#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include <omp.h>

#include "include/uni_nav_graph.h"
#include "include/MethodSelector.h"
#include "include/ung_build_settings.h"
#include "include/ung_gpu_cover_frontier_provider.h"
#include "include/ung_prof_log.h"


namespace ANNS
{
namespace
{

void print_label_nav_graph_summary(const std::shared_ptr<LabelNavGraph> &label_nav_graph,
                                   IdxType num_groups)
{
   if (!label_nav_graph)
      return;

   uint64_t total_lng_edges = 0;
   for (IdxType group_id = 1; group_id <= num_groups; ++group_id)
      total_lng_edges += label_nav_graph->out_neighbors[group_id].size();

   std::cout << "[label_nav_graph] groups=" << num_groups
             << " edges=" << total_lng_edges;
   if (num_groups > 0)
      std::cout << " avg_out_degree=" << static_cast<double>(total_lng_edges) / num_groups;
   std::cout << std::endl;

   prof_logf("[PROF] label_nav_graph.summary groups=%u edges=%llu avg_out_degree=%.6f",
             static_cast<unsigned>(num_groups),
             static_cast<unsigned long long>(total_lng_edges),
             num_groups > 0 ? static_cast<double>(total_lng_edges) / num_groups : 0.0);
}

} // namespace

   UniNavGraph::UniNavGraph(IdxType num_nodes)
       : _label_nav_graph(std::make_shared<LabelNavGraph>(num_nodes))
   {
   }

   UniNavGraph::UniNavGraph() = default;
   UniNavGraph::~UniNavGraph() = default;

   void UniNavGraph::build(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler,
                           std::string scenario, std::string index_name, uint32_t num_threads, IdxType num_cross_edges,
                           IdxType max_degree, IdxType Lbuild, float alpha, std::string dataset,
                           ANNS::AcornInUng new_cross_edge)
   {
      auto all_start_time = std::chrono::high_resolution_clock::now();
      _base_storage = base_storage;
      _num_points = base_storage->get_num_points();
      _distance_handler = distance_handler;
      std::cout << "- Scenario: " << scenario << std::endl;

      _dataset = dataset;

      // index parameters
      _index_name = index_name;
      _num_cross_edges = num_cross_edges;
      _max_degree = max_degree;
      _Lbuild = Lbuild;
      _alpha = alpha;
      _num_threads = num_threads;
      _scenario = scenario;
      _build_config = UngBuildConfig::from_env(num_threads);
      _build_config.print(std::cout);

      std::cout << "Dividing groups and building the trie tree index ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      build_trie_and_divide_groups();
      _graph = std::make_shared<ANNS::Graph>(base_storage->get_num_points());
      _global_graph = std::make_shared<ANNS::Graph>(base_storage->get_num_points());
      std::cout << "begin prepare_group_storages_graphs" << std::endl;
      prepare_group_storages_graphs();
      build_special_blocks();
      reserve_graph_neighbor_capacity();
      _label_processing_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
      std::cout << "- Finished in " << _label_processing_time << " ms" << std::endl;

      // build graph index for each group
      build_graph_for_all_groups();
      if (!_build_config.is_original_cpu_pipeline())
         build_vector_and_attr_graph();

      // for label equality scenario, there is no need for label navigating graph and cross-group edges
      if (_scenario == "equality")
      {
         add_offset_for_uni_nav_graph();
      }
      else
      {

         // build the label navigating graph
         build_label_nav_graph();
         if (!_build_config.is_original_cpu_pipeline())
         {
            get_descendants_info();
            print_label_nav_graph_summary(_label_nav_graph, _num_groups);

            // calculate the coverage ratio
            cal_f_coverage_ratio();

            // initialize_lng_descendants_coverage_bitsets();
            auto roaring_start_time = std::chrono::high_resolution_clock::now();
            initialize_roaring_bitsets();
            _build_roaring_bitsets_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - roaring_start_time).count();

         }
         else
         {
            _build_roaring_bitsets_time = 0.0;
         }

         std::cout << "new_cross_edge.ung_and_acorn: " << new_cross_edge.ung_and_acorn << std::endl;
         if (_build_config.is_original_cpu_pipeline() || !new_cross_edge.ung_and_acorn)
            build_cross_group_edges();
         else
         {
            _num_cross_edges = _num_cross_edges / 2;
            finalize_intra_group_graphs();

            add_new_distance_oriented_edges(
                dataset,
                num_threads,
                new_cross_edge);
         }
      }

      if (_build_config.special_blocks_enabled)
         build_special_edge_overlay();

      // index time
      _index_time = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - all_start_time)
                        .count();
   }

   void UniNavGraph::prepare_group_storages_graphs()
   {
      _new_vec_id_to_group_id.resize(_num_points);

      // reorder the vectors
      _group_id_to_range.resize(_num_groups + 1);
      _new_to_old_vec_ids.resize(_num_points);
      IdxType new_vec_id = 0;
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         _group_id_to_range[group_id].first = new_vec_id;
         for (auto old_vec_id : _group_id_to_vec_ids[group_id])
         {
            _new_to_old_vec_ids[new_vec_id] = old_vec_id;
            _new_vec_id_to_group_id[new_vec_id] = group_id;
            ++new_vec_id;
         }
         _group_id_to_range[group_id].second = new_vec_id;
      }

      // reorder the underlying storage
      _base_storage->reorder_data(_new_to_old_vec_ids);

      // init storage and graph for each group
      _group_storages.resize(_num_groups + 1);
      _group_graphs.resize(_num_groups + 1);
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         auto start = _group_id_to_range[group_id].first;
         auto end = _group_id_to_range[group_id].second;
         _group_storages[group_id] = create_storage(_base_storage, start, end);
         _group_graphs[group_id] = std::make_shared<Graph>(_graph, start, end);
      }
   }

   void UniNavGraph::reserve_graph_neighbor_capacity()
   {
      const GraphReserveSettings reserve_cfg =
          make_graph_reserve_settings(_max_degree, _num_cross_edges);
      uint64_t small_groups = 0;
      uint64_t small_points = 0;
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         const auto &range = _group_id_to_range[group_id];
         const IdxType group_size = range.second - range.first;
         if (group_size <= reserve_cfg.complete_threshold)
         {
            ++small_groups;
            small_points += static_cast<uint64_t>(group_size);
         }
      }

      const bool enable_reserve = reserve_cfg.enabled(_num_points, _num_groups, small_points);
      const std::string reserve_summary =
          reserve_cfg.summary(_num_points, _num_groups, small_points);
      std::cout << "[graph_reserve] config " << reserve_summary << std::endl;
      prof_logf("[PROF] graph.reserve_config %s", reserve_summary.c_str());
      if (!enable_reserve || !_graph || !_graph->neighbors)
      {
         std::cout << "[graph_reserve] enabled=0 mode="
                   << reserve_cfg.mode_name()
                   << " groups=" << _num_groups
                   << " small_groups=" << small_groups
                   << " small_points=" << small_points
                   << " complete_threshold=" << reserve_cfg.complete_threshold
                   << std::endl;
         prof_logf("[PROF] graph.reserve_capacity enabled=0 mode=%s groups=%u small_groups=%llu small_points=%llu complete_threshold=%u",
                   reserve_cfg.mode_name(),
                   static_cast<unsigned>(_num_groups),
                   static_cast<unsigned long long>(small_groups),
                   static_cast<unsigned long long>(small_points),
                   static_cast<unsigned>(reserve_cfg.complete_threshold));
         return;
      }

      auto start = std::chrono::high_resolution_clock::now();
      uint64_t reserved_edges = 0;
#pragma omp parallel for schedule(dynamic, 4096) reduction(+ : reserved_edges)
      for (IdxType point_id = 0; point_id < _num_points; ++point_id)
      {
         const IdxType group_id = _new_vec_id_to_group_id[point_id];
         const auto &range = _group_id_to_range[group_id];
         const IdxType group_size = range.second - range.first;
         IdxType intra_cap = std::min<IdxType>(_max_degree, group_size > 0 ? group_size - 1 : 0);
         if (group_size <= _max_degree && group_size > 0)
            intra_cap = group_size - 1;
         const IdxType reserve_cap = std::min<IdxType>(reserve_cfg.hard_cap,
                                                       std::max<IdxType>(0, intra_cap) +
                                                           _num_cross_edges + reserve_cfg.additional_slack);
         _graph->neighbors[point_id].reserve(static_cast<size_t>(reserve_cap));
         reserved_edges += static_cast<uint64_t>(reserve_cap);
      }
      const double reserve_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::high_resolution_clock::now() - start)
                                    .count();
      std::cout << "[graph_reserve] enabled=1 capacity_edges=" << reserved_edges
                << " mode=" << reserve_cfg.mode_name()
                << " groups=" << _num_groups
                << " small_groups=" << small_groups
                << " small_points=" << small_points
                << " ms=" << reserve_ms
                << " slack=" << reserve_cfg.additional_slack
                << " hard_cap=" << reserve_cfg.hard_cap << std::endl;
      prof_logf("[PROF] graph.reserve_capacity enabled=1 mode=%s capacity_edges=%llu ms=%.3f slack=%u hard_cap=%u groups=%u small_groups=%llu small_points=%llu complete_threshold=%u",
                reserve_cfg.mode_name(),
                static_cast<unsigned long long>(reserved_edges), reserve_ms,
                static_cast<unsigned>(reserve_cfg.additional_slack),
                static_cast<unsigned>(reserve_cfg.hard_cap),
                static_cast<unsigned>(_num_groups),
                static_cast<unsigned long long>(small_groups),
                static_cast<unsigned long long>(small_points),
                static_cast<unsigned>(reserve_cfg.complete_threshold));
   }

   bool UniNavGraph::should_write_intra_group_global_ids() const
   {
      const IntraGroupIdSettings cfg = make_intra_group_id_settings(_build_config);
      prof_logf("[PROF] group_graph.intra_global_id_config %s", cfg.summary().c_str());
      return cfg.enabled();
   }

   void UniNavGraph::finalize_intra_group_graphs()
   {
      if (_intra_group_graph_ids_are_global)
      {
         std::cout << "Finalizing intra-group graphs skipped: neighbor IDs are already global." << std::endl;
         return;
      }
      std::cout << "Finalizing intra-group graphs by converting neighbor IDs to global..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      add_offset_for_uni_nav_graph();
      auto duration = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
      std::cout << "- Finished in " << duration << " ms" << std::endl;
   }



   void UniNavGraph::add_offset_for_uni_nav_graph()
   {
      if (_intra_group_graph_ids_are_global)
      {
         prof_logf("[PROF] graph.add_offset skipped=1 reason=intra_global_ids");
         return;
      }
      omp_set_num_threads(_num_threads);
#pragma omp parallel for schedule(dynamic, 4096)
      for (auto i = 0; i < _num_points; ++i)
         for (auto &neighbor : _graph->neighbors[i])
            neighbor += _group_id_to_range[_new_vec_id_to_group_id[i]].first;
   }

}
