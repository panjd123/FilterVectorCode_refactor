#include <omp.h>

#include "include/uni_nav_graph.h"
#include "include/tagore_graph_builder.h"
#include "include/ung_build_settings.h"
#include "include/ung_prof_log.h"
#include "vamana/vamana.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ANNS
{

void UniNavGraph::build_complete_graph(std::shared_ptr<Graph> graph, IdxType num_points, IdxType base_offset)
{
   if (num_points == 0)
      return;
   for (IdxType i = 0; i < num_points; ++i)
   {
      auto &neighbors = graph->neighbors[i];
      neighbors.resize(num_points > 0 ? num_points - 1 : 0);
      IdxType write = 0;
      for (IdxType j = 0; j < num_points; ++j)
      {
         if (i != j)
            neighbors[write++] = j + base_offset;
      }
   }
}

void UniNavGraph::build_bounded_complete_graph(std::shared_ptr<Graph> graph,
                                               IdxType num_points,
                                               IdxType max_degree,
                                               IdxType base_offset)
{
   if (num_points == 0)
      return;
   if (num_points <= max_degree + 1)
   {
      build_complete_graph(graph, num_points, base_offset);
      return;
   }

   const IdxType degree = std::min(max_degree, num_points - 1);
   for (IdxType i = 0; i < num_points; ++i)
   {
      auto &neighbors = graph->neighbors[i];
      neighbors.resize(degree);
      for (IdxType j = 0; j < degree; ++j)
         neighbors[j] = ((i + j + 1) % num_points) + base_offset;
   }
}

void UniNavGraph::build_graph_for_all_groups()
{
   std::cout << "Building graph for each group ..." << std::endl;
   omp_set_num_threads(_num_threads);
   auto start_time = std::chrono::high_resolution_clock::now();
   _intra_group_graph_ids_are_global = false;

   if (_index_name != "Vamana")
   {
      std::cerr << "Error: invalid index name " << _index_name << std::endl;
      std::exit(-1);
   }

   const GroupGraphBuildContext build_context =
       make_group_graph_build_context(_build_config, _max_degree, _num_threads);
   const GroupGraphRoute &group_route = build_context.route;
   _vamana_instances.resize(_num_groups + 1);
   _special_block_target_graphs.assign(_num_groups + 1, nullptr);
   _group_entry_points.resize(_num_groups + 1);
   const CpuGroupGraphSettings &group_cfg = build_context.cpu_settings;
   const std::string group_cfg_summary = group_cfg.summary();
   std::cout << "[group_graph] cpu_config " << group_cfg_summary << std::endl;
   prof_logf("[PROF] group_graph.cpu_config %s", group_cfg_summary.c_str());

   const IdxType small_group_complete_threshold = group_cfg.complete_threshold;
   const bool bounded_complete = group_cfg.bounded_complete;
   std::atomic<size_t> complete_fast_groups{0};
   std::atomic<size_t> complete_fast_points{0};
   std::atomic<size_t> vamana_build_groups{0};
   std::atomic<size_t> vamana_build_points{0};
   std::atomic<size_t> trivial_special_skipped_groups{0};
   std::atomic<size_t> trivial_special_skipped_points{0};
   const bool profile_group_graph = group_cfg.profile;
   std::vector<double> group_graph_ms(profile_group_graph ? _num_groups + 1 : 0, 0.0);
   std::vector<IdxType> group_graph_nx(profile_group_graph ? _num_groups + 1 : 0, 0);
   std::vector<uint8_t> group_graph_route(profile_group_graph ? _num_groups + 1 : 0, 0);
   const int large_group_inner_threads = group_cfg.large_inner_threads;
   const IdxType large_group_inner_threshold = group_cfg.large_inner_threshold;
   const bool split_large_group_build = group_cfg.split_large_group_build();
   const int large_group_outer_threads = group_cfg.large_outer_threads;
   std::vector<IdxType> large_group_ids;

   if (split_large_group_build)
   {
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         const auto &range = _group_id_to_range[group_id];
         const IdxType group_size = range.second - range.first;
         if (group_size > small_group_complete_threshold && group_size >= large_group_inner_threshold)
            large_group_ids.push_back(group_id);
      }
      std::sort(large_group_ids.begin(), large_group_ids.end(), [&](IdxType a, IdxType b) {
         const auto &ra = _group_id_to_range[a];
         const auto &rb = _group_id_to_range[b];
         return (ra.second - ra.first) > (rb.second - rb.first);
      });
      std::cout << "[group_graph] large_inner_threads=" << large_group_inner_threads
                << " large_outer_threads=" << large_group_outer_threads
                << " large_inner_nx=" << large_group_inner_threshold
                << " large_inner_groups=" << large_group_ids.size()
                << std::endl;
   }

   if (group_route.uses_cuda_backend() && !_build_config.special_blocks_enabled)
   {
      build_graph_for_all_groups_tagore_cuda();
      _build_graph_time = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - start_time)
                              .count();
      std::cout << "\r- Finished in " << _build_graph_time << " ms" << std::endl;
      return;
   }
   if (group_route.uses_cuda_backend() && _build_config.special_blocks_enabled &&
       _build_config.special_block_skip_trivial)
   {
      std::cout << "[group_graph] special_blocks=1 forces CPU group-graph path so trivial special blocks can be skipped without writing ordinary edges." << std::endl;
   }

#pragma omp parallel for schedule(dynamic, 1)
   for (auto group_id = 1; group_id <= _num_groups; ++group_id)
   {
      const auto &range = _group_id_to_range[group_id];
      const IdxType group_size = range.second - range.first;
      if (split_large_group_build && group_size > small_group_complete_threshold &&
          group_size >= large_group_inner_threshold)
         continue;
      const auto group_start_time = profile_group_graph ? std::chrono::high_resolution_clock::now()
                                                        : std::chrono::high_resolution_clock::time_point{};
      if (_build_config.special_blocks_enabled && _build_config.special_block_skip_trivial &&
          is_trivial_special_block_root_group(group_id))
      {
         auto scratch_graph = std::make_shared<Graph>(group_size);
         if (group_size <= small_group_complete_threshold)
         {
            if (bounded_complete)
               build_bounded_complete_graph(scratch_graph, group_size, _max_degree);
            else
               build_complete_graph(scratch_graph, group_size);
            _vamana_instances[group_id] = std::make_shared<Vamana>(_group_storages[group_id], _distance_handler,
                                                                   scratch_graph, 0);
         }
         else
         {
            _vamana_instances[group_id] = std::make_shared<Vamana>(false);
            _vamana_instances[group_id]->build(_group_storages[group_id], _distance_handler,
                                               scratch_graph, _max_degree, _Lbuild, _alpha, 1);
         }
         _special_block_target_graphs[group_id] = scratch_graph;
         _group_entry_points[group_id] = _vamana_instances[group_id]->get_entry_point() + range.first;
         trivial_special_skipped_groups.fetch_add(1, std::memory_order_relaxed);
         trivial_special_skipped_points.fetch_add(static_cast<size_t>(group_size), std::memory_order_relaxed);
         if (profile_group_graph)
         {
            group_graph_ms[group_id] = std::chrono::duration<double, std::milli>(
                                          std::chrono::high_resolution_clock::now() - group_start_time)
                                          .count();
            group_graph_nx[group_id] = group_size;
            group_graph_route[group_id] = 4;
         }
         continue;
      }
      if (group_size <= small_group_complete_threshold)
      {
         if (bounded_complete)
            build_bounded_complete_graph(_group_graphs[group_id], group_size, _max_degree);
         else
            build_complete_graph(_group_graphs[group_id], group_size);
         _vamana_instances[group_id] = std::make_shared<Vamana>(_group_storages[group_id], _distance_handler,
                                                                _group_graphs[group_id], 0);
         complete_fast_groups.fetch_add(1, std::memory_order_relaxed);
         complete_fast_points.fetch_add(static_cast<size_t>(group_size), std::memory_order_relaxed);
      }
      else
      {
         _vamana_instances[group_id] = std::make_shared<Vamana>(false);
         _vamana_instances[group_id]->build(_group_storages[group_id], _distance_handler,
                                            _group_graphs[group_id], _max_degree, _Lbuild, _alpha, 1);
         vamana_build_groups.fetch_add(1, std::memory_order_relaxed);
         vamana_build_points.fetch_add(static_cast<size_t>(group_size), std::memory_order_relaxed);
      }
      if (profile_group_graph)
      {
         group_graph_ms[group_id] = std::chrono::duration<double, std::milli>(
                                       std::chrono::high_resolution_clock::now() - group_start_time)
                                       .count();
         group_graph_nx[group_id] = group_size;
         group_graph_route[group_id] = group_size <= small_group_complete_threshold ? 1 : 2;
      }

      _group_entry_points[group_id] = _vamana_instances[group_id]->get_entry_point() + range.first;
   }

   if (split_large_group_build)
   {
      omp_set_dynamic(0);
      omp_set_max_active_levels(2);
   }
#pragma omp parallel for schedule(dynamic, 1) num_threads(large_group_outer_threads)
   for (size_t large_idx = 0; large_idx < large_group_ids.size(); ++large_idx)
   {
      const auto group_id = large_group_ids[large_idx];
      const auto &range = _group_id_to_range[group_id];
      const IdxType group_size = range.second - range.first;
      const auto group_start_time = profile_group_graph ? std::chrono::high_resolution_clock::now()
                                                        : std::chrono::high_resolution_clock::time_point{};
      if (_build_config.special_blocks_enabled && _build_config.special_block_skip_trivial &&
          is_trivial_special_block_root_group(group_id))
      {
         auto scratch_graph = std::make_shared<Graph>(group_size);
         _vamana_instances[group_id] = std::make_shared<Vamana>(false);
         _vamana_instances[group_id]->build(_group_storages[group_id], _distance_handler,
                                            scratch_graph, _max_degree, _Lbuild, _alpha,
                                            static_cast<uint32_t>(large_group_inner_threads));
         _special_block_target_graphs[group_id] = scratch_graph;
         _group_entry_points[group_id] = _vamana_instances[group_id]->get_entry_point() + range.first;
         trivial_special_skipped_groups.fetch_add(1, std::memory_order_relaxed);
         trivial_special_skipped_points.fetch_add(static_cast<size_t>(group_size), std::memory_order_relaxed);
         if (profile_group_graph)
         {
            group_graph_ms[group_id] = std::chrono::duration<double, std::milli>(
                                          std::chrono::high_resolution_clock::now() - group_start_time)
                                          .count();
            group_graph_nx[group_id] = group_size;
            group_graph_route[group_id] = 4;
         }
         continue;
      }
      _vamana_instances[group_id] = std::make_shared<Vamana>(false);
      _vamana_instances[group_id]->build(_group_storages[group_id], _distance_handler,
                                         _group_graphs[group_id], _max_degree, _Lbuild, _alpha,
                                         static_cast<uint32_t>(large_group_inner_threads));
      vamana_build_groups.fetch_add(1, std::memory_order_relaxed);
      vamana_build_points.fetch_add(static_cast<size_t>(group_size), std::memory_order_relaxed);
      if (profile_group_graph)
      {
         group_graph_ms[group_id] = std::chrono::duration<double, std::milli>(
                                       std::chrono::high_resolution_clock::now() - group_start_time)
                                       .count();
         group_graph_nx[group_id] = group_size;
         group_graph_route[group_id] = 3;
      }
      _group_entry_points[group_id] = _vamana_instances[group_id]->get_entry_point() + range.first;
   }

   _build_graph_time = std::chrono::duration<double, std::milli>(
                           std::chrono::high_resolution_clock::now() - start_time)
                           .count();
   _special_block_summary.group_graph_trivial_skipped_groups =
       static_cast<IdxType>(trivial_special_skipped_groups.load());
   _special_block_summary.group_graph_trivial_skipped_points =
       static_cast<IdxType>(trivial_special_skipped_points.load());
   std::cout << "[group_graph] complete_threshold_nx=" << small_group_complete_threshold
             << " complete_groups=" << complete_fast_groups.load()
             << " complete_points=" << complete_fast_points.load()
             << " vamana_groups=" << vamana_build_groups.load()
             << " vamana_points=" << vamana_build_points.load()
             << " trivial_special_skipped_groups=" << trivial_special_skipped_groups.load()
             << " trivial_special_skipped_points=" << trivial_special_skipped_points.load()
             << std::endl;

   if (profile_group_graph)
   {
      struct BinStat
      {
         const char *name;
         IdxType lo;
         IdxType hi;
         size_t groups = 0;
         size_t points = 0;
         double sum_ms = 0.0;
         double max_ms = 0.0;
         IdxType max_gid = 0;
         IdxType max_nx = 0;
      };
      std::vector<BinStat> bins = {
          {"<=64", 0, 64},
          {"65-128", 65, 128},
          {"129-256", 129, 256},
          {"257-512", 257, 512},
          {"513-1024", 513, 1024},
          {"1025-2048", 1025, 2048},
          {">2048", 2049, std::numeric_limits<IdxType>::max()}};
      std::vector<IdxType> order;
      order.reserve(_num_groups);
      double total_group_ms = 0.0;
      double total_vamana_ms = 0.0;
      double total_complete_ms = 0.0;
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         const IdxType nx = group_graph_nx[group_id];
         const double ms = group_graph_ms[group_id];
         total_group_ms += ms;
         if (group_graph_route[group_id] == 1)
            total_complete_ms += ms;
         else if (group_graph_route[group_id] == 2 || group_graph_route[group_id] == 3)
            total_vamana_ms += ms;
         order.push_back(group_id);
         for (auto &bin : bins)
         {
            if (nx >= bin.lo && nx <= bin.hi)
            {
               bin.groups += 1;
               bin.points += static_cast<size_t>(nx);
               bin.sum_ms += ms;
               if (ms > bin.max_ms)
               {
                  bin.max_ms = ms;
                  bin.max_gid = group_id;
                  bin.max_nx = nx;
               }
               break;
            }
         }
      }
      std::sort(order.begin(), order.end(), [&](IdxType a, IdxType b) {
         return group_graph_ms[a] > group_graph_ms[b];
      });
      std::cout << "[group_graph_profile] total_group_ms_sum=" << total_group_ms
                << " complete_ms_sum=" << total_complete_ms
                << " vamana_ms_sum=" << total_vamana_ms
                << " wall_ms=" << _build_graph_time
                << std::endl;
      for (const auto &bin : bins)
      {
         if (bin.groups == 0)
            continue;
         std::cout << "[group_graph_profile_bin] nx=" << bin.name
                   << " groups=" << bin.groups
                   << " points=" << bin.points
                   << " sum_ms=" << bin.sum_ms
                   << " avg_ms=" << (bin.sum_ms / static_cast<double>(bin.groups))
                   << " max_ms=" << bin.max_ms
                   << " max_gid=" << bin.max_gid
                   << " max_nx=" << bin.max_nx
                   << std::endl;
      }
      const size_t topn = std::min<size_t>(20, order.size());
      for (size_t i = 0; i < topn; ++i)
      {
         const IdxType gid = order[i];
         std::cout << "[group_graph_profile_top] rank=" << (i + 1)
                   << " gid=" << gid
                   << " nx=" << group_graph_nx[gid]
                   << " route="
                   << (group_graph_route[gid] == 1 ? "complete"
                                                    : (group_graph_route[gid] == 3 ? "vamana_inner_mt"
                                                                                   : (group_graph_route[gid] == 4 ? "trivial_special_skip"
                                                                                                                   : "vamana")))
                   << " ms=" << group_graph_ms[gid]
                   << std::endl;
      }
   }
   std::cout << "\r- Finished in " << _build_graph_time << " ms" << std::endl;
}

void UniNavGraph::build_graph_for_all_groups_tagore_cuda()
{
   if (_base_storage->get_data_type() != DataType::FLOAT)
      throw std::runtime_error("TagoreCuda supports only float vectors.");

   _intra_group_graph_ids_are_global = should_write_intra_group_global_ids();
   std::cout << "[group_graph] intra_global_ids=" << (_intra_group_graph_ids_are_global ? 1 : 0)
             << std::endl;
   prof_logf("[PROF] group_graph.intra_global_ids enabled=%d",
             _intra_group_graph_ids_are_global ? 1 : 0);

   _tagore_groups = 0.0;
   _tagore_points = 0.0;
   _tagore_direct_build_wall_time_ms = 0.0;
   _tagore_no_alloc_build_time_ms = 0.0;
   _tagore_pack_time_ms = 0.0;
   _tagore_convert_time_ms = 0.0;
   _tagore_workspace_alloc_time_ms = 0.0;
   _tagore_h2d_time_ms = 0.0;
   _tagore_memset_time_ms = 0.0;
   _tagore_gnn_time_ms = 0.0;
   _tagore_prune_time_ms = 0.0;
   _tagore_grnnd_refine_time_ms = 0.0;
   _tagore_d2h_time_ms = 0.0;
   _tagore_fill_time_ms = 0.0;
   _tagore_workspace_free_time_ms = 0.0;
   _tagore_unaccounted_time_ms = 0.0;
   _tagore_h2d_effective_gbps = 0.0;
   _tagore_d2h_effective_gbps = 0.0;
   _tagore_gnn_mpts_s = 0.0;
   _tagore_prune_mpts_s = 0.0;

   double tagore_wall_ms = 0.0;
   double tagore_pack_ms = 0.0;
   double tagore_fill_ms = 0.0;
   TagoreFallbackBuildStats fallback_stats;
   TagoreBuildResult timing_acc;

   const TagoreGroupBuildContext context =
       make_tagore_group_build_context(_build_config, _max_degree, _num_threads);
   const std::string tagore_cfg_summary = context.settings.summary();
   std::cout << "[TagoreCuda] config " << tagore_cfg_summary << std::endl;
   prof_logf("[PROF] tagore.config %s", tagore_cfg_summary.c_str());

   const TagoreGroupPartition group_partition = partition_tagore_groups(context);

   auto build_fallback_groups = [&]() {
      fallback_stats = build_tagore_fallback_groups(group_partition.fallback_group_ids, context);
   };

   const bool overlap_fallback =
       context.settings.should_overlap_fallback(!group_partition.fallback_group_ids.empty(),
                                                !group_partition.tagore_requests.empty());
   std::exception_ptr fallback_error;
   std::thread fallback_thread;
   if (overlap_fallback)
   {
      fallback_thread = std::thread([&]() {
         try
         {
            build_fallback_groups();
         }
         catch (...)
         {
            fallback_error = std::current_exception();
         }
      });
   }
   else
   {
      build_fallback_groups();
   }

   struct FallbackThreadJoiner
   {
      std::thread &thread;
      ~FallbackThreadJoiner()
      {
         if (thread.joinable())
            thread.join();
      }
   } fallback_joiner{fallback_thread};

   TagoreBatchBuildArtifacts tagore_batch =
       build_tagore_batch_artifacts(group_partition.tagore_requests, context);
   tagore_wall_ms = tagore_batch.wall_ms;
   if (fallback_thread.joinable())
      fallback_thread.join();
   if (fallback_error)
      std::rethrow_exception(fallback_error);

   double tagore_fill_cpu_sum_ms = 0.0;
   fill_tagore_batch_results(group_partition.tagore_group_ids, tagore_batch.results,
                             tagore_batch.exact_packed_rank,
                             tagore_batch.exact_packed_graph,
                             tagore_batch.exact_packed_offsets,
                             tagore_batch.exact_packed_stride,
                             context,
                             timing_acc, tagore_fill_ms, tagore_fill_cpu_sum_ms);

   timing_acc.alloc_ms = tagore_batch.batch_timing.batch_alloc_ms;
   timing_acc.free_ms = tagore_batch.batch_timing.batch_free_ms;
   tagore_pack_ms += tagore_batch.batch_timing.batch_pack_ms;

   const uint64_t h2d_bytes =
       group_partition.tagore_total_points * static_cast<uint64_t>(_base_storage->get_dim()) * sizeof(float);
   uint32_t effective_k = _build_config.tagore_k <= _max_degree ? static_cast<uint32_t>(_max_degree + 1)
                                                                : _build_config.tagore_k;
   if (!tagore_batch.results.empty() && tagore_batch.results.front().graph_stride != 0)
      effective_k = tagore_batch.results.front().graph_stride;
   const uint64_t d2h_bytes =
       group_partition.tagore_total_points * static_cast<uint64_t>(effective_k) * sizeof(uint32_t) +
       group_partition.tagore_group_ids.size() * sizeof(uint32_t);
   const double h2d_gbps =
       timing_acc.h2d_ms > 0.0 ? (static_cast<double>(h2d_bytes) / 1.0e6) / timing_acc.h2d_ms : 0.0;
   const double d2h_gbps =
       timing_acc.d2h_ms > 0.0 ? (static_cast<double>(d2h_bytes) / 1.0e6) / timing_acc.d2h_ms : 0.0;
   const double gnn_mpts_s =
       timing_acc.gnn_ms > 0.0
           ? (static_cast<double>(group_partition.tagore_total_points) / 1.0e3) / timing_acc.gnn_ms
           : 0.0;
   const double prune_mpts_s =
       timing_acc.prune_ms > 0.0
           ? (static_cast<double>(group_partition.tagore_total_points) / 1.0e3) / timing_acc.prune_ms
           : 0.0;
   const double tagore_total_wall_ms = tagore_wall_ms + tagore_fill_ms;
   const double tagore_unaccounted_ms =
       tagore_total_wall_ms - tagore_pack_ms - timing_acc.convert_ms - timing_acc.alloc_ms -
       timing_acc.h2d_ms - timing_acc.memset_ms - timing_acc.gnn_ms - timing_acc.prune_ms -
       timing_acc.grnnd_refine_ms - timing_acc.d2h_ms - tagore_fill_ms - timing_acc.free_ms;

   _tagore_groups = static_cast<double>(group_partition.tagore_group_ids.size());
   _tagore_points = static_cast<double>(group_partition.tagore_total_points);
   _tagore_direct_build_wall_time_ms = tagore_wall_ms;
   _tagore_no_alloc_build_time_ms = tagore_total_wall_ms - timing_acc.alloc_ms;
   _tagore_pack_time_ms = tagore_pack_ms;
   _tagore_convert_time_ms = timing_acc.convert_ms;
   _tagore_workspace_alloc_time_ms = timing_acc.alloc_ms;
   _tagore_h2d_time_ms = timing_acc.h2d_ms;
   _tagore_memset_time_ms = timing_acc.memset_ms;
   _tagore_gnn_time_ms = timing_acc.gnn_ms;
   _tagore_prune_time_ms = timing_acc.prune_ms;
   _tagore_grnnd_refine_time_ms = timing_acc.grnnd_refine_ms;
   _tagore_d2h_time_ms = timing_acc.d2h_ms;
   _tagore_fill_time_ms = tagore_fill_ms;
   _tagore_workspace_free_time_ms = timing_acc.free_ms;
   _tagore_unaccounted_time_ms = tagore_unaccounted_ms;
   _tagore_h2d_effective_gbps = h2d_gbps;
   _tagore_d2h_effective_gbps = d2h_gbps;
   _tagore_gnn_mpts_s = gnn_mpts_s;
   _tagore_prune_mpts_s = prune_mpts_s;

   std::cout << "- TagoreCuda groups: " << group_partition.tagore_group_ids.size()
             << ", fallback_complete_groups: " << fallback_stats.complete_groups
             << ", fallback_complete_points: " << fallback_stats.complete_points
             << ", fallback_cpu_groups: " << fallback_stats.cpu_groups
             << ", fallback_cpu_points: " << fallback_stats.cpu_points
             << ", fallback_threads: " << context.settings.fallback_threads
             << ", fallback_wall: " << fallback_stats.wall_ms
             << ", direct build wall: " << tagore_wall_ms
             << " ms, no_alloc_total: " << _tagore_no_alloc_build_time_ms
             << " ms, pack: " << tagore_pack_ms
             << " ms, convert: " << timing_acc.convert_ms
             << " ms, alloc: " << timing_acc.alloc_ms
             << " ms, h2d: " << timing_acc.h2d_ms << " ms (" << h2d_gbps << " GB/s)"
             << " ms, memset: " << timing_acc.memset_ms
             << " ms, gnn: " << timing_acc.gnn_ms << " ms (" << gnn_mpts_s << " Mpts/s)"
             << " ms, prune: " << timing_acc.prune_ms << " ms (" << prune_mpts_s << " Mpts/s)"
             << " ms, grnnd_refine: " << timing_acc.grnnd_refine_ms
             << " ms, d2h: " << timing_acc.d2h_ms << " ms (" << d2h_gbps << " GB/s)"
             << " ms, fill: " << tagore_fill_ms
             << " ms, fill_cpu_sum: " << tagore_fill_cpu_sum_ms
             << " ms, free: " << timing_acc.free_ms
             << " ms, unaccounted: " << tagore_unaccounted_ms
             << " ms" << std::endl;
}

TagoreFallbackBuildStats
UniNavGraph::build_tagore_fallback_groups(const std::vector<IdxType> &fallback_group_ids,
                                          const TagoreGroupBuildContext &context)
{
   TagoreFallbackBuildStats stats;
   size_t complete_groups = 0;
   size_t complete_points = 0;
   size_t cpu_groups = 0;
   size_t cpu_points = 0;
   const IdxType complete_threshold = context.settings.complete_threshold;
   const int fallback_impl = context.settings.fallback_impl;
   const bool bounded_complete = context.settings.bounded_complete;
   const int fallback_threads = context.settings.fallback_threads;
   const auto fallback_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1) num_threads(fallback_threads) reduction(+ : complete_groups, complete_points, cpu_groups, cpu_points)
   for (size_t i = 0; i < fallback_group_ids.size(); ++i)
   {
      const IdxType group_id = fallback_group_ids[i];
      const auto &range = _group_id_to_range[group_id];
      const IdxType n = range.second - range.first;
      if (n <= complete_threshold || fallback_impl == 0)
      {
         if (bounded_complete)
            build_bounded_complete_graph(_group_graphs[group_id], n, _max_degree,
                                         _intra_group_graph_ids_are_global ? range.first : 0);
         else
            build_complete_graph(_group_graphs[group_id], n,
                                 _intra_group_graph_ids_are_global ? range.first : 0);
         _vamana_instances[group_id] = std::make_shared<Vamana>(_group_storages[group_id], _distance_handler,
                                                                _group_graphs[group_id], 0);
         _group_entry_points[group_id] = range.first;
         complete_groups += 1;
         complete_points += static_cast<size_t>(n);
      }
      else
      {
         _vamana_instances[group_id] = std::make_shared<Vamana>(false);
         _vamana_instances[group_id]->build(_group_storages[group_id], _distance_handler,
                                            _group_graphs[group_id], _max_degree, _Lbuild, _alpha, 1);
         if (_intra_group_graph_ids_are_global)
         {
            for (IdxType local_id = 0; local_id < n; ++local_id)
               for (auto &neighbor : _group_graphs[group_id]->neighbors[local_id])
                  neighbor += range.first;
         }
         _group_entry_points[group_id] = _vamana_instances[group_id]->get_entry_point() + range.first;
         cpu_groups += 1;
         cpu_points += static_cast<size_t>(n);
      }
   }
   stats.wall_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - fallback_start)
                       .count();
   stats.complete_groups = complete_groups;
   stats.complete_points = complete_points;
   stats.cpu_groups = cpu_groups;
   stats.cpu_points = cpu_points;
   return stats;
}

TagoreGroupPartition
UniNavGraph::partition_tagore_groups(const TagoreGroupBuildContext &context) const
{
   TagoreGroupPartition partition;
   partition.fallback_group_ids.reserve(_num_groups);
   partition.tagore_group_ids.reserve(_num_groups);
   partition.tagore_requests.reserve(_num_groups);
   for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
   {
      const auto &range = _group_id_to_range[group_id];
      const IdxType n = range.second - range.first;
      if (n <= context.settings.complete_threshold || n < _build_config.tagore_min_group_size)
      {
         partition.fallback_group_ids.emplace_back(group_id);
      }
      else
      {
         partition.tagore_group_ids.emplace_back(group_id);
         partition.tagore_requests.push_back({reinterpret_cast<const float *>(_base_storage->get_vector(range.first)),
                                              static_cast<uint32_t>(n)});
         partition.tagore_total_points += n;
      }
   }
   return partition;
}

TagoreBatchBuildArtifacts
UniNavGraph::build_tagore_batch_artifacts(const std::vector<TagoreGroupRequest> &tagore_requests,
                                          const TagoreGroupBuildContext &context)
{
   TagoreBatchBuildArtifacts artifacts;
   artifacts.results.resize(tagore_requests.size());
   constexpr size_t npos = static_cast<size_t>(-1);
   artifacts.exact_packed_rank.assign(tagore_requests.size(), npos);
   if (tagore_requests.empty())
      return artifacts;

   const auto batch_start = std::chrono::high_resolution_clock::now();
   const uint32_t dim = static_cast<uint32_t>(_base_storage->get_dim());
   if (context.exact_batch_threshold > 0)
   {
      std::vector<TagoreGroupRequest> exact_requests;
      std::vector<TagoreGroupRequest> gnn_requests;
      std::vector<size_t> exact_positions;
      std::vector<size_t> gnn_positions;
      exact_requests.reserve(tagore_requests.size());
      gnn_requests.reserve(tagore_requests.size());
      exact_positions.reserve(tagore_requests.size());
      gnn_positions.reserve(tagore_requests.size());
      for (size_t i = 0; i < tagore_requests.size(); ++i)
      {
         if (tagore_requests[i].num_points <= context.exact_batch_threshold)
         {
            exact_positions.push_back(i);
            exact_requests.push_back(tagore_requests[i]);
         }
         else
         {
            gnn_positions.push_back(i);
            gnn_requests.push_back(tagore_requests[i]);
         }
      }

      auto merge_batch = [&](TagoreBatchBuildResult &part, const std::vector<size_t> &positions) {
         artifacts.batch_timing.batch_pack_ms += part.batch_pack_ms;
         artifacts.batch_timing.batch_alloc_ms += part.batch_alloc_ms;
         artifacts.batch_timing.batch_free_ms += part.batch_free_ms;
         for (size_t i = 0; i < positions.size(); ++i)
            artifacts.results[positions[i]] = std::move(part.groups[i]);
      };

      if (!exact_requests.empty())
      {
         TagoreBatchBuildResult exact_batch = build_tagore_vamana_cuda_batch(
             exact_requests, dim, _build_config.tagore_k, static_cast<uint32_t>(_max_degree),
             _build_config.tagore_m, _build_config.tagore_iter, _alpha, context.prune_mode, context.runtime_config);
         for (size_t i = 0; i < exact_positions.size(); ++i)
            artifacts.exact_packed_rank[exact_positions[i]] = i;
         artifacts.exact_packed_graph = std::move(exact_batch.packed_graph);
         artifacts.exact_packed_offsets = std::move(exact_batch.packed_offsets);
         artifacts.exact_packed_stride = exact_batch.packed_graph_stride;
         merge_batch(exact_batch, exact_positions);
      }
      if (!gnn_requests.empty())
      {
         TagoreBatchBuildResult gnn_batch = build_tagore_vamana_cuda_batch(
             gnn_requests, dim, _build_config.tagore_k, static_cast<uint32_t>(_max_degree),
             _build_config.tagore_m, _build_config.tagore_iter, _alpha, context.prune_mode, context.runtime_config);
         merge_batch(gnn_batch, gnn_positions);
      }
      std::cout << "[TagoreCuda] mixed FastGrnnd batches: exact_groups=" << exact_requests.size()
                << " gnn_groups=" << gnn_requests.size()
                << " exact_nx_threshold=" << context.exact_batch_threshold
                << std::endl;
   }
   else
   {
      artifacts.batch_timing = build_tagore_vamana_cuda_batch(
          tagore_requests,
          dim,
          _build_config.tagore_k,
          static_cast<uint32_t>(_max_degree),
          _build_config.tagore_m,
          _build_config.tagore_iter,
          _alpha,
          context.prune_mode,
          context.runtime_config);
      if (!artifacts.batch_timing.packed_graph.empty())
      {
         artifacts.exact_packed_graph = std::move(artifacts.batch_timing.packed_graph);
         artifacts.exact_packed_offsets = std::move(artifacts.batch_timing.packed_offsets);
         artifacts.exact_packed_stride = artifacts.batch_timing.packed_graph_stride;
         for (size_t i = 0; i < artifacts.results.size(); ++i)
            artifacts.exact_packed_rank[i] = i;
      }
      artifacts.results = std::move(artifacts.batch_timing.groups);
   }
   artifacts.wall_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::high_resolution_clock::now() - batch_start)
                           .count();
   return artifacts;
}

void UniNavGraph::fill_tagore_batch_results(const std::vector<IdxType> &tagore_group_ids,
                                            const std::vector<TagoreBuildResult> &tagore_results,
                                            const std::vector<size_t> &exact_packed_rank,
                                            const std::vector<uint32_t> &exact_packed_graph,
                                            const std::vector<uint32_t> &exact_packed_offsets,
                                            uint32_t exact_packed_stride,
                                            const TagoreGroupBuildContext &context,
                                            TagoreBuildResult &timing_acc,
                                            double &fill_wall_ms,
                                            double &fill_cpu_sum_ms)
{
   double timing_convert_ms = 0.0;
   double timing_h2d_ms = 0.0;
   double timing_memset_ms = 0.0;
   double timing_gnn_ms = 0.0;
   double timing_prune_ms = 0.0;
   double timing_grnnd_refine_ms = 0.0;
   double timing_d2h_ms = 0.0;
   fill_cpu_sum_ms = 0.0;
   constexpr size_t npos = static_cast<size_t>(-1);
   const bool fast_fill = context.settings.fast_fill;
   const int fill_threads = context.settings.fill_threads;
   const auto fill_wall_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1) num_threads(fill_threads) reduction(+ : fill_cpu_sum_ms, timing_convert_ms, timing_h2d_ms, timing_memset_ms, timing_gnn_ms, timing_prune_ms, timing_grnnd_refine_ms, timing_d2h_ms)
   for (size_t gi = 0; gi < tagore_group_ids.size(); ++gi)
   {
      const IdxType group_id = tagore_group_ids[gi];
      const auto &range = _group_id_to_range[group_id];
      const IdxType num_points = range.second - range.first;
      const TagoreBuildResult &result = tagore_results[gi];
      auto graph = _group_graphs[group_id];
      const IdxType base_offset = _intra_group_graph_ids_are_global ? range.first : 0;
      const uint32_t k = result.graph_stride != 0
                             ? result.graph_stride
                             : (_build_config.tagore_k <= _max_degree ? static_cast<uint32_t>(_max_degree + 1)
                                                                       : _build_config.tagore_k);
      const uint32_t *graph_data = result.graph.empty() ? nullptr : result.graph.data();
      bool exact_packed_source = false;
      if (!graph_data && gi < exact_packed_rank.size() && exact_packed_rank[gi] != npos &&
          exact_packed_stride == k && !exact_packed_graph.empty() &&
          exact_packed_rank[gi] + 1 < exact_packed_offsets.size())
      {
         graph_data = exact_packed_graph.data() + static_cast<size_t>(exact_packed_offsets[exact_packed_rank[gi]]) * k;
         exact_packed_source = true;
      }
      if (!graph_data)
         throw std::runtime_error("TagoreCuda result graph is empty and no packed graph source is available.");
      const auto fill_start = std::chrono::high_resolution_clock::now();
      for (IdxType local_id = 0; local_id < num_points; ++local_id)
      {
         auto &neighbors = graph->neighbors[local_id];
         neighbors.clear();
         const uint32_t degree = graph_data[static_cast<size_t>(local_id) * k];
         if (fast_fill || exact_packed_source)
         {
            const uint32_t limit = std::min<uint32_t>(degree, _max_degree);
            if (exact_packed_source)
            {
               const uint32_t *begin = graph_data + static_cast<size_t>(local_id) * k + 1;
               neighbors.resize(limit);
               if (limit > 0 && base_offset == 0)
                  std::memcpy(neighbors.data(), begin, static_cast<size_t>(limit) * sizeof(IdxType));
               else
               {
                  for (uint32_t j = 0; j < limit; ++j)
                     neighbors[j] = static_cast<IdxType>(begin[j]) + base_offset;
               }
            }
            else
            {
               neighbors.resize(limit);
               uint32_t write = 0;
               for (uint32_t j = 0; j < degree && write < limit; ++j)
               {
                  const uint32_t neighbor = graph_data[static_cast<size_t>(local_id) * k + 1 + j];
                  if (neighbor < num_points && neighbor != local_id)
                     neighbors[write++] = static_cast<IdxType>(neighbor) + base_offset;
               }
               neighbors.resize(write);
            }
         }
         else
         {
            neighbors.reserve(std::min<uint32_t>(degree, _max_degree));
            for (uint32_t j = 0; j < degree && neighbors.size() < _max_degree; ++j)
            {
               const uint32_t neighbor = graph_data[static_cast<size_t>(local_id) * k + 1 + j];
               const IdxType neighbor_id = static_cast<IdxType>(neighbor) + base_offset;
               bool duplicate = false;
               for (IdxType existing : neighbors)
               {
                  if (existing == neighbor_id)
                  {
                     duplicate = true;
                     break;
                  }
               }
               if (neighbor < num_points && neighbor != local_id && !duplicate)
                  neighbors.emplace_back(neighbor_id);
            }
         }
      }
      _vamana_instances[group_id] = std::make_shared<Vamana>(_group_storages[group_id], _distance_handler,
                                                             _group_graphs[group_id], result.entry_point);
      _group_entry_points[group_id] = result.entry_point + range.first;
      fill_cpu_sum_ms += std::chrono::duration<double, std::milli>(
                             std::chrono::high_resolution_clock::now() - fill_start)
                             .count();

      timing_convert_ms += result.convert_ms;
      timing_h2d_ms += result.h2d_ms;
      timing_memset_ms += result.memset_ms;
      timing_gnn_ms += result.gnn_ms;
      timing_prune_ms += result.prune_ms;
      timing_grnnd_refine_ms += result.grnnd_refine_ms;
      timing_d2h_ms += result.d2h_ms;
   }
   fill_wall_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::high_resolution_clock::now() - fill_wall_start)
                      .count();
   timing_acc.convert_ms = timing_convert_ms;
   timing_acc.h2d_ms = timing_h2d_ms;
   timing_acc.memset_ms = timing_memset_ms;
   timing_acc.gnn_ms = timing_gnn_ms;
   timing_acc.prune_ms = timing_prune_ms;
   timing_acc.grnnd_refine_ms = timing_grnnd_refine_ms;
   timing_acc.d2h_ms = timing_d2h_ms;
}

} // namespace ANNS
