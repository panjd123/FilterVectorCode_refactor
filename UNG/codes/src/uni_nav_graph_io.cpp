#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/filesystem.hpp>

#include "utils.h"
#include "include/uni_nav_graph.h"
#include "include/ung_group_topology.h"

namespace fs = boost::filesystem;

namespace ANNS
{

   void UniNavGraph::save(std::string index_path_prefix, std::string results_path_prefix)
   {
      fs::create_directories(index_path_prefix);
      auto start_time = std::chrono::high_resolution_clock::now();
      auto save_stage_start_time = start_time;
      auto log_save_stage = [&](const char *stage_name) {
         const auto now = std::chrono::high_resolution_clock::now();
         std::cout << "[save][stage] " << stage_name << "_ms="
                   << std::chrono::duration<double, std::milli>(now - save_stage_start_time).count()
                   << std::endl;
         save_stage_start_time = now;
      };

      // save meta data
      std::map<std::string, std::string> meta_data;
      statistics();
      meta_data["num_points"] = std::to_string(_num_points);
      meta_data["num_groups"] = std::to_string(_num_groups);
      meta_data["index_name"] = _index_name;
      meta_data["max_degree"] = std::to_string(_max_degree);
      meta_data["Lbuild"] = std::to_string(_Lbuild);
      meta_data["alpha"] = std::to_string(_alpha);
      meta_data["build_num_threads"] = std::to_string(_num_threads);
      meta_data["scenario"] = _scenario;
      meta_data["num_cross_edges"] = std::to_string(_num_cross_edges);
      meta_data["graph_num_edges"] = std::to_string(_graph_num_edges);
      meta_data["LNG_num_edges"] = std::to_string(_LNG_num_edges);
      meta_data["index_size(MB)"] = std::to_string(_index_size);
      meta_data["_index_size_add_rb(MB)"] = std::to_string(_index_size_add_rb);
      meta_data["build_profile"] = to_string(_build_config.profile);
      meta_data["group_graph_impl"] = to_string(_build_config.group_graph_impl);
      meta_data["get_min_super_sets_impl"] = to_string(_build_config.get_min_super_sets_impl);
      meta_data["lng_impl"] = to_string(_build_config.lng_impl);
      meta_data["base_group_topology"] = to_string(_build_config.hierarchy.base_topology);
      meta_data["lng_constructed"] =
          _build_config.hierarchy.base_topology == GroupTopologyKind::Lng ? "1" : "0";
      meta_data["hierarchy_layers"] = _build_config.hierarchy.encode();
      meta_data["search_auxiliary_built"] = "0";
      meta_data["cross_edge_impl"] = to_string(_build_config.cross_edge_impl);
      meta_data["additional_edges_impl"] = to_string(_build_config.additional_edges_impl);
      meta_data["trie_cross_source_exact"] =
          _build_config.trie_cross_source_exact ? "1" : "0";
      meta_data["gpu_topk_impl"] = to_string(_build_config.gpu_topk_impl);
      meta_data["index_time(ms)"] = std::to_string(_index_time);
      meta_data["index_time_add_rb(ms)"] = std::to_string(_index_time);
      meta_data["label_processing_time(ms)"] = std::to_string(_label_processing_time);
      meta_data["build_graph_time(ms)"] = std::to_string(_build_graph_time);
      meta_data["build_LNG_time(ms)"] = std::to_string(_build_LNG_time);
      meta_data["build_cross_edges_time(ms)"] = std::to_string(_build_cross_edges_time);
      meta_data["tagore_groups"] = std::to_string(_tagore_groups);
      meta_data["tagore_points"] = std::to_string(_tagore_points);
      meta_data["tagore_direct_build_wall_time(ms)"] = std::to_string(_tagore_direct_build_wall_time_ms);
      meta_data["tagore_no_alloc_build_time(ms)"] = std::to_string(_tagore_no_alloc_build_time_ms);
      meta_data["tagore_pack_time(ms)"] = std::to_string(_tagore_pack_time_ms);
      meta_data["tagore_convert_time(ms)"] = std::to_string(_tagore_convert_time_ms);
      meta_data["tagore_workspace_alloc_time(ms)"] = std::to_string(_tagore_workspace_alloc_time_ms);
      meta_data["tagore_h2d_time(ms)"] = std::to_string(_tagore_h2d_time_ms);
      meta_data["tagore_memset_time(ms)"] = std::to_string(_tagore_memset_time_ms);
      meta_data["tagore_gnn_time(ms)"] = std::to_string(_tagore_gnn_time_ms);
      meta_data["tagore_prune_time(ms)"] = std::to_string(_tagore_prune_time_ms);
      meta_data["tagore_grnnd_refine_time(ms)"] = std::to_string(_tagore_grnnd_refine_time_ms);
      meta_data["tagore_d2h_time(ms)"] = std::to_string(_tagore_d2h_time_ms);
      meta_data["tagore_fill_time(ms)"] = std::to_string(_tagore_fill_time_ms);
      meta_data["tagore_workspace_free_time(ms)"] = std::to_string(_tagore_workspace_free_time_ms);
      meta_data["tagore_unaccounted_time(ms)"] = std::to_string(_tagore_unaccounted_time_ms);
      meta_data["tagore_h2d_effective_gbps"] = std::to_string(_tagore_h2d_effective_gbps);
      meta_data["tagore_d2h_effective_gbps"] = std::to_string(_tagore_d2h_effective_gbps);
      meta_data["tagore_gnn_mpts_s"] = std::to_string(_tagore_gnn_mpts_s);
      meta_data["tagore_prune_mpts_s"] = std::to_string(_tagore_prune_mpts_s);
      meta_data["cross_edge_step1_time(ms)"] = std::to_string(_cross_edge_step1_time_ms);
      meta_data["cross_edge_step2_acorn_time(ms)"] = std::to_string(_cross_edge_step2_acorn_time_ms);
      meta_data["cross_edge_step3_add_dist_edges_time(ms)"] = std::to_string(_cross_edge_step3_add_dist_edges_time_ms);
      meta_data["cross_edge_step4_add_hierarchy_edges_time(ms)"] = std::to_string(_cross_edge_step4_add_hierarchy_edges_time_ms);
      fs::remove(results_path_prefix + "trie_label_frequency.csv");

      std::string meta_filename = index_path_prefix + "meta";
      write_kv_file(meta_filename, meta_data);

      // save build_time to csv
      std::string build_time_filename = results_path_prefix + "build_time.csv";
      std::ofstream build_time_file(build_time_filename);
      build_time_file << "Index Name,Build Time (ms)\n";
      build_time_file << "index_time" << "," << _index_time << "\n";
      build_time_file << "label_processing_time" << "," << _label_processing_time << "\n";
      build_time_file << "build_graph_time" << "," << _build_graph_time << "\n";
      build_time_file << "build_LNG_time" << "," << _build_LNG_time << "\n";
      build_time_file << "build_cross_edges_time" << "," << _build_cross_edges_time << "\n";
      build_time_file << "tagore_groups" << "," << _tagore_groups << "\n";
      build_time_file << "tagore_points" << "," << _tagore_points << "\n";
      build_time_file << "tagore_direct_build_wall_time" << "," << _tagore_direct_build_wall_time_ms << "\n";
      build_time_file << "tagore_no_alloc_build_time" << "," << _tagore_no_alloc_build_time_ms << "\n";
      build_time_file << "tagore_pack_time" << "," << _tagore_pack_time_ms << "\n";
      build_time_file << "tagore_convert_time" << "," << _tagore_convert_time_ms << "\n";
      build_time_file << "tagore_workspace_alloc_time" << "," << _tagore_workspace_alloc_time_ms << "\n";
      build_time_file << "tagore_h2d_time" << "," << _tagore_h2d_time_ms << "\n";
      build_time_file << "tagore_memset_time" << "," << _tagore_memset_time_ms << "\n";
      build_time_file << "tagore_gnn_time" << "," << _tagore_gnn_time_ms << "\n";
      build_time_file << "tagore_prune_time" << "," << _tagore_prune_time_ms << "\n";
      build_time_file << "tagore_grnnd_refine_time" << "," << _tagore_grnnd_refine_time_ms << "\n";
      build_time_file << "tagore_d2h_time" << "," << _tagore_d2h_time_ms << "\n";
      build_time_file << "tagore_fill_time" << "," << _tagore_fill_time_ms << "\n";
      build_time_file << "tagore_workspace_free_time" << "," << _tagore_workspace_free_time_ms << "\n";
      build_time_file << "tagore_unaccounted_time" << "," << _tagore_unaccounted_time_ms << "\n";
      build_time_file << "tagore_h2d_effective_gbps" << "," << _tagore_h2d_effective_gbps << "\n";
      build_time_file << "tagore_d2h_effective_gbps" << "," << _tagore_d2h_effective_gbps << "\n";
      build_time_file << "tagore_gnn_mpts_s" << "," << _tagore_gnn_mpts_s << "\n";
      build_time_file << "tagore_prune_mpts_s" << "," << _tagore_prune_mpts_s << "\n";
      build_time_file << "cross_edge_step1_time" << "," << _cross_edge_step1_time_ms << "\n";
      build_time_file << "cross_edge_step2_acorn_time" << "," << _cross_edge_step2_acorn_time_ms << "\n";
      build_time_file << "cross_edge_step3_add_dist_edges_time" << "," << _cross_edge_step3_add_dist_edges_time_ms << "\n";
      build_time_file << "cross_edge_step4_add_hierarchy_edges_time" << "," << _cross_edge_step4_add_hierarchy_edges_time_ms << "\n";
      build_time_file.close();
      log_save_stage("meta_and_build_time");

      // save vectors and label sets
      std::string bin_file = index_path_prefix + "vecs.bin";
      std::string label_file = index_path_prefix + "labels.txt";
      _base_storage->write_to_file(bin_file, label_file);
      log_save_stage("base_storage");

      // save group id to label set
      std::string group_id_to_label_set_filename = index_path_prefix + "group_id_to_label_set";
      write_2d_vectors(group_id_to_label_set_filename, _group_id_to_label_set);

      // save group id to range
      std::string group_id_to_range_filename = index_path_prefix + "group_id_to_range";
      write_2d_vectors(group_id_to_range_filename, _group_id_to_range);

      // save group id to entry point
      std::string group_entry_points_filename = index_path_prefix + "group_entry_points";
      write_1d_vector(group_entry_points_filename, _group_entry_points);

      // save group id to vec ids
      std::string group_id_to_vec_ids_filename = index_path_prefix + "group_id_to_vec_ids.dat";
      write_2d_vectors(group_id_to_vec_ids_filename, _group_id_to_vec_ids);
      std::cout << "group_id_to_vec_ids saved." << std::endl;
      log_save_stage("group_metadata");

      auto graph_stage_start = std::chrono::high_resolution_clock::now();
      auto log_graph_substage = [&](const char *stage_name) {
         const auto now = std::chrono::high_resolution_clock::now();
         std::cout << "[save][graph_trie] " << stage_name << "_ms="
                   << std::chrono::duration<double, std::milli>(now - graph_stage_start).count()
                   << std::endl;
         graph_stage_start = now;
      };

      // save new to old vec ids
      std::string new_to_old_vec_ids_filename = index_path_prefix + "new_to_old_vec_ids";
      write_1d_vector(new_to_old_vec_ids_filename, _new_to_old_vec_ids);
      log_graph_substage("new_to_old_vec_ids");

      // save trie index
      std::string trie_filename = index_path_prefix + "trie";
      _trie_index.save(trie_filename);
      log_graph_substage("trie");

      // save graph data
      std::string graph_filename = index_path_prefix + "graph";
      _graph->save(graph_filename);
      log_graph_substage("graph");

      std::string global_graph_filename = index_path_prefix + "global_graph";
      _global_graph->save(global_graph_filename);
      log_graph_substage("global_graph");

      std::string global_vamana_entry_point_filename = index_path_prefix + "global_vamana_entry_point";
      write_one_T(global_vamana_entry_point_filename, _global_vamana_entry_point);
      log_graph_substage("global_entry_point");
      log_save_stage("graph_and_trie");

      // Remove obsolete caches when rebuilding an existing index prefix.
      for (const char *name : {"lng_coverage_ratio", "covered_sets", "lng_descendants_num",
                               "lng_descendants", "lng_descendants_rb.bin", "covered_sets_rb.bin",
                               "lng_descendants_bits", "covered_sets_bits", "vector_attr_graph",
                               "reordered_vecs.fvecs", "reordered_labels.txt"})
         fs::remove(index_path_prefix + name);

      // 保存 LNG 的核心图结构 (邻接表)
      std::string lng_out_neighbors_filename = index_path_prefix + "lng_out_neighbors.dat";
      write_2d_vectors(lng_out_neighbors_filename, _label_nav_graph->out_neighbors);
      std::cout << "LNG out_neighbors saved." << std::endl; // 增加一个打印，确认保存成功
      log_save_stage("lng_out_neighbors");

      // print
      std::cout << "- Index saved in " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
   }

   void UniNavGraph::load(std::string index_path_prefix, const std::string &data_type, const std::string &dataset)
   {
      std::cout << "Loading index from " << index_path_prefix << " ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      _dataset = dataset;

      // load meta data
      std::string meta_filename = index_path_prefix + "meta";
      auto meta_data = parse_kv_file(meta_filename);
      _num_points = std::stoi(meta_data["num_points"]);
      _num_groups = std::stoi(meta_data["num_groups"]);
      _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups + 1);
      const auto base_topology_it = meta_data.find("base_group_topology");
      const GroupTopologyKind base_topology = parse_group_topology(
          base_topology_it == meta_data.end() ? "lng" : base_topology_it->second);
      const auto hierarchy_layers_it = meta_data.find("hierarchy_layers");
      _build_config.hierarchy = parse_hierarchy_plan(
          hierarchy_layers_it == meta_data.end() ? "" : hierarchy_layers_it->second,
          base_topology);

      // load vectors and label sets
      std::string bin_file = index_path_prefix + "vecs.bin";
      std::string label_file = index_path_prefix + "labels.txt";
      _base_storage = create_storage(data_type, false);
      _base_storage->load_from_file(bin_file, label_file);

      // load group id to label set
      std::string group_id_to_label_set_filename = index_path_prefix + "group_id_to_label_set";
      load_2d_vectors(group_id_to_label_set_filename, _group_id_to_label_set);
      _group_trie_index.build(_group_id_to_label_set, _num_groups);

      // load group id to range
      std::string group_id_to_range_filename = index_path_prefix + "group_id_to_range";
      load_2d_vectors(group_id_to_range_filename, _group_id_to_range);
      _new_vec_id_to_group_id.assign(_num_points, 0);
      for (IdxType group_id = 1; group_id < _group_id_to_range.size(); ++group_id)
      {
         const auto &range = _group_id_to_range[group_id];
         for (IdxType point_id = range.first;
              point_id < range.second && point_id < _new_vec_id_to_group_id.size();
              ++point_id)
            _new_vec_id_to_group_id[point_id] = group_id;
      }

      load_special_blocks(index_path_prefix, meta_data);

      // load group id to entry point
      std::string group_entry_points_filename = index_path_prefix + "group_entry_points";
      load_1d_vector(group_entry_points_filename, _group_entry_points);

      // load group id to vec ids
      std::string group_id_to_vec_ids_filename = index_path_prefix + "group_id_to_vec_ids.dat";
      load_2d_vectors(group_id_to_vec_ids_filename, _group_id_to_vec_ids);
      std::cout << "group_id_to_vec_ids loaded." << std::endl;

      // load new to old vec ids
      std::string new_to_old_vec_ids_filename = index_path_prefix + "new_to_old_vec_ids";
      load_1d_vector(new_to_old_vec_ids_filename, _new_to_old_vec_ids);

      std::cout << "- Building reverse ID map (original_id -> new_id)..." << std::endl;
      _old_to_new_vec_ids.resize(_num_points);
      for (IdxType new_id = 0; new_id < _num_points; ++new_id)
      {
         IdxType old_id = _new_to_old_vec_ids[new_id];
         _old_to_new_vec_ids[old_id] = new_id;
      }
      std::cout << "- Reverse ID map built successfully." << std::endl;

      // load trie index
      std::string trie_filename = index_path_prefix + "trie";
      _trie_index.load(trie_filename);
      // load graph data
      std::string graph_filename = index_path_prefix + "graph";
      _graph = std::make_shared<Graph>(_base_storage->get_num_points());
      _graph->load(graph_filename);

      std::string global_graph_filename = index_path_prefix + "global_graph";
      _global_graph = std::make_shared<Graph>(_base_storage->get_num_points());
      _global_graph->load(global_graph_filename);

      std::string global_vamana_entry_point_filename = index_path_prefix + "global_vamana_entry_point";
      load_one_T(global_vamana_entry_point_filename, _global_vamana_entry_point);

      std::string lng_out_neighbors_filename = index_path_prefix + "lng_out_neighbors.dat";
      load_2d_vectors(lng_out_neighbors_filename, _label_nav_graph->out_neighbors);
      _label_nav_graph->in_neighbors.assign(_num_groups + 1, {});
      for (IdxType source_group_id = 1;
           source_group_id < _label_nav_graph->out_neighbors.size(); ++source_group_id)
      {
         for (IdxType target_group_id : _label_nav_graph->out_neighbors[source_group_id])
         {
            if (target_group_id > 0 && target_group_id <= _num_groups)
               _label_nav_graph->in_neighbors[target_group_id].push_back(source_group_id);
         }
      }
      _group_topology_graph = build_group_topology(
          _build_config.hierarchy.base_topology, _group_id_to_label_set,
          _num_groups, _label_nav_graph);
      std::cout << "LNG out_neighbors loaded." << std::endl;

      // print
      std::cout << "- Index loaded in " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
   }

   void UniNavGraph::statistics()
   {
      // number of edges in the unified navigating graph
      _graph_num_edges = 0;
      for (IdxType i = 0; i < _num_points; ++i)
         _graph_num_edges += _graph->neighbors[i].size();

      // number of edges in the label navigating graph
      _LNG_num_edges = 0;
      if (_label_nav_graph != nullptr)
         for (IdxType i = 1; i <= _num_groups; ++i)
               _LNG_num_edges += _label_nav_graph->out_neighbors[i].size();

      // index size
      _index_size = 0;
      for (IdxType i = 1; i <= _num_groups; ++i)
         _index_size += _group_id_to_label_set[i].size() * sizeof(LabelType);
      _index_size += _group_id_to_range.size() * sizeof(IdxType) * 2;
      _index_size += _group_entry_points.size() * sizeof(IdxType);
      _index_size += _new_to_old_vec_ids.size() * sizeof(IdxType);
      _index_size += _trie_index.get_index_size();
      _index_size += _graph->get_index_size();

      // --- 统计 _label_nav_graph (LNG) 的大小 ---
      size_t lng_graph_size = 0;
      if (_label_nav_graph != nullptr)
      {
         // 1. 统计 out_neighbors (基于 .size())
         for (const auto &neighbors : _label_nav_graph->out_neighbors)
         {
               lng_graph_size += neighbors.size() * sizeof(ANNS::IdxType);
         }
         // 2. 统计 in_neighbors (基于 .size())
         for (const auto &neighbors : _label_nav_graph->in_neighbors)
         {
               lng_graph_size += neighbors.size() * sizeof(ANNS::IdxType);
         }
      }
      _index_size += lng_graph_size;

      size_t group_to_vec_size = 0;
      for (const auto &vec_ids : _group_id_to_vec_ids)
      {
         group_to_vec_size += vec_ids.size() * sizeof(ANNS::IdxType);
      }
      _index_size += group_to_vec_size;

      size_t special_block_size = 0;
      special_block_size += _special_blocks.size() * sizeof(SpecialBlock);
      for (const SpecialBlock &block : _special_blocks)
      {
         special_block_size += block.root_labels.size() * sizeof(LabelType);
         special_block_size += block.common_labels.size() * sizeof(LabelType);
         special_block_size += block.member_group_ids.size() * sizeof(IdxType);
         special_block_size += block.child_block_ids.size() * sizeof(IdxType);
      }
      for (const auto &owners : _group_to_special_block_by_level)
         special_block_size += owners.size() * sizeof(IdxType);
      special_block_size += _group_is_special_block_root.size() * sizeof(uint8_t);
      special_block_size += _group_is_trivial_special_block_root.size() * sizeof(uint8_t);
      for (const auto &owners : _point_to_special_block_by_level)
         special_block_size += owners.size() * sizeof(IdxType);
      special_block_size += _point_is_special_block_root.size() * sizeof(uint8_t);
      special_block_size += _special_edges_by_point.size() * sizeof(std::vector<SpecialEdge>);
      for (const auto &edges : _special_edges_by_point)
         special_block_size += edges.size() * sizeof(SpecialEdge);
      special_block_size += _special_heavy_edges_by_point.size() * sizeof(std::vector<SpecialEdge>);
      for (const auto &edges : _special_heavy_edges_by_point)
         special_block_size += edges.size() * sizeof(SpecialEdge);
      special_block_size += _special_edges_csr.offsets.capacity() * sizeof(uint64_t);
      special_block_size += _special_edges_csr.edges.capacity() * sizeof(PackedSpecialEdge);
      special_block_size += _special_heavy_edges_csr.offsets.capacity() * sizeof(uint64_t);
      special_block_size += _special_heavy_edges_csr.edges.capacity() * sizeof(PackedSpecialEdge);
      special_block_size += static_cast<size_t>(_special_block_trie_index.memory_size_bytes());
      _special_block_summary.index_bytes = static_cast<uint64_t>(special_block_size);
      _index_size += special_block_size;

      _index_size_add_rb = _index_size;

      // return as MB
      _index_size /= 1024 * 1024;
      _index_size_add_rb /= 1024 * 1024;
   }
}
