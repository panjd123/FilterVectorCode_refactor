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

#include "../../../ACORN/faiss/IndexACORN.h"
#include "../../../ACORN/faiss/index_io.h"
#include "utils.h"
#include "include/uni_nav_graph.h"
#include "include/MethodSelector.h"

namespace fs = boost::filesystem;

namespace ANNS
{
   size_t UniNavGraph::count_graph_edges() const
   {
      size_t total_edges = 0;
      for (const auto &neighbors : _vector_attr_graph)
      {
         total_edges += neighbors.size();
      }
      return total_edges / 2;
   }

   uint32_t UniNavGraph::compute_checksum() const
   {
      uint32_t sum = 0;
      for (const auto &neighbors : _vector_attr_graph)
      {
         for (IdxType node : neighbors)
         {
            sum ^= (node << (sum % 32));
         }
      }
      return sum;
   }

   void UniNavGraph::save_bipartite_graph(const std::string &filename)
   {
      std::ofstream out(filename, std::ios::binary);
      if (!out)
      {
         throw std::runtime_error("Cannot open file for writing: " + filename);
      }

      const char header[8] = {'B', 'I', 'P', 'G', 'R', 'P', 'H', '1'};
      out.write(header, 8);

      out.write(reinterpret_cast<const char *>(&_num_points), sizeof(IdxType));
      out.write(reinterpret_cast<const char *>(&_num_attributes), sizeof(AtrType));

      uint64_t map_size = _attr_to_id.size();
      out.write(reinterpret_cast<const char *>(&map_size), sizeof(uint64_t));

      for (const auto &[label, id] : _attr_to_id)
      {
         out.write(reinterpret_cast<const char *>(&label), sizeof(LabelType));
         out.write(reinterpret_cast<const char *>(&id), sizeof(AtrType));
      }

      uint64_t total_nodes = _vector_attr_graph.size();
      out.write(reinterpret_cast<const char *>(&total_nodes), sizeof(uint64_t));

      for (const auto &neighbors : _vector_attr_graph)
      {
         uint32_t neighbor_count = neighbors.size();
         out.write(reinterpret_cast<const char *>(&neighbor_count), sizeof(uint32_t));

         if (!neighbors.empty())
         {
            out.write(reinterpret_cast<const char *>(neighbors.data()),
                      neighbors.size() * sizeof(IdxType));
         }
      }

      uint32_t checksum = compute_checksum();
      out.write(reinterpret_cast<const char *>(&checksum), sizeof(uint32_t));

      std::cout << "Successfully saved bipartite graph to " << filename
                << " (" << out.tellp() << " bytes)" << std::endl;
   }

   void UniNavGraph::load_bipartite_graph(const std::string &filename)
   {
      std::cout << "Loading bipartite graph from " << filename << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      std::ifstream in(filename, std::ios::binary);
      if (!in)
      {
         throw std::runtime_error("Cannot open file for reading: " + filename);
      }

      char header[8];
      in.read(header, 8);
      if (std::string(header, 8) != "BIPGRPH1")
      {
         throw std::runtime_error("Invalid file format");
      }

      in.read(reinterpret_cast<char *>(&_num_points), sizeof(IdxType));
      in.read(reinterpret_cast<char *>(&_num_attributes), sizeof(AtrType));

      _attr_to_id.clear();
      _id_to_attr.clear();

      uint64_t map_size;
      in.read(reinterpret_cast<char *>(&map_size), sizeof(uint64_t));

      for (uint64_t i = 0; i < map_size; ++i)
      {
         LabelType label;
         AtrType id;

         in.read(reinterpret_cast<char *>(&label), sizeof(LabelType));
         in.read(reinterpret_cast<char *>(&id), sizeof(AtrType));

         _attr_to_id[label] = id;
         _id_to_attr[id] = label;
      }

      _vector_attr_graph.clear();

      uint64_t total_nodes;
      in.read(reinterpret_cast<char *>(&total_nodes), sizeof(uint64_t));
      _vector_attr_graph.resize(total_nodes);

      for (uint64_t i = 0; i < total_nodes; ++i)
      {
         uint32_t neighbor_count;
         in.read(reinterpret_cast<char *>(&neighbor_count), sizeof(uint32_t));

         _vector_attr_graph[i].resize(neighbor_count);
         if (neighbor_count > 0)
         {
            in.read(reinterpret_cast<char *>(_vector_attr_graph[i].data()),
                    neighbor_count * sizeof(IdxType));
         }
      }

      uint32_t stored_checksum;
      in.read(reinterpret_cast<char *>(&stored_checksum), sizeof(uint32_t));

      uint32_t computed_checksum = compute_checksum();
      if (stored_checksum != computed_checksum)
      {
         throw std::runtime_error("Checksum verification failed");
      }

      std::cout << "- Loaded bipartite graph with " << _num_points << " vectors and "
                << _num_attributes << " attributes in "
                << std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - start_time)
                       .count()
                << " ms" << std::endl;
   }
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
      meta_data["descendants_impl"] = to_string(_build_config.descendants_impl);
      meta_data["coverage_impl"] = to_string(_build_config.coverage_impl);
      meta_data["cross_edge_impl"] = to_string(_build_config.cross_edge_impl);
      meta_data["gpu_topk_impl"] = to_string(_build_config.gpu_topk_impl);
      meta_data["index_time(ms)"] = std::to_string(_index_time - _build_roaring_bitsets_time);
      meta_data["index_time_add_rb(ms)"] = std::to_string(_index_time);
      meta_data["label_processing_time(ms)"] = std::to_string(_label_processing_time);
      meta_data["build_graph_time(ms)"] = std::to_string(_build_graph_time);
      meta_data["build_vector_attr_graph_time(ms)"] = std::to_string(_build_vector_attr_graph_time);
      meta_data["cal_descendants_time(ms)"] = std::to_string(_cal_descendants_time);
      meta_data["cal_coverage_ratio_time(ms)"] = std::to_string(_cal_coverage_ratio_time);
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
      meta_data["special_blocks_enabled"] = std::to_string(_build_config.special_blocks_enabled ? 1 : 0);
      meta_data["special_block_skip_trivial"] = std::to_string(_build_config.special_block_skip_trivial ? 1 : 0);
      meta_data["special_block_min_points"] = std::to_string(_special_block_summary.threshold);
      meta_data["special_block_count"] = std::to_string(_special_block_summary.num_blocks);
      meta_data["special_block_trivial_count"] = std::to_string(_special_block_summary.trivial_blocks);
      meta_data["special_block_member_groups"] = std::to_string(_special_block_summary.member_groups);
      meta_data["special_block_member_points"] = std::to_string(_special_block_summary.member_points);
      meta_data["special_block_child_edges"] = std::to_string(_special_block_summary.child_block_edges);
      meta_data["special_edge_count"] = std::to_string(_special_block_summary.special_edges);
      meta_data["special_edge_intra_count"] = std::to_string(_special_block_summary.intra_special_edges);
      meta_data["special_edge_inter_count"] = std::to_string(_special_block_summary.inter_special_edges);
      meta_data["special_block_metadata_time(ms)"] = std::to_string(_special_block_summary.metadata_ms);
      meta_data["special_edge_overlay_time(ms)"] = std::to_string(_special_block_summary.edge_overlay_ms);
      meta_data["special_edge_intra_build_time(ms)"] = std::to_string(_special_block_summary.intra_edge_build_ms);
      meta_data["special_edge_inter_build_time(ms)"] = std::to_string(_special_block_summary.inter_edge_build_ms);
      meta_data["special_group_graph_trivial_skipped_groups"] = std::to_string(_special_block_summary.group_graph_trivial_skipped_groups);
      meta_data["special_group_graph_trivial_skipped_points"] = std::to_string(_special_block_summary.group_graph_trivial_skipped_points);
      meta_data["special_cross_trivial_skipped_pairs"] = std::to_string(_special_block_summary.cross_trivial_skipped_pairs);
      meta_data["special_cross_trivial_skipped_query_vectors"] = std::to_string(_special_block_summary.cross_trivial_skipped_query_vectors);
      meta_data["special_additional_trivial_skipped_groups"] = std::to_string(_special_block_summary.additional_trivial_skipped_groups);
      meta_data["special_additional_trivial_skipped_points"] = std::to_string(_special_block_summary.additional_trivial_skipped_points);

      std::cout << "Calculating and saving Trie static metrics..." << std::endl;
      TrieStaticMetrics trie_metrics = _trie_index.calculate_static_metrics();
      meta_data["trie_label_cardinality"] = std::to_string(trie_metrics.label_cardinality);
      meta_data["trie_total_nodes"] = std::to_string(trie_metrics.total_nodes);
      meta_data["trie_avg_path_length"] = std::to_string(trie_metrics.avg_path_length);
      meta_data["trie_avg_branching_factor"] = std::to_string(trie_metrics.avg_branching_factor);
      // Save the detailed label frequency distribution to its own file for easier analysis
      std::string trie_freq_filename = results_path_prefix + "trie_label_frequency.csv";
      std::ofstream freq_file(trie_freq_filename);
      freq_file << "LabelID,Frequency\n";
      for (const auto &pair : trie_metrics.label_frequency)
         freq_file << pair.first << "," << pair.second << "\n";
      freq_file.close();
      std::cout << "- Trie label frequency distribution saved to " << trie_freq_filename << std::endl;

      std::string meta_filename = index_path_prefix + "meta";
      write_kv_file(meta_filename, meta_data);

      // save build_time to csv
      std::string build_time_filename = results_path_prefix + "build_time.csv";
      std::ofstream build_time_file(build_time_filename);
      build_time_file << "Index Name,Build Time (ms)\n";
      build_time_file << "index_time" << "," << _index_time << "\n";
      build_time_file << "label_processing_time" << "," << _label_processing_time << "\n";
      build_time_file << "build_graph_time" << "," << _build_graph_time << "\n";
      build_time_file << "build_vector_attr_graph_time" << "," << _build_vector_attr_graph_time << "\n";
      build_time_file << "cal_descendants_time" << "," << _cal_descendants_time << "\n";
      build_time_file << "cal_coverage_ratio_time" << "," << _cal_coverage_ratio_time << "\n";
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
      build_time_file << "special_blocks_enabled" << "," << (_build_config.special_blocks_enabled ? 1 : 0) << "\n";
      build_time_file << "special_block_skip_trivial" << "," << (_build_config.special_block_skip_trivial ? 1 : 0) << "\n";
      build_time_file << "special_block_min_points" << "," << _special_block_summary.threshold << "\n";
      build_time_file << "special_block_metadata_time" << "," << _special_block_summary.metadata_ms << "\n";
      build_time_file << "special_block_count" << "," << _special_block_summary.num_blocks << "\n";
      build_time_file << "special_block_trivial_count" << "," << _special_block_summary.trivial_blocks << "\n";
      build_time_file << "special_block_member_groups" << "," << _special_block_summary.member_groups << "\n";
      build_time_file << "special_block_member_points" << "," << _special_block_summary.member_points << "\n";
      build_time_file << "special_block_child_edges" << "," << _special_block_summary.child_block_edges << "\n";
      build_time_file << "special_edge_overlay_time" << "," << _special_block_summary.edge_overlay_ms << "\n";
      build_time_file << "special_edge_intra_build_time" << "," << _special_block_summary.intra_edge_build_ms << "\n";
      build_time_file << "special_edge_inter_build_time" << "," << _special_block_summary.inter_edge_build_ms << "\n";
      build_time_file << "special_edge_count" << "," << _special_block_summary.special_edges << "\n";
      build_time_file << "special_edge_intra_count" << "," << _special_block_summary.intra_special_edges << "\n";
      build_time_file << "special_edge_inter_count" << "," << _special_block_summary.inter_special_edges << "\n";
      build_time_file << "special_group_graph_trivial_skipped_groups" << "," << _special_block_summary.group_graph_trivial_skipped_groups << "\n";
      build_time_file << "special_group_graph_trivial_skipped_points" << "," << _special_block_summary.group_graph_trivial_skipped_points << "\n";
      build_time_file << "special_cross_trivial_skipped_pairs" << "," << _special_block_summary.cross_trivial_skipped_pairs << "\n";
      build_time_file << "special_cross_trivial_skipped_query_vectors" << "," << _special_block_summary.cross_trivial_skipped_query_vectors << "\n";
      build_time_file << "special_additional_trivial_skipped_groups" << "," << _special_block_summary.additional_trivial_skipped_groups << "\n";
      build_time_file << "special_additional_trivial_skipped_points" << "," << _special_block_summary.additional_trivial_skipped_points << "\n";
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

      save_special_blocks(index_path_prefix);
      log_save_stage("special_blocks");

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

      if (!_build_config.is_original_cpu_pipeline())
      {
         // save LNG coverage ratio
         std::string coverage_ratio_filename = index_path_prefix + "lng_coverage_ratio";
         write_1d_vector(coverage_ratio_filename, _label_nav_graph->coverage_ratio);

         const bool skip_lng_text_sets = std::getenv("UNG_SKIP_LNG_TEXT_SETS") != nullptr;
         if (skip_lng_text_sets)
         {
            std::cout << "Skipping LNG text covered_sets/lng_descendants (UNG_SKIP_LNG_TEXT_SETS=1)." << std::endl;
         }
         else
         {
            // save covered_sets in LNG
            std::string covered_sets_filename = index_path_prefix + "covered_sets";
            write_2d_vectors(covered_sets_filename, _label_nav_graph->covered_sets);
            std::cout << "LNG covered_sets saved." << std::endl;

            // save LNG descendant num
            std::string lng_descendants_num_filename = index_path_prefix + "lng_descendants_num";
            write_1d_pair_vector(lng_descendants_num_filename, _label_nav_graph->_lng_descendants_num);

            // save LNG descendants
            std::string lng_descendants_filename = index_path_prefix + "lng_descendants";
            write_2d_vectors(lng_descendants_filename, _label_nav_graph->_lng_descendants);
         }
      }
      log_save_stage("lng_sets");

      // 保存 LNG 的核心图结构 (邻接表)
      std::string lng_out_neighbors_filename = index_path_prefix + "lng_out_neighbors.dat";
      write_2d_vectors(lng_out_neighbors_filename, _label_nav_graph->out_neighbors);
      std::cout << "LNG out_neighbors saved." << std::endl; // 增加一个打印，确认保存成功
      log_save_stage("lng_out_neighbors");

      if (!_build_config.is_original_cpu_pipeline())
      {
         // save vector attr graph data
         std::string vector_attr_graph_filename = index_path_prefix + "vector_attr_graph";
         save_bipartite_graph(vector_attr_graph_filename);
      }
      log_save_stage("vector_attr_graph");

      // // save _lng_descendants_bits and _covered_sets_bits
      // std::string lng_descendants_bits_filename = index_path_prefix + "lng_descendants_bits";
      // write_bitset_vector(lng_descendants_bits_filename, _lng_descendants_bits);
      // std::string covered_sets_bits_filename = index_path_prefix + "covered_sets_bits";
      // write_bitset_vector(covered_sets_bits_filename, _covered_sets_bits);

      if (!_build_config.is_original_cpu_pipeline())
      {
         // save lng_descendants_rb and _covered_sets_rb
         std::string lng_descendants_rb_filename = index_path_prefix + "lng_descendants_rb.bin";
         save_roaring_vector(lng_descendants_rb_filename, _lng_descendants_rb);
         std::string covered_sets_rb_filename = index_path_prefix + "covered_sets_rb.bin";
         save_roaring_vector(covered_sets_rb_filename, _covered_sets_rb);
      }
      log_save_stage("roaring_bitmaps");

      if (!_build_config.is_original_cpu_pipeline())
      {
      const bool skip_reordered_export = std::getenv("UNG_SKIP_REORDERED_EXPORT") != nullptr;
      if (skip_reordered_export)
      {
         std::cout << "\n--- Skipping reordered data export for ACORN index building (UNG_SKIP_REORDERED_EXPORT=1) ---"
                   << std::endl;
      }
      else
      {
      // save acorn index
      std::cout << "\n--- Exporting reordered data for ACORN index building ---" << std::endl;

      // 1. 导出重排后的向量
      std::string reordered_vec_path = index_path_prefix + "reordered_vecs.fvecs";
      std::cout << "- Saving reordered vectors to: " << reordered_vec_path << std::endl;
      try
      {
         uint32_t dim = static_cast<uint32_t>(_base_storage->get_dim());
         std::ofstream out(reordered_vec_path, std::ios::binary);
         if (!out)
         {
            throw std::runtime_error("Cannot open file for writing: " + reordered_vec_path);
         }

         // 遍历所有已重排的向量并按 .fvecs 格式写入
         for (ANNS::IdxType new_id = 0; new_id < _num_points; ++new_id)
         {
            const float *vec_data = reinterpret_cast<const float *>(_base_storage->get_vector(new_id));
            // 先写入维度
            out.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));
            // 再写入向量数据
            out.write(reinterpret_cast<const char *>(vec_data), dim * sizeof(float));
         }
         out.close();
         std::cout << "- Reordered vectors saved successfully." << std::endl;
      }
      catch (const std::exception &e)
      {
         std::cerr << "Error while saving reordered vectors: " << e.what() << std::endl;
      }

      // 2. 导出与重排后向量顺序完全一致的标签
      std::string reordered_label_path = index_path_prefix + "reordered_labels.txt";
      std::ofstream reordered_label_file(reordered_label_path);
      if (!reordered_label_file.is_open())
      {
         std::cerr << "Error: Could not open file to save reordered labels: " << reordered_label_path << std::endl;
      }
      else
      {
         for (ANNS::IdxType new_id = 0; new_id < _num_points; ++new_id)
         {
            const auto &label_set = _base_storage->get_label_set(new_id);
            for (size_t i = 0; i < label_set.size(); ++i)
            {
               reordered_label_file << label_set[i] << (i == label_set.size() - 1 ? "" : ",");
            }
            reordered_label_file << "\n";
         }
         reordered_label_file.close();
         std::cout << "- Reordered labels saved to: " << reordered_label_path << std::endl;
      }
	      std::cout << "--- Finished exporting reordered data ---\n"
	                << std::endl;
	      }
      }
      log_save_stage("reordered_export");

      // print
      std::cout << "- Index saved in " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
   }

   std::unordered_map<int, std::vector<int>> load_inverted_index(const std::string &input_path)
   {
      std::ifstream in(input_path, std::ios::binary);
      if (!in)
      {
         fprintf(stderr, "Error: Cannot open file for reading inverted index: %s\n", input_path.c_str());
         exit(1);
      }

      std::unordered_map<int, std::vector<int>> inverted_index;

      uint64_t map_size;
      in.read(reinterpret_cast<char *>(&map_size), sizeof(map_size));
      inverted_index.reserve(map_size);

      for (uint64_t i = 0; i < map_size; ++i)
      {
         int attr_id;
         uint64_t list_size;

         in.read(reinterpret_cast<char *>(&attr_id), sizeof(attr_id));
         in.read(reinterpret_cast<char *>(&list_size), sizeof(list_size));

         std::vector<int> vec_list(list_size);
         in.read(reinterpret_cast<char *>(vec_list.data()), list_size * sizeof(int));

         inverted_index[attr_id] = std::move(vec_list);
      }

      in.close();
      printf("Inverted index loaded successfully.\n");
      return inverted_index;
   }

   void UniNavGraph::load(std::string index_path_prefix, std::string selector_model_prefix, const std::string &data_type, const std::string &acorn_index_path, const std::string &acorn_1_index_path, const std::string &dataset)
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

      // load vectors and label sets
      std::string bin_file = index_path_prefix + "vecs.bin";
      std::string label_file = index_path_prefix + "labels.txt";
      _base_storage = create_storage(data_type, false);
      _base_storage->load_from_file(bin_file, label_file);

      // load group id to label set
      std::string group_id_to_label_set_filename = index_path_prefix + "group_id_to_label_set";
      load_2d_vectors(group_id_to_label_set_filename, _group_id_to_label_set);

      // load group id to range
      std::string group_id_to_range_filename = index_path_prefix + "group_id_to_range";
      load_2d_vectors(group_id_to_range_filename, _group_id_to_range);

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
      // --- 预计算并缓存 Trie 静态指标 ---
      std::cout << "Pre-calculating and caching Trie static metrics..." << std::endl;
      _trie_static_metrics = _trie_index.calculate_static_metrics();
      std::cout << "- Caching complete." << std::endl;

      // load graph data
      std::string graph_filename = index_path_prefix + "graph";
      _graph = std::make_shared<Graph>(_base_storage->get_num_points());
      _graph->load(graph_filename);

      std::string global_graph_filename = index_path_prefix + "global_graph";
      _global_graph = std::make_shared<Graph>(_base_storage->get_num_points());
      _global_graph->load(global_graph_filename);

      std::string global_vamana_entry_point_filename = index_path_prefix + "global_vamana_entry_point";
      load_one_T(global_vamana_entry_point_filename, _global_vamana_entry_point);

      std::string coverage_ratio_filename = index_path_prefix + "lng_coverage_ratio";
      std::cout << "_label_nav_graph->coverage_ratio size: " << _label_nav_graph->coverage_ratio.size() << std::endl;
      load_1d_vector(coverage_ratio_filename, _label_nav_graph->coverage_ratio);
      std::cout << "LNG coverage ratio loaded." << std::endl;

      std::string lng_descendants_num_filename = index_path_prefix + "lng_descendants_num";
      std::string lng_descendants_filename = index_path_prefix + "lng_descendants";
      std::string covered_sets_filename = index_path_prefix + "covered_sets";
      const bool has_lng_text_sets = fs::exists(lng_descendants_num_filename) &&
                                     fs::exists(lng_descendants_filename) &&
                                     fs::exists(covered_sets_filename);
      if (has_lng_text_sets)
      {
         load_1d_pair_vector(lng_descendants_num_filename, _label_nav_graph->_lng_descendants_num);
         std::cout << "LNG descendants num loaded." << std::endl;

         load_2d_vectors(lng_descendants_filename, _label_nav_graph->_lng_descendants);

         load_2d_vectors(covered_sets_filename, _label_nav_graph->covered_sets);
         std::cout << "LNG covered_sets loaded." << std::endl;
      }
      else
      {
         std::cout << "LNG text covered_sets/lng_descendants are absent; relying on roaring caches and out_neighbors."
                   << std::endl;
      }

      std::string lng_out_neighbors_filename = index_path_prefix + "lng_out_neighbors.dat";
      load_2d_vectors(lng_out_neighbors_filename, _label_nav_graph->out_neighbors);
      std::cout << "LNG out_neighbors loaded." << std::endl;

      // Optional legacy bitset caches; roaring caches are loaded below.
      // std::string lng_descendants_bits_filename = index_path_prefix + "lng_descendants_bits";
      // load_bitset_vector(lng_descendants_bits_filename, _lng_descendants_bits);
      // std::cout << "_lng_descendants_bits loaded." << std::endl;
      // std::string covered_sets_bits_filename = index_path_prefix + "covered_sets_bits";
      // load_bitset_vector(covered_sets_bits_filename, _covered_sets_bits);
      // std::cout << "_covered_sets_bits loaded." << std::endl;

      std::string lng_descendants_rb_filename = index_path_prefix + "lng_descendants_rb.bin";
      load_roaring_vector(lng_descendants_rb_filename, _lng_descendants_rb);
      std::cout << "_lng_descendants_rb loaded." << std::endl;
      std::string covered_sets_rb_filename = index_path_prefix + "covered_sets_rb.bin";
      load_roaring_vector(covered_sets_rb_filename, _covered_sets_rb);
      std::cout << "_covered_sets_rb loaded." << std::endl;
      std::cout << " _label_nav_graph->out_neighbors.size() = " << _label_nav_graph->out_neighbors.size() << std::endl;
      std::cout << " _num_groups = " << _num_groups << std::endl;

      // load idea1 selector
      std::string model_path = selector_model_prefix + "/idea1/idea1_selector_model_final.onnx";
      std::cout << "Loading Trie method selector model from " << model_path << " ..." << std::endl;
      try
      {
         if (fs::exists(model_path))
         {
            _trie_method_selector = std::make_unique<MethodSelector>(model_path);
            std::cout << "- Model loaded successfully." << std::endl;
         }
         else
         {
            std::cerr << "- WARNING: Model file not found. Will use default method passed by command line." << std::endl;
            _trie_method_selector = nullptr;
         }
      }
      catch (const std::exception &e)
      {
         std::cerr << "- ERROR: Failed to load model: " << e.what() << ". Will use default method." << std::endl;
         _trie_method_selector = nullptr;
      }

      // load idea2 selector
      std::string idea2_model_path = selector_model_prefix + "/idea2/idea2_selector_model_final.onnx";
      std::cout << "Loading Idea2 Selector model from " << idea2_model_path << " ..." << std::endl;
      if (fs::exists(idea2_model_path))
      {
         std::cout << "- Loading Idea2 Selector model from: " << idea2_model_path << std::endl;
         try
         {
            _ung_acorn_selector = std::make_unique<MethodSelector>(idea2_model_path);
         }
         catch (const std::runtime_error &e)
         {
            std::cerr << "  - [ERROR] Failed to initialize Idea2 Selector: " << e.what() << std::endl;
            _ung_acorn_selector = nullptr;
         }
      }
      else
      {
         std::cout << "- [INFO] Idea2 Selector model not found at: " << idea2_model_path << ". Selector will be disabled." << std::endl;
         _ung_acorn_selector = nullptr;
      }

      // === load ACORN index ===
      // Declare a shared inverted index and a flag to track if it's loaded.
      std::unordered_map<int, std::vector<int>> shared_inverted_index;
      bool inverted_index_loaded = false;

      if (!acorn_index_path.empty() && fs::exists(acorn_index_path))
      {
         std::cout << "Loading REORDERED ACORN index from " << acorn_index_path << " ..." << std::endl;
         try
         {
            faiss::Index *raw_index = faiss::read_index(acorn_index_path.c_str());
            _acorn_index = std::shared_ptr<faiss::IndexACORNFlat>(dynamic_cast<faiss::IndexACORNFlat *>(raw_index));

            if (_acorn_index)
            {
               std::cout << "- Associating metadata directly (reordered)..." << std::endl;
               std::vector<std::vector<int>> metadata(_num_points);
               for (size_t new_id = 0; new_id < _num_points; ++new_id)
               {
                  const auto &label_set = _base_storage->get_label_set(new_id);
                  metadata[new_id].assign(label_set.begin(), label_set.end());
                  std::sort(metadata[new_id].begin(), metadata[new_id].end());
               }
               _acorn_index->set_metadata(metadata);
               std::cout << "- ACORN index loaded and metadata associated successfully." << std::endl;

               std::string inverted_index_path = acorn_index_path + ".inverted_index";
               if (fs::exists(inverted_index_path))
               {
                  std::cout << "- Loading inverted index from: " << inverted_index_path << std::endl;
                  shared_inverted_index = load_inverted_index(inverted_index_path);
                  _acorn_index->set_inverted_index(shared_inverted_index);
                  inverted_index_loaded = true;
                  std::cout << "- Inverted index loaded and set successfully for ACORN." << std::endl;
               }
               else
               {
                  std::cerr << "  - [WARNING] Inverted index file not found at: " << inverted_index_path
                            << ". ACORN's original filtering (force_use_alg=3) will not work." << std::endl;
               }
            }
            else
            {
               std::cerr << "ERROR: Failed to cast loaded index to faiss::IndexACORNFlat." << std::endl;
               delete raw_index;
            }
         }
         catch (const std::exception &e)
         {
            std::cerr << "ERROR: Exception caught while loading ACORN index: " << e.what() << std::endl;
            _acorn_index = nullptr;
         }
      }
      else if (!acorn_index_path.empty())
      {
         std::cerr << "Warning: ACORN index path provided, but file not found at: " << acorn_index_path << std::endl;
      }

      // --- load ACORN-1 index ---
      if (!acorn_1_index_path.empty() && fs::exists(acorn_1_index_path))
      {
         std::cout << "Loading REORDERED ACORN-1 index from " << acorn_1_index_path << " ..." << std::endl;
         try
         {
            faiss::Index *raw_index_1 = faiss::read_index(acorn_1_index_path.c_str());
            _acorn_1_index = std::shared_ptr<faiss::IndexACORNFlat>(dynamic_cast<faiss::IndexACORNFlat *>(raw_index_1));

            if (_acorn_1_index)
            {
               std::cout << "- Associating metadata directly with ACORN-1 index (reordered)..." << std::endl;
               std::vector<std::vector<int>> metadata(_num_points);
               for (size_t new_id = 0; new_id < _num_points; ++new_id)
               {
                  const auto &label_set = _base_storage->get_label_set(new_id);
                  metadata[new_id].assign(label_set.begin(), label_set.end());
                  std::sort(metadata[new_id].begin(), metadata[new_id].end());
               }
               _acorn_1_index->set_metadata(metadata);
               std::cout << "- ACORN-1 index loaded and metadata associated successfully." << std::endl;

               // Replace the old loading logic to reuse the inverted index.
               if (inverted_index_loaded)
               {
                  std::cout << "- Reusing the loaded inverted index for ACORN-1." << std::endl;
                  _acorn_1_index->set_inverted_index(shared_inverted_index);
                  std::cout << "- Inverted index set successfully for ACORN-1." << std::endl;
               }
               else
               {
                  std::cerr << "  - [WARNING] Inverted index could not be set for ACORN-1 because the primary inverted index file was not found." << std::endl;
               }
            }
            else
            {
               std::cerr << "ERROR: Failed to cast loaded index to faiss::IndexACORNFlat from " << acorn_1_index_path << std::endl;
               delete raw_index_1;
            }
         }
         catch (const std::exception &e)
         {
            std::cerr << "ERROR: Exception caught while loading ACORN-1 index: " << e.what() << std::endl;
            _acorn_1_index = nullptr;
         }
      }
      else if (!acorn_1_index_path.empty())
      {
         std::cerr << "Warning: ACORN-1 index file not found at: " << acorn_1_index_path << std::endl;
      }

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

      _index_size_add_rb = _index_size;

      if (!_lng_descendants_rb.empty())
      {
         for (const auto &rb : _lng_descendants_rb)
         {
               _index_size_add_rb += rb.getSizeInBytes();
         }
      }
      if (!_covered_sets_rb.empty())
      {
         for (const auto &rb : _covered_sets_rb)
         {
               _index_size_add_rb += rb.getSizeInBytes();
         }
      }

      // return as MB
      _index_size /= 1024 * 1024;
      _index_size_add_rb /= 1024 * 1024;
   }
}
