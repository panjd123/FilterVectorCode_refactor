#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils.h"
#include "include/uni_nav_graph.h"

namespace ANNS
{
namespace
{
   std::string get_required_acorn_script_path()
   {
      const char *script_path = std::getenv("UNG_ACORN_SCRIPT_PATH");
      if (script_path != nullptr && script_path[0] != '\0')
         return script_path;
      return {};
   }

   std::string shell_quote(const std::string &value)
   {
      std::string quoted = "'";
      for (char c : value)
      {
         if (c == '\'')
            quoted += "'\\''";
         else
            quoted += c;
      }
      quoted += "'";
      return quoted;
   }

   struct NewEdgeCandidate
   {
      IdxType from;
      IdxType to;
      float distance;

      bool operator<(const NewEdgeCandidate &other) const
      {
         return distance < other.distance;
      }
   };
}

   void UniNavGraph::add_new_distance_oriented_edges(
       const std::string &dataset,
       uint32_t num_threads,
       ANNS::AcornInUng new_cross_edge)
   {
      assert(_base_storage != nullptr && "Error: _base_storage is not initialized!");
      assert(_label_nav_graph != nullptr && "Error: _label_nav_graph is not initialized! Did you call build_label_nav_graph()?");
      assert(!_vamana_instances.empty() && "Error: _vamana_instances is empty! Did you call build_graph_for_all_groups()?");
      assert(_vamana_instances.size() == _num_groups + 1 && "Error: _vamana_instances has incorrect size!");

      bool add_acorn_edges = (new_cross_edge.new_edge_policy == "acorn_and_guarantee" || new_cross_edge.new_edge_policy == "just_acorn");
      bool add_guarantee_edges = (new_cross_edge.new_edge_policy == "acorn_and_guarantee" || new_cross_edge.new_edge_policy == "just_guarantee");
      auto all_start_time = std::chrono::high_resolution_clock::now();

      _my_new_edges_set.clear();

      // Legacy ACORN augmentation path: identify candidate query vectors, run
      // the external ACORN script, then convert its output into graph edges.
      if (add_acorn_edges)
      {
         std::cout << "\n========================================================" << std::endl;
         std::cout << "--- Augmenting Graph with New Distance-Oriented Edges (ACORN) ---" << std::endl;
         std::cout << "========================================================" << std::endl;
         auto acorn_part_start_time = std::chrono::high_resolution_clock::now();

         const size_t MAX_ID_SAMPLES = 15;
         const size_t SAMPLING_THRESHOLD = 100000;

         std::cout << "\n--- [Cross-Edge Sub-step 1: Identifying & Sampling Query Vectors] ---" << std::endl;
         auto step1_start_time = std::chrono::high_resolution_clock::now();

         std::cout << "[Step 0] Building reverse ID map for safe lookup..." << std::endl;
         std::unordered_map<IdxType, IdxType> old_to_new_map;
         old_to_new_map.reserve(_num_points);
         for (IdxType new_id = 0; new_id < _num_points; ++new_id)
         {
            old_to_new_map[_new_to_old_vec_ids[new_id]] = new_id;
         }
         std::cout << "  - Reverse ID map built successfully." << std::endl;

         std::cout << "[Step 1.1] Discovering conceptual root attributes via global coverage..." << std::endl;
         std::unordered_map<LabelType, size_t> attribute_counts;
         for (IdxType i = 0; i < _num_points; ++i)
         {
            for (const auto &label : _base_storage->get_label_set(i))
            {
               attribute_counts[label]++;
            }
         }
         std::unordered_set<LabelType> conceptual_root_labels;
         for (const auto &pair : attribute_counts)
         {
            if (static_cast<float>(pair.second) / _num_points >= new_cross_edge.root_coverage_threshold)
            {
               conceptual_root_labels.insert(pair.first);
            }
         }
         if (conceptual_root_labels.empty())
         {
            std::cerr << "  - WARNING: No conceptual root attributes found. Aborting edge addition." << std::endl;
            return;
         }
         std::cout << "  - Discovered " << conceptual_root_labels.size() << " conceptual root attributes." << std::endl;

         std::cout << "  	- Conceptual Root Labels are: { ";
         for (const auto &label : conceptual_root_labels)
         {
            std::cout << label << " ";
         }
         std::cout << "}" << std::endl;

         // --- Step 1.2: 根据层级比例识别候选组 ---
         std::cout << "[Step 1.2] Precisely identifying candidate groups using layer depth ratio..." << std::endl;
         size_t max_observed_depth = 0;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            if (_group_id_to_label_set[group_id].size() > max_observed_depth)
            {
               max_observed_depth = _group_id_to_label_set[group_id].size();
            }
         }
         int target_max_depth = std::max(1, static_cast<int>(max_observed_depth * new_cross_edge.layer_depth_retio));
         std::cout << "  - Max observed layer depth in dataset: " << max_observed_depth << std::endl;
         std::cout << "  - Using layer_depth_ratio=" << new_cross_edge.layer_depth_retio << ", target max depth is " << target_max_depth << std::endl;
         std::unordered_set<IdxType> candidate_groups;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            const auto &label_set = _group_id_to_label_set[group_id];
            if (label_set.empty() || label_set.size() > target_max_depth)
               continue;
            for (const auto &label : label_set)
            {
               if (conceptual_root_labels.count(label))
               {
                  candidate_groups.insert(group_id);
                  break;
               }
            }
         }
         std::unordered_set<IdxType> topmost_group_ids;
         for (IdxType group_id : candidate_groups)
         {
            bool is_topmost = true;
            for (IdxType parent_id : _label_nav_graph->in_neighbors[group_id])
            {
               if (candidate_groups.count(parent_id))
               {
                  is_topmost = false;
                  break;
               }
            }
            if (is_topmost)
            {
               topmost_group_ids.insert(group_id);
            }
         }
         std::cout << "  - Identified " << candidate_groups.size() << " total candidate groups within target depth." << std::endl;
         std::cout << "  - Identified " << topmost_group_ids.size() << " true topmost groups among them." << std::endl;

         std::cout << "[Step 1.3] Separating vectors into 'topmost' and 'subsequent' buckets..." << std::endl;
         std::vector<IdxType> topmost_vec_ids, subsequent_vec_ids;
         std::vector<float *> topmost_vec_data, subsequent_vec_data;
         std::vector<std::vector<LabelType>> topmost_vec_labels, subsequent_vec_labels;
         for (IdxType group_id : candidate_groups)
         {
            const auto &range = _group_id_to_range[group_id];
            const auto &label_set = _group_id_to_label_set[group_id];
            if (topmost_group_ids.count(group_id))
            {
               for (IdxType u = range.first; u < range.second; ++u)
               {
                  topmost_vec_ids.push_back(u);
                  topmost_vec_data.push_back(const_cast<float *>(reinterpret_cast<const float *>(_base_storage->get_vector(u))));
                  topmost_vec_labels.push_back(label_set);
               }
            }
            else
            {
               for (IdxType u = range.first; u < range.second; ++u)
               {
                  subsequent_vec_ids.push_back(u);
                  subsequent_vec_data.push_back(const_cast<float *>(reinterpret_cast<const float *>(_base_storage->get_vector(u))));
                  subsequent_vec_labels.push_back(label_set);
               }
            }
         }
         std::cout << "  - Topmost-layer bucket size: " << topmost_vec_ids.size() << " vectors." << std::endl;
         std::cout << "  - Subsequent-layer bucket size: " << subsequent_vec_ids.size() << " vectors." << std::endl;

         std::cout << "[Step 1.4] Applying stratified sampling logic..." << std::endl;
         std::vector<IdxType> final_query_vec_ids;
         std::vector<float *> final_query_vec_data;
         std::vector<std::vector<LabelType>> final_query_vec_labels;
         size_t total_candidates_vecs = topmost_vec_ids.size() + subsequent_vec_ids.size();
         const auto original_topmost_vec_ids = topmost_vec_ids;

         if (total_candidates_vecs < SAMPLING_THRESHOLD || new_cross_edge.query_vector_ratio >= 1.0f)
         {
            std::cout << "  - Decision: Using ALL candidates without sampling." << std::endl;
            final_query_vec_ids = std::move(topmost_vec_ids);
            final_query_vec_ids.insert(final_query_vec_ids.end(), subsequent_vec_ids.begin(), subsequent_vec_ids.end());
            final_query_vec_data = std::move(topmost_vec_data);
            final_query_vec_data.insert(final_query_vec_data.end(), subsequent_vec_data.begin(), subsequent_vec_data.end());
            final_query_vec_labels = std::move(topmost_vec_labels);
            final_query_vec_labels.insert(final_query_vec_labels.end(), subsequent_vec_labels.begin(), subsequent_vec_labels.end());
         }
         else
         {
            std::cout << "  - Decision: Performing stratified sampling." << std::endl;
            size_t target_total_count = static_cast<size_t>(total_candidates_vecs * new_cross_edge.query_vector_ratio);
            target_total_count = std::max(target_total_count, topmost_vec_ids.size());

            final_query_vec_ids = std::move(topmost_vec_ids);
            final_query_vec_data = std::move(topmost_vec_data);
            final_query_vec_labels = std::move(topmost_vec_labels);

            size_t num_to_sample = target_total_count > final_query_vec_ids.size() ? target_total_count - final_query_vec_ids.size() : 0;
            if (num_to_sample > 0 && !subsequent_vec_ids.empty())
            {
               num_to_sample = std::min(num_to_sample, subsequent_vec_ids.size());
               std::vector<size_t> indices(subsequent_vec_ids.size());
               std::iota(indices.begin(), indices.end(), 0);
               std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

               for (size_t i = 0; i < num_to_sample; ++i)
               {
                  size_t original_idx = indices[i];
                  final_query_vec_ids.push_back(subsequent_vec_ids[original_idx]);
                  final_query_vec_data.push_back(subsequent_vec_data[original_idx]);
                  final_query_vec_labels.push_back(subsequent_vec_labels[original_idx]);
               }
            }
         }
         std::cout << "  - Final number of query vectors for edge augmentation: " << final_query_vec_ids.size() << std::endl;

         if (final_query_vec_ids.empty())
         {
            std::cout << "  - No vectors to process. Skipping edge addition." << std::endl;
            return;
         }

         _cross_edge_step1_time_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::high_resolution_clock::now() - step1_start_time)
                                         .count();
         std::cout << "--- [Finished Sub-step 1 in " << _cross_edge_step1_time_ms << " ms] ---" << std::endl;

         std::cout << "\n--- [Cross-Edge Sub-step 2: Executing ACORN Script] ---" << std::endl;
         auto step2_start_time = std::chrono::high_resolution_clock::now();
         std::cout << "[Step 1.5] Preparing input files and executing ACORN script..." << std::endl;
         const std::string acorn_script_path = get_required_acorn_script_path();
         if (acorn_script_path.empty())
         {
            std::cerr << "  - FATAL ERROR: legacy ACORN augmentation requires UNG_ACORN_SCRIPT_PATH to point to run_acorn_for_ung.sh." << std::endl;
            return;
         }
         if (new_cross_edge.acorn_in_ung_output_path.empty() || new_cross_edge.acorn_in_ung_output_path == "false")
         {
            std::cerr << "  - FATAL ERROR: legacy ACORN augmentation requires a valid acorn_in_ung_output_path." << std::endl;
            return;
         }
         std::string query_vec_path = new_cross_edge.acorn_in_ung_output_path + "/acorn_query_vectors.fvecs";
         std::string query_attr_path = new_cross_edge.acorn_in_ung_output_path + "/acorn_query_attrs.txt";
         write_fvecs(query_vec_path, final_query_vec_data, _base_storage->get_dim());
         write_labels_txt(query_attr_path, final_query_vec_labels);
         std::string output_neighbors_file = new_cross_edge.acorn_in_ung_output_path + "/R_neighbors_with_vectors.txt";
         std::stringstream cmd_ss;
         cmd_ss << shell_quote(acorn_script_path) << " "
                << shell_quote(dataset) << " " << new_cross_edge.R_in_add_new_edge << " " << _base_storage->get_num_points() << " "
                << new_cross_edge.M << " " << new_cross_edge.M_beta << " " << new_cross_edge.gamma << " " << new_cross_edge.efs << " "
                << num_threads << " "
                << shell_quote(query_vec_path) << " " << shell_quote(query_attr_path) << " " << shell_quote(new_cross_edge.acorn_in_ung_output_path) << " 1";
         std::string cmd = cmd_ss.str();
         std::cout << "  - Executing command: " << cmd << std::endl;
         int ret = system(cmd.c_str());
         if (ret != 0)
         {
            std::cerr << "  - FATAL ERROR: ACORN script execution failed! Check logs in " << new_cross_edge.acorn_in_ung_output_path << std::endl;
            return;
         }
         std::cout << "  - ACORN script finished successfully." << std::endl;

         _cross_edge_step2_acorn_time_ms = std::chrono::duration<double, std::milli>(
                                               std::chrono::high_resolution_clock::now() - step2_start_time)
                                               .count();
         std::cout << "--- [Finished Sub-step 2 in " << _cross_edge_step2_acorn_time_ms << " ms] ---" << std::endl;

         std::cout << "\n--- [Cross-Edge Sub-step 3: Adding Distance-Oriented Edges] ---" << std::endl;
         auto step3_start_time = std::chrono::high_resolution_clock::now();

         std::cout << "[Step 1.6] Parsing ACORN output and generating candidates..." << std::endl;
         std::vector<NewEdgeCandidate> candidates;
         std::ifstream neighbor_file(output_neighbors_file);
         if (!neighbor_file.is_open())
         {
            std::cerr << "  - FATAL ERROR: Cannot open ACORN output file: " << output_neighbors_file << std::endl;
            return;
         }
         std::string line;
         long long current_query_local_idx = -1;
         IdxType u_id_new = 0;
         size_t intra_group_edges_skipped = 0;
         while (std::getline(neighbor_file, line))
         {
            if (line.rfind("Query: ", 0) == 0)
            {
               current_query_local_idx = std::stoll(line.substr(7));
               if (current_query_local_idx >= 0 && current_query_local_idx < final_query_vec_ids.size())
               {
                  u_id_new = final_query_vec_ids[current_query_local_idx];
               }
            }
            else if (line.rfind("NeighborID: ", 0) == 0)
            {
               std::stringstream line_ss(line);
               std::string token;
               IdxType v_id_old;
               line_ss >> token >> v_id_old;
               if (v_id_old < 0)
                  continue;
               auto it = old_to_new_map.find(v_id_old);
               if (it == old_to_new_map.end())
                  continue;
               IdxType v_id_new = it->second;
               IdxType group_u = _new_vec_id_to_group_id[u_id_new];
               IdxType group_v = _new_vec_id_to_group_id[v_id_new];
               if (group_u != group_v)
               {
                  float dist = _distance_handler->compute(
                      _base_storage->get_vector(u_id_new),
                      _base_storage->get_vector(v_id_new),
                      _base_storage->get_dim());
                  candidates.push_back({u_id_new, v_id_new, dist});
               }
               else
               {
                  intra_group_edges_skipped++;
               }
            }
         }
         neighbor_file.close();
         std::cout << "  - Generated " << candidates.size() << " initial cross-group candidates from ACORN results." << std::endl;
         std::cout << "  - Skipped " << intra_group_edges_skipped << " potential edges because they were intra-group." << std::endl;

         std::cout << "[Step 2] Adding new edges with smart selection logic and in-degree control (M=" << new_cross_edge.M_in_add_new_edge << ")..." << std::endl;

         // Group candidates by source vector so each query vector gets its own
         // bounded set of distance-oriented outgoing edges.
         std::unordered_map<IdxType, std::vector<NewEdgeCandidate>> query_to_candidates;
         for (const auto &cand : candidates)
         {
            query_to_candidates[cand.from].push_back(cand);
         }

         _num_distance_oriented_edges = 0;

         // Bound how many newly added edges can point to the same vector.
         std::vector<std::atomic<IdxType>> new_edge_in_degree(_num_points);
         for (size_t i = 0; i < _num_points; ++i)
         {
            new_edge_in_degree[i] = 0;
         }

         for (auto const &[query_id, cand_list] : query_to_candidates)
         {
            std::vector<NewEdgeCandidate> hierarchical_candidates;
            std::vector<NewEdgeCandidate> non_hierarchical_candidates;

            IdxType from_group = _new_vec_id_to_group_id[query_id];

            // Prefer parent->child label-hierarchy edges, then use ordinary
            // cross-group neighbors to fill the remaining degree budget.
            for (const auto &cand : cand_list)
            {
               IdxType to_group = _new_vec_id_to_group_id[cand.to];
               const auto &children = _label_nav_graph->out_neighbors[from_group];
               bool is_hierarchical = false;
               for (const auto &true_child : children)
               {
                  if (to_group == true_child)
                  {
                     is_hierarchical = true;
                     break;
                  }
               }
               if (is_hierarchical)
               {
                  hierarchical_candidates.push_back(cand);
               }
               else
               {
                  non_hierarchical_candidates.push_back(cand);
               }
            }

            std::sort(hierarchical_candidates.begin(), hierarchical_candidates.end(),
                      [](const auto &a, const auto &b)
                      { return a.distance < b.distance; });
            std::sort(non_hierarchical_candidates.begin(), non_hierarchical_candidates.end(),
                      [](const auto &a, const auto &b)
                      { return a.distance < b.distance; });

            size_t edges_added_for_this_query = 0;
            const size_t max_out_degree = new_cross_edge.M_in_add_new_edge;
            const size_t max_in_degree = new_cross_edge.M_in_add_new_edge;

            for (const auto &cand : hierarchical_candidates)
            {
               if (edges_added_for_this_query >= max_out_degree)
                  break;

               if (new_edge_in_degree[cand.to].load() < max_in_degree)
               {
                  uint64_t edge_key = (uint64_t)cand.from << 32 | cand.to;
                  if (_my_new_edges_set.find(edge_key) == _my_new_edges_set.end())
                  {
                     _graph->neighbors[cand.from].push_back(cand.to);
                     _num_distance_oriented_edges++;
                     edges_added_for_this_query++;
                     new_edge_in_degree[cand.to]++;
                     _my_new_edges_set.insert(edge_key);
                  }
               }
            }

            for (const auto &cand : non_hierarchical_candidates)
            {
               if (edges_added_for_this_query >= max_out_degree)
                  break;

               if (new_edge_in_degree[cand.to].load() < max_in_degree)
               {
                  uint64_t edge_key = (uint64_t)cand.from << 32 | cand.to;
                  if (_my_new_edges_set.find(edge_key) == _my_new_edges_set.end())
                  {
                     _graph->neighbors[cand.from].push_back(cand.to);
                     _num_distance_oriented_edges++;
                     edges_added_for_this_query++;
                     new_edge_in_degree[cand.to]++;
                     _my_new_edges_set.insert(edge_key);
                  }
               }
            }
         }

         std::cout << "  - Added a total of " << _num_distance_oriented_edges << " new distance-oriented edges to the graph with smart selection and in-degree control." << std::endl;

         // Diagnostic summary: compare ACORN-added group pairs against the
         // required parent->child hierarchy links.
         std::set<std::pair<IdxType, IdxType>> acorn_connected_group_pairs;
         for (const auto &edge_key : _my_new_edges_set)
         {
            IdxType from_id = edge_key >> 32;
            IdxType to_id = edge_key & 0xFFFFFFFF;
            acorn_connected_group_pairs.insert({_new_vec_id_to_group_id[from_id], _new_vec_id_to_group_id[to_id]});
         }

         std::cout << "\n--- [Cross-Edge Sub-step 3.5: ACORN Edge Connectivity Analysis] ---" << std::endl;
         // ... (诊断分析的 cout 代码完全不变) ...
         size_t total_required_hierarchical_links = 0;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            total_required_hierarchical_links += _label_nav_graph->out_neighbors[group_id].size();
         }
         std::cout << "  - Total parent->child links in hierarchy: " << total_required_hierarchical_links << std::endl;
         std::cout << "  - ACORN edges created " << acorn_connected_group_pairs.size() << " unique group-to-group connections." << std::endl;

         size_t satisfied_hierarchical_links = 0;
         for (const auto &group_pair : acorn_connected_group_pairs)
         {
            IdxType parent_group = group_pair.first;
            IdxType child_group = group_pair.second;
            const auto &children = _label_nav_graph->out_neighbors[parent_group];
            bool is_hierarchical = false;
            for (const auto &true_child : children)
            {
               if (child_group == true_child)
               {
                  is_hierarchical = true;
                  break;
               }
            }
            if (is_hierarchical)
            {
               satisfied_hierarchical_links++;
            }
         }
         std::cout << "  - Among them, " << satisfied_hierarchical_links << " connections are valid parent->child links." << std::endl;
         double satisfaction_rate = total_required_hierarchical_links > 0 ? (double)satisfied_hierarchical_links / total_required_hierarchical_links * 100.0 : 0.0;
         std::cout << "  - Satisfaction rate of hierarchical links by ACORN: " << satisfaction_rate << "%" << std::endl;
         std::cout << "--- [Finished ACORN Edge Analysis] ---\n"
                   << std::endl;

         _cross_edge_step3_add_dist_edges_time_ms = std::chrono::duration<double, std::milli>(
                                                        std::chrono::high_resolution_clock::now() - step3_start_time)
                                                        .count();
         std::cout << "--- [Finished Sub-step 3 in " << _cross_edge_step3_add_dist_edges_time_ms << " ms] ---" << std::endl;

         _build_cross_edges_time = std::chrono::duration<double, std::milli>(
                                       std::chrono::high_resolution_clock::now() - acorn_part_start_time)
                                       .count();
      }

      // Guarantee repair path: ensure parent groups have at least one edge into
      // each missing child group required by the label hierarchy.
      if (add_guarantee_edges)
      {
         std::cout << "\n--- [Cross-Edge Sub-step 4: Adding Targeted Edges for Unconnected Children] ---" << std::endl;
         auto step4_start_time = std::chrono::high_resolution_clock::now();

         size_t max_group_size = 0;
         if (_num_groups > 0)
         {
            for (auto group_id = 1; group_id <= _num_groups; ++group_id)
               max_group_size = std::max(max_group_size, static_cast<size_t>(_group_id_to_range[group_id].second - _group_id_to_range[group_id].first));
         }
         SearchCacheList search_cache_list(1, max_group_size, _Lbuild);

         std::vector<std::vector<std::pair<IdxType, IdxType>>> additional_edges_hierarchical(_num_groups + 1);

         std::cout << "  - Checking for and repairing broken parent->child links (in single-thread mode)..." << std::endl;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            const auto &cur_range = _group_id_to_range[group_id];
            if (cur_range.first >= cur_range.second || _label_nav_graph->out_neighbors[group_id].empty())
            {
               continue;
            }

            std::unordered_set<IdxType> connected_groups;
            for (IdxType i = cur_range.first; i < cur_range.second; ++i)
            {
               for (const auto &neighbor : _graph->neighbors[i])
                  connected_groups.insert(_new_vec_id_to_group_id[neighbor]);
            }

            for (IdxType out_group_id : _label_nav_graph->out_neighbors[group_id])
            {
               if (connected_groups.find(out_group_id) == connected_groups.end())
               {
                  const auto &out_range = _group_id_to_range[out_group_id];
                  size_t child_group_size = out_range.second - out_range.first;
                  if (child_group_size == 0)
                     continue;
                  if (child_group_size <= _max_degree)
                  { // 小规模组逻辑（蛮力）
                     IdxType query_vec_id = _group_entry_points[group_id];
                     const char *query = _base_storage->get_vector(query_vec_id);
                     float min_dist = std::numeric_limits<float>::max();
                     IdxType best_neighbor_id = -1;
                     for (IdxType i = out_range.first; i < out_range.second; ++i)
                     {
                        float dist = _distance_handler->compute(query, _base_storage->get_vector(i), _base_storage->get_dim());
                        if (dist < min_dist)
                        {
                           min_dist = dist;
                           best_neighbor_id = i;
                        }
                     }
                     if (best_neighbor_id != -1)
                     {
                        additional_edges_hierarchical[group_id].emplace_back(query_vec_id, best_neighbor_id);
                     }
                  }
                  else
                  { // 大规模组逻辑：使用手动图搜索，并配合正确的“已访问”集合
                     IdxType cnt = 0;
                     for (auto vec_id = cur_range.first; vec_id < cur_range.second && cnt < _num_cross_edges; ++vec_id)
                     {
                        const char *query_vec = _base_storage->get_vector(vec_id);
                        auto search_cache = search_cache_list.get_free_cache();
                        search_cache->search_queue.clear();
                        std::unordered_set<IdxType> visited_nodes_global;
                        IdxType child_group_entry_point = _group_entry_points[out_group_id];
                        float dist = _distance_handler->compute(query_vec, _base_storage->get_vector(child_group_entry_point), _base_storage->get_dim());

                        visited_nodes_global.insert(child_group_entry_point);
                        search_cache->search_queue.insert(child_group_entry_point, dist);
                        while (search_cache->search_queue.has_unexpanded_node())
                        {
                           const Candidate &cur = search_cache->search_queue.get_closest_unexpanded();
                           for (auto neighbor_id : _graph->neighbors[cur.id])
                           {
                              if (neighbor_id >= out_range.first && neighbor_id < out_range.second)
                              {
                                 if (visited_nodes_global.find(neighbor_id) == visited_nodes_global.end())
                                 {
                                    visited_nodes_global.insert(neighbor_id);
                                    float d = _distance_handler->compute(query_vec, _base_storage->get_vector(neighbor_id), _base_storage->get_dim());
                                    search_cache->search_queue.insert(neighbor_id, d);
                                 }
                              }
                           }
                        }
                        for (size_t k = 0; k < search_cache->search_queue.size() && k < _num_cross_edges / 2; ++k)
                        {
                           additional_edges_hierarchical[group_id].emplace_back(vec_id, search_cache->search_queue[k].id);
                           cnt++;
                        }
                        search_cache_list.release_cache(search_cache);
                     }
                  }
               }
            }
         }
         std::cout << std::endl;

         // 4. 单线程合并所有补充边...
         std::cout << "  - Merging newly created fill-in edges..." << std::endl;
         size_t num_added_hierarchical_edges = 0;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            for (const auto &[from_id, to_id] : additional_edges_hierarchical[group_id])
            {
               uint64_t edge_key = (uint64_t)from_id << 32 | to_id;
               if (_my_new_edges_set.find(edge_key) == _my_new_edges_set.end())
               {
                  _graph->neighbors[from_id].push_back(to_id);
                  _my_new_edges_set.insert(edge_key);
                  num_added_hierarchical_edges++;
               }
            }
         }
         std::cout << "  - Merged " << num_added_hierarchical_edges << " new hierarchical edges to repair the graph." << std::endl;

         _cross_edge_step4_add_hierarchy_edges_time_ms = std::chrono::duration<double, std::milli>(
                                                             std::chrono::high_resolution_clock::now() - step4_start_time)
                                                             .count();
         std::cout << "--- [Finished Sub-step 4 in " << _cross_edge_step4_add_hierarchy_edges_time_ms << " ms] ---" << std::endl;
      }

      auto total_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - all_start_time).count();
      std::cout << "\n--- Finished augmenting graph in " << total_time_ms << " ms. ---" << std::endl;
      std::cout << "========================================================" << std::endl;
   }

}
