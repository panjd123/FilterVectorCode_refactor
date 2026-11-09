#include <algorithm>
#include <atomic>
#include <bitset>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <omp.h>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <roaring/roaring.hh>

#include "include/uni_nav_graph.h"

namespace ANNS
{
   void UniNavGraph::build_trie_and_divide_groups()
   {
      // create groups for base label sets
      IdxType new_group_id = 1;
      for (auto vec_id = 0; vec_id < _num_points; ++vec_id)
      {
         const auto &label_set = _base_storage->get_label_set(vec_id);
         auto group_id = _trie_index.insert(label_set, new_group_id);

         // deal with new label setinver
         if (group_id + 1 > _group_id_to_vec_ids.size())
         {
            _group_id_to_vec_ids.resize(group_id + 1);
            _group_id_to_label_set.resize(group_id + 1);
            _group_id_to_label_set[group_id] = label_set;
         }
         _group_id_to_vec_ids[group_id].emplace_back(vec_id);
      }

      // logs
      _num_groups = new_group_id - 1;
      std::cout << "- Number of groups: " << _num_groups << std::endl;
   }

   void UniNavGraph::get_min_super_sets(const std::vector<LabelType> &query_label_set, std::vector<IdxType> &min_super_set_ids,
                                        bool avoid_self, bool need_containment)
   {
      if (_build_config.get_min_super_sets_impl == UngGetMinSuperSetsImpl::OriginalSort)
      {
         get_min_super_sets_original_sort(query_label_set, min_super_set_ids, avoid_self, need_containment);
         return;
      }
      get_min_super_sets_optimized_bucket(query_label_set, min_super_set_ids, avoid_self, need_containment);
   }

   void UniNavGraph::get_min_super_sets_original_sort(const std::vector<LabelType> &query_label_set, std::vector<IdxType> &min_super_set_ids,
                                                      bool avoid_self, bool need_containment)
   {
      min_super_set_ids.clear();

      std::vector<std::shared_ptr<TrieNode>> candidates;
      _trie_index.get_super_set_entrances(query_label_set, candidates, avoid_self, need_containment);

      if (candidates.empty())
         return;
      if (candidates.size() == 1)
      {
         min_super_set_ids.emplace_back(candidates[0]->group_id);
         return;
      }

      std::sort(candidates.begin(), candidates.end(),
                [](const std::shared_ptr<TrieNode> &a, const std::shared_ptr<TrieNode> &b)
                {
                   return a->label_set_size < b->label_set_size;
                });
      auto min_size = _group_id_to_label_set[candidates[0]->group_id].size();

      for (auto candidate : candidates)
      {
         const auto &cur_group_id = candidate->group_id;
         const auto &cur_label_set = _group_id_to_label_set[cur_group_id];
         bool is_min = true;

         if (cur_label_set.size() > min_size)
         {
            for (auto min_group_id : min_super_set_ids)
            {
               const auto &min_label_set = _group_id_to_label_set[min_group_id];
               if (std::includes(cur_label_set.begin(), cur_label_set.end(), min_label_set.begin(), min_label_set.end()))
               {
                  is_min = false;
                  break;
               }
            }
         }

         if (is_min)
            min_super_set_ids.emplace_back(cur_group_id);
      }
   }

   void UniNavGraph::get_min_super_sets_optimized_bucket(const std::vector<LabelType> &query_label_set, std::vector<IdxType> &min_super_set_ids,
                                                        bool avoid_self, bool need_containment)
   {
      min_super_set_ids.clear();

      // Reuse thread-local buffers to avoid repeated candidate allocations.
      thread_local std::vector<std::shared_ptr<TrieNode>> candidates;
      candidates.clear();
      _trie_index.get_super_set_entrances(query_label_set, candidates, avoid_self, need_containment);

      if (candidates.size() > 1000000)
      {
         #pragma omp critical
         {
            std::cout << "[WARNING] Performance Hazard! Group Label Size: " << query_label_set.size()
                        << " | Candidates Found: " << candidates.size()
                        << " (This will be slow!)" << std::endl;
            if (!query_label_set.empty()) {
                  std::cout << "   -> Labels: " << query_label_set[0] << " ... " << std::endl;
            }
         }
      }
      // special cases
      if (candidates.empty())
         return;
      if (candidates.size() == 1)
      {
         min_super_set_ids.emplace_back(candidates[0]->group_id);
         return;
      }

      thread_local std::vector<TrieNode*> candidate_nodes;
      candidate_nodes.clear();
      candidate_nodes.reserve(candidates.size());
      LabelType min_size = candidates[0]->label_set_size;
      LabelType max_size = min_size;
      for (const auto &sp : candidates)
      {
         TrieNode *node = sp.get();
         candidate_nodes.push_back(node);
         min_size = std::min(min_size, node->label_set_size);
         max_size = std::max(max_size, node->label_set_size);
      }

      thread_local std::vector<const std::vector<LabelType>*> min_label_sets;
      min_label_sets.clear();
      min_super_set_ids.reserve(candidate_nodes.size());

      auto try_add_candidate = [&](const TrieNode *candidate) {
         const auto cur_group_id = candidate->group_id;
         const auto &cur_label_set = _group_id_to_label_set[cur_group_id];
         bool is_min = true;

         // check whether contains existing minimum super sets (label ids are in ascending order)
         if (candidate->label_set_size > min_size)
         {
            for (const auto *min_label_set : min_label_sets)
            {
               if (std::includes(cur_label_set.begin(), cur_label_set.end(), min_label_set->begin(), min_label_set->end()))
               {
                  is_min = false;
                  break;
               }
            }
         }

         if (is_min)
         {
            min_super_set_ids.emplace_back(cur_group_id);
            min_label_sets.emplace_back(&cur_label_set);
         }
      };

      // 当 label_set_size 跨度较小时，使用桶化遍历替代 O(n log n) 排序。
      static constexpr LabelType kBucketSpanThreshold = 256;
      if (max_size >= min_size && (max_size - min_size) <= kBucketSpanThreshold)
      {
         const size_t bucket_span = static_cast<size_t>(max_size - min_size + 1);
         thread_local std::vector<std::vector<TrieNode*>> size_buckets;
         if (size_buckets.size() < bucket_span)
            size_buckets.resize(bucket_span);
         for (size_t i = 0; i < bucket_span; ++i)
            size_buckets[i].clear();

         for (TrieNode *node : candidate_nodes)
            size_buckets[static_cast<size_t>(node->label_set_size - min_size)].push_back(node);

         for (size_t i = 0; i < bucket_span; ++i)
            for (const TrieNode *candidate : size_buckets[i])
               try_add_candidate(candidate);
      }
      else
      {
         // 跨度较大时回退原排序路径，保证稳定性。
         std::sort(candidate_nodes.begin(), candidate_nodes.end(),
                   [](const TrieNode *a, const TrieNode *b)
                   {
                      return a->label_set_size < b->label_set_size;
                   });

         for (const TrieNode *candidate : candidate_nodes)
            try_add_candidate(candidate);
      }
   }

   void UniNavGraph::get_min_super_sets_debug(const std::vector<LabelType> &query_label_set,
                                              std::vector<IdxType> &min_super_set_ids,
                                              bool avoid_self, bool need_containment,
                                              std::atomic<int> &print_counter, bool is_new_trie_method, bool is_rec_more_start, QueryStats &stats,
                                              bool skip_group_id_check) const
   {
#if ENABLE_ENTRY_DEBUG_OUTPUT
      // --- 计时器变量定义 ---
      double time_trie_lookup = 0.0;
      double time_sorting = 0.0;
      double time_filtering_loop = 0.0;
      auto function_start_time = std::chrono::high_resolution_clock::now();
#endif

      min_super_set_ids.clear();

      // --- 1. 测量Trie查找候选者的时间 ---
#if ENABLE_ENTRY_DEBUG_OUTPUT
      auto start_trie = std::chrono::high_resolution_clock::now();
#endif

      std::vector<std::shared_ptr<TrieNode>> candidates;

      if (is_new_trie_method)
      {
         // --- 调用新方法 (Recursive) ---
         TrieSearchMetricsRecursive trie_metrics_m2 = {};
         if (is_rec_more_start)
            _trie_index.get_super_set_entrances_new_more_sp_debug(query_label_set, candidates, avoid_self, need_containment, print_counter, trie_metrics_m2);
         else
            _trie_index.get_super_set_entrances_new_debug(query_label_set, candidates, avoid_self, need_containment, print_counter, trie_metrics_m2);
         stats.recursive_calls = trie_metrics_m2.recursive_calls;
         stats.pruning_events = trie_metrics_m2.pruning_events;
         stats.trie_nodes_traversed = trie_metrics_m2.recursive_calls + trie_metrics_m2.nodes_processed_in_bfs;
         if (stats.recursive_calls > 0)
            stats.pruning_efficiency = static_cast<float>(stats.pruning_events) / stats.recursive_calls;
         else
            stats.pruning_efficiency = 0.0f;
      }
      else
      {
         // --- 调用旧方法 (Shortcut) ---
         TrieMethod1Metrics trie_metrics_m1 = {};
         _trie_index.get_super_set_entrances_debug(query_label_set, candidates, avoid_self, need_containment, print_counter, trie_metrics_m1, skip_group_id_check);
         stats.successful_checks = trie_metrics_m1.successful_checks;
         stats.trie_nodes_traversed = trie_metrics_m1.upward_traversals + trie_metrics_m1.bfs_nodes_processed;
         stats.redundant_upward_steps = trie_metrics_m1.redundant_upward_steps;
         if (stats.candidate_set_size > 0)
            stats.shortcut_hit_ratio = static_cast<float>(stats.successful_checks) / stats.candidate_set_size;
         else
            stats.shortcut_hit_ratio = 0.0f;
      }

#if ENABLE_ENTRY_DEBUG_OUTPUT
      auto end_trie = std::chrono::high_resolution_clock::now();
      time_trie_lookup = std::chrono::duration<double, std::milli>(end_trie - start_trie).count();
#endif

      // 特殊情况处理
      if (candidates.empty())
         return;
      if (candidates.size() == 1)
      {
         min_super_set_ids.emplace_back(candidates[0]->group_id);
         return;
      }

      // --- 2. 测量排序时间 ---
#if ENABLE_ENTRY_DEBUG_OUTPUT
      auto start_sort = std::chrono::high_resolution_clock::now();
#endif
      std::sort(candidates.begin(), candidates.end(),
                [](const std::shared_ptr<TrieNode> &a, const std::shared_ptr<TrieNode> &b)
                {
                   return a->label_set_size < b->label_set_size;
                });
#if ENABLE_ENTRY_DEBUG_OUTPUT
      auto end_sort = std::chrono::high_resolution_clock::now();
      time_sorting = std::chrono::duration<double, std::milli>(end_sort - start_sort).count();
#endif

      auto min_size = _group_id_to_label_set[candidates[0]->group_id].size();

      // --- 3. 测量过滤循环的时间 ---
#if ENABLE_ENTRY_DEBUG_OUTPUT
      auto start_filter = std::chrono::high_resolution_clock::now();
#endif
      for (auto candidate : candidates)
      {
         const auto &cur_group_id = candidate->group_id;
         const auto &cur_label_set = _group_id_to_label_set[cur_group_id];
         bool is_min = true;

         if (cur_label_set.size() > min_size)
         {
            for (auto min_group_id : min_super_set_ids)
            {
               const auto &min_label_set = _group_id_to_label_set[min_group_id];
               if (std::includes(cur_label_set.begin(), cur_label_set.end(), min_label_set.begin(), min_label_set.end()))
               {
                  is_min = false;
                  break;
               }
            }
         }

         if (is_min)
         {
            min_super_set_ids.emplace_back(cur_group_id);
         }
      }
#if ENABLE_ENTRY_DEBUG_OUTPUT
      auto end_filter = std::chrono::high_resolution_clock::now();
      time_filtering_loop = std::chrono::duration<double, std::milli>(end_filter - start_filter).count();

      auto function_total_time = std::chrono::duration<double, std::milli>(end_filter - function_start_time).count();
#endif
   }

   // 为了方便排序，定义一个结构体来存储候选组及其评分
   struct ScoredCandidate
   {
      double score;
      ANNS::IdxType group_id;

      // 重载小于运算符，方便使用 std::sort 进行降序排序
      bool operator<(const ScoredCandidate &other) const
      {
         return score > other.score; // score 越大越靠前
      }
   };

   std::vector<ANNS::IdxType> UniNavGraph::select_entry_groups(
       const std::vector<ANNS::IdxType> &minimum_entry_sets,
       ANNS::SelectionMode mode,
       size_t top_k,
       double beta,
       ANNS::IdxType true_query_group_id) const
   {
      bool verbose = false;
      assert(_label_nav_graph != nullptr && "Error: _label_nav_graph should not be null.");

      if (top_k == 0 && true_query_group_id == 0)
      {
         return minimum_entry_sets;
      }
      if (minimum_entry_sets.empty() && true_query_group_id == 0)
      {
         return {};
      }

      std::queue<ANNS::IdxType> q;
      std::vector<int> lng_distance(_num_groups + 1, -1);

      // Score only nodes reached by the BFS frontier instead of rescanning every group.
      std::vector<ANNS::IdxType> reachable_nodes;
      reachable_nodes.reserve(_num_groups / 10);

      for (const auto &group_id : minimum_entry_sets)
      {
         if (group_id > 0 && group_id <= _num_groups && lng_distance[group_id] == -1)
         {
            q.push(group_id);
            lng_distance[group_id] = 0;
         }
      }

      while (!q.empty())
      {
         ANNS::IdxType current_group = q.front();
         q.pop();

         if (current_group >= _label_nav_graph->out_neighbors.size())
            continue;

         for (const auto &child_group : _label_nav_graph->out_neighbors[current_group])
         {
            if (child_group > 0 && child_group <= _num_groups && lng_distance[child_group] == -1)
            {
               lng_distance[child_group] = lng_distance[current_group] + 1;
               q.push(child_group);
               reachable_nodes.push_back(child_group);
            }
         }
      }

      std::vector<ANNS::ScoredCandidate> candidates;
      candidates.reserve(reachable_nodes.size());

      for (const auto group_id : reachable_nodes)
      {
         const double group_size = static_cast<double>(_group_id_to_vec_ids[group_id].size());
         double score = 0.0;
         if (mode == ANNS::SelectionMode::SizeOnly)
         {
            score = group_size;
         }
         else
         { // SizeAndDistance with new formula
            const int dist = lng_distance[group_id];
            score = static_cast<double>(dist) * static_cast<double>(_num_points) + group_size;
         }
         candidates.push_back({score, group_id});
      }

      size_t num_to_sort = std::min(candidates.size(), top_k + 5);
      if (num_to_sort > 0)
      {
         std::partial_sort(candidates.begin(),
                           candidates.begin() + num_to_sort,
                           candidates.end());
      }

      std::vector<ANNS::IdxType> final_entry_groups = minimum_entry_sets;
      std::unordered_set<ANNS::IdxType> existing_groups(minimum_entry_sets.begin(), minimum_entry_sets.end());

      bool oracle_injected = false;
      bool oracle_existed = false;

      if (true_query_group_id > 0 && true_query_group_id <= _num_groups)
      {
         if (existing_groups.find(true_query_group_id) == existing_groups.end())
         {
            final_entry_groups.push_back(true_query_group_id);
            existing_groups.insert(true_query_group_id);
            oracle_injected = true;
         }
         else
         {
            oracle_existed = true;
         }
      }

      for (size_t i = 0; i < num_to_sort; ++i)
      {
         const auto &candidate = candidates[i];
         if (final_entry_groups.size() >= minimum_entry_sets.size() + top_k)
         {
            break;
         }
         if (existing_groups.find(candidate.group_id) == existing_groups.end())
         {
            final_entry_groups.push_back(candidate.group_id);
            existing_groups.insert(candidate.group_id);
         }
      }

      // --- 精简后的日志输出 ---
      if (verbose)
      {
         if (oracle_injected)
         {
            std::cout << "[Info] Oracle group " << true_query_group_id << " was successfully injected." << std::endl;
         }
         else if (oracle_existed && true_query_group_id > 0)
         {
            std::cout << "[Info] Oracle group " << true_query_group_id << " was already in the initial set." << std::endl;
         }

         size_t added_count = final_entry_groups.size() > minimum_entry_sets.size() ? final_entry_groups.size() - minimum_entry_sets.size() : 0;
         if (oracle_injected)
            added_count--;

         std::cout << "[Info] Final selected entry groups count: " << final_entry_groups.size()
                   << " (Initial: " << minimum_entry_sets.size()
                   << ", Heuristically Added: " << added_count << ")" << std::endl;
      }

      return final_entry_groups;
   }

   void UniNavGraph::build_label_nav_graph()
   {
      if (_build_config.lng_impl == UngLngImpl::OriginalCpu)
      {
         build_label_nav_graph_original_cpu();
         return;
      }
      if (_build_config.lng_impl == UngLngImpl::LegacyAllocating)
      {
         build_label_nav_graph_legacy_allocating();
         return;
      }
      build_label_nav_graph_optimized_phase1();
   }

   void UniNavGraph::build_label_nav_graph_optimized_phase1()
   {
      std::cout << "Building label navigation graph... " << std::endl;
      std::cout << "- LNG impl: " << to_string(_build_config.lng_impl) << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups + 1);
      omp_set_num_threads(_num_threads);

      std::atomic<size_t> processed_count{0};
      size_t total_groups = _num_groups;
      size_t log_interval = 100000;

      std::cout << "--- Phase 1: Calculating Out-Neighbors (Parallel) ---" << std::endl;
#pragma omp parallel for schedule(dynamic, 256)
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         thread_local std::vector<IdxType> min_super_set_ids;
         get_min_super_sets(_group_id_to_label_set[group_id], min_super_set_ids, true);
         _label_nav_graph->out_neighbors[group_id].swap(min_super_set_ids);
         min_super_set_ids.clear();

         size_t current = ++processed_count;

         bool normal_log = (current % log_interval == 0);
         bool final_sprint = (current > total_groups - 100000) && (current % 1000 == 0);

         if (normal_log || final_sprint || current == total_groups)
         {
            #pragma omp critical
            {
               double percentage = (double)current / total_groups * 100.0;
               std::cout << "[Phase 1] Processed " << current << " / " << total_groups
                         << " (" << std::fixed << std::setprecision(4) << percentage << "%)"
                         << std::endl;
            }
         }
      }

      std::cout << "--- Phase 2: Calculating In-Neighbors (Serial) ---" << std::endl;

      processed_count = 0;
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         for (auto each : _label_nav_graph->out_neighbors[group_id])
            _label_nav_graph->in_neighbors[each].emplace_back(group_id);

         size_t current = ++processed_count;
         if (current % log_interval == 0 || current == total_groups)
         {
             double percentage = (double)current / total_groups * 100.0;
             std::cout << "[Phase 2] Processed " << current << " / " << total_groups
                       << " (" << std::fixed << std::setprecision(1) << percentage << "%)"
                       << std::endl;
         }
      }

      _build_LNG_time = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - start_time)
                            .count();
      std::cout << "- Finished building LNG in " << _build_LNG_time << " ms" << std::endl;
   }

   void UniNavGraph::build_label_nav_graph_original_cpu()
   {
      std::cout << "Building label navigation graph... " << std::endl;
      std::cout << "- LNG impl: " << to_string(_build_config.lng_impl) << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups + 1);
      omp_set_num_threads(_num_threads);

#pragma omp parallel for schedule(dynamic, 256)
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         if (group_id % 100 == 0)
            std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
         std::vector<IdxType> min_super_set_ids;
         get_min_super_sets_original_sort(_group_id_to_label_set[group_id], min_super_set_ids, true);
         _label_nav_graph->out_neighbors[group_id] = min_super_set_ids;
      }

      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
         for (auto each : _label_nav_graph->out_neighbors[group_id])
            _label_nav_graph->in_neighbors[each].emplace_back(group_id);

      _build_LNG_time = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - start_time)
                            .count();
      std::cout << "\r- Finished in " << _build_LNG_time << " ms" << std::endl;
   }

   void UniNavGraph::build_label_nav_graph_legacy_allocating()
   {
      std::cout << "Building label navigation graph... " << std::endl;
      std::cout << "- LNG impl: " << to_string(_build_config.lng_impl)
                << " (compatibility slow path)" << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups + 1);
      omp_set_num_threads(_num_threads);

      std::atomic<size_t> processed_count{0};
      size_t total_groups = _num_groups;
      size_t log_interval = 100000;

      std::cout << "--- Phase 1: Calculating Out-Neighbors (Parallel, legacy allocating) ---" << std::endl;
#pragma omp parallel for schedule(dynamic, 1)
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         std::vector<IdxType> min_super_set_ids;
         get_min_super_sets(_group_id_to_label_set[group_id], min_super_set_ids, true);
         _label_nav_graph->out_neighbors[group_id] = min_super_set_ids;

         size_t current = ++processed_count;
         if (current % log_interval == 0 || current == total_groups)
         {
#pragma omp critical
            {
               double percentage = (double)current / total_groups * 100.0;
               std::cout << "[Phase 1] Processed " << current << " / " << total_groups
                         << " (" << std::fixed << std::setprecision(4) << percentage << "%)" << std::endl;
            }
         }
      }

      std::cout << "--- Phase 2: Calculating In-Neighbors (Serial) ---" << std::endl;
      processed_count = 0;
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         for (auto each : _label_nav_graph->out_neighbors[group_id])
            _label_nav_graph->in_neighbors[each].emplace_back(group_id);

         size_t current = ++processed_count;
         if (current % log_interval == 0 || current == total_groups)
         {
            double percentage = (double)current / total_groups * 100.0;
            std::cout << "[Phase 2] Processed " << current << " / " << total_groups
                      << " (" << std::fixed << std::setprecision(1) << percentage << "%)" << std::endl;
         }
      }

      _build_LNG_time = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - start_time)
                            .count();
      std::cout << "- Finished building LNG in " << _build_LNG_time << " ms" << std::endl;
   }

   roaring::Roaring UniNavGraph::collect_target_points(const std::vector<IdxType> &group_ids) const
   {
      roaring::Roaring targets;
      roaring::Roaring visited_groups;
      std::vector<IdxType> pending(group_ids);
      while (!pending.empty())
      {
         const IdxType group_id = pending.back();
         pending.pop_back();
         if (group_id == 0 || group_id > _num_groups || !visited_groups.addChecked(group_id))
            continue;
         const auto &range = _group_id_to_range[group_id];
         // Block backend membership maps use original dataset ids.
         for (IdxType point_id = range.first; point_id < range.second; ++point_id)
            targets.add(_new_to_old_vec_ids[point_id]);
         for (IdxType child : _label_nav_graph->out_neighbors[group_id])
            pending.push_back(child);
      }
      return targets;
   }
}
