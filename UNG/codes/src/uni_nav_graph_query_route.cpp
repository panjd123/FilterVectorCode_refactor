#include "include/uni_nav_graph.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "include/MethodSelector.h"

namespace ANNS
{

   bool ung_env_flag_enabled(const char *name)
   {
      const char *value = std::getenv(name);
      if (value == nullptr)
         return false;
      return !(value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
               value[0] == 'n' || value[0] == 'N' || value[0] == 'o' || value[0] == 'O');
   }

   bool should_stop_special_block_search(bool enabled,
                                         size_t result_size,
                                         IdxType k,
                                         float best_unexpanded_distance,
                                         float second_best_unexpanded_distance,
                                         float worst_result_distance,
                                         size_t nodes_visited,
                                         size_t min_nodes_visited)
   {
      if (!enabled || k == 0 || result_size < static_cast<size_t>(k))
         return false;
      if (nodes_visited < min_nodes_visited)
         return false;
      return best_unexpanded_distance > worst_result_distance &&
             second_best_unexpanded_distance > worst_result_distance;
   }

   const char *search_graph_backend_impl_name(SearchGraphBackendImpl impl)
   {
      switch (impl)
      {
      case SearchGraphBackendImpl::NeighborList:
         return "neighbor_list";
      case SearchGraphBackendImpl::Csr:
         return "csr";
      }
      return "unknown";
   }

   SearchGraphBackendImpl parse_search_graph_backend_impl(const std::string &value)
   {
      if (value == "0" || value == "neighbor_list" || value == "graph" || value == "default")
         return SearchGraphBackendImpl::NeighborList;
      if (value == "1" || value == "csr")
         return SearchGraphBackendImpl::Csr;
      throw std::invalid_argument(
          "Invalid graph_search_backend: " + value +
          " (expected neighbor_list/graph/default/0 or csr/1)");
   }

   SearchRuntimeConfig make_search_runtime_config(uint32_t num_threads,
                                                   IdxType Lsearch,
                                                   IdxType num_entry_points,
                                                   std::string scenario,
                                                   IdxType K,
                                                   bool idea2_available,
                                                   bool use_new_trie_method,
                                                   bool recursive_more_start,
                                                   bool ung_more_entry,
                                                   bool bfs_filter,
                                                   int lsearch_start,
                                                   int lsearch_step,
                                                   int efs_start,
                                                   int efs_step_slow,
                                                   int efs_step_fast,
                                                   int lsearch_threshold,
                                                   int force_use_alg,
                                                   EntryGroupStrategy entry_group_strategy,
                                                   SearchGraphBackendImpl graph_backend)
   {
      SearchRuntimeConfig runtime;
      runtime.num_threads = num_threads;
      runtime.Lsearch = Lsearch;
      runtime.num_entry_points = num_entry_points;
      runtime.scenario = std::move(scenario);
      runtime.K = K;
      runtime.idea2_available = idea2_available;
      runtime.use_new_trie_method = use_new_trie_method;
      runtime.recursive_more_start = recursive_more_start;
      runtime.ung_more_entry = ung_more_entry;
      runtime.bfs_filter = bfs_filter;
      runtime.entry_group_strategy = entry_group_strategy;
      runtime.graph_backend = graph_backend;
      runtime.special_block_search = ung_env_flag_enabled("UNG_SPECIAL_BLOCK_SEARCH");
      runtime.special_block_early_stop = ung_env_flag_enabled("UNG_SPECIAL_EARLY_STOP");
      runtime.special_light_stats = true;
      if (const char *value = std::getenv("UNG_SPECIAL_LIGHT_STATS"))
      {
         runtime.special_light_stats = !(value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
                                         value[0] == 'n' || value[0] == 'N' || value[0] == 'o' || value[0] == 'O');
      }
      if (const char *value = std::getenv("UNG_SPECIAL_PREFETCH"))
      {
         runtime.special_block_prefetch = !(value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
                                            value[0] == 'n' || value[0] == 'N' || value[0] == 'o' || value[0] == 'O');
      }
      if (const char *value = std::getenv("UNG_SPECIAL_SEARCH_MODE"))
      {
         const std::string mode(value);
         if (mode == "favor_blocks")
            runtime.special_search_mode = SpecialSearchMode::FavorBlocks;
         else if (mode == "free_state" || mode == "free" || mode == "0")
            runtime.special_search_mode = SpecialSearchMode::FreeState;
         else
            throw std::invalid_argument("Invalid UNG_SPECIAL_SEARCH_MODE: " + mode +
                                        " (expected free_state or favor_blocks)");
      }
      if (const char *value = std::getenv("UNG_SPECIAL_MAX_ACTIVE_LEVEL"))
      {
         const unsigned long configured_level = std::strtoul(value, nullptr, 10);
         runtime.special_max_activation_level = static_cast<uint8_t>(
             std::min<unsigned long>(configured_level,
                                     std::numeric_limits<uint8_t>::max()));
      }
      if (const char *value = std::getenv("UNG_SPECIAL_UPPER_MIN_COVERED_POINTS"))
         runtime.special_upper_min_covered_points =
             static_cast<size_t>(std::strtoull(value, nullptr, 10));
      runtime.special_heavy_edge_search = ung_env_flag_enabled("UNG_SPECIAL_HEAVY_EDGE_SEARCH");
      if (const char *value = std::getenv("UNG_SPECIAL_HEAVY_EDGE_MIN_QUERY_SIZE"))
         runtime.special_heavy_edge_min_query_size = static_cast<size_t>(std::strtoull(value, nullptr, 10));
      if (const char *value = std::getenv("UNG_SPECIAL_HEAVY_EDGE_MIN_MATCHED_POINTS"))
         runtime.special_heavy_edge_min_matched_points = static_cast<size_t>(std::strtoull(value, nullptr, 10));
      if (const char *value = std::getenv("UNG_SPECIAL_HEAVY_EDGE_MIN_ENTRIES"))
         runtime.special_heavy_edge_min_entries = static_cast<size_t>(std::strtoull(value, nullptr, 10));
      runtime.lsearch_start = lsearch_start;
      runtime.lsearch_step = lsearch_step;
      runtime.efs_start = efs_start;
      runtime.efs_step_slow = efs_step_slow;
      runtime.efs_step_fast = efs_step_fast;
      runtime.lsearch_threshold = lsearch_threshold;
      runtime.force_use_alg = force_use_alg;
      return runtime;
   }

   QueryRouteDecision UniNavGraph::decide_query_route(const std::vector<LabelType> &query_labels,
                                                      bool is_idea2_available,
                                                      bool is_new_trie_method,
                                                      bool is_rec_more_start,
                                                      int force_use_alg,
                                                      bool is_bfs_filter,
                                                      std::vector<IdxType> &entry_group_ids,
                                                      QueryStats &stats)
   {
      QueryRouteDecision decision;

      stats.query_length = query_labels.size();
      stats.candidate_set_size = query_labels.empty() ? 0 : get_candidate_count_for_label(query_labels.back());

      const bool is_method3_mode = (force_use_alg == 0 && is_idea2_available && is_new_trie_method);
      if (is_method3_mode)
         decision.pre_trie_heuristic = check_pre_trie_heuristic(_dataset, stats.query_length, stats.candidate_set_size);

      if (decision.apply_pre_trie_heuristic())
      {
         stats.is_idea1_used = decision.use_new_trie;
         stats.is_idea2_used = static_cast<float>(decision.algorithm);
      }
      else if (force_use_alg == 0)
      {
         if (_trie_method_selector != nullptr && is_new_trie_method)
         {
            auto idea1_flag_start_time = std::chrono::high_resolution_clock::now();

            const auto &trie_metrics = _trie_static_metrics;
            stats.trie_total_nodes = trie_metrics.total_nodes;
            stats.trie_label_cardinality = trie_metrics.label_cardinality;
            stats.trie_avg_path_length = trie_metrics.avg_path_length;
            stats.trie_avg_branching_factor = trie_metrics.avg_branching_factor;

            std::vector<float> idea1_features = calculate_idea1_features(stats);
            if (!idea1_features.empty())
            {
               auto idea1_selector_start_time = std::chrono::high_resolution_clock::now();
               decision.use_new_trie = _trie_method_selector->predict(idea1_features);
               stats.idea1_selector_pred_time_ms =
                   std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - idea1_selector_start_time)
                       .count();
            }
            else
            {
               std::cout << "[Warning] No Idea1 features for dataset " << _dataset
                         << ". Reverting to default behavior." << std::endl;
               decision.use_new_trie = is_new_trie_method;
            }
            stats.idea1_flag_time_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - idea1_flag_start_time)
                    .count();
         }
         else
         {
            decision.use_new_trie = is_new_trie_method;
         }

         if (is_idea2_available)
         {
            static std::atomic<int> temp_counter{0};
            QueryStats temp_stats;
            auto get_entry_group_start_time = std::chrono::high_resolution_clock::now();
            const bool use_nT_for_group_get =
                (force_use_alg == 2) ? true : ((force_use_alg == 0) ? decision.use_new_trie : is_new_trie_method);
            get_min_super_sets_debug(query_labels, entry_group_ids, false, true, temp_counter,
                                     use_nT_for_group_get, is_rec_more_start, temp_stats, is_new_trie_method);
            decision.entry_groups_calculated = true;
            stats.get_min_super_sets_time_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - get_entry_group_start_time)
                    .count();

            auto idea2_flag_start_time = std::chrono::high_resolution_clock::now();
            // Route statistics are optional metadata.  They are not required
            // to execute the graph search after entry_group_ids are known.
            // Keep the selector path usable when the caller explicitly asks
            // to skip the expensive descendants/coverage bitmap unions.
            stats.num_entry_points = entry_group_ids.size();
            if (!ung_env_flag_enabled("UNG_DISABLE_ENTRY_ROUTE_STATS"))
               populate_entry_group_route_stats(entry_group_ids, stats);

            std::vector<float> idea2_features = calculate_idea2_features(stats);
            if (!idea2_features.empty() && _ung_acorn_selector != nullptr && force_use_alg == 0)
            {
               auto idea2_selector_start_time = std::chrono::high_resolution_clock::now();
               const float pred_val = _ung_acorn_selector->predict(idea2_features);
               decision.algorithm = static_cast<int>(std::round(pred_val));
               stats.idea2_selector_pred_time_ms =
                   std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - idea2_selector_start_time)
                       .count();
            }
            stats.idea2_flag_time_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - idea2_flag_start_time)
                    .count();
         }
      }

      if (force_use_alg > 0)
      {
         decision.apply_force_route(force_use_alg, is_bfs_filter);
      }

      stats.is_idea1_used = decision.use_new_trie;
      stats.is_idea2_used = static_cast<float>(decision.algorithm);
      return decision;
   }

} // namespace ANNS
