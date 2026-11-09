#include "include/uni_nav_graph.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

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
      if (idea2_available || (force_use_alg != 1 && force_use_alg != 2))
         throw std::invalid_argument("Adaptive and bitmap routes were removed; use force_use_alg=1 or 2 and is_idea2_available=false");
      SearchRuntimeConfig runtime;
      runtime.num_threads = num_threads;
      runtime.Lsearch = Lsearch;
      runtime.num_entry_points = num_entry_points;
      runtime.scenario = std::move(scenario);
      runtime.K = K;
      runtime.recursive_more_start = recursive_more_start;
      runtime.ung_more_entry = ung_more_entry;
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
         else if (mode == "hnsw_blocks" || mode == "upper_hnsw" || mode == "2")
            runtime.special_search_mode = SpecialSearchMode::HnswBlocks;
         else if (mode == "auto" || mode == "3")
            runtime.special_search_mode = SpecialSearchMode::Auto;
         else if (mode == "free_state" || mode == "free" || mode == "0")
            runtime.special_search_mode = SpecialSearchMode::FreeState;
         else
            throw std::invalid_argument("Invalid UNG_SPECIAL_SEARCH_MODE: " + mode +
                                        " (expected free_state, favor_blocks, hnsw_blocks, or auto)");
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
      if (const char *value = std::getenv("UNG_SPECIAL_HEAVY_EDGE_MIN_ENTRIES"))
         runtime.special_heavy_edge_min_entries = static_cast<size_t>(std::strtoull(value, nullptr, 10));
      runtime.force_use_alg = force_use_alg;
      return runtime;
   }

   QueryRouteDecision UniNavGraph::decide_query_route(const std::vector<LabelType> &query_labels,
                                                      int force_use_alg,
                                                      QueryStats &stats)
   {
      if (force_use_alg != 1 && force_use_alg != 2)
         throw std::invalid_argument("Adaptive and bitmap routes were removed; use force_use_alg=1 or 2");
      stats.query_length = query_labels.size();
      stats.candidate_set_size = query_labels.empty() ? 0 : _trie_index.get_candidate_count_for_label(query_labels.back());
      QueryRouteDecision decision;
      decision.use_new_trie = force_use_alg == 2;
      return decision;
   }

} // namespace ANNS
