#ifndef ANNS_UNG_QUERY_ROUTE_H
#define ANNS_UNG_QUERY_ROUTE_H

#include "config.h"
#include "ung_entry_group.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace ANNS
{

struct QueryRouteDecision
{
   bool use_new_trie = false;
   int algorithm = 0; // 0: UNG, 1: ACORN(BFS), 2: ACORN(NoBFS)
   bool entry_groups_calculated = false;
   std::optional<bool> pre_trie_heuristic;

   bool uses_acorn() const
   {
      return algorithm == 1 || algorithm == 2;
   }

   bool use_old_bitmap_search(int force_use_alg) const
   {
      return pre_trie_heuristic.has_value() || force_use_alg == 3 || force_use_alg == 4;
   }

   bool apply_pre_trie_heuristic()
   {
      if (!pre_trie_heuristic.has_value())
         return false;
      algorithm = pre_trie_heuristic.value() ? 1 : 0;
      use_new_trie = true;
      return true;
   }

   void apply_force_route(int force_use_alg, bool bfs_filter)
   {
      switch (force_use_alg)
      {
      case 1:
         use_new_trie = false;
         algorithm = 0;
         break;
      case 2:
         use_new_trie = true;
         algorithm = 0;
         break;
      case 3:
      case 4:
         algorithm = bfs_filter ? 1 : 2;
         break;
      default:
         break;
      }
   }
};

enum class SearchGraphBackendImpl : int
{
   NeighborList = 0,
   Csr = 1,
};

const char *search_graph_backend_impl_name(SearchGraphBackendImpl impl);
SearchGraphBackendImpl parse_search_graph_backend_impl(const std::string &value);

enum class SpecialSearchMode : int
{
   FreeState = 0,
   FavorBlocks = 1,
};

bool ung_env_flag_enabled(const char *name);
bool should_stop_special_block_search(bool enabled,
                                      size_t result_size,
                                      IdxType k,
                                      float best_unexpanded_distance,
                                      float second_best_unexpanded_distance,
                                      float worst_result_distance,
                                      size_t nodes_visited,
                                      size_t min_nodes_visited);

struct SearchRuntimeConfig
{
   uint32_t num_threads = 1;
   IdxType Lsearch = 0;
   IdxType num_entry_points = 0;
   std::string scenario;
   IdxType K = 0;

   bool idea2_available = false;
   bool use_new_trie_method = false;
   bool recursive_more_start = false;
   bool ung_more_entry = false;
   bool bfs_filter = false;
   EntryGroupProviderImpl entry_group_provider = EntryGroupProviderImpl::CpuMinSuperSets;
   SearchGraphBackendImpl graph_backend = SearchGraphBackendImpl::NeighborList;
   size_t scalar_els_cap = 0;
   bool special_block_search = false;
   bool special_block_early_stop = false;
   bool special_block_prefetch = true;
   bool special_light_stats = false;
   SpecialSearchMode special_search_mode = SpecialSearchMode::FreeState;
   bool special_block_free_use_regular = false;
   // Highest Special Block activation level allowed for this query. The
   // current fixed two-layer implementation uses 1 for the historical/middle
   // overlay and 2 for the single upper overlay. UINT8_MAX enables both.
   uint8_t special_max_activation_level = std::numeric_limits<uint8_t>::max();
   // Suppress all upper overlays unless the query fully covers at least this
   // many direct member points across upper blocks. Zero disables the gate.
   size_t special_upper_min_covered_points = 0;
   bool special_heavy_edge_search = false;
   size_t special_heavy_edge_min_query_size = 0;
   size_t special_heavy_edge_min_matched_points = 0;
   size_t special_heavy_edge_min_entries = 0;

   int lsearch_start = 0;
   int lsearch_step = 0;
   int efs_start = 0;
   int efs_step_slow = 0;
   int efs_step_fast = 0;
   int lsearch_threshold = 0;
   int force_use_alg = 0;
};

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
                                                EntryGroupProviderImpl entry_group_provider =
                                                    EntryGroupProviderImpl::CpuMinSuperSets,
                                                SearchGraphBackendImpl graph_backend =
                                                    SearchGraphBackendImpl::NeighborList,
                                                size_t scalar_els_cap = 0);

} // namespace ANNS

#endif // ANNS_UNG_QUERY_ROUTE_H
