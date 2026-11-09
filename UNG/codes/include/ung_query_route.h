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

// Fixed graph route and entry-provider state. No learned or bitmap route.
struct QueryRouteDecision
{
   bool use_new_trie = false;
   int algorithm = 0; // Graph expansion; retained for entry-provider state.
   bool entry_groups_calculated = false;
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
   HnswBlocks = 2,
   Auto = 3,
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

   bool recursive_more_start = false;
   bool ung_more_entry = false;
   EntryGroupStrategy entry_group_strategy = EntryGroupStrategy::OptimizedLng;
   SearchGraphBackendImpl graph_backend = SearchGraphBackendImpl::NeighborList;
   bool special_block_search = false;
   bool special_block_early_stop = false;
   bool special_block_prefetch = true;
   bool special_light_stats = false;
   SpecialSearchMode special_search_mode = SpecialSearchMode::FreeState;
   // Highest materialized activation level allowed for this query. Level zero
   // is the base graph; materialized hierarchy layer i uses activation i + 1.
   uint8_t special_max_activation_level = std::numeric_limits<uint8_t>::max();
   // Suppress all upper overlays unless the query fully covers at least this
   // many direct member points across upper blocks. Zero disables the gate.
   size_t special_upper_min_covered_points = 0;
   bool special_heavy_edge_search = false;
   size_t special_heavy_edge_min_query_size = 0;
   size_t special_heavy_edge_min_entries = 0;

   int force_use_alg = 1;
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
                                                EntryGroupStrategy entry_group_strategy =
                                                    EntryGroupStrategy::OptimizedLng,
                                                SearchGraphBackendImpl graph_backend =
                                                    SearchGraphBackendImpl::NeighborList);

} // namespace ANNS

#endif // ANNS_UNG_QUERY_ROUTE_H
