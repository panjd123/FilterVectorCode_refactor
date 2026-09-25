#include "include/uni_nav_graph.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace
{
bool disable_entry_route_stats()
{
   const char *value = std::getenv("UNG_DISABLE_ENTRY_ROUTE_STATS");
   if (value == nullptr)
      return false;
   return value[0] != '\0' && !(value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
                                value[0] == 'n' || value[0] == 'N' || value[0] == 'o' ||
                                value[0] == 'O');
}
}

namespace ANNS
{

void UniNavGraph::prepare_entry_groups_for_execution(
    const EntryGroupProviderRequest &request,
    std::vector<IdxType> &entry_group_ids,
    QueryStats &stats)
{
   EntryGroupProviderResult result = run_entry_group_provider(request, stats);
   entry_group_ids = std::move(result.group_ids);
   stats.num_entry_points = entry_group_ids.size();
   if (disable_entry_route_stats())
      return;
   if (result.route_stats.valid)
      apply_entry_group_route_stats(result.route_stats, stats);
   else
      populate_entry_group_route_stats(entry_group_ids, stats);
}

EntryGroupProviderResult UniNavGraph::run_entry_group_provider(
    const EntryGroupProviderRequest &request,
    QueryStats &stats)
{
   request.validate();
   EntryGroupProviderResult result;
   switch (request.strategy)
   {
   case EntryGroupStrategy::Original:
      result = compute_original_entry_groups_for_execution(request, stats);
      break;
   case EntryGroupStrategy::OptimizedLng:
      result = compute_optimized_lng_entry_groups_for_execution(request, stats);
      break;
   case EntryGroupStrategy::Trie:
      result = compute_trie_entry_groups_for_execution(request, stats);
      break;
   }

   // Preserve the legacy optional expansion as a strategy-independent
   // post-processing policy. It must not select a different implementation.
   if (request.ung_more_entry && request.decision->algorithm == 0)
   {
      const IdxType true_group_id = request.true_group_id_or(0);
      const size_t extra_k = result.group_ids.size() / 5;
      result.group_ids = select_entry_groups(result.group_ids,
                                             SelectionMode::SizeAndDistance,
                                             extra_k, 1.0, true_group_id);
      result.exact_minimal = false;
   }
   return result;
}

EntryGroupProviderResult UniNavGraph::compute_original_entry_groups_for_execution(
    const EntryGroupProviderRequest &request,
    QueryStats &stats)
{
   const auto start = std::chrono::high_resolution_clock::now();
   EntryGroupProviderResult result;
   result.requested_strategy = request.strategy;
   result.provider = EntryGroupProviderKind::Original;
   result.exact_minimal = true;
   if (request.has_current_group_ids())
      result.group_ids = *request.current_group_ids;

   const QueryRouteDecision &decision = *request.decision;
   if (!decision.entry_groups_calculated && decision.algorithm == 0)
      get_min_super_sets_original_sort(*request.query_labels, result.group_ids,
                                       false, true);
   else if (!decision.entry_groups_calculated)
      result.provider = EntryGroupProviderKind::Passthrough;

   result.elapsed_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::high_resolution_clock::now() - start)
                           .count();
   stats.get_min_super_sets_time_ms = result.elapsed_ms;
   return result;
}

EntryGroupProviderResult UniNavGraph::compute_optimized_lng_entry_groups_for_execution(
    const EntryGroupProviderRequest &request,
    QueryStats &stats)
{
   const auto start = std::chrono::high_resolution_clock::now();
   EntryGroupProviderResult result;
   result.requested_strategy = request.strategy;
   result.provider = EntryGroupProviderKind::OptimizedLng;
   result.exact_minimal = true;
   if (request.has_current_group_ids())
      result.group_ids = *request.current_group_ids;

   const QueryRouteDecision &decision = *request.decision;
   if (!decision.entry_groups_calculated && decision.algorithm == 0)
      get_min_super_sets_optimized_bucket(*request.query_labels, result.group_ids,
                                          false, true);
   else if (!decision.entry_groups_calculated)
      result.provider = EntryGroupProviderKind::Passthrough;

   result.elapsed_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::high_resolution_clock::now() - start)
                           .count();
   stats.get_min_super_sets_time_ms = result.elapsed_ms;
   return result;
}

EntryGroupProviderResult UniNavGraph::compute_trie_entry_groups_for_execution(
    const EntryGroupProviderRequest &request,
    QueryStats &stats)
{
   const auto start = std::chrono::high_resolution_clock::now();
   EntryGroupProviderResult result;
   result.requested_strategy = request.strategy;
   result.provider = EntryGroupProviderKind::Trie;
   result.exact_minimal = false;
   if (request.has_current_group_ids())
      result.group_ids = *request.current_group_ids;

   const QueryRouteDecision &decision = *request.decision;
   SpecialBlockTrieSearchStats trie_stats;
   if (!decision.entry_groups_calculated && decision.algorithm == 0)
   {
      if (_group_trie_index.empty())
         throw std::runtime_error("trie entry strategy requires the group trie index");
      result.group_ids = _group_trie_index.find_entry_groups_bitset(
          *request.query_labels, &trie_stats);
   }
   else if (!decision.entry_groups_calculated)
   {
      result.provider = EntryGroupProviderKind::Passthrough;
   }

   result.elapsed_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::high_resolution_clock::now() - start)
                           .count();
   stats.get_min_super_sets_time_ms = result.elapsed_ms;
   stats.candidate_set_size = trie_stats.pivot_postings;
   stats.successful_checks = trie_stats.matching_pivots;
   stats.trie_nodes_traversed = static_cast<long long>(
       trie_stats.upward_nodes_visited + trie_stats.downward_nodes_visited);
   stats.special_trie_pivot_postings = trie_stats.pivot_postings;
   stats.special_trie_matching_pivots = trie_stats.matching_pivots;
   stats.special_trie_upward_nodes_visited = trie_stats.upward_nodes_visited;
   stats.special_trie_downward_nodes_visited = trie_stats.downward_nodes_visited;
   stats.special_trie_branches_pruned = trie_stats.branches_pruned;
   stats.special_trie_terminal_candidates = trie_stats.terminal_candidates;
   stats.special_trie_terminal_descendants_pruned =
       trie_stats.terminal_descendants_pruned;
   stats.special_trie_final_entries = result.group_ids.size();
   stats.special_trie_time_ms = result.elapsed_ms;
   return result;
}

} // namespace ANNS
