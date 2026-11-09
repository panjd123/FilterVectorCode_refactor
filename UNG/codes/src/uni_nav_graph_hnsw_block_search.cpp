#include "include/uni_nav_graph.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <queue>
#include <vector>

#include <roaring/roaring.hh>

namespace ANNS
{
namespace
{
uint32_t upper_env_u32(const char *name, uint32_t fallback)
{
   if (const char *value = std::getenv(name))
      return static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
   return fallback;
}

float upper_env_float(const char *name, float fallback)
{
   if (const char *value = std::getenv(name))
      return std::strtof(value, nullptr);
   return fallback;
}

bool upper_env_bool(const char *name, bool fallback)
{
   const char *value = std::getenv(name);
   if (value == nullptr)
      return fallback;
   return !(value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
            value[0] == 'n' || value[0] == 'N' || value[0] == 'o' || value[0] == 'O');
}

struct HnswBlockCandidate
{
   IdxType id = 0;
   IdxType block_id = 0;
   float score = 0.0f;
   float raw_distance = 0.0f;
   bool target = false;
};

struct HnswBlockBestFirst
{
   bool operator()(const HnswBlockCandidate &a, const HnswBlockCandidate &b) const
   {
      if (a.score != b.score)
         return a.score > b.score;
      return a.id > b.id;
   }
};

struct HnswBlockWorstFirst
{
   bool operator()(const HnswBlockCandidate &a, const HnswBlockCandidate &b) const
   {
      if (a.score != b.score)
         return a.score < b.score;
      return a.id < b.id;
   }
};

struct HnswBlockResult
{
   IdxType id = 0;
   float distance = 0.0f;
};

struct HnswBlockWorstResult
{
   bool operator()(const HnswBlockResult &a, const HnswBlockResult &b) const
   {
      if (a.distance != b.distance)
         return a.distance < b.distance;
      return a.id < b.id;
   }
};
} // namespace

bool UniNavGraph::execute_hnsw_block_ung_query(
    const char *query,
    std::shared_ptr<SearchCache> search_cache,
    const SearchRuntimeConfig &runtime,
    const GraphSearchBackend &graph_backend,
    const std::vector<IdxType> &entry_group_ids,
    IdxType query_id,
    std::vector<float> &num_cmps,
    SearchQueue &cur_result,
    QueryStats &stats)
{
   const auto search_start = std::chrono::high_resolution_clock::now();
   stats.special_search_enabled = true;
   stats.favor_block_search_enabled = true;
   stats.special_free_use_regular = false;

   if (!has_upper_hnsw() || runtime.Lsearch == 0 || entry_group_ids.empty())
   {
      stats.num_distance_calcs = 0;
      return false;
   }

   const size_t upper_top_n = std::max<size_t>(
       1, upper_env_u32("UNG_HNSW_BLOCK_UPPER_TOPN", 128));
   const size_t ef_upper = std::max<size_t>(
       upper_top_n, upper_env_u32("UNG_HNSW_BLOCK_EF_UPPER", 256));
   const size_t requested_entries = std::max<size_t>(
       1, upper_env_u32("UNG_HNSW_BLOCK_ENTRIES", 32));
   const size_t max_entries_per_block = std::max<size_t>(
       1, upper_env_u32("UNG_HNSW_BLOCK_MAX_ENTRIES_PER_BLOCK", 1));
   const size_t min_block_quota = upper_env_u32("UNG_HNSW_BLOCK_MIN_QUOTA", 4);
   const float penalty = std::max(0.0f, upper_env_float("UNG_HNSW_BLOCK_PENALTY", 0.0f));
   const float bound_factor = upper_env_float("UNG_HNSW_BLOCK_BOUND_FACTOR", 0.0f);
   const bool use_regular_edges = upper_env_bool("UNG_HNSW_BLOCK_USE_REGULAR_EDGES", false);
   const size_t expansion_budget = std::max<size_t>(
       runtime.Lsearch,
       static_cast<size_t>(std::ceil(
           static_cast<double>(runtime.Lsearch) *
           std::max(1.0f, upper_env_float("UNG_HNSW_BLOCK_EXPANSION_FACTOR", 1.0f)))));
   const size_t nav_capacity = std::max<size_t>(runtime.Lsearch, requested_entries);
   const size_t result_capacity = std::max<size_t>(runtime.Lsearch, runtime.K);
   const IdxType dim = _base_storage->get_dim();

   roaring::Roaring target_bitmap = collect_target_points(entry_group_ids);
   stats.favor_target_points = target_bitmap.cardinality();
   stats.favor_selectivity = _num_points == 0 ? 0.0f :
       static_cast<float>(stats.favor_target_points) / static_cast<float>(_num_points);
   stats.favor_exclusion_distance = penalty;

   auto &target_map = search_cache->favor_target_map;
   auto &target_touched = search_cache->favor_target_touched;
   if (target_map.size() < _num_points)
      target_map.assign(_num_points, 0);
   for (IdxType point_id : target_touched)
      if (point_id < target_map.size())
         target_map[point_id] = 0;
   target_touched.clear();
   auto is_target_point = [&](IdxType point_id) {
      if (point_id >= _num_points || point_id >= target_map.size())
         return false;
      uint8_t &cached = target_map[point_id];
      if (cached == 0)
      {
         bool is_target = false;
         if (point_id < _new_to_old_vec_ids.size())
         {
            const IdxType original_id = _new_to_old_vec_ids[point_id];
            is_target = target_bitmap.contains(static_cast<uint32_t>(original_id));
         }
         cached = is_target ? 2 : 1;
         target_touched.push_back(point_id);
      }
      return cached == 2;
   };

   const std::vector<UpperHnswCandidate> upper_candidates =
       _upper_hnsw->search(reinterpret_cast<const float *>(query), upper_top_n, ef_upper);
   if (upper_candidates.empty())
      return false;

   auto &selected_block = search_cache->favor_selected_blocks;
   if (selected_block.size() < _special_blocks.size() + 1)
      selected_block.assign(_special_blocks.size() + 1, 0);
   else
      std::fill(selected_block.begin(), selected_block.end(), 0);
   std::vector<size_t> entries_per_block(_special_blocks.size() + 1, 0);
   std::vector<UpperHnswCandidate> entries;
   entries.reserve(requested_entries);

   auto select_entries = [&](bool require_target, bool enforce_block_cap) {
      for (const UpperHnswCandidate &candidate : upper_candidates)
      {
         if (entries.size() >= requested_entries)
            break;
         if (candidate.point_id >= target_map.size() || candidate.block_id == 0 ||
             candidate.block_id >= entries_per_block.size())
            continue;
         const bool is_target = is_target_point(candidate.point_id);
         if (is_target != require_target)
            continue;
         if (enforce_block_cap && entries_per_block[candidate.block_id] >= max_entries_per_block)
            continue;
         bool duplicate = false;
         for (const UpperHnswCandidate &entry : entries)
            if (entry.point_id == candidate.point_id)
               duplicate = true;
         if (duplicate)
            continue;
         entries.push_back(candidate);
         entries_per_block[candidate.block_id]++;
         selected_block[candidate.block_id] = 1;
      }
   };
   select_entries(true, true);
   select_entries(false, true);
   select_entries(true, false);
   select_entries(false, false);
   if (entries.empty())
      return false;

   std::vector<IdxType> selected_block_ids;
   for (IdxType block_id = 1; block_id < selected_block.size(); ++block_id)
   {
      if (selected_block[block_id] == 0)
         continue;
      selected_block_ids.push_back(block_id);
      stats.favor_block_count++;
      if (block_id <= _special_blocks.size())
         stats.favor_block_points += _special_blocks[block_id - 1].point_count;
   }

   auto &block_expanded = search_cache->favor_block_source_expanded;
   if (block_expanded.size() < selected_block.size())
      block_expanded.assign(selected_block.size(), 0);
   else
      std::fill(block_expanded.begin(), block_expanded.end(), 0);

   auto &visited = search_cache->special_visited_regular;
   visited.clear();
   auto &active = search_cache->favor_candidate_active;
   if (active.size() < _num_points)
      active.assign(_num_points, 0);
   else
      std::fill(active.begin(), active.end(), 0);
   auto &expanded = search_cache->special_free_state_cache;
   auto &expanded_touched = search_cache->special_free_state_touched;
   if (expanded.size() < _num_points)
      expanded.assign(_num_points, 0);
   for (IdxType point_id : expanded_touched)
      if (point_id < expanded.size())
         expanded[point_id] = 0;
   expanded_touched.clear();

   using CandidateHeap = std::priority_queue<
       HnswBlockCandidate, std::vector<HnswBlockCandidate>, HnswBlockBestFirst>;
   CandidateHeap frontier;
   std::priority_queue<HnswBlockCandidate,
                       std::vector<HnswBlockCandidate>,
                       HnswBlockWorstFirst> frontier_worst;
   std::vector<CandidateHeap> quota_frontiers(selected_block.size());
   std::priority_queue<HnswBlockResult,
                       std::vector<HnswBlockResult>,
                       HnswBlockWorstResult> result_beam;
   std::priority_queue<HnswBlockResult,
                       std::vector<HnswBlockResult>,
                       HnswBlockWorstResult> topk_bound;
   size_t frontier_active = 0;
   size_t comparisons = 0;

   auto clean_frontier_worst = [&]() {
      while (!frontier_worst.empty())
      {
         const IdxType id = frontier_worst.top().id;
         if (id < active.size() && active[id] != 0)
            break;
         frontier_worst.pop();
      }
   };

   auto offer_result = [&](IdxType id, float distance) {
      result_beam.push(HnswBlockResult{id, distance});
      if (result_beam.size() > result_capacity)
         result_beam.pop();
      topk_bound.push(HnswBlockResult{id, distance});
      if (topk_bound.size() > static_cast<size_t>(runtime.K))
         topk_bound.pop();
   };

   auto discover = [&](IdxType id, float known_distance, bool has_known_distance) {
      if (id >= _num_points || visited.check(id))
         return false;
      visited.set(id);
      const bool is_target = is_target_point(id);
      const float raw_distance = has_known_distance
          ? known_distance
          : _distance_handler->compute(query, _base_storage->get_vector(id), dim);
      if (!has_known_distance)
         comparisons++;
      stats.num_nodes_visited++;
      if (is_target)
      {
         stats.favor_td_candidates_inserted++;
         offer_result(id, raw_distance);
      }
      else
      {
         stats.favor_ntd_candidates_inserted++;
      }

      IdxType block_id = 0;
      if (!_point_to_special_block_by_level.empty() &&
          id < _point_to_special_block_by_level.front().size())
         block_id = _point_to_special_block_by_level.front()[id];
      const HnswBlockCandidate candidate{
          id, block_id, raw_distance + (is_target ? 0.0f : penalty), raw_distance, is_target};
      const bool quota_protected =
          block_id > 0 && block_id < selected_block.size() &&
          selected_block[block_id] != 0 && block_expanded[block_id] < min_block_quota;
      clean_frontier_worst();
      if (!quota_protected && frontier_active >= nav_capacity &&
          !frontier_worst.empty() &&
          candidate.score >= frontier_worst.top().score)
         return false;

      active[id] = 1;
      frontier.push(candidate);
      frontier_worst.push(candidate);
      frontier_active++;
      if (block_id < quota_frontiers.size() && selected_block[block_id] != 0)
         quota_frontiers[block_id].push(candidate);

      clean_frontier_worst();
      if (frontier_active > nav_capacity && !frontier_worst.empty())
      {
         const IdxType removed = frontier_worst.top().id;
         frontier_worst.pop();
         if (removed < active.size() && active[removed] != 0)
         {
            active[removed] = 0;
            frontier_active--;
         }
      }
      return true;
   };

   for (const UpperHnswCandidate &entry : entries)
      discover(entry.point_id, entry.distance, true);
   comparisons += upper_candidates.size();

   size_t expansions = 0;
   auto expand_candidate = [&](const HnswBlockCandidate &candidate) {
      if (candidate.id >= expanded.size() || expanded[candidate.id] != 0)
         return false;
      expanded[candidate.id] = 1;
      expanded_touched.push_back(candidate.id);
      if (candidate.id < active.size() && active[candidate.id] != 0)
      {
         active[candidate.id] = 0;
         if (frontier_active > 0)
            frontier_active--;
      }
      expansions++;
      if (candidate.target)
         stats.special_regular_nodes_expanded++;
      else
         stats.special_free_nodes_expanded++;
      if (candidate.block_id > 0 && candidate.block_id < block_expanded.size())
      {
         block_expanded[candidate.block_id]++;
         stats.favor_blocks_expanded++;
      }

      const auto special_edges = special_edges_for_point(candidate.id);
      for (const SpecialEdge edge : special_edges)
      {
         stats.special_edges_scanned++;
         if (edge.kind == SpecialEdgeKind::InterBlock)
            stats.special_inter_edges_scanned++;
         else
            stats.special_intra_edges_scanned++;
         if (discover(edge.target_point_id, 0.0f, false))
            stats.special_edges_accepted++;
      }
      if (use_regular_edges)
      {
         const GraphNeighborView neighbors = graph_backend.neighbors(candidate.id);
         for (size_t i = 0; i < neighbors.size; ++i)
         {
            stats.special_regular_edges_scanned++;
            if (discover(neighbors.ids[i], 0.0f, false))
               stats.special_regular_edges_accepted++;
         }
      }
      return true;
   };

   for (IdxType block_id : selected_block_ids)
   {
      CandidateHeap &quota_frontier = quota_frontiers[block_id];
      while (block_expanded[block_id] < min_block_quota &&
             expansions < expansion_budget && !quota_frontier.empty())
      {
         const HnswBlockCandidate candidate = quota_frontier.top();
         quota_frontier.pop();
         if (candidate.id >= active.size() || active[candidate.id] == 0)
            continue;
         expand_candidate(candidate);
      }
   }

   while (expansions < expansion_budget && !frontier.empty())
   {
      const HnswBlockCandidate candidate = frontier.top();
      frontier.pop();
      if (candidate.id >= active.size() || active[candidate.id] == 0)
         continue;
      const bool quota_satisfied =
          candidate.block_id == 0 || candidate.block_id >= block_expanded.size() ||
          selected_block[candidate.block_id] == 0 ||
          block_expanded[candidate.block_id] >= min_block_quota;
      if (bound_factor > 0.0f && quota_satisfied &&
          topk_bound.size() >= static_cast<size_t>(runtime.K) &&
          candidate.raw_distance > topk_bound.top().distance * bound_factor)
      {
         active[candidate.id] = 0;
         if (frontier_active > 0)
            frontier_active--;
         stats.favor_block_edge_scans_skipped++;
         continue;
      }
      expand_candidate(candidate);
   }

   std::vector<HnswBlockResult> final_results;
   final_results.reserve(result_beam.size());
   while (!result_beam.empty())
   {
      final_results.push_back(result_beam.top());
      result_beam.pop();
   }
   std::sort(final_results.begin(), final_results.end(),
             [](const HnswBlockResult &a, const HnswBlockResult &b) {
                if (a.distance != b.distance)
                   return a.distance < b.distance;
                return a.id < b.id;
             });
   cur_result.clear();
   for (const HnswBlockResult &result : final_results)
   {
      cur_result.insert(result.id, result.distance);
      stats.favor_td_results++;
      if (cur_result.size() >= static_cast<int32_t>(runtime.K))
         break;
   }

   stats.special_regular_distance_calcs = stats.favor_td_candidates_inserted;
   stats.special_free_distance_calcs = stats.favor_ntd_candidates_inserted;
   stats.num_distance_calcs = comparisons;
   num_cmps[query_id] = static_cast<float>(comparisons);
   stats.core_search_time_ms =
       std::chrono::duration<double, std::milli>(
           std::chrono::high_resolution_clock::now() - search_start)
           .count();
   stats.search_time_ms = stats.core_search_time_ms;
   stats.regular_edges_scanned = stats.special_regular_edges_scanned;
   stats.free_edges_scanned = stats.special_edges_scanned;
   return cur_result.size() > 0;
}

} // namespace ANNS
