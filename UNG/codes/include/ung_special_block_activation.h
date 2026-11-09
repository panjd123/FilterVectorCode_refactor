#ifndef ANNS_UNG_SPECIAL_BLOCK_ACTIVATION_H
#define ANNS_UNG_SPECIAL_BLOCK_ACTIVATION_H

#include "config.h"
#include "ung_special_blocks.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ANNS
{

struct SpecialBlockUpperAuthorization
{
   uint8_t level = 0;
   size_t block_count = 0;
   size_t direct_points = 0;
};

// Query-free index construction can still make a deterministic per-query
// routing decision from persisted block metadata. Measure only the highest
// upper layer authorized by the query: direct members form a disjoint
// partition within that layer, so their mass does not double-count points
// represented again at finer layers.
inline SpecialBlockUpperAuthorization special_block_query_upper_authorization(
    const std::string &scenario,
    const std::vector<LabelType> &query_labels,
    const std::vector<SpecialBlock> &blocks)
{
   SpecialBlockUpperAuthorization result;
   if (scenario != "containment")
      return result;
   std::vector<LabelType> sorted_query = query_labels;
   std::sort(sorted_query.begin(), sorted_query.end());
   for (const SpecialBlock &block : blocks)
   {
      if (block.level == 0)
         continue;
      if (std::includes(block.root_labels.begin(), block.root_labels.end(),
                        sorted_query.begin(), sorted_query.end()))
      {
         if (block.level > result.level)
         {
            result.level = block.level;
            result.block_count = 0;
            result.direct_points = 0;
         }
         if (block.level != result.level)
            continue;
         ++result.block_count;
         result.direct_points += static_cast<size_t>(block.point_count);
      }
   }
   return result;
}

inline bool special_block_upper_route_allowed(
    const SpecialBlockUpperAuthorization &authorization,
    size_t minimum_direct_points)
{
   return authorization.block_count > 0 &&
          authorization.direct_points >= minimum_direct_points;
}

inline bool special_block_query_authorizes_upper(
    const std::string &scenario,
    const std::vector<LabelType> &query_labels,
    const std::vector<SpecialBlock> &blocks)
{
   return special_block_upper_route_allowed(
       special_block_query_upper_authorization(scenario, query_labels, blocks), 0);
}

inline bool special_block_member_is_free(const std::string &scenario,
                                         const std::vector<IdxType> &owner_to_block,
                                         IdxType owner_id,
                                         const std::vector<uint8_t> &query_covers_block)
{
   if (scenario != "containment" || owner_id >= owner_to_block.size())
      return false;
   const IdxType block_id = owner_to_block[owner_id];
   return block_id > 0 && block_id < query_covers_block.size() &&
          query_covers_block[block_id] != 0;
}

inline uint8_t special_block_activation_level(const SpecialBlock &block)
{
   return static_cast<uint8_t>(block.level + 1);
}

// Return every query-authorized materialized state for a point, ordered from
// the coarsest (highest activation level) to the finest. Callers select the
// levels at or above the point's current state, making cross-layer movement
// monotone while sharing one cached vector distance across graph states.
inline std::vector<uint8_t> special_block_authorized_levels_for_point(
    IdxType point_id,
    const std::vector<std::vector<IdxType>> &point_to_block_by_level,
    const std::vector<uint8_t> &query_free_block)
{
   std::vector<uint8_t> levels;
   levels.reserve(point_to_block_by_level.size());
   for (size_t level_index = point_to_block_by_level.size(); level_index > 0;
        --level_index)
   {
      const std::vector<IdxType> &owners =
          point_to_block_by_level[level_index - 1];
      if (point_id >= owners.size())
         continue;
      const IdxType block_id = owners[point_id];
      if (block_id == 0 || block_id >= query_free_block.size() ||
          query_free_block[block_id] == 0)
         continue;
      levels.push_back(static_cast<uint8_t>(level_index));
   }
   return levels;
}

// A point discovered while expanding current_level may remain in that layer
// and may be promoted to any query-authorized coarser layer. It must never
// create a finer-layer state: once traversal has moved upward, search is
// monotone and cannot fall back to a lower overlay or to the base graph.
inline std::vector<uint8_t> special_block_monotone_levels_for_point(
    IdxType point_id, uint8_t current_level,
    const std::vector<std::vector<IdxType>> &point_to_block_by_level,
    const std::vector<uint8_t> &query_free_block)
{
   std::vector<uint8_t> levels = special_block_authorized_levels_for_point(
       point_id, point_to_block_by_level, query_free_block);
   levels.erase(
       std::remove_if(levels.begin(), levels.end(),
                      [current_level](uint8_t level) {
                         return level < current_level;
                      }),
       levels.end());
   if (std::find(levels.begin(), levels.end(), current_level) == levels.end())
      levels.push_back(current_level);
   std::sort(levels.begin(), levels.end(), std::greater<uint8_t>());
   return levels;
}

struct SpecialBlockEdgeTransition
{
   bool allowed = false;
   uint8_t successor_activation_level = 0;
};

// Authorize only the overlay owned by the candidate's current activation
// level. Cross-layer movement is represented by inserting another state for
// the same point; an edge itself never silently changes layer.
inline SpecialBlockEdgeTransition special_block_edge_transition(
    uint8_t current_level, const SpecialBlock &owner, bool query_covers_owner)
{
   SpecialBlockEdgeTransition transition;
   transition.successor_activation_level = current_level;
   if (!query_covers_owner)
      return transition;

   const uint8_t owner_level = special_block_activation_level(owner);
   if (owner_level != current_level)
      return transition;

   transition.allowed = true;
   return transition;
}

inline bool special_batch_gpu_search_is_allowed(bool requested,
                                                size_t upper_block_count)
{
   return requested && upper_block_count == 0;
}

// FAVOR's block-selection backend predates activation levels and indexes only
// the middle ownership map. Running it over a multilevel sidecar would appear
// successful while silently ignoring every upper block.
inline bool special_favor_block_search_is_allowed(size_t upper_block_count)
{
   return upper_block_count == 0;
}

struct SpecialBlockLevelGateResult
{
   size_t upper_covered_blocks = 0;
   size_t upper_covered_points = 0;
   bool upper_enabled = true;
};

// Apply a per-query gate after coverage has been propagated through each
// layer. point_count contains direct members, so summing covered blocks within
// one partition level does not double-count descendants. This gate changes
// only which overlay levels may activate; it never changes ELS, ordinary graph
// edges, or the middle-layer coverage mask.
inline SpecialBlockLevelGateResult special_block_apply_level_gate(
    std::vector<uint8_t> &query_free_block,
    const std::vector<SpecialBlock> &blocks,
    uint8_t max_activation_level,
    size_t upper_min_covered_points)
{
   SpecialBlockLevelGateResult result;
   for (IdxType block_id = 1; block_id < query_free_block.size(); ++block_id)
   {
      if (query_free_block[block_id] == 0 || block_id > blocks.size())
         continue;
      const SpecialBlock &block = blocks[block_id - 1];
      if (block.level == 0)
         continue;
      ++result.upper_covered_blocks;
      result.upper_covered_points += static_cast<size_t>(block.point_count);
   }

   result.upper_enabled = max_activation_level >= 2 &&
                          result.upper_covered_points >= upper_min_covered_points;
   for (IdxType block_id = 1; block_id < query_free_block.size(); ++block_id)
   {
      if (block_id > blocks.size())
         continue;
      const uint8_t activation_level = special_block_activation_level(blocks[block_id - 1]);
      if (activation_level > max_activation_level ||
          (blocks[block_id - 1].level > 0 && !result.upper_enabled))
         query_free_block[block_id] = 0;
   }
   return result;
}

} // namespace ANNS

#endif // ANNS_UNG_SPECIAL_BLOCK_ACTIVATION_H
