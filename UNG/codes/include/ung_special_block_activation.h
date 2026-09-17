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

// Query-free index construction can still make a deterministic per-query
// routing decision: use a multilevel overlay only when the containment
// predicate fully authorizes at least one upper block.  This asks no latency,
// Recall, or selectivity model; it reuses the exact root-label condition that
// guards upper-level traversal.
inline bool special_block_query_authorizes_upper(
    const std::string &scenario,
    const std::vector<LabelType> &query_labels,
    const std::vector<SpecialBlock> &blocks)
{
   if (scenario != "containment")
      return false;
   std::vector<LabelType> sorted_query = query_labels;
   std::sort(sorted_query.begin(), sorted_query.end());
   for (const SpecialBlock &block : blocks)
   {
      if (block.level == 0)
         continue;
      if (std::includes(block.root_labels.begin(), block.root_labels.end(),
                        sorted_query.begin(), sorted_query.end()))
         return true;
   }
   return false;
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

inline bool special_block_successor_is_free(bool current_is_free,
                                            bool target_block_is_covered)
{
   return current_is_free || target_block_is_covered;
}

// Ordinary traversal may activate only the middle layer. Once a candidate is
// already middle-active, arriving at any point directly owned by a covered
// upper block promotes it to level two. This point-membership rule is needed
// even when independently partitioned middle and upper blocks do not form a
// strict refinement.
inline uint8_t special_block_point_activation_level(
    const std::string &scenario, uint8_t current_level, IdxType point_id,
    const std::vector<IdxType> &point_to_upper_block,
    const std::vector<uint8_t> &query_covers_block)
{
   if (current_level == 0 || scenario != "containment" ||
       point_id >= point_to_upper_block.size())
      return current_level;
   const IdxType upper_block_id = point_to_upper_block[point_id];
   if (upper_block_id == 0 || upper_block_id >= query_covers_block.size() ||
       query_covers_block[upper_block_id] == 0)
      return current_level;
   return std::max<uint8_t>(current_level, 2);
}

inline uint8_t special_block_activation_level(const SpecialBlock &block)
{
   return static_cast<uint8_t>(block.level + 1);
}

struct SpecialBlockEdgeTransition
{
   bool allowed = false;
   uint8_t successor_activation_level = 0;
};

// Authorize only the overlay owned by the candidate's current activation
// level. Promotion is a point-state transition performed before the target is
// queued, so one expansion never mixes edges from two overlay levels.
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

inline std::vector<uint8_t> special_block_lazy_seed_mask(
    const std::vector<uint8_t> &free_frontier,
    const std::vector<SpecialBlock> &blocks,
    size_t descendant_depth)
{
   std::vector<uint8_t> selected(free_frontier.size(), 0);
   std::vector<std::pair<IdxType, size_t>> pending;
   pending.reserve(blocks.size());
   for (IdxType block_id = 1; block_id < free_frontier.size(); ++block_id)
   {
      if (free_frontier[block_id] == 0)
         continue;
      selected[block_id] = 1;
      pending.emplace_back(block_id, 0);
   }

   for (size_t cursor = 0; cursor < pending.size(); ++cursor)
   {
      const IdxType block_id = pending[cursor].first;
      const size_t depth = pending[cursor].second;
      if (depth >= descendant_depth || block_id == 0 || block_id > blocks.size())
         continue;
      for (IdxType child_block_id : blocks[block_id - 1].child_block_ids)
      {
         if (child_block_id == 0 || child_block_id >= selected.size() ||
             selected[child_block_id] != 0)
            continue;
         selected[child_block_id] = 1;
         pending.emplace_back(child_block_id, depth + 1);
      }
   }
   return selected;
}

} // namespace ANNS

#endif // ANNS_UNG_SPECIAL_BLOCK_ACTIVATION_H
