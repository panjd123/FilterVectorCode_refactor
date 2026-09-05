#ifndef ANNS_UNG_SPECIAL_BLOCK_ACTIVATION_H
#define ANNS_UNG_SPECIAL_BLOCK_ACTIVATION_H

#include "config.h"
#include "ung_special_blocks.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ANNS
{

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

inline uint8_t special_block_activation_level(const SpecialBlock &block)
{
   return static_cast<uint8_t>(block.level + 1);
}

inline bool special_block_edge_is_allowed(uint8_t current_level,
                                          const SpecialBlock &owner,
                                          bool query_covers_owner)
{
   return query_covers_owner &&
          special_block_activation_level(owner) <= current_level + 1;
}

inline uint8_t special_block_successor_activation_level(
    uint8_t current_level, const SpecialBlock &owner)
{
   return std::max(current_level, special_block_activation_level(owner));
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
