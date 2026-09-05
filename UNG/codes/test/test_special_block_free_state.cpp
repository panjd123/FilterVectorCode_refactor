#include "ung_special_block_activation.h"
#include "ung_special_blocks.h"
#include "ung_special_trie_regular_search.h"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace
{
void expect(bool condition, const char *message)
{
   if (!condition)
   {
      std::cerr << "FAILED: " << message << '\n';
      std::exit(1);
   }
}
} // namespace

int main()
{
   const std::vector<ANNS::IdxType> group_to_block{0, 1, 0};
   const std::vector<ANNS::IdxType> point_to_block{0, 0, 0, 1};
   const std::vector<uint8_t> covered_blocks{0, 1};

   expect(ANNS::special_block_member_is_free("containment", group_to_block, 1,
                                             covered_blocks),
          "a covered member entry group must be free");
   expect(ANNS::special_block_member_is_free("containment", point_to_block, 3,
                                             covered_blocks),
          "a covered member neighbour point must be free");
   expect(!ANNS::special_block_member_is_free("containment", group_to_block, 2,
                                              covered_blocks),
          "a group outside the block must remain regular");
   expect(!ANNS::special_block_member_is_free("equality", group_to_block, 1,
                                              covered_blocks),
          "non-containment routes must not activate free state");
   expect(!ANNS::special_block_member_is_free("containment", group_to_block, 1,
                                              std::vector<uint8_t>{0, 0}),
          "an uncovered block must remain regular");

   expect(ANNS::special_block_successor_is_free(true, false),
          "a child block must inherit free state from its parent block");
   expect(ANNS::special_block_successor_is_free(false, true),
          "a covered block must activate free state from regular search");
   expect(!ANNS::special_block_successor_is_free(false, false),
          "regular search must stay regular before reaching a covered block");

   std::vector<ANNS::SpecialBlock> activation_blocks(4);
   activation_blocks[0].block_id = 1;
   activation_blocks[0].child_block_ids = {2, 3};
   activation_blocks[1].block_id = 2;
   activation_blocks[1].child_block_ids = {4};
   activation_blocks[2].block_id = 3;
   activation_blocks[3].block_id = 4;
   const std::vector<uint8_t> free_frontier{0, 1, 0, 0, 0};
   expect(ANNS::special_block_lazy_seed_mask(free_frontier, activation_blocks, 0) ==
              std::vector<uint8_t>({0, 1, 0, 0, 0}),
          "lazy depth zero must seed only the free frontier");
   expect(ANNS::special_block_lazy_seed_mask(free_frontier, activation_blocks, 1) ==
              std::vector<uint8_t>({0, 1, 1, 1, 0}),
          "lazy depth one must include direct child blocks");
   expect(ANNS::special_block_lazy_seed_mask(free_frontier, activation_blocks, 2) ==
              std::vector<uint8_t>({0, 1, 1, 1, 1}),
          "lazy depth two must include the next descendant layer");

   const std::vector<std::vector<ANNS::LabelType>> group_labels{
       {}, {1, 3, 5}, {1, 2, 3, 5}, {1, 4}};
   expect(ANNS::compute_direct_member_common_labels({1, 2}, group_labels) ==
              std::vector<ANNS::LabelType>({1, 3, 5}),
          "block common labels must be the exact direct-member intersection");
   expect(ANNS::compute_direct_member_common_labels({1, 2, 3}, group_labels) ==
              std::vector<ANNS::LabelType>({1}),
          "labels from an explicitly included group must participate in the intersection");
   expect(ANNS::compute_direct_member_common_labels({}, group_labels).empty(),
          "a block without direct member groups has no common labels");

   const std::vector<ANNS::IdxType> point_to_group{1, 1, 2};
   expect(ANNS::special_trie_regular_main_edge_allowed(point_to_group, 0, 1),
          "Trie regular search must retain same-group main-graph edges");
   expect(!ANNS::special_trie_regular_main_edge_allowed(point_to_group, 0, 2),
          "Trie regular search must reject LNG cross-group main-graph edges");
   expect(!ANNS::special_trie_regular_main_edge_allowed(point_to_group, 0, 3),
          "Trie regular search must reject out-of-range main-graph edges");

   std::cout << "special-block free-state activation checks passed\n";
   return 0;
}
