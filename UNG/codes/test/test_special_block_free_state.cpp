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

   const std::vector<ANNS::IdxType> point_to_upper{0, 3, 0, 3};
   const std::vector<uint8_t> covered_multilevel_blocks{0, 1, 0, 1};
   expect(ANNS::special_block_point_activation_level(
              "containment", 0, 1, point_to_upper, covered_multilevel_blocks) == 0,
          "ordinary traversal must not activate an upper-owned point directly");
   expect(ANNS::special_block_point_activation_level(
              "containment", 1, 1, point_to_upper, covered_multilevel_blocks) == 2,
          "middle traversal must activate a covered upper-owned point even across partition boundaries");
   expect(ANNS::special_block_point_activation_level(
              "containment", 1, 2, point_to_upper, covered_multilevel_blocks) == 1,
          "a point outside upper ownership must retain middle activation");
   expect(ANNS::special_block_point_activation_level(
              "containment", 1, 1, point_to_upper,
              std::vector<uint8_t>{0, 1, 0, 0}) == 1,
          "an uncovered upper block must not activate");
   expect(ANNS::special_block_point_activation_level(
              "equality", 1, 1, point_to_upper, covered_multilevel_blocks) == 1,
          "upper membership activation is containment-only");

   expect(ANNS::special_block_successor_is_free(true, false),
          "a child block must inherit free state from its parent block");
   expect(ANNS::special_block_successor_is_free(false, true),
          "a covered block must activate free state from regular search");
   expect(!ANNS::special_block_successor_is_free(false, false),
          "regular search must stay regular before reaching a covered block");

   ANNS::SpecialBlock middle_block;
   middle_block.level = 0;
   ANNS::SpecialBlock upper_block;
   upper_block.level = 1;
   const auto ordinary_to_middle =
       ANNS::special_block_edge_transition(0, middle_block, true);
   expect(ordinary_to_middle.allowed &&
              ordinary_to_middle.successor_activation_level == 1,
          "ordinary search may enter a covered middle block at level one");
   const auto ordinary_to_upper =
       ANNS::special_block_edge_transition(0, upper_block, true);
   expect(!ordinary_to_upper.allowed &&
              ordinary_to_upper.successor_activation_level == 0,
          "ordinary search must not bypass the middle layer");
   const auto middle_to_upper =
       ANNS::special_block_edge_transition(1, upper_block, true);
   expect(middle_to_upper.allowed &&
              middle_to_upper.successor_activation_level == 2,
          "middle search may enter a covered upper block at level two");
   const auto uncovered_upper =
       ANNS::special_block_edge_transition(1, upper_block, false);
   expect(!uncovered_upper.allowed &&
              uncovered_upper.successor_activation_level == 1,
          "query coverage remains mandatory and rejected edges preserve source state");
   const auto upper_to_middle =
       ANNS::special_block_edge_transition(2, middle_block, true);
   expect(upper_to_middle.allowed &&
              upper_to_middle.successor_activation_level == 2,
          "activation must be monotone when traversing a lower-layer edge");

   expect(ANNS::special_batch_gpu_search_is_allowed(true, 0),
          "single-layer indexes must retain the requested batch GPU path");
   expect(!ANNS::special_batch_gpu_search_is_allowed(true, 1),
          "multilevel indexes must use the activation-gated search path");
   expect(!ANNS::special_batch_gpu_search_is_allowed(false, 0),
          "an unset batch GPU option must remain disabled");
   expect(ANNS::special_favor_block_search_is_allowed(0),
          "FAVOR block mode remains available for legacy single-layer indexes");
   expect(!ANNS::special_favor_block_search_is_allowed(1),
          "FAVOR block mode must not silently ignore an upper layer");

   std::vector<ANNS::SpecialBlock> gate_blocks(3);
   gate_blocks[0].block_id = 1;
   gate_blocks[0].level = 0;
   gate_blocks[0].point_count = 1000;
   gate_blocks[1].block_id = 2;
   gate_blocks[1].level = 1;
   gate_blocks[1].point_count = 6000;
   gate_blocks[2].block_id = 3;
   gate_blocks[2].level = 1;
   gate_blocks[2].point_count = 5000;
   std::vector<uint8_t> gated_coverage{0, 1, 1, 1};
   const auto gate_enabled = ANNS::special_block_apply_level_gate(
       gated_coverage, gate_blocks, 2, 10000);
   expect(gate_enabled.upper_enabled && gate_enabled.upper_covered_blocks == 2 &&
              gate_enabled.upper_covered_points == 11000 &&
              gated_coverage == std::vector<uint8_t>({0, 1, 1, 1}),
          "an upper layer above the covered-point threshold must remain enabled");
   gated_coverage = {0, 1, 1, 0};
   const auto gate_below_threshold = ANNS::special_block_apply_level_gate(
       gated_coverage, gate_blocks, 2, 10000);
   expect(!gate_below_threshold.upper_enabled &&
              gated_coverage == std::vector<uint8_t>({0, 1, 0, 0}),
          "an upper layer below the covered-point threshold must be suppressed");
   gated_coverage = {0, 1, 1, 1};
   const auto gate_max_level = ANNS::special_block_apply_level_gate(
       gated_coverage, gate_blocks, 1, 0);
   expect(!gate_max_level.upper_enabled &&
              gated_coverage == std::vector<uint8_t>({0, 1, 0, 0}),
          "max activation level one must provide a strict same-index upper ablation");

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
