#include "ung_special_block_activation.h"
#include "ung_special_blocks.h"

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
   std::vector<ANNS::SpecialBlock> routing_blocks(3);
   routing_blocks[0].level = 0;
   routing_blocks[0].root_labels = {1, 2};
   routing_blocks[1].level = 1;
   routing_blocks[1].root_labels = {1, 3, 5};
   routing_blocks[1].point_count = 12000;
   routing_blocks[2].level = 2;
   routing_blocks[2].root_labels = {1, 3, 6};
   routing_blocks[2].point_count = 70000;
   const auto upper_authorization =
       ANNS::special_block_query_upper_authorization(
           "containment", {1, 3}, routing_blocks);
   expect(upper_authorization.level == 2 &&
              upper_authorization.block_count == 1 &&
              upper_authorization.direct_points == 70000,
          "upper authorization must expose exact highest-layer direct mass");
   expect(ANNS::special_block_upper_route_allowed(upper_authorization, 70000),
          "the upper route must be allowed at the declared mass boundary");
   expect(!ANNS::special_block_upper_route_allowed(upper_authorization, 70001),
          "the upper route must fall back below the declared mass boundary");
   expect(ANNS::special_block_query_authorizes_upper(
              "containment", {1, 3}, routing_blocks),
          "a containment query covered by an upper root must authorize multilevel search");
   expect(!ANNS::special_block_query_authorizes_upper(
              "containment", {1, 4}, routing_blocks),
          "a query outside every upper root must stay on the plain route");
   expect(!ANNS::special_block_query_authorizes_upper(
              "equality", {1, 3}, routing_blocks),
          "the upper authorization router is containment-only");
   routing_blocks[1].level = 0;
   routing_blocks[2].level = 0;
   expect(!ANNS::special_block_query_authorizes_upper(
              "containment", {1}, routing_blocks),
          "a single-level index must not claim an upper-level route");

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

   ANNS::SpecialBlock middle_block;
   middle_block.level = 0;
   ANNS::SpecialBlock upper_block;
   upper_block.level = 1;
   const auto ordinary_to_middle =
       ANNS::special_block_edge_transition(0, middle_block, true);
   expect(!ordinary_to_middle.allowed &&
              ordinary_to_middle.successor_activation_level == 0,
          "ordinary search must not consume overlay edges before activation");
   const auto ordinary_to_upper =
       ANNS::special_block_edge_transition(0, upper_block, true);
   expect(!ordinary_to_upper.allowed &&
              ordinary_to_upper.successor_activation_level == 0,
          "ordinary search must not bypass the middle layer");
   const auto middle_to_upper =
       ANNS::special_block_edge_transition(1, upper_block, true);
   expect(!middle_to_upper.allowed &&
              middle_to_upper.successor_activation_level == 1,
          "middle expansion must not mix in upper-layer edges");
   const auto middle_to_middle =
       ANNS::special_block_edge_transition(1, middle_block, true);
   expect(middle_to_middle.allowed &&
              middle_to_middle.successor_activation_level == 1,
          "middle search must retain edges owned by its current layer");
   const auto uncovered_upper =
       ANNS::special_block_edge_transition(1, upper_block, false);
   expect(!uncovered_upper.allowed &&
              uncovered_upper.successor_activation_level == 1,
          "query coverage remains mandatory and rejected edges preserve source state");
   const auto upper_to_middle =
       ANNS::special_block_edge_transition(2, middle_block, true);
   expect(!upper_to_middle.allowed &&
              upper_to_middle.successor_activation_level == 2,
          "upper search must not fall back to middle-layer edges");
   const auto upper_to_upper =
       ANNS::special_block_edge_transition(2, upper_block, true);
   expect(upper_to_upper.allowed &&
              upper_to_upper.successor_activation_level == 2,
          "upper search must retain edges owned by its current layer");

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

   std::cout << "special-block free-state activation checks passed\n";
   return 0;
}
