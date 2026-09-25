#include "ung_group_topology.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
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
}

int main()
{
   std::vector<std::vector<ANNS::LabelType>> labels{
       {}, {1}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}};
   auto lng = std::make_shared<ANNS::LabelNavGraph>(labels.size());
   lng->out_neighbors[1] = {2, 3};
   lng->out_neighbors[2] = {5};
   lng->out_neighbors[3] = {5};
   lng->out_neighbors[4] = {5};
   lng->in_neighbors[2] = {1};
   lng->in_neighbors[3] = {1};
   lng->in_neighbors[5] = {2, 3, 4};

   ANNS::GroupTopologyStats lng_stats;
   const auto lng_result = ANNS::build_group_topology(
       ANNS::GroupTopologyKind::Lng, labels, 5, lng, &lng_stats);
   expect(lng_result.get() == lng.get(), "LNG topology should share the semantic LNG");
   expect(lng_stats.edge_count == 5, "LNG edge count must be reported");

   ANNS::GroupTopologyStats trie_stats;
   const auto trie = ANNS::build_group_topology(
       ANNS::GroupTopologyKind::Trie, labels, 5, lng, &trie_stats);
   expect(trie.get() != lng.get(), "Trie topology must be independent from LNG");
   expect(trie->out_neighbors[1] == std::vector<ANNS::IdxType>({2, 3}),
          "Trie should connect a terminal to nearest prefix-terminal descendants");
   expect(trie->out_neighbors[2] == std::vector<ANNS::IdxType>({5}),
          "Trie should preserve the canonical prefix chain");
   expect(trie->out_neighbors[4].empty(),
          "non-prefix set containment is not a Trie edge");
   expect(trie_stats.edge_count == 3, "Trie edge count must expose its sparser topology");

   const ANNS::HierarchyPlan plan = ANNS::parse_hierarchy_plan(
       "1000:trie,16000:lng,400000:trie", ANNS::GroupTopologyKind::Trie);
   expect(plan.layer_count() == 3, "hierarchy parser must support arbitrary depth");
   expect(plan.encode() == "1000:trie,16000:lng,400000:trie",
          "hierarchy plan must round-trip");

   std::vector<ANNS::SpecialBlock> blocks(6);
   for (size_t i = 0; i < blocks.size(); ++i)
      blocks[i].block_id = static_cast<ANNS::IdxType>(i + 1);
   blocks[0].level = 0;
   blocks[0].root_labels = {1};
   blocks[0].child_block_ids = {2, 3};
   blocks[1].level = 0;
   blocks[1].root_labels = {1, 2};
   blocks[2].level = 0;
   blocks[2].root_labels = {1, 3};
   blocks[3].level = 1;
   blocks[3].root_labels = {2};
   blocks[4].level = 1;
   blocks[4].root_labels = {2, 4};
   blocks[4].child_block_ids = {6};
   blocks[5].level = 1;
   blocks[5].root_labels = {2, 4, 5};

   ANNS::apply_layer_topology(ANNS::GroupTopologyKind::Lng, 0, blocks);
   expect(blocks[0].child_block_ids == std::vector<ANNS::IdxType>({2, 3}),
          "LNG layer topology must select minimal strict supersets");
   expect(blocks[4].child_block_ids == std::vector<ANNS::IdxType>({6}),
          "configuring one level must not mutate another level");

   blocks[0].child_block_ids = {3, 2, 2, 5};
   ANNS::apply_layer_topology(ANNS::GroupTopologyKind::Trie, 0, blocks);
   expect(blocks[0].child_block_ids == std::vector<ANNS::IdxType>({2, 3}),
          "Trie layer topology must retain only unique same-level trie edges");

   bool rejected_non_increasing = false;
   try
   {
      (void)ANNS::parse_hierarchy_plan(
          "1000:trie,1000:lng", ANNS::GroupTopologyKind::Lng);
   }
   catch (const std::invalid_argument &)
   {
      rejected_non_increasing = true;
   }
   expect(rejected_non_increasing,
          "hierarchy parser must reject non-increasing thresholds");

   std::cout << "UNG group topology checks passed\n";
   return 0;
}
