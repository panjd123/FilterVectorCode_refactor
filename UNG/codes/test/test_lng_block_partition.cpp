#include "ung_lng_block_partition.h"

#include <algorithm>
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

bool contains_group(const std::vector<ANNS::IdxType> &groups, ANNS::IdxType group_id)
{
   return std::find(groups.begin(), groups.end(), group_id) != groups.end();
}

bool contains_block(const std::vector<ANNS::IdxType> &blocks, ANNS::IdxType block_id)
{
   return std::find(blocks.begin(), blocks.end(), block_id) != blocks.end();
}
} // namespace

int main()
{
   {
   std::vector<std::vector<ANNS::IdxType>> out_neighbors(5);
   out_neighbors[1] = {2, 3};
   out_neighbors[2] = {4};
   out_neighbors[3] = {4};

   std::vector<ANNS::IdxType> group_points{0, 1, 1, 1, 3};
   std::vector<std::vector<ANNS::LabelType>> group_labels(5);
   group_labels[1] = {10};
   group_labels[2] = {10, 20};
   group_labels[3] = {10, 30};
   group_labels[4] = {10, 20, 30};

   ANNS::LngBlockPartitionInput input;
   input.out_neighbors = &out_neighbors;
   input.group_points = &group_points;
   input.group_labels = &group_labels;
   input.min_points = 2;
   input.tree_mode = ANNS::LngBlockTreeMode::Bfs;

   ANNS::LngBlockPartitionResult result = ANNS::build_lng_special_blocks(input);

   expect(result.blocks.size() == 2, "DFS postorder LNG partition must cut child and parent blocks");

   const ANNS::SpecialBlock &child = result.blocks[0];
   const ANNS::SpecialBlock &parent = result.blocks[1];

   expect(child.root_group_id == 4, "first postorder block must be rooted at the lower LNG group");
   expect(child.point_count == 3, "child block must contain the lower group points");
   expect(contains_group(child.member_group_ids, 4), "child block must own group 4");

   expect(parent.root_group_id == 1, "parent block must be rooted at the upper LNG group");
   expect(parent.point_count == 3, "parent block must not duplicate child block points");
   expect(contains_group(parent.member_group_ids, 1), "parent block must own group 1");
   expect(contains_group(parent.member_group_ids, 2), "parent block must own group 2");
   expect(contains_group(parent.member_group_ids, 3), "parent block must own group 3");
   expect(!contains_group(parent.member_group_ids, 4), "parent block must exclude groups owned by child blocks");

   expect(result.group_to_block.size() == 5, "group-to-block map must include group ids");
   expect(result.group_to_block[4] == child.block_id, "group 4 must map to the child block");
   expect(result.group_to_block[1] == parent.block_id, "group 1 must map to the parent block");
   expect(result.group_to_block[2] == parent.block_id, "group 2 must map to the parent block");
   expect(result.group_to_block[3] == parent.block_id, "group 3 must map to the parent block");

   expect(parent.child_block_ids.size() == 1, "parent block must keep one direct LNG block edge");
   expect(contains_block(parent.child_block_ids, child.block_id),
          "parent block must link to the child block through block-label LNG construction");
   }

   {
   std::vector<std::vector<ANNS::IdxType>> out_neighbors(4);
   out_neighbors[1] = {2};
   out_neighbors[2] = {3};

   std::vector<ANNS::IdxType> group_points{0, 3, 3, 3};
   std::vector<std::vector<ANNS::LabelType>> group_labels(4);
   group_labels[1] = {10};
   group_labels[2] = {10, 20, 30};
   group_labels[3] = {10, 20};

   ANNS::LngBlockPartitionInput input;
   input.out_neighbors = &out_neighbors;
   input.group_points = &group_points;
   input.group_labels = &group_labels;
   input.min_points = 2;
   input.tree_mode = ANNS::LngBlockTreeMode::Bfs;

   ANNS::LngBlockPartitionResult result = ANNS::build_lng_special_blocks(input);

   expect(result.blocks.size() == 3, "BFS tree partition must cut each large generated-tree node");

   const ANNS::SpecialBlock &root = result.blocks[2];
   expect(root.root_group_id == 1, "top generated-tree node must become the root block");
   expect(root.child_block_ids.size() == 1,
          "block-level LNG must choose only the minimal label superset block");
   expect(contains_block(root.child_block_ids, result.blocks[0].block_id),
          "root block must link to label [10,20], not directly to non-minimal [10,20,30]");
   expect(!contains_block(root.child_block_ids, result.blocks[1].block_id),
          "block-level LNG must prune non-minimal block supersets");
   }

   std::cout << "LNG block partition checks passed\n";
   return 0;
}
