#ifndef ANNS_UNG_SPECIAL_BLOCKS_H
#define ANNS_UNG_SPECIAL_BLOCKS_H

#include "config.h"

#include <cstdint>
#include <vector>

namespace ANNS
{

struct SpecialBlock
{
   IdxType block_id = 0;
   IdxType root_group_id = 0;
   IdxType point_count = 0;
   IdxType subtree_point_count = 0;
   std::vector<LabelType> root_labels;
   std::vector<IdxType> member_group_ids;
   std::vector<IdxType> child_block_ids;

   bool is_trivial() const
   {
      return root_group_id > 0 && member_group_ids.size() == 1 &&
             member_group_ids.front() == root_group_id && child_block_ids.empty();
   }
};

enum class SpecialEdgeKind : uint8_t
{
   IntraBlock = 0,
   InterBlock = 1,
};

struct SpecialEdge
{
   IdxType target_point_id = 0;
   IdxType special_block_id = 0;
   SpecialEdgeKind kind = SpecialEdgeKind::IntraBlock;
};

struct SpecialBlockBuildSummary
{
   IdxType threshold = 0;
   IdxType num_blocks = 0;
   IdxType trivial_blocks = 0;
   IdxType member_groups = 0;
   IdxType member_points = 0;
   IdxType child_block_edges = 0;
   IdxType special_edges = 0;
   IdxType intra_special_edges = 0;
   IdxType inter_special_edges = 0;
   IdxType group_graph_trivial_skipped_groups = 0;
   IdxType group_graph_trivial_skipped_points = 0;
   IdxType cross_trivial_skipped_pairs = 0;
   IdxType cross_trivial_skipped_query_vectors = 0;
   IdxType additional_trivial_skipped_groups = 0;
   IdxType additional_trivial_skipped_points = 0;
   double metadata_ms = 0.0;
   double intra_edge_build_ms = 0.0;
   double inter_edge_build_ms = 0.0;
   double edge_overlay_ms = 0.0;
};

} // namespace ANNS

#endif // ANNS_UNG_SPECIAL_BLOCKS_H
