#ifndef ANNS_UNG_SPECIAL_BLOCKS_H
#define ANNS_UNG_SPECIAL_BLOCKS_H

#include "config.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace ANNS
{

struct SpecialBlock
{
   static constexpr IdxType kInvalidEntryPoint = std::numeric_limits<IdxType>::max();

   IdxType block_id = 0;
   IdxType root_group_id = 0;
   // Global point id of the entry used by the block-local graph.
   IdxType entry_point_id = kInvalidEntryPoint;
   IdxType point_count = 0;
   IdxType subtree_point_count = 0;
   std::vector<LabelType> root_labels;
   // Exact label intersection of this block's direct member groups. Child
   // blocks are separate coverage regions and are intentionally excluded.
   std::vector<LabelType> common_labels;
   std::vector<IdxType> member_group_ids;
   std::vector<IdxType> child_block_ids;

   bool is_trivial() const
   {
      return root_group_id > 0 && member_group_ids.size() == 1 &&
             member_group_ids.front() == root_group_id && child_block_ids.empty();
   }
};

inline std::vector<LabelType> compute_direct_member_common_labels(
    const std::vector<IdxType> &member_group_ids,
    const std::vector<std::vector<LabelType>> &group_labels)
{
   std::vector<LabelType> common;
   bool initialized = false;
   for (IdxType group_id : member_group_ids)
   {
      if (group_id >= group_labels.size())
         return {};
      std::vector<LabelType> labels = group_labels[group_id];
      std::sort(labels.begin(), labels.end());
      labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
      if (!initialized)
      {
         common = std::move(labels);
         initialized = true;
         continue;
      }
      std::vector<LabelType> intersection;
      intersection.reserve(std::min(common.size(), labels.size()));
      std::set_intersection(common.begin(), common.end(),
                            labels.begin(), labels.end(),
                            std::back_inserter(intersection));
      common.swap(intersection);
      if (common.empty())
         break;
   }
   return common;
}

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
   uint64_t trie_regular_group_edges = 0;
   uint64_t trie_regular_vector_edges = 0;
   uint64_t trie_regular_portal_edges = 0;
   IdxType group_graph_trivial_skipped_groups = 0;
   IdxType group_graph_trivial_skipped_points = 0;
   IdxType cross_trivial_skipped_pairs = 0;
   IdxType cross_trivial_skipped_query_vectors = 0;
   IdxType additional_trivial_skipped_groups = 0;
   IdxType additional_trivial_skipped_points = 0;
   uint64_t index_bytes = 0;
   uint64_t disk_bytes = 0;
   uint64_t memory_logical_bytes = 0;
   uint64_t memory_allocated_bytes = 0;
   uint64_t build_memory_logical_bytes = 0;
   uint64_t build_memory_allocated_bytes = 0;
   uint64_t trie_serialized_bytes = 0;
   uint64_t trie_node_count = 0;
   uint64_t trie_child_count = 0;
   double metadata_ms = 0.0;
   double trie_build_ms = 0.0;
   double trie_save_ms = 0.0;
   double trie_load_ms = 0.0;
   double edge_overlay_ms = 0.0;
   double block_indexes_prepare_ms = 0.0;
   double intra_edge_build_ms = 0.0;
   double inter_edge_build_ms = 0.0;
   double trie_regular_edge_build_ms = 0.0;
   double save_metadata_ms = 0.0;
   double save_members_ms = 0.0;
   double save_children_ms = 0.0;
   double save_edges_ms = 0.0;
   double save_trie_regular_edges_ms = 0.0;
   double save_total_ms = 0.0;
   double load_total_ms = 0.0;
   double compact_reload_ms = 0.0;
};

void save_special_block_metadata_binary(
    const std::string &path,
    const std::vector<SpecialBlock> &blocks);
bool load_special_block_metadata_binary(
    const std::string &path,
    std::vector<SpecialBlock> &blocks,
    std::string &error);

} // namespace ANNS

#endif // ANNS_UNG_SPECIAL_BLOCKS_H
