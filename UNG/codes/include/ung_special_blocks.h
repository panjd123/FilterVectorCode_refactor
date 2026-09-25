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

inline bool is_supported_special_block_index_format(const std::string &format)
{
   return format == "special_block_trie_v1" ||
          format == "special_block_trie_v2" ||
          format == "special_block_trie_multilevel_v1";
}

struct SpecialBlock
{
   static constexpr IdxType kInvalidEntryPoint = std::numeric_limits<IdxType>::max();

   IdxType block_id = 0;
   // Zero is the finest materialized layer; larger values are progressively
   // coarser. Activation level is layer + 1 because zero denotes the ordinary
   // graph during search.
   uint8_t level = 0;
   // Optional containing block in the next coarser layer. This relation is
   // diagnostic/navigation metadata; child_block_ids remains the direct
   // parent->child relation inside one layer.
   IdxType parent_block_id = 0;
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

// Resolve the target's direct owner in the layer declared by the edge owner.
// Block ids are global across the middle/upper vectors, while point ownership
// is stored separately per layer.  Callers that classify inter edges (for
// example, light/heavy sidecar routing) must not assume the middle layer.
inline IdxType special_edge_target_owner(
    const SpecialEdge &edge,
    const std::vector<SpecialBlock> &blocks,
    const std::vector<std::vector<IdxType>> &point_to_block_by_level)
{
   if (edge.special_block_id == 0 || edge.special_block_id > blocks.size())
      return 0;
   const SpecialBlock &owner = blocks[edge.special_block_id - 1];
   if (owner.level >= point_to_block_by_level.size())
      return 0;
   const std::vector<IdxType> &point_owner = point_to_block_by_level[owner.level];
   if (edge.target_point_id >= point_owner.size())
      return 0;
   return point_owner[edge.target_point_id];
}

// Check that an edge's declared owner agrees with the per-layer partition.
// Intra edges stay among the owner's direct members. Inter edges originate in
// the owner and target a direct child block in the same partition layer. This
// is intentionally independent of vector distances and can therefore reject
// a structurally valid but semantically mis-tagged persisted sidecar.
inline bool validate_special_edge_semantics(
    IdxType source_point_id,
    const SpecialEdge &edge,
    const std::vector<SpecialBlock> &blocks,
    const std::vector<std::vector<IdxType>> &point_to_block_by_level,
    std::string &error)
{
   if (edge.special_block_id == 0 || edge.special_block_id > blocks.size())
   {
      error = "special edge source, target, or owner is out of range";
      return false;
   }

   const SpecialBlock &owner = blocks[edge.special_block_id - 1];
   if (owner.level >= point_to_block_by_level.size())
   {
      error = "special edge owner level has no ownership map";
      return false;
   }
   const std::vector<IdxType> &point_owner = point_to_block_by_level[owner.level];
   if (source_point_id >= point_owner.size() || edge.target_point_id >= point_owner.size())
   {
      error = "special edge source or target is out of range";
      return false;
   }
   if (point_owner[source_point_id] != owner.block_id)
   {
      error = "special edge source is not a direct member of its declared owner";
      return false;
   }

   const IdxType target_owner = special_edge_target_owner(
       edge, blocks, point_to_block_by_level);
   if (edge.kind == SpecialEdgeKind::IntraBlock)
   {
      if (target_owner != owner.block_id)
      {
         error = "intra special edge target is outside its declared owner";
         return false;
      }
   }
   else if (edge.kind == SpecialEdgeKind::InterBlock)
   {
      if (target_owner == 0 ||
          !std::binary_search(owner.child_block_ids.begin(),
                              owner.child_block_ids.end(), target_owner))
      {
         error = "inter special edge target is not in a direct child of its declared owner";
         return false;
      }
   }
   else
   {
      error = "special edge kind is invalid";
      return false;
   }

   error.clear();
   return true;
}

struct SpecialBlockBuildSummary
{
   std::vector<IdxType> layer_thresholds;
   std::vector<IdxType> layer_block_counts;
   IdxType threshold = 0;
   IdxType upper_threshold = 0;
   IdxType num_blocks = 0;
   IdxType upper_blocks = 0;
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
   double save_metadata_ms = 0.0;
   double save_members_ms = 0.0;
   double save_children_ms = 0.0;
   double save_edges_ms = 0.0;
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
// Validate the fixed middle/upper topology represented by the sidecar. This is
// intentionally independent of a loaded UniNavGraph so the same invariant is
// enforced before serialization and immediately after deserialization.
bool validate_special_block_metadata(
    const std::vector<SpecialBlock> &blocks,
    std::string &error);
// Validate source-graph relationships that are already defined immediately
// after partitioning.  The local intra graphs have not been built at this
// point, so entry_point_id is intentionally outside this preflight contract.
bool validate_special_block_partition_semantics(
    const std::vector<SpecialBlock> &blocks,
    IdxType num_points,
    IdxType num_groups,
    const std::vector<std::vector<LabelType>> &group_labels,
    const std::vector<std::pair<IdxType, IdxType>> &group_ranges,
    const std::vector<IdxType> &point_to_group,
    std::string &error);
// Validate invariants that can only be checked after the sidecar is attached
// to its source UNG graph.  Metadata I/O deliberately stays graph-independent;
// the full loader calls this second gate before rebuilding ownership indexes.
// Unlike partition preflight, this also requires a valid block-local graph
// entry point for every block.
bool validate_special_block_graph_semantics(
    const std::vector<SpecialBlock> &blocks,
    IdxType num_points,
    IdxType num_groups,
    const std::vector<std::vector<LabelType>> &group_labels,
    const std::vector<std::pair<IdxType, IdxType>> &group_ranges,
    const std::vector<IdxType> &point_to_group,
    std::string &error);

} // namespace ANNS

#endif // ANNS_UNG_SPECIAL_BLOCKS_H
