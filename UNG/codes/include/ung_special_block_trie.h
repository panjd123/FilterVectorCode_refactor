#ifndef ANNS_UNG_SPECIAL_BLOCK_TRIE_H
#define ANNS_UNG_SPECIAL_BLOCK_TRIE_H

#include "config.h"
#include <roaring/roaring.hh>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace ANNS
{

struct SpecialBlockTrieNode
{
   LabelType label = 0;
   uint32_t parent_id = UINT32_MAX;
   uint32_t first_child = 0;
   uint32_t child_count = 0;
   IdxType terminal_group_id = 0;
   IdxType block_id = 0;
};

static_assert(sizeof(SpecialBlockTrieNode) == 24,
              "special-block trie node format must stay 24 bytes");

struct SpecialBlockTrieSearchStats
{
   size_t pivot_postings = 0;
   size_t matching_pivots = 0;
   size_t upward_nodes_visited = 0;
   size_t downward_nodes_visited = 0;
   size_t branches_pruned = 0;
   size_t terminal_candidates = 0;
   size_t block_frontier_candidates = 0;
   size_t terminal_descendants_pruned = 0;
   size_t final_entries = 0;
   size_t final_block_entries = 0;
   double elapsed_ms = 0.0;
};

class SpecialBlockTrieIndex
{
public:
   using NodeId = uint32_t;
   static constexpr NodeId kInvalidNodeId = UINT32_MAX;
   static constexpr uint32_t kFormatVersion = 1;

   void clear();
   void build(const std::vector<std::vector<LabelType>> &group_labels,
              IdxType num_groups);

   bool empty() const { return nodes_.empty(); }
   size_t node_count() const { return nodes_.size(); }
   size_t child_count() const { return child_ids_.size(); }
   IdxType num_groups() const { return num_groups_; }
   uint64_t serialized_size_bytes() const;
   uint64_t logical_memory_size_bytes() const;
   uint64_t memory_size_bytes() const;

   const SpecialBlockTrieNode &node(NodeId node_id) const;
   NodeId child_node_id(NodeId node_id, size_t child_offset) const;
   NodeId find_exact_node(const std::vector<LabelType> &labels) const;
   std::vector<LabelType> labels_for_node(NodeId node_id) const;
   void set_block_id(NodeId node_id, IdxType block_id);
   std::vector<std::pair<IdxType, IdxType>> terminal_successor_pairs() const;
   // Map every terminal group inside a block to its nearest block ancestor.
   // The search uses this topology as a query-conditional portal: the block
   // entry is activated only when the query is covered by that block root.
   std::vector<std::pair<IdxType, IdxType>> terminal_block_portal_pairs() const;

   std::vector<IdxType> find_entry_groups(
       const std::vector<LabelType> &query_labels,
       SpecialBlockTrieSearchStats *stats = nullptr,
       std::vector<IdxType> *entry_block_ids = nullptr,
       bool include_block_frontier = false) const;

   // Fast path for queries whose label->group bitsets have already been
   // materialized.  The returned groups are the minimal candidate terminals
   // on the trie branches (i.e. no returned group has a candidate terminal
   // ancestor).  An empty query is the universal containment predicate and
   // returns the precomputed root terminal frontier.
   std::vector<IdxType> find_entry_groups_bitset(
       const std::vector<LabelType> &query_labels,
       SpecialBlockTrieSearchStats *stats = nullptr) const;

   // Apply the trie minimal-frontier operation to an already intersected
   // candidate bitmap.  This is useful when the caller owns a larger bitmap
   // cache and wants to avoid rebuilding the label intersection.
   std::vector<IdxType> select_entry_groups_from_candidates(
       const roaring::Roaring &candidate_groups,
       SpecialBlockTrieSearchStats *stats = nullptr) const;

   void save(const std::string &path) const;
   void load(const std::string &path,
             IdxType expected_num_groups,
             IdxType max_block_id);

private:
   void rebuild_label_postings();
   void rebuild_group_bitsets();
   void rebuild_terminal_ancestor_cache();
   void rebuild_block_descendant_flags();
   void validate(IdxType max_block_id) const;

   IdxType num_groups_ = 0;
   std::vector<SpecialBlockTrieNode> nodes_;
   std::vector<NodeId> child_ids_;
   std::vector<uint8_t> subtree_has_block_;
   std::unordered_map<LabelType, std::vector<NodeId>> label_postings_;
   std::unordered_map<LabelType, roaring::Roaring> label_group_bitsets_;
   std::vector<NodeId> group_terminal_nodes_;
   // CSR layout keeps each group's terminal-ancestor chain contiguous and
   // avoids one allocation/indirection per group on the query hot path.
   std::vector<uint64_t> terminal_ancestor_offsets_;
   std::vector<IdxType> terminal_ancestor_groups_;
   // Minimal terminal groups reachable from the trie root.  This makes the
   // universal empty predicate proportional to its entry frontier rather than
   // to the number of groups or trie nodes.
   std::vector<IdxType> root_entry_groups_;
};

} // namespace ANNS

#endif // ANNS_UNG_SPECIAL_BLOCK_TRIE_H
