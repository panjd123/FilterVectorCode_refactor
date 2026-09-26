#include "include/ung_special_block_trie.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace ANNS
{
namespace
{

constexpr std::array<char, 8> kMagic{{'S', 'B', 'T', 'R', 'I', 'E', '1', '\0'}};
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

struct SpecialBlockTrieFileHeader
{
   char magic[8];
   uint32_t version = 0;
   uint32_t header_bytes = 0;
   uint64_t node_count = 0;
   uint64_t child_count = 0;
   uint32_t num_groups = 0;
   uint32_t max_block_id = 0;
   uint64_t payload_checksum = 0;
};

static_assert(sizeof(SpecialBlockTrieFileHeader) == 48,
              "special-block trie file header must stay 48 bytes");

void checksum_bytes(uint64_t &hash, const void *data, size_t size)
{
   const auto *bytes = static_cast<const uint8_t *>(data);
   for (size_t i = 0; i < size; ++i)
   {
      hash ^= bytes[i];
      hash *= kFnvPrime;
   }
}

uint64_t payload_checksum(const std::vector<SpecialBlockTrieNode> &nodes,
                          const std::vector<SpecialBlockTrieIndex::NodeId> &children)
{
   uint64_t hash = kFnvOffset;
   if (!nodes.empty())
      checksum_bytes(hash, nodes.data(), nodes.size() * sizeof(nodes.front()));
   if (!children.empty())
      checksum_bytes(hash, children.data(), children.size() * sizeof(children.front()));
   return hash;
}

std::vector<LabelType> canonical_labels(const std::vector<LabelType> &labels)
{
   std::vector<LabelType> canonical = labels;
   std::sort(canonical.begin(), canonical.end());
   canonical.erase(std::unique(canonical.begin(), canonical.end()), canonical.end());
   return canonical;
}

uint64_t checked_payload_size(uint64_t node_count, uint64_t child_count)
{
   if (node_count > (std::numeric_limits<uint64_t>::max() -
                     sizeof(SpecialBlockTrieFileHeader)) /
                        sizeof(SpecialBlockTrieNode))
      throw std::runtime_error("special-block trie node count overflows file size");
   uint64_t size = sizeof(SpecialBlockTrieFileHeader) +
                   node_count * sizeof(SpecialBlockTrieNode);
   if (child_count > (std::numeric_limits<uint64_t>::max() - size) /
                         sizeof(SpecialBlockTrieIndex::NodeId))
      throw std::runtime_error("special-block trie child count overflows file size");
   return size + child_count * sizeof(SpecialBlockTrieIndex::NodeId);
}

} // namespace

void SpecialBlockTrieIndex::clear()
{
   num_groups_ = 0;
   nodes_.clear();
   child_ids_.clear();
   subtree_has_block_.clear();
   label_postings_.clear();
   label_group_bitsets_.clear();
   group_terminal_nodes_.clear();
   terminal_ancestor_offsets_.clear();
   terminal_ancestor_groups_.clear();
   root_entry_groups_.clear();
}

void SpecialBlockTrieIndex::build(
    const std::vector<std::vector<LabelType>> &group_labels,
    IdxType num_groups)
{
   clear();
   if (group_labels.size() <= static_cast<size_t>(num_groups))
      throw std::invalid_argument("special-block trie requires labels for every group");

   num_groups_ = num_groups;
   nodes_.push_back(SpecialBlockTrieNode{});
   std::vector<std::unordered_map<LabelType, NodeId>> child_maps(1);

   for (IdxType group_id = 1; group_id <= num_groups_; ++group_id)
   {
      const std::vector<LabelType> labels = canonical_labels(group_labels[group_id]);
      if (labels.empty())
         throw std::invalid_argument("special-block trie does not support an empty terminal label set");

      NodeId current = 0;
      for (LabelType label : labels)
      {
         auto child_it = child_maps[current].find(label);
         if (child_it == child_maps[current].end())
         {
            if (nodes_.size() >= static_cast<size_t>(kInvalidNodeId))
               throw std::overflow_error("special-block trie has too many nodes");
            const NodeId next = static_cast<NodeId>(nodes_.size());
            SpecialBlockTrieNode node;
            node.label = label;
            node.parent_id = current;
            nodes_.push_back(node);
            child_maps.emplace_back();
            child_maps[current].emplace(label, next);
            current = next;
         }
         else
         {
            current = child_it->second;
         }
      }
      if (nodes_[current].terminal_group_id != 0)
         throw std::invalid_argument("special-block trie received duplicate group label sets");
      nodes_[current].terminal_group_id = group_id;
   }

   for (NodeId node_id = 0; node_id < nodes_.size(); ++node_id)
   {
      std::vector<std::pair<LabelType, NodeId>> sorted_children(
          child_maps[node_id].begin(), child_maps[node_id].end());
      std::sort(sorted_children.begin(), sorted_children.end(),
                [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
      nodes_[node_id].first_child = static_cast<uint32_t>(child_ids_.size());
      nodes_[node_id].child_count = static_cast<uint32_t>(sorted_children.size());
      for (const auto &child : sorted_children)
         child_ids_.push_back(child.second);
   }

   validate(0);
   subtree_has_block_.assign(nodes_.size(), 0);
   rebuild_label_postings();
   rebuild_group_bitsets();
   rebuild_terminal_ancestor_cache();
}

uint64_t SpecialBlockTrieIndex::serialized_size_bytes() const
{
   if (nodes_.empty())
      return 0;
   return checked_payload_size(nodes_.size(), child_ids_.size());
}

uint64_t SpecialBlockTrieIndex::memory_size_bytes() const
{
   uint64_t bytes = sizeof(SpecialBlockTrieIndex) +
                    nodes_.capacity() * sizeof(SpecialBlockTrieNode) +
                    child_ids_.capacity() * sizeof(NodeId) +
                    subtree_has_block_.capacity() * sizeof(uint8_t) +
                    label_postings_.bucket_count() * sizeof(void *);
   for (const auto &posting : label_postings_)
      bytes += sizeof(posting) + posting.second.capacity() * sizeof(NodeId);
   bytes += group_terminal_nodes_.capacity() * sizeof(NodeId);
   bytes += terminal_ancestor_offsets_.capacity() * sizeof(uint64_t);
   bytes += terminal_ancestor_groups_.capacity() * sizeof(IdxType);
   bytes += root_entry_groups_.capacity() * sizeof(IdxType);
   bytes += label_group_bitsets_.bucket_count() * sizeof(void *);
   for (const auto &posting : label_group_bitsets_)
      bytes += sizeof(posting) + posting.second.getSizeInBytes();
   return bytes;
}

uint64_t SpecialBlockTrieIndex::logical_memory_size_bytes() const
{
   uint64_t bytes = sizeof(SpecialBlockTrieIndex) +
                    nodes_.size() * sizeof(SpecialBlockTrieNode) +
                    child_ids_.size() * sizeof(NodeId) +
                    subtree_has_block_.size() * sizeof(uint8_t) +
                    label_postings_.size() * sizeof(decltype(label_postings_)::value_type);
   for (const auto &posting : label_postings_)
      bytes += posting.second.size() * sizeof(NodeId);
   bytes += group_terminal_nodes_.size() * sizeof(NodeId);
   bytes += terminal_ancestor_offsets_.size() * sizeof(uint64_t);
   bytes += terminal_ancestor_groups_.size() * sizeof(IdxType);
   bytes += root_entry_groups_.size() * sizeof(IdxType);
   bytes += label_group_bitsets_.size() * sizeof(decltype(label_group_bitsets_)::value_type);
   for (const auto &posting : label_group_bitsets_)
      bytes += posting.second.getSizeInBytes();
   return bytes;
}

const SpecialBlockTrieNode &SpecialBlockTrieIndex::node(NodeId node_id) const
{
   if (node_id >= nodes_.size())
      throw std::out_of_range("special-block trie node id is out of range");
   return nodes_[node_id];
}

SpecialBlockTrieIndex::NodeId SpecialBlockTrieIndex::child_node_id(
    NodeId node_id, size_t child_offset) const
{
   const SpecialBlockTrieNode &parent = node(node_id);
   if (child_offset >= parent.child_count)
      throw std::out_of_range("special-block trie child offset is out of range");
   return child_ids_[static_cast<size_t>(parent.first_child) + child_offset];
}

SpecialBlockTrieIndex::NodeId SpecialBlockTrieIndex::find_exact_node(
    const std::vector<LabelType> &labels) const
{
   if (nodes_.empty())
      return kInvalidNodeId;
   const std::vector<LabelType> canonical = canonical_labels(labels);
   NodeId current = 0;
   for (LabelType label : canonical)
   {
      const SpecialBlockTrieNode &parent = nodes_[current];
      const auto begin = child_ids_.begin() + parent.first_child;
      const auto end = begin + parent.child_count;
      const auto found = std::lower_bound(
          begin, end, label,
          [this](NodeId child_id, LabelType target) {
             return nodes_[child_id].label < target;
          });
      if (found == end || nodes_[*found].label != label)
         return kInvalidNodeId;
      current = *found;
   }
   return current;
}

std::vector<LabelType> SpecialBlockTrieIndex::labels_for_node(NodeId node_id) const
{
   node(node_id);
   std::vector<LabelType> labels;
   while (node_id != 0)
   {
      labels.push_back(nodes_[node_id].label);
      node_id = nodes_[node_id].parent_id;
   }
   std::reverse(labels.begin(), labels.end());
   return labels;
}

void SpecialBlockTrieIndex::set_block_id(NodeId node_id, IdxType block_id)
{
   if (node_id == 0 || node_id >= nodes_.size())
      throw std::out_of_range("special-block trie block root node is invalid");
   nodes_[node_id].block_id = block_id;
   if (subtree_has_block_.size() != nodes_.size())
      subtree_has_block_.assign(nodes_.size(), 0);
   NodeId current = node_id;
   while (true)
   {
      if (subtree_has_block_[current] != 0)
         break;
      subtree_has_block_[current] = 1;
      if (current == 0)
         break;
      current = nodes_[current].parent_id;
   }
}

std::vector<std::pair<IdxType, IdxType>>
SpecialBlockTrieIndex::terminal_successor_pairs() const
{
   std::vector<std::pair<IdxType, IdxType>> pairs;
   if (nodes_.empty())
      return pairs;

   std::vector<IdxType> nearest_terminal(nodes_.size(), 0);
   pairs.reserve(num_groups_);
   for (NodeId node_id = 1; node_id < nodes_.size(); ++node_id)
   {
      const SpecialBlockTrieNode &trie_node = nodes_[node_id];
      const IdxType parent_terminal = nearest_terminal[trie_node.parent_id];
      if (trie_node.terminal_group_id == 0)
      {
         nearest_terminal[node_id] = parent_terminal;
         continue;
      }

      if (parent_terminal != 0)
         pairs.emplace_back(parent_terminal, trie_node.terminal_group_id);
      nearest_terminal[node_id] = trie_node.terminal_group_id;
   }
   return pairs;
}

std::vector<std::pair<IdxType, IdxType>>
SpecialBlockTrieIndex::terminal_block_portal_pairs() const
{
   std::vector<std::pair<IdxType, IdxType>> pairs;
   if (nodes_.empty())
      return pairs;

   struct Pending
   {
      NodeId node_id = 0;
      IdxType nearest_block = 0;
   };
   std::vector<Pending> stack;
   stack.reserve(nodes_.size());
   stack.push_back(Pending{0, 0});
   while (!stack.empty())
   {
      const Pending pending = stack.back();
      stack.pop_back();
      const SpecialBlockTrieNode &trie_node = nodes_[pending.node_id];
      const IdxType nearest_block = trie_node.block_id != 0
                                        ? trie_node.block_id
                                        : pending.nearest_block;
      if (trie_node.terminal_group_id != 0 && nearest_block != 0)
         pairs.emplace_back(trie_node.terminal_group_id, nearest_block);

      for (size_t offset = 0; offset < trie_node.child_count; ++offset)
         stack.push_back(Pending{child_ids_[static_cast<size_t>(trie_node.first_child) + offset],
                                 nearest_block});
   }
   std::sort(pairs.begin(), pairs.end());
   pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
   return pairs;
}

std::vector<IdxType> SpecialBlockTrieIndex::find_entry_groups(
    const std::vector<LabelType> &query_labels,
    SpecialBlockTrieSearchStats *stats,
    std::vector<IdxType> *entry_block_ids,
    bool include_block_frontier) const
{
   const auto start = std::chrono::high_resolution_clock::now();
   SpecialBlockTrieSearchStats local_stats;
   std::vector<IdxType> entries;
   if (entry_block_ids != nullptr)
      entry_block_ids->clear();
   const std::vector<LabelType> &query = query_labels;
   if (nodes_.empty())
   {
      local_stats.elapsed_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::high_resolution_clock::now() - start)
                                   .count();
      if (stats != nullptr)
         *stats = local_stats;
      return entries;
   }
   if (query.empty())
   {
      entries = root_entry_groups_;
      local_stats.pivot_postings = num_groups_;
      local_stats.matching_pivots = num_groups_;
      local_stats.terminal_candidates = entries.size();

      // Block-aware callers also need the first block below each root entry.
      // The production Trie entry provider does not request this path.
      if (include_block_frontier && entry_block_ids != nullptr)
      {
         struct PendingNode
         {
            NodeId node_id = 0;
            bool terminal_seen = false;
         };
         std::vector<PendingNode> queue;
         queue.reserve(nodes_[0].child_count);
         for (size_t offset = 0; offset < nodes_[0].child_count; ++offset)
            queue.push_back(PendingNode{
                child_ids_[static_cast<size_t>(nodes_[0].first_child) + offset], false});

         size_t head = 0;
         while (head < queue.size())
         {
            const PendingNode pending = queue[head++];
            const SpecialBlockTrieNode &trie_node = nodes_[pending.node_id];
            ++local_stats.downward_nodes_visited;
            if (pending.terminal_seen && trie_node.block_id != 0)
            {
               ++local_stats.block_frontier_candidates;
               entry_block_ids->push_back(trie_node.block_id);
               continue;
            }

            const bool terminal_seen = pending.terminal_seen ||
                                       trie_node.terminal_group_id != 0;
            for (size_t child_offset = 0; child_offset < trie_node.child_count;
                 ++child_offset)
            {
               const NodeId child =
                   child_ids_[static_cast<size_t>(trie_node.first_child) + child_offset];
               if (terminal_seen &&
                   (subtree_has_block_.size() != nodes_.size() ||
                    subtree_has_block_[child] == 0))
               {
                  ++local_stats.terminal_descendants_pruned;
                  continue;
               }
               queue.push_back(PendingNode{child, terminal_seen});
            }
         }
      }

      local_stats.final_entries = entries.size();
      local_stats.final_block_entries =
          entry_block_ids == nullptr ? 0 : entry_block_ids->size();
      local_stats.elapsed_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::high_resolution_clock::now() - start)
                                   .count();
      if (stats != nullptr)
         *stats = local_stats;
      return entries;
   }
#ifndef NDEBUG
   if (!std::is_sorted(query.begin(), query.end()) ||
       std::adjacent_find(query.begin(), query.end()) != query.end())
      throw std::invalid_argument(
          "special-block trie query labels must be strictly increasing");
#endif

   const auto posting_it = label_postings_.find(query.back());
   if (posting_it == label_postings_.end())
   {
      local_stats.elapsed_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::high_resolution_clock::now() - start)
                                   .count();
      if (stats != nullptr)
         *stats = local_stats;
      return entries;
   }
   local_stats.pivot_postings = posting_it->second.size();

   // Strictly increasing paths contain the pivot label at most once. Nodes in
   // one pivot posting are therefore incomparable and their subtrees cannot
   // overlap, so a visited array proportional to the full index is unnecessary.
   struct PendingNode
   {
      NodeId node_id = 0;
      bool terminal_seen = false;
   };
   thread_local std::vector<PendingNode> queue;
   queue.clear();
   if (queue.capacity() < posting_it->second.size())
      queue.reserve(posting_it->second.size());
   for (NodeId pivot_node : posting_it->second)
   {
      NodeId current = nodes_[pivot_node].parent_id;
      bool matches = true;
      for (size_t reverse_idx = query.size() - 1; reverse_idx > 0; --reverse_idx)
      {
         const LabelType required = query[reverse_idx - 1];
         while (current != 0 && nodes_[current].label > required)
         {
            ++local_stats.upward_nodes_visited;
            current = nodes_[current].parent_id;
         }
         if (current == 0 || nodes_[current].label != required)
         {
            matches = false;
            break;
         }
         ++local_stats.upward_nodes_visited;
         current = nodes_[current].parent_id;
      }
      if (!matches)
      {
         ++local_stats.branches_pruned;
         continue;
      }
      ++local_stats.matching_pivots;
      queue.push_back(PendingNode{pivot_node, false});
   }

   size_t head = 0;
   while (head < queue.size())
   {
      const PendingNode pending = queue[head++];
      const NodeId current = pending.node_id;
      ++local_stats.downward_nodes_visited;
      const SpecialBlockTrieNode &trie_node = nodes_[current];
      if (include_block_frontier && pending.terminal_seen && trie_node.block_id != 0)
      {
         ++local_stats.block_frontier_candidates;
         if (entry_block_ids != nullptr)
            entry_block_ids->push_back(trie_node.block_id);
         continue;
      }

      bool terminal_seen = pending.terminal_seen;
      if (trie_node.terminal_group_id != 0 && !terminal_seen)
      {
         ++local_stats.terminal_candidates;
         entries.push_back(trie_node.terminal_group_id);
         terminal_seen = true;
         if (!include_block_frontier)
            continue;
      }

      for (size_t child_offset = 0; child_offset < trie_node.child_count; ++child_offset)
      {
         const NodeId child = child_ids_[static_cast<size_t>(trie_node.first_child) + child_offset];
         if (terminal_seen &&
             (subtree_has_block_.size() != nodes_.size() || subtree_has_block_[child] == 0))
         {
            ++local_stats.terminal_descendants_pruned;
            continue;
         }
         queue.push_back(PendingNode{child, terminal_seen});
      }
   }

   local_stats.final_entries = entries.size();
   local_stats.final_block_entries =
       entry_block_ids == nullptr ? 0 : entry_block_ids->size();
   local_stats.elapsed_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - start)
                                .count();
   if (stats != nullptr)
      *stats = local_stats;
   return entries;
}

std::vector<IdxType> SpecialBlockTrieIndex::select_entry_groups_from_candidates(
    const roaring::Roaring &candidate_groups,
    SpecialBlockTrieSearchStats *stats) const
{
   const auto start = std::chrono::high_resolution_clock::now();
   SpecialBlockTrieSearchStats local;
   std::vector<IdxType> entries;
   if (nodes_.empty() || candidate_groups.isEmpty())
   {
      local.elapsed_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::high_resolution_clock::now() - start)
                             .count();
      if (stats != nullptr)
         *stats = local;
      return entries;
   }

   entries.reserve(candidate_groups.cardinality());
   local.matching_pivots = candidate_groups.cardinality();
   for (auto it = candidate_groups.begin(); it != candidate_groups.end(); ++it)
   {
      const IdxType gid = static_cast<IdxType>(*it);
      if (gid == 0 || gid > num_groups_ || gid + 1 >= terminal_ancestor_offsets_.size())
         continue;

      bool covered = false;
      const uint64_t begin = terminal_ancestor_offsets_[gid];
      const uint64_t end = terminal_ancestor_offsets_[gid + 1];
      for (uint64_t offset = begin; offset < end; ++offset)
      {
         ++local.upward_nodes_visited;
         if (candidate_groups.contains(terminal_ancestor_groups_[offset]))
         {
            covered = true;
            break;
         }
      }
      if (!covered)
      {
         entries.push_back(gid);
         ++local.terminal_candidates;
      }
   }
   local.final_entries = entries.size();
   local.elapsed_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - start)
                          .count();
   if (stats != nullptr)
      *stats = local;
   return entries;
}

std::vector<IdxType> SpecialBlockTrieIndex::find_entry_groups_bitset(
    const std::vector<LabelType> &query_labels,
    SpecialBlockTrieSearchStats *stats) const
{
   const auto start = std::chrono::high_resolution_clock::now();
   SpecialBlockTrieSearchStats local;
   if (nodes_.empty())
   {
      local.elapsed_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::high_resolution_clock::now() - start)
                             .count();
      if (stats != nullptr)
         *stats = local;
      return {};
   }
   if (query_labels.empty())
   {
      local.pivot_postings = num_groups_;
      local.matching_pivots = num_groups_;
      local.terminal_candidates = root_entry_groups_.size();
      local.final_entries = root_entry_groups_.size();
      local.elapsed_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::high_resolution_clock::now() - start)
                             .count();
      if (stats != nullptr)
         *stats = local;
      return root_entry_groups_;
   }

   // Intersect the smallest postings first.  This minimizes roaring work and
   // usually makes the subsequent ancestor checks proportional to the answer
   // size rather than to the total number of groups.
   const roaring::Roaring *seed = nullptr;
   for (LabelType label : query_labels)
   {
      const auto it = label_group_bitsets_.find(label);
      if (it == label_group_bitsets_.end())
      {
         local.elapsed_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - start)
                                .count();
         if (stats != nullptr)
            *stats = local;
         return {};
      }
      if (seed == nullptr || it->second.cardinality() < seed->cardinality())
         seed = &it->second;
   }

   roaring::Roaring candidates = *seed;
   local.pivot_postings = seed->cardinality();
   for (LabelType label : query_labels)
   {
      const auto it = label_group_bitsets_.find(label);
      if (&it->second != seed)
         candidates &= it->second;
      if (candidates.isEmpty())
         break;
   }
   local.matching_pivots = candidates.cardinality();
   SpecialBlockTrieSearchStats select_stats;
   std::vector<IdxType> result = select_entry_groups_from_candidates(candidates, &select_stats);
   local.upward_nodes_visited = select_stats.upward_nodes_visited;
   local.terminal_candidates = select_stats.terminal_candidates;
   local.final_entries = result.size();
   local.elapsed_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - start)
                          .count();
   if (stats != nullptr)
      *stats = local;
   return result;
}

void SpecialBlockTrieIndex::save(const std::string &path) const
{
   if (nodes_.empty())
      throw std::runtime_error("cannot save an empty special-block trie index");
   validate(std::numeric_limits<IdxType>::max());

   SpecialBlockTrieFileHeader header;
   std::memcpy(header.magic, kMagic.data(), kMagic.size());
   header.version = kFormatVersion;
   header.header_bytes = sizeof(header);
   header.node_count = nodes_.size();
   header.child_count = child_ids_.size();
   header.num_groups = num_groups_;
   for (const SpecialBlockTrieNode &trie_node : nodes_)
      header.max_block_id = std::max(header.max_block_id, trie_node.block_id);
   header.payload_checksum = payload_checksum(nodes_, child_ids_);

   const std::string temporary_path = path + ".tmp";
   std::ofstream out(temporary_path, std::ios::binary | std::ios::trunc);
   if (!out)
      throw std::runtime_error("cannot open special-block trie temporary output: " + temporary_path);
   out.write(reinterpret_cast<const char *>(&header), sizeof(header));
   out.write(reinterpret_cast<const char *>(nodes_.data()),
             static_cast<std::streamsize>(nodes_.size() * sizeof(nodes_.front())));
   if (!child_ids_.empty())
      out.write(reinterpret_cast<const char *>(child_ids_.data()),
                static_cast<std::streamsize>(child_ids_.size() * sizeof(child_ids_.front())));
   out.flush();
   if (!out)
   {
      out.close();
      std::remove(temporary_path.c_str());
      throw std::runtime_error("failed while writing special-block trie index");
   }
   out.close();
   if (std::rename(temporary_path.c_str(), path.c_str()) != 0)
   {
      std::remove(temporary_path.c_str());
      throw std::runtime_error("cannot install validated special-block trie index: " + path);
   }
}

void SpecialBlockTrieIndex::load(const std::string &path,
                                 IdxType expected_num_groups,
                                 IdxType max_block_id)
{
   clear();
   std::ifstream in(path, std::ios::binary | std::ios::ate);
   if (!in)
      throw std::runtime_error("missing special-block trie index: " + path);
   const std::streamoff file_size = in.tellg();
   if (file_size < static_cast<std::streamoff>(sizeof(SpecialBlockTrieFileHeader)))
      throw std::runtime_error("special-block trie index is shorter than its header");
   in.seekg(0);

   SpecialBlockTrieFileHeader header;
   in.read(reinterpret_cast<char *>(&header), sizeof(header));
   if (!in || std::memcmp(header.magic, kMagic.data(), kMagic.size()) != 0)
      throw std::runtime_error("special-block trie index has invalid magic");
   if (header.version != kFormatVersion || header.header_bytes != sizeof(header))
      throw std::runtime_error("special-block trie index format version is not supported");
   if (header.node_count == 0 || header.node_count >= kInvalidNodeId)
      throw std::runtime_error("special-block trie index has invalid node count");
   if (header.child_count >= kInvalidNodeId)
      throw std::runtime_error("special-block trie index has invalid child count");
   if (header.num_groups != expected_num_groups)
      throw std::runtime_error("special-block trie group count does not match index metadata");
   if (header.max_block_id > max_block_id)
      throw std::runtime_error("special-block trie references an unknown special block");
   const uint64_t expected_size = checked_payload_size(header.node_count, header.child_count);
   if (static_cast<uint64_t>(file_size) != expected_size)
      throw std::runtime_error("special-block trie file size does not match its header");

   nodes_.resize(static_cast<size_t>(header.node_count));
   child_ids_.resize(static_cast<size_t>(header.child_count));
   in.read(reinterpret_cast<char *>(nodes_.data()),
           static_cast<std::streamsize>(nodes_.size() * sizeof(nodes_.front())));
   if (!child_ids_.empty())
      in.read(reinterpret_cast<char *>(child_ids_.data()),
              static_cast<std::streamsize>(child_ids_.size() * sizeof(child_ids_.front())));
   if (!in)
      throw std::runtime_error("special-block trie payload read failed");
   if (payload_checksum(nodes_, child_ids_) != header.payload_checksum)
      throw std::runtime_error("special-block trie checksum mismatch");

   num_groups_ = header.num_groups;
   validate(max_block_id);
   rebuild_label_postings();
   rebuild_group_bitsets();
   rebuild_terminal_ancestor_cache();
   rebuild_block_descendant_flags();
}

void SpecialBlockTrieIndex::rebuild_label_postings()
{
   label_postings_.clear();
   for (NodeId node_id = 1; node_id < nodes_.size(); ++node_id)
      label_postings_[nodes_[node_id].label].push_back(node_id);
}

void SpecialBlockTrieIndex::rebuild_group_bitsets()
{
   label_group_bitsets_.clear();
   group_terminal_nodes_.assign(static_cast<size_t>(num_groups_) + 1, kInvalidNodeId);
   for (NodeId node_id = 1; node_id < nodes_.size(); ++node_id)
   {
      const IdxType gid = nodes_[node_id].terminal_group_id;
      if (gid == 0)
         continue;
      group_terminal_nodes_[gid] = node_id;
      NodeId current = node_id;
      while (current != 0)
      {
         label_group_bitsets_[nodes_[current].label].add(gid);
         current = nodes_[current].parent_id;
      }
   }
}

void SpecialBlockTrieIndex::rebuild_terminal_ancestor_cache()
{
   terminal_ancestor_offsets_.assign(static_cast<size_t>(num_groups_) + 2, 0);
   terminal_ancestor_groups_.clear();
   terminal_ancestor_groups_.reserve(num_groups_);
   root_entry_groups_.clear();
   root_entry_groups_.reserve(num_groups_);
   for (IdxType gid = 1; gid <= num_groups_; ++gid)
   {
      terminal_ancestor_offsets_[gid] = terminal_ancestor_groups_.size();
      if (gid >= group_terminal_nodes_.size() ||
          group_terminal_nodes_[gid] == kInvalidNodeId)
         continue;
      NodeId current = nodes_[group_terminal_nodes_[gid]].parent_id;
      bool has_terminal_ancestor = false;
      while (current != 0)
      {
         const IdxType ancestor_gid = nodes_[current].terminal_group_id;
         if (ancestor_gid != 0)
         {
            terminal_ancestor_groups_.push_back(ancestor_gid);
            has_terminal_ancestor = true;
         }
         current = nodes_[current].parent_id;
      }
      if (!has_terminal_ancestor)
         root_entry_groups_.push_back(gid);
   }
   terminal_ancestor_offsets_[num_groups_ + 1] = terminal_ancestor_groups_.size();
}

void SpecialBlockTrieIndex::rebuild_block_descendant_flags()
{
   subtree_has_block_.assign(nodes_.size(), 0);
   for (size_t reverse_idx = nodes_.size(); reverse_idx > 0; --reverse_idx)
   {
      const NodeId node_id = static_cast<NodeId>(reverse_idx - 1);
      uint8_t has_block = nodes_[node_id].block_id != 0 ? 1 : 0;
      const SpecialBlockTrieNode &trie_node = nodes_[node_id];
      for (size_t child_offset = 0; child_offset < trie_node.child_count && has_block == 0;
           ++child_offset)
      {
         const NodeId child =
             child_ids_[static_cast<size_t>(trie_node.first_child) + child_offset];
         has_block = subtree_has_block_[child];
      }
      subtree_has_block_[node_id] = has_block;
   }
}

void SpecialBlockTrieIndex::validate(IdxType max_block_id) const
{
   if (nodes_.empty())
      throw std::runtime_error("special-block trie index is empty");
   if (nodes_[0].parent_id != kInvalidNodeId || nodes_[0].label != 0 ||
       nodes_[0].terminal_group_id != 0 || nodes_[0].block_id != 0)
      throw std::runtime_error("special-block trie root is invalid");

   std::vector<uint8_t> child_seen(nodes_.size(), 0);
   std::vector<uint8_t> group_seen(static_cast<size_t>(num_groups_) + 1, 0);
   size_t terminal_count = 0;
   for (NodeId node_id = 0; node_id < nodes_.size(); ++node_id)
   {
      const SpecialBlockTrieNode &trie_node = nodes_[node_id];
      if (node_id > 0 && (trie_node.parent_id >= node_id || trie_node.parent_id >= nodes_.size()))
         throw std::runtime_error("special-block trie parent reference is invalid");
      if (trie_node.first_child > child_ids_.size() ||
          trie_node.child_count > child_ids_.size() - trie_node.first_child)
         throw std::runtime_error("special-block trie child span is invalid");
      LabelType previous_label = 0;
      bool have_previous = false;
      for (size_t child_offset = 0; child_offset < trie_node.child_count; ++child_offset)
      {
         const NodeId child = child_ids_[static_cast<size_t>(trie_node.first_child) + child_offset];
         if (child == 0 || child >= nodes_.size() || nodes_[child].parent_id != node_id)
            throw std::runtime_error("special-block trie child reference is invalid");
         if (child_seen[child] != 0)
            throw std::runtime_error("special-block trie node has multiple parents");
         child_seen[child] = 1;
         if (have_previous && nodes_[child].label <= previous_label)
            throw std::runtime_error("special-block trie children are not strictly ordered");
         if (node_id != 0 && nodes_[child].label <= trie_node.label)
            throw std::runtime_error("special-block trie path labels are not strictly ordered");
         previous_label = nodes_[child].label;
         have_previous = true;
      }
      if (trie_node.terminal_group_id != 0)
      {
         if (trie_node.terminal_group_id > num_groups_ ||
             group_seen[trie_node.terminal_group_id] != 0)
            throw std::runtime_error("special-block trie terminal group is invalid");
         group_seen[trie_node.terminal_group_id] = 1;
         ++terminal_count;
      }
      if (trie_node.block_id > max_block_id)
         throw std::runtime_error("special-block trie block id is invalid");
   }
   for (NodeId node_id = 1; node_id < nodes_.size(); ++node_id)
      if (child_seen[node_id] == 0)
         throw std::runtime_error("special-block trie contains an unreachable node");
   if (terminal_count != num_groups_)
      throw std::runtime_error("special-block trie does not contain every group terminal");
}

} // namespace ANNS
