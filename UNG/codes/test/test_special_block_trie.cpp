#include "ung_entry_group.h"
#include "ung_special_block_trie.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
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

std::vector<ANNS::IdxType> sorted(std::vector<ANNS::IdxType> values)
{
   std::sort(values.begin(), values.end());
   return values;
}

bool load_throws(const std::filesystem::path &path,
                 ANNS::IdxType num_groups,
                 ANNS::IdxType max_block_id)
{
   try
   {
      ANNS::SpecialBlockTrieIndex loaded;
      loaded.load(path.string(), num_groups, max_block_id);
   }
   catch (const std::runtime_error &)
   {
      return true;
   }
   return false;
}
} // namespace

int main()
{
   const std::vector<std::vector<ANNS::LabelType>> group_labels{
       {},
       {1, 3},
       {1, 3, 4},
       {1, 2, 3},
       {2, 3},
       {5},
   };

   ANNS::SpecialBlockTrieIndex index;
   index.build(group_labels, 5);
   expect(index.node_count() == 9, "fixture must build the expected number of flat trie nodes");
   expect(index.child_count() + 1 == index.node_count(), "a trie must have one child edge per non-root node");

   ANNS::SpecialBlockTrieSearchStats stats;
   expect(sorted(index.find_entry_groups({1}, &stats)) ==
              std::vector<ANNS::IdxType>({1, 3}),
          "query must keep first terminals on distinct branches without global filtering");
   expect(stats.pivot_postings == 1 && stats.matching_pivots == 1,
          "single-label pivot posting statistics must be exact");
   expect(stats.final_entries == 2, "final entry count must match emitted terminals");
   expect(index.terminal_successor_pairs() ==
              std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>>({{1, 2}}),
          "terminal navigation must connect a group only to its nearest terminal descendants");

   expect(sorted(index.find_entry_groups({1, 3}, &stats)) ==
              std::vector<ANNS::IdxType>({1, 3}),
          "upward matching must allow extra path labels and prune terminal descendants");
   expect(stats.pivot_postings == 3 && stats.matching_pivots == 2 && stats.branches_pruned == 1,
          "multi-label upward containment statistics must be exact");
   expect(sorted(index.find_entry_groups_bitset({1, 3}, &stats)) ==
              std::vector<ANNS::IdxType>({1, 3}),
          "roaring candidate intersection must keep one entry per matching trie branch");
   expect(sorted(index.find_entry_groups_bitset({1}, &stats)) ==
              std::vector<ANNS::IdxType>({1, 3}),
          "roaring candidate intersection must remove candidate terminal descendants");
   expect(index.find_entry_groups_bitset({99}, nullptr).empty(),
          "roaring candidate intersection must reject unknown labels");
   expect(sorted(index.find_entry_groups({3}, nullptr)) ==
              std::vector<ANNS::IdxType>({1, 3, 4}),
          "every matching pivot branch must emit its first terminal");
   expect(index.find_entry_groups({}, nullptr).empty(), "empty query must not scan the whole trie");
   expect(index.find_entry_groups({99}, nullptr).empty(), "unknown pivot label must return no entries");

   const std::vector<std::vector<ANNS::LabelType>> terminal_at_query_labels{
       {},
       {1},
       {1, 2},
       {1, 3},
   };
   ANNS::SpecialBlockTrieIndex terminal_at_query;
   terminal_at_query.build(terminal_at_query_labels, 3);
   const auto descendant_block = terminal_at_query.find_exact_node({1, 2});
   expect(descendant_block != ANNS::SpecialBlockTrieIndex::kInvalidNodeId,
          "terminal fixture must contain the descendant block node");
   terminal_at_query.set_block_id(descendant_block, 1);
   expect(terminal_at_query.terminal_block_portal_pairs() ==
              std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>>({{2, 1}}),
          "a terminal must map to its nearest block ancestor");
   expect(terminal_at_query.terminal_successor_pairs() ==
              std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>>({{1, 2}, {1, 3}}),
          "a query terminal must connect to the first terminal on every child branch");
   expect(terminal_at_query.find_entry_groups({1}, &stats) ==
              std::vector<ANNS::IdxType>({1}),
          "a terminal at the query node must stop the default downward search");
   std::vector<ANNS::IdxType> terminal_descendant_blocks;
   expect(terminal_at_query.find_entry_groups(
              {1}, &stats, &terminal_descendant_blocks, true) ==
              std::vector<ANNS::IdxType>({1}) &&
              terminal_descendant_blocks == std::vector<ANNS::IdxType>({1}),
          "explicit block-frontier search may continue below a query terminal");

   const auto block_node = index.find_exact_node({1, 2, 3});
   expect(block_node != ANNS::SpecialBlockTrieIndex::kInvalidNodeId,
          "exact node lookup must find a terminal path");
   index.set_block_id(block_node, 7);
   expect(index.node(block_node).block_id == 7, "block-root metadata must be attached to the trie node");
   expect(index.labels_for_node(block_node) == std::vector<ANNS::LabelType>({1, 2, 3}),
          "node path labels must round-trip");

   const auto descendant_block_node = index.find_exact_node({1, 3, 4});
   expect(descendant_block_node != ANNS::SpecialBlockTrieIndex::kInvalidNodeId,
          "fixture must contain a block below a first terminal");
   index.set_block_id(descendant_block_node, 8);
   expect(index.terminal_block_portal_pairs() ==
              std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>>({{2, 8}, {3, 7}}),
          "each terminal must map to its nearest block ancestor");

   std::vector<ANNS::IdxType> block_frontier;
   expect(sorted(index.find_entry_groups({1}, &stats, &block_frontier, true)) ==
              std::vector<ANNS::IdxType>({1, 3}),
          "block-aware frontier must preserve every terminal-only entry");
   expect(block_frontier == std::vector<ANNS::IdxType>({8}),
          "block-aware frontier must emit only blocks below a first terminal");
   expect(stats.block_frontier_candidates == 1 && stats.final_block_entries == 1,
          "block frontier statistics must be exact");
   expect(stats.terminal_descendants_pruned == 0,
          "the only terminal descendant branch in the fixture contains a block");

   const auto root = std::filesystem::temp_directory_path() / "ung_special_block_trie_test";
   std::filesystem::create_directories(root);
   const auto binary = root / "special_block_trie.bin";
   index.save(binary.string());
   expect(std::filesystem::file_size(binary) == index.serialized_size_bytes(),
          "reported serialized size must match the binary file");

   ANNS::SpecialBlockTrieIndex loaded;
   loaded.load(binary.string(), 5, 8);
   expect(loaded.node_count() == index.node_count(), "loaded trie node count must be exact");
   expect(loaded.node(block_node).block_id == 7 &&
              loaded.node(descendant_block_node).block_id == 8,
          "block-root metadata must survive binary I/O");
   expect(loaded.find_entry_groups({1, 3}, nullptr) == index.find_entry_groups({1, 3}, nullptr),
          "loaded trie query results must equal constructed results");
   expect(loaded.terminal_block_portal_pairs() == index.terminal_block_portal_pairs(),
          "loaded trie portal topology must equal constructed topology");
   block_frontier.clear();
   expect(sorted(loaded.find_entry_groups({1}, &stats, &block_frontier, true)) ==
              std::vector<ANNS::IdxType>({1, 3}) &&
              block_frontier == std::vector<ANNS::IdxType>({8}),
          "loaded trie must rebuild descendant-block pruning metadata");

   const auto truncated = root / "truncated.bin";
   std::filesystem::copy_file(binary, truncated,
                              std::filesystem::copy_options::overwrite_existing);
   std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated) - 1);
   expect(load_throws(truncated, 5, 8), "truncated trie file must be rejected");
   expect(load_throws(binary, 4, 8), "group-count mismatch must be rejected");
   expect(load_throws(binary, 5, 7), "unknown block-root metadata must be rejected");

   const ANNS::EntryGroupProviderImpl provider =
       ANNS::parse_entry_group_provider_impl("special_block_trie");
   expect(provider == ANNS::EntryGroupProviderImpl::SpecialBlockTrie,
          "special_block_trie must parse to the dedicated provider");
   expect(std::string(ANNS::entry_group_provider_impl_name(provider)) == "special_block_trie",
          "special-block trie provider must round-trip to its public name");

   std::filesystem::remove_all(root);
   std::cout << "special-block trie index checks passed\n";
   return 0;
}
