#include "ung_special_edge_io.h"
#include "uni_nav_graph.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
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
} // namespace

int main()
{
   expect(ANNS::is_supported_special_block_index_format("special_block_trie_v1") &&
              ANNS::is_supported_special_block_index_format("special_block_trie_v2") &&
              ANNS::is_supported_special_block_index_format("special_block_trie_multilevel_v1") &&
              !ANNS::is_supported_special_block_index_format("special_block_trie_unknown"),
          "loader format policy must accept legacy and multilevel indexes only");

   const auto root = std::filesystem::temp_directory_path() / "ung_special_edge_io_test";
   std::filesystem::create_directories(root);
   const auto csv = root / "edges.csv";
   const auto bin = root / "edges.bin";
   {
      std::ofstream out(csv);
      out << "source_point_id,target_point_id,special_block_id,edge_kind\n";
      out << "1,2,3,intra\n";
      out << "4,5,6,inter\n";
      out << "7,8,9,intra\n";
   }

   uint64_t converted = 0;
   std::string error;
   expect(ANNS::convert_special_edge_csv_to_binary(csv.string(), bin.string(), converted, error),
          "CSV-to-binary conversion must succeed");
   expect(converted == 3, "converted edge count must be exact");

   const auto validation = ANNS::validate_special_edge_binary_file(bin.string());
   expect(validation.valid, "converted binary must validate");
   expect(validation.count == 3, "validated count must be exact");

   std::vector<ANNS::SpecialEdgeBinaryRecord> records;
   uint64_t read_count = 0;
   expect(ANNS::for_each_special_edge_binary_record(
              bin.string(),
              [&](const ANNS::SpecialEdgeBinaryRecord &record) { records.push_back(record); },
              read_count, error),
          "validated binary records must be readable");
   expect(read_count == 3 && records.size() == 3, "all binary records must be visited");
   expect(records[0].source == 1 && records[0].target == 2 && records[0].block == 3 && records[0].kind == 0,
          "intra record fields must round-trip");
   expect(records[1].source == 4 && records[1].target == 5 && records[1].block == 6 && records[1].kind == 1,
          "inter record fields must round-trip");

   const auto truncated = root / "truncated.bin";
   {
      std::ofstream out(truncated, std::ios::binary);
      const uint64_t declared_count = 2;
      out.write(reinterpret_cast<const char *>(&declared_count), sizeof(declared_count));
      out.write(reinterpret_cast<const char *>(&records[0]), sizeof(records[0]));
   }
   expect(!ANNS::validate_special_edge_binary_file(truncated.string()).valid,
          "truncated record data must fail validation");

   const auto regular_csr = root / "regular_csr.bin";
   std::vector<std::vector<ANNS::IdxType>> regular_edges = {
       {}, {2, 3}, {}, {0}};
   uint64_t regular_count = 0;
   expect(ANNS::write_regular_edge_csr_file(
              regular_csr.string(), regular_edges, regular_count, error),
          "regular CSR write must succeed");
   const auto regular_validation = ANNS::validate_special_edge_binary_file(regular_csr.string());
   expect(regular_validation.valid && regular_validation.format_version == 2,
          "regular CSR must validate as version 2");
   expect(regular_validation.regular_targets_only && regular_validation.count == 3 &&
              regular_validation.num_sources == regular_edges.size(),
          "regular CSR header fields must be exact");
   records.clear();
   expect(ANNS::for_each_special_edge_binary_record(
              regular_csr.string(),
              [&](const ANNS::SpecialEdgeBinaryRecord &record) { records.push_back(record); },
              read_count, error),
          "regular CSR records must be readable");
   expect(records.size() == 3 && records[0].source == 1 && records[0].target == 2 &&
              records[2].source == 3 && records[2].target == 0,
          "regular CSR source offsets and targets must round-trip");
   ANNS::RegularEdgeCsr loaded_regular_csr;
   expect(ANNS::read_regular_edge_csr_file(
              regular_csr.string(), loaded_regular_csr, error),
          "regular CSR must support direct flat loading");
   expect(loaded_regular_csr.offsets == std::vector<uint64_t>({0, 0, 2, 2, 3}) &&
              loaded_regular_csr.targets == std::vector<ANNS::IdxType>({2, 3, 0}),
          "direct regular CSR loading must preserve offsets and flat targets");

   const auto special_csr = root / "special_csr.bin";
   std::vector<std::vector<ANNS::SpecialEdge>> special_edges(4);
   special_edges[1].push_back({2, 3, ANNS::SpecialEdgeKind::IntraBlock});
   special_edges[1].push_back({3, 4, ANNS::SpecialEdgeKind::InterBlock});
   special_edges[3].push_back({0, 5, ANNS::SpecialEdgeKind::IntraBlock});
   uint64_t special_count = 0;
   expect(ANNS::write_special_edge_csr_file(
              special_csr.string(), special_edges,
              [](ANNS::IdxType, const ANNS::SpecialEdge &) { return true; },
              special_count, error),
          "special CSR write must succeed");
   const auto special_validation = ANNS::validate_special_edge_binary_file(special_csr.string());
   expect(special_validation.valid && special_validation.format_version == 2 &&
              !special_validation.regular_targets_only && special_validation.count == 3,
          "special CSR must validate as version 2");
   records.clear();
   expect(ANNS::for_each_special_edge_binary_record(
              special_csr.string(),
              [&](const ANNS::SpecialEdgeBinaryRecord &record) { records.push_back(record); },
              read_count, error),
          "special CSR records must be readable");
   expect(records.size() == 3 && records[0].source == 1 && records[0].block == 3 &&
              records[0].kind == 0 && records[1].block == 4 && records[1].kind == 1,
          "special CSR fields must round-trip");
   ANNS::SpecialEdgeCsr loaded_special_csr;
   expect(ANNS::read_special_edge_csr_file(
              special_csr.string(), loaded_special_csr, error),
          "special CSR must support direct flat loading");
   expect(loaded_special_csr.offsets == std::vector<uint64_t>({0, 0, 2, 2, 3}) &&
              loaded_special_csr.edges.size() == 3 &&
              loaded_special_csr.edges[0].unpack().target_point_id == 2 &&
              loaded_special_csr.edges[1].unpack().special_block_id == 4 &&
              loaded_special_csr.edges[1].unpack().kind == ANNS::SpecialEdgeKind::InterBlock,
          "direct special CSR loading must preserve offsets and flat payloads");

   const auto block_metadata = root / "special_blocks.bin";
   std::vector<ANNS::SpecialBlock> blocks(2);
   blocks[0].block_id = 1;
   blocks[0].level = 0;
   blocks[0].parent_block_id = 2;
   blocks[0].root_group_id = 7;
   blocks[0].entry_point_id = 11;
   blocks[0].point_count = 20;
   blocks[0].subtree_point_count = 31;
   blocks[0].root_labels = {1, 3};
   blocks[0].common_labels = {1};
   blocks[0].member_group_ids = {7, 8, 9};
   blocks[1].block_id = 2;
   blocks[1].level = 1;
   blocks[1].root_group_id = 10;
   blocks[1].entry_point_id = 12;
   blocks[1].point_count = 5;
   blocks[1].subtree_point_count = 5;
   blocks[1].root_labels = {1, 3, 5};
   blocks[1].common_labels = {1, 3};
   blocks[1].member_group_ids = {10};
   ANNS::save_special_block_metadata_binary(block_metadata.string(), blocks);
   std::vector<ANNS::SpecialBlock> loaded_blocks;
   expect(ANNS::load_special_block_metadata_binary(
              block_metadata.string(), loaded_blocks, error),
          "special block metadata binary must load");
   expect(loaded_blocks.size() == 2 && loaded_blocks[0].block_id == 1 &&
              loaded_blocks[0].root_labels == blocks[0].root_labels &&
              loaded_blocks[0].common_labels == blocks[0].common_labels &&
              loaded_blocks[0].member_group_ids == blocks[0].member_group_ids &&
              loaded_blocks[0].child_block_ids == blocks[0].child_block_ids &&
              loaded_blocks[0].level == 0 && loaded_blocks[0].parent_block_id == 2 &&
              loaded_blocks[1].entry_point_id == 12,
          "special block metadata fields must round-trip");

   auto expect_invalid_blocks = [&](std::vector<ANNS::SpecialBlock> invalid,
                                    const char *message) {
      std::string validation_error;
      expect(!ANNS::validate_special_block_metadata(invalid, validation_error) &&
                 !validation_error.empty(),
             message);
   };
   auto invalid_level = blocks;
   invalid_level[1].level = 2;
   expect_invalid_blocks(std::move(invalid_level),
                         "metadata must reject unsupported third block layer");
   auto invalid_parent = blocks;
   invalid_parent[0].parent_block_id = 1;
   expect_invalid_blocks(std::move(invalid_parent),
                         "middle parent must reference an upper block");
   auto duplicate_owner = blocks;
   duplicate_owner.push_back(blocks[0]);
   duplicate_owner.back().block_id = 3;
   duplicate_owner.back().parent_block_id = 2;
   expect_invalid_blocks(std::move(duplicate_owner),
                         "same-layer duplicate group ownership must be rejected");
   auto cross_layer_child = blocks;
   cross_layer_child[0].child_block_ids = {2};
   expect_invalid_blocks(std::move(cross_layer_child),
                         "child relation must stay within one block layer");

   std::vector<ANNS::SpecialBlock> same_layer(3, blocks[0]);
   for (size_t i = 0; i < same_layer.size(); ++i)
   {
      same_layer[i].block_id = static_cast<ANNS::IdxType>(i + 1);
      same_layer[i].parent_block_id = 0;
      same_layer[i].root_group_id = static_cast<ANNS::IdxType>(20 + i);
      same_layer[i].member_group_ids = {same_layer[i].root_group_id};
      same_layer[i].root_labels = {1, static_cast<ANNS::LabelType>(10 + i)};
      same_layer[i].point_count = 1;
      same_layer[i].subtree_point_count = 1;
      same_layer[i].child_block_ids.clear();
   }
   auto multiple_parent = same_layer;
   multiple_parent[0].child_block_ids = {3};
   multiple_parent[1].child_block_ids = {3};
   expect_invalid_blocks(std::move(multiple_parent),
                         "same-layer child must have a unique parent");
   auto cyclic_children = same_layer;
   cyclic_children[0].child_block_ids = {2};
   cyclic_children[1].child_block_ids = {3};
   cyclic_children[2].child_block_ids = {1};
   expect_invalid_blocks(std::move(cyclic_children),
                         "same-layer child topology must be acyclic");

   std::vector<ANNS::SpecialBlock> edge_blocks(3);
   edge_blocks[0].block_id = 1;
   edge_blocks[0].level = 0;
   edge_blocks[0].child_block_ids = {2};
   edge_blocks[1].block_id = 2;
   edge_blocks[1].level = 0;
   edge_blocks[2].block_id = 3;
   edge_blocks[2].level = 1;
   const std::vector<ANNS::IdxType> point_middle_owner = {1, 1, 2, 2};
   const std::vector<ANNS::IdxType> point_upper_owner = {3, 3, 3, 3};
   auto expect_edge_semantics = [&](ANNS::IdxType source,
                                    const ANNS::SpecialEdge &edge,
                                    bool expected, const char *message) {
      std::string semantic_error;
      expect(ANNS::validate_special_edge_semantics(
                 source, edge, edge_blocks, point_middle_owner,
                 point_upper_owner, semantic_error) == expected,
             message);
   };
   expect_edge_semantics(0, {1, 1, ANNS::SpecialEdgeKind::IntraBlock}, true,
                         "intra edge may stay within its direct owner");
   expect_edge_semantics(0, {2, 1, ANNS::SpecialEdgeKind::InterBlock}, true,
                         "inter edge may target a direct child block");
   expect(ANNS::special_edge_target_owner(
              {2, 1, ANNS::SpecialEdgeKind::InterBlock}, edge_blocks,
              point_middle_owner, point_upper_owner) == 2,
          "middle inter target owner must use middle ownership");
   expect(ANNS::special_edge_target_owner(
              {2, 3, ANNS::SpecialEdgeKind::InterBlock}, edge_blocks,
              point_middle_owner, point_upper_owner) == 3,
          "upper inter target owner must use upper ownership");
   expect_edge_semantics(2, {3, 1, ANNS::SpecialEdgeKind::IntraBlock}, false,
                         "intra edge source must belong to its declared owner");
   expect_edge_semantics(0, {2, 1, ANNS::SpecialEdgeKind::IntraBlock}, false,
                         "intra edge target must belong to its declared owner");
   expect_edge_semantics(0, {3, 2, ANNS::SpecialEdgeKind::InterBlock}, false,
                         "inter edge source must belong to its declared owner");
   expect_edge_semantics(0, {2, 3, ANNS::SpecialEdgeKind::InterBlock}, false,
                         "inter edge target must belong to a same-layer direct child");
   expect_edge_semantics(0, {1, 0, ANNS::SpecialEdgeKind::IntraBlock}, false,
                         "zero edge owner must be rejected");
   expect_edge_semantics(4, {1, 1, ANNS::SpecialEdgeKind::IntraBlock}, false,
                         "out-of-range edge source must be rejected");

   // The graph-aware validation gate checks semantic relationships which the
   // standalone binary format cannot know.  This fixture models two nested
   // middle blocks and one containing upper block over four contiguous groups.
   std::vector<std::vector<ANNS::LabelType>> group_labels(5);
   group_labels[1] = {1, 2};
   group_labels[2] = {1, 2, 3};
   group_labels[3] = {1, 4};
   group_labels[4] = {1, 5};
   std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>> group_ranges(5);
   group_ranges[1] = {0, 2};
   group_ranges[2] = {2, 5};
   group_ranges[3] = {5, 6};
   group_ranges[4] = {6, 8};
   const std::vector<ANNS::IdxType> point_to_group = {1, 1, 2, 2, 2, 3, 4, 4};

   std::vector<ANNS::SpecialBlock> semantic_blocks(3);
   semantic_blocks[0].block_id = 1;
   semantic_blocks[0].level = 0;
   semantic_blocks[0].parent_block_id = 3;
   semantic_blocks[0].root_group_id = 1;
   semantic_blocks[0].entry_point_id = 0;
   semantic_blocks[0].point_count = 2;
   semantic_blocks[0].subtree_point_count = 5;
   semantic_blocks[0].root_labels = {1, 2};
   semantic_blocks[0].member_group_ids = {1};
   semantic_blocks[0].child_block_ids = {2};
   semantic_blocks[1].block_id = 2;
   semantic_blocks[1].level = 0;
   semantic_blocks[1].parent_block_id = 3;
   semantic_blocks[1].root_group_id = 2;
   semantic_blocks[1].entry_point_id = 2;
   semantic_blocks[1].point_count = 3;
   semantic_blocks[1].subtree_point_count = 3;
   semantic_blocks[1].root_labels = {1, 2, 3};
   semantic_blocks[1].member_group_ids = {2};
   semantic_blocks[2].block_id = 3;
   semantic_blocks[2].level = 1;
   semantic_blocks[2].root_group_id = 3;
   semantic_blocks[2].entry_point_id = 5;
   semantic_blocks[2].point_count = 3;
   semantic_blocks[2].subtree_point_count = 8;
   semantic_blocks[2].root_labels = {1};
   semantic_blocks[2].member_group_ids = {3, 4};

   auto expect_graph_semantics = [&](const std::vector<ANNS::SpecialBlock> &candidate,
                                     bool expected,
                                     const char *message) {
      std::string validation_error;
      expect(ANNS::validate_special_block_metadata(candidate, validation_error),
             "graph-semantics fixtures must retain valid standalone metadata");
      const bool valid = ANNS::validate_special_block_graph_semantics(
          candidate, 8, 4, group_labels, group_ranges, point_to_group,
          validation_error);
      expect(valid == expected && (valid || !validation_error.empty()), message);
   };
   expect_graph_semantics(semantic_blocks, true,
                          "valid nested block metadata must match its source graph");
   auto unrelated_child = semantic_blocks;
   unrelated_child[1].root_labels = {1, 4};
   expect_graph_semantics(unrelated_child, false,
                          "child root must be a strict descendant of its parent root");
   auto unrelated_upper = semantic_blocks;
   unrelated_upper[2].root_labels = {7};
   expect_graph_semantics(unrelated_upper, false,
                          "upper root must contain linked middle roots");
   auto false_root = semantic_blocks;
   false_root[0].root_labels = {1, 9};
   expect_graph_semantics(false_root, false,
                          "block root labels must contain every direct member group");
   auto external_entry = semantic_blocks;
   external_entry[0].entry_point_id = 5;
   expect_graph_semantics(external_entry, false,
                          "entry point must belong to a direct member group");
   auto wrong_point_count = semantic_blocks;
   wrong_point_count[0].point_count = 3;
   expect_graph_semantics(wrong_point_count, false,
                          "point_count must equal the direct member range total");

   const auto explicit_index = root / "explicit_index";
   std::filesystem::create_directories(explicit_index);
   {
      std::ofstream out(explicit_index / "meta");
      out << "num_points=0\n"
          << "num_groups=0\n"
          << "special_blocks_enabled=1\n"
          << "special_block_partition=trie\n";
   }
   const auto explicit_load_error = [&]() {
      try
      {
         ANNS::UniNavGraph index;
         index.load_special_block_index(explicit_index.string());
      }
      catch (const std::runtime_error &ex)
      {
         return std::string(ex.what());
      }
      return std::string();
   };
   expect(!explicit_load_error().empty(),
          "explicit sidecar load must reject missing format and fingerprint");
   {
      std::ofstream out(explicit_index / "meta");
      out << "index_format=special_block_trie_multilevel_v1\n"
          << "num_points=0\n"
          << "num_groups=0\n"
          << "special_blocks_enabled=1\n"
          << "special_block_partition=trie\n";
   }
   expect(!explicit_load_error().empty(),
          "explicit sidecar load must reject a missing source fingerprint");
   {
      std::ofstream out(explicit_index / "meta");
      out << "index_format=special_block_trie_multilevel_v1\n"
          // FNV-1a fingerprint for an empty source UNG: append num_points=0
          // and num_groups=0 using the production little-endian encoding.
          << "source_ung_fingerprint=a31e272015f12c43\n"
          << "num_points=0\n"
          << "num_groups=0\n"
          << "special_blocks_enabled=1\n"
          << "special_block_partition=trie\n";
   }
   expect(!explicit_load_error().empty(),
          "explicit sidecar load must reject missing binary metadata");
   {
      std::ofstream out(explicit_index / "special_blocks.bin", std::ios::binary);
      out << "corrupt";
   }
   expect(!explicit_load_error().empty(),
          "explicit sidecar load must reject corrupt binary metadata");

   // Preflight the whole immutable bundle.  Placeholder files are sufficient
   // here because the missing-file diagnostic must happen before parsing the
   // intentionally corrupt metadata/trie payloads.
   {
      std::ofstream out(explicit_index / "special_block_trie.bin", std::ios::binary);
      out << "placeholder";
   }
   {
      std::ofstream out(explicit_index / "special_edges.bin", std::ios::binary);
      out << "placeholder";
   }
   std::string load_error = explicit_load_error();
   expect(load_error.find("special_trie_regular_edges.bin") != std::string::npos,
          "explicit bundle must report a missing regular-edge sidecar before parsing");
   {
      std::ofstream out(explicit_index / "special_trie_regular_edges.bin", std::ios::binary);
      out << "placeholder";
   }
   std::filesystem::remove(explicit_index / "special_edges.bin");
   load_error = explicit_load_error();
   expect(load_error.find("special_edges.bin") != std::string::npos,
          "explicit bundle must report a missing special-edge sidecar before parsing");

   std::filesystem::remove_all(root);
   std::cout << "special edge binary I/O checks passed\n";
   return 0;
}
