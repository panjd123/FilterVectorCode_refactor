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
   // A prior assertion failure exits before the normal cleanup below. Start
   // from an empty fixture so reruns cannot inherit stale sidecar files.
   std::filesystem::remove_all(root);
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

   ANNS::SpecialEdgeBinaryRecord parsed_csv_record;
   expect(ANNS::parse_special_edge_csv_record(
              "10,11,12,inter\r", parsed_csv_record, error) &&
              parsed_csv_record.source == 10 && parsed_csv_record.target == 11 &&
              parsed_csv_record.block == 12 && parsed_csv_record.kind == 1,
          "shared CSV parser must accept one exact CRLF record");
   expect(!ANNS::parse_special_edge_csv_record(
              "10,11,inter", parsed_csv_record, error),
          "shared CSV parser must reject a missing numeric field");
   expect(!ANNS::parse_special_edge_csv_record(
              "10,11,12,unknown", parsed_csv_record, error),
          "shared CSV parser must reject an unknown edge kind");
   expect(!ANNS::parse_special_edge_csv_record(
              "10,11,12,inter,extra", parsed_csv_record, error),
          "shared CSV parser must reject trailing columns");

   const auto malformed_csv = root / "malformed_edges.csv";
   const auto malformed_bin = root / "malformed_edges.bin";
   {
      std::ofstream out(malformed_csv);
      out << "source_point_id,target_point_id,special_block_id,edge_kind\n";
      out << "10,11,12,inter,extra\n";
   }
   converted = 0;
   expect(!ANNS::convert_special_edge_csv_to_binary(
              malformed_csv.string(), malformed_bin.string(), converted, error),
          "converter must use the strict shared CSV parser");
   expect(!std::filesystem::exists(malformed_bin),
          "failed CSV conversion must not publish a partial binary");

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
          "upper inter target owner and per-target cap key must use upper ownership");
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
   semantic_blocks[0].common_labels = {1, 2};
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
   semantic_blocks[1].common_labels = {1, 2, 3};
   semantic_blocks[1].member_group_ids = {2};
   semantic_blocks[2].block_id = 3;
   semantic_blocks[2].level = 1;
   semantic_blocks[2].root_group_id = 1;
   semantic_blocks[2].entry_point_id = 5;
   semantic_blocks[2].point_count = 8;
   semantic_blocks[2].subtree_point_count = 8;
   semantic_blocks[2].root_labels = {1};
   semantic_blocks[2].common_labels = {1};
   semantic_blocks[2].member_group_ids = {1, 2, 3, 4};

   auto expect_graph_semantics = [&](const std::vector<ANNS::SpecialBlock> &candidate,
                                     bool expected,
                                     const char *message) {
      std::string validation_error;
      const bool metadata_valid =
          ANNS::validate_special_block_metadata(candidate, validation_error);
      if (!metadata_valid)
         std::cerr << "standalone-metadata mismatch: " << message
                   << "; validator: " << validation_error << '\n';
      expect(metadata_valid,
             "graph-semantics fixtures must retain valid standalone metadata");
      const bool valid = ANNS::validate_special_block_graph_semantics(
          candidate, 8, 4, group_labels, group_ranges, point_to_group,
          validation_error);
      if (valid != expected)
         std::cerr << "graph-semantics mismatch: " << message
                   << "; validator: " << validation_error << '\n';
      expect(valid == expected && (valid || !validation_error.empty()), message);
   };
   {
      std::string validation_error;
      expect(ANNS::validate_special_block_graph_semantics(
                 {}, 0, 0, {}, {}, {}, validation_error),
             "empty graph and empty block metadata must be a valid degenerate bundle");
   }
   expect_graph_semantics(semantic_blocks, true,
                          "valid nested block metadata must match its source graph");
   auto expect_invalid_source_layout =
       [&](const std::vector<std::vector<ANNS::LabelType>> &candidate_labels,
           const std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>> &candidate_ranges,
           const std::vector<ANNS::IdxType> &candidate_point_to_group,
           const char *message) {
          std::string validation_error;
          expect(!ANNS::validate_special_block_partition_semantics(
                     semantic_blocks, 8, 4, candidate_labels, candidate_ranges,
                     candidate_point_to_group, validation_error) &&
                     !validation_error.empty(),
                 message);
       };
   {
      auto invalid_sentinel_labels = group_labels;
      invalid_sentinel_labels[0] = {99};
      expect_invalid_source_layout(invalid_sentinel_labels, group_ranges,
                                   point_to_group,
                                   "group zero must remain an empty label sentinel");
   }
   {
      auto invalid_sentinel_range = group_ranges;
      invalid_sentinel_range[0] = {0, 1};
      expect_invalid_source_layout(group_labels, invalid_sentinel_range,
                                   point_to_group,
                                   "group zero must remain an empty range sentinel");
   }
   {
      auto empty_group_labels = group_labels;
      empty_group_labels[4].clear();
      expect_invalid_source_layout(empty_group_labels, group_ranges,
                                   point_to_group,
                                   "every source group must have a non-empty label set");
   }
   {
      auto unsorted_group_labels = group_labels;
      unsorted_group_labels[2] = {2, 1, 3};
      expect_invalid_source_layout(unsorted_group_labels, group_ranges,
                                   point_to_group,
                                   "source group label sets must use canonical order");
   }
   {
      auto duplicate_group_labels = group_labels;
      duplicate_group_labels[4] = group_labels[3];
      expect_invalid_source_layout(duplicate_group_labels, group_ranges,
                                   point_to_group,
                                   "source group label sets must be globally unique");
   }
   {
      auto ranges_with_gap = group_ranges;
      ranges_with_gap[2].first = 3;
      expect_invalid_source_layout(group_labels, ranges_with_gap,
                                   point_to_group,
                                   "source group ranges must not contain holes");
   }
   {
      auto overlapping_ranges = group_ranges;
      overlapping_ranges[2].first = 1;
      expect_invalid_source_layout(group_labels, overlapping_ranges,
                                   point_to_group,
                                   "source group ranges must not overlap");
   }
   {
      auto truncated_ranges = group_ranges;
      truncated_ranges[4].second = 7;
      expect_invalid_source_layout(group_labels, truncated_ranges,
                                   point_to_group,
                                   "source group ranges must cover every point");
   }
   {
      auto inconsistent_point_owner = point_to_group;
      inconsistent_point_owner[2] = 1;
      expect_invalid_source_layout(group_labels, group_ranges,
                                   inconsistent_point_owner,
                                   "point ownership must agree with the source group ranges");
   }
   {
      auto invalid_level = semantic_blocks;
      invalid_level[0].level = 2;
      std::string validation_error;
      expect(!ANNS::validate_special_block_graph_semantics(
                 invalid_level, 8, 4, group_labels, group_ranges,
                 point_to_group, validation_error) &&
                 !validation_error.empty(),
             "the public graph validator must reject invalid standalone metadata safely");
   }
   auto missing_direct_child = semantic_blocks;
   missing_direct_child[0].child_block_ids.clear();
   expect_graph_semantics(missing_direct_child, false,
                          "same-layer topology must not omit its nearest block ancestor");

   auto nested_upper = semantic_blocks;
   ANNS::SpecialBlock upper_child;
   upper_child.block_id = 4;
   upper_child.level = 1;
   upper_child.root_group_id = 1;
   upper_child.entry_point_id = 0;
   upper_child.point_count = 5;
   upper_child.subtree_point_count = 5;
   upper_child.root_labels = {1, 2};
   upper_child.common_labels = {1, 2};
   upper_child.member_group_ids = {1, 2};
   ANNS::SpecialBlock outer_middle;
   outer_middle.block_id = 5;
   outer_middle.level = 0;
   outer_middle.parent_block_id = 3;
   outer_middle.root_group_id = 3;
   outer_middle.entry_point_id = 5;
   outer_middle.point_count = 1;
   outer_middle.subtree_point_count = 1;
   outer_middle.root_labels = {1, 4};
   outer_middle.common_labels = {1, 4};
   outer_middle.member_group_ids = {3};
   nested_upper[2].child_block_ids = {4};
   nested_upper[2].member_group_ids = {3, 4};
   nested_upper[2].root_group_id = 3;
   nested_upper[2].point_count = 3;
   nested_upper[0].parent_block_id = 4;
   nested_upper[1].parent_block_id = 4;
   nested_upper.push_back(upper_child);
   nested_upper.push_back(outer_middle);
   expect_graph_semantics(nested_upper, true,
                          "middle blocks may link to their nearest nested upper ancestor");
   auto skipped_upper_parent = nested_upper;
   skipped_upper_parent[0].parent_block_id = 3;
   expect_graph_semantics(skipped_upper_parent, false,
                          "middle parent must not skip a nearer upper block ancestor");
   // Independent partitions need not align with middle ownership. Model an
   // upper child nested inside a middle block: group 1 remains owned by the
   // {1,2} upper block while group 2 moves to its {1,2,3} upper child.
   auto crossing_partition = nested_upper;
   crossing_partition[3].member_group_ids = {1};
   crossing_partition[3].point_count = 2;
   crossing_partition[3].common_labels = {1, 2};
   crossing_partition[3].child_block_ids = {6};
   ANNS::SpecialBlock crossing_upper_child;
   crossing_upper_child.block_id = 6;
   crossing_upper_child.level = 1;
   crossing_upper_child.root_group_id = 2;
   crossing_upper_child.entry_point_id = 2;
   crossing_upper_child.point_count = 3;
   crossing_upper_child.subtree_point_count = 3;
   crossing_upper_child.root_labels = {1, 2, 3};
   crossing_upper_child.common_labels = {1, 2, 3};
   crossing_upper_child.member_group_ids = {2};
   crossing_partition[1].parent_block_id = 6;
   crossing_partition.push_back(crossing_upper_child);
   expect_graph_semantics(crossing_partition, true,
                          "independent layers may split one middle region across nested upper owners");
   auto missing_upper_member = nested_upper;
   missing_upper_member[3].member_group_ids = {1};
   missing_upper_member[3].point_count = 2;
   expect_graph_semantics(missing_upper_member, false,
                          "a block cannot silently omit a group whose nearest root it owns");
   auto unreachable_upper = semantic_blocks;
   unreachable_upper[2].member_group_ids = {3, 4};
   unreachable_upper[2].root_group_id = 3;
   unreachable_upper[2].point_count = 3;
   expect_graph_semantics(unreachable_upper, false,
                          "every upper block needs a middle-owned direct member activation point");
   std::vector<ANNS::SpecialBlock> ancestor_only_activation(2);
   ancestor_only_activation[0].block_id = 1;
   ancestor_only_activation[0].level = 0;
   ancestor_only_activation[0].root_group_id = 1;
   ancestor_only_activation[0].entry_point_id = 0;
   ancestor_only_activation[0].point_count = 2;
   ancestor_only_activation[0].subtree_point_count = 5;
   ancestor_only_activation[0].root_labels = {1};
   ancestor_only_activation[0].common_labels = {1, 2};
   ancestor_only_activation[0].member_group_ids = {1};
   ancestor_only_activation[1].block_id = 2;
   ancestor_only_activation[1].level = 1;
   ancestor_only_activation[1].root_group_id = 1;
   ancestor_only_activation[1].entry_point_id = 0;
   ancestor_only_activation[1].point_count = 2;
   ancestor_only_activation[1].subtree_point_count = 5;
   ancestor_only_activation[1].root_labels = {1, 2};
   ancestor_only_activation[1].common_labels = {1, 2};
   ancestor_only_activation[1].member_group_ids = {1};
   expect_graph_semantics(ancestor_only_activation, false,
                          "an upper block cannot rely on a larger middle ancestor for activation");
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
   {
      auto partition_without_entries = semantic_blocks;
      for (auto &block : partition_without_entries)
         block.entry_point_id = ANNS::SpecialBlock::kInvalidEntryPoint;
      std::string validation_error;
      expect(ANNS::validate_special_block_partition_semantics(
                 partition_without_entries, 8, 4, group_labels, group_ranges,
                 point_to_group, validation_error),
             "partition preflight must not require intra-graph entry points");
      expect(!ANNS::validate_special_block_graph_semantics(
                 partition_without_entries, 8, 4, group_labels, group_ranges,
                 point_to_group, validation_error),
             "the complete graph gate must require intra-graph entry points");
      auto invalid_partition_count = partition_without_entries;
      invalid_partition_count[0].point_count += 1;
      expect(!ANNS::validate_special_block_partition_semantics(
                 invalid_partition_count, 8, 4, group_labels, group_ranges,
                 point_to_group, validation_error),
             "partition preflight must still reject wrong direct point counts");
      auto invalid_partition_reachability = partition_without_entries;
      invalid_partition_reachability[2].member_group_ids = {3, 4};
      invalid_partition_reachability[2].point_count = 3;
      expect(!ANNS::validate_special_block_partition_semantics(
                 invalid_partition_reachability, 8, 4, group_labels,
                 group_ranges, point_to_group, validation_error),
             "partition preflight must still reject unreachable upper blocks");
   }
   auto wrong_point_count = semantic_blocks;
   wrong_point_count[0].point_count = 3;
   expect_graph_semantics(wrong_point_count, false,
                          "point_count must equal the direct member range total");
   auto wrong_subtree_point_count = semantic_blocks;
   wrong_subtree_point_count[0].subtree_point_count = 6;
   expect_graph_semantics(wrong_subtree_point_count, false,
                          "subtree_point_count must equal the complete source Trie subtree");

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

   // A requested optional heavy sidecar must fail closed when the binary is
   // present but corrupt and there is no CSV fallback.  Use a valid empty
   // graph bundle so this reaches the production heavy-edge loader rather
   // than failing earlier metadata checks.
   const auto corrupt_heavy_index = root / "corrupt_heavy_index";
   std::filesystem::create_directories(corrupt_heavy_index);
   {
      std::ofstream out(corrupt_heavy_index / "meta");
      out << "index_format=special_block_trie_multilevel_v1\n"
          << "source_ung_fingerprint=a31e272015f12c43\n"
          << "num_points=0\n"
          << "num_groups=0\n"
          << "special_blocks_enabled=1\n"
          << "special_block_partition=none\n";
   }
   ANNS::save_special_block_metadata_binary(
       (corrupt_heavy_index / "special_blocks.bin").string(), {});
   uint64_t empty_edge_count = 0;
   expect(ANNS::write_special_edge_csr_file(
              (corrupt_heavy_index / "special_edges.bin").string(), {},
              [](ANNS::IdxType, const ANNS::SpecialEdge &) { return true; },
              empty_edge_count, error),
          "empty special-edge CSR fixture must be writable");
   {
      std::ofstream out(corrupt_heavy_index / "special_heavy_edges.bin",
                        std::ios::binary);
      out << "corrupt";
   }
   const char *old_heavy_search = std::getenv("UNG_SPECIAL_HEAVY_EDGE_SEARCH");
   const std::string old_heavy_search_value =
       old_heavy_search == nullptr ? std::string() : std::string(old_heavy_search);
   setenv("UNG_SPECIAL_HEAVY_EDGE_SEARCH", "1", 1);
   {
      std::ofstream out(corrupt_heavy_index / "special_heavy_edges.csv");
      out << "source_point_id,target_point_id,special_block_id,edge_kind\n"
          << "0,0,0,inter,extra\n";
   }
   std::string malformed_heavy_error;
   try
   {
      ANNS::UniNavGraph index;
      index.load_special_block_index(corrupt_heavy_index.string());
   }
   catch (const std::runtime_error &ex)
   {
      malformed_heavy_error = ex.what();
   }
   expect(malformed_heavy_error.find("invalid heavy special-edge CSV") !=
              std::string::npos,
          "runtime heavy CSV fallback must use the strict shared parser");
   {
      std::ofstream out(corrupt_heavy_index / "special_heavy_edges.csv");
      out << "source_point_id,target_point_id,special_block_id,edge_kind\n";
   }
   bool valid_heavy_csv_loaded = true;
   try
   {
      ANNS::UniNavGraph index;
      index.load_special_block_index(corrupt_heavy_index.string());
   }
   catch (const std::runtime_error &)
   {
      valid_heavy_csv_loaded = false;
   }
   expect(valid_heavy_csv_loaded,
          "a valid heavy CSV must remain a usable fallback after reaching EOF");
   std::filesystem::remove(corrupt_heavy_index / "special_heavy_edges.csv");

   std::string corrupt_heavy_error;
   try
   {
      ANNS::UniNavGraph index;
      index.load_special_block_index(corrupt_heavy_index.string());
   }
   catch (const std::runtime_error &ex)
   {
      corrupt_heavy_error = ex.what();
   }
   if (old_heavy_search == nullptr)
      unsetenv("UNG_SPECIAL_HEAVY_EDGE_SEARCH");
   else
      setenv("UNG_SPECIAL_HEAVY_EDGE_SEARCH",
             old_heavy_search_value.c_str(), 1);
   if (corrupt_heavy_error.find("heavy special-edge binary is invalid") ==
       std::string::npos)
      std::cerr << "unexpected corrupt-heavy load outcome: "
                << (corrupt_heavy_error.empty() ? "<no exception>"
                                                : corrupt_heavy_error)
                << '\n';
   expect(corrupt_heavy_error.find("heavy special-edge binary is invalid") !=
              std::string::npos,
          "requested corrupt heavy binary without CSV fallback must fail closed");

   std::filesystem::remove_all(root);
   std::cout << "special edge binary I/O checks passed\n";
   return 0;
}
