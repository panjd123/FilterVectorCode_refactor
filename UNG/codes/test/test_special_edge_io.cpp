#include "ung_special_edge_io.h"

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
   blocks[0].root_group_id = 7;
   blocks[0].entry_point_id = 11;
   blocks[0].point_count = 20;
   blocks[0].subtree_point_count = 31;
   blocks[0].root_labels = {1, 3};
   blocks[0].common_labels = {1};
   blocks[0].member_group_ids = {7, 8, 9};
   blocks[0].child_block_ids = {2};
   blocks[1].block_id = 2;
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
              loaded_blocks[1].entry_point_id == 12,
          "special block metadata fields must round-trip");

   std::filesystem::remove_all(root);
   std::cout << "special edge binary I/O checks passed\n";
   return 0;
}
