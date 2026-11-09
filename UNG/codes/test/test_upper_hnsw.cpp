#include "ung_upper_hnsw.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
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
   constexpr ANNS::IdxType kPoints = 512;
   constexpr ANNS::IdxType kDim = 8;
   constexpr ANNS::IdxType kBlocks = 16;
   const std::filesystem::path root = "/tmp/ung_upper_hnsw_test";
   std::filesystem::create_directories(root);
   const std::string bin = (root / "base.bin").string();
   const std::string labels = (root / "labels.txt").string();
   {
      std::ofstream out(bin, std::ios::binary | std::ios::trunc);
      out.write(reinterpret_cast<const char *>(&kPoints), sizeof(kPoints));
      out.write(reinterpret_cast<const char *>(&kDim), sizeof(kDim));
      for (ANNS::IdxType point = 0; point < kPoints; ++point)
      {
         float vector[kDim];
         for (ANNS::IdxType dim = 0; dim < kDim; ++dim)
            vector[dim] = static_cast<float>(point) * 0.01f +
                          static_cast<float>(dim) * 0.001f;
         out.write(reinterpret_cast<const char *>(vector), sizeof(vector));
      }
   }
   {
      std::ofstream out(labels, std::ios::trunc);
      for (ANNS::IdxType point = 0; point < kPoints; ++point)
         out << "1\n";
   }

   auto storage = ANNS::create_storage("float", false);
   storage->load_from_file(bin, labels);
   std::vector<ANNS::IdxType> point_to_block(kPoints, 0);
   std::vector<ANNS::IdxType> block_entries(kBlocks + 1, 0);
   for (ANNS::IdxType point = 0; point < kPoints; ++point)
      point_to_block[point] = point / (kPoints / kBlocks) + 1;
   for (ANNS::IdxType block = 1; block <= kBlocks; ++block)
      block_entries[block] = (block - 1) * (kPoints / kBlocks);

   ANNS::UpperHnswIndex index;
   index.build(storage, point_to_block, block_entries, kBlocks,
               4, 4, 32, 12345, 2);
   const ANNS::UpperHnswStats stats = index.statistics();
   expect(stats.num_nodes >= 100 && stats.num_nodes <= 180,
          "sampling should produce the expected upper-layer scale");
   expect(stats.covered_blocks == kBlocks,
          "forced block portals must cover every block");
   expect(stats.max_level >= 1 && stats.nodes_per_level.size() >= 2,
          "the upper index must contain multiple HNSW levels");
   expect(stats.nodes_per_level[0] == stats.num_nodes,
          "every sampled node must exist at upper level zero");
   expect(stats.max_degree_per_level[0] <= 8,
          "FAISS HNSW level zero degree must not exceed 2M");
   for (size_t level = 1; level < stats.max_degree_per_level.size(); ++level)
      expect(stats.max_degree_per_level[level] <= 4,
             "FAISS HNSW upper degree must not exceed M");
   expect(stats.reachable_level0_nodes * 100 >= stats.num_nodes * 95,
          "upper level-zero graph should be connected from its entry point");

   index.save(root.string() + "/", kPoints, kBlocks);
   ANNS::UpperHnswIndex loaded;
   loaded.load(root.string() + "/", kPoints, kBlocks);
   const ANNS::IdxType query_point = block_entries[4];
   const auto candidates = loaded.search(
       reinterpret_cast<const float *>(storage->get_vector(query_point)), 8, 32);
   expect(!candidates.empty(), "loaded upper HNSW must return candidates");
   expect(candidates.front().point_id == query_point,
          "exact sampled query vector must be the nearest upper candidate");
   expect(candidates.front().block_id == point_to_block[query_point],
          "upper candidate must preserve its bottom-block mapping");

   storage->clean();
   std::cout << "upper HNSW build, statistics, persistence, and search checks passed\n";
   return 0;
}
