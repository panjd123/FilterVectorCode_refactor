#ifndef ANNS_UNG_UPPER_HNSW_H
#define ANNS_UNG_UPPER_HNSW_H

#include "config.h"
#include "storage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace faiss
{
struct IndexHNSWFlat;
}

namespace ANNS
{

struct UpperHnswCandidate
{
   IdxType point_id = 0;
   IdxType block_id = 0;
   float distance = 0.0f;
};

struct UpperHnswStats
{
   size_t num_nodes = 0;
   size_t covered_blocks = 0;
   int max_level = -1;
   size_t reachable_level0_nodes = 0;
   std::vector<size_t> nodes_per_level;
   std::vector<double> average_degree_per_level;
   std::vector<size_t> max_degree_per_level;
};

class UpperHnswIndex
{
public:
   UpperHnswIndex();
   ~UpperHnswIndex();
   UpperHnswIndex(UpperHnswIndex &&) noexcept;
   UpperHnswIndex &operator=(UpperHnswIndex &&) noexcept;
   UpperHnswIndex(const UpperHnswIndex &) = delete;
   UpperHnswIndex &operator=(const UpperHnswIndex &) = delete;

   void build(const std::shared_ptr<IStorage> &base_storage,
              const std::vector<IdxType> &point_to_block,
              const std::vector<IdxType> &block_entry_points,
              IdxType num_blocks,
              uint32_t sample_denominator,
              uint32_t max_degree,
              uint32_t ef_construction,
              uint64_t seed,
              uint32_t num_threads);

   void save(const std::string &prefix, IdxType base_num_points,
             IdxType num_blocks) const;
   void load(const std::string &prefix, IdxType expected_base_num_points,
             IdxType expected_num_blocks);

   std::vector<UpperHnswCandidate> search(const float *query,
                                          size_t top_n,
                                          size_t ef_search) const;

   UpperHnswStats statistics() const;
   bool empty() const;
   size_t size() const;
   const std::vector<IdxType> &point_ids() const { return point_ids_; }
   const std::vector<IdxType> &block_ids() const { return block_ids_; }

   static const char *index_filename();
   static const char *mapping_filename();

private:
   void validate_mapping(IdxType base_num_points, IdxType num_blocks) const;

   std::unique_ptr<faiss::IndexHNSWFlat> index_;
   std::vector<IdxType> point_ids_;
   std::vector<IdxType> block_ids_;
};

} // namespace ANNS

#endif // ANNS_UNG_UPPER_HNSW_H
