#include "include/ung_upper_hnsw.h"

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include <omp.h>

namespace ANNS
{
namespace
{
constexpr std::array<char, 8> kMappingMagic{{'U', 'N', 'G', 'U', 'H', 'N', 'S', 'W'}};
constexpr uint32_t kMappingVersion = 1;

uint64_t splitmix64(uint64_t value)
{
   value += 0x9e3779b97f4a7c15ULL;
   value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
   value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
   return value ^ (value >> 31);
}

template <typename T>
void write_value(std::ofstream &out, const T &value)
{
   out.write(reinterpret_cast<const char *>(&value), sizeof(T));
   if (!out)
      throw std::runtime_error("failed to write upper-HNSW mapping");
}

template <typename T>
void read_value(std::ifstream &in, T &value)
{
   in.read(reinterpret_cast<char *>(&value), sizeof(T));
   if (!in)
      throw std::runtime_error("truncated upper-HNSW mapping");
}
} // namespace

UpperHnswIndex::UpperHnswIndex() = default;
UpperHnswIndex::~UpperHnswIndex() = default;
UpperHnswIndex::UpperHnswIndex(UpperHnswIndex &&) noexcept = default;
UpperHnswIndex &UpperHnswIndex::operator=(UpperHnswIndex &&) noexcept = default;

const char *UpperHnswIndex::index_filename()
{
   return "upper_hnsw.faiss";
}

const char *UpperHnswIndex::mapping_filename()
{
   return "upper_hnsw_mapping.bin";
}

void UpperHnswIndex::build(const std::shared_ptr<IStorage> &base_storage,
                           const std::vector<IdxType> &point_to_block,
                           const std::vector<IdxType> &block_entry_points,
                           IdxType num_blocks,
                           uint32_t sample_denominator,
                           uint32_t max_degree,
                           uint32_t ef_construction,
                           uint64_t seed,
                           uint32_t num_threads)
{
   if (!base_storage || base_storage->get_data_type() != DataType::FLOAT)
      throw std::invalid_argument("upper HNSW currently requires float vectors");
   const IdxType num_points = base_storage->get_num_points();
   const IdxType dim = base_storage->get_dim();
   if (point_to_block.size() != static_cast<size_t>(num_points))
      throw std::invalid_argument("upper HNSW point-to-block mapping has the wrong size");
   if (sample_denominator == 0 || max_degree < 2 || ef_construction < max_degree)
      throw std::invalid_argument("invalid upper HNSW build parameters");

   std::vector<uint8_t> selected(num_points, 0);
   for (IdxType block_id = 1; block_id <= num_blocks; ++block_id)
   {
      if (block_id >= block_entry_points.size())
         break;
      const IdxType point_id = block_entry_points[block_id];
      if (point_id < num_points && point_to_block[point_id] != 0)
         selected[point_id] = 1;
   }
   for (IdxType point_id = 0; point_id < num_points; ++point_id)
   {
      if (point_to_block[point_id] == 0)
         continue;
      if (splitmix64(static_cast<uint64_t>(point_id) ^ seed) % sample_denominator == 0)
         selected[point_id] = 1;
   }

   point_ids_.clear();
   block_ids_.clear();
   for (IdxType point_id = 0; point_id < num_points; ++point_id)
   {
      if (selected[point_id] == 0)
         continue;
      point_ids_.push_back(point_id);
      block_ids_.push_back(point_to_block[point_id]);
   }
   if (point_ids_.empty())
      throw std::runtime_error("upper HNSW sampling selected no points");

   std::vector<float> sampled_vectors(
       static_cast<size_t>(point_ids_.size()) * static_cast<size_t>(dim));
#pragma omp parallel for schedule(static) num_threads(num_threads)
   for (size_t local_id = 0; local_id < point_ids_.size(); ++local_id)
   {
      const float *source = reinterpret_cast<const float *>(
          base_storage->get_vector(point_ids_[local_id]));
      std::memcpy(sampled_vectors.data() + local_id * static_cast<size_t>(dim),
                  source, static_cast<size_t>(dim) * sizeof(float));
   }

   index_ = std::make_unique<faiss::IndexHNSWFlat>(
       static_cast<int>(dim), static_cast<int>(max_degree), 1, faiss::METRIC_L2);
   index_->hnsw.efConstruction = static_cast<int>(ef_construction);
   omp_set_num_threads(static_cast<int>(std::max<uint32_t>(num_threads, 1)));
   index_->add(static_cast<faiss::idx_t>(point_ids_.size()), sampled_vectors.data());

   validate_mapping(num_points, num_blocks);
   const UpperHnswStats stats = statistics();
   std::cout << "[upper_hnsw][build] nodes=" << stats.num_nodes
             << " covered_blocks=" << stats.covered_blocks
             << " max_level=" << stats.max_level
             << " reachable_level0=" << stats.reachable_level0_nodes
             << std::endl;
}

void UpperHnswIndex::save(const std::string &prefix,
                          IdxType base_num_points,
                          IdxType num_blocks) const
{
   if (empty())
      throw std::runtime_error("cannot save an empty upper HNSW index");
   validate_mapping(base_num_points, num_blocks);
   faiss::write_index(index_.get(), (prefix + index_filename()).c_str());

   std::ofstream out(prefix + mapping_filename(), std::ios::binary | std::ios::trunc);
   if (!out)
      throw std::runtime_error("cannot create upper-HNSW mapping sidecar");
   out.write(kMappingMagic.data(), static_cast<std::streamsize>(kMappingMagic.size()));
   write_value(out, kMappingVersion);
   write_value(out, base_num_points);
   write_value(out, num_blocks);
   const uint64_t count = point_ids_.size();
   write_value(out, count);
   out.write(reinterpret_cast<const char *>(point_ids_.data()),
             static_cast<std::streamsize>(point_ids_.size() * sizeof(IdxType)));
   out.write(reinterpret_cast<const char *>(block_ids_.data()),
             static_cast<std::streamsize>(block_ids_.size() * sizeof(IdxType)));
   if (!out)
      throw std::runtime_error("failed to write upper-HNSW mapping payload");
}

void UpperHnswIndex::load(const std::string &prefix,
                          IdxType expected_base_num_points,
                          IdxType expected_num_blocks)
{
   std::ifstream in(prefix + mapping_filename(), std::ios::binary);
   if (!in)
      throw std::runtime_error("missing upper-HNSW mapping sidecar");
   std::array<char, 8> magic{};
   in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
   if (!in || magic != kMappingMagic)
      throw std::runtime_error("invalid upper-HNSW mapping magic");
   uint32_t version = 0;
   IdxType base_num_points = 0;
   IdxType num_blocks = 0;
   uint64_t count = 0;
   read_value(in, version);
   read_value(in, base_num_points);
   read_value(in, num_blocks);
   read_value(in, count);
   if (version != kMappingVersion || base_num_points != expected_base_num_points ||
       num_blocks != expected_num_blocks || count > expected_base_num_points)
      throw std::runtime_error("upper-HNSW mapping metadata mismatch");
   point_ids_.resize(static_cast<size_t>(count));
   block_ids_.resize(static_cast<size_t>(count));
   in.read(reinterpret_cast<char *>(point_ids_.data()),
           static_cast<std::streamsize>(point_ids_.size() * sizeof(IdxType)));
   in.read(reinterpret_cast<char *>(block_ids_.data()),
           static_cast<std::streamsize>(block_ids_.size() * sizeof(IdxType)));
   if (!in)
      throw std::runtime_error("truncated upper-HNSW mapping payload");

   std::unique_ptr<faiss::Index> loaded(
       faiss::read_index((prefix + index_filename()).c_str()));
   auto *hnsw = dynamic_cast<faiss::IndexHNSWFlat *>(loaded.get());
   if (hnsw == nullptr)
      throw std::runtime_error("upper-HNSW FAISS sidecar has the wrong index type");
   loaded.release();
   index_.reset(hnsw);
   validate_mapping(expected_base_num_points, expected_num_blocks);
}

std::vector<UpperHnswCandidate> UpperHnswIndex::search(
    const float *query, size_t top_n, size_t ef_search) const
{
   if (empty() || query == nullptr || top_n == 0)
      return {};
   top_n = std::min(top_n, point_ids_.size());
   std::vector<float> distances(top_n, std::numeric_limits<float>::infinity());
   std::vector<faiss::idx_t> labels(top_n, -1);
   faiss::SearchParametersHNSW parameters;
   parameters.efSearch = static_cast<int>(std::max(top_n, ef_search));
   index_->search(1, query, static_cast<faiss::idx_t>(top_n),
                  distances.data(), labels.data(), &parameters);

   std::vector<UpperHnswCandidate> candidates;
   candidates.reserve(top_n);
   for (size_t rank = 0; rank < top_n; ++rank)
   {
      if (labels[rank] < 0 || static_cast<size_t>(labels[rank]) >= point_ids_.size())
         continue;
      const size_t local_id = static_cast<size_t>(labels[rank]);
      candidates.push_back(
          UpperHnswCandidate{point_ids_[local_id], block_ids_[local_id], distances[rank]});
   }
   return candidates;
}

UpperHnswStats UpperHnswIndex::statistics() const
{
   UpperHnswStats stats;
   if (empty())
      return stats;
   const faiss::HNSW &hnsw = index_->hnsw;
   stats.num_nodes = point_ids_.size();
   stats.max_level = hnsw.max_level;
   const size_t layer_count = static_cast<size_t>(std::max(0, hnsw.max_level) + 1);
   stats.nodes_per_level.assign(layer_count, 0);
   stats.average_degree_per_level.assign(layer_count, 0.0);
   stats.max_degree_per_level.assign(layer_count, 0);
   std::vector<uint64_t> degree_sum(layer_count, 0);
   std::vector<uint8_t> covered;
   IdxType max_block = 0;
   for (IdxType block_id : block_ids_)
      max_block = std::max(max_block, block_id);
   covered.assign(static_cast<size_t>(max_block) + 1, 0);

   for (size_t node = 0; node < point_ids_.size(); ++node)
   {
      if (block_ids_[node] < covered.size())
         covered[block_ids_[node]] = 1;
      const int node_levels = hnsw.levels[node];
      for (int layer = 0; layer < node_levels; ++layer)
      {
         if (static_cast<size_t>(layer) >= layer_count)
            break;
         size_t begin = 0;
         size_t end = 0;
         hnsw.neighbor_range(static_cast<faiss::idx_t>(node), layer, &begin, &end);
         size_t degree = 0;
         for (size_t offset = begin; offset < end; ++offset)
            if (hnsw.neighbors[offset] >= 0)
               ++degree;
         ++stats.nodes_per_level[static_cast<size_t>(layer)];
         degree_sum[static_cast<size_t>(layer)] += degree;
         stats.max_degree_per_level[static_cast<size_t>(layer)] =
             std::max(stats.max_degree_per_level[static_cast<size_t>(layer)], degree);
      }
   }
   stats.covered_blocks = static_cast<size_t>(
       std::count(covered.begin(), covered.end(), uint8_t{1}));
   for (size_t layer = 0; layer < layer_count; ++layer)
      if (stats.nodes_per_level[layer] > 0)
         stats.average_degree_per_level[layer] =
             static_cast<double>(degree_sum[layer]) / stats.nodes_per_level[layer];

   if (hnsw.entry_point >= 0 &&
       static_cast<size_t>(hnsw.entry_point) < point_ids_.size())
   {
      std::vector<uint8_t> visited(point_ids_.size(), 0);
      std::deque<size_t> frontier;
      frontier.push_back(static_cast<size_t>(hnsw.entry_point));
      visited[static_cast<size_t>(hnsw.entry_point)] = 1;
      while (!frontier.empty())
      {
         const size_t node = frontier.front();
         frontier.pop_front();
         ++stats.reachable_level0_nodes;
         size_t begin = 0;
         size_t end = 0;
         hnsw.neighbor_range(static_cast<faiss::idx_t>(node), 0, &begin, &end);
         for (size_t offset = begin; offset < end; ++offset)
         {
            const auto neighbor = hnsw.neighbors[offset];
            if (neighbor < 0 || static_cast<size_t>(neighbor) >= visited.size() ||
                visited[static_cast<size_t>(neighbor)] != 0)
               continue;
            visited[static_cast<size_t>(neighbor)] = 1;
            frontier.push_back(static_cast<size_t>(neighbor));
         }
      }
   }
   return stats;
}

bool UpperHnswIndex::empty() const
{
   return !index_ || point_ids_.empty();
}

size_t UpperHnswIndex::size() const
{
   return point_ids_.size();
}

void UpperHnswIndex::validate_mapping(IdxType base_num_points,
                                      IdxType num_blocks) const
{
   if (!index_ || point_ids_.size() != block_ids_.size() ||
       index_->ntotal != static_cast<faiss::idx_t>(point_ids_.size()))
      throw std::runtime_error("upper-HNSW index/mapping size mismatch");
   for (size_t local_id = 0; local_id < point_ids_.size(); ++local_id)
   {
      if (point_ids_[local_id] >= base_num_points || block_ids_[local_id] == 0 ||
          block_ids_[local_id] > num_blocks)
         throw std::runtime_error("upper-HNSW mapping contains an out-of-range id");
   }
}

} // namespace ANNS
