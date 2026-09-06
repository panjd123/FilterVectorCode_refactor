#include "include/uni_nav_graph.h"
#include "include/utils.h"
#include "include/ung_lng_block_partition.h"

#include "include/tagore_graph_builder.h"
#include "include/ung_build_settings.h"
#include "include/ung_prof_log.h"
#include "include/ung_special_edge_io.h"
#include "vamana/vamana.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cerrno>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <unordered_map>
#include <type_traits>
#include <utility>

#include <omp.h>

namespace ANNS
{
namespace
{

std::string block_source_fingerprint(
    IdxType num_points,
    IdxType num_groups,
    const std::vector<std::vector<LabelType>> &group_labels,
    const std::vector<std::pair<IdxType, IdxType>> &group_ranges)
{
   uint64_t hash = 1469598103934665603ULL;
   auto append = [&](uint64_t value) {
      for (unsigned shift = 0; shift < 64; shift += 8)
      {
         hash ^= (value >> shift) & 0xffU;
         hash *= 1099511628211ULL;
      }
   };
   append(num_points);
   append(num_groups);
   for (IdxType group_id = 1; group_id <= num_groups; ++group_id)
   {
      if (group_id >= group_labels.size() || group_id >= group_ranges.size())
         throw std::runtime_error("cannot fingerprint incomplete UNG group metadata");
      append(group_id);
      append(group_ranges[group_id].first);
      append(group_ranges[group_id].second);
      append(group_labels[group_id].size());
      for (LabelType label : group_labels[group_id])
         append(label);
   }
   std::ostringstream out;
   out << std::hex << std::setw(16) << std::setfill('0') << hash;
   return out.str();
}

uint64_t special_block_disk_bytes(const std::string &prefix)
{
   static const char *files[] = {
       "meta",
       "special_blocks.bin",
       "special_blocks.csv",
       "special_block_members.csv",
       "special_block_children.csv",
       "special_block_trie.bin",
       "special_trie_regular_edges.bin",
       "special_edges.bin",
       "special_edges.csv",
       "special_heavy_edges.bin",
       "special_heavy_edges.csv",
   };
   uint64_t total = 0;
   for (const char *name : files)
   {
      std::error_code ec;
      const auto size = std::filesystem::file_size(prefix + name, ec);
      if (!ec)
         total += size;
   }
   return total;
}

constexpr char kSpecialBlockMetadataMagicV2[8] = {'S', 'B', 'L', 'K', '0', '0', '2', '\0'};
constexpr char kSpecialBlockMetadataMagicV3[8] = {'S', 'B', 'L', 'K', '0', '0', '3', '\0'};
constexpr uint32_t kSpecialBlockMetadataVersion = 3;

bool validate_special_block_metadata_impl(
    const std::vector<SpecialBlock> &blocks,
    std::string &error)
{
   std::unordered_map<IdxType, IdxType> middle_group_owner;
   std::unordered_map<IdxType, IdxType> upper_group_owner;
   std::vector<IdxType> child_parent(blocks.size() + 1, 0);
   for (size_t index = 0; index < blocks.size(); ++index)
   {
      const SpecialBlock &block = blocks[index];
      if (block.block_id != index + 1)
      {
         error = "special block ids must be dense and one-based";
         return false;
      }
      if (block.level > 1)
      {
         error = "special block metadata supports only middle level 0 and upper level 1";
         return false;
      }
      if (block.member_group_ids.empty())
      {
         error = "special block must contain at least one direct member group";
         return false;
      }
      if (block.root_labels.empty())
      {
         error = "special block root label path cannot be empty";
         return false;
      }
      if (block.point_count == 0 || block.subtree_point_count < block.point_count)
      {
         error = "special block point counts are inconsistent";
         return false;
      }
      if (block.root_group_id == 0 ||
          std::find(block.member_group_ids.begin(), block.member_group_ids.end(),
                    block.root_group_id) == block.member_group_ids.end())
      {
         error = "special block root group must be a direct member";
         return false;
      }
      if (block.level == 1 && block.parent_block_id != 0)
      {
         error = "upper special block cannot have a parent block";
         return false;
      }
      if (block.parent_block_id != 0)
      {
         if (block.level != 0 || block.parent_block_id > blocks.size() ||
             blocks[block.parent_block_id - 1].level != 1)
         {
            error = "middle special block parent must reference an upper block";
            return false;
         }
      }
      auto &owners = block.level == 0 ? middle_group_owner : upper_group_owner;
      for (IdxType group_id : block.member_group_ids)
      {
         if (group_id == 0 || !owners.emplace(group_id, block.block_id).second)
         {
            error = "a group may belong to at most one special block per layer";
            return false;
         }
      }
      for (IdxType child_id : block.child_block_ids)
      {
         if (child_id == 0 || child_id > blocks.size() || child_id == block.block_id ||
             blocks[child_id - 1].level != block.level)
         {
            error = "special block child must reference a distinct block in the same layer";
            return false;
         }
         if (child_parent[child_id] != 0)
         {
            error = "a special block may have at most one same-layer parent";
            return false;
         }
         child_parent[child_id] = block.block_id;
      }
   }

   // A valid layer is a forest.  Unique parents alone do not exclude a cycle
   // (for example 1->2->1), so walk the parent chain from every block.
   std::vector<uint8_t> visit(blocks.size() + 1, 0);
   for (IdxType start = 1; start <= blocks.size(); ++start)
   {
      IdxType current = start;
      while (current != 0 && visit[current] == 0)
      {
         visit[current] = 1;
         current = child_parent[current];
      }
      if (current != 0 && visit[current] == 1)
      {
         error = "special block child topology contains a cycle";
         return false;
      }
      current = start;
      while (current != 0 && visit[current] == 1)
      {
         visit[current] = 2;
         current = child_parent[current];
      }
   }
   error.clear();
   return true;
}

bool is_label_path_prefix(const std::vector<LabelType> &prefix,
                          const std::vector<LabelType> &path,
                          bool require_strict)
{
   if (prefix.size() > path.size() || (require_strict && prefix.size() == path.size()))
      return false;
   return std::equal(prefix.begin(), prefix.end(), path.begin());
}

bool validate_special_block_graph_semantics_impl(
    const std::vector<SpecialBlock> &blocks,
    IdxType num_points,
    IdxType num_groups,
    const std::vector<std::vector<LabelType>> &group_labels,
    const std::vector<std::pair<IdxType, IdxType>> &group_ranges,
    const std::vector<IdxType> &point_to_group,
    std::string &error)
{
   // An empty source graph has no one-based group sentinel to validate. It is
   // still a well-formed degenerate input when it contains no points, groups,
   // blocks, or point ownership. Keeping this case explicit also lets bundle
   // preflight exercise the same production loader without synthetic graph
   // state.
   if (num_points == 0 && num_groups == 0 && blocks.empty() &&
       point_to_group.empty())
   {
      error.clear();
      return true;
   }
   if (group_labels.size() <= num_groups || group_ranges.size() <= num_groups ||
       point_to_group.size() != num_points)
   {
      error = "source UNG group metadata is incomplete";
      return false;
   }

   for (const SpecialBlock &block : blocks)
   {
      uint64_t actual_point_count = 0;
      for (IdxType group_id : block.member_group_ids)
      {
         if (group_id == 0 || group_id > num_groups)
         {
            error = "special block contains an invalid direct member group";
            return false;
         }
         const auto &range = group_ranges[group_id];
         if (range.first > range.second || range.second > num_points)
         {
            error = "special block direct member has an invalid point range";
            return false;
         }
         if (!is_label_path_prefix(block.root_labels, group_labels[group_id], false))
         {
            error = "special block root labels do not contain every direct member group";
            return false;
         }
         actual_point_count += static_cast<uint64_t>(range.second - range.first);
      }
      if (actual_point_count != block.point_count)
      {
         error = "special block point_count does not match its direct member ranges";
         return false;
      }
      if (block.entry_point_id == SpecialBlock::kInvalidEntryPoint ||
          block.entry_point_id >= num_points ||
          std::find(block.member_group_ids.begin(), block.member_group_ids.end(),
                    point_to_group[block.entry_point_id]) == block.member_group_ids.end())
      {
         error = "special block entry point is not in a direct member group";
         return false;
      }

      for (IdxType child_id : block.child_block_ids)
      {
         const SpecialBlock &child = blocks[child_id - 1];
         if (!is_label_path_prefix(block.root_labels, child.root_labels, true))
         {
            error = "same-layer child root is not a strict trie descendant";
            return false;
         }
      }
      if (block.level == 0 && block.parent_block_id != 0)
      {
         const SpecialBlock &upper = blocks[block.parent_block_id - 1];
         if (!is_label_path_prefix(upper.root_labels, block.root_labels, false))
         {
            error = "upper block does not contain its middle-layer member block";
            return false;
         }
      }
   }
   error.clear();
   return true;
}

void write_special_block_metadata_binary_impl(const std::string &path,
                                              const std::vector<SpecialBlock> &blocks)
{
   std::string validation_error;
   if (!validate_special_block_metadata_impl(blocks, validation_error))
      throw std::runtime_error("invalid special block metadata: " + validation_error);
   const std::string temporary_path = path + ".tmp";
   std::ofstream out(temporary_path, std::ios::binary | std::ios::trunc);
   if (!out)
      throw std::runtime_error("cannot open special block metadata output: " + temporary_path);
   const uint64_t count = blocks.size();
   out.write(kSpecialBlockMetadataMagicV3, sizeof(kSpecialBlockMetadataMagicV3));
   out.write(reinterpret_cast<const char *>(&kSpecialBlockMetadataVersion), sizeof(uint32_t));
   const uint32_t reserved = 0;
   out.write(reinterpret_cast<const char *>(&reserved), sizeof(reserved));
   out.write(reinterpret_cast<const char *>(&count), sizeof(count));
   for (const auto &block : blocks)
   {
      const uint32_t fixed[] = {
          static_cast<uint32_t>(block.block_id),
          static_cast<uint32_t>(block.level),
          static_cast<uint32_t>(block.parent_block_id),
          static_cast<uint32_t>(block.root_group_id),
          static_cast<uint32_t>(block.entry_point_id),
          static_cast<uint32_t>(block.point_count),
          static_cast<uint32_t>(block.subtree_point_count),
          static_cast<uint32_t>(block.root_labels.size()),
          static_cast<uint32_t>(block.common_labels.size()),
          static_cast<uint32_t>(block.member_group_ids.size()),
          static_cast<uint32_t>(block.child_block_ids.size()),
      };
      out.write(reinterpret_cast<const char *>(fixed), sizeof(fixed));
      const auto write_values = [&](const auto &values) {
         for (const auto value : values)
         {
            const uint32_t encoded = static_cast<uint32_t>(value);
            out.write(reinterpret_cast<const char *>(&encoded), sizeof(encoded));
         }
      };
      write_values(block.root_labels);
      write_values(block.common_labels);
      write_values(block.member_group_ids);
      write_values(block.child_block_ids);
   }
   out.flush();
   if (!out)
      throw std::runtime_error("failed while writing special block metadata: " + temporary_path);
   out.close();
   if (std::rename(temporary_path.c_str(), path.c_str()) != 0)
   {
      std::remove(temporary_path.c_str());
      throw std::runtime_error("cannot publish special block metadata: " + path);
   }
}

bool read_special_block_metadata_binary_impl(const std::string &path,
                                             std::vector<SpecialBlock> &blocks,
                                             std::string &error)
{
   std::ifstream in(path, std::ios::binary);
   if (!in)
      return false;
   char magic[8];
   uint32_t version = 0;
   uint32_t reserved = 0;
   uint64_t count = 0;
   in.read(magic, sizeof(magic));
   in.read(reinterpret_cast<char *>(&version), sizeof(version));
   in.read(reinterpret_cast<char *>(&reserved), sizeof(reserved));
   in.read(reinterpret_cast<char *>(&count), sizeof(count));
   const bool version2 = version == 2 &&
                         std::equal(std::begin(magic), std::end(magic),
                                    std::begin(kSpecialBlockMetadataMagicV2));
   const bool version3 = version == 3 &&
                         std::equal(std::begin(magic), std::end(magic),
                                    std::begin(kSpecialBlockMetadataMagicV3));
   if (!in || (!version2 && !version3) || count > UINT32_MAX)
   {
      error = "invalid special block metadata header";
      return false;
   }
   blocks.clear();
   blocks.reserve(static_cast<size_t>(count));
   for (uint64_t index = 0; index < count; ++index)
   {
      uint32_t fixed[11] = {};
      const size_t fixed_count = version3 ? 11 : 9;
      in.read(reinterpret_cast<char *>(fixed), fixed_count * sizeof(uint32_t));
      if (!in || fixed[0] != index + 1)
      {
         error = "invalid special block metadata record";
         return false;
      }
      SpecialBlock block;
      block.block_id = fixed[0];
      const size_t offset = version3 ? 2 : 0;
      if (version3)
      {
         block.level = static_cast<uint8_t>(fixed[1]);
         block.parent_block_id = fixed[2];
      }
      block.root_group_id = fixed[1 + offset];
      block.entry_point_id = fixed[2 + offset];
      block.point_count = fixed[3 + offset];
      block.subtree_point_count = fixed[4 + offset];
      const auto read_values = [&](uint32_t value_count, auto &values) {
         values.resize(value_count);
         for (uint32_t i = 0; i < value_count; ++i)
         {
            uint32_t encoded = 0;
            in.read(reinterpret_cast<char *>(&encoded), sizeof(encoded));
            values[i] = static_cast<typename std::decay_t<decltype(values)>::value_type>(encoded);
         }
      };
      read_values(fixed[5 + offset], block.root_labels);
      read_values(fixed[6 + offset], block.common_labels);
      read_values(fixed[7 + offset], block.member_group_ids);
      read_values(fixed[8 + offset], block.child_block_ids);
      if (!in)
      {
         error = "truncated special block metadata payload";
         return false;
      }
      blocks.push_back(std::move(block));
   }
   char trailing = 0;
   if (in.read(&trailing, 1))
   {
      error = "special block metadata has trailing bytes";
      return false;
   }
   if (!validate_special_block_metadata_impl(blocks, error))
      return false;
   error.clear();
   return true;
}

IdxType read_env_idxtype(const char *key, IdxType fallback, IdxType min_v, IdxType max_v)
{
   const char *s = std::getenv(key);
   if (!s || !*s)
      return fallback;
   errno = 0;
   char *end = nullptr;
   unsigned long long v = std::strtoull(s, &end, 10);
   if (errno != 0 || end == s || *end != '\0')
      return fallback;
   unsigned long long lo = static_cast<unsigned long long>(min_v);
   unsigned long long hi = static_cast<unsigned long long>(max_v);
   if (v < lo)
      v = lo;
   if (v > hi)
      v = hi;
   return static_cast<IdxType>(v);
}

bool read_env_bool_default(const char *key, bool fallback)
{
   const char *s = std::getenv(key);
   if (!s || !*s)
      return fallback;
   return std::atoi(s) != 0;
}

std::string read_env_string(const char *key, const std::string &fallback)
{
   const char *s = std::getenv(key);
   if (!s || !*s)
      return fallback;
   return std::string(s);
}

void build_exact_topk_graph_for_points(std::shared_ptr<Graph> graph,
                                       const std::shared_ptr<IStorage> &storage,
                                       const std::shared_ptr<DistanceHandler> &distance_handler,
                                       const std::vector<IdxType> &points,
                                       IdxType max_degree)
{
   const IdxType n = static_cast<IdxType>(points.size());
   if (n <= 1)
      return;
   const IdxType degree = std::min<IdxType>(max_degree, n - 1);
   const IdxType dim = storage->get_dim();
   std::vector<std::pair<float, IdxType>> candidates;
   candidates.reserve(n > 0 ? n - 1 : 0);
   for (IdxType i = 0; i < n; ++i)
   {
      candidates.clear();
      const char *src = storage->get_vector(points[i]);
      for (IdxType j = 0; j < n; ++j)
      {
         if (i == j)
            continue;
         const float dist = distance_handler->compute(src, storage->get_vector(points[j]), dim);
         candidates.emplace_back(dist, j);
      }
      if (static_cast<IdxType>(candidates.size()) > degree)
      {
         std::nth_element(candidates.begin(), candidates.begin() + degree, candidates.end(),
                          [](const auto &a, const auto &b) {
                             return a.first < b.first || (a.first == b.first && a.second < b.second);
                          });
         candidates.resize(degree);
      }
      std::sort(candidates.begin(), candidates.end(),
                [](const auto &a, const auto &b) {
                   return a.first < b.first || (a.first == b.first && a.second < b.second);
                });
      auto &neighbors = graph->neighbors[i];
      neighbors.resize(candidates.size());
      for (size_t k = 0; k < candidates.size(); ++k)
         neighbors[k] = candidates[k].second;
   }
}

void build_sampled_topk_graph_for_points(std::shared_ptr<Graph> graph,
                                         const std::shared_ptr<IStorage> &storage,
                                         const std::shared_ptr<DistanceHandler> &distance_handler,
                                         const std::vector<IdxType> &points,
                                         IdxType max_degree,
                                         size_t requested_candidates)
{
   const IdxType n = static_cast<IdxType>(points.size());
   if (n <= 1)
      return;
   if (requested_candidates == 0 || requested_candidates >= static_cast<size_t>(n - 1))
   {
      build_exact_topk_graph_for_points(graph, storage, distance_handler, points, max_degree);
      return;
   }

   const IdxType degree = std::min<IdxType>(max_degree, static_cast<IdxType>(requested_candidates));
   const IdxType dim = storage->get_dim();
   std::vector<std::pair<float, IdxType>> candidates;
   std::vector<IdxType> sampled_local_ids;
   candidates.reserve(requested_candidates);
   sampled_local_ids.reserve(requested_candidates);

   for (IdxType i = 0; i < n; ++i)
   {
      candidates.clear();
      sampled_local_ids.clear();
      for (size_t t = 0; t < requested_candidates && sampled_local_ids.size() < requested_candidates; ++t)
      {
         IdxType local_id = static_cast<IdxType>((t * static_cast<size_t>(n)) / requested_candidates);
         local_id = (local_id + i + 1) % n;
         if (local_id == i)
            local_id = (local_id + 1) % n;
         if (local_id == i ||
             std::find(sampled_local_ids.begin(), sampled_local_ids.end(), local_id) != sampled_local_ids.end())
            continue;
         sampled_local_ids.push_back(local_id);
      }

      for (IdxType fallback = 0; sampled_local_ids.size() < requested_candidates && fallback < n; ++fallback)
      {
         if (fallback == i ||
             std::find(sampled_local_ids.begin(), sampled_local_ids.end(), fallback) != sampled_local_ids.end())
            continue;
         sampled_local_ids.push_back(fallback);
      }

      const char *src = storage->get_vector(points[i]);
      for (IdxType local_id : sampled_local_ids)
      {
         const float dist = distance_handler->compute(src, storage->get_vector(points[local_id]), dim);
         candidates.emplace_back(dist, local_id);
      }
      if (static_cast<IdxType>(candidates.size()) > degree)
      {
         std::nth_element(candidates.begin(), candidates.begin() + degree, candidates.end(),
                          [](const auto &a, const auto &b) {
                             return a.first < b.first || (a.first == b.first && a.second < b.second);
                          });
         candidates.resize(degree);
      }
      std::sort(candidates.begin(), candidates.end(),
                [](const auto &a, const auto &b) {
                   return a.first < b.first || (a.first == b.first && a.second < b.second);
                });
      auto &neighbors = graph->neighbors[i];
      neighbors.resize(candidates.size());
      for (size_t k = 0; k < candidates.size(); ++k)
         neighbors[k] = candidates[k].second;
   }
}

std::vector<IdxType> sample_local_candidate_ids(IdxType source_id, IdxType n, size_t requested_candidates)
{
   std::vector<IdxType> sampled_local_ids;
   if (n <= 1 || requested_candidates == 0)
      return sampled_local_ids;
   requested_candidates = std::min<size_t>(requested_candidates, static_cast<size_t>(n - 1));
   sampled_local_ids.reserve(requested_candidates);

   for (size_t t = 0; t < requested_candidates && sampled_local_ids.size() < requested_candidates; ++t)
   {
      IdxType local_id = static_cast<IdxType>((t * static_cast<size_t>(n)) / requested_candidates);
      local_id = (local_id + source_id + 1) % n;
      if (local_id == source_id)
         local_id = (local_id + 1) % n;
      if (local_id == source_id ||
          std::find(sampled_local_ids.begin(), sampled_local_ids.end(), local_id) != sampled_local_ids.end())
         continue;
      sampled_local_ids.push_back(local_id);
   }

   for (IdxType fallback = 0; sampled_local_ids.size() < requested_candidates && fallback < n; ++fallback)
   {
      if (fallback == source_id ||
          std::find(sampled_local_ids.begin(), sampled_local_ids.end(), fallback) != sampled_local_ids.end())
         continue;
      sampled_local_ids.push_back(fallback);
   }
   return sampled_local_ids;
}

std::shared_ptr<Vamana> build_sampled_vamana_graph_for_points(std::shared_ptr<Graph> graph,
                                                              const std::shared_ptr<IStorage> &local_storage,
                                                              const std::shared_ptr<DistanceHandler> &distance_handler,
                                                              IdxType max_degree,
                                                              IdxType Lbuild,
                                                              float alpha,
                                                              uint32_t num_threads,
                                                              IdxType max_candidate_size,
                                                              size_t requested_candidates)
{
   const IdxType n = local_storage->get_num_points();
   auto local_index = std::make_shared<Vamana>(false);
   if (n <= 1)
      return local_index;

   requested_candidates = std::min<size_t>(std::max<size_t>(requested_candidates, static_cast<size_t>(max_degree)),
                                           static_cast<size_t>(n - 1));
   const IdxType dim = local_storage->get_dim();
   std::vector<std::vector<Candidate>> candidate_pools(n);

   omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
   for (IdxType source_id = 0; source_id < n; ++source_id)
   {
      const auto sampled_local_ids = sample_local_candidate_ids(source_id, n, requested_candidates);
      auto &candidates = candidate_pools[source_id];
      candidates.reserve(sampled_local_ids.size());
      const char *src = local_storage->get_vector(source_id);
      for (IdxType local_id : sampled_local_ids)
      {
         const float dist = distance_handler->compute(src, local_storage->get_vector(local_id), dim);
         candidates.emplace_back(local_id, dist);
      }
   }

   local_index->build_from_candidate_pools(local_storage, distance_handler, graph, candidate_pools,
                                           max_degree, Lbuild, alpha, num_threads, max_candidate_size);
   return local_index;
}

uint64_t mix_hash64(uint64_t x)
{
   x += 0x9e3779b97f4a7c15ULL;
   x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
   x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
   return x ^ (x >> 31);
}

float hash_projection_value(const std::shared_ptr<IStorage> &storage, IdxType local_id, uint64_t seed)
{
   const IdxType dim = storage->get_dim();
   if (storage->get_data_type() != DataType::FLOAT)
      return static_cast<float>(mix_hash64(seed ^ static_cast<uint64_t>(local_id)) & 0xffffffULL);
   const float *vec = reinterpret_cast<const float *>(storage->get_vector(local_id));
   float acc = 0.0f;
   for (IdxType d = 0; d < dim; ++d)
   {
      const uint64_t h = mix_hash64(seed ^ (static_cast<uint64_t>(d) * 0x9e3779b185ebca87ULL));
      acc += (h & 1ULL) ? vec[d] : -vec[d];
   }
   return acc;
}

uint32_t hashprune_direction_hash(const std::shared_ptr<IStorage> &storage,
                                  IdxType src,
                                  IdxType dst,
                                  uint32_t bits,
                                  IdxType sampled_dims,
                                  uint64_t seed)
{
   bits = std::min<uint32_t>(bits, 24);
   if (bits == 0)
      return 0;
   if (storage->get_data_type() != DataType::FLOAT)
      return static_cast<uint32_t>(mix_hash64(seed ^ (static_cast<uint64_t>(src) << 32) ^ dst));
   const IdxType dim = storage->get_dim();
   sampled_dims = std::min<IdxType>(std::max<IdxType>(sampled_dims, 1), dim);
   const float *src_vec = reinterpret_cast<const float *>(storage->get_vector(src));
   const float *dst_vec = reinterpret_cast<const float *>(storage->get_vector(dst));
   uint32_t out = 0;
   for (uint32_t bit = 0; bit < bits; ++bit)
   {
      float dot = 0.0f;
      const uint64_t bit_seed = seed ^ (static_cast<uint64_t>(bit) * 0xd6e8feb86659fd93ULL);
      for (IdxType sample = 0; sample < sampled_dims; ++sample)
      {
         const uint64_t dim_hash = mix_hash64(bit_seed ^ (static_cast<uint64_t>(sample) * 0x9e3779b185ebca87ULL));
         const IdxType d = static_cast<IdxType>(dim_hash % static_cast<uint64_t>(dim));
         dot += (dim_hash & 1ULL) ? (dst_vec[d] - src_vec[d]) : (src_vec[d] - dst_vec[d]);
      }
      if (dot >= 0.0f)
         out |= (1U << bit);
   }
   return out;
}

struct HashPruneGraphDiagnostics
{
   unsigned long long points = 0;
   unsigned long long candidate_count_sum = 0;
   unsigned long long bucket_coverage_sum = 0;
   unsigned long long repaired_points = 0;
   unsigned long long repair_candidates_added = 0;
   unsigned long long low_2hop_points = 0;
   unsigned long long graph_repair_candidates_added = 0;
   unsigned long long hash_repair_candidates_added = 0;
   unsigned long long reservoir_repair_candidates_added = 0;
   unsigned long long sample_repair_candidates_added = 0;
   unsigned long long final_degree_sum = 0;
   unsigned long long low_degree_lt16 = 0;
   unsigned long long low_degree_lt32 = 0;
   unsigned long long full_degree_points = 0;
   IdxType min_degree = std::numeric_limits<IdxType>::max();
   IdxType max_degree = 0;
};

struct HashPruneReservoir
{
   HashPruneReservoir(size_t cap = 0, size_t per_bucket = 1, size_t nearest = 0)
       : capacity(cap), bucket_capacity(std::max<size_t>(per_bucket, 1)), nearest_capacity(std::min(nearest, cap)) {}

   void offer(IdxType id, float distance, uint32_t bucket)
   {
      if (capacity == 0)
         return;
      offer_to_nearest(id, distance, bucket);
      offer_to_diversity(id, distance, bucket);
   }

   std::vector<Candidate> to_candidates() const
   {
      std::vector<Entry> merged;
      merged.reserve(nearest_entries.size() + diversity_entries.size());
      auto add_or_update = [&](const Entry &entry) {
         for (auto &cur : merged)
            if (cur.id == entry.id)
            {
               if (entry.distance < cur.distance)
                  cur = entry;
               return;
            }
         merged.push_back(entry);
      };
      for (const auto &entry : nearest_entries)
         add_or_update(entry);
      for (const auto &entry : diversity_entries)
         add_or_update(entry);

      std::vector<Candidate> out;
      out.reserve(merged.size());
      for (const auto &entry : merged)
         out.emplace_back(entry.id, entry.distance);
      return out;
   }

   size_t candidate_count() const
   {
      return to_candidates().size();
   }

   size_t bucket_coverage() const
   {
      std::vector<uint32_t> buckets;
      buckets.reserve(nearest_entries.size() + diversity_entries.size());
      for (const auto &entry : nearest_entries)
         buckets.push_back(entry.bucket);
      for (const auto &entry : diversity_entries)
         buckets.push_back(entry.bucket);
      std::sort(buckets.begin(), buckets.end());
      buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());
      return buckets.size();
   }

   struct Entry
   {
      IdxType id = 0;
      float distance = 0.0f;
      uint32_t bucket = 0;
   };

   size_t capacity = 0;
   size_t bucket_capacity = 1;
   size_t nearest_capacity = 0;
   std::vector<Entry> nearest_entries;
   std::vector<Entry> diversity_entries;

private:
   void offer_to_nearest(IdxType id, float distance, uint32_t bucket)
   {
      if (nearest_capacity == 0)
         return;
      for (auto &entry : nearest_entries)
      {
         if (entry.id == id)
         {
            if (distance < entry.distance)
               entry = Entry{id, distance, bucket};
            return;
         }
      }
      if (nearest_entries.size() < nearest_capacity)
      {
         nearest_entries.push_back(Entry{id, distance, bucket});
         return;
      }
      auto worst = std::max_element(nearest_entries.begin(), nearest_entries.end(),
                                    [](const Entry &a, const Entry &b) { return a.distance < b.distance; });
      if (worst != nearest_entries.end() && distance < worst->distance)
         *worst = Entry{id, distance, bucket};
   }

   void offer_to_diversity(IdxType id, float distance, uint32_t bucket)
   {
      const size_t diversity_capacity = capacity > nearest_capacity ? capacity - nearest_capacity : 0;
      if (diversity_capacity == 0)
         return;
      for (auto &entry : diversity_entries)
      {
         if (entry.id == id)
         {
            if (distance < entry.distance)
               entry = Entry{id, distance, bucket};
            return;
         }
      }

      size_t same_bucket_count = 0;
      auto worst_same_bucket = diversity_entries.end();
      for (auto it = diversity_entries.begin(); it != diversity_entries.end(); ++it)
      {
         if (it->bucket != bucket)
            continue;
         same_bucket_count += 1;
         if (worst_same_bucket == diversity_entries.end() || worst_same_bucket->distance < it->distance)
            worst_same_bucket = it;
      }

      if (same_bucket_count >= bucket_capacity)
      {
         if (worst_same_bucket != diversity_entries.end() && distance < worst_same_bucket->distance)
            *worst_same_bucket = Entry{id, distance, bucket};
         return;
      }

      if (diversity_entries.size() < diversity_capacity)
      {
         diversity_entries.push_back(Entry{id, distance, bucket});
         return;
      }

      auto worst = std::max_element(diversity_entries.begin(), diversity_entries.end(),
                                    [](const Entry &a, const Entry &b) { return a.distance < b.distance; });
      if (worst != diversity_entries.end() && distance < worst->distance)
         *worst = Entry{id, distance, bucket};
   }
};

void hashprune_stream_edge(const std::shared_ptr<IStorage> &storage,
                           std::vector<HashPruneReservoir> &reservoirs,
                           IdxType src,
                           IdxType dst,
                           float distance,
                           uint32_t hash_bits,
                           IdxType hash_sample_dims,
                           uint64_t seed)
{
   if (src == dst || src >= static_cast<IdxType>(reservoirs.size()))
      return;
   const uint32_t bucket = hashprune_direction_hash(storage, src, dst, hash_bits, hash_sample_dims, seed);
   reservoirs[src].offer(dst, distance, bucket);
}

size_t add_sampled_candidates_to_pool(const std::shared_ptr<IStorage> &local_storage,
                                      const std::shared_ptr<DistanceHandler> &distance_handler,
                                      IdxType source_id,
                                      size_t requested_candidates,
                                      size_t target_candidate_count,
                                      std::vector<Candidate> &pool)
{
   const IdxType n = local_storage->get_num_points();
   if (n <= 1 || requested_candidates == 0 || pool.size() >= target_candidate_count)
      return 0;

   target_candidate_count = std::min<size_t>(target_candidate_count, static_cast<size_t>(n - 1));
   const size_t before = pool.size();
   const auto sampled = sample_local_candidate_ids(source_id, n, requested_candidates);
   const char *src = local_storage->get_vector(source_id);
   const IdxType dim = local_storage->get_dim();
   for (IdxType dst : sampled)
   {
      bool exists = false;
      for (const auto &candidate : pool)
      {
         if (candidate.id == dst)
         {
            exists = true;
            break;
         }
      }
      if (!exists)
         pool.emplace_back(dst, distance_handler->compute(src, local_storage->get_vector(dst), dim));
      if (pool.size() >= target_candidate_count)
         break;
   }
   return pool.size() - before;
}

bool candidate_pool_contains(const std::vector<Candidate> &pool, IdxType id)
{
   for (const auto &candidate : pool)
      if (candidate.id == id)
         return true;
   return false;
}

size_t add_candidate_ids_to_pool(const std::shared_ptr<IStorage> &local_storage,
                                 const std::shared_ptr<DistanceHandler> &distance_handler,
                                 IdxType source_id,
                                 const std::vector<IdxType> &candidate_ids,
                                 size_t target_candidate_count,
                                 std::vector<Candidate> &pool)
{
   const IdxType n = local_storage->get_num_points();
   if (n <= 1 || pool.size() >= target_candidate_count)
      return 0;
   target_candidate_count = std::min<size_t>(target_candidate_count, static_cast<size_t>(n - 1));
   const size_t before = pool.size();
   const char *src = local_storage->get_vector(source_id);
   const IdxType dim = local_storage->get_dim();
   for (IdxType dst : candidate_ids)
   {
      if (dst == source_id || dst >= n || candidate_pool_contains(pool, dst))
         continue;
      pool.emplace_back(dst, distance_handler->compute(src, local_storage->get_vector(dst), dim));
      if (pool.size() >= target_candidate_count)
         break;
   }
   return pool.size() - before;
}

std::vector<IdxType> collect_hash_near_candidates(IdxType source_id,
                                                  const std::vector<std::vector<IdxType>> &hash_orders,
                                                  const std::vector<std::vector<IdxType>> &hash_positions,
                                                  size_t requested_candidates)
{
   std::vector<IdxType> out;
   if (requested_candidates == 0 || hash_orders.empty())
      return out;
   out.reserve(requested_candidates);
   for (size_t replica = 0; replica < hash_orders.size() && out.size() < requested_candidates; ++replica)
   {
      if (source_id >= static_cast<IdxType>(hash_positions[replica].size()))
         continue;
      const auto &order = hash_orders[replica];
      const IdxType pos = hash_positions[replica][source_id];
      for (size_t radius = 1; out.size() < requested_candidates && radius < order.size(); ++radius)
      {
         bool checked = false;
         if (static_cast<size_t>(pos) >= radius)
         {
            const IdxType cand = order[static_cast<size_t>(pos) - radius];
            if (cand != source_id && std::find(out.begin(), out.end(), cand) == out.end())
               out.push_back(cand);
            checked = true;
         }
         if (static_cast<size_t>(pos) + radius < order.size())
         {
            const IdxType cand = order[static_cast<size_t>(pos) + radius];
            if (cand != source_id && std::find(out.begin(), out.end(), cand) == out.end())
               out.push_back(cand);
            checked = true;
         }
         if (!checked)
            break;
      }
   }
   return out;
}

size_t two_hop_coverage_until(const std::shared_ptr<Graph> &graph,
                              IdxType source_id,
                              IdxType n,
                              size_t enough_coverage)
{
   std::vector<IdxType> seen;
   seen.reserve(enough_coverage + 1);
   const auto &first_hop = graph->neighbors[source_id];
   for (IdxType mid : first_hop)
   {
      if (mid == source_id || mid >= n)
         continue;
      if (std::find(seen.begin(), seen.end(), mid) == seen.end())
         seen.push_back(mid);
      if (seen.size() >= enough_coverage)
         return seen.size();
      for (IdxType dst : graph->neighbors[mid])
      {
         if (dst == source_id)
            continue;
         if (std::find(seen.begin(), seen.end(), dst) == seen.end())
            seen.push_back(dst);
         if (seen.size() >= enough_coverage)
            return seen.size();
      }
   }
   return seen.size();
}

size_t add_graph_expansion_candidates_to_pool(const std::shared_ptr<IStorage> &local_storage,
                                              const std::shared_ptr<DistanceHandler> &distance_handler,
                                              const std::shared_ptr<Graph> &graph,
                                              IdxType source_id,
                                              IdxType n,
                                              size_t requested_candidates,
                                              size_t target_candidate_count,
                                              std::vector<Candidate> &pool)
{
   std::vector<IdxType> candidates;
   candidates.reserve(requested_candidates);
   const auto &first_hop = graph->neighbors[source_id];
   for (IdxType mid : first_hop)
   {
      if (candidates.size() >= requested_candidates)
         break;
      if (mid != source_id && std::find(candidates.begin(), candidates.end(), mid) == candidates.end())
         candidates.push_back(mid);
      if (mid >= n)
         continue;
      for (IdxType dst : graph->neighbors[mid])
      {
         if (candidates.size() >= requested_candidates)
            break;
         if (dst == source_id || std::find(candidates.begin(), candidates.end(), dst) != candidates.end())
            continue;
         candidates.push_back(dst);
      }
   }
   return add_candidate_ids_to_pool(local_storage, distance_handler, source_id, candidates,
                                    target_candidate_count, pool);
}

size_t add_neighbor_reservoir_candidates_to_pool(const std::shared_ptr<IStorage> &local_storage,
                                                 const std::shared_ptr<DistanceHandler> &distance_handler,
                                                 const std::shared_ptr<Graph> &graph,
                                                 const std::vector<HashPruneReservoir> &reservoirs,
                                                 IdxType source_id,
                                                 size_t requested_candidates,
                                                 size_t target_candidate_count,
                                                 std::vector<Candidate> &pool)
{
   std::vector<Candidate> ranked;
   const auto &first_hop = graph->neighbors[source_id];
   for (IdxType mid : first_hop)
   {
      if (mid >= static_cast<IdxType>(reservoirs.size()))
         continue;
      auto neighbor_candidates = reservoirs[mid].to_candidates();
      ranked.insert(ranked.end(), neighbor_candidates.begin(), neighbor_candidates.end());
   }
   std::sort(ranked.begin(), ranked.end(), [](const Candidate &a, const Candidate &b) {
      return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
   });
   std::vector<IdxType> candidate_ids;
   candidate_ids.reserve(requested_candidates);
   for (const auto &candidate : ranked)
   {
      if (candidate_ids.size() >= requested_candidates)
         break;
      if (candidate.id == source_id || std::find(candidate_ids.begin(), candidate_ids.end(), candidate.id) != candidate_ids.end())
         continue;
      candidate_ids.push_back(candidate.id);
   }
   return add_candidate_ids_to_pool(local_storage, distance_handler, source_id, candidate_ids,
                                    target_candidate_count, pool);
}

std::shared_ptr<Vamana> build_hashprune_graph_for_points(std::shared_ptr<Graph> graph,
                                                         const std::shared_ptr<IStorage> &local_storage,
                                                         const std::shared_ptr<DistanceHandler> &distance_handler,
                                                         IdxType max_degree,
                                                         IdxType Lbuild,
                                                         float alpha,
                                                         uint32_t num_threads,
                                                         IdxType max_candidate_size,
                                                         size_t leaf_size,
                                                         size_t replicas,
                                                         size_t reservoir_size,
                                                         size_t leaf_topk,
                                                         size_t fanout,
                                                         uint32_t hash_bits,
                                                         IdxType hash_sample_dims,
                                                         size_t bucket_capacity,
                                                         size_t nearest_capacity,
                                                         size_t mix_sample_candidates,
                                                         size_t repair_degree,
                                                         size_t repair_sample_candidates,
                                                         size_t repair_max_points,
                                                         bool nav_repair_enabled,
                                                         size_t repair_2hop_min,
                                                         size_t repair_graph_hops,
                                                         size_t repair_hash_near,
                                                         size_t repair_reservoir_near,
                                                         bool collect_diagnostics,
                                                         HashPruneGraphDiagnostics &diagnostics,
                                                         unsigned long long &hashprune_edges_streamed)
{
   const IdxType n = local_storage->get_num_points();
   auto local_index = std::make_shared<Vamana>(false);
   hashprune_edges_streamed = 0;
   if (n <= 1)
      return local_index;

   leaf_size = std::min<size_t>(std::max<size_t>(leaf_size, 2), static_cast<size_t>(n));
   replicas = std::max<size_t>(replicas, 1);
   fanout = std::max<size_t>(fanout, 1);
   reservoir_size = std::max<size_t>(reservoir_size, static_cast<size_t>(max_degree));
   leaf_topk = std::min<size_t>(std::max<size_t>(leaf_topk, static_cast<size_t>(max_degree)), leaf_size - 1);

   bucket_capacity = std::max<size_t>(bucket_capacity, 1);
   nearest_capacity = std::min<size_t>(nearest_capacity, reservoir_size);
   mix_sample_candidates = std::min<size_t>(mix_sample_candidates, static_cast<size_t>(n - 1));
   repair_degree = std::min<size_t>(repair_degree, reservoir_size);
   repair_sample_candidates = std::min<size_t>(repair_sample_candidates, static_cast<size_t>(n - 1));
   repair_2hop_min = std::min<size_t>(repair_2hop_min, static_cast<size_t>(n - 1));
   repair_graph_hops = std::min<size_t>(repair_graph_hops, static_cast<size_t>(n - 1));
   repair_hash_near = std::min<size_t>(repair_hash_near, static_cast<size_t>(n - 1));
   repair_reservoir_near = std::min<size_t>(repair_reservoir_near, static_cast<size_t>(n - 1));

   std::vector<HashPruneReservoir> reservoirs;
   reservoirs.reserve(n);
   for (IdxType i = 0; i < n; ++i)
      reservoirs.emplace_back(reservoir_size, bucket_capacity, nearest_capacity);
   std::vector<std::mutex> reservoir_locks(n);
   std::vector<std::vector<IdxType>> hash_orders;
   std::vector<std::vector<IdxType>> hash_positions;
   hash_orders.reserve(replicas);
   hash_positions.reserve(replicas);
   const IdxType dim = local_storage->get_dim();

   for (size_t replica = 0; replica < replicas; ++replica)
   {
      const uint64_t seed = mix_hash64(0x1234abcddcba4321ULL ^ replica);
      std::vector<std::pair<float, IdxType>> order;
      order.reserve(n);
      for (IdxType local_id = 0; local_id < n; ++local_id)
         order.emplace_back(hash_projection_value(local_storage, local_id, seed), local_id);
      std::sort(order.begin(), order.end(), [](const auto &a, const auto &b) {
         return a.first < b.first || (a.first == b.first && a.second < b.second);
      });
      hash_orders.emplace_back();
      hash_positions.emplace_back(static_cast<size_t>(n));
      hash_orders.back().reserve(n);
      for (size_t pos = 0; pos < order.size(); ++pos)
      {
         hash_orders.back().push_back(order[pos].second);
         hash_positions.back()[order[pos].second] = static_cast<IdxType>(pos);
      }

      for (size_t lane = 0; lane < fanout; ++lane)
      {
         const size_t shift = (lane * leaf_size) / fanout;
         omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : hashprune_edges_streamed)
         for (size_t begin = shift; begin < static_cast<size_t>(n); begin += leaf_size)
         {
            const size_t end = std::min<size_t>(begin + leaf_size, static_cast<size_t>(n));
            if (end <= begin + 1)
               continue;
            std::vector<std::pair<float, IdxType>> candidates;
            candidates.reserve(end - begin - 1);
            for (size_t a = begin; a < end; ++a)
            {
               const IdxType src = order[a].second;
               const char *src_vec = local_storage->get_vector(src);
               candidates.clear();
               for (size_t b = begin; b < end; ++b)
               {
                  if (a == b)
                     continue;
                  const IdxType dst = order[b].second;
                  const float dist = distance_handler->compute(src_vec, local_storage->get_vector(dst), dim);
                  candidates.emplace_back(dist, dst);
               }
               if (candidates.size() > leaf_topk)
               {
                  std::nth_element(candidates.begin(), candidates.begin() + leaf_topk, candidates.end(),
                                   [](const auto &x, const auto &y) {
                                      return x.first < y.first || (x.first == y.first && x.second < y.second);
                                   });
                  candidates.resize(leaf_topk);
               }
               for (const auto &candidate : candidates)
               {
                  std::lock_guard<std::mutex> lock(reservoir_locks[src]);
                  hashprune_stream_edge(local_storage, reservoirs, src, candidate.second, candidate.first,
                                        hash_bits, hash_sample_dims, seed);
                  hashprune_edges_streamed += 1;
               }
            }
         }
      }
   }

   std::vector<std::vector<Candidate>> candidate_pools(n);
   for (IdxType local_id = 0; local_id < n; ++local_id)
   {
      candidate_pools[local_id] = reservoirs[local_id].to_candidates();
      if (mix_sample_candidates > 0)
      {
         const size_t target_candidate_count = candidate_pools[local_id].size() + mix_sample_candidates;
         add_sampled_candidates_to_pool(local_storage, distance_handler, local_id, mix_sample_candidates,
                                        target_candidate_count, candidate_pools[local_id]);
      }
      if (collect_diagnostics)
         diagnostics.bucket_coverage_sum += reservoirs[local_id].bucket_coverage();
   }

   local_index->build_from_candidate_pools(local_storage, distance_handler, graph, candidate_pools,
                                           max_degree, Lbuild, alpha, num_threads, max_candidate_size);

   auto rebuild_hashprune_after_low_degree_repair = [&]() {
      for (IdxType local_id = 0; local_id < n; ++local_id)
         graph->neighbors[local_id].clear();
      local_index = std::make_shared<Vamana>(false);
      local_index->build_from_candidate_pools(local_storage, distance_handler, graph, candidate_pools,
                                              max_degree, Lbuild, alpha, num_threads, max_candidate_size);
   };

   auto repair_low_final_degree_points = [&]() {
      const bool hashprune_nav_repair = nav_repair_enabled;
      if (repair_degree == 0 && repair_2hop_min == 0)
         return false;
      bool repaired_any = false;
      const size_t candidate_limit = std::min<size_t>(static_cast<size_t>(max_candidate_size), static_cast<size_t>(n - 1));
      for (IdxType local_id = 0; local_id < n; ++local_id)
      {
         const size_t final_degree = graph->neighbors[local_id].size();
         bool low_degree = repair_degree > 0 && final_degree < repair_degree;
         bool low_two_hop = false;
         if (!low_degree && hashprune_nav_repair && repair_2hop_min > 0)
         {
            const size_t two_hop = two_hop_coverage_until(graph, local_id, n, repair_2hop_min);
            low_two_hop = two_hop < repair_2hop_min;
            if (low_two_hop)
               diagnostics.low_2hop_points += 1;
         }
         if (!low_degree && !low_two_hop)
            continue;
         if (repair_max_points > 0 && diagnostics.repaired_points >= repair_max_points)
            break;
         if (candidate_pools[local_id].size() >= candidate_limit)
            continue;

         const size_t before_repair = candidate_pools[local_id].size();
         if (hashprune_nav_repair)
         {
            size_t target_candidate_count = std::min<size_t>(candidate_limit, candidate_pools[local_id].size() + repair_graph_hops);
            const size_t graph_added = add_graph_expansion_candidates_to_pool(
                local_storage, distance_handler, graph, local_id, n, repair_graph_hops,
                target_candidate_count, candidate_pools[local_id]);
            diagnostics.graph_repair_candidates_added += graph_added;

            target_candidate_count = std::min<size_t>(candidate_limit, candidate_pools[local_id].size() + repair_reservoir_near);
            const size_t reservoir_added = add_neighbor_reservoir_candidates_to_pool(
                local_storage, distance_handler, graph, reservoirs, local_id, repair_reservoir_near,
                target_candidate_count, candidate_pools[local_id]);
            diagnostics.reservoir_repair_candidates_added += reservoir_added;

            const auto hash_candidates = collect_hash_near_candidates(
                local_id, hash_orders, hash_positions, repair_hash_near);
            target_candidate_count = std::min<size_t>(candidate_limit, candidate_pools[local_id].size() + repair_hash_near);
            const size_t hash_added = add_candidate_ids_to_pool(
                local_storage, distance_handler, local_id, hash_candidates,
                target_candidate_count, candidate_pools[local_id]);
            diagnostics.hash_repair_candidates_added += hash_added;
         }

         if (repair_sample_candidates > 0 && candidate_pools[local_id].size() < candidate_limit)
         {
            const size_t target_candidate_count = std::min<size_t>(candidate_limit, candidate_pools[local_id].size() + repair_sample_candidates);
            const size_t sample_added = add_sampled_candidates_to_pool(
                local_storage, distance_handler, local_id,
                std::max<size_t>(repair_sample_candidates, repair_degree),
                target_candidate_count, candidate_pools[local_id]);
            diagnostics.sample_repair_candidates_added += sample_added;
         }

         if (candidate_pools[local_id].size() > before_repair)
         {
            diagnostics.repaired_points += 1;
            diagnostics.repair_candidates_added += candidate_pools[local_id].size() - before_repair;
            repaired_any = true;
         }
      }
      return repaired_any;
   };

   if (repair_low_final_degree_points())
      rebuild_hashprune_after_low_degree_repair();

   if (collect_diagnostics)
   {
      for (IdxType local_id = 0; local_id < n; ++local_id)
      {
         diagnostics.points += 1;
         diagnostics.candidate_count_sum += candidate_pools[local_id].size();
         const IdxType degree = static_cast<IdxType>(graph->neighbors[local_id].size());
         diagnostics.final_degree_sum += degree;
         diagnostics.min_degree = std::min(diagnostics.min_degree, degree);
         diagnostics.max_degree = std::max(diagnostics.max_degree, degree);
         if (degree < 16)
            diagnostics.low_degree_lt16 += 1;
         if (degree < 32)
            diagnostics.low_degree_lt32 += 1;
         if (degree >= max_degree)
            diagnostics.full_degree_points += 1;
      }
   }
   return local_index;
}

class SpecialBlockStorageView : public IStorage
{
public:
   SpecialBlockStorageView(std::shared_ptr<IStorage> base_storage,
                           std::vector<IdxType> point_ids)
       : base_storage_(std::move(base_storage)), point_ids_(std::move(point_ids))
   {
      labels_.reserve(point_ids_.size());
      for (IdxType point_id : point_ids_)
         labels_.push_back(base_storage_->get_label_set(point_id));
   }

   void load_from_file(const std::string &, const std::string &, IdxType) override
   {
      throw std::runtime_error("SpecialBlockStorageView does not support load_from_file");
   }

   void write_to_file(const std::string &, const std::string &) override
   {
      throw std::runtime_error("SpecialBlockStorageView does not support write_to_file");
   }

   void reorder_data(const std::vector<IdxType> &) override
   {
      throw std::runtime_error("SpecialBlockStorageView does not support reorder_data");
   }

   DataType get_data_type() const override { return base_storage_->get_data_type(); }
   IdxType get_num_points() const override { return static_cast<IdxType>(point_ids_.size()); }
   IdxType get_dim() const override { return base_storage_->get_dim(); }

   std::vector<LabelType> *get_offseted_label_sets(IdxType idx) override { return labels_.data() + idx; }
   char *get_vector(IdxType idx) override { return base_storage_->get_vector(point_ids_[idx]); }
   std::vector<LabelType> &get_label_set(IdxType idx) override { return labels_[idx]; }
   void prefetch_vec_by_id(IdxType idx) const override { base_storage_->prefetch_vec_by_id(point_ids_[idx]); }

   IdxType choose_medoid(uint32_t num_threads, std::shared_ptr<DistanceHandler> distance_handler) override
   {
      if (point_ids_.empty())
         return 0;
      const IdxType n = static_cast<IdxType>(point_ids_.size());
      const IdxType dim = base_storage_->get_dim();
      std::vector<float> mean(dim, 0.0f);
      if (base_storage_->get_data_type() != DataType::FLOAT)
         return 0;
      for (IdxType local_id = 0; local_id < n; ++local_id)
      {
         const float *vec = reinterpret_cast<const float *>(get_vector(local_id));
         for (IdxType d = 0; d < dim; ++d)
            mean[d] += vec[d];
      }
      for (IdxType d = 0; d < dim; ++d)
         mean[d] /= static_cast<float>(n);

      std::vector<float> dists(n, 0.0f);
      omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 2048)
      for (IdxType local_id = 0; local_id < n; ++local_id)
      {
         dists[local_id] = distance_handler->compute(reinterpret_cast<const char *>(mean.data()),
                                                     get_vector(local_id), dim);
      }
      return static_cast<IdxType>(std::min_element(dists.begin(), dists.end()) - dists.begin());
   }

   void clean() override {}

private:
   std::shared_ptr<IStorage> base_storage_;
   std::vector<IdxType> point_ids_;
   std::vector<std::vector<LabelType>> labels_;
};

std::string join_labels_for_special_path(const std::vector<LabelType> &labels)
{
   std::string out;
   for (size_t i = 0; i < labels.size(); ++i)
   {
      if (i)
         out.push_back(' ');
      out += std::to_string(labels[i]);
   }
   return out;
}

std::vector<std::string> split_csv_line(const std::string &line)
{
   std::vector<std::string> out;
   std::string cur;
   std::stringstream ss(line);
   while (std::getline(ss, cur, ','))
      out.push_back(cur);
   return out;
}

std::vector<LabelType> parse_label_list(const std::string &text)
{
   std::vector<LabelType> labels;
   std::stringstream ss(text);
   std::string item;
   while (std::getline(ss, item, ' '))
   {
      if (!item.empty())
         labels.push_back(static_cast<LabelType>(std::stoul(item)));
   }
   return labels;
}

void collect_special_block_members(const SpecialBlockTrieIndex &trie,
                                   SpecialBlockTrieIndex::NodeId root,
                                   SpecialBlock &block,
                                   const std::vector<IdxType> &node_group_points,
                                   const std::vector<IdxType> &node_to_block)
{
   std::vector<SpecialBlockTrieIndex::NodeId> stack{root};
   while (!stack.empty())
   {
      const SpecialBlockTrieIndex::NodeId cur = stack.back();
      stack.pop_back();
      const SpecialBlockTrieNode &node = trie.node(cur);
      if (cur != root && cur < node_to_block.size() && node_to_block[cur] != 0)
      {
         block.child_block_ids.push_back(node_to_block[cur]);
         continue;
      }
      if (node.terminal_group_id != 0)
      {
         block.member_group_ids.push_back(node.terminal_group_id);
         block.point_count += node_group_points[cur];
      }
      for (size_t child_offset = 0; child_offset < node.child_count; ++child_offset)
         stack.push_back(trie.child_node_id(cur, child_offset));
   }
   std::sort(block.member_group_ids.begin(), block.member_group_ids.end());
   std::sort(block.child_block_ids.begin(), block.child_block_ids.end());
   block.child_block_ids.erase(std::unique(block.child_block_ids.begin(), block.child_block_ids.end()),
                               block.child_block_ids.end());
}

std::vector<IdxType> collect_block_points(const SpecialBlock &block,
                                          const std::vector<std::pair<IdxType, IdxType>> &group_ranges)
{
   std::vector<IdxType> points;
   points.reserve(block.point_count);
   for (IdxType group_id : block.member_group_ids)
   {
      if (group_id >= group_ranges.size())
         continue;
      const auto &range = group_ranges[group_id];
      for (IdxType point_id = range.first; point_id < range.second; ++point_id)
         points.push_back(point_id);
   }
   return points;
}

bool env_flag(const char *key)
{
   const char *value = std::getenv(key);
   return value && *value && std::string(value) != "0";
}

unsigned long long read_env_ull(const char *key, unsigned long long fallback = 0)
{
   const char *value = std::getenv(key);
   if (!value || !*value)
      return fallback;
   char *end = nullptr;
   unsigned long long parsed = std::strtoull(value, &end, 10);
   if (end == value || *end != 0)
      return fallback;
   return parsed;
}

struct SpecialInterPairProfile
{
   double ms = 0.0;
   IdxType parent_block_id = 0;
   IdxType child_block_id = 0;
   size_t source_points = 0;
   size_t target_points = 0;
   unsigned long long pair_work = 0;
   unsigned long long edges = 0;
   bool exact_scan = false;
   bool sampled = false;
   bool forced_graph = false;
};

void record_slowest_inter_pair(std::vector<SpecialInterPairProfile> &slowest_pairs,
                               const SpecialInterPairProfile &profile,
                               size_t max_pairs = 8)
{
   slowest_pairs.push_back(profile);
   std::sort(slowest_pairs.begin(), slowest_pairs.end(),
             [](const auto &a, const auto &b) {
                return a.ms > b.ms;
             });
   if (slowest_pairs.size() > max_pairs)
      slowest_pairs.resize(max_pairs);
}

std::vector<IdxType> sample_special_inter_candidates(const std::vector<IdxType> &dst_points,
                                                     size_t requested_candidates)
{
   if (requested_candidates == 0 || requested_candidates >= dst_points.size())
      return dst_points;

   std::vector<IdxType> sampled_dst_points;
   sampled_dst_points.reserve(requested_candidates);
   if (requested_candidates == 1)
   {
      sampled_dst_points.push_back(dst_points[dst_points.size() / 2]);
      return sampled_dst_points;
   }

   const size_t last = dst_points.size() - 1;
   size_t prev_idx = dst_points.size();
   for (size_t i = 0; i < requested_candidates; ++i)
   {
      const size_t idx = (i * last) / (requested_candidates - 1);
      if (idx != prev_idx)
      {
         sampled_dst_points.push_back(dst_points[idx]);
         prev_idx = idx;
      }
   }
   return sampled_dst_points;
}

bool read_special_edge_binary_file(const std::string &path,
                                   std::vector<std::vector<SpecialEdge>> &edges_by_point,
                                   SpecialBlockBuildSummary &summary,
                                   size_t &loaded_edges,
                                   const std::vector<SpecialBlock> &blocks,
                                   const std::vector<IdxType> &point_to_middle_block,
                                   const std::vector<IdxType> &point_to_upper_block)
{
   uint64_t binary_count = 0;
   std::string error;
   std::string invalid_record;
   std::vector<std::vector<SpecialEdge>> staged_edges(edges_by_point.size());
   size_t staged_edges_count = 0;
   size_t staged_intra_count = 0;
   size_t staged_inter_count = 0;
   const bool loaded = for_each_special_edge_binary_record(
       path,
       [&](const SpecialEdgeBinaryRecord &rec) {
          if (rec.source >= edges_by_point.size())
          {
             invalid_record = "source id is out of range";
             return;
          }
          if (rec.kind > 1)
          {
             invalid_record = "edge kind is invalid";
             return;
          }
          SpecialEdge edge;
          edge.target_point_id = static_cast<IdxType>(rec.target);
          edge.special_block_id = static_cast<IdxType>(rec.block);
          edge.kind = rec.kind != 0 ? SpecialEdgeKind::InterBlock : SpecialEdgeKind::IntraBlock;
          staged_edges[rec.source].push_back(edge);
          staged_edges_count += 1;
          if (edge.kind == SpecialEdgeKind::InterBlock)
             staged_inter_count += 1;
          else
             staged_intra_count += 1;
       },
       binary_count, error);
   if (!loaded || !invalid_record.empty())
   {
      std::cerr << "[special_edges][load] invalid binary sidecar " << path
                << ": " << (!invalid_record.empty() ? invalid_record : error)
                << "; falling back to CSV" << std::endl;
      return false;
   }
   for (IdxType source = 0; source < staged_edges.size(); ++source)
   {
      for (const SpecialEdge &edge : staged_edges[source])
      {
         std::string semantic_error;
         if (!validate_special_edge_semantics(
                 source, edge, blocks, point_to_middle_block,
                 point_to_upper_block, semantic_error))
         {
            std::cerr << "[special_edges][load] invalid binary sidecar " << path
                      << ": " << semantic_error << std::endl;
            return false;
         }
      }
   }
   edges_by_point.swap(staged_edges);
   summary.special_edges += static_cast<IdxType>(staged_edges_count);
   summary.intra_special_edges += static_cast<IdxType>(staged_intra_count);
   summary.inter_special_edges += static_cast<IdxType>(staged_inter_count);
   loaded_edges += staged_edges_count;
   std::cout << "[special_edges][load] binary=" << path
             << " declared_edges=" << binary_count
             << " loaded_edges=" << loaded_edges << std::endl;
   return true;
}

std::vector<float> pack_special_block_vectors(const std::shared_ptr<IStorage> &base_storage,
                                               const std::vector<IdxType> &points)
{
   if (base_storage->get_data_type() != DataType::FLOAT)
      throw std::runtime_error("special block GPU intra build supports only float vectors.");
   const IdxType dim = base_storage->get_dim();
   std::vector<float> packed(static_cast<size_t>(points.size()) * static_cast<size_t>(dim));
   for (size_t local_id = 0; local_id < points.size(); ++local_id)
   {
      const float *src = reinterpret_cast<const float *>(base_storage->get_vector(points[local_id]));
      std::memcpy(packed.data() + static_cast<size_t>(local_id) * static_cast<size_t>(dim),
                  src,
                  static_cast<size_t>(dim) * sizeof(float));
   }
   return packed;
}

void fill_special_graph_from_tagore_result(std::shared_ptr<Graph> graph,
                                           const TagoreBuildResult &result,
                                           IdxType n,
                                           IdxType max_degree,
                                           uint32_t fallback_stride)
{
   const uint32_t stride = result.graph_stride != 0 ? result.graph_stride : fallback_stride;
   if (stride == 0 || result.graph.empty())
      throw std::runtime_error("special block GPU intra build returned an empty graph.");
   for (IdxType local_id = 0; local_id < n; ++local_id)
   {
      auto &neighbors = graph->neighbors[local_id];
      neighbors.clear();
      const uint32_t degree = result.graph[static_cast<size_t>(local_id) * stride];
      for (uint32_t j = 0; j < degree && neighbors.size() < max_degree; ++j)
      {
         const uint32_t neighbor = result.graph[static_cast<size_t>(local_id) * stride + 1 + j];
         if (neighbor < n && neighbor != local_id)
            neighbors.emplace_back(static_cast<IdxType>(neighbor));
      }
   }
}

} // namespace

void save_special_block_metadata_binary(
    const std::string &path,
    const std::vector<SpecialBlock> &blocks)
{
   write_special_block_metadata_binary_impl(path, blocks);
}

bool load_special_block_metadata_binary(
    const std::string &path,
    std::vector<SpecialBlock> &blocks,
    std::string &error)
{
   return read_special_block_metadata_binary_impl(path, blocks, error);
}

bool validate_special_block_metadata(
    const std::vector<SpecialBlock> &blocks,
    std::string &error)
{
   return validate_special_block_metadata_impl(blocks, error);
}

bool validate_special_block_graph_semantics(
    const std::vector<SpecialBlock> &blocks,
    IdxType num_points,
    IdxType num_groups,
    const std::vector<std::vector<LabelType>> &group_labels,
    const std::vector<std::pair<IdxType, IdxType>> &group_ranges,
    const std::vector<IdxType> &point_to_group,
    std::string &error)
{
   return validate_special_block_graph_semantics_impl(
       blocks, num_points, num_groups, group_labels, group_ranges,
       point_to_group, error);
}

void UniNavGraph::refresh_special_block_memory_stats()
{
   uint64_t logical = 0;
   uint64_t allocated = 0;

   logical += _special_blocks.size() * sizeof(SpecialBlock);
   allocated += sizeof(_special_blocks) + _special_blocks.capacity() * sizeof(SpecialBlock);
   for (const auto &block : _special_blocks)
   {
      logical += block.root_labels.size() * sizeof(LabelType);
      logical += block.common_labels.size() * sizeof(LabelType);
      logical += block.member_group_ids.size() * sizeof(IdxType);
      logical += block.child_block_ids.size() * sizeof(IdxType);
      allocated += block.root_labels.capacity() * sizeof(LabelType);
      allocated += block.common_labels.capacity() * sizeof(LabelType);
      allocated += block.member_group_ids.capacity() * sizeof(IdxType);
      allocated += block.child_block_ids.capacity() * sizeof(IdxType);
   }

   const auto add_flat_vector = [&](const auto &values) {
      using Vector = std::decay_t<decltype(values)>;
      using Value = typename Vector::value_type;
      logical += values.size() * sizeof(Value);
      allocated += sizeof(Vector) + values.capacity() * sizeof(Value);
   };
   add_flat_vector(_group_id_to_special_block);
   add_flat_vector(_group_id_to_upper_special_block);
   add_flat_vector(_group_is_special_block_root);
   add_flat_vector(_group_is_trivial_special_block_root);
   add_flat_vector(_point_to_special_block);
   add_flat_vector(_point_to_upper_special_block);
   add_flat_vector(_point_is_special_block_root);

   const auto add_nested_vector = [&](const auto &rows) {
      using Outer = std::decay_t<decltype(rows)>;
      using Inner = typename Outer::value_type;
      using Value = typename Inner::value_type;
      logical += rows.size() * sizeof(Inner);
      allocated += sizeof(Outer) + rows.capacity() * sizeof(Inner);
      for (const auto &row : rows)
      {
         logical += row.size() * sizeof(Value);
         allocated += row.capacity() * sizeof(Value);
      }
   };
   add_nested_vector(_special_edges_by_point);
   add_nested_vector(_special_heavy_edges_by_point);
   add_nested_vector(_special_trie_regular_edges_by_point);
   add_flat_vector(_special_edges_csr.offsets);
   add_flat_vector(_special_edges_csr.edges);
   add_flat_vector(_special_heavy_edges_csr.offsets);
   add_flat_vector(_special_heavy_edges_csr.edges);
   add_flat_vector(_special_trie_regular_edges_csr.offsets);
   add_flat_vector(_special_trie_regular_edges_csr.targets);

   logical += _special_block_trie_index.logical_memory_size_bytes();
   allocated += _special_block_trie_index.memory_size_bytes();
   _special_block_summary.memory_logical_bytes = logical;
   _special_block_summary.memory_allocated_bytes = allocated;
   _special_block_summary.index_bytes = allocated;
}

UniNavGraph::SpecialEdgeView
UniNavGraph::special_edges_for_point(IdxType point_id) const
{
   if (!_special_edges_csr.empty() && point_id < _special_edges_csr.num_sources())
   {
      const uint64_t begin = _special_edges_csr.offsets[point_id];
      const uint64_t end = _special_edges_csr.offsets[point_id + 1];
      if (begin == end)
         return {};
      return {nullptr, _special_edges_csr.edges.data() + begin, static_cast<size_t>(end - begin)};
   }
   if (point_id < _special_edges_by_point.size())
      return {_special_edges_by_point[point_id].data(), nullptr, _special_edges_by_point[point_id].size()};
   return {};
}

UniNavGraph::SpecialEdgeView
UniNavGraph::special_heavy_edges_for_point(IdxType point_id) const
{
   if (!_special_heavy_edges_csr.empty() && point_id < _special_heavy_edges_csr.num_sources())
   {
      const uint64_t begin = _special_heavy_edges_csr.offsets[point_id];
      const uint64_t end = _special_heavy_edges_csr.offsets[point_id + 1];
      if (begin == end)
         return {};
      return {nullptr, _special_heavy_edges_csr.edges.data() + begin, static_cast<size_t>(end - begin)};
   }
   if (point_id < _special_heavy_edges_by_point.size())
      return {_special_heavy_edges_by_point[point_id].data(), nullptr, _special_heavy_edges_by_point[point_id].size()};
   return {};
}

UniNavGraph::ContiguousView<IdxType>
UniNavGraph::special_regular_edges_for_point(IdxType point_id) const
{
   if (!_special_trie_regular_edges_csr.empty() &&
       point_id < _special_trie_regular_edges_csr.num_sources())
   {
      const uint64_t begin = _special_trie_regular_edges_csr.offsets[point_id];
      const uint64_t end = _special_trie_regular_edges_csr.offsets[point_id + 1];
      if (begin == end)
         return {};
      return {_special_trie_regular_edges_csr.targets.data() + begin,
              static_cast<size_t>(end - begin)};
   }
   if (point_id < _special_trie_regular_edges_by_point.size())
      return {_special_trie_regular_edges_by_point[point_id].data(),
              _special_trie_regular_edges_by_point[point_id].size()};
   return {};
}

bool UniNavGraph::has_special_edges() const
{
   return !_special_edges_csr.empty() || !_special_edges_by_point.empty();
}

bool UniNavGraph::is_special_block_root_group(IdxType group_id) const
{
   return group_id < _group_is_special_block_root.size() &&
          _group_is_special_block_root[group_id] != 0;
}

bool UniNavGraph::is_trivial_special_block_root_group(IdxType group_id) const
{
   return group_id < _group_is_trivial_special_block_root.size() &&
          _group_is_trivial_special_block_root[group_id] != 0;
}

bool UniNavGraph::is_special_block_root_point(IdxType point_id) const
{
   return point_id < _point_is_special_block_root.size() &&
          _point_is_special_block_root[point_id] != 0;
}

bool UniNavGraph::query_covers_special_block(const std::vector<LabelType> &query_labels,
                                             const SpecialBlock &block) const
{
   std::vector<LabelType> sorted_query = query_labels;
   std::vector<LabelType> sorted_coverage =
       (_special_block_summary.upper_blocks > 0 ||
        env_flag("UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE"))
           ? block.root_labels
           : block.common_labels;
   std::sort(sorted_query.begin(), sorted_query.end());
   std::sort(sorted_coverage.begin(), sorted_coverage.end());
   return std::includes(sorted_coverage.begin(), sorted_coverage.end(),
                        sorted_query.begin(), sorted_query.end());
}

void UniNavGraph::populate_special_query_stats(const std::vector<LabelType> &query_labels,
                                               QueryStats &stats) const
{
   stats.special_query_matched_points = 0;
   stats.special_query_points = 0;
   stats.special_query_trivial_points = 0;
   stats.special_query_nontrivial_points = 0;
   stats.special_query_block_count = 0;
   stats.special_query_trivial_block_count = 0;
   stats.special_query_nontrivial_block_count = 0;
   stats.special_query_ratio = 0.0f;
   stats.special_query_nontrivial_ratio = 0.0f;

   stats.special_query_matched_points = stats.entry_group_matched_points;

   if (_special_blocks.empty() || stats.special_query_matched_points == 0)
      return;

   for (const SpecialBlock &block : _special_blocks)
   {
      if (!query_covers_special_block(query_labels, block))
         continue;
      const size_t points = static_cast<size_t>(block.point_count);
      stats.special_query_points += points;
      stats.special_query_block_count += 1;
      if (block.is_trivial())
      {
         stats.special_query_trivial_points += points;
         stats.special_query_trivial_block_count += 1;
      }
      else
      {
         stats.special_query_nontrivial_points += points;
         stats.special_query_nontrivial_block_count += 1;
      }
   }

   stats.special_query_ratio =
       static_cast<float>(stats.special_query_points) /
       static_cast<float>(stats.special_query_matched_points);
   stats.special_query_nontrivial_ratio =
       static_cast<float>(stats.special_query_nontrivial_points) /
       static_cast<float>(stats.special_query_matched_points);
}

void UniNavGraph::rebuild_special_block_indexes()
{
   _group_id_to_special_block.assign(_num_groups + 1, 0);
   _group_id_to_upper_special_block.assign(_num_groups + 1, 0);
   _group_is_special_block_root.assign(_num_groups + 1, 0);
   _group_is_trivial_special_block_root.assign(_num_groups + 1, 0);
   _point_to_special_block.assign(_num_points, 0);
   _point_to_upper_special_block.assign(_num_points, 0);
   _point_is_special_block_root.assign(_num_points, 0);
   _special_block_summary.num_blocks = static_cast<IdxType>(_special_blocks.size());
   _special_block_summary.upper_blocks = 0;
   _special_block_summary.trivial_blocks = 0;
   _special_block_summary.member_groups = 0;
   _special_block_summary.member_points = 0;
   _special_block_summary.child_block_edges = 0;

   for (const SpecialBlock &block : _special_blocks)
   {
      if (block.level > 0)
         _special_block_summary.upper_blocks += 1;
      _special_block_summary.member_groups += static_cast<IdxType>(block.member_group_ids.size());
      _special_block_summary.member_points += block.point_count;
      _special_block_summary.child_block_edges += static_cast<IdxType>(block.child_block_ids.size());
      if (block.is_trivial())
         _special_block_summary.trivial_blocks += 1;
      if (block.level == 0 && block.root_group_id > 0 &&
          block.root_group_id < _group_is_special_block_root.size())
      {
         _group_is_special_block_root[block.root_group_id] = 1;
         if (block.is_trivial())
            _group_is_trivial_special_block_root[block.root_group_id] = 1;
         const auto &root_range = _group_id_to_range[block.root_group_id];
         for (IdxType point_id = root_range.first; point_id < root_range.second && point_id < _point_is_special_block_root.size(); ++point_id)
            _point_is_special_block_root[point_id] = 1;
      }
      for (IdxType group_id : block.member_group_ids)
      {
         if (group_id >= _group_id_to_special_block.size())
            continue;
         std::vector<IdxType> &group_map = block.level == 0
                                               ? _group_id_to_special_block
                                               : _group_id_to_upper_special_block;
         std::vector<IdxType> &point_map = block.level == 0
                                               ? _point_to_special_block
                                               : _point_to_upper_special_block;
         group_map[group_id] = block.block_id;
         const auto &range = _group_id_to_range[group_id];
         for (IdxType point_id = range.first; point_id < range.second && point_id < point_map.size(); ++point_id)
            point_map[point_id] = block.block_id;
      }
   }
}

void UniNavGraph::build_special_blocks()
{
   _special_blocks.clear();
   _special_block_trie_index.clear();
   _group_id_to_special_block.assign(_num_groups + 1, 0);
   _group_id_to_upper_special_block.assign(_num_groups + 1, 0);
   _group_is_special_block_root.assign(_num_groups + 1, 0);
   _group_is_trivial_special_block_root.assign(_num_groups + 1, 0);
   _point_to_special_block.assign(_num_points, 0);
   _point_to_upper_special_block.assign(_num_points, 0);
   _point_is_special_block_root.assign(_num_points, 0);
   _special_edges_by_point.clear();
   _special_heavy_edges_by_point.clear();
   _special_block_summary = {};
   _special_block_summary.threshold = static_cast<IdxType>(_build_config.special_block_min_points);
   _special_block_summary.upper_threshold =
       static_cast<IdxType>(_build_config.special_block_upper_min_points);

   if (!_build_config.special_blocks_enabled)
      return;

   const auto start = std::chrono::high_resolution_clock::now();
   if (_build_config.special_block_partition == UngSpecialBlockPartition::Lng)
   {
      if (_build_config.special_block_upper_min_points > 0)
         throw std::runtime_error(
             "multi-level Special Blocks currently require trie partitioning");
      if (!_label_nav_graph)
         throw std::runtime_error("LNG special block partition requires the label navigation graph to be built first.");
      std::vector<IdxType> group_points(_num_groups + 1, 0);
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         const auto &range = _group_id_to_range[group_id];
         group_points[group_id] = range.second - range.first;
      }

      LngBlockPartitionInput input;
      input.out_neighbors = &_label_nav_graph->out_neighbors;
      input.group_points = &group_points;
      input.group_labels = &_group_id_to_label_set;
      input.min_points = static_cast<IdxType>(_build_config.special_block_min_points);
      input.tree_mode = _build_config.special_block_tree_mode == "bfs"
                            ? LngBlockTreeMode::Bfs
                            : LngBlockTreeMode::Random;
      input.tree_seed = _build_config.special_block_tree_seed;
      LngBlockPartitionResult partition = build_lng_special_blocks(input);
      _special_blocks = std::move(partition.blocks);
      rebuild_special_block_indexes();

      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - start)
                            .count();
      _special_block_summary.metadata_ms = ms;
      std::cout << "[special_blocks] enabled=1 partition=lng tree_mode="
                << _build_config.special_block_tree_mode
                << " tree_seed=" << _build_config.special_block_tree_seed
                << " threshold=" << _special_block_summary.threshold
                << " blocks=" << _special_block_summary.num_blocks
                << " trivial_blocks=" << _special_block_summary.trivial_blocks
                << " member_groups=" << _special_block_summary.member_groups
                << " member_points=" << _special_block_summary.member_points
                << " child_block_edges=" << _special_block_summary.child_block_edges
                << " ms=" << ms << std::endl;
      prof_logf("[PROF] special_blocks partition=lng tree_mode=%s tree_seed=%llu threshold=%u blocks=%u trivial_blocks=%u member_groups=%u member_points=%u child_block_edges=%u ms=%.3f",
                _build_config.special_block_tree_mode.c_str(),
                static_cast<unsigned long long>(_build_config.special_block_tree_seed),
                static_cast<unsigned>(_special_block_summary.threshold),
                static_cast<unsigned>(_special_block_summary.num_blocks),
                static_cast<unsigned>(_special_block_summary.trivial_blocks),
                static_cast<unsigned>(_special_block_summary.member_groups),
                static_cast<unsigned>(_special_block_summary.member_points),
                static_cast<unsigned>(_special_block_summary.child_block_edges),
                ms);
      return;
   }

   const auto trie_build_start = std::chrono::high_resolution_clock::now();
   _special_block_trie_index.build(_group_id_to_label_set, _num_groups);
   _special_block_summary.trie_build_ms = std::chrono::duration<double, std::milli>(
                                               std::chrono::high_resolution_clock::now() - trie_build_start)
                                               .count();
   _special_block_summary.trie_node_count = _special_block_trie_index.node_count();
   _special_block_summary.trie_child_count = _special_block_trie_index.child_count();
   _special_block_summary.trie_serialized_bytes = _special_block_trie_index.serialized_size_bytes();

   std::vector<IdxType> node_group_points(_special_block_trie_index.node_count(), 0);
   std::vector<IdxType> subtree_points(_special_block_trie_index.node_count(), 0);
   for (SpecialBlockTrieIndex::NodeId node_id = 1;
        node_id < _special_block_trie_index.node_count(); ++node_id)
   {
      const IdxType group_id = _special_block_trie_index.node(node_id).terminal_group_id;
      if (group_id == 0)
         continue;
      const auto &range = _group_id_to_range[group_id];
      node_group_points[node_id] = range.second - range.first;
   }

   // Subtree cardinality is shared by all layers. Each layer has an
   // independent uncovered counter, so adding a coarse layer does not alter
   // the historical 1k partition or its block ids.
   for (size_t reverse_idx = _special_block_trie_index.node_count(); reverse_idx > 0; --reverse_idx)
   {
      const auto idx = static_cast<SpecialBlockTrieIndex::NodeId>(reverse_idx - 1);
      const SpecialBlockTrieNode &node = _special_block_trie_index.node(idx);
      IdxType total = node_group_points[idx];
      for (size_t child_offset = 0; child_offset < node.child_count; ++child_offset)
         total += subtree_points[_special_block_trie_index.child_node_id(idx, child_offset)];
      subtree_points[idx] = total;
   }

   auto append_layer = [&](IdxType threshold, uint8_t level) {
      std::vector<IdxType> uncovered_points(_special_block_trie_index.node_count(), 0);
      std::vector<IdxType> node_to_block(_special_block_trie_index.node_count(), 0);
      for (size_t reverse_idx = _special_block_trie_index.node_count(); reverse_idx > 0; --reverse_idx)
      {
         const auto idx = static_cast<SpecialBlockTrieIndex::NodeId>(reverse_idx - 1);
         const SpecialBlockTrieNode &node = _special_block_trie_index.node(idx);
         IdxType uncovered = node_group_points[idx];
         for (size_t child_offset = 0; child_offset < node.child_count; ++child_offset)
            uncovered += uncovered_points[
                _special_block_trie_index.child_node_id(idx, child_offset)];
         uncovered_points[idx] = uncovered;
         if (idx == 0 || uncovered <= threshold)
            continue;

         SpecialBlock block;
         block.block_id = static_cast<IdxType>(_special_blocks.size() + 1);
         block.level = level;
         block.subtree_point_count = subtree_points[idx];
         block.root_labels = _special_block_trie_index.labels_for_node(idx);
         collect_special_block_members(_special_block_trie_index, idx, block,
                                       node_group_points, node_to_block);
         if (block.member_group_ids.empty())
            continue;
         block.root_group_id = block.member_group_ids.front();
         block.common_labels = compute_direct_member_common_labels(
             block.member_group_ids, _group_id_to_label_set);
         _special_blocks.push_back(std::move(block));
         node_to_block[idx] = static_cast<IdxType>(_special_blocks.size());
         if (level == 0)
            _special_block_trie_index.set_block_id(idx, node_to_block[idx]);
         uncovered_points[idx] = 0;
      }
      return node_to_block;
   };

   const std::vector<IdxType> middle_node_to_block = append_layer(
       static_cast<IdxType>(_build_config.special_block_min_points), 0);
   if (_build_config.special_block_upper_min_points > 0)
   {
      const std::vector<IdxType> upper_node_to_block = append_layer(
          static_cast<IdxType>(_build_config.special_block_upper_min_points), 1);
      // Link each middle block to its nearest containing upper-layer block.
      for (const IdxType middle_block_id : middle_node_to_block)
      {
         if (middle_block_id == 0 || middle_block_id > _special_blocks.size())
            continue;
         SpecialBlock &middle = _special_blocks[middle_block_id - 1];
         auto node_id = _special_block_trie_index.find_exact_node(middle.root_labels);
         while (node_id != SpecialBlockTrieIndex::kInvalidNodeId)
         {
            if (node_id < upper_node_to_block.size() &&
                upper_node_to_block[node_id] != 0)
            {
               middle.parent_block_id = upper_node_to_block[node_id];
               break;
            }
            if (node_id == 0)
               break;
            node_id = _special_block_trie_index.node(node_id).parent_id;
         }
      }
   }

   rebuild_special_block_indexes();

   const double ms = std::chrono::duration<double, std::milli>(
                         std::chrono::high_resolution_clock::now() - start)
                         .count();
   _special_block_summary.metadata_ms = ms;
   std::cout << "[special_blocks] enabled=1 partition=trie threshold=" << _special_block_summary.threshold
             << " upper_threshold=" << _special_block_summary.upper_threshold
             << " trie_nodes=" << _special_block_summary.trie_node_count
             << " trie_children=" << _special_block_summary.trie_child_count
             << " trie_bytes=" << _special_block_summary.trie_serialized_bytes
             << " trie_build_ms=" << _special_block_summary.trie_build_ms
             << " blocks=" << _special_block_summary.num_blocks
             << " upper_blocks=" << _special_block_summary.upper_blocks
             << " trivial_blocks=" << _special_block_summary.trivial_blocks
             << " member_groups=" << _special_block_summary.member_groups
             << " member_points=" << _special_block_summary.member_points
             << " child_block_edges=" << _special_block_summary.child_block_edges
             << " ms=" << ms << std::endl;
   prof_logf("[PROF] special_blocks partition=trie threshold=%u trie_nodes=%llu trie_children=%llu trie_bytes=%llu trie_build_ms=%.3f blocks=%u trivial_blocks=%u member_groups=%u member_points=%u child_block_edges=%u ms=%.3f",
             static_cast<unsigned>(_special_block_summary.threshold),
             static_cast<unsigned long long>(_special_block_summary.trie_node_count),
             static_cast<unsigned long long>(_special_block_summary.trie_child_count),
             static_cast<unsigned long long>(_special_block_summary.trie_serialized_bytes),
             _special_block_summary.trie_build_ms,
             static_cast<unsigned>(_special_block_summary.num_blocks),
             static_cast<unsigned>(_special_block_summary.trivial_blocks),
             static_cast<unsigned>(_special_block_summary.member_groups),
             static_cast<unsigned>(_special_block_summary.member_points),
             static_cast<unsigned>(_special_block_summary.child_block_edges),
             ms);
}

void UniNavGraph::build_special_block_index(
    const std::string &ung_index_path_prefix,
    const std::string &base_bin_file,
    const std::string &base_label_file,
    const std::string &block_index_path_prefix,
    const std::string &results_path_prefix,
    const std::string &data_type,
    const std::shared_ptr<DistanceHandler> &distance_handler,
    uint32_t num_threads,
    const UngBuildConfig &build_config,
    IdxType max_degree,
    IdxType num_cross_edges,
    IdxType Lbuild,
    float alpha)
{
   namespace fs = std::filesystem;
   const auto normalize_prefix = [](std::string prefix) {
      if (!prefix.empty() && prefix.back() != '/' && prefix.back() != '\\')
         prefix.push_back('/');
      return prefix;
   };
   const std::string ung_prefix = normalize_prefix(ung_index_path_prefix);
   const std::string block_prefix = normalize_prefix(block_index_path_prefix);
   const std::string result_prefix = normalize_prefix(results_path_prefix);
   const auto all_start = std::chrono::high_resolution_clock::now();
   _max_degree = max_degree;
   _num_cross_edges = num_cross_edges;
   _Lbuild = Lbuild;
   _alpha = alpha;
   _num_threads = std::max<uint32_t>(1, num_threads);
   _distance_handler = distance_handler;
   _build_config = build_config;
   _build_config.special_blocks_enabled = true;
   _build_config.special_block_partition = UngSpecialBlockPartition::Trie;
   _build_config.special_block_max_degree = max_degree;
   _build_config.special_block_num_cross_edges = num_cross_edges;
   _build_config.special_block_data_mode = "x1";

   const bool has_ung_source =
       fs::is_regular_file(ung_prefix + "meta") &&
       fs::is_regular_file(ung_prefix + "vecs.bin") &&
       fs::is_regular_file(ung_prefix + "labels.txt") &&
       fs::is_regular_file(ung_prefix + "group_id_to_label_set") &&
       fs::is_regular_file(ung_prefix + "group_id_to_range");
   bool reused_ung_source = has_ung_source;
   if (reused_ung_source)
   {
      try
      {
         const auto ung_meta = parse_kv_file(ung_prefix + "meta");
         const auto num_points_it = ung_meta.find("num_points");
         const auto num_groups_it = ung_meta.find("num_groups");
         if (num_points_it == ung_meta.end() || num_points_it->second.empty() ||
             num_groups_it == ung_meta.end() || num_groups_it->second.empty())
            throw std::runtime_error("UNG metadata is incomplete");
         _num_points = static_cast<IdxType>(std::stoul(num_points_it->second));
         _num_groups = static_cast<IdxType>(std::stoul(num_groups_it->second));
         _base_storage = create_storage(data_type, false);
         _base_storage->load_from_file(ung_prefix + "vecs.bin", ung_prefix + "labels.txt");
         if (_base_storage->get_num_points() != _num_points)
            throw std::runtime_error("UNG vector count does not match its metadata");
         load_2d_vectors(ung_prefix + "group_id_to_label_set", _group_id_to_label_set);
         load_2d_vectors(ung_prefix + "group_id_to_range", _group_id_to_range);
         if (_group_id_to_label_set.size() <= _num_groups || _group_id_to_range.size() <= _num_groups)
            throw std::runtime_error("UNG group metadata is incomplete");

         _new_vec_id_to_group_id.assign(_num_points, 0);
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            const auto &range = _group_id_to_range[group_id];
            if (range.first > range.second || range.second > _num_points)
               throw std::runtime_error("UNG group range is invalid");
            for (IdxType point_id = range.first; point_id < range.second; ++point_id)
               _new_vec_id_to_group_id[point_id] = group_id;
         }
      }
      catch (const std::exception &ex)
      {
         std::cerr << "[special_blocks] cannot reuse UNG index input: " << ex.what()
                   << "; falling back to dataset files." << std::endl;
         reused_ung_source = false;
         _base_storage.reset();
         _group_id_to_label_set.clear();
         _group_id_to_range.clear();
         _new_vec_id_to_group_id.clear();
      }
   }
   if (!reused_ung_source)
   {
      if (base_bin_file.empty() || base_label_file.empty())
         throw std::runtime_error(
             "UNG index input is incomplete; --base_bin_file and --base_label_file are required for dataset fallback");
      if (!fs::is_regular_file(base_bin_file) || !fs::is_regular_file(base_label_file))
         throw std::runtime_error("dataset fallback input files are unavailable");

      std::cout << "[special_blocks] UNG index input is incomplete; rebuilding block source from dataset files." << std::endl;
      _base_storage = create_storage(data_type, false);
      _base_storage->load_from_file(base_bin_file, base_label_file);
      _num_points = _base_storage->get_num_points();
      build_trie_and_divide_groups();
      _group_id_to_range.resize(_num_groups + 1);
      _new_vec_id_to_group_id.assign(_num_points, 0);
      _new_to_old_vec_ids.resize(_num_points);
      IdxType new_vec_id = 0;
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         _group_id_to_range[group_id].first = new_vec_id;
         for (IdxType old_vec_id : _group_id_to_vec_ids[group_id])
         {
            _new_to_old_vec_ids[new_vec_id] = old_vec_id;
            _new_vec_id_to_group_id[new_vec_id] = group_id;
            ++new_vec_id;
         }
         _group_id_to_range[group_id].second = new_vec_id;
      }
      if (new_vec_id != _num_points)
         throw std::runtime_error("dataset fallback failed to assign every vector to a label group");
      _base_storage->reorder_data(_new_to_old_vec_ids);
   }

   build_special_blocks();
   build_special_edge_overlay();
   build_special_trie_regular_edge_overlay();
   refresh_special_block_memory_stats();
   _special_block_summary.build_memory_logical_bytes =
       _special_block_summary.memory_logical_bytes;
   _special_block_summary.build_memory_allocated_bytes =
       _special_block_summary.memory_allocated_bytes;

   fs::create_directories(block_prefix);
   fs::create_directories(result_prefix);
   save_special_blocks(block_prefix);

   const auto compact_reload_start = std::chrono::high_resolution_clock::now();
   std::vector<std::vector<SpecialEdge>>().swap(_special_edges_by_point);
   std::vector<std::vector<SpecialEdge>>().swap(_special_heavy_edges_by_point);
   std::vector<std::vector<IdxType>>().swap(_special_trie_regular_edges_by_point);
   std::string reload_error;
   if (!read_regular_edge_csr_file(
           block_prefix + "special_trie_regular_edges.bin",
           _special_trie_regular_edges_csr, reload_error))
      throw std::runtime_error("cannot reload saved regular-edge CSR: " + reload_error);
   if (!read_special_edge_csr_file(
           block_prefix + "special_edges.bin", _special_edges_csr, reload_error))
      throw std::runtime_error("cannot reload saved special-edge CSR: " + reload_error);
   const std::string heavy_path = block_prefix + "special_heavy_edges.bin";
   if (fs::exists(heavy_path) &&
       !read_special_edge_csr_file(heavy_path, _special_heavy_edges_csr, reload_error))
      throw std::runtime_error("cannot reload saved heavy-edge CSR: " + reload_error);
   refresh_special_block_memory_stats();
   _special_block_summary.compact_reload_ms = std::chrono::duration<double, std::milli>(
                                                   std::chrono::high_resolution_clock::now() - compact_reload_start)
                                                   .count();

   std::map<std::string, std::string> meta;
   meta["index_format"] = _special_block_summary.upper_blocks > 0
                                ? "special_block_trie_multilevel_v1"
                                : "special_block_trie_v2";
   meta["edge_storage_format"] = "csr_v2";
   meta["source_ung_index"] = reused_ung_source ? ung_prefix : "dataset_fallback";
   meta["source_input"] = reused_ung_source ? "ung_index" : "dataset_fallback";
   meta["source_ung_fingerprint"] = block_source_fingerprint(
       _num_points, _num_groups, _group_id_to_label_set, _group_id_to_range);
   meta["num_points"] = std::to_string(_num_points);
   meta["num_groups"] = std::to_string(_num_groups);
   meta["special_blocks_enabled"] = "1";
   meta["special_block_partition"] = "trie";
   meta["special_block_min_points"] =
       std::to_string(_build_config.special_block_min_points);
   meta["special_block_upper_min_points"] =
       std::to_string(_special_block_summary.upper_threshold);
   meta["special_block_max_degree"] = std::to_string(max_degree);
   meta["special_block_num_cross_edges"] = std::to_string(num_cross_edges);
   meta["special_block_count"] = std::to_string(_special_block_summary.num_blocks);
   meta["special_block_upper_count"] = std::to_string(_special_block_summary.upper_blocks);
   meta["special_block_trivial_count"] = std::to_string(_special_block_summary.trivial_blocks);
   meta["special_block_member_groups"] = std::to_string(_special_block_summary.member_groups);
   meta["special_block_member_points"] = std::to_string(_special_block_summary.member_points);
   meta["special_block_child_edges"] = std::to_string(_special_block_summary.child_block_edges);
   meta["special_edge_count"] = std::to_string(_special_block_summary.special_edges);
   meta["special_edge_intra_count"] = std::to_string(_special_block_summary.intra_special_edges);
   meta["special_edge_inter_count"] = std::to_string(_special_block_summary.inter_special_edges);
   meta["special_trie_regular_group_edge_count"] =
       std::to_string(_special_block_summary.trie_regular_group_edges);
   meta["special_trie_regular_vector_edge_count"] =
       std::to_string(_special_block_summary.trie_regular_vector_edges);
   meta["special_trie_regular_portal_edge_count"] =
       std::to_string(_special_block_summary.trie_regular_portal_edges);
   meta["special_block_trie_format_version"] =
       std::to_string(SpecialBlockTrieIndex::kFormatVersion);
   meta["special_block_trie_nodes"] = std::to_string(_special_block_summary.trie_node_count);
   meta["special_block_trie_children"] = std::to_string(_special_block_summary.trie_child_count);
   meta["special_block_trie_serialized_bytes"] =
       std::to_string(_special_block_summary.trie_serialized_bytes);
   meta["special_block_metadata_time(ms)"] = std::to_string(_special_block_summary.metadata_ms);
   meta["special_block_trie_build_time(ms)"] = std::to_string(_special_block_summary.trie_build_ms);
   meta["special_edge_overlay_time(ms)"] = std::to_string(_special_block_summary.edge_overlay_ms);
   meta["special_edge_intra_build_time(ms)"] = std::to_string(_special_block_summary.intra_edge_build_ms);
   meta["special_edge_inter_build_time(ms)"] = std::to_string(_special_block_summary.inter_edge_build_ms);
   meta["special_trie_regular_edge_build_time(ms)"] =
       std::to_string(_special_block_summary.trie_regular_edge_build_ms);
   meta["special_blocks_save_time(ms)"] = std::to_string(_special_block_summary.save_total_ms);
   meta["build_memory_logical_bytes"] =
       std::to_string(_special_block_summary.build_memory_logical_bytes);
   meta["build_memory_allocated_bytes"] =
       std::to_string(_special_block_summary.build_memory_allocated_bytes);
   meta["loaded_memory_logical_bytes"] =
       std::to_string(_special_block_summary.memory_logical_bytes);
   meta["loaded_memory_allocated_bytes"] =
       std::to_string(_special_block_summary.memory_allocated_bytes);
   meta["compact_reload_time(ms)"] = std::to_string(_special_block_summary.compact_reload_ms);
   meta["disk_bytes"] = "0";
   meta["build_time(ms)"] = "0";
   write_kv_file(block_prefix + "meta", meta);

   _special_block_summary.disk_bytes = special_block_disk_bytes(block_prefix);
   const double total_build_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::high_resolution_clock::now() - all_start)
                                     .count();
   meta["disk_bytes"] = std::to_string(_special_block_summary.disk_bytes);
   meta["build_time(ms)"] = std::to_string(total_build_ms);
   write_kv_file(block_prefix + "meta", meta);
   _special_block_summary.disk_bytes = special_block_disk_bytes(block_prefix);
   meta["disk_bytes"] = std::to_string(_special_block_summary.disk_bytes);
   write_kv_file(block_prefix + "meta", meta);

   std::ofstream timing(result_prefix + "build_time.csv");
   timing << "Metric,Value\n"
          << "special_block_metadata_time," << _special_block_summary.metadata_ms << '\n'
          << "special_block_trie_build_time," << _special_block_summary.trie_build_ms << '\n'
          << "special_edge_intra_build_time," << _special_block_summary.intra_edge_build_ms << '\n'
          << "special_edge_inter_build_time," << _special_block_summary.inter_edge_build_ms << '\n'
          << "special_edge_overlay_time," << _special_block_summary.edge_overlay_ms << '\n'
          << "special_trie_regular_edge_build_time," << _special_block_summary.trie_regular_edge_build_ms << '\n'
          << "special_blocks_save_time," << _special_block_summary.save_total_ms << '\n'
          << "disk_bytes," << _special_block_summary.disk_bytes << '\n'
          << "build_memory_logical_bytes," << _special_block_summary.build_memory_logical_bytes << '\n'
          << "build_memory_allocated_bytes," << _special_block_summary.build_memory_allocated_bytes << '\n'
          << "loaded_memory_logical_bytes," << _special_block_summary.memory_logical_bytes << '\n'
          << "loaded_memory_allocated_bytes," << _special_block_summary.memory_allocated_bytes << '\n'
          << "compact_reload_time," << _special_block_summary.compact_reload_ms << '\n'
          << "total_time," << total_build_ms
          << '\n';
   std::cout << "[special_block_index][build] total_ms=" << total_build_ms
             << " disk_bytes=" << _special_block_summary.disk_bytes
             << " build_memory_allocated_bytes=" << _special_block_summary.build_memory_allocated_bytes
             << " loaded_memory_logical_bytes=" << _special_block_summary.memory_logical_bytes
             << " loaded_memory_allocated_bytes=" << _special_block_summary.memory_allocated_bytes
             << " compact_reload_ms=" << _special_block_summary.compact_reload_ms
             << std::endl;
}

void UniNavGraph::load_special_block_index(const std::string &block_index_path_prefix)
{
   const auto load_start = std::chrono::high_resolution_clock::now();
   std::string prefix = block_index_path_prefix;
   if (!prefix.empty() && prefix.back() != '/' && prefix.back() != '\\')
      prefix.push_back('/');
   const auto block_meta = parse_kv_file(prefix + "meta");
   const auto format_it = block_meta.find("index_format");
   if (format_it == block_meta.end())
      throw std::runtime_error(
          "independent special block metadata is missing index_format");
   if (!is_supported_special_block_index_format(format_it->second))
      throw std::runtime_error("unsupported special block index format: " + format_it->second);
   const auto points_it = block_meta.find("num_points");
   const auto groups_it = block_meta.find("num_groups");
   if (points_it == block_meta.end() || groups_it == block_meta.end())
      throw std::runtime_error("special block metadata is missing num_points or num_groups");
   if (static_cast<IdxType>(std::stoul(points_it->second)) != _num_points ||
       static_cast<IdxType>(std::stoul(groups_it->second)) != _num_groups)
      throw std::runtime_error("special block index does not match the loaded UNG index");
   const auto fingerprint_it = block_meta.find("source_ung_fingerprint");
   if (fingerprint_it == block_meta.end())
      throw std::runtime_error("independent special block metadata is missing its UNG fingerprint");
   if (fingerprint_it != block_meta.end() &&
       fingerprint_it->second != block_source_fingerprint(
                                    _num_points, _num_groups,
                                    _group_id_to_label_set, _group_id_to_range))
      throw std::runtime_error("special block index was built from a different UNG group layout");
   load_special_blocks(prefix, block_meta, true);
   refresh_special_block_memory_stats();
   _special_block_summary.disk_bytes = special_block_disk_bytes(prefix);
   _special_block_summary.load_total_ms = std::chrono::duration<double, std::milli>(
                                                 std::chrono::high_resolution_clock::now() - load_start)
                                                 .count();
   std::cout << "[special_block_index][load] total_ms=" << _special_block_summary.load_total_ms
             << " disk_bytes=" << _special_block_summary.disk_bytes
             << " memory_logical_bytes=" << _special_block_summary.memory_logical_bytes
             << " memory_allocated_bytes=" << _special_block_summary.memory_allocated_bytes
             << std::endl;
}

void UniNavGraph::build_special_trie_regular_edge_overlay()
{
   _special_trie_regular_edges_by_point.clear();
   _special_trie_regular_edges_available = false;
   _special_block_summary.trie_regular_group_edges = 0;
   _special_block_summary.trie_regular_vector_edges = 0;
   _special_block_summary.trie_regular_portal_edges = 0;
   _special_block_summary.trie_regular_edge_build_ms = 0.0;

   if (_build_config.special_block_partition != UngSpecialBlockPartition::Trie ||
       _special_block_trie_index.empty() || !_base_storage)
      return;

   const auto start = std::chrono::high_resolution_clock::now();
   std::vector<std::vector<IdxType>> group_successors(_num_groups + 1);
   for (const auto &pair : _special_block_trie_index.terminal_successor_pairs())
   {
      if (pair.first == 0 || pair.first > _num_groups ||
          pair.second == 0 || pair.second > _num_groups || pair.first == pair.second)
         continue;
      group_successors[pair.first].push_back(pair.second);
   }

   const auto portal_pairs = _special_block_trie_index.terminal_block_portal_pairs();
   uint64_t portal_vector_edge_count = 0;
   for (const auto &pair : portal_pairs)
   {
      const IdxType source_group = pair.first;
      const IdxType block_id = pair.second;
      if (source_group == 0 || source_group > _num_groups ||
          block_id == 0 || block_id > _special_blocks.size())
         continue;
      const IdxType entry_point = _special_blocks[block_id - 1].entry_point_id;
      if (entry_point == SpecialBlock::kInvalidEntryPoint || entry_point >= _num_points)
         throw std::runtime_error(
             "special Trie portal requires a valid block-local graph entry point");
      const auto &source_range = _group_id_to_range[source_group];
      portal_vector_edge_count += source_range.second - source_range.first;
   }

   uint64_t successor_group_edge_count = 0;
   for (auto &successors : group_successors)
   {
      std::sort(successors.begin(), successors.end());
      successors.erase(std::unique(successors.begin(), successors.end()), successors.end());
      successor_group_edge_count += successors.size();
   }
   const uint64_t portal_group_edge_count = portal_pairs.size();
   const uint64_t group_edge_count = successor_group_edge_count + portal_group_edge_count;

   _special_trie_regular_edges_by_point.assign(_num_points, {});
   const IdxType edge_degree = std::max<IdxType>(
       1, _build_config.special_block_num_cross_edges > 0
              ? static_cast<IdxType>(_build_config.special_block_num_cross_edges)
              : _num_cross_edges);
   const IdxType dim = _base_storage->get_dim();
   uint64_t vector_edge_count = 0;
#pragma omp parallel for schedule(dynamic, 64) num_threads(_num_threads) reduction(+ : vector_edge_count)
   for (IdxType source_group = 1; source_group <= _num_groups; ++source_group)
   {
      const auto &successors = group_successors[source_group];
      if (successors.empty())
         continue;
      const auto &source_range = _group_id_to_range[source_group];
      for (IdxType source = source_range.first; source < source_range.second; ++source)
      {
         auto &output = _special_trie_regular_edges_by_point[source];
         output.reserve(successors.size() * static_cast<size_t>(edge_degree));
         const char *source_vector = _base_storage->get_vector(source);
         for (IdxType target_group : successors)
         {
            const auto &target_range = _group_id_to_range[target_group];
            SearchQueue nearest;
            nearest.reserve(edge_degree);
            for (IdxType target = target_range.first; target < target_range.second; ++target)
            {
               nearest.insert(target,
                              _distance_handler->compute(
                                  source_vector, _base_storage->get_vector(target), dim));
            }
            for (int32_t rank = 0; rank < nearest.size(); ++rank)
               output.push_back(nearest[rank].id);
         }
         std::sort(output.begin(), output.end());
         output.erase(std::unique(output.begin(), output.end()), output.end());
         vector_edge_count += output.size();
      }
   }

   _special_trie_regular_edges_available = true;
   _special_block_summary.trie_regular_group_edges = group_edge_count;
   _special_block_summary.trie_regular_vector_edges = vector_edge_count;
   _special_block_summary.trie_regular_portal_edges = portal_vector_edge_count;
   _special_block_summary.trie_regular_edge_build_ms =
       std::chrono::duration<double, std::milli>(
           std::chrono::high_resolution_clock::now() - start)
           .count();
   std::cout << "[special_trie_regular] group_edges=" << group_edge_count
             << " successor_group_edges=" << successor_group_edge_count
             << " portal_group_edges=" << portal_group_edge_count
             << " vector_edges=" << vector_edge_count
             << " implicit_portal_vector_edges=" << portal_vector_edge_count
             << " degree_per_group=" << edge_degree
             << " build_ms=" << _special_block_summary.trie_regular_edge_build_ms
             << std::endl;
}

void UniNavGraph::build_special_edge_overlay()
{
   _special_edges_by_point.assign(_num_points, {});
   _special_heavy_edges_by_point.clear();
   _special_block_summary.special_edges = 0;
   _special_block_summary.intra_special_edges = 0;
   _special_block_summary.inter_special_edges = 0;

   if (_special_blocks.empty())
      return;

   const auto overlay_start = std::chrono::high_resolution_clock::now();
   const auto block_indexes_prepare_start = std::chrono::high_resolution_clock::now();
   std::vector<std::vector<IdxType>> block_points(_special_blocks.size() + 1);
   for (const SpecialBlock &block : _special_blocks)
      if (block.block_id < block_points.size())
         block_points[block.block_id] = collect_block_points(block, _group_id_to_range);
   std::vector<std::shared_ptr<Vamana>> block_indexes(_special_blocks.size() + 1);
   _special_block_summary.block_indexes_prepare_ms = std::chrono::duration<double, std::milli>(
                                                         std::chrono::high_resolution_clock::now() - block_indexes_prepare_start)
                                                         .count();

   double intra_ms = 0.0;
   double inter_ms = 0.0;
   const bool gpu_intra_enabled = env_flag("UNG_SPECIAL_BLOCK_GPU_INTRA");
   if (gpu_intra_enabled && _build_config.tagore_k <= _max_degree)
   {
      const std::string msg = "special block GPU intra requires UNG_TAGORE_K > max_degree; lower values currently trigger GPU fallback and are unsupported.";
      std::cerr << "[special_edges][GPU_INTRA][unsupported_config] " << msg << std::endl;
      throw std::runtime_error(msg);
   }
   const IdxType special_block_max_degree = _build_config.special_block_max_degree > 0
                                             ? static_cast<IdxType>(_build_config.special_block_max_degree)
                                             : _max_degree;
   const IdxType special_block_num_cross_edges = _build_config.special_block_num_cross_edges > 0
                                                ? static_cast<IdxType>(_build_config.special_block_num_cross_edges)
                                                : _num_cross_edges;
   const CpuGroupGraphSettings group_cfg = make_cpu_group_graph_settings(special_block_max_degree, _num_threads);
   const IdxType legacy_special_complete_threshold =
       std::max<IdxType>(group_cfg.complete_threshold,
                         static_cast<IdxType>(_build_config.special_block_min_points));
   const IdxType cpu_special_complete_default =
       std::max<IdxType>(legacy_special_complete_threshold,
                         static_cast<IdxType>(2048));
   const IdxType gpu_special_complete_default =
       std::max<IdxType>(legacy_special_complete_threshold, static_cast<IdxType>(128));
   const IdxType special_intra_complete_threshold = read_env_idxtype(
       "UNG_SPECIAL_INTRA_COMPLETE_NX",
       gpu_intra_enabled ? gpu_special_complete_default : cpu_special_complete_default,
       special_block_max_degree,
       1 << 20);
   const bool special_intra_bounded_complete =
       read_env_bool_default("UNG_SPECIAL_INTRA_BOUNDED_COMPLETE", true);
   const bool special_intra_exact_topk =
       read_env_bool_default("UNG_SPECIAL_INTRA_EXACT_TOPK", true);
   const bool special_intra_route =
       read_env_bool_default("UNG_SPECIAL_INTRA_ROUTE", false);
   const IdxType special_intra_small_nx =
       read_env_idxtype("UNG_SPECIAL_INTRA_SMALL_NX",
                        special_intra_complete_threshold,
                        special_block_max_degree,
                        1 << 20);
   const IdxType special_intra_large_nx =
       read_env_idxtype("UNG_SPECIAL_INTRA_LARGE_NX",
                        std::max<IdxType>(special_intra_small_nx + 1, 8192),
                        special_intra_small_nx + 1,
                        1 << 20);
   const std::string special_intra_large_backend =
       read_env_string("UNG_SPECIAL_INTRA_LARGE_BACKEND", "jasper_style");
   const unsigned long long intra_sample_candidates_env =
       read_env_ull("UNG_SPECIAL_INTRA_SAMPLE_CANDIDATES", 0);
   const unsigned long long intra_mid_sample_candidates_env =
       read_env_ull("UNG_SPECIAL_INTRA_MID_SAMPLE_CANDIDATES",
                    intra_sample_candidates_env == 0 ? 256ULL : intra_sample_candidates_env);
   const IdxType intra_sample_min_n =
       read_env_idxtype("UNG_SPECIAL_INTRA_SAMPLE_MIN_N", 1, 1, 1 << 20);
   const std::string intra_sample_mode =
       read_env_string("UNG_SPECIAL_INTRA_SAMPLE_MODE", "topk");
   const bool intra_sample_uses_vamana =
       intra_sample_mode == "vamana" || intra_sample_mode == "vamana_prune";
   const bool intra_sample_uses_hashprune = intra_sample_mode == "hashprune";
   const IdxType hashprune_leaf_size =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_LEAF_SIZE", 256, 2, 1 << 20);
   const IdxType hashprune_replicas =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_REPLICAS", 2, 1, 1024);
   const IdxType hashprune_reservoir =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_RESERVOIR", std::max<IdxType>(special_block_max_degree * 4, 128), special_block_max_degree, 1 << 20);
   const IdxType hashprune_leaf_topk =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_LEAF_TOPK", special_block_max_degree, special_block_max_degree, 1 << 20);
   const IdxType hashprune_fanout =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_FANOUT", 1, 1, 64);
   const IdxType hashprune_hash_bits =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_HASH_BITS", 8, 1, 24);
   const IdxType hashprune_hash_sample_dims =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_HASH_SAMPLE_DIMS", 32, 1, 1 << 20);
   const IdxType hashprune_bucket_cap =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_BUCKET_CAP", 2, 1, 1024);
   const IdxType hashprune_nearest =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_NEAREST", special_block_max_degree, 0, 1 << 20);
   const IdxType hashprune_mix_sample =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_MIX_SAMPLE", 0, 0, 1 << 20);
   const IdxType hashprune_repair_degree =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_REPAIR_DEGREE", 0, 0, 1 << 20);
   const IdxType hashprune_repair_sample =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_REPAIR_SAMPLE", 0, 0, 1 << 20);
   const IdxType hashprune_repair_max_points =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_REPAIR_MAX_POINTS", 0, 0, 1 << 30);
   const bool hashprune_nav_repair = read_env_bool_default("UNG_SPECIAL_HASHPRUNE_NAV_REPAIR", false);
   const IdxType hashprune_repair_2hop_min =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_REPAIR_2HOP_MIN", 0, 0, 1 << 20);
   const IdxType hashprune_repair_graph_hops =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_REPAIR_GRAPH_HOPS", 0, 0, 1 << 20);
   const IdxType hashprune_repair_hash_near =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_REPAIR_HASH_NEAR", 0, 0, 1 << 20);
   const IdxType hashprune_repair_reservoir_near =
       read_env_idxtype("UNG_SPECIAL_HASHPRUNE_REPAIR_RESERVOIR_NEAR", 0, 0, 1 << 20);
   const bool hashprune_diag = read_env_bool_default("UNG_SPECIAL_HASHPRUNE_DIAG", true);
   size_t complete_intra_blocks = 0;
   size_t complete_intra_points = 0;
   size_t sampled_intra_blocks = 0;
   size_t sampled_intra_points = 0;
   size_t route_small_blocks = 0;
   size_t route_small_points = 0;
   size_t route_mid_blocks = 0;
   size_t route_mid_points = 0;
   size_t route_large_blocks = 0;
   size_t route_large_points = 0;
   unsigned long long hashprune_edges_streamed = 0;
   HashPruneGraphDiagnostics hashprune_diag_stats;
   size_t cpu_intra_vamana_blocks = 0;
   size_t cpu_intra_vamana_points = 0;
   size_t gpu_intra_blocks = 0;
   size_t gpu_intra_points = 0;
   size_t gpu_intra_fallback_blocks = 0;
   bool route_large_gpu_disabled = false;
   const auto intra_start = std::chrono::high_resolution_clock::now();
   for (SpecialBlock &block : _special_blocks)
   {
      if (block.block_id >= block_points.size())
         continue;
      const auto &points = block_points[block.block_id];
      const IdxType n = static_cast<IdxType>(points.size());
      block.entry_point_id = SpecialBlock::kInvalidEntryPoint;
      if (n == 0)
         continue;
      if (n == 1)
      {
         block.entry_point_id = points.front();
         continue;
      }

      auto local_graph = std::make_shared<Graph>(n);
      auto local_storage = std::make_shared<SpecialBlockStorageView>(_base_storage, points);
      bool routed_intra = false;
      if (special_intra_route)
      {
         routed_intra = true;
         if (n <= special_intra_small_nx)
         {
            if (special_intra_exact_topk)
               build_exact_topk_graph_for_points(local_graph, _base_storage, _distance_handler, points, special_block_max_degree);
            else if (special_intra_bounded_complete)
               build_bounded_complete_graph(local_graph, n, special_block_max_degree);
            else
               build_complete_graph(local_graph, n);
            block_indexes[block.block_id] = std::make_shared<Vamana>(
                local_storage,
                _distance_handler,
                local_graph,
                0);
            complete_intra_blocks += 1;
            complete_intra_points += static_cast<size_t>(n);
            route_small_blocks += 1;
            route_small_points += static_cast<size_t>(n);
         }
         else if (n < special_intra_large_nx)
         {
            const size_t mid_sample_candidates =
                std::min<size_t>(std::max<size_t>(static_cast<size_t>(special_block_max_degree),
                                                  static_cast<size_t>(intra_mid_sample_candidates_env)),
                                 static_cast<size_t>(n - 1));
            block_indexes[block.block_id] = build_sampled_vamana_graph_for_points(
                local_graph, local_storage, _distance_handler, special_block_max_degree, _Lbuild, _alpha,
                _num_threads, default_paras::MAX_CANDIDATE_SIZE, mid_sample_candidates);
            sampled_intra_blocks += 1;
            sampled_intra_points += static_cast<size_t>(n);
            route_mid_blocks += 1;
            route_mid_points += static_cast<size_t>(n);
         }
         else
         {
            bool built_by_gpu = false;
            if (!route_large_gpu_disabled &&
                (special_intra_large_backend == "jasper_style" ||
                 special_intra_large_backend == "tagore" ||
                 special_intra_large_backend == "gpu"))
            {
               try
               {
                  if (_build_config.tagore_k <= special_block_max_degree)
                     throw std::runtime_error("large-block CUDA route requires UNG_TAGORE_K > UNG_SPECIAL_BLOCK_MAX_DEGREE");
                  std::vector<float> packed = pack_special_block_vectors(_base_storage, points);
                  TagoreGroupRequest request{packed.data(), static_cast<uint32_t>(n)};
                  const TagoreCudaRuntimeConfig runtime_cfg =
                      make_tagore_cuda_runtime_config(_build_config.tagore_k,
                                                      static_cast<uint32_t>(special_block_max_degree),
                                                      false);
                  TagoreBatchBuildResult batch = build_tagore_vamana_cuda_batch(
                      std::vector<TagoreGroupRequest>{request},
                      static_cast<uint32_t>(_base_storage->get_dim()),
                      _build_config.tagore_k,
                      static_cast<uint32_t>(special_block_max_degree),
                      _build_config.tagore_m,
                      _build_config.tagore_iter,
                      _alpha,
                      TagorePruneMode::FastGrnnd,
                      runtime_cfg);
                  if (batch.groups.empty())
                     throw std::runtime_error("empty CUDA batch result for special block");
                  fill_special_graph_from_tagore_result(local_graph,
                                                        batch.groups.front(),
                                                        n,
                                                        special_block_max_degree,
                                                        _build_config.tagore_k);
                  block_indexes[block.block_id] = std::make_shared<Vamana>(
                      local_storage,
                      _distance_handler,
                      local_graph,
                      batch.groups.front().entry_point < n ? batch.groups.front().entry_point : 0);
                  gpu_intra_blocks += 1;
                  gpu_intra_points += static_cast<size_t>(n);
                  built_by_gpu = true;
               }
               catch (const std::exception &ex)
               {
                  gpu_intra_fallback_blocks += 1;
                  route_large_gpu_disabled = true;
                  std::cerr << "[special_edges][INTRA_ROUTE][large_fallback] block=" << block.block_id
                            << " n=" << n << " backend=" << special_intra_large_backend
                            << " disable_remaining_gpu=1"
                            << " reason=" << ex.what() << std::endl;
               }
            }
            if (!built_by_gpu)
            {
               auto local_index = std::make_shared<Vamana>(false);
               local_index->build(local_storage, _distance_handler, local_graph, special_block_max_degree, _Lbuild, _alpha, 1);
               block_indexes[block.block_id] = local_index;
               cpu_intra_vamana_blocks += 1;
               cpu_intra_vamana_points += static_cast<size_t>(n);
            }
            route_large_blocks += 1;
            route_large_points += static_cast<size_t>(n);
         }
      }
      if (!routed_intra && n <= special_intra_complete_threshold)
      {
         const size_t intra_sample_candidates = intra_sample_candidates_env == 0
                                                  ? 0
                                                  : std::max<size_t>(static_cast<size_t>(special_block_max_degree),
                                                                     static_cast<size_t>(intra_sample_candidates_env));
         const bool use_intra_sampling = intra_sample_candidates > 0 &&
                                         n >= intra_sample_min_n &&
                                         intra_sample_candidates < static_cast<size_t>(n - 1) &&
                                         (special_intra_exact_topk || intra_sample_uses_vamana || intra_sample_uses_hashprune);
         if (intra_sample_uses_hashprune)
         {
            unsigned long long block_hashprune_edges_streamed = 0;
            block_indexes[block.block_id] = build_hashprune_graph_for_points(
                local_graph, local_storage, _distance_handler, special_block_max_degree, _Lbuild, _alpha,
                _num_threads, default_paras::MAX_CANDIDATE_SIZE,
                static_cast<size_t>(hashprune_leaf_size), static_cast<size_t>(hashprune_replicas),
                static_cast<size_t>(hashprune_reservoir), static_cast<size_t>(hashprune_leaf_topk),
                static_cast<size_t>(hashprune_fanout), static_cast<uint32_t>(hashprune_hash_bits),
                hashprune_hash_sample_dims, static_cast<size_t>(hashprune_bucket_cap),
                static_cast<size_t>(hashprune_nearest), static_cast<size_t>(hashprune_mix_sample),
                static_cast<size_t>(hashprune_repair_degree), static_cast<size_t>(hashprune_repair_sample),
                static_cast<size_t>(hashprune_repair_max_points), hashprune_nav_repair,
                static_cast<size_t>(hashprune_repair_2hop_min), static_cast<size_t>(hashprune_repair_graph_hops),
                static_cast<size_t>(hashprune_repair_hash_near), static_cast<size_t>(hashprune_repair_reservoir_near),
                hashprune_diag, hashprune_diag_stats, block_hashprune_edges_streamed);
            hashprune_edges_streamed += block_hashprune_edges_streamed;
            sampled_intra_blocks += 1;
            sampled_intra_points += static_cast<size_t>(n);
         }
         else if (use_intra_sampling && intra_sample_uses_vamana)
         {
            block_indexes[block.block_id] = build_sampled_vamana_graph_for_points(
                local_graph, local_storage, _distance_handler, special_block_max_degree, _Lbuild, _alpha,
                _num_threads, default_paras::MAX_CANDIDATE_SIZE, intra_sample_candidates);
            sampled_intra_blocks += 1;
            sampled_intra_points += static_cast<size_t>(n);
         }
         else
         {
            if (use_intra_sampling)
            {
               build_sampled_topk_graph_for_points(local_graph, _base_storage, _distance_handler, points,
                                                   special_block_max_degree, intra_sample_candidates);
               sampled_intra_blocks += 1;
               sampled_intra_points += static_cast<size_t>(n);
            }
            else if (special_intra_exact_topk)
               build_exact_topk_graph_for_points(local_graph, _base_storage, _distance_handler, points, special_block_max_degree);
            else if (special_intra_bounded_complete)
               build_bounded_complete_graph(local_graph, n, special_block_max_degree);
            else
               build_complete_graph(local_graph, n);
            block_indexes[block.block_id] = std::make_shared<Vamana>(
                local_storage,
                _distance_handler,
                local_graph,
                0);
         }
         complete_intra_blocks += 1;
         complete_intra_points += static_cast<size_t>(n);
      }
      else if (!routed_intra)
      {
         bool built_by_gpu = false;
         if (gpu_intra_enabled && n > special_intra_complete_threshold)
         {
            try
            {
               std::vector<float> packed = pack_special_block_vectors(_base_storage, points);
               TagoreGroupRequest request{packed.data(), static_cast<uint32_t>(n)};
               const TagoreCudaRuntimeConfig runtime_cfg =
                   make_tagore_cuda_runtime_config(_build_config.tagore_k,
                                                   static_cast<uint32_t>(special_block_max_degree),
                                                   false);
               TagoreBatchBuildResult batch = build_tagore_vamana_cuda_batch(
                   std::vector<TagoreGroupRequest>{request},
                   static_cast<uint32_t>(_base_storage->get_dim()),
                   _build_config.tagore_k,
                   static_cast<uint32_t>(special_block_max_degree),
                   _build_config.tagore_m,
                   _build_config.tagore_iter,
                   _alpha,
                   TagorePruneMode::FastGrnnd,
                   runtime_cfg);
               if (batch.groups.empty())
                  throw std::runtime_error("empty Tagore batch result for special block");
               fill_special_graph_from_tagore_result(local_graph,
                                                     batch.groups.front(),
                                                     n,
                                                     special_block_max_degree,
                                                     _build_config.tagore_k <= special_block_max_degree
                                                         ? static_cast<uint32_t>(special_block_max_degree + 1)
                                                         : _build_config.tagore_k);
               block_indexes[block.block_id] = std::make_shared<Vamana>(
                   std::make_shared<SpecialBlockStorageView>(_base_storage, points),
                   _distance_handler,
                   local_graph,
                   batch.groups.front().entry_point < n ? batch.groups.front().entry_point : 0);
               gpu_intra_blocks += 1;
               gpu_intra_points += static_cast<size_t>(n);
               built_by_gpu = true;
            }
            catch (const std::exception &ex)
            {
               gpu_intra_fallback_blocks += 1;
               std::cerr << "[special_edges][GPU_INTRA] fallback block=" << block.block_id
                         << " n=" << n << " reason=" << ex.what() << std::endl;
            }
         }
         if (!built_by_gpu)
         {
            auto local_storage = std::make_shared<SpecialBlockStorageView>(_base_storage, points);
            auto local_index = std::make_shared<Vamana>(false);
            local_index->build(local_storage, _distance_handler, local_graph, special_block_max_degree, _Lbuild, _alpha, 1);
            block_indexes[block.block_id] = local_index;
            cpu_intra_vamana_blocks += 1;
            cpu_intra_vamana_points += static_cast<size_t>(n);
         }
      }

      const auto &block_index = block_indexes[block.block_id];
      if (!block_index)
         throw std::runtime_error("special block local graph did not produce an index");
      const IdxType local_entry_point = block_index->get_entry_point();
      if (local_entry_point >= n)
         throw std::runtime_error("special block local graph produced an invalid entry point");
      block.entry_point_id = points[local_entry_point];

      for (IdxType local_id = 0; local_id < n; ++local_id)
      {
         const IdxType source = points[local_id];
         auto &special_edges = _special_edges_by_point[source];
         for (IdxType local_neighbor : local_graph->neighbors[local_id])
         {
            if (local_neighbor >= n)
               continue;
            const IdxType target = points[local_neighbor];
            special_edges.push_back({target, block.block_id, SpecialEdgeKind::IntraBlock});
            _special_block_summary.intra_special_edges += 1;
         }
      }
   }
   intra_ms = std::chrono::duration<double, std::milli>(
                  std::chrono::high_resolution_clock::now() - intra_start)
                  .count();
   std::cout << "[special_edges] gpu_intra_enabled=" << (gpu_intra_enabled ? 1 : 0)
             << " gpu_intra_blocks=" << gpu_intra_blocks
             << " gpu_intra_points=" << gpu_intra_points
             << " gpu_intra_fallback_blocks=" << gpu_intra_fallback_blocks
             << " group_complete_threshold_nx=" << group_cfg.complete_threshold
             << " special_complete_threshold_nx=" << special_intra_complete_threshold
             << " special_bounded_complete=" << (special_intra_bounded_complete ? 1 : 0)
             << " special_exact_topk=" << (special_intra_exact_topk ? 1 : 0)
             << " intra_route=" << (special_intra_route ? 1 : 0)
             << " intra_small_nx=" << special_intra_small_nx
             << " intra_large_nx=" << special_intra_large_nx
             << " intra_large_backend=" << special_intra_large_backend
             << " intra_mid_sample_candidates=" << intra_mid_sample_candidates_env
             << " route_small_blocks=" << route_small_blocks
             << " route_small_points=" << route_small_points
             << " route_mid_blocks=" << route_mid_blocks
             << " route_mid_points=" << route_mid_points
             << " route_large_blocks=" << route_large_blocks
             << " route_large_points=" << route_large_points
             << " route_large_gpu_disabled=" << (route_large_gpu_disabled ? 1 : 0)
             << " intra_sample_mode=" << intra_sample_mode
             << " intra_sample_candidates=" << intra_sample_candidates_env
             << " intra_sample_min_n=" << intra_sample_min_n
             << " hashprune_leaf_size=" << hashprune_leaf_size
             << " hashprune_replicas=" << hashprune_replicas
             << " hashprune_reservoir=" << hashprune_reservoir
             << " hashprune_leaf_topk=" << hashprune_leaf_topk
             << " hashprune_fanout=" << hashprune_fanout
             << " hashprune_hash_bits=" << hashprune_hash_bits
             << " hashprune_hash_sample_dims=" << hashprune_hash_sample_dims
             << " hashprune_bucket_cap=" << hashprune_bucket_cap
             << " hashprune_nearest=" << hashprune_nearest
             << " hashprune_mix_sample=" << hashprune_mix_sample
             << " hashprune_repair_degree=" << hashprune_repair_degree
             << " hashprune_repair_sample=" << hashprune_repair_sample
             << " hashprune_repair_max_points=" << hashprune_repair_max_points
             << " hashprune_nav_repair=" << (hashprune_nav_repair ? 1 : 0)
             << " hashprune_repair_2hop_min=" << hashprune_repair_2hop_min
             << " hashprune_repair_graph_hops=" << hashprune_repair_graph_hops
             << " hashprune_repair_hash_near=" << hashprune_repair_hash_near
             << " hashprune_repair_reservoir_near=" << hashprune_repair_reservoir_near
             << " hashprune_diag=" << (hashprune_diag ? 1 : 0)
             << " hashprune_edges_streamed=" << hashprune_edges_streamed
             << " sampled_blocks=" << sampled_intra_blocks
             << " sampled_points=" << sampled_intra_points
             << " complete_blocks=" << complete_intra_blocks
             << " complete_points=" << complete_intra_points
             << " cpu_vamana_blocks=" << cpu_intra_vamana_blocks
             << " cpu_vamana_points=" << cpu_intra_vamana_points
             << std::endl;
   if (hashprune_diag && hashprune_diag_stats.points > 0)
   {
      const double hashprune_avg_candidate_count =
          static_cast<double>(hashprune_diag_stats.candidate_count_sum) / hashprune_diag_stats.points;
      const double hashprune_avg_bucket_coverage =
          static_cast<double>(hashprune_diag_stats.bucket_coverage_sum) / hashprune_diag_stats.points;
      const double hashprune_final_avg_degree =
          static_cast<double>(hashprune_diag_stats.final_degree_sum) / hashprune_diag_stats.points;
      std::cout << "[special_edges][hashprune_diag] points=" << hashprune_diag_stats.points
                << " avg_candidates=" << hashprune_avg_candidate_count
                << " avg_bucket_coverage=" << hashprune_avg_bucket_coverage
                << " hashprune_repaired_points=" << hashprune_diag_stats.repaired_points
                << " hashprune_repair_candidates_added=" << hashprune_diag_stats.repair_candidates_added
                << " hashprune_low_2hop_points=" << hashprune_diag_stats.low_2hop_points
                << " hashprune_graph_repair_candidates_added=" << hashprune_diag_stats.graph_repair_candidates_added
                << " hashprune_hash_repair_candidates_added=" << hashprune_diag_stats.hash_repair_candidates_added
                << " hashprune_reservoir_repair_candidates_added=" << hashprune_diag_stats.reservoir_repair_candidates_added
                << " hashprune_sample_repair_candidates_added=" << hashprune_diag_stats.sample_repair_candidates_added
                << " final_avg_degree=" << hashprune_final_avg_degree
                << " min_degree=" << hashprune_diag_stats.min_degree
                << " max_degree=" << hashprune_diag_stats.max_degree
                << " low_degree_lt16=" << hashprune_diag_stats.low_degree_lt16
                << " low_degree_lt32=" << hashprune_diag_stats.low_degree_lt32
                << " full_degree_points=" << hashprune_diag_stats.full_degree_points
                << std::endl;
   }
   prof_logf("[PROF] special_edges.gpu_intra enabled=%d blocks=%zu points=%zu fallback_blocks=%zu group_complete_threshold_nx=%u special_complete_threshold_nx=%u special_bounded_complete=%d special_exact_topk=%d intra_route=%d intra_small_nx=%u intra_large_nx=%u intra_large_backend=%s intra_mid_sample_candidates=%llu route_small_blocks=%zu route_small_points=%zu route_mid_blocks=%zu route_mid_points=%zu route_large_blocks=%zu route_large_points=%zu intra_sample_mode=%s intra_sample_candidates=%llu intra_sample_min_n=%u hashprune_leaf_size=%u hashprune_replicas=%u hashprune_reservoir=%u hashprune_leaf_topk=%u hashprune_fanout=%u hashprune_hash_bits=%u hashprune_hash_sample_dims=%u hashprune_bucket_cap=%u hashprune_nearest=%u hashprune_mix_sample=%u hashprune_repair_degree=%u hashprune_repair_sample=%u hashprune_repair_max_points=%u hashprune_nav_repair=%u hashprune_repair_2hop_min=%u hashprune_repair_graph_hops=%u hashprune_repair_hash_near=%u hashprune_repair_reservoir_near=%u hashprune_diag=%u hashprune_edges_streamed=%llu sampled_blocks=%zu sampled_points=%zu complete_blocks=%zu complete_points=%zu cpu_vamana_blocks=%zu cpu_vamana_points=%zu",
             gpu_intra_enabled ? 1 : 0,
             gpu_intra_blocks,
             gpu_intra_points,
             gpu_intra_fallback_blocks,
             static_cast<unsigned>(group_cfg.complete_threshold),
             static_cast<unsigned>(special_intra_complete_threshold),
             special_intra_bounded_complete ? 1 : 0,
             special_intra_exact_topk ? 1 : 0,
             special_intra_route ? 1 : 0,
             static_cast<unsigned>(special_intra_small_nx),
             static_cast<unsigned>(special_intra_large_nx),
             special_intra_large_backend.c_str(),
             intra_mid_sample_candidates_env,
             route_small_blocks,
             route_small_points,
             route_mid_blocks,
             route_mid_points,
             route_large_blocks,
             route_large_points,
             intra_sample_mode.c_str(),
             intra_sample_candidates_env,
             static_cast<unsigned>(intra_sample_min_n),
             static_cast<unsigned>(hashprune_leaf_size),
             static_cast<unsigned>(hashprune_replicas),
             static_cast<unsigned>(hashprune_reservoir),
             static_cast<unsigned>(hashprune_leaf_topk),
             static_cast<unsigned>(hashprune_fanout),
             static_cast<unsigned>(hashprune_hash_bits),
             static_cast<unsigned>(hashprune_hash_sample_dims),
             static_cast<unsigned>(hashprune_bucket_cap),
             static_cast<unsigned>(hashprune_nearest),
             static_cast<unsigned>(hashprune_mix_sample),
             static_cast<unsigned>(hashprune_repair_degree),
             static_cast<unsigned>(hashprune_repair_sample),
             static_cast<unsigned>(hashprune_repair_max_points),
             static_cast<unsigned>(hashprune_nav_repair ? 1 : 0),
             static_cast<unsigned>(hashprune_repair_2hop_min),
             static_cast<unsigned>(hashprune_repair_graph_hops),
             static_cast<unsigned>(hashprune_repair_hash_near),
             static_cast<unsigned>(hashprune_repair_reservoir_near),
             static_cast<unsigned>(hashprune_diag ? 1 : 0),
             hashprune_edges_streamed,
             sampled_intra_blocks,
             sampled_intra_points,
             complete_intra_blocks,
             complete_intra_points,
             cpu_intra_vamana_blocks,
             cpu_intra_vamana_points);

   const CpuHybridCrossSettings hybrid_cfg = make_cpu_hybrid_cross_settings();
   const unsigned long long force_graph_pair_work = read_env_ull("UNG_SPECIAL_INTER_FORCE_GRAPH_PAIR_WORK", 10000);
   const IdxType inter_search_ef = read_env_idxtype(
       "UNG_SPECIAL_INTER_SEARCH_EF",
       std::max<IdxType>(_Lbuild, special_block_num_cross_edges),
       std::max<IdxType>(special_block_num_cross_edges, 1),
       1 << 20);
   const unsigned long long sample_candidates_env = read_env_ull("UNG_SPECIAL_INTER_SAMPLE_CANDIDATES", 0);
   const unsigned long long sample_min_pair_work = read_env_ull("UNG_SPECIAL_INTER_SAMPLE_MIN_PAIR_WORK", 10000);
   const unsigned long long inter_pair_work_cap = read_env_ull("UNG_SPECIAL_INTER_MAX_PAIR_WORK", 0);
   size_t inter_pair_work_cap_skipped_pairs = 0;
   unsigned long long inter_pair_work_cap_skipped_queries = 0;
   unsigned long long inter_pair_work_cap_skipped_work = 0;
   const auto inter_start = std::chrono::high_resolution_clock::now();
   const bool gpu_inter_enabled = env_flag("UNG_SPECIAL_BLOCK_GPU_INTER");
   bool gpu_inter_used = false;
   double gpu_inter_ms = 0.0;
   if (gpu_inter_enabled)
   {
      try
      {
         gpu_inter_used = gpu_build_special_inter_edges(_special_blocks,
                                                        block_points,
                                                        _special_edges_by_point,
                                                        _special_block_summary.inter_special_edges,
                                                        gpu_inter_ms);
      }
      catch (const std::exception &ex)
      {
         gpu_inter_used = false;
         std::cerr << "[special_edges][GPU_INTER] fallback reason=" << ex.what() << std::endl;
      }
   }
   const bool profile_inter_routes = env_flag("UNG_SPECIAL_INTER_ROUTE_PROFILE");
   const bool profile_inter_no_output = env_flag("UNG_SPECIAL_INTER_PROFILE_NO_OUTPUT");
   if (profile_inter_no_output)
      std::cout << "[special_edges][profile_no_output] enabled=1 index_output_invalid=1" << std::endl;
   size_t inter_exact_pairs = 0, inter_graph_pairs = 0;
   size_t inter_exact_queries = 0, inter_graph_queries = 0;
   unsigned long long inter_exact_pair_work = 0, inter_graph_pair_work = 0;
   unsigned long long inter_exact_edges = 0, inter_graph_edges = 0;
   double inter_exact_ms = 0.0, inter_graph_ms = 0.0;
   double inter_exact_setup_ms = 0.0, inter_graph_setup_ms = 0.0;
   double inter_exact_loop_ms = 0.0, inter_graph_loop_ms = 0.0;
   size_t inter_sample_pairs = 0, inter_force_graph_pairs = 0;
   size_t inter_sample_queries = 0, inter_force_graph_queries = 0;
   unsigned long long inter_sample_pair_work = 0, inter_force_graph_pair_work = 0;
   unsigned long long inter_sample_edges = 0, inter_force_graph_edges = 0;
   double inter_sample_ms = 0.0, inter_force_graph_ms = 0.0;
   unsigned long long max_pair_work = 0;
   IdxType max_pair_parent_block = 0, max_pair_child_block = 0;
   std::vector<SpecialInterPairProfile> slowest_pairs;
   if (!gpu_inter_used)
      omp_set_num_threads(_num_threads);
   if (env_flag("UNG_SPECIAL_INTER_RESERVE") && !gpu_inter_used)
   {
      const auto reserve_start = std::chrono::high_resolution_clock::now();
      std::vector<IdxType> extra_edges_by_point(_num_points, 0);
      for (const SpecialBlock &block : _special_blocks)
      {
         if (block.block_id >= block_points.size() || block.child_block_ids.empty())
            continue;
         const auto &src_points = block_points[block.block_id];
         if (src_points.empty())
            continue;
         IdxType child_count = 0;
         for (IdxType child_block_id : block.child_block_ids)
         {
            if (child_block_id < block_points.size() && !block_points[child_block_id].empty())
               child_count += 1;
         }
         if (child_count == 0)
            continue;
         const IdxType extra = child_count * special_block_num_cross_edges;
         for (IdxType source : src_points)
            if (source < extra_edges_by_point.size())
               extra_edges_by_point[source] += extra;
      }
      size_t reserved_points = 0;
      unsigned long long reserved_edges = 0;
      for (IdxType point_id = 0; point_id < static_cast<IdxType>(extra_edges_by_point.size()); ++point_id)
      {
         const IdxType extra = extra_edges_by_point[point_id];
         if (extra == 0)
            continue;
         auto &edges = _special_edges_by_point[point_id];
         edges.reserve(edges.size() + static_cast<size_t>(extra));
         reserved_points += 1;
         reserved_edges += extra;
      }
      const double reserve_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - reserve_start)
                                  .count();
      std::cout << "[special_edges][inter_reserve] points=" << reserved_points
                << " extra_edges=" << reserved_edges
                << " ms=" << reserve_ms
                << std::endl;
   }
   if (!gpu_inter_used)
   {
      for (const SpecialBlock &block : _special_blocks)
      {
         if (block.block_id >= block_points.size())
            continue;
         const auto &src_points = block_points[block.block_id];
         if (src_points.empty())
            continue;
         for (IdxType child_block_id : block.child_block_ids)
         {
            if (child_block_id >= block_points.size())
               continue;
            const auto &dst_points = block_points[child_block_id];
            if (dst_points.empty())
               continue;
            auto child_index = child_block_id < block_indexes.size() ? block_indexes[child_block_id] : nullptr;
            const unsigned long long pair_work =
                static_cast<unsigned long long>(src_points.size()) *
                static_cast<unsigned long long>(dst_points.size());
            if (inter_pair_work_cap > 0 && pair_work > inter_pair_work_cap)
            {
               inter_pair_work_cap_skipped_pairs += 1;
               inter_pair_work_cap_skipped_queries += static_cast<unsigned long long>(src_points.size());
               inter_pair_work_cap_skipped_work += pair_work;
               continue;
            }
            if (pair_work > max_pair_work)
            {
               max_pair_work = pair_work;
               max_pair_parent_block = block.block_id;
               max_pair_child_block = child_block_id;
            }
            const bool force_graph = force_graph_pair_work > 0 &&
                                     pair_work > force_graph_pair_work &&
                                     child_index != nullptr;
            const bool use_exact_scan =
                !force_graph &&
                (dst_points.size() <= static_cast<size_t>(hybrid_cfg.target_exact_max_nx) ||
                 pair_work <= static_cast<unsigned long long>(hybrid_cfg.work_threshold));
            const size_t sample_candidates = sample_candidates_env == 0
                                                ? 0
                                                : std::max<size_t>(static_cast<size_t>(special_block_num_cross_edges),
                                                                   static_cast<size_t>(sample_candidates_env));
            const bool use_sampling = sample_candidates > 0 &&
                                      sample_candidates < dst_points.size() &&
                                      pair_work > sample_min_pair_work;
            std::vector<IdxType> sampled_dst_points;
            const std::vector<IdxType> *target_points_for_scan = &dst_points;
            if (use_sampling)
            {
               sampled_dst_points = sample_special_inter_candidates(dst_points, sample_candidates);
               target_points_for_scan = &sampled_dst_points;
            }
            const auto pair_start = std::chrono::high_resolution_clock::now();
            SearchCacheList search_cache_list(std::max<uint32_t>(1, _num_threads),
                                              static_cast<IdxType>(dst_points.size()),
                                              inter_search_ef);
            const auto pair_loop_start = std::chrono::high_resolution_clock::now();

            unsigned long long inter_edges_added = 0;
#pragma omp parallel for schedule(dynamic, 64) num_threads(_num_threads) reduction(+ : inter_edges_added)
            for (IdxType src_idx = 0; src_idx < static_cast<IdxType>(src_points.size()); ++src_idx)
            {
               const IdxType source = src_points[src_idx];
               SearchQueue local_topk;
               local_topk.reserve(special_block_num_cross_edges);
               if (use_sampling)
               {
                  append_cross_edges_from_target_points(source,
                                                        *target_points_for_scan,
                                                        nullptr,
                                                        nullptr,
                                                        true,
                                                        local_topk,
                                                        special_block_num_cross_edges);
               }
               else
               {
                  append_cross_edges_from_target_points(source,
                                                        dst_points,
                                                        child_index,
                                                        &search_cache_list,
                                                        use_exact_scan,
                                                        local_topk,
                                                        special_block_num_cross_edges);
               }
               if (profile_inter_no_output)
               {
                  inter_edges_added += static_cast<unsigned long long>(local_topk.size());
               }
               else
               {
                  auto &special_edges = _special_edges_by_point[source];
                  for (int k = 0; k < local_topk.size(); ++k)
                  {
                     special_edges.push_back({local_topk[k].id, block.block_id, SpecialEdgeKind::InterBlock});
                     inter_edges_added += 1;
                  }
               }
            }
            const auto pair_loop_end = std::chrono::high_resolution_clock::now();
            _special_block_summary.inter_special_edges += static_cast<IdxType>(inter_edges_added);
            if (profile_inter_routes)
            {
               const double pair_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::high_resolution_clock::now() - pair_start)
                                      .count();
               const double setup_ms = std::chrono::duration<double, std::milli>(
                                       pair_loop_start - pair_start)
                                       .count();
               const double loop_ms = std::chrono::duration<double, std::milli>(
                                      pair_loop_end - pair_loop_start)
                                      .count();
               SpecialInterPairProfile pair_profile;
               pair_profile.ms = pair_ms;
               pair_profile.parent_block_id = block.block_id;
               pair_profile.child_block_id = child_block_id;
               pair_profile.source_points = src_points.size();
               pair_profile.target_points = dst_points.size();
               pair_profile.pair_work = pair_work;
               pair_profile.edges = inter_edges_added;
               pair_profile.exact_scan = use_exact_scan;
               pair_profile.sampled = use_sampling;
               pair_profile.forced_graph = force_graph;
               record_slowest_inter_pair(slowest_pairs, pair_profile);
               if (use_sampling)
               {
                  inter_sample_pairs += 1;
                  inter_sample_queries += src_points.size();
                  inter_sample_pair_work += pair_work;
                  inter_sample_edges += inter_edges_added;
                  inter_sample_ms += pair_ms;
               }
               else if (use_exact_scan)
               {
                  inter_exact_pairs += 1;
                  inter_exact_queries += src_points.size();
                  inter_exact_pair_work += pair_work;
                  inter_exact_edges += inter_edges_added;
                  inter_exact_ms += pair_ms;
                  inter_exact_setup_ms += setup_ms;
                  inter_exact_loop_ms += loop_ms;
               }
               else
               {
                  inter_graph_pairs += 1;
                  inter_graph_queries += src_points.size();
                  inter_graph_pair_work += pair_work;
                  inter_graph_edges += inter_edges_added;
                  inter_graph_ms += pair_ms;
                  inter_graph_setup_ms += setup_ms;
                  inter_graph_loop_ms += loop_ms;
                  if (force_graph)
                  {
                     inter_force_graph_pairs += 1;
                     inter_force_graph_queries += src_points.size();
                     inter_force_graph_pair_work += pair_work;
                     inter_force_graph_edges += inter_edges_added;
                     inter_force_graph_ms += pair_ms;
                  }
               }
            }
         }
      }
   }
   if (inter_pair_work_cap > 0 && !gpu_inter_used)
   {
      std::cout << "[special_edges][inter_pair_work_cap] max_pair_work=" << inter_pair_work_cap
                << " skipped_pairs=" << inter_pair_work_cap_skipped_pairs
                << " skipped_queries=" << inter_pair_work_cap_skipped_queries
                << " skipped_pair_work=" << inter_pair_work_cap_skipped_work
                << std::endl;
   }
   if (profile_inter_routes && !gpu_inter_used)
   {
      std::cout << "[special_edges][cpu_inter_profile] exact_pairs=" << inter_exact_pairs
                << " exact_queries=" << inter_exact_queries
                << " exact_pair_work=" << inter_exact_pair_work
                << " exact_edges=" << inter_exact_edges
                << " exact_ms=" << inter_exact_ms
                << " exact_setup_ms=" << inter_exact_setup_ms
                << " exact_loop_ms=" << inter_exact_loop_ms
                << " graph_pairs=" << inter_graph_pairs
                << " graph_queries=" << inter_graph_queries
                << " graph_pair_work=" << inter_graph_pair_work
                << " graph_edges=" << inter_graph_edges
                << " graph_ms=" << inter_graph_ms
                << " graph_setup_ms=" << inter_graph_setup_ms
                << " graph_loop_ms=" << inter_graph_loop_ms
                << " force_graph_pair_work=" << force_graph_pair_work
                << " force_graph_pairs=" << inter_force_graph_pairs
                << " force_graph_queries=" << inter_force_graph_queries
                << " force_graph_pair_work_sum=" << inter_force_graph_pair_work
                << " force_graph_edges=" << inter_force_graph_edges
                << " force_graph_ms=" << inter_force_graph_ms
                << " inter_search_ef=" << inter_search_ef
                << " sample_candidates=" << sample_candidates_env
                << " sample_min_pair_work=" << sample_min_pair_work
                << " sample_pairs=" << inter_sample_pairs
                << " sample_queries=" << inter_sample_queries
                << " sample_pair_work=" << inter_sample_pair_work
                << " sample_edges=" << inter_sample_edges
                << " sample_ms=" << inter_sample_ms
                << " max_pair_work=" << max_pair_work
                << " max_pair_parent_block=" << max_pair_parent_block
                << " max_pair_child_block=" << max_pair_child_block
                << std::endl;
      for (size_t rank = 0; rank < slowest_pairs.size(); ++rank)
      {
         const auto &pair = slowest_pairs[rank];
         std::cout << "[special_edges][cpu_inter_slowest_pair] rank=" << (rank + 1)
                   << " parent_block=" << pair.parent_block_id
                   << " child_block=" << pair.child_block_id
                   << " ms=" << pair.ms
                   << " pair_work=" << pair.pair_work
                   << " source_points=" << pair.source_points
                   << " target_points=" << pair.target_points
                   << " edges=" << pair.edges
                   << " route=" << (pair.sampled ? "sample" : (pair.exact_scan ? "exact" : "graph"))
                   << " forced_graph=" << (pair.forced_graph ? 1 : 0)
                   << std::endl;
      }
   }
   inter_ms = std::chrono::duration<double, std::milli>(
                  std::chrono::high_resolution_clock::now() - inter_start)
                  .count();
   std::cout << "[special_edges] gpu_inter_enabled=" << (gpu_inter_enabled ? 1 : 0)
             << " gpu_inter_used=" << (gpu_inter_used ? 1 : 0)
             << " gpu_inter_ms=" << gpu_inter_ms
             << std::endl;

   _special_block_summary.special_edges =
       _special_block_summary.intra_special_edges + _special_block_summary.inter_special_edges;
   const double overlay_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::high_resolution_clock::now() - overlay_start)
                                 .count();
   _special_block_summary.intra_edge_build_ms = intra_ms;
   _special_block_summary.inter_edge_build_ms = inter_ms;
   _special_block_summary.edge_overlay_ms = overlay_ms;
   std::cout << "[special_edges] sidecar_edges=" << _special_block_summary.special_edges
             << " intra=" << _special_block_summary.intra_special_edges
             << " inter=" << _special_block_summary.inter_special_edges
             << " block_indexes_prepare_ms=" << _special_block_summary.block_indexes_prepare_ms
             << " intra_ms=" << intra_ms
             << " inter_ms=" << inter_ms
             << " ms=" << overlay_ms << std::endl;
   prof_logf("[PROF] special_edges sidecar_edges=%u intra=%u inter=%u prepare_ms=%.3f intra_ms=%.3f inter_ms=%.3f ms=%.3f",
             static_cast<unsigned>(_special_block_summary.special_edges),
             static_cast<unsigned>(_special_block_summary.intra_special_edges),
             static_cast<unsigned>(_special_block_summary.inter_special_edges),
             _special_block_summary.block_indexes_prepare_ms,
             intra_ms,
             inter_ms,
             overlay_ms);
}

void UniNavGraph::save_special_blocks(const std::string &prefix)
{
   if (!_build_config.special_blocks_enabled)
      return;
   const auto save_start = std::chrono::high_resolution_clock::now();
   double metadata_ms = 0.0;
   double member_ms = 0.0;
   double child_ms = 0.0;
   double edge_ms = 0.0;
   double trie_ms = 0.0;
   double trie_regular_edge_ms = 0.0;
   size_t light_edge_rows = 0;
   size_t heavy_edge_rows = 0;
   const auto metadata_start = std::chrono::high_resolution_clock::now();
   save_special_block_metadata_binary(prefix + "special_blocks.bin", _special_blocks);
   const bool write_metadata_csv = env_flag("UNG_SPECIAL_BLOCK_METADATA_CSV");
   if (!write_metadata_csv)
   {
      std::error_code ec;
      std::filesystem::remove(prefix + "special_blocks.csv", ec);
      std::filesystem::remove(prefix + "special_block_members.csv", ec);
      std::filesystem::remove(prefix + "special_block_children.csv", ec);
   }
   if (write_metadata_csv)
   {
      std::ofstream out(prefix + "special_blocks.csv");
      out << "block_id,level,parent_block_id,root_group_id,root_labels,point_count,subtree_point_count,is_trivial,member_group_count,child_block_count,entry_point_id,common_labels\n";
      for (const SpecialBlock &block : _special_blocks)
      {
         out << block.block_id << ','
             << static_cast<unsigned>(block.level) << ','
             << block.parent_block_id << ','
             << block.root_group_id << ','
             << join_labels_for_special_path(block.root_labels) << ','
             << block.point_count << ','
             << block.subtree_point_count << ','
             << (block.is_trivial() ? 1 : 0) << ','
             << block.member_group_ids.size() << ','
             << block.child_block_ids.size() << ','
             << block.entry_point_id << ','
             << join_labels_for_special_path(block.common_labels) << '\n';
      }
   }
   metadata_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - metadata_start)
                     .count();
   const auto member_start = std::chrono::high_resolution_clock::now();
   if (write_metadata_csv)
   {
      std::ofstream out(prefix + "special_block_members.csv");
      out << "block_id,group_id\n";
      for (const SpecialBlock &block : _special_blocks)
         for (IdxType group_id : block.member_group_ids)
            out << block.block_id << ',' << group_id << '\n';
   }
   member_ms = std::chrono::duration<double, std::milli>(
                  std::chrono::high_resolution_clock::now() - member_start)
                  .count();
   const auto child_start = std::chrono::high_resolution_clock::now();
   if (write_metadata_csv)
   {
      std::ofstream out(prefix + "special_block_children.csv");
      out << "parent_block_id,child_block_id\n";
      for (const SpecialBlock &block : _special_blocks)
         for (IdxType child : block.child_block_ids)
            out << block.block_id << ',' << child << '\n';
   }
   child_ms = std::chrono::duration<double, std::milli>(
                 std::chrono::high_resolution_clock::now() - child_start)
                 .count();
   if (!_special_block_trie_index.empty())
   {
      const auto trie_start = std::chrono::high_resolution_clock::now();
      _special_block_trie_index.save(prefix + "special_block_trie.bin");
      trie_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - trie_start)
                    .count();
   }
   if (_special_trie_regular_edges_available)
   {
      const auto regular_edge_start = std::chrono::high_resolution_clock::now();
      const std::string path = prefix + "special_trie_regular_edges.bin";
      uint64_t count = 0;
      std::string error;
      if (!write_regular_edge_csr_file(
              path, _special_trie_regular_edges_by_point, count, error))
         throw std::runtime_error("cannot save special Trie regular CSR sidecar: " + error);
      _special_block_summary.trie_regular_vector_edges = count;
      trie_regular_edge_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::high_resolution_clock::now() - regular_edge_start)
                                 .count();
   }
   const auto edge_start = std::chrono::high_resolution_clock::now();
   {
      const bool binary_only_sidecar = env_flag("UNG_SPECIAL_EDGE_BINARY_ONLY");
      const bool write_binary_sidecar = true;
      std::ofstream out;
      if (!binary_only_sidecar)
         out.open(prefix + "special_edges.csv");
      else
      {
         std::error_code ec;
         std::filesystem::remove(prefix + "special_edges.csv", ec);
         std::filesystem::remove(prefix + "special_heavy_edges.csv", ec);
      }
      std::unique_ptr<std::ofstream> heavy_out;
      uint64_t binary_light_count = 0;
      uint64_t binary_heavy_count = 0;
      const unsigned long long save_heavy_pair_work_threshold =
          read_env_ull("UNG_SPECIAL_HEAVY_EDGE_SAVE_PAIR_WORK", 0);
      std::unordered_map<unsigned long long, unsigned long long> child_pair_work_by_key;
      if (save_heavy_pair_work_threshold > 0)
      {
         if (!binary_only_sidecar)
         {
            heavy_out = std::make_unique<std::ofstream>(prefix + "special_heavy_edges.csv");
            *heavy_out << "source_point_id,target_point_id,special_block_id,edge_kind\n";
         }
         for (const SpecialBlock &parent : _special_blocks)
         {
            if (parent.block_id == 0)
               continue;
            for (IdxType child_id : parent.child_block_ids)
            {
               if (child_id == 0 || child_id > _special_blocks.size())
                  continue;
               const SpecialBlock &child = _special_blocks[child_id - 1];
               const unsigned long long key =
                   (static_cast<unsigned long long>(parent.block_id) << 32) |
                   static_cast<unsigned long long>(child_id);
               child_pair_work_by_key[key] =
                   static_cast<unsigned long long>(parent.point_count) *
                   static_cast<unsigned long long>(child.point_count);
            }
         }
      }
      if (!binary_only_sidecar)
         out << "source_point_id,target_point_id,special_block_id,edge_kind\n";
      for (IdxType source = 0; source < _special_edges_by_point.size(); ++source)
         for (const SpecialEdge &edge : _special_edges_by_point[source])
         {
            bool write_heavy = false;
            if (save_heavy_pair_work_threshold > 0 &&
                edge.kind == SpecialEdgeKind::InterBlock)
            {
               const IdxType child_id = special_edge_target_owner(
                   edge, _special_blocks, _point_to_special_block,
                   _point_to_upper_special_block);
               const unsigned long long key =
                   (static_cast<unsigned long long>(edge.special_block_id) << 32) |
                   static_cast<unsigned long long>(child_id);
               auto work_it = child_pair_work_by_key.find(key);
               write_heavy = work_it != child_pair_work_by_key.end() &&
                             work_it->second > save_heavy_pair_work_threshold;
            }
            if (write_heavy)
               heavy_edge_rows += 1;
            else
               light_edge_rows += 1;
            if (!binary_only_sidecar)
            {
               std::ostream &target_out = write_heavy ? *heavy_out : out;
               target_out << source << ','
                          << edge.target_point_id << ','
                          << edge.special_block_id << ','
                          << (edge.kind == SpecialEdgeKind::IntraBlock ? "intra" : "inter")
                          << '\n';
            }
         }
      if (write_binary_sidecar)
      {
         const auto is_heavy = [&](IdxType, const SpecialEdge &edge) {
            if (save_heavy_pair_work_threshold == 0 ||
                edge.kind != SpecialEdgeKind::InterBlock)
               return false;
            const IdxType child_id = special_edge_target_owner(
                edge, _special_blocks, _point_to_special_block,
                _point_to_upper_special_block);
            const unsigned long long key =
                (static_cast<unsigned long long>(edge.special_block_id) << 32) |
                static_cast<unsigned long long>(child_id);
            const auto work_it = child_pair_work_by_key.find(key);
            return work_it != child_pair_work_by_key.end() &&
                   work_it->second > save_heavy_pair_work_threshold;
         };
         std::string error;
         if (!write_special_edge_csr_file(
                 prefix + "special_edges.bin", _special_edges_by_point,
                 [&](IdxType source, const SpecialEdge &edge) { return !is_heavy(source, edge); },
                 binary_light_count, error))
            throw std::runtime_error("cannot save special-edge CSR sidecar: " + error);
         if (save_heavy_pair_work_threshold > 0 &&
             !write_special_edge_csr_file(
                 prefix + "special_heavy_edges.bin", _special_edges_by_point,
                 is_heavy, binary_heavy_count, error))
            throw std::runtime_error("cannot save heavy special-edge CSR sidecar: " + error);
         if (save_heavy_pair_work_threshold == 0)
         {
            std::error_code ec;
            std::filesystem::remove(prefix + "special_heavy_edges.bin", ec);
         }
      }
   }
   edge_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - edge_start)
                .count();
   const double total_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::high_resolution_clock::now() - save_start)
                               .count();
   _special_block_summary.save_metadata_ms = metadata_ms;
   _special_block_summary.save_members_ms = member_ms;
   _special_block_summary.save_children_ms = child_ms;
   _special_block_summary.save_edges_ms = edge_ms;
   _special_block_summary.save_trie_regular_edges_ms = trie_regular_edge_ms;
   _special_block_summary.trie_save_ms = trie_ms;
   _special_block_summary.save_total_ms = total_ms;
   std::cout << "[special_blocks][save] metadata_ms=" << metadata_ms
             << " members_ms=" << member_ms
             << " children_ms=" << child_ms
             << " trie_ms=" << trie_ms
             << " trie_regular_edges_ms=" << trie_regular_edge_ms
             << " edges_ms=" << edge_ms
             << " total_ms=" << total_ms
             << " light_edges=" << light_edge_rows
             << " heavy_edges=" << heavy_edge_rows
             << std::endl;
}

void UniNavGraph::load_special_blocks(const std::string &prefix,
                                      const std::map<std::string, std::string> &meta_data,
                                      bool require_binary_sidecars)
{
   _special_blocks.clear();
   _special_block_trie_index.clear();
   _special_edges_by_point.clear();
   _special_heavy_edges_by_point.clear();
   _special_trie_regular_edges_by_point.clear();
   _special_edges_csr.clear();
   _special_heavy_edges_csr.clear();
   _special_trie_regular_edges_csr.clear();
   _special_trie_regular_edges_available = false;
   _special_block_summary = {};
   auto meta_it = meta_data.find("special_blocks_enabled");
   if (meta_it == meta_data.end() || meta_it->second != "1")
   {
      if (require_binary_sidecars)
         throw std::runtime_error(
             "explicit special block index metadata does not enable special blocks");
      rebuild_special_block_indexes();
      return;
   }

   const auto partition_it = meta_data.find("special_block_partition");
   const bool trie_partition =
       partition_it != meta_data.end() && partition_it->second == "trie";
   if (require_binary_sidecars)
   {
      // An explicitly loaded overlay is an immutable bundle.  Check the
      // complete required file set before parsing any one member so a corrupt
      // metadata file cannot mask a missing graph sidecar.
      const auto require_file = [&](const char *name) {
         const std::string path = prefix + name;
         if (!std::filesystem::is_regular_file(path))
            throw std::runtime_error(
                "required special block sidecar is missing: " + path);
      };
      require_file("special_blocks.bin");
      if (trie_partition)
      {
         require_file("special_block_trie.bin");
         require_file("special_trie_regular_edges.bin");
      }
      require_file("special_edges.bin");
   }

   auto threshold_it = meta_data.find("special_block_min_points");
   if (threshold_it != meta_data.end())
      _special_block_summary.threshold = static_cast<IdxType>(std::stoul(threshold_it->second));
   auto upper_threshold_it = meta_data.find("special_block_upper_min_points");
   if (upper_threshold_it != meta_data.end())
      _special_block_summary.upper_threshold =
          static_cast<IdxType>(std::stoul(upper_threshold_it->second));

   std::string binary_metadata_error;
   const bool loaded_binary_metadata = load_special_block_metadata_binary(
       prefix + "special_blocks.bin", _special_blocks, binary_metadata_error);
   if (!loaded_binary_metadata)
   {
      if (require_binary_sidecars)
         throw std::runtime_error(
             "cannot load required special block metadata sidecar: " +
             (binary_metadata_error.empty()
                  ? std::string("file is missing: ") + prefix + "special_blocks.bin"
                  : binary_metadata_error));
      std::ifstream in(prefix + "special_blocks.csv");
      if (!in)
      {
         std::cerr << "[special_blocks] binary metadata unavailable ("
                   << binary_metadata_error << ") and legacy special_blocks.csv is missing."
                   << std::endl;
         rebuild_special_block_indexes();
         return;
      }
      std::string line;
      std::getline(in, line);
      while (std::getline(in, line))
      {
         if (line.empty())
            continue;
         const auto cols = split_csv_line(line);
         if (cols.size() < 8)
            continue;
         SpecialBlock block;
         block.block_id = static_cast<IdxType>(std::stoul(cols[0]));
         const bool multilevel_csv = cols.size() >= 12;
         const size_t offset = multilevel_csv ? 2 : 0;
         if (multilevel_csv)
         {
            block.level = static_cast<uint8_t>(std::stoul(cols[1]));
            block.parent_block_id = static_cast<IdxType>(std::stoul(cols[2]));
         }
         block.root_group_id = static_cast<IdxType>(std::stoul(cols[1 + offset]));
         block.root_labels = parse_label_list(cols[2 + offset]);
         block.point_count = static_cast<IdxType>(std::stoul(cols[3 + offset]));
         block.subtree_point_count = static_cast<IdxType>(std::stoul(cols[4 + offset]));
         if (cols.size() >= 9 + offset)
            block.entry_point_id = static_cast<IdxType>(std::stoull(cols[8 + offset]));
         if (cols.size() >= 10 + offset)
            block.common_labels = parse_label_list(cols[9 + offset]);
         if (block.block_id > _special_blocks.size() + 1)
            _special_blocks.resize(block.block_id - 1);
         if (block.block_id == 0)
            continue;
         if (block.block_id > _special_blocks.size())
            _special_blocks.push_back(std::move(block));
         else
            _special_blocks[block.block_id - 1] = std::move(block);
      }
   }

   if (!loaded_binary_metadata)
   {
      std::ifstream in(prefix + "special_block_members.csv");
      std::string line;
      std::getline(in, line);
      while (std::getline(in, line))
      {
         const auto cols = split_csv_line(line);
         if (cols.size() < 2)
            continue;
         const IdxType block_id = static_cast<IdxType>(std::stoul(cols[0]));
         const IdxType group_id = static_cast<IdxType>(std::stoul(cols[1]));
         if (block_id > 0 && block_id <= _special_blocks.size())
            _special_blocks[block_id - 1].member_group_ids.push_back(group_id);
      }
   }

   if (!loaded_binary_metadata)
   {
      std::ifstream in(prefix + "special_block_children.csv");
      std::string line;
      std::getline(in, line);
      while (std::getline(in, line))
      {
         const auto cols = split_csv_line(line);
         if (cols.size() < 2)
            continue;
         const IdxType parent_id = static_cast<IdxType>(std::stoul(cols[0]));
         const IdxType child_id = static_cast<IdxType>(std::stoul(cols[1]));
         if (parent_id > 0 && parent_id <= _special_blocks.size())
            _special_blocks[parent_id - 1].child_block_ids.push_back(child_id);
      }
   }

   for (SpecialBlock &block : _special_blocks)
   {
      if (block.entry_point_id != SpecialBlock::kInvalidEntryPoint &&
          block.entry_point_id >= _num_points)
         throw std::runtime_error("special block metadata contains an invalid entry point");
      std::sort(block.root_labels.begin(), block.root_labels.end());
      block.root_labels.erase(std::unique(block.root_labels.begin(), block.root_labels.end()),
                              block.root_labels.end());
      std::sort(block.member_group_ids.begin(), block.member_group_ids.end());
      block.member_group_ids.erase(std::unique(block.member_group_ids.begin(), block.member_group_ids.end()),
                                   block.member_group_ids.end());
      const std::vector<LabelType> exact_common = compute_direct_member_common_labels(
          block.member_group_ids, _group_id_to_label_set);
      std::sort(block.common_labels.begin(), block.common_labels.end());
      block.common_labels.erase(std::unique(block.common_labels.begin(), block.common_labels.end()),
                                block.common_labels.end());
      if (!block.common_labels.empty() && block.common_labels != exact_common)
         throw std::runtime_error("special block metadata common-label intersection mismatch");
      block.common_labels = exact_common;
      std::sort(block.child_block_ids.begin(), block.child_block_ids.end());
      block.child_block_ids.erase(std::unique(block.child_block_ids.begin(), block.child_block_ids.end()),
                                  block.child_block_ids.end());
   }
   std::string topology_error;
   if (!validate_special_block_metadata(_special_blocks, topology_error))
      throw std::runtime_error("invalid special block topology: " + topology_error);
   std::string graph_semantics_error;
   if (!validate_special_block_graph_semantics(
           _special_blocks, _num_points, _num_groups, _group_id_to_label_set,
           _group_id_to_range, _new_vec_id_to_group_id, graph_semantics_error))
      throw std::runtime_error(
          "special block metadata does not match its source UNG graph: " +
          graph_semantics_error);
   rebuild_special_block_indexes();

   if (trie_partition)
   {
      const auto trie_load_start = std::chrono::high_resolution_clock::now();
      _special_block_trie_index.load(prefix + "special_block_trie.bin",
                                     _num_groups,
                                     static_cast<IdxType>(_special_blocks.size()));
      _special_block_summary.trie_load_ms = std::chrono::duration<double, std::milli>(
                                                std::chrono::high_resolution_clock::now() - trie_load_start)
                                                .count();
      _special_block_summary.trie_node_count = _special_block_trie_index.node_count();
      _special_block_summary.trie_child_count = _special_block_trie_index.child_count();
      _special_block_summary.trie_serialized_bytes =
          _special_block_trie_index.serialized_size_bytes();

      const auto regular_group_count_it = meta_data.find("special_trie_regular_group_edge_count");
      _special_block_summary.trie_regular_group_edges =
          regular_group_count_it == meta_data.end()
              ? _special_block_trie_index.terminal_successor_pairs().size()
              : std::stoull(regular_group_count_it->second);
      const auto portal_edge_count_it = meta_data.find("special_trie_regular_portal_edge_count");
      if (portal_edge_count_it != meta_data.end())
         _special_block_summary.trie_regular_portal_edges =
             std::stoull(portal_edge_count_it->second);
      const std::string regular_edge_path = prefix + "special_trie_regular_edges.bin";
      if (std::ifstream(regular_edge_path, std::ios::binary))
      {
         uint64_t loaded = 0;
         std::string error;
         bool invalid_record = false;
         const auto validation = validate_special_edge_binary_file(regular_edge_path);
         bool read_ok = false;
         if (validation.valid && validation.format_version == 2)
         {
            read_ok = validation.regular_targets_only && validation.num_sources == _num_points &&
                      read_regular_edge_csr_file(
                          regular_edge_path, _special_trie_regular_edges_csr, error);
            if (read_ok)
            {
               loaded = _special_trie_regular_edges_csr.targets.size();
               invalid_record = std::any_of(
                   _special_trie_regular_edges_csr.targets.begin(),
                   _special_trie_regular_edges_csr.targets.end(),
                   [&](IdxType target) { return target >= _num_points; });
            }
         }
         else
         {
            _special_trie_regular_edges_by_point.assign(_num_points, {});
            read_ok = for_each_special_edge_binary_record(
                regular_edge_path,
                [&](const SpecialEdgeBinaryRecord &record) {
                   if (record.source >= _num_points || record.target >= _num_points ||
                       record.block != 0 || record.kind != 0)
                   {
                      invalid_record = true;
                      return;
                   }
                   _special_trie_regular_edges_by_point[record.source].push_back(record.target);
                },
                loaded, error);
         }
         if (!read_ok || invalid_record)
            throw std::runtime_error(
                "invalid special Trie regular edge sidecar: " +
                (invalid_record ? std::string("record field is out of range") : error));
         _special_trie_regular_edges_available = true;
         _special_block_summary.trie_regular_vector_edges = loaded;
      }
      else
      {
         if (require_binary_sidecars)
            throw std::runtime_error(
                "required special Trie regular edge sidecar is missing: " +
                regular_edge_path);
         std::cerr << "[special_trie_regular] sidecar missing; rebuild the index before using the special_block_trie provider."
                   << std::endl;
      }
   }

   const unsigned long long heavy_pair_work_threshold =
       read_env_ull("UNG_SPECIAL_HEAVY_EDGE_PAIR_WORK", 0);
   std::unordered_map<unsigned long long, unsigned long long> child_pair_work_by_key;
   if (heavy_pair_work_threshold > 0)
   {
      for (const SpecialBlock &parent : _special_blocks)
      {
         if (parent.block_id == 0)
            continue;
         for (IdxType child_id : parent.child_block_ids)
         {
            if (child_id == 0 || child_id > _special_blocks.size())
               continue;
            const SpecialBlock &child = _special_blocks[child_id - 1];
            const unsigned long long pair_work =
                static_cast<unsigned long long>(parent.point_count) *
                static_cast<unsigned long long>(child.point_count);
            const unsigned long long key =
                (static_cast<unsigned long long>(parent.block_id) << 32) |
                static_cast<unsigned long long>(child_id);
            child_pair_work_by_key[key] = pair_work;
         }
      }
   }

   const auto load_binary_edges = [&](const std::string &path,
                                      SpecialEdgeCsr &csr,
                                      std::vector<std::vector<SpecialEdge>> &legacy,
                                      size_t &loaded_edges) {
      const auto validation = validate_special_edge_binary_file(path);
      if (validation.valid && validation.format_version == 2)
      {
         std::string error;
         bool read_ok = false;
         if (validation.regular_targets_only)
            error = "expected special-edge payload, found regular targets";
         else if (validation.num_sources != _num_points)
            error = "CSR source count does not match the UNG point count";
         else
            read_ok = read_special_edge_csr_file(path, csr, error);
         if (!read_ok)
            throw std::runtime_error("invalid special-edge CSR sidecar: " + error);
         for (IdxType source = 0; source < validation.num_sources; ++source)
         {
            for (uint64_t edge_idx = csr.offsets[source];
                 edge_idx < csr.offsets[source + 1]; ++edge_idx)
            {
               const SpecialEdge edge = csr.edges[edge_idx].unpack();
               std::string semantic_error;
               if (!validate_special_edge_semantics(
                       source, edge, _special_blocks,
                       _point_to_special_block, _point_to_upper_special_block,
                       semantic_error))
                  throw std::runtime_error(
                      "invalid special-edge CSR semantics: " + semantic_error);
               _special_block_summary.special_edges += 1;
               if (edge.kind == SpecialEdgeKind::InterBlock)
                  _special_block_summary.inter_special_edges += 1;
               else
                  _special_block_summary.intra_special_edges += 1;
            }
         }
         loaded_edges += csr.edges.size();
         std::cout << "[special_edges][load] csr=" << path
                   << " loaded_edges=" << csr.edges.size() << std::endl;
         return true;
      }
      legacy.assign(_num_points, {});
      const bool loaded = read_special_edge_binary_file(
          path, legacy, _special_block_summary, loaded_edges, _special_blocks,
          _point_to_special_block, _point_to_upper_special_block);
      return loaded;
   };

   size_t heavy_edges_loaded = 0;
   const bool load_persisted_heavy_edges = env_flag("UNG_SPECIAL_HEAVY_EDGE_SEARCH");
   const bool light_binary_edges_exist =
       !env_flag("UNG_SPECIAL_EDGE_FORCE_CSV") &&
       static_cast<bool>(std::ifstream(prefix + "special_edges.bin", std::ios::binary));
   const bool heavy_binary_edges_exist =
       static_cast<bool>(std::ifstream(prefix + "special_heavy_edges.bin", std::ios::binary));
   size_t light_binary_edges_loaded = 0;
   const bool loaded_light_from_binary =
       light_binary_edges_exist &&
       load_binary_edges(prefix + "special_edges.bin",
                         _special_edges_csr,
                         _special_edges_by_point,
                         light_binary_edges_loaded);
   if (require_binary_sidecars && !loaded_light_from_binary)
      throw std::runtime_error(
          "required special-edge binary sidecar is missing or invalid: " +
          prefix + "special_edges.bin");
   std::ifstream edge_in(prefix + "special_edges.csv");
   if (!loaded_light_from_binary && edge_in)
   {
      _special_edges_by_point.assign(_num_points, {});
      std::string line;
      std::getline(edge_in, line);
      size_t line_number = 1;
      while (std::getline(edge_in, line))
      {
         ++line_number;
         if (line.empty())
            continue;
         SpecialEdgeBinaryRecord record;
         std::string parse_error;
         if (!parse_special_edge_csv_record(line, record, parse_error))
            throw std::runtime_error(
                "invalid special-edge CSV at line " +
                std::to_string(line_number) + ": " + parse_error);
         const IdxType source = static_cast<IdxType>(record.source);
         if (source >= _special_edges_by_point.size())
            throw std::runtime_error("special-edge CSV source is out of range");
         SpecialEdge edge;
         edge.target_point_id = static_cast<IdxType>(record.target);
         edge.special_block_id = static_cast<IdxType>(record.block);
         edge.kind = record.kind != 0 ? SpecialEdgeKind::InterBlock : SpecialEdgeKind::IntraBlock;
         std::string semantic_error;
         if (!validate_special_edge_semantics(
                 source, edge, _special_blocks,
                 _point_to_special_block, _point_to_upper_special_block,
                 semantic_error))
            throw std::runtime_error(
                "invalid special-edge CSV semantics: " + semantic_error);
         bool use_heavy_sidecar = false;
         if (heavy_pair_work_threshold > 0 && edge.kind == SpecialEdgeKind::InterBlock)
         {
            const IdxType child_id = special_edge_target_owner(
                edge, _special_blocks, _point_to_special_block,
                _point_to_upper_special_block);
            if (child_id > 0)
            {
               const unsigned long long key =
                   (static_cast<unsigned long long>(edge.special_block_id) << 32) |
                   static_cast<unsigned long long>(child_id);
               auto work_it = child_pair_work_by_key.find(key);
               use_heavy_sidecar = work_it != child_pair_work_by_key.end() &&
                                   work_it->second > heavy_pair_work_threshold;
            }
         }
         if (use_heavy_sidecar)
         {
            if (_special_heavy_edges_by_point.empty())
               _special_heavy_edges_by_point.assign(_num_points, {});
            _special_heavy_edges_by_point[source].push_back(edge);
            heavy_edges_loaded += 1;
         }
         else
         {
            _special_edges_by_point[source].push_back(edge);
         }
         _special_block_summary.special_edges += 1;
         if (edge.kind == SpecialEdgeKind::InterBlock)
            _special_block_summary.inter_special_edges += 1;
         else
            _special_block_summary.intra_special_edges += 1;
      }
   }
   bool loaded_heavy_from_binary = false;
   if (load_persisted_heavy_edges && heavy_binary_edges_exist)
      loaded_heavy_from_binary = load_binary_edges(
          prefix + "special_heavy_edges.bin", _special_heavy_edges_csr,
          _special_heavy_edges_by_point, heavy_edges_loaded);
   std::ifstream heavy_edge_in(prefix + "special_heavy_edges.csv");
   const bool heavy_csv_edges_exist = static_cast<bool>(heavy_edge_in);
   if (load_persisted_heavy_edges && !loaded_heavy_from_binary &&
       heavy_csv_edges_exist)
   {
      if (_special_heavy_edges_by_point.empty())
         _special_heavy_edges_by_point.assign(_num_points, {});
      std::string line;
      std::getline(heavy_edge_in, line);
      size_t line_number = 1;
      while (std::getline(heavy_edge_in, line))
      {
         ++line_number;
         if (line.empty())
            continue;
         SpecialEdgeBinaryRecord record;
         std::string parse_error;
         if (!parse_special_edge_csv_record(line, record, parse_error))
            throw std::runtime_error(
                "invalid heavy special-edge CSV at line " +
                std::to_string(line_number) + ": " + parse_error);
         const IdxType source = static_cast<IdxType>(record.source);
         if (source >= _special_heavy_edges_by_point.size())
            throw std::runtime_error("heavy special-edge CSV source is out of range");
         SpecialEdge edge;
         edge.target_point_id = static_cast<IdxType>(record.target);
         edge.special_block_id = static_cast<IdxType>(record.block);
         edge.kind = record.kind != 0 ? SpecialEdgeKind::InterBlock : SpecialEdgeKind::IntraBlock;
         std::string semantic_error;
         if (!validate_special_edge_semantics(
                 source, edge, _special_blocks,
                 _point_to_special_block, _point_to_upper_special_block,
                 semantic_error))
            throw std::runtime_error(
                "invalid heavy special-edge CSV semantics: " + semantic_error);
         _special_heavy_edges_by_point[source].push_back(edge);
         _special_block_summary.special_edges += 1;
         if (edge.kind == SpecialEdgeKind::InterBlock)
            _special_block_summary.inter_special_edges += 1;
         else
            _special_block_summary.intra_special_edges += 1;
         heavy_edges_loaded += 1;
      }
   }
   if (load_persisted_heavy_edges && heavy_binary_edges_exist &&
       !loaded_heavy_from_binary && !heavy_csv_edges_exist)
      throw std::runtime_error(
          "requested heavy special-edge binary is invalid and no CSV fallback exists: " +
          prefix + "special_heavy_edges.bin");
   std::cout << "[special_blocks] loaded blocks=" << _special_block_summary.num_blocks
             << " trivial_blocks=" << _special_block_summary.trivial_blocks
             << " trie_nodes=" << _special_block_summary.trie_node_count
             << " trie_children=" << _special_block_summary.trie_child_count
             << " trie_bytes=" << _special_block_summary.trie_serialized_bytes
             << " trie_load_ms=" << _special_block_summary.trie_load_ms
             << " trie_regular_group_edges=" << _special_block_summary.trie_regular_group_edges
             << " trie_regular_vector_edges=" << _special_block_summary.trie_regular_vector_edges
             << " sidecar_edges=" << _special_block_summary.special_edges
             << " binary_light_edges=" << light_binary_edges_loaded
             << " heavy_edges=" << heavy_edges_loaded
             << " heavy_pair_work_threshold=" << heavy_pair_work_threshold
             << std::endl;
}

} // namespace ANNS
