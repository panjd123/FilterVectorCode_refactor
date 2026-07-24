#include "include/uni_nav_graph.h"

#include "include/tagore_graph_builder.h"
#include "include/ung_build_settings.h"
#include "include/ung_prof_log.h"
#include "vamana/vamana.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cerrno>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <unordered_map>
#include <utility>

#include <omp.h>

namespace ANNS
{
namespace
{

struct SpecialTrieNode
{
   LabelType label = 0;
   int parent = -1;
   std::unordered_map<LabelType, int> children;
   IdxType group_id = 0;
   IdxType group_points = 0;
   IdxType subtree_points = 0;
   IdxType uncovered_points = 0;
   IdxType block_id = 0;
   std::vector<LabelType> labels;
};

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

void collect_special_block_members(const std::vector<SpecialTrieNode> &nodes,
                                   int root,
                                   SpecialBlock &block,
                                   const std::vector<IdxType> &node_to_block)
{
   std::vector<int> stack{root};
   while (!stack.empty())
   {
      const int cur = stack.back();
      stack.pop_back();
      if (cur != root && node_to_block[cur] != 0)
      {
         block.child_block_ids.push_back(node_to_block[cur]);
         continue;
      }
      if (nodes[cur].group_id != 0)
      {
         block.member_group_ids.push_back(nodes[cur].group_id);
         block.point_count += nodes[cur].group_points;
      }
      for (const auto &kv : nodes[cur].children)
         stack.push_back(kv.second);
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

struct SpecialEdgeBinaryRecord
{
   uint32_t source = 0;
   uint32_t target = 0;
   uint32_t block = 0;
   uint8_t kind = 0;
   uint8_t pad[3] = {0, 0, 0};
};

void write_special_edge_binary_record(std::ofstream &out, IdxType source, const SpecialEdge &edge)
{
   SpecialEdgeBinaryRecord rec;
   rec.source = static_cast<uint32_t>(source);
   rec.target = static_cast<uint32_t>(edge.target_point_id);
   rec.block = static_cast<uint32_t>(edge.special_block_id);
   rec.kind = edge.kind == SpecialEdgeKind::InterBlock ? 1 : 0;
   out.write(reinterpret_cast<const char *>(&rec), sizeof(rec));
}

bool read_special_edge_binary_file(const std::string &path,
                                   std::vector<std::vector<SpecialEdge>> &edges_by_point,
                                   SpecialBlockBuildSummary &summary,
                                   size_t &loaded_edges)
{
   std::ifstream in(path, std::ios::binary);
   if (!in)
      return false;
   uint64_t count = 0;
   in.read(reinterpret_cast<char *>(&count), sizeof(count));
   for (uint64_t i = 0; i < count; ++i)
   {
      SpecialEdgeBinaryRecord rec;
      in.read(reinterpret_cast<char *>(&rec), sizeof(rec));
      if (!in)
         break;
      if (rec.source >= edges_by_point.size())
         continue;
      SpecialEdge edge;
      edge.target_point_id = static_cast<IdxType>(rec.target);
      edge.special_block_id = static_cast<IdxType>(rec.block);
      edge.kind = rec.kind != 0 ? SpecialEdgeKind::InterBlock : SpecialEdgeKind::IntraBlock;
      edges_by_point[rec.source].push_back(edge);
      summary.special_edges += 1;
      if (edge.kind == SpecialEdgeKind::InterBlock)
         summary.inter_special_edges += 1;
      else
         summary.intra_special_edges += 1;
      loaded_edges += 1;
   }
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
   std::vector<LabelType> sorted_root = block.root_labels;
   std::sort(sorted_query.begin(), sorted_query.end());
   std::sort(sorted_root.begin(), sorted_root.end());
   return std::includes(sorted_root.begin(), sorted_root.end(),
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
   _group_is_special_block_root.assign(_num_groups + 1, 0);
   _group_is_trivial_special_block_root.assign(_num_groups + 1, 0);
   _point_to_special_block.assign(_num_points, 0);
   _point_is_special_block_root.assign(_num_points, 0);
   _special_block_summary.num_blocks = static_cast<IdxType>(_special_blocks.size());
   _special_block_summary.trivial_blocks = 0;
   _special_block_summary.member_groups = 0;
   _special_block_summary.member_points = 0;
   _special_block_summary.child_block_edges = 0;

   for (const SpecialBlock &block : _special_blocks)
   {
      _special_block_summary.member_groups += static_cast<IdxType>(block.member_group_ids.size());
      _special_block_summary.member_points += block.point_count;
      _special_block_summary.child_block_edges += static_cast<IdxType>(block.child_block_ids.size());
      if (block.is_trivial())
         _special_block_summary.trivial_blocks += 1;
      if (block.root_group_id > 0 && block.root_group_id < _group_is_special_block_root.size())
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
         _group_id_to_special_block[group_id] = block.block_id;
         const auto &range = _group_id_to_range[group_id];
         for (IdxType point_id = range.first; point_id < range.second && point_id < _point_to_special_block.size(); ++point_id)
            _point_to_special_block[point_id] = block.block_id;
      }
   }
}

void UniNavGraph::build_special_blocks()
{
   _special_blocks.clear();
   _group_id_to_special_block.assign(_num_groups + 1, 0);
   _group_is_special_block_root.assign(_num_groups + 1, 0);
   _group_is_trivial_special_block_root.assign(_num_groups + 1, 0);
   _point_to_special_block.assign(_num_points, 0);
   _point_is_special_block_root.assign(_num_points, 0);
   _special_edges_by_point.clear();
   _special_heavy_edges_by_point.clear();
   _special_block_summary = {};
   _special_block_summary.threshold = static_cast<IdxType>(_build_config.special_block_min_points);

   if (!_build_config.special_blocks_enabled)
      return;

   const auto start = std::chrono::high_resolution_clock::now();
   std::vector<SpecialTrieNode> nodes(1);
   nodes[0].parent = -1;
   for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
   {
      int cur = 0;
      const auto &labels = _group_id_to_label_set[group_id];
      for (LabelType label : labels)
      {
         auto it = nodes[cur].children.find(label);
         if (it == nodes[cur].children.end())
         {
            const int next = static_cast<int>(nodes.size());
            nodes[cur].children[label] = next;
            SpecialTrieNode node;
            node.label = label;
            node.parent = cur;
            node.labels = nodes[cur].labels;
            node.labels.push_back(label);
            nodes.push_back(std::move(node));
            cur = next;
         }
         else
         {
            cur = it->second;
         }
      }
      const auto &range = _group_id_to_range[group_id];
      nodes[cur].group_id = group_id;
      nodes[cur].group_points = range.second - range.first;
   }

   std::vector<IdxType> node_to_block(nodes.size(), 0);
   for (int idx = static_cast<int>(nodes.size()) - 1; idx >= 0; --idx)
   {
      IdxType subtree_points = nodes[idx].group_points;
      IdxType uncovered_points = nodes[idx].group_points;
      for (const auto &kv : nodes[idx].children)
      {
         subtree_points += nodes[kv.second].subtree_points;
         uncovered_points += nodes[kv.second].uncovered_points;
      }
      nodes[idx].subtree_points = subtree_points;
      nodes[idx].uncovered_points = uncovered_points;
      if (idx == 0)
         continue;
      if (uncovered_points > static_cast<IdxType>(_build_config.special_block_min_points))
      {
         SpecialBlock block;
         block.block_id = static_cast<IdxType>(_special_blocks.size() + 1);
         block.subtree_point_count = subtree_points;
         block.root_labels = nodes[idx].labels;
         collect_special_block_members(nodes, idx, block, node_to_block);
         if (block.member_group_ids.empty())
            continue;
         block.root_group_id = block.member_group_ids.front();
         _special_blocks.push_back(std::move(block));
         node_to_block[idx] = static_cast<IdxType>(_special_blocks.size());
         nodes[idx].uncovered_points = 0;
      }
   }

   rebuild_special_block_indexes();

   const double ms = std::chrono::duration<double, std::milli>(
                         std::chrono::high_resolution_clock::now() - start)
                         .count();
   _special_block_summary.metadata_ms = ms;
   std::cout << "[special_blocks] enabled=1 threshold=" << _special_block_summary.threshold
             << " blocks=" << _special_block_summary.num_blocks
             << " trivial_blocks=" << _special_block_summary.trivial_blocks
             << " member_groups=" << _special_block_summary.member_groups
             << " member_points=" << _special_block_summary.member_points
             << " child_block_edges=" << _special_block_summary.child_block_edges
             << " ms=" << ms << std::endl;
   prof_logf("[PROF] special_blocks enabled=1 threshold=%u blocks=%u trivial_blocks=%u member_groups=%u member_points=%u child_block_edges=%u ms=%.3f",
             static_cast<unsigned>(_special_block_summary.threshold),
             static_cast<unsigned>(_special_block_summary.num_blocks),
             static_cast<unsigned>(_special_block_summary.trivial_blocks),
             static_cast<unsigned>(_special_block_summary.member_groups),
             static_cast<unsigned>(_special_block_summary.member_points),
             static_cast<unsigned>(_special_block_summary.child_block_edges),
             ms);
}

void UniNavGraph::build_special_edge_overlay()
{
   _special_edges_by_point.assign(_num_points, {});
   _special_heavy_edges_by_point.clear();
   _special_block_summary.special_edges = 0;
   _special_block_summary.intra_special_edges = 0;
   _special_block_summary.inter_special_edges = 0;

   if (_special_blocks.empty() || !_graph)
      return;

   std::vector<std::vector<IdxType>> block_points(_special_blocks.size() + 1);
   for (const SpecialBlock &block : _special_blocks)
      if (block.block_id < block_points.size())
         block_points[block.block_id] = collect_block_points(block, _group_id_to_range);
   std::vector<std::shared_ptr<Vamana>> block_indexes(_special_blocks.size() + 1);

   const auto overlay_start = std::chrono::high_resolution_clock::now();
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
   size_t complete_intra_blocks = 0;
   size_t complete_intra_points = 0;
   size_t cpu_intra_vamana_blocks = 0;
   size_t cpu_intra_vamana_points = 0;
   size_t gpu_intra_blocks = 0;
   size_t gpu_intra_points = 0;
   size_t gpu_intra_fallback_blocks = 0;
   const auto intra_start = std::chrono::high_resolution_clock::now();
   for (const SpecialBlock &block : _special_blocks)
   {
      if (block.block_id >= block_points.size())
         continue;
      const auto &points = block_points[block.block_id];
      const IdxType n = static_cast<IdxType>(points.size());
      if (n <= 1)
         continue;

      auto local_graph = std::make_shared<Graph>(n);
      if (n <= special_intra_complete_threshold)
      {
         if (special_intra_exact_topk)
            build_exact_topk_graph_for_points(local_graph, _base_storage, _distance_handler, points, special_block_max_degree);
         else if (special_intra_bounded_complete)
            build_bounded_complete_graph(local_graph, n, special_block_max_degree);
         else
            build_complete_graph(local_graph, n);
         block_indexes[block.block_id] = std::make_shared<Vamana>(
             std::make_shared<SpecialBlockStorageView>(_base_storage, points),
             _distance_handler,
             local_graph,
             0);
         complete_intra_blocks += 1;
         complete_intra_points += static_cast<size_t>(n);
      }
      else
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
             << " complete_blocks=" << complete_intra_blocks
             << " complete_points=" << complete_intra_points
             << " cpu_vamana_blocks=" << cpu_intra_vamana_blocks
             << " cpu_vamana_points=" << cpu_intra_vamana_points
             << std::endl;
   prof_logf("[PROF] special_edges.gpu_intra enabled=%d blocks=%zu points=%zu fallback_blocks=%zu group_complete_threshold_nx=%u special_complete_threshold_nx=%u special_bounded_complete=%d special_exact_topk=%d complete_blocks=%zu complete_points=%zu cpu_vamana_blocks=%zu cpu_vamana_points=%zu",
             gpu_intra_enabled ? 1 : 0,
             gpu_intra_blocks,
             gpu_intra_points,
             gpu_intra_fallback_blocks,
             static_cast<unsigned>(group_cfg.complete_threshold),
             static_cast<unsigned>(special_intra_complete_threshold),
             special_intra_bounded_complete ? 1 : 0,
             special_intra_exact_topk ? 1 : 0,
             complete_intra_blocks,
             complete_intra_points,
             cpu_intra_vamana_blocks,
             cpu_intra_vamana_points);

   const CpuHybridCrossSettings hybrid_cfg = make_cpu_hybrid_cross_settings();
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
            const bool use_exact_scan =
                dst_points.size() <= static_cast<size_t>(hybrid_cfg.target_exact_max_nx) ||
                pair_work <= static_cast<unsigned long long>(hybrid_cfg.work_threshold);
            const auto pair_start = std::chrono::high_resolution_clock::now();
            SearchCacheList search_cache_list(std::max<uint32_t>(1, _num_threads),
                                              static_cast<IdxType>(dst_points.size()),
                                              _Lbuild);
            const auto pair_loop_start = std::chrono::high_resolution_clock::now();

            unsigned long long inter_edges_added = 0;
#pragma omp parallel for schedule(dynamic, 64) num_threads(_num_threads) reduction(+ : inter_edges_added)
            for (IdxType src_idx = 0; src_idx < static_cast<IdxType>(src_points.size()); ++src_idx)
            {
               const IdxType source = src_points[src_idx];
               SearchQueue local_topk;
               local_topk.reserve(special_block_num_cross_edges);
               append_cross_edges_from_target_points(source,
                                                     dst_points,
                                                     child_index,
                                                     &search_cache_list,
                                                     use_exact_scan,
                                                     local_topk,
                                                     special_block_num_cross_edges);
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
               if (use_exact_scan)
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
                << std::endl;
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
             << " intra_ms=" << intra_ms
             << " inter_ms=" << inter_ms
             << " ms=" << overlay_ms << std::endl;
   prof_logf("[PROF] special_edges sidecar_edges=%u intra=%u inter=%u intra_ms=%.3f inter_ms=%.3f ms=%.3f",
             static_cast<unsigned>(_special_block_summary.special_edges),
             static_cast<unsigned>(_special_block_summary.intra_special_edges),
             static_cast<unsigned>(_special_block_summary.inter_special_edges),
             intra_ms,
             inter_ms,
             overlay_ms);
}

void UniNavGraph::save_special_blocks(const std::string &prefix) const
{
   if (!_build_config.special_blocks_enabled)
      return;
   const auto save_start = std::chrono::high_resolution_clock::now();
   double metadata_ms = 0.0;
   double member_ms = 0.0;
   double child_ms = 0.0;
   double edge_ms = 0.0;
   size_t light_edge_rows = 0;
   size_t heavy_edge_rows = 0;
   const auto metadata_start = std::chrono::high_resolution_clock::now();
   {
      std::ofstream out(prefix + "special_blocks.csv");
      out << "block_id,root_group_id,root_labels,point_count,subtree_point_count,is_trivial,member_group_count,child_block_count\n";
      for (const SpecialBlock &block : _special_blocks)
      {
         out << block.block_id << ','
             << block.root_group_id << ','
             << join_labels_for_special_path(block.root_labels) << ','
             << block.point_count << ','
             << block.subtree_point_count << ','
             << (block.is_trivial() ? 1 : 0) << ','
             << block.member_group_ids.size() << ','
             << block.child_block_ids.size() << '\n';
      }
   }
   metadata_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - metadata_start)
                     .count();
   const auto member_start = std::chrono::high_resolution_clock::now();
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
   const auto edge_start = std::chrono::high_resolution_clock::now();
   {
      const bool binary_only_sidecar = env_flag("UNG_SPECIAL_EDGE_BINARY_ONLY");
      const bool write_binary_sidecar = binary_only_sidecar || env_flag("UNG_SPECIAL_EDGE_BINARY");
      std::ofstream out;
      if (!binary_only_sidecar)
         out.open(prefix + "special_edges.csv");
      std::unique_ptr<std::ofstream> heavy_out;
      std::unique_ptr<std::ofstream> binary_out;
      std::unique_ptr<std::ofstream> heavy_binary_out;
      uint64_t binary_light_count = 0;
      uint64_t binary_heavy_count = 0;
      if (write_binary_sidecar)
      {
         binary_out = std::make_unique<std::ofstream>(prefix + "special_edges.bin", std::ios::binary);
         binary_out->write(reinterpret_cast<const char *>(&binary_light_count), sizeof(binary_light_count));
      }
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
         if (write_binary_sidecar)
         {
            heavy_binary_out = std::make_unique<std::ofstream>(prefix + "special_heavy_edges.bin", std::ios::binary);
            heavy_binary_out->write(reinterpret_cast<const char *>(&binary_heavy_count), sizeof(binary_heavy_count));
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
            if (save_heavy_pair_work_threshold > 0 && edge.kind == SpecialEdgeKind::InterBlock &&
                edge.target_point_id < _point_to_special_block.size())
            {
               const IdxType child_id = _point_to_special_block[edge.target_point_id];
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
            if (write_binary_sidecar)
            {
               if (write_heavy && heavy_binary_out)
               {
                  write_special_edge_binary_record(*heavy_binary_out, source, edge);
                  binary_heavy_count += 1;
               }
               else if (!write_heavy && binary_out)
               {
                  write_special_edge_binary_record(*binary_out, source, edge);
                  binary_light_count += 1;
               }
            }
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
      if (binary_out)
      {
         binary_out->seekp(0);
         binary_out->write(reinterpret_cast<const char *>(&binary_light_count), sizeof(binary_light_count));
      }
      if (heavy_binary_out)
      {
         heavy_binary_out->seekp(0);
         heavy_binary_out->write(reinterpret_cast<const char *>(&binary_heavy_count), sizeof(binary_heavy_count));
      }
   }
   edge_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - edge_start)
                .count();
   const double total_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::high_resolution_clock::now() - save_start)
                               .count();
   std::cout << "[special_blocks][save] metadata_ms=" << metadata_ms
             << " members_ms=" << member_ms
             << " children_ms=" << child_ms
             << " edges_ms=" << edge_ms
             << " total_ms=" << total_ms
             << " light_edges=" << light_edge_rows
             << " heavy_edges=" << heavy_edge_rows
             << std::endl;
}

void UniNavGraph::load_special_blocks(const std::string &prefix,
                                      const std::map<std::string, std::string> &meta_data)
{
   _special_blocks.clear();
   _special_edges_by_point.clear();
   _special_heavy_edges_by_point.clear();
   _special_block_summary = {};
   auto meta_it = meta_data.find("special_blocks_enabled");
   if (meta_it == meta_data.end() || meta_it->second != "1")
   {
      rebuild_special_block_indexes();
      return;
   }

   auto threshold_it = meta_data.find("special_block_min_points");
   if (threshold_it != meta_data.end())
      _special_block_summary.threshold = static_cast<IdxType>(std::stoul(threshold_it->second));

   {
      std::ifstream in(prefix + "special_blocks.csv");
      if (!in)
      {
         std::cerr << "[special_blocks] metadata enabled in meta but special_blocks.csv is missing." << std::endl;
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
         block.root_group_id = static_cast<IdxType>(std::stoul(cols[1]));
         block.root_labels = parse_label_list(cols[2]);
         block.point_count = static_cast<IdxType>(std::stoul(cols[3]));
         block.subtree_point_count = static_cast<IdxType>(std::stoul(cols[4]));
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
      std::sort(block.member_group_ids.begin(), block.member_group_ids.end());
      block.member_group_ids.erase(std::unique(block.member_group_ids.begin(), block.member_group_ids.end()),
                                   block.member_group_ids.end());
      std::sort(block.child_block_ids.begin(), block.child_block_ids.end());
      block.child_block_ids.erase(std::unique(block.child_block_ids.begin(), block.child_block_ids.end()),
                                  block.child_block_ids.end());
   }
   rebuild_special_block_indexes();

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
      _special_heavy_edges_by_point.assign(_num_points, {});
   }

   _special_edges_by_point.assign(_num_points, {});
   size_t heavy_edges_loaded = 0;
   const bool load_persisted_heavy_edges = env_flag("UNG_SPECIAL_HEAVY_EDGE_SEARCH");
   const bool csv_edges_exist = static_cast<bool>(std::ifstream(prefix + "special_edges.csv"));
   const bool read_binary_sidecar = env_flag("UNG_SPECIAL_EDGE_BINARY") || !csv_edges_exist;
   size_t light_binary_edges_loaded = 0;
   const bool loaded_light_from_binary =
       read_binary_sidecar &&
       read_special_edge_binary_file(prefix + "special_edges.bin",
                                     _special_edges_by_point,
                                     _special_block_summary,
                                     light_binary_edges_loaded);
   std::ifstream edge_in(prefix + "special_edges.csv");
   if (!loaded_light_from_binary && edge_in)
   {
      std::string line;
      std::getline(edge_in, line);
      while (std::getline(edge_in, line))
      {
         const auto cols = split_csv_line(line);
         if (cols.size() < 4)
            continue;
         const IdxType source = static_cast<IdxType>(std::stoul(cols[0]));
         if (source >= _special_edges_by_point.size())
            continue;
         SpecialEdge edge;
         edge.target_point_id = static_cast<IdxType>(std::stoul(cols[1]));
         edge.special_block_id = static_cast<IdxType>(std::stoul(cols[2]));
         edge.kind = cols[3] == "inter" ? SpecialEdgeKind::InterBlock : SpecialEdgeKind::IntraBlock;
         bool use_heavy_sidecar = false;
         if (heavy_pair_work_threshold > 0 && edge.kind == SpecialEdgeKind::InterBlock &&
             edge.target_point_id < _point_to_special_block.size())
         {
            const IdxType child_id = _point_to_special_block[edge.target_point_id];
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
   if (load_persisted_heavy_edges && read_binary_sidecar)
   {
      if (_special_heavy_edges_by_point.empty())
         _special_heavy_edges_by_point.assign(_num_points, {});
      read_special_edge_binary_file(prefix + "special_heavy_edges.bin",
                                    _special_heavy_edges_by_point,
                                    _special_block_summary,
                                    heavy_edges_loaded);
   }
   std::ifstream heavy_edge_in(prefix + "special_heavy_edges.csv");
   if (load_persisted_heavy_edges && heavy_edges_loaded == 0 && heavy_edge_in)
   {
      if (_special_heavy_edges_by_point.empty())
         _special_heavy_edges_by_point.assign(_num_points, {});
      std::string line;
      std::getline(heavy_edge_in, line);
      while (std::getline(heavy_edge_in, line))
      {
         const auto cols = split_csv_line(line);
         if (cols.size() < 4)
            continue;
         const IdxType source = static_cast<IdxType>(std::stoul(cols[0]));
         if (source >= _special_heavy_edges_by_point.size())
            continue;
         SpecialEdge edge;
         edge.target_point_id = static_cast<IdxType>(std::stoul(cols[1]));
         edge.special_block_id = static_cast<IdxType>(std::stoul(cols[2]));
         edge.kind = cols[3] == "inter" ? SpecialEdgeKind::InterBlock : SpecialEdgeKind::IntraBlock;
         _special_heavy_edges_by_point[source].push_back(edge);
         _special_block_summary.special_edges += 1;
         if (edge.kind == SpecialEdgeKind::InterBlock)
            _special_block_summary.inter_special_edges += 1;
         else
            _special_block_summary.intra_special_edges += 1;
         heavy_edges_loaded += 1;
      }
   }
   std::cout << "[special_blocks] loaded blocks=" << _special_block_summary.num_blocks
             << " trivial_blocks=" << _special_block_summary.trivial_blocks
             << " sidecar_edges=" << _special_block_summary.special_edges
             << " binary_light_edges=" << light_binary_edges_loaded
             << " heavy_edges=" << heavy_edges_loaded
             << " heavy_pair_work_threshold=" << heavy_pair_work_threshold
             << std::endl;
}

} // namespace ANNS
