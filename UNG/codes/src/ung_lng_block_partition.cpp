#include "include/ung_lng_block_partition.h"

#include <algorithm>
#include <deque>
#include <random>
#include <stdexcept>
#include <utility>

namespace ANNS
{
namespace
{

struct TreeEdge
{
   IdxType src = 0;
   IdxType dst = 0;
};

struct LngBlockAccumulator
{
   IdxType uncovered_points = 0;
   IdxType subtree_points = 0;
   IdxType produced_block_id = 0;
   std::vector<IdxType> member_group_ids;
   std::vector<IdxType> child_block_ids;
};

IdxType group_point_count(const std::vector<IdxType> &group_points, IdxType group_id)
{
   return group_id < group_points.size() ? group_points[group_id] : 0;
}

std::vector<IdxType> sorted_unique(std::vector<IdxType> values)
{
   std::sort(values.begin(), values.end());
   values.erase(std::unique(values.begin(), values.end()), values.end());
   return values;
}

std::vector<LabelType> sorted_unique_labels(std::vector<LabelType> values)
{
   std::sort(values.begin(), values.end());
   values.erase(std::unique(values.begin(), values.end()), values.end());
   return values;
}

void append_all(std::vector<IdxType> &dst, const std::vector<IdxType> &src)
{
   dst.insert(dst.end(), src.begin(), src.end());
}

bool includes_all(const std::vector<LabelType> &superset,
                  const std::vector<LabelType> &subset)
{
   return std::includes(superset.begin(), superset.end(), subset.begin(), subset.end());
}

void add_out_edges(IdxType src,
                   const std::vector<std::vector<IdxType>> &out_neighbors,
                   const std::vector<uint8_t> &visited,
                   std::deque<TreeEdge> &bfs_edges,
                   std::vector<TreeEdge> &random_edges,
                   LngBlockTreeMode mode,
                   IdxType num_groups)
{
   if (src >= out_neighbors.size())
      return;
   for (IdxType dst : out_neighbors[src])
   {
      if (dst == 0 || dst > num_groups || visited[dst])
         continue;
      TreeEdge edge{src, dst};
      if (mode == LngBlockTreeMode::Bfs)
         bfs_edges.push_back(edge);
      else
         random_edges.push_back(edge);
   }
}

std::vector<IdxType> build_start_order(const std::vector<std::vector<IdxType>> &out_neighbors,
                                       IdxType num_groups)
{
   std::vector<IdxType> in_degree(static_cast<size_t>(num_groups) + 1, 0);
   for (IdxType group_id = 1; group_id <= num_groups; ++group_id)
   {
      if (group_id >= out_neighbors.size())
         continue;
      for (IdxType dst : out_neighbors[group_id])
      {
         if (dst > 0 && dst <= num_groups)
            in_degree[dst] += 1;
      }
   }

   std::vector<IdxType> starts;
   starts.reserve(num_groups);
   for (IdxType group_id = 1; group_id <= num_groups; ++group_id)
      if (in_degree[group_id] == 0)
         starts.push_back(group_id);
   for (IdxType group_id = 1; group_id <= num_groups; ++group_id)
      if (in_degree[group_id] != 0)
         starts.push_back(group_id);
   return starts;
}

void build_lng_spanning_forest(const std::vector<std::vector<IdxType>> &out_neighbors,
                               LngBlockTreeMode mode,
                               uint64_t seed,
                               std::vector<std::vector<IdxType>> &tree_children,
                               std::vector<IdxType> &visit_order)
{
   const IdxType num_groups = static_cast<IdxType>(out_neighbors.size() - 1);
   tree_children.assign(static_cast<size_t>(num_groups) + 1, {});
   visit_order.clear();
   visit_order.reserve(num_groups);

   std::vector<uint8_t> visited(static_cast<size_t>(num_groups) + 1, 0);
   std::deque<TreeEdge> bfs_edges;
   std::vector<TreeEdge> random_edges;
   std::mt19937_64 rng(seed);
   const std::vector<IdxType> starts = build_start_order(out_neighbors, num_groups);
   size_t start_idx = 0;
   IdxType visited_count = 0;

   auto start_component = [&]() -> bool {
      while (start_idx < starts.size())
      {
         const IdxType start = starts[start_idx++];
         if (start == 0 || start > num_groups || visited[start])
            continue;
         visited[start] = 1;
         visited_count += 1;
         visit_order.push_back(start);
         add_out_edges(start, out_neighbors, visited, bfs_edges, random_edges, mode, num_groups);
         return true;
      }
      return false;
   };

   while (visited_count < num_groups)
   {
      if (bfs_edges.empty() && random_edges.empty())
      {
         if (!start_component())
            break;
      }

      TreeEdge edge;
      bool have_edge = false;
      if (mode == LngBlockTreeMode::Bfs)
      {
         while (!bfs_edges.empty())
         {
            edge = bfs_edges.front();
            bfs_edges.pop_front();
            if (edge.dst > 0 && edge.dst <= num_groups && !visited[edge.dst])
            {
               have_edge = true;
               break;
            }
         }
      }
      else
      {
         while (!random_edges.empty())
         {
            std::uniform_int_distribution<size_t> pick(0, random_edges.size() - 1);
            const size_t pos = pick(rng);
            edge = random_edges[pos];
            random_edges[pos] = random_edges.back();
            random_edges.pop_back();
            if (edge.dst > 0 && edge.dst <= num_groups && !visited[edge.dst])
            {
               have_edge = true;
               break;
            }
         }
      }

      if (!have_edge)
         continue;

      visited[edge.dst] = 1;
      visited_count += 1;
      tree_children[edge.src].push_back(edge.dst);
      visit_order.push_back(edge.dst);
      add_out_edges(edge.dst, out_neighbors, visited, bfs_edges, random_edges, mode, num_groups);
   }
}

std::vector<IdxType> build_block_label_lng(const std::vector<SpecialBlock> &blocks,
                                           size_t block_idx)
{
   std::vector<IdxType> out;
   const auto &src_labels = blocks[block_idx].root_labels;
   for (size_t cand_idx = 0; cand_idx < blocks.size(); ++cand_idx)
   {
      if (cand_idx == block_idx)
         continue;
      const auto &cand_labels = blocks[cand_idx].root_labels;
      if (cand_labels.size() <= src_labels.size())
         continue;
      if (!includes_all(cand_labels, src_labels))
         continue;

      bool minimal = true;
      for (size_t other_idx = 0; other_idx < blocks.size(); ++other_idx)
      {
         if (other_idx == block_idx || other_idx == cand_idx)
            continue;
         const auto &other_labels = blocks[other_idx].root_labels;
         if (other_labels.size() <= src_labels.size() || other_labels.size() >= cand_labels.size())
            continue;
         if (includes_all(other_labels, src_labels) && includes_all(cand_labels, other_labels))
         {
            minimal = false;
            break;
         }
      }

      if (minimal)
         out.push_back(blocks[cand_idx].block_id);
   }
   return sorted_unique(std::move(out));
}

} // namespace

LngBlockPartitionResult build_lng_special_blocks(const LngBlockPartitionInput &input)
{
   if (!input.out_neighbors || !input.group_points || !input.group_labels)
      throw std::invalid_argument("build_lng_special_blocks requires non-null input vectors.");
   const auto &out_neighbors = *input.out_neighbors;
   const auto &group_points = *input.group_points;
   const auto &group_labels = *input.group_labels;
   if (out_neighbors.empty())
      return {};

   const IdxType num_groups = static_cast<IdxType>(out_neighbors.size() - 1);
   LngBlockPartitionResult result;
   result.group_to_block.assign(static_cast<size_t>(num_groups) + 1, 0);

   std::vector<std::vector<IdxType>> tree_children;
   std::vector<IdxType> visit_order;
   build_lng_spanning_forest(out_neighbors, input.tree_mode, input.tree_seed,
                             tree_children, visit_order);

   std::vector<LngBlockAccumulator> acc(static_cast<size_t>(num_groups) + 1);
   for (auto it = visit_order.rbegin(); it != visit_order.rend(); ++it)
   {
      const IdxType group_id = *it;
      LngBlockAccumulator cur;
      const IdxType own_points = group_point_count(group_points, group_id);
      cur.uncovered_points = own_points;
      cur.subtree_points = own_points;
      if (own_points > 0)
         cur.member_group_ids.push_back(group_id);

      for (IdxType child : tree_children[group_id])
      {
         cur.subtree_points += acc[child].subtree_points;
         if (acc[child].produced_block_id != 0)
         {
            cur.child_block_ids.push_back(acc[child].produced_block_id);
            continue;
         }
         cur.uncovered_points += acc[child].uncovered_points;
         append_all(cur.member_group_ids, acc[child].member_group_ids);
         append_all(cur.child_block_ids, acc[child].child_block_ids);
      }

      if (cur.uncovered_points > input.min_points)
      {
         SpecialBlock block;
         block.block_id = static_cast<IdxType>(result.blocks.size() + 1);
         block.root_group_id = group_id;
         block.point_count = cur.uncovered_points;
         block.subtree_point_count = cur.subtree_points;
         if (group_id < group_labels.size())
            block.root_labels = sorted_unique_labels(group_labels[group_id]);
         block.member_group_ids = sorted_unique(std::move(cur.member_group_ids));
         block.common_labels = compute_direct_member_common_labels(
             block.member_group_ids, group_labels);
         block.child_block_ids = sorted_unique(std::move(cur.child_block_ids));
         for (IdxType member_group_id : block.member_group_ids)
            if (member_group_id < result.group_to_block.size())
               result.group_to_block[member_group_id] = block.block_id;
         result.blocks.push_back(std::move(block));
         cur.produced_block_id = static_cast<IdxType>(result.blocks.size());
         cur.uncovered_points = 0;
         cur.member_group_ids.clear();
         cur.child_block_ids.clear();
      }

      acc[group_id] = std::move(cur);
   }

   for (size_t block_idx = 0; block_idx < result.blocks.size(); ++block_idx)
      result.blocks[block_idx].child_block_ids = build_block_label_lng(result.blocks, block_idx);

   return result;
}

} // namespace ANNS
