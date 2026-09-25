#include "include/ung_group_topology.h"

#include "include/ung_special_block_trie.h"

#include <algorithm>
#include <stdexcept>

namespace ANNS
{
namespace
{
bool includes_all(const std::vector<LabelType> &superset,
                  const std::vector<LabelType> &subset)
{
   return std::includes(superset.begin(), superset.end(),
                        subset.begin(), subset.end());
}

std::vector<IdxType> minimal_supersets(
    size_t source_index,
    const std::vector<size_t> &layer_indices,
    const std::vector<SpecialBlock> &blocks)
{
   std::vector<IdxType> result;
   const auto &source_labels = blocks[source_index].root_labels;
   for (size_t candidate_index : layer_indices)
   {
      if (candidate_index == source_index)
         continue;
      const auto &candidate_labels = blocks[candidate_index].root_labels;
      if (candidate_labels.size() <= source_labels.size() ||
          !includes_all(candidate_labels, source_labels))
         continue;
      bool minimal = true;
      for (size_t other_index : layer_indices)
      {
         if (other_index == source_index || other_index == candidate_index)
            continue;
         const auto &other_labels = blocks[other_index].root_labels;
         if (other_labels.size() <= source_labels.size() ||
             other_labels.size() >= candidate_labels.size())
            continue;
         if (includes_all(other_labels, source_labels) &&
             includes_all(candidate_labels, other_labels))
         {
            minimal = false;
            break;
         }
      }
      if (minimal)
         result.push_back(blocks[candidate_index].block_id);
   }
   std::sort(result.begin(), result.end());
   result.erase(std::unique(result.begin(), result.end()), result.end());
   return result;
}
}

std::shared_ptr<LabelNavGraph> build_group_topology(
    GroupTopologyKind kind,
    const std::vector<std::vector<LabelType>> &group_labels,
    IdxType num_groups,
    const std::shared_ptr<LabelNavGraph> &lng_graph,
    GroupTopologyStats *stats)
{
   std::shared_ptr<LabelNavGraph> graph;
   if (kind == GroupTopologyKind::Lng)
   {
      if (!lng_graph)
         throw std::invalid_argument("LNG group topology requires an LNG graph");
      graph = lng_graph;
   }
   else
   {
      SpecialBlockTrieIndex trie;
      trie.build(group_labels, num_groups);
      graph = std::make_shared<LabelNavGraph>(num_groups + 1);
      for (const auto &edge : trie.terminal_successor_pairs())
      {
         graph->out_neighbors[edge.first].push_back(edge.second);
         graph->in_neighbors[edge.second].push_back(edge.first);
      }
   }

   if (stats != nullptr)
   {
      stats->kind = kind;
      stats->group_count = num_groups;
      stats->edge_count = 0;
      for (IdxType group_id = 1; group_id <= num_groups; ++group_id)
         stats->edge_count += graph->out_neighbors[group_id].size();
   }
   return graph;
}

void apply_layer_topology(GroupTopologyKind kind,
                          uint8_t level,
                          std::vector<SpecialBlock> &blocks)
{
   std::vector<size_t> layer_indices;
   for (size_t index = 0; index < blocks.size(); ++index)
      if (blocks[index].level == level)
         layer_indices.push_back(index);

   if (kind == GroupTopologyKind::Lng)
   {
      for (size_t index : layer_indices)
         blocks[index].child_block_ids = minimal_supersets(index, layer_indices, blocks);
      return;
   }

   for (size_t index : layer_indices)
   {
      auto &children = blocks[index].child_block_ids;
      children.erase(std::remove_if(children.begin(), children.end(),
                                    [&](IdxType child_id) {
                                       return child_id == 0 || child_id > blocks.size() ||
                                              blocks[child_id - 1].level != level;
                                    }),
                     children.end());
      std::sort(children.begin(), children.end());
      children.erase(std::unique(children.begin(), children.end()), children.end());
   }
}

} // namespace ANNS
