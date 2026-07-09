#include "include/ung_cross_edge_output_writer.h"

#include "include/search_queue.h"
#include "include/ung_prof_log.h"

#include <limits>
#include <omp.h>

namespace ANNS
{

CrossEdgeGraphMaterializationWriter::CrossEdgeGraphMaterializationWriter(Graph &graph, IdxType topk)
    : graph_(graph), topk_(topk)
{
}

void CrossEdgeGraphMaterializationWriter::append_cross_edges(
    const CrossEdgeHostOutputView &outputs,
    IdxType num_points,
    CrossEdgeBuildTiming &timing)
{
   ScopedTimerMs t("cross_edges.merge_cross_ms", &timing.merge_cross_ms);
#pragma omp parallel for schedule(dynamic, 4096)
   for (IdxType point_id = 0; point_id < num_points; ++point_id)
   {
      append_cross_edges_for_point(outputs, point_id);
   }
}

void CrossEdgeGraphMaterializationWriter::append_materialized_additional_edges(
    const std::vector<std::vector<std::pair<IdxType, IdxType>>> &additional_edges,
    IdxType num_groups,
    CrossEdgeBuildTiming &timing)
{
   ScopedTimerMs t("cross_edges.merge_additional_ms", &timing.merge_additional_ms);
#pragma omp parallel for schedule(dynamic, 256)
   for (IdxType group_id = 1; group_id <= num_groups; ++group_id)
   {
      for (const auto &[from_id, to_id] : additional_edges[group_id])
         graph_.neighbors[from_id].emplace_back(to_id);
   }
}

CrossEdgeCsrOutput CrossEdgeGraphMaterializationWriter::build_csr_adjacency(
    const CrossEdgeHostOutputView &outputs,
    IdxType num_points) const
{
   CrossEdgeCsrOutput csr;
   csr.row_offsets.resize(static_cast<size_t>(num_points) + 1, 0);

   for (IdxType point_id = 0; point_id < num_points; ++point_id)
   {
      csr.row_offsets[static_cast<size_t>(point_id) + 1] =
          csr.row_offsets[static_cast<size_t>(point_id)] +
          count_cross_edges_for_point(outputs, point_id);
   }

   csr.col_indices.resize(csr.row_offsets.back());
#pragma omp parallel for schedule(dynamic, 4096)
   for (IdxType point_id = 0; point_id < num_points; ++point_id)
   {
      std::size_t write_offset = csr.row_offsets[static_cast<size_t>(point_id)];
      append_cross_edges_for_point_to_csr(outputs, point_id, csr.col_indices, write_offset);
   }
   return csr;
}

void CrossEdgeGraphMaterializationWriter::append_cross_edges_for_point(
    const CrossEdgeHostOutputView &outputs,
    IdxType point_id)
{
   auto &dst = graph_.neighbors[point_id];
   if (outputs.uses_id_vectors())
   {
      const auto &src = (*outputs.id_vectors)[point_id];
      dst.insert(dst.end(), src.begin(), src.end());
   }
   else if (outputs.uses_flat_ids())
   {
      const auto &flat_ids = *outputs.flat_ids;
      const size_t base = static_cast<size_t>(point_id) * static_cast<size_t>(outputs.topk);
      for (IdxType k = 0; k < topk_; ++k)
      {
         const IdxType neighbor_id = flat_ids[base + static_cast<size_t>(k)];
         if (neighbor_id != std::numeric_limits<IdxType>::max())
            dst.emplace_back(neighbor_id);
      }
   }
   else
   {
      const auto &cross_group_neighbors = *outputs.search_queues;
      for (auto k = 0; k < cross_group_neighbors[point_id].size(); ++k)
         dst.emplace_back(cross_group_neighbors[point_id][k].id);
   }
}

std::size_t CrossEdgeGraphMaterializationWriter::count_cross_edges_for_point(
    const CrossEdgeHostOutputView &outputs,
    IdxType point_id) const
{
   if (outputs.uses_id_vectors())
      return (*outputs.id_vectors)[point_id].size();
   if (outputs.uses_flat_ids())
   {
      const auto &flat_ids = *outputs.flat_ids;
      const size_t base = static_cast<size_t>(point_id) * static_cast<size_t>(outputs.topk);
      std::size_t count = 0;
      for (IdxType k = 0; k < topk_; ++k)
      {
         if (flat_ids[base + static_cast<size_t>(k)] != std::numeric_limits<IdxType>::max())
            ++count;
      }
      return count;
   }
   return static_cast<std::size_t>((*outputs.search_queues)[point_id].size());
}

void CrossEdgeGraphMaterializationWriter::append_cross_edges_for_point_to_csr(
    const CrossEdgeHostOutputView &outputs,
    IdxType point_id,
    std::vector<IdxType> &col_indices,
    std::size_t &write_offset) const
{
   if (outputs.uses_id_vectors())
   {
      const auto &src = (*outputs.id_vectors)[point_id];
      for (IdxType neighbor_id : src)
         col_indices[write_offset++] = neighbor_id;
   }
   else if (outputs.uses_flat_ids())
   {
      const auto &flat_ids = *outputs.flat_ids;
      const size_t base = static_cast<size_t>(point_id) * static_cast<size_t>(outputs.topk);
      for (IdxType k = 0; k < topk_; ++k)
      {
         const IdxType neighbor_id = flat_ids[base + static_cast<size_t>(k)];
         if (neighbor_id != std::numeric_limits<IdxType>::max())
            col_indices[write_offset++] = neighbor_id;
      }
   }
   else
   {
      const auto &cross_group_neighbors = *outputs.search_queues;
      for (auto k = 0; k < cross_group_neighbors[point_id].size(); ++k)
         col_indices[write_offset++] = cross_group_neighbors[point_id][k].id;
   }
}

} // namespace ANNS
