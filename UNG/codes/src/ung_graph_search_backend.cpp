#include "include/ung_graph_search_backend.h"

#include "include/graph.h"
#include "include/ung_cross_edge_output_writer.h"

namespace ANNS
{

GraphSearchBackend::GraphSearchBackend(const Graph &graph)
    : storage_kind_(StorageKind::Graph), graph_(&graph)
{
}

GraphSearchBackend::GraphSearchBackend(const CrossEdgeCsrOutput &csr)
    : storage_kind_(StorageKind::Csr),
      csr_row_offsets_(&csr.row_offsets),
      csr_col_indices_(&csr.col_indices)
{
}

GraphNeighborView GraphSearchBackend::neighbors(IdxType point_id) const
{
   if (storage_kind_ == StorageKind::Csr)
   {
      const std::size_t row = static_cast<std::size_t>(point_id);
      const std::size_t begin = (*csr_row_offsets_)[row];
      const std::size_t end = (*csr_row_offsets_)[row + 1];
      return GraphNeighborView{csr_col_indices_->data() + begin, end - begin};
   }

   const auto &neighbors = graph_->neighbors[point_id];
   return GraphNeighborView{neighbors.data(), neighbors.size()};
}

} // namespace ANNS
