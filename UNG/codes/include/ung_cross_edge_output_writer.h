#ifndef ANNS_UNG_CROSS_EDGE_OUTPUT_WRITER_H
#define ANNS_UNG_CROSS_EDGE_OUTPUT_WRITER_H

#include "graph.h"
#include "ung_cross_edge_result.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace ANNS
{

struct CrossEdgeCsrOutput
{
   std::vector<std::size_t> row_offsets;
   std::vector<IdxType> col_indices;
};

class CrossEdgeGraphMaterializationWriter
{
public:
   CrossEdgeGraphMaterializationWriter(Graph &graph, IdxType topk);

   void append_cross_edges(const CrossEdgeHostOutputView &outputs,
                           IdxType num_points,
                           CrossEdgeBuildTiming &timing);

   void append_materialized_additional_edges(
       const std::vector<std::vector<std::pair<IdxType, IdxType>>> &additional_edges,
       IdxType num_groups,
       CrossEdgeBuildTiming &timing);

   CrossEdgeCsrOutput build_csr_adjacency(const CrossEdgeHostOutputView &outputs,
                                          IdxType num_points) const;

private:
   Graph &graph_;
   IdxType topk_ = 0;

   void append_cross_edges_for_point(const CrossEdgeHostOutputView &outputs, IdxType point_id);
   std::size_t count_cross_edges_for_point(const CrossEdgeHostOutputView &outputs, IdxType point_id) const;
   void append_cross_edges_for_point_to_csr(const CrossEdgeHostOutputView &outputs,
                                            IdxType point_id,
                                            std::vector<IdxType> &col_indices,
                                            std::size_t &write_offset) const;
};

} // namespace ANNS

#endif // ANNS_UNG_CROSS_EDGE_OUTPUT_WRITER_H
