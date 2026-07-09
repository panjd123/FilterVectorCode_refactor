#ifndef ANNS_UNG_GRAPH_SEARCH_BACKEND_H
#define ANNS_UNG_GRAPH_SEARCH_BACKEND_H

#include "config.h"

#include <cstddef>
#include <vector>

namespace ANNS
{

class Graph;
struct CrossEdgeCsrOutput;

struct GraphNeighborView
{
   const IdxType *ids = nullptr;
   std::size_t size = 0;
};

class GraphSearchBackend
{
public:
   explicit GraphSearchBackend(const Graph &graph);
   explicit GraphSearchBackend(const CrossEdgeCsrOutput &csr);

   GraphNeighborView neighbors(IdxType point_id) const;

private:
   enum class StorageKind
   {
      Graph,
      Csr
   };

   StorageKind storage_kind_ = StorageKind::Graph;
   const Graph *graph_ = nullptr;
   const std::vector<std::size_t> *csr_row_offsets_ = nullptr;
   const std::vector<IdxType> *csr_col_indices_ = nullptr;
};

} // namespace ANNS

#endif // ANNS_UNG_GRAPH_SEARCH_BACKEND_H
