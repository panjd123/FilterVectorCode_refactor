#ifndef ANNS_NAVIX_HNSW_H
#define ANNS_NAVIX_HNSW_H

#include "config.h"
#include "distance.h"
#include "graph.h"
#include "search_queue.h"
#include "storage.h"
#include "visited_set.h"

#include <memory>
#include <vector>

namespace ANNS
{

class NavixHnsw
{
public:
   explicit NavixHnsw(bool verbose = true) : _verbose(verbose) {}

   void build(std::shared_ptr<IStorage> base_storage,
              std::shared_ptr<DistanceHandler> distance_handler,
              std::shared_ptr<Graph> graph,
              IdxType max_degree,
              IdxType ef_construction,
              IdxType max_candidates,
              uint32_t num_threads);

   IdxType get_entry_point() const { return _entry_point; }

private:
   std::vector<Candidate> search_layer(const char *query,
                                       IdxType entry_point,
                                       IdxType ef,
                                       IdxType exclude_id,
                                       VisitedSet &visited) const;
   std::vector<IdxType> select_neighbors(IdxType src,
                                         std::vector<Candidate> candidates,
                                         IdxType max_degree) const;
   void prune_node(IdxType node, IdxType max_degree);
   void connect_bidirectional(IdxType src, const std::vector<IdxType> &neighbors, IdxType max_degree);

   std::shared_ptr<IStorage> _base_storage;
   std::shared_ptr<DistanceHandler> _distance_handler;
   std::shared_ptr<Graph> _graph;
   IdxType _entry_point = 0;
   bool _verbose = true;
};

} // namespace ANNS

#endif // ANNS_NAVIX_HNSW_H
