#ifndef ANNS_UNG_NAVIX_SEARCH_H
#define ANNS_UNG_NAVIX_SEARCH_H

#include "config.h"
#include "distance.h"
#include "search_cache.h"
#include "search_queue.h"
#include "storage.h"
#include "ung_graph_search_backend.h"

#include <memory>
#include <vector>

namespace ANNS
{

struct NavixSearchStats
{
   size_t distance_calcs = 0;
   size_t nodes_visited = 0;
   size_t one_hop_calls = 0;
   size_t directed_two_hop_calls = 0;
   size_t full_two_hop_calls = 0;
};

NavixSearchStats navix_adaptive_local_search(const char *query,
                                             const std::shared_ptr<IStorage> &base_storage,
                                             const std::shared_ptr<DistanceHandler> &distance_handler,
                                             const GraphSearchBackend &graph_backend,
                                             const std::vector<uint8_t> &filter_map,
                                             const std::vector<IdxType> &entry_points,
                                             IdxType ef_search,
                                             IdxType K,
                                             SearchCache &search_cache,
                                             SearchQueue &result);

} // namespace ANNS

#endif // ANNS_UNG_NAVIX_SEARCH_H
