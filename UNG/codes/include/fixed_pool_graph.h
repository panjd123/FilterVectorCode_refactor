#ifndef ANNS_FIXED_POOL_GRAPH_H
#define ANNS_FIXED_POOL_GRAPH_H

#include <memory>
#include "config.h"
#include "graph.h"
#include "storage.h"
#include "distance.h"

namespace ANNS {

struct FixedPoolBuildStats {
    double total_ms = 0.0;
    double h2d_ms = 0.0;
    double init_ms = 0.0;
    double refine_ms = 0.0;
    double prune_ms = 0.0;
    double d2h_ms = 0.0;
    IdxType pool_size = 0;
    IdxType iterations = 0;
    IdxType num_points = 0;
    IdxType dim = 0;
};

// Experimental GPU NN-Descent-style graph builder.
// It keeps a fixed-size neighbor pool per point, repeatedly expands through
// neighbors-of-neighbors, then robust-prunes the pool to graph edges.
FixedPoolBuildStats build_fixed_pool_graph_gpu(
    std::shared_ptr<IStorage> storage,
    std::shared_ptr<DistanceHandler> distance_handler,
    std::shared_ptr<Graph> graph,
    IdxType max_degree,
    IdxType requested_pool_size,
    float alpha,
    uint32_t iterations,
    uint64_t seed);

} // namespace ANNS

#endif // ANNS_FIXED_POOL_GRAPH_H
