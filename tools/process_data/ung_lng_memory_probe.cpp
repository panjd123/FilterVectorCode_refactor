// Count the LNG representation using the current native loader's operations.
// This probe loads only group adjacency; no vectors, graph rebuild or queries.
#include <bits/stdc++.h>
#include "label_nav_graph.h"
#include "utils.h"

using namespace ANNS;
using Bytes = uint64_t;

template<class T> Bytes allocated(const std::vector<T>& values) {
    return values.capacity() * sizeof(T);
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const std::string path = argv[1];
    const auto groups = static_cast<IdxType>(std::stoul(argv[2]));
    const auto expected_edges = std::stoull(argv[3]);
    LabelNavGraph lng(groups + 1);
    if (!std::ifstream(path)) throw std::runtime_error("Missing LNG adjacency file");
    // Same construction, loading and reverse-adjacency reconstruction as
    // UniNavGraph::load. In/out vectors share one LNG object with topology.
    load_2d_vectors(path, lng.out_neighbors);
    if (lng.out_neighbors.size() < groups + 1ULL ||
        lng.out_neighbors.size() > groups + 2ULL)
        throw std::runtime_error("Unexpected LNG group count");
    lng.in_neighbors.assign(groups + 1, {});
    Bytes edge_count = 0, outgoing = 0, incoming = 0;
    for (IdxType source = 1; source < lng.out_neighbors.size(); ++source) {
        for (const auto target : lng.out_neighbors[source]) {
            if (target == 0 || target > groups)
                throw std::runtime_error("LNG target outside group range");
            lng.in_neighbors[target].push_back(source);
            ++edge_count;
        }
    }
    if (edge_count != expected_edges)
        throw std::runtime_error("LNG edge count differs from native metadata");
    for (const auto& row : lng.out_neighbors) outgoing += allocated(row);
    for (const auto& row : lng.in_neighbors) incoming += allocated(row);
    const Bytes outer = allocated(lng.out_neighbors) + allocated(lng.in_neighbors);
    const Bytes degrees = allocated(lng.in_degree) + allocated(lng.out_degree);
    using Allocation = std::_Sp_counted_ptr_inplace<LabelNavGraph, std::allocator<void>, __gnu_cxx::_S_atomic>;
    const Bytes control = sizeof(Allocation) - sizeof(LabelNavGraph);
    const Bytes total = sizeof(lng) + outer + outgoing + incoming + degrees + control;
    std::cout << "LNG_MEMORY bytes=" << total << " edges=" << edge_count
              << " object_bytes=" << sizeof(lng) << " outer_array_bytes=" << outer
              << " outgoing_capacity_bytes=" << outgoing << " incoming_capacity_bytes=" << incoming
              << " degree_capacity_bytes=" << degrees << " shared_control_bytes=" << control << std::endl;
}
