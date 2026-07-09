#ifndef ANNS_GPU_CROSS_EDGE_CPU_TINY_CUH
#define ANNS_GPU_CROSS_EDGE_CPU_TINY_CUH

// CPU exact fallback for very small cross-edge groups.
//
// Planning decides which target groups are too small to amortize GPU launch and
// transfer overhead. This helper executes those groups with the same topK
// semantics as the old inline lambda, keeping exact scan details out of the
// all-batched GPU route.

namespace {

__host__ __forceinline__
void cross_edge_host_topk_insert(float* best_dist, int* best_idx, int K, float dist, int idx)
{
    if (dist >= best_dist[K - 1]) return;
    int pos = K - 1;
    while (pos > 0 && dist < best_dist[pos - 1]) {
        best_dist[pos] = best_dist[pos - 1];
        best_idx[pos] = best_idx[pos - 1];
        --pos;
    }
    best_dist[pos] = dist;
    best_idx[pos] = idx;
}

double run_cross_edge_cpu_tiny_groups(
    const std::vector<int>& cpu_tiny_group_indices,
    const std::vector<ANNS::IdxType>& target_group_ids,
    const std::vector<int>& target_group_nx,
    int dim,
    int topk,
    ANNS::IStorage* base_storage,
    const std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>>& group_id_to_range,
    const ANNS::LabelNavGraph& label_nav_graph,
    std::vector<ANNS::SearchQueue>& cross_group_neighbors)
{
    if (cpu_tiny_group_indices.empty()) return 0.0;
    const auto t0 = std::chrono::high_resolution_clock::now();

    for (int gi_i : cpu_tiny_group_indices) {
        const size_t gi = (size_t)gi_i;
        const ANNS::IdxType tgt_gid = target_group_ids[gi];
        const auto& tgt_rng = group_id_to_range[tgt_gid];
        const int nx = target_group_nx[gi];
        if (nx <= 0) continue;
        const int valid_k = std::min<int>(topk, nx);
        if (valid_k <= 0) continue;

        std::vector<const float*> x_ptrs((size_t)nx);
        for (int j = 0; j < nx; ++j) {
            x_ptrs[(size_t)j] =
                reinterpret_cast<const float*>(base_storage->get_vector(tgt_rng.first + (ANNS::IdxType)j));
        }

        for (auto in_gid : label_nav_graph.in_neighbors[tgt_gid]) {
            const auto& qrng = group_id_to_range[in_gid];
            for (ANNS::IdxType qid = qrng.first; qid < qrng.second; ++qid) {
                const float* q = reinterpret_cast<const float*>(base_storage->get_vector(qid));

                constexpr int KMAX = 64;
                const int K = std::min(valid_k, KMAX);
                float best_dist[KMAX];
                int best_idx[KMAX];
                for (int k = 0; k < KMAX; ++k) {
                    best_dist[k] = FLT_MAX;
                    best_idx[k] = -1;
                }

                for (int j = 0; j < nx; ++j) {
                    const float* x = x_ptrs[(size_t)j];
                    float dist = 0.f;
                    for (int d = 0; d < dim; ++d) {
                        const float diff = q[d] - x[d];
                        dist += diff * diff;
                    }
                    cross_edge_host_topk_insert(best_dist, best_idx, K, dist, j);
                }

                for (int r = 0; r < K; ++r) {
                    const int j = best_idx[r];
                    if (j < 0) continue;
                    cross_group_neighbors[qid].insert(tgt_rng.first + (ANNS::IdxType)j, best_dist[r]);
                }
            }
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace

#endif // ANNS_GPU_CROSS_EDGE_CPU_TINY_CUH
