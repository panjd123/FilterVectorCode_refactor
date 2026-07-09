// Device-side topK insertion primitives shared by cross-edge kernels.
// Keep this include before any kernel family that calls ung_topk_insert* or
// ung_global_topk_insert_locked.

__device__ __forceinline__
void ung_topk_insert32(float* best_dist, int* best_idx, int K, float dist, int idx)
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

__device__ __forceinline__
void ung_topk_insert16(float* best_dist, int* best_idx, int K, float dist, int idx)
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

__device__ __forceinline__
void ung_global_topk_insert_locked(int* out_i, float* out_d, int topk, int gid, float dist)
{
    bool exists = false;
    for (int k = 0; k < topk; ++k) {
        if (out_i[k] == gid) {
            exists = true;
            break;
        }
    }
    if (exists) return;

    const int last = topk - 1;
    const bool better_than_tail =
        (dist < out_d[last]) || (dist == out_d[last] && gid < out_i[last]);
    if (!better_than_tail) return;

    int pos = last;
    while (pos > 0 &&
           (dist < out_d[pos - 1] || (dist == out_d[pos - 1] && gid < out_i[pos - 1]))) {
        out_d[pos] = out_d[pos - 1];
        out_i[pos] = out_i[pos - 1];
        --pos;
    }
    out_d[pos] = dist;
    out_i[pos] = gid;
}
