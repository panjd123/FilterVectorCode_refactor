// Naive, small, medium, and descriptor fused group kernels.
//
// Include after gpu_cross_edge_topk_device.cuh and descriptor definitions. These
// kernels are launched by per-group, bucket, double-buffer, and X-streaming helpers;
// route selection and launch policy stay outside this file.

extern "C" __global__ void ung_group_fused_topk_global_kernel(
    const float* __restrict__ Q,          // [nq, dim]
    const float* __restrict__ q_norm,     // [nq]
    const float* __restrict__ X,          // [nx, dim]
    const float* __restrict__ x_norm,     // [nx]
    int nq,
    int nx,
    int dim,
    int topk,
    int* __restrict__ out_idx,            // [nq, topk]
    float* __restrict__ out_dist)         // [nq, topk]
{
    int qi = blockIdx.x;
    if (qi >= nq) return;

    extern __shared__ unsigned char smem_raw[];
    float* s_q = reinterpret_cast<float*>(smem_raw);
    float* s_dist = s_q + dim;
    int* s_idx = reinterpret_cast<int*>(s_dist + (size_t)blockDim.x * (size_t)topk);

    const float* q = Q + (size_t)qi * (size_t)dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        s_q[d] = q[d];
    }
    __syncthreads();

    float best_d[32];
    int best_i[32];
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        best_d[k] = FLT_MAX;
        best_i[k] = -1;
    }

    for (int j = threadIdx.x; j < nx; j += blockDim.x) {
        const float* x = X + (size_t)j * (size_t)dim;
        float dot = 0.f;
        for (int d = 0; d < dim; ++d) {
            dot += s_q[d] * x[d];
        }
        float dist = q_norm[qi] + x_norm[j] - 2.f * dot;

        if (dist < best_d[topk - 1]) {
            int pos = topk - 1;
            while (pos > 0 && dist < best_d[pos - 1]) {
                best_d[pos] = best_d[pos - 1];
                best_i[pos] = best_i[pos - 1];
                --pos;
            }
            best_d[pos] = dist;
            best_i[pos] = j;
        }
    }

    size_t off = (size_t)threadIdx.x * (size_t)topk;
    for (int k = 0; k < topk; ++k) {
        s_dist[off + (size_t)k] = best_d[k];
        s_idx[off + (size_t)k] = best_i[k];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        size_t total = (size_t)blockDim.x * (size_t)topk;
        size_t out_base = (size_t)qi * (size_t)topk;
        for (int r = 0; r < topk; ++r) {
            size_t pick = total;
            float best = FLT_MAX;
            for (size_t t = 0; t < total; ++t) {
                if (s_idx[t] >= 0 && s_dist[t] < best) {
                    best = s_dist[t];
                    pick = t;
                }
            }
            if (pick == total) {
                out_idx[out_base + (size_t)r] = -1;
                out_dist[out_base + (size_t)r] = FLT_MAX;
            } else {
                out_idx[out_base + (size_t)r] = s_idx[pick];
                out_dist[out_base + (size_t)r] = s_dist[pick];
                s_idx[pick] = -1;
            }
        }
    }
}

extern "C" __global__ void ung_small_group_topk_fused_global_kernel(
    const float* __restrict__ Q,          // [nq, dim]
    const float* __restrict__ q_norm,     // [nq]
    const float* __restrict__ X,          // [nx, dim]
    const float* __restrict__ x_norm,     // [nx]
    int nq,
    int nx,
    int dim,
    int topk,
    int max_nx,
    int* __restrict__ out_idx,            // [nq, topk]
    float* __restrict__ out_dist)         // [nq, topk]
{
    int qi = blockIdx.x;
    if (qi >= nq) return;
    if (nx <= 0 || max_nx <= 0) return;

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    extern __shared__ unsigned char smem_raw[];
    float* s_q = reinterpret_cast<float*>(smem_raw);                // [dim]
    float* s_dist = s_q + dim;                                      // [max_nx]
    int* s_idx = reinterpret_cast<int*>(s_dist + max_nx);           // [max_nx]

    const float* q = Q + (size_t)qi * (size_t)dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        s_q[d] = q[d];
    }
    for (int j = threadIdx.x; j < max_nx; j += blockDim.x) {
        s_dist[j] = FLT_MAX;
        s_idx[j] = -1;
    }
    __syncthreads();

    if (wid < nx) {
        const float* x = X + (size_t)wid * (size_t)dim;
        float acc = 0.f;
        for (int d = lane; d < dim; d += 32) {
            acc += s_q[d] * x[d];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            s_dist[wid] = q_norm[qi] + x_norm[wid] - 2.f * acc;
            s_idx[wid] = wid;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        constexpr int KMAX = 32;
        int K = topk;
        if (K > KMAX) K = KMAX;

        float best_d[KMAX];
        int best_i[KMAX];
        #pragma unroll
        for (int k = 0; k < KMAX; ++k) {
            best_d[k] = FLT_MAX;
            best_i[k] = -1;
        }

        for (int j = 0; j < nx; ++j) {
            float dist = s_dist[j];
            int idx = s_idx[j];
            if (idx < 0) continue;
            if (dist < best_d[K - 1]) {
                int pos = K - 1;
                while (pos > 0 && dist < best_d[pos - 1]) {
                    best_d[pos] = best_d[pos - 1];
                    best_i[pos] = best_i[pos - 1];
                    --pos;
                }
                best_d[pos] = dist;
                best_i[pos] = idx;
            }
        }

        size_t base = (size_t)qi * (size_t)topk;
        for (int k = 0; k < K; ++k) {
            out_idx[base + (size_t)k] = best_i[k];
            out_dist[base + (size_t)k] = best_d[k];
        }
        for (int k = K; k < topk; ++k) {
            out_idx[base + (size_t)k] = -1;
            out_dist[base + (size_t)k] = FLT_MAX;
        }
    }
}

extern "C" __global__ void ung_small_group_topk_fused_global_merge_kernel(
    const float* __restrict__ Q,          // [nq, dim]
    const float* __restrict__ q_norm,     // [nq]
    const float* __restrict__ X,          // [nx, dim]
    const float* __restrict__ x_norm,     // [nx]
    const uint32_t* __restrict__ q_ids,   // [total_queries]
    int q_global_offset,
    int x_global_offset,
    int nq,
    int nx,
    int dim,
    int topk,
    int max_nx,
    int* __restrict__ global_idx,         // [num_points, topk]
    float* __restrict__ global_dist,
    int* __restrict__ locks)
{
    int qi = blockIdx.x;
    if (qi >= nq) return;
    if (nx <= 0 || max_nx <= 0) return;

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    extern __shared__ unsigned char smem_raw[];
    float* s_q = reinterpret_cast<float*>(smem_raw);
    float* s_dist = s_q + dim;
    int* s_idx = reinterpret_cast<int*>(s_dist + max_nx);

    const float* q = Q + (size_t)qi * (size_t)dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        s_q[d] = q[d];
    }
    for (int j = threadIdx.x; j < max_nx; j += blockDim.x) {
        s_dist[j] = FLT_MAX;
        s_idx[j] = -1;
    }
    __syncthreads();

    if (wid < nx) {
        const float* x = X + (size_t)wid * (size_t)dim;
        float acc = 0.f;
        for (int d = lane; d < dim; d += 32) {
            acc += s_q[d] * x[d];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            s_dist[wid] = q_norm[qi] + x_norm[wid] - 2.f * acc;
            s_idx[wid] = wid;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        constexpr int KMAX = 32;
        int K = topk;
        if (K > KMAX) K = KMAX;

        float best_d[KMAX];
        int best_i[KMAX];
        #pragma unroll
        for (int k = 0; k < KMAX; ++k) {
            best_d[k] = FLT_MAX;
            best_i[k] = -1;
        }

        for (int j = 0; j < nx; ++j) {
            float dist = s_dist[j];
            int idx = s_idx[j];
            if (idx < 0) continue;
            ung_topk_insert32(best_d, best_i, K, dist, idx);
        }

        const uint32_t qid = q_ids[q_global_offset + qi];
        (void)locks;
        int* out_i = global_idx + (size_t)qid * (size_t)topk;
        float* out_d = global_dist + (size_t)qid * (size_t)topk;
        for (int k = 0; k < K; ++k) {
            if (best_i[k] >= 0) {
                ung_global_topk_insert_locked(out_i, out_d, topk,
                                              x_global_offset + best_i[k], best_d[k]);
            }
        }
    }
}

extern "C" __global__ void ung_medium_group_topk_fused_global_kernel(
    const float* __restrict__ Q,          // [nq, dim]
    const float* __restrict__ q_norm,     // [nq]
    const float* __restrict__ X,          // [nx, dim]
    const float* __restrict__ x_norm,     // [nx]
    int nq,
    int nx,
    int dim,
    int topk,
    int max_nx,
    int* __restrict__ out_idx,            // [nq, topk]
    float* __restrict__ out_dist)         // [nq, topk]
{
    int qi = blockIdx.x;
    if (qi >= nq) return;
    if (nx <= 0 || max_nx <= 0) return;
    if (topk <= 0 || topk > 32) return;

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    if (nwarps <= 0) return;

    extern __shared__ unsigned char smem_raw[];
    float* s_q = reinterpret_cast<float*>(smem_raw);                // [dim]
    float* s_dist = s_q + dim;                                      // [nwarps, topk]
    int* s_idx = reinterpret_cast<int*>(s_dist + (size_t)nwarps * (size_t)topk);

    const float* q = Q + (size_t)qi * (size_t)dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        s_q[d] = q[d];
    }
    __syncthreads();

    constexpr int KMAX = 32;
    float best_d[KMAX];
    int best_i[KMAX];
    #pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        best_d[k] = FLT_MAX;
        best_i[k] = -1;
    }

    const float qn = q_norm[qi];
    for (int j = wid; j < nx; j += nwarps) {
        const float* x = X + (size_t)j * (size_t)dim;
        float acc = 0.f;
        for (int d = lane; d < dim; d += 32) {
            acc += s_q[d] * x[d];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            const float dist = qn + x_norm[j] - 2.f * acc;
            ung_topk_insert32(best_d, best_i, topk, dist, j);
        }
    }

    if (lane == 0) {
        const size_t off = (size_t)wid * (size_t)topk;
        for (int k = 0; k < topk; ++k) {
            s_dist[off + (size_t)k] = best_d[k];
            s_idx[off + (size_t)k] = best_i[k];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float final_d[KMAX];
        int final_i[KMAX];
        #pragma unroll
        for (int k = 0; k < KMAX; ++k) {
            final_d[k] = FLT_MAX;
            final_i[k] = -1;
        }

        for (int w = 0; w < nwarps; ++w) {
            const size_t off = (size_t)w * (size_t)topk;
            for (int k = 0; k < topk; ++k) {
                const int idx = s_idx[off + (size_t)k];
                if (idx >= 0) {
                    ung_topk_insert32(final_d, final_i, topk, s_dist[off + (size_t)k], idx);
                }
            }
        }

        size_t base = (size_t)qi * (size_t)topk;
        for (int k = 0; k < topk; ++k) {
            out_idx[base + (size_t)k] = final_i[k];
            out_dist[base + (size_t)k] = final_d[k];
        }
    }
}

extern "C" __global__ void ung_medium_group_topk_fused_global_merge_kernel(
    const float* __restrict__ Q,          // [nq, dim]
    const float* __restrict__ q_norm,     // [nq]
    const float* __restrict__ X,          // [nx, dim]
    const float* __restrict__ x_norm,     // [nx]
    const uint32_t* __restrict__ q_ids,   // [total_queries]
    int q_global_offset,
    int x_global_offset,
    int nq,
    int nx,
    int dim,
    int topk,
    int max_nx,
    int* __restrict__ global_idx,         // [num_points, topk]
    float* __restrict__ global_dist,
    int* __restrict__ locks)
{
    int qi = blockIdx.x;
    if (qi >= nq) return;
    if (nx <= 0 || max_nx <= 0) return;
    if (topk <= 0 || topk > 32) return;

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    if (nwarps <= 0) return;

    extern __shared__ unsigned char smem_raw[];
    float* s_q = reinterpret_cast<float*>(smem_raw);
    float* s_dist = s_q + dim;
    int* s_idx = reinterpret_cast<int*>(s_dist + (size_t)nwarps * (size_t)topk);

    const float* q = Q + (size_t)qi * (size_t)dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        s_q[d] = q[d];
    }
    __syncthreads();

    constexpr int KMAX = 32;
    float best_d[KMAX];
    int best_i[KMAX];
    #pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        best_d[k] = FLT_MAX;
        best_i[k] = -1;
    }

    const float qn = q_norm[qi];
    for (int j = wid; j < nx; j += nwarps) {
        const float* x = X + (size_t)j * (size_t)dim;
        float acc = 0.f;
        for (int d = lane; d < dim; d += 32) {
            acc += s_q[d] * x[d];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            const float dist = qn + x_norm[j] - 2.f * acc;
            ung_topk_insert32(best_d, best_i, topk, dist, j);
        }
    }

    if (lane == 0) {
        const size_t off = (size_t)wid * (size_t)topk;
        for (int k = 0; k < topk; ++k) {
            s_dist[off + (size_t)k] = best_d[k];
            s_idx[off + (size_t)k] = best_i[k];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float final_d[KMAX];
        int final_i[KMAX];
        #pragma unroll
        for (int k = 0; k < KMAX; ++k) {
            final_d[k] = FLT_MAX;
            final_i[k] = -1;
        }

        for (int w = 0; w < nwarps; ++w) {
            const size_t off = (size_t)w * (size_t)topk;
            for (int k = 0; k < topk; ++k) {
                const int idx = s_idx[off + (size_t)k];
                if (idx >= 0) {
                    ung_topk_insert32(final_d, final_i, topk, s_dist[off + (size_t)k], idx);
                }
            }
        }

        const uint32_t qid = q_ids[q_global_offset + qi];
        (void)locks;
        int* out_i = global_idx + (size_t)qid * (size_t)topk;
        float* out_d = global_dist + (size_t)qid * (size_t)topk;
        for (int k = 0; k < topk; ++k) {
            if (final_i[k] >= 0) {
                ung_global_topk_insert_locked(out_i, out_d, topk,
                                              x_global_offset + final_i[k], final_d[k]);
            }
        }
    }
}

extern "C" __global__ void ung_group_desc_topk_fused_global_kernel(
    const UngGroupQueryDesc* __restrict__ descs, // [num_desc]
    int num_desc,
    const float* __restrict__ Q,                 // [total_queries, dim]
    const float* __restrict__ q_norm,            // [total_queries]
    const float* __restrict__ all_x,             // [total_points, dim]
    const float* __restrict__ all_norm,          // [total_points]
    const uint32_t* __restrict__ query_ids,       // [total_queries], optional global q ids
    int dim,
    int topk,
    int max_nx,
    int direct_qid,
    int* __restrict__ out_idx,                   // [total_queries, topk]
    float* __restrict__ out_dist)                // [total_queries, topk]
{
    int did = blockIdx.x;
    if (did >= num_desc) return;
    if (max_nx <= 0) return;

    UngGroupQueryDesc desc = descs[did];
    const int qi = (int)desc.qi;
    const int x_off = (int)desc.x_off;
    const int nx = (int)desc.nx;
    if (nx <= 0 || nx > max_nx) return;

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;

    extern __shared__ unsigned char smem_raw[];
    float* s_q = reinterpret_cast<float*>(smem_raw);                // [dim]
    float* s_dist = s_q + dim;                                      // [max_nx]
    int* s_idx = reinterpret_cast<int*>(s_dist + max_nx);           // [max_nx]

    const int qid = direct_qid ? (int)query_ids[qi] : qi;
    const float* q = direct_qid ? (all_x + (size_t)qid * (size_t)dim)
                                : (Q + (size_t)qi * (size_t)dim);
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        s_q[d] = q[d];
    }
    for (int j = threadIdx.x; j < max_nx; j += blockDim.x) {
        s_dist[j] = FLT_MAX;
        s_idx[j] = -1;
    }
    __syncthreads();

    const float* x_base = all_x + (size_t)x_off * (size_t)dim;
    const float* x_norm = all_norm + (size_t)x_off;
    for (int j = wid; j < nx; j += nwarps) {
        const float* x = x_base + (size_t)j * (size_t)dim;
        float acc = 0.f;
        for (int d = lane; d < dim; d += 32) {
            acc += s_q[d] * x[d];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            const float qn = direct_qid ? all_norm[qid] : q_norm[qi];
            s_dist[j] = qn + x_norm[j] - 2.f * acc;
            s_idx[j] = j;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        constexpr int KMAX = 32;
        int K = topk;
        if (K > KMAX) K = KMAX;

        float best_d[KMAX];
        int best_i[KMAX];
        #pragma unroll
        for (int k = 0; k < KMAX; ++k) {
            best_d[k] = FLT_MAX;
            best_i[k] = -1;
        }

        for (int j = 0; j < nx; ++j) {
            float dist = s_dist[j];
            int idx = s_idx[j];
            if (idx < 0) continue;
            if (dist < best_d[K - 1]) {
                int pos = K - 1;
                while (pos > 0 && dist < best_d[pos - 1]) {
                    best_d[pos] = best_d[pos - 1];
                    best_i[pos] = best_i[pos - 1];
                    --pos;
                }
                best_d[pos] = dist;
                best_i[pos] = idx;
            }
        }

        size_t base = (size_t)qi * (size_t)topk;
        for (int k = 0; k < K; ++k) {
            out_idx[base + (size_t)k] = best_i[k];
            out_dist[base + (size_t)k] = best_d[k];
        }
        for (int k = K; k < topk; ++k) {
            out_idx[base + (size_t)k] = -1;
            out_dist[base + (size_t)k] = FLT_MAX;
        }
    }
}
