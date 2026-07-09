// Large target-group fused kernels used by the immediate per-group route.
//
// Include after gpu_cross_edge_topk_device.cuh, because these kernels call
// ung_topk_insert32(), ung_topk_insert16(), and ung_global_topk_insert_locked().

extern "C" __global__ void ung_large_group_warp_fused_topk_global_kernel(
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
    const int qi = blockIdx.x;
    if (qi >= nq || topk <= 0 || topk > 32) return;

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    if (nwarps <= 0) return;

    extern __shared__ unsigned char lg_smem_raw[];
    float* s_dist = reinterpret_cast<float*>(lg_smem_raw);       // [nwarps, topk]
    int* s_idx = reinterpret_cast<int*>(s_dist + (size_t)nwarps * (size_t)topk);

    constexpr int KMAX = 32;
    float best_d[KMAX];
    int best_i[KMAX];
    #pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        best_d[k] = FLT_MAX;
        best_i[k] = -1;
    }

    const float* q = Q + (size_t)qi * (size_t)dim;
    const float qn = q_norm[qi];

    // Each warp owns a strided subset of X. Dot products are reduced inside
    // the warp; only lane 0 maintains that warp's register-resident topK.
    for (int j = wid; j < nx; j += nwarps) {
        const float* x = X + (size_t)j * (size_t)dim;
        float acc = 0.f;

        if ((dim & 3) == 0) {
            const int dim4 = dim >> 2;
            const float4* q4 = reinterpret_cast<const float4*>(q);
            const float4* x4 = reinterpret_cast<const float4*>(x);
            for (int d4 = lane; d4 < dim4; d4 += 32) {
                const float4 qv = q4[d4];
                const float4 xv = x4[d4];
                acc += qv.x * xv.x + qv.y * xv.y + qv.z * xv.z + qv.w * xv.w;
            }
        } else {
            for (int d = lane; d < dim; d += 32) {
                acc += q[d] * x[d];
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }

        if (lane == 0) {
            float dist = qn + x_norm[j] - 2.f * acc;
            ung_topk_insert32(best_d, best_i, topk, dist, j);
        }
    }

    if (lane == 0) {
        size_t off = (size_t)wid * (size_t)topk;
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
            size_t off = (size_t)w * (size_t)topk;
            for (int k = 0; k < topk; ++k) {
                int idx = s_idx[off + (size_t)k];
                if (idx >= 0) {
                    ung_topk_insert32(final_d, final_i, topk, s_dist[off + (size_t)k], idx);
                }
            }
        }

        size_t out_base = (size_t)qi * (size_t)topk;
        for (int k = 0; k < topk; ++k) {
            out_idx[out_base + (size_t)k] = final_i[k];
            if (out_dist) out_dist[out_base + (size_t)k] = final_d[k];
        }
    }
}

extern "C" __global__ void ung_large_group_warp_fused_topk_global_merge_kernel(
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
    int* __restrict__ global_idx,         // [num_points, topk]
    float* __restrict__ global_dist,
    int* __restrict__ locks)
{
    const int qi = blockIdx.x;
    if (qi >= nq || topk <= 0 || topk > 32) return;

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    if (nwarps <= 0) return;

    extern __shared__ unsigned char lg_smem_raw[];
    float* s_dist = reinterpret_cast<float*>(lg_smem_raw);
    int* s_idx = reinterpret_cast<int*>(s_dist + (size_t)nwarps * (size_t)topk);

    constexpr int KMAX = 32;
    float best_d[KMAX];
    int best_i[KMAX];
    #pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        best_d[k] = FLT_MAX;
        best_i[k] = -1;
    }

    const float* q = Q + (size_t)qi * (size_t)dim;
    const float qn = q_norm[qi];

    for (int j = wid; j < nx; j += nwarps) {
        const float* x = X + (size_t)j * (size_t)dim;
        float acc = 0.f;

        if ((dim & 3) == 0) {
            const int dim4 = dim >> 2;
            const float4* q4 = reinterpret_cast<const float4*>(q);
            const float4* x4 = reinterpret_cast<const float4*>(x);
            for (int d4 = lane; d4 < dim4; d4 += 32) {
                const float4 qv = q4[d4];
                const float4 xv = x4[d4];
                acc += qv.x * xv.x + qv.y * xv.y + qv.z * xv.z + qv.w * xv.w;
            }
        } else {
            for (int d = lane; d < dim; d += 32) {
                acc += q[d] * x[d];
            }
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
        int* lock = locks + qid;
        while (atomicCAS(lock, 0, 1) != 0) { }
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

extern "C" __global__ void ung_large_group_warp_query_topk_global_kernel(
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
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int qi = blockIdx.x * nwarps + wid;
    if (qi >= nq || topk <= 0 || topk > 32) return;

    constexpr int KMAX = 32;
    float best_d[KMAX];
    int best_i[KMAX];
    #pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        best_d[k] = FLT_MAX;
        best_i[k] = -1;
    }

    const float* q = Q + (size_t)qi * (size_t)dim;
    const float qn = q_norm[qi];

    for (int j = 0; j < nx; ++j) {
        const float* x = X + (size_t)j * (size_t)dim;
        float acc = 0.f;

        if ((dim & 3) == 0) {
            const int dim4 = dim >> 2;
            const float4* q4 = reinterpret_cast<const float4*>(q);
            const float4* x4 = reinterpret_cast<const float4*>(x);
            for (int d4 = lane; d4 < dim4; d4 += 32) {
                const float4 qv = q4[d4];
                const float4 xv = x4[d4];
                acc += qv.x * xv.x + qv.y * xv.y + qv.z * xv.z + qv.w * xv.w;
            }
        } else {
            for (int d = lane; d < dim; d += 32) {
                acc += q[d] * x[d];
            }
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
        const size_t out_base = (size_t)qi * (size_t)topk;
        for (int k = 0; k < topk; ++k) {
            out_idx[out_base + (size_t)k] = best_i[k];
            if (out_dist) out_dist[out_base + (size_t)k] = best_d[k];
        }
    }
}

extern "C" __global__ void ung_large_group_warp_query_topk_global_merge_kernel(
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
    int* __restrict__ global_idx,         // [num_points, topk]
    float* __restrict__ global_dist,
    int* __restrict__ locks)
{
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int qi = blockIdx.x * nwarps + wid;
    if (qi >= nq || topk <= 0 || topk > 32) return;

    constexpr int KMAX = 32;
    float best_d[KMAX];
    int best_i[KMAX];
    #pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        best_d[k] = FLT_MAX;
        best_i[k] = -1;
    }

    const float* q = Q + (size_t)qi * (size_t)dim;
    const float qn = q_norm[qi];

    for (int j = 0; j < nx; ++j) {
        const float* x = X + (size_t)j * (size_t)dim;
        float acc = 0.f;

        if ((dim & 3) == 0) {
            const int dim4 = dim >> 2;
            const float4* q4 = reinterpret_cast<const float4*>(q);
            const float4* x4 = reinterpret_cast<const float4*>(x);
            for (int d4 = lane; d4 < dim4; d4 += 32) {
                const float4 qv = q4[d4];
                const float4 xv = x4[d4];
                acc += qv.x * xv.x + qv.y * xv.y + qv.z * xv.z + qv.w * xv.w;
            }
        } else {
            for (int d = lane; d < dim; d += 32) {
                acc += q[d] * x[d];
            }
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
        const uint32_t qid = q_ids[q_global_offset + qi];
        (void)locks;
        int* out_i = global_idx + (size_t)qid * (size_t)topk;
        float* out_d = global_dist + (size_t)qid * (size_t)topk;
        for (int k = 0; k < topk; ++k) {
            if (best_i[k] >= 0) {
                ung_global_topk_insert_locked(out_i, out_d, topk,
                                              x_global_offset + best_i[k], best_d[k]);
            }
        }
    }
}

extern "C" __global__ void ung_large_group_tf32_wmma_topk_global_kernel(
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
    using namespace nvcuda;
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 8;
    constexpr int KT = 1;
    constexpr int KMAX = 16;

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    if (topk <= 0 || topk > KMAX || nwarps <= 0) return;

    const int q_base = (blockIdx.x * nwarps + wid) * M;
    const bool warp_active = (q_base < nq);

    extern __shared__ float smem[];
    float* sB = smem;
    const size_t per_warp = (size_t)KT * M * K + (size_t)M * N;
    float* sA = smem + (size_t)KT * K * N + (size_t)wid * per_warp;
    float* sC = sA + (size_t)KT * M * K;

    float best_d[KMAX];
    int best_i[KMAX];
    #pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        best_d[k] = FLT_MAX;
        best_i[k] = -1;
    }

    const int row_lane = lane; // lanes 0..15 own topK for rows 0..15.
    const bool owns_row = (warp_active && row_lane < M && q_base + row_lane < nq);
    const float qn = owns_row ? q_norm[q_base + row_lane] : 0.f;

    for (int x_base = 0; x_base < nx; x_base += N) {
        wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int k0 = 0; k0 < dim; k0 += KT * K) {
            for (int t = lane; t < KT * M * K; t += 32) {
                const int kt = t / (M * K);
                const int tr = t - kt * M * K;
                const int r = tr / K;
                const int kk = tr - r * K;
                const int qid = q_base + r;
                const int d = k0 + kt * K + kk;
                float v = 0.f;
                if (warp_active && qid < nq && d < dim) {
                    v = Q[(size_t)qid * (size_t)dim + (size_t)d];
                }
                sA[t] = wmma::__float_to_tf32(v);
            }

            // B is KxN in column-major layout: B[kk + col*K] = X[x_base+col, k0+kk].
            for (int t = threadIdx.x; t < KT * K * N; t += blockDim.x) {
                const int kt = t / (K * N);
                const int tb = t - kt * K * N;
                const int col = tb / K;
                const int kk = tb - col * K;
                const int xid = x_base + col;
                const int d = k0 + kt * K + kk;
                float v = 0.f;
                if (xid < nx && d < dim) {
                    v = X[(size_t)xid * (size_t)dim + (size_t)d];
                }
                sB[t] = wmma::__float_to_tf32(v);
            }
            __syncthreads();

            #pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                wmma::load_matrix_sync(a_frag, sA + (size_t)kt * M * K, K);
                wmma::load_matrix_sync(b_frag, sB + (size_t)kt * K * N, K);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            __syncthreads();
        }

        wmma::store_matrix_sync(sC, c_frag, N, wmma::mem_row_major);
        __syncwarp();

        if (owns_row) {
            const int valid_cols = min(N, nx - x_base);
            const float* row = sC + (size_t)row_lane * N;
            for (int col = 0; col < valid_cols; ++col) {
                const int xid = x_base + col;
                const float dist = qn + x_norm[xid] - 2.f * row[col];
                ung_topk_insert16(best_d, best_i, topk, dist, xid);
            }
        }
        __syncwarp();
    }

    if (owns_row) {
        const size_t out_base = (size_t)(q_base + row_lane) * (size_t)topk;
        for (int k = 0; k < topk; ++k) {
            out_idx[out_base + (size_t)k] = best_i[k];
            out_dist[out_base + (size_t)k] = best_d[k];
        }
    }
}

extern "C" __global__ void ung_large_group_tf32_wmma_topk_global_merge_kernel(
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
    int* __restrict__ global_idx,         // [num_points, topk]
    float* __restrict__ global_dist,
    int* __restrict__ locks)
{
    using namespace nvcuda;
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 8;
    constexpr int KT = 4;
    constexpr int KMAX = 16;

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    if (topk <= 0 || topk > KMAX || nwarps <= 0) return;

    const int q_base = (blockIdx.x * nwarps + wid) * M;
    const bool warp_active = (q_base < nq);

    extern __shared__ float smem[];
    float* sB = smem;
    const size_t per_warp = (size_t)KT * M * K + (size_t)M * N;
    float* sA = smem + (size_t)KT * K * N + (size_t)wid * per_warp;
    float* sC = sA + (size_t)KT * M * K;

    float best_d[KMAX];
    int best_i[KMAX];
    #pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        best_d[k] = FLT_MAX;
        best_i[k] = -1;
    }

    const int row_lane = lane;
    const bool owns_row = (warp_active && row_lane < M && q_base + row_lane < nq);
    const float qn = owns_row ? q_norm[q_base + row_lane] : 0.f;

    for (int x_base = 0; x_base < nx; x_base += N) {
        wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int k0 = 0; k0 < dim; k0 += KT * K) {
            for (int t = lane; t < KT * M * K; t += 32) {
                const int kt = t / (M * K);
                const int tr = t - kt * M * K;
                const int r = tr / K;
                const int kk = tr - r * K;
                const int qid = q_base + r;
                const int d = k0 + kt * K + kk;
                float v = 0.f;
                if (warp_active && qid < nq && d < dim) {
                    v = Q[(size_t)qid * (size_t)dim + (size_t)d];
                }
                sA[t] = wmma::__float_to_tf32(v);
            }
            for (int t = threadIdx.x; t < KT * K * N; t += blockDim.x) {
                const int kt = t / (K * N);
                const int tb = t - kt * K * N;
                const int col = tb / K;
                const int kk = tb - col * K;
                const int xid = x_base + col;
                const int d = k0 + kt * K + kk;
                float v = 0.f;
                if (xid < nx && d < dim) {
                    v = X[(size_t)xid * (size_t)dim + (size_t)d];
                }
                sB[t] = wmma::__float_to_tf32(v);
            }
            __syncthreads();

            #pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                wmma::load_matrix_sync(a_frag, sA + (size_t)kt * M * K, K);
                wmma::load_matrix_sync(b_frag, sB + (size_t)kt * K * N, K);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            __syncthreads();
        }

        wmma::store_matrix_sync(sC, c_frag, N, wmma::mem_row_major);
        __syncwarp();

        if (owns_row) {
            const int valid_cols = min(N, nx - x_base);
            const float* row = sC + (size_t)row_lane * N;
            for (int col = 0; col < valid_cols; ++col) {
                const int xid = x_base + col;
                const float dist = qn + x_norm[xid] - 2.f * row[col];
                ung_topk_insert16(best_d, best_i, topk, dist, xid);
            }
        }
        __syncwarp();
    }

    if (owns_row) {
        const uint32_t qid = q_ids[q_global_offset + q_base + row_lane];
        (void)locks;
        int* out_i = global_idx + (size_t)qid * (size_t)topk;
        float* out_d = global_dist + (size_t)qid * (size_t)topk;
        for (int k = 0; k < topk; ++k) {
            if (best_i[k] >= 0) {
                ung_global_topk_insert_locked(out_i, out_d, topk,
                                              x_global_offset + best_i[k], best_d[k]);
            }
        }
    }
}
