#ifndef ANNS_GPU_CROSS_EDGE_SOURCE_EXACT_KERNELS_CUH
#define ANNS_GPU_CROSS_EDGE_SOURCE_EXACT_KERNELS_CUH

// Source-centric exact experimental kernels for cross-edge search.
//
// These kernels belong to the negative/experimental source-exact route. They
// stay separate from the default target-centric fused path so the main CUDA
// file does not present them as part of the paper path.

extern "C" __global__ void ung_source_exact_topk_global_kernel(
    const UngSourceQueryDesc* __restrict__ queries,
    int num_queries,
    const UngTargetSegmentDesc* __restrict__ segments,
    const float* __restrict__ all_x,
    const float* __restrict__ all_norm,
    int dim,
    int topk,
    int* __restrict__ out_idx,
    float* __restrict__ out_dist)
{
    const int qi = blockIdx.x;
    if (qi >= num_queries || topk <= 0 || topk > 32) return;

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    if (nwarps <= 0) return;

    const UngSourceQueryDesc qdesc = queries[qi];
    const uint32_t qid = qdesc.qid;
    const float* q = all_x + (size_t)qid * (size_t)dim;
    const float qn = all_norm[qid];

    constexpr int KMAX = 32;
    float best_d[KMAX];
    int best_i[KMAX];
#pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        best_d[k] = FLT_MAX;
        best_i[k] = -1;
    }

    for (uint32_t e = 0; e < qdesc.edge_count; ++e) {
        const UngTargetSegmentDesc seg = segments[qdesc.edge_start + e];
        const float* x_base = all_x + (size_t)seg.x_off * (size_t)dim;
        const float* xn_base = all_norm + (size_t)seg.x_off;

        for (uint32_t j = (uint32_t)wid; j < seg.nx; j += (uint32_t)nwarps) {
            const float* x = x_base + (size_t)j * (size_t)dim;
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
                const int gid = (int)(seg.x_off + j);
                const float dist = qn + xn_base[j] - 2.f * acc;
                ung_topk_insert32(best_d, best_i, topk, dist, gid);
            }
        }
    }

    extern __shared__ unsigned char smem_raw[];
    float* s_dist = reinterpret_cast<float*>(smem_raw);
    int* s_idx = reinterpret_cast<int*>(s_dist + (size_t)nwarps * (size_t)topk);

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

        const size_t out_base = (size_t)qi * (size_t)topk;
        for (int k = 0; k < topk; ++k) {
            out_idx[out_base + (size_t)k] = final_i[k];
            if (out_dist) out_dist[out_base + (size_t)k] = final_d[k];
        }
    }
}

extern "C" __global__ void ung_source_tf32_wmma_topk_global_kernel(
    const UngSourceTileDesc* __restrict__ tiles,
    int num_tiles,
    const UngTargetSegmentDesc* __restrict__ segments,
    const float* __restrict__ all_x,
    const float* __restrict__ all_norm,
    int dim,
    int topk,
    int* __restrict__ out_idx,
    float* __restrict__ out_dist)
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
    const int base_tid = (int)blockIdx.x * nwarps;
    const int tid = base_tid + wid;
    if (base_tid >= num_tiles || topk <= 0 || topk > KMAX || nwarps <= 0) return;

    UngSourceTileDesc td;
    td.q_start = 0;
    td.q_count = 0;
    td.out_start = 0;
    td.edge_start = 0;
    td.edge_count = 0;
    if (tid < num_tiles) {
        td = tiles[tid];
    }
    const bool warp_active = (tid < num_tiles && td.q_count != 0 && td.edge_count != 0);

    extern __shared__ float smem[];
    float* const sB_shared = smem;
    float* const warp_smem = smem + (size_t)KT * K * N;
    const size_t per_warp = (size_t)KT * M * K + (size_t)M * N;
    float* sA = warp_smem + (size_t)wid * per_warp;
    float* sC = sA + (size_t)KT * M * K;

    const int row_lane = lane;
    const bool owns_row = (warp_active && row_lane < M && (uint32_t)row_lane < td.q_count);
    const uint32_t qid = td.q_start + (uint32_t)row_lane;
    const float qn = owns_row ? all_norm[qid] : 0.f;

    float best_d[KMAX];
    int best_i[KMAX];
#pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        best_d[k] = FLT_MAX;
        best_i[k] = -1;
    }

    for (uint32_t e = 0; e < td.edge_count; ++e) {
        const UngTargetSegmentDesc seg = segments[td.edge_start + e];
        const float* x_base_ptr = all_x + (size_t)seg.x_off * (size_t)dim;
        const float* x_norm_ptr = all_norm + (size_t)seg.x_off;

        for (uint32_t x_base = 0; x_base < seg.nx; x_base += N) {
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
                    const uint32_t row_qid = td.q_start + (uint32_t)r;
                    const int d = k0 + kt * K + kk;
                    float v = 0.f;
                    if (warp_active && (uint32_t)r < td.q_count && d < dim) {
                        v = all_x[(size_t)row_qid * (size_t)dim + (size_t)d];
                    }
                    sA[t] = wmma::__float_to_tf32(v);
                }

                for (int t = threadIdx.x; t < KT * K * N; t += blockDim.x) {
                    const int kt = t / (K * N);
                    const int tb = t - kt * K * N;
                    const int col = tb / K;
                    const int kk = tb - col * K;
                    const uint32_t xlocal = x_base + (uint32_t)col;
                    const int d = k0 + kt * K + kk;
                    float v = 0.f;
                    if (xlocal < seg.nx && d < dim) {
                        v = x_base_ptr[(size_t)xlocal * (size_t)dim + (size_t)d];
                    }
                    sB_shared[t] = wmma::__float_to_tf32(v);
                }
                __syncthreads();

#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
                    wmma::load_matrix_sync(a_frag, sA + (size_t)kt * M * K, K);
                    wmma::load_matrix_sync(b_frag, sB_shared + (size_t)kt * K * N, K);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                __syncthreads();
            }

            wmma::store_matrix_sync(sC, c_frag, N, wmma::mem_row_major);
            __syncwarp();

            if (owns_row) {
                const int valid_cols = min((uint32_t)N, seg.nx - x_base);
                const float* row = sC + (size_t)row_lane * N;
                for (int col = 0; col < valid_cols; ++col) {
                    const uint32_t xlocal = x_base + (uint32_t)col;
                    const int gid = (int)(seg.x_off + xlocal);
                    const float dist = qn + x_norm_ptr[xlocal] - 2.f * row[col];
                    ung_topk_insert16(best_d, best_i, topk, dist, gid);
                }
            }
            __syncwarp();
        }
    }

    if (owns_row) {
        const uint32_t out_row = td.out_start + (uint32_t)row_lane;
        const size_t out_base = (size_t)out_row * (size_t)topk;
        for (int k = 0; k < topk; ++k) {
            out_idx[out_base + (size_t)k] = best_i[k];
            if (out_dist) out_dist[out_base + (size_t)k] = best_d[k];
        }
    }
}

#endif // ANNS_GPU_CROSS_EDGE_SOURCE_EXACT_KERNELS_CUH
