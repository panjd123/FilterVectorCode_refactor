// Descriptor-batched TF32 group kernels used by bucket, double-buffer, and X-streaming routes.
//
// Include after gpu_cross_edge_topk_device.cuh and descriptor definitions, because these
// kernels use UngGroupTileDesc and ung_topk_insert16().

extern "C" __global__ void ung_group_tile_tf32_wmma_topk_global_kernel(
    const UngGroupTileDesc* __restrict__ descs,
    int num_desc,
    const float* __restrict__ Q,          // [total_queries, dim]
    const float* __restrict__ q_norm,     // [total_queries]
    const float* __restrict__ all_x,      // [total_points, dim]
    const float* __restrict__ all_norm,   // [total_points]
    const uint32_t* __restrict__ query_ids,// [total_queries], optional global q ids
    int dim,
    int topk,
    int direct_qid,
    int* __restrict__ out_idx,            // [total_queries, topk], local index within target group
    float* __restrict__ out_dist)         // [total_queries, topk]
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
    const int base_did = blockIdx.x * nwarps;
    const int did = base_did + wid;
    if (base_did >= num_desc || topk <= 0 || topk > KMAX || nwarps <= 0) return;

    const UngGroupTileDesc desc0 = descs[base_did];
    UngGroupTileDesc desc;
    desc.q_start = 0;
    desc.q_count = 0;
    desc.x_off = desc0.x_off;
    desc.nx = desc0.nx;
    if (did < num_desc) {
        desc = descs[did];
    }

    bool block_shared_x = (desc0.q_count != 0 && desc0.nx != 0);
    for (int w = 0; w < nwarps; ++w) {
        const int other_did = base_did + w;
        if (other_did >= num_desc) continue;
        const UngGroupTileDesc other = descs[other_did];
        if (other.q_count == 0 || other.nx == 0) continue;
        if (other.x_off != desc0.x_off || other.nx != desc0.nx) {
            block_shared_x = false;
        }
    }
    const bool warp_active = (did < num_desc && desc.q_count != 0 && desc.nx != 0);

    extern __shared__ float smem[];
    float* const sB_shared = smem;
    float* const warp_smem = smem + (size_t)KT * K * N;
    const size_t per_warp_shared = (size_t)KT * M * K + (size_t)M * N;
    const size_t per_warp_private = (size_t)KT * M * K + (size_t)KT * K * N + (size_t)M * N;
    float* sA = warp_smem + (size_t)wid * (block_shared_x ? per_warp_shared : per_warp_private);
    float* sB = block_shared_x ? sB_shared : (sA + (size_t)KT * M * K);
    float* sC = block_shared_x ? (sA + (size_t)KT * M * K) : (sB + (size_t)KT * K * N);

    float best_d[KMAX];
    int best_i[KMAX];
    #pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        best_d[k] = FLT_MAX;
        best_i[k] = -1;
    }

    const int row_lane = lane;
    const bool owns_row = (warp_active && row_lane < M && (uint32_t)row_lane < desc.q_count);
    const uint32_t qi = desc.q_start + (uint32_t)row_lane;
    const uint32_t qid = (owns_row && direct_qid) ? query_ids[qi] : qi;
    const float qn = owns_row ? (direct_qid ? all_norm[qid] : q_norm[qi]) : 0.f;
    const uint32_t block_x_off = block_shared_x ? desc0.x_off : desc.x_off;
    const uint32_t block_nx = block_shared_x ? desc0.nx : desc.nx;
    const float* x_base_ptr = all_x + (size_t)block_x_off * (size_t)dim;
    const float* x_norm_ptr = all_norm + (size_t)block_x_off;

    for (uint32_t x_base = 0; x_base < block_nx; x_base += N) {
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
                const uint32_t qrow = desc.q_start + (uint32_t)r;
                const int d = k0 + kt * K + kk;
                float v = 0.f;
                if (warp_active && (uint32_t)r < desc.q_count && d < dim) {
                    const uint32_t src = direct_qid ? query_ids[qrow] : qrow;
                    const float* q_base_ptr = direct_qid ? all_x : Q;
                    v = q_base_ptr[(size_t)src * (size_t)dim + (size_t)d];
                }
                sA[t] = wmma::__float_to_tf32(v);
            }
            const int b_start = block_shared_x ? threadIdx.x : lane;
            const int b_step = block_shared_x ? blockDim.x : 32;
            for (int t = b_start; t < KT * K * N; t += b_step) {
                const int kt = t / (K * N);
                const int tb = t - kt * K * N;
                const int col = tb / K;
                const int kk = tb - col * K;
                const uint32_t xlocal = x_base + (uint32_t)col;
                const int d = k0 + kt * K + kk;
                float v = 0.f;
                if (xlocal < block_nx && d < dim) {
                    v = x_base_ptr[(size_t)xlocal * (size_t)dim + (size_t)d];
                }
                sB[t] = wmma::__float_to_tf32(v);
            }
            if (block_shared_x) {
                __syncthreads();
            } else {
                __syncwarp();
            }

            #pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                wmma::load_matrix_sync(a_frag, sA + (size_t)kt * M * K, K);
                wmma::load_matrix_sync(b_frag, sB + (size_t)kt * K * N, K);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            if (block_shared_x) {
                __syncthreads();
            } else {
                __syncwarp();
            }
        }

        wmma::store_matrix_sync(sC, c_frag, N, wmma::mem_row_major);
        __syncwarp();

        if (owns_row) {
            const int valid_cols = min((uint32_t)N, block_nx - x_base);
            const float* row = sC + (size_t)row_lane * N;
            for (int col = 0; col < valid_cols; ++col) {
                const uint32_t xlocal = x_base + (uint32_t)col;
                const float dist = qn + x_norm_ptr[xlocal] - 2.f * row[col];
                ung_topk_insert16(best_d, best_i, topk, dist, (int)xlocal);
            }
        }
        __syncwarp();
    }

    if (owns_row) {
        const size_t out_base = (size_t)qi * (size_t)topk;
        for (int k = 0; k < topk; ++k) {
            out_idx[out_base + (size_t)k] = best_i[k];
            out_dist[out_base + (size_t)k] = best_d[k];
        }
    }
}

extern "C" __global__ void ung_group_prefix_tf32_wmma_topk_global_kernel(
    const UngGroupTileDesc* __restrict__ groups,
    const uint32_t* __restrict__ tile_offsets,
    int num_groups,
    uint32_t total_tiles,
    const float* __restrict__ Q,          // [total_queries, dim]
    const float* __restrict__ q_norm,     // [total_queries]
    const float* __restrict__ all_x,      // [total_points, dim]
    const float* __restrict__ all_norm,   // [total_points]
    int dim,
    int topk,
    int* __restrict__ out_idx,            // [total_queries, topk], local index within target group
    float* __restrict__ out_dist)         // [total_queries, topk]
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
    const uint32_t tile_id = (uint32_t)blockIdx.x * (uint32_t)nwarps + (uint32_t)wid;
    if (tile_id >= total_tiles || topk <= 0 || topk > KMAX || nwarps <= 0) return;

    int lo = 0;
    int hi = num_groups;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (tile_offsets[mid] <= tile_id) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    const UngGroupTileDesc g = groups[lo];
    const uint32_t local_tile = tile_id - tile_offsets[lo];
    const uint32_t q_start = g.q_start + local_tile * (uint32_t)M;
    const uint32_t q_count = (q_start < g.q_start + g.q_count)
        ? min((uint32_t)M, g.q_start + g.q_count - q_start)
        : 0u;
    const bool warp_active = (q_count != 0 && g.nx != 0);

    extern __shared__ float smem[];
    float* const sB_shared = smem;
    float* const warp_smem = smem + (size_t)KT * K * N;
    const size_t per_warp = (size_t)KT * M * K + (size_t)M * N;
    float* sA = warp_smem + (size_t)wid * per_warp;
    float* sC = sA + (size_t)KT * M * K;

    float best_d[KMAX];
    int best_i[KMAX];
    #pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        best_d[k] = FLT_MAX;
        best_i[k] = -1;
    }

    const int row_lane = lane;
    const bool owns_row = (warp_active && row_lane < M && (uint32_t)row_lane < q_count);
    const uint32_t qi = q_start + (uint32_t)row_lane;
    const float qn = owns_row ? q_norm[qi] : 0.f;
    const float* x_base_ptr = all_x + (size_t)g.x_off * (size_t)dim;
    const float* x_norm_ptr = all_norm + (size_t)g.x_off;

    for (uint32_t x_base = 0; x_base < g.nx; x_base += N) {
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
                const uint32_t qrow = q_start + (uint32_t)r;
                const int d = k0 + kt * K + kk;
                float v = 0.f;
                if (warp_active && (uint32_t)r < q_count && d < dim) {
                    v = Q[(size_t)qrow * (size_t)dim + (size_t)d];
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
                if (xlocal < g.nx && d < dim) {
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
            const int valid_cols = min((uint32_t)N, g.nx - x_base);
            const float* row = sC + (size_t)row_lane * N;
            for (int col = 0; col < valid_cols; ++col) {
                const uint32_t xlocal = x_base + (uint32_t)col;
                const float dist = qn + x_norm_ptr[xlocal] - 2.f * row[col];
                ung_topk_insert16(best_d, best_i, topk, dist, (int)xlocal);
            }
        }
        __syncwarp();
    }

    if (owns_row) {
        const size_t out_base = (size_t)qi * (size_t)topk;
        for (int k = 0; k < topk; ++k) {
            out_idx[out_base + (size_t)k] = best_i[k];
            out_dist[out_base + (size_t)k] = best_d[k];
        }
    }
}

extern "C" __global__ void ung_group_2d_tf32_wmma_topk_global_kernel(
    const UngGroupTileDesc* __restrict__ groups,
    int num_groups,
    const float* __restrict__ Q,          // [total_queries, dim]
    const float* __restrict__ q_norm,     // [total_queries]
    const float* __restrict__ all_x,      // [total_points, dim]
    const float* __restrict__ all_norm,   // [total_points]
    const uint32_t* __restrict__ query_ids,// [total_queries], optional global q ids
    int dim,
    int topk,
    int direct_qid,
    int* __restrict__ out_idx,            // [total_queries, topk], local index within target group
    float* __restrict__ out_dist)         // [total_queries, topk]
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
    const int gid = (int)blockIdx.y;
    if (gid >= num_groups || topk <= 0 || topk > KMAX || nwarps <= 0) return;

    const UngGroupTileDesc g = groups[gid];
    const uint32_t local_tile = (uint32_t)blockIdx.x * (uint32_t)nwarps + (uint32_t)wid;
    const uint32_t q_start = g.q_start + local_tile * (uint32_t)M;
    const uint32_t q_end = g.q_start + g.q_count;
    const uint32_t q_count = (q_start < q_end) ? min((uint32_t)M, q_end - q_start) : 0u;
    const bool warp_active = (q_count != 0 && g.nx != 0);
    if (!warp_active) return;

    extern __shared__ float smem[];
    float* const sB_shared = smem;
    float* const warp_smem = smem + (size_t)KT * K * N;
    const size_t per_warp = (size_t)KT * M * K + (size_t)M * N;
    float* sA = warp_smem + (size_t)wid * per_warp;
    float* sC = sA + (size_t)KT * M * K;

    float best_d[KMAX];
    int best_i[KMAX];
    #pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        best_d[k] = FLT_MAX;
        best_i[k] = -1;
    }

    const int row_lane = lane;
    const bool owns_row = (row_lane < M && (uint32_t)row_lane < q_count);
    const uint32_t qi = q_start + (uint32_t)row_lane;
    const uint32_t qid = (owns_row && direct_qid) ? query_ids[qi] : qi;
    const float qn = owns_row ? (direct_qid ? all_norm[qid] : q_norm[qi]) : 0.f;
    const float* x_base_ptr = all_x + (size_t)g.x_off * (size_t)dim;
    const float* x_norm_ptr = all_norm + (size_t)g.x_off;

    for (uint32_t x_base = 0; x_base < g.nx; x_base += N) {
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
                const uint32_t qrow = q_start + (uint32_t)r;
                const int d = k0 + kt * K + kk;
                float v = 0.f;
                if ((uint32_t)r < q_count && d < dim) {
                    const uint32_t src = direct_qid ? query_ids[qrow] : qrow;
                    const float* q_base_ptr = direct_qid ? all_x : Q;
                    v = q_base_ptr[(size_t)src * (size_t)dim + (size_t)d];
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
                if (xlocal < g.nx && d < dim) {
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
            const int valid_cols = min((uint32_t)N, g.nx - x_base);
            const float* row = sC + (size_t)row_lane * N;
            for (int col = 0; col < valid_cols; ++col) {
                const uint32_t xlocal = x_base + (uint32_t)col;
                const float dist = qn + x_norm_ptr[xlocal] - 2.f * row[col];
                ung_topk_insert16(best_d, best_i, topk, dist, (int)xlocal);
            }
        }
        __syncwarp();
    }

    if (owns_row) {
        const size_t out_base = (size_t)qi * (size_t)topk;
        for (int k = 0; k < topk; ++k) {
            out_idx[out_base + (size_t)k] = best_i[k];
            out_dist[out_base + (size_t)k] = best_d[k];
        }
    }
}
