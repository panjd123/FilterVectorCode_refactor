// Shared norm/topK update kernels for cross-edge GPU routes.
//
// This file is included from gpu_gemm_topk.cu inside its anonymous namespace,
// before double-buffer and SGEMM baseline helpers that launch these kernels.

// Compute squared L2 norm for each row: out[i] = sum_d A[i,d]^2.
__global__ void l2_norm_sq_kernel(
    const float* __restrict__ A,
    int n,
    int dim,
    float* __restrict__ out)
{
    constexpr int WARP = 32;
    const int lane = threadIdx.x & (WARP - 1);
    const int warp_id = threadIdx.x / WARP;
    const int warps_per_block = blockDim.x / WARP;
    int i = blockIdx.x * warps_per_block + warp_id;
    if (i >= n) return;

    float acc = 0.f;
    const float* a = A + (size_t)i * dim;

    for (int d = lane; d < dim; d += WARP) {
        float v = a[d];
        acc += v * v;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) out[i] = acc;
}

// Initialize topK output arrays to idx=-1 and dist=INF.
__global__ void init_topk_kernel(int nq, int topk, int* out_idx, float* out_dist) {
    const long long total = (long long)nq * (long long)topk;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)blockDim.x * gridDim.x) {
        out_idx[(size_t)idx] = -1;
        out_dist[(size_t)idx] = FLT_MAX;
    }
}

__device__ __forceinline__
void topk_insert_linear(float* best_dist, int* best_idx, int K, float dist, int idx)
{
    if (dist >= best_dist[K - 1]) return;
    int pos = K - 1;
    while (pos > 0 && dist < best_dist[pos - 1]) {
        best_dist[pos] = best_dist[pos - 1];
        best_idx[pos]  = best_idx[pos - 1];
        --pos;
    }
    best_dist[pos] = dist;
    best_idx[pos]  = idx;
}

// Update running topK from one GEMM dot tile.
__global__ void update_topk_from_dot_tile_kernel(
    const float* __restrict__ dot,
    const float* __restrict__ qn,
    const float* __restrict__ xn_tile,
    int nq,
    int tile,
    int topk,
    int x_base,
    int* __restrict__ best_idx,
    float* __restrict__ best_dist)
{
    int qi = blockIdx.x;
    if (qi >= nq) return;

    constexpr int KMAX = 64;
    int K = topk;
    if (K > KMAX) K = KMAX;

    float cur_best[KMAX];
    int cur_idx[KMAX];
    if (threadIdx.x == 0) {
        for (int k = 0; k < K; ++k) {
            cur_best[k] = best_dist[(size_t)qi * topk + k];
            cur_idx[k] = best_idx[(size_t)qi * topk + k];
        }
        for (int k = K; k < KMAX; ++k) {
            cur_best[k] = FLT_MAX;
            cur_idx[k] = -1;
        }
    }
    __syncthreads();

    float local_best[KMAX];
    int local_idx[KMAX];
    for (int k = 0; k < K; ++k) {
        local_best[k] = FLT_MAX;
        local_idx[k] = -1;
    }
    for (int k = K; k < KMAX; ++k) {
        local_best[k] = FLT_MAX;
        local_idx[k] = -1;
    }

    float qnorm = qn[qi];
    const float* row = dot + (size_t)qi * tile;

    for (int j = threadIdx.x; j < tile; j += blockDim.x) {
        float d = qnorm + xn_tile[j] - 2.0f * row[j];
        topk_insert_linear(local_best, local_idx, K, d, x_base + j);
    }

    extern __shared__ unsigned char smem[];
    float* sdist = reinterpret_cast<float*>(smem);
    int* sidx = reinterpret_cast<int*>(sdist + blockDim.x * K);

    for (int k = 0; k < K; ++k) {
        int off = threadIdx.x * K + k;
        sdist[off] = local_best[k];
        sidx[off] = local_idx[k];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = blockDim.x * K;
        for (int r = 0; r < K; ++r) {
            int best = r;
            for (int t = r + 1; t < total; ++t) {
                if (sdist[t] < sdist[best]) best = t;
            }
            if (best != r) {
                float td = sdist[best];
                sdist[best] = sdist[r];
                sdist[r] = td;
                int ti = sidx[best];
                sidx[best] = sidx[r];
                sidx[r] = ti;
            }
        }

        for (int r = 0; r < K; ++r) {
            if (sidx[r] < 0) continue;
            topk_insert_linear(cur_best, cur_idx, K, sdist[r], sidx[r]);
        }

        for (int k = 0; k < K; ++k) {
            best_dist[(size_t)qi * topk + k] = cur_best[k];
            best_idx[(size_t)qi * topk + k] = cur_idx[k];
        }
        for (int k = K; k < topk; ++k) {
            best_dist[(size_t)qi * topk + k] = FLT_MAX;
            best_idx[(size_t)qi * topk + k] = -1;
        }
    }
}

// Same semantics as update_topk_from_dot_tile_kernel, but consumes multiple
// same-shape dot tiles in one launch.
__global__ void update_topk_from_dot_grouped_tiles_kernel(
    const float* __restrict__ dot_batches,
    const float* __restrict__ qn,
    const float* __restrict__ xn_group,
    int nq,
    int tile,
    int num_tiles,
    int topk,
    int x_base,
    int* __restrict__ best_idx,
    float* __restrict__ best_dist)
{
    int qi = blockIdx.x;
    if (qi >= nq) return;

    constexpr int KMAX = 64;
    int K = topk;
    if (K > KMAX) K = KMAX;

    float cur_best[KMAX];
    int cur_idx[KMAX];
    if (threadIdx.x == 0) {
        for (int k = 0; k < K; ++k) {
            cur_best[k] = best_dist[(size_t)qi * topk + k];
            cur_idx[k] = best_idx[(size_t)qi * topk + k];
        }
        for (int k = K; k < KMAX; ++k) {
            cur_best[k] = FLT_MAX;
            cur_idx[k] = -1;
        }
    }
    __syncthreads();

    float local_best[KMAX];
    int local_idx[KMAX];
    for (int k = 0; k < K; ++k) {
        local_best[k] = FLT_MAX;
        local_idx[k] = -1;
    }
    for (int k = K; k < KMAX; ++k) {
        local_best[k] = FLT_MAX;
        local_idx[k] = -1;
    }

    const float qnorm = qn[qi];
    const int total_cols = tile * num_tiles;
    const size_t dot_stride = (size_t)nq * (size_t)tile;

    for (int p = threadIdx.x; p < total_cols; p += blockDim.x) {
        int tile_id = p / tile;
        int j = p - tile_id * tile;
        const float* row = dot_batches + (size_t)tile_id * dot_stride + (size_t)qi * tile;
        float d = qnorm + xn_group[p] - 2.0f * row[j];
        topk_insert_linear(local_best, local_idx, K, d, x_base + p);
    }

    extern __shared__ unsigned char smem[];
    float* sdist = reinterpret_cast<float*>(smem);
    int* sidx = reinterpret_cast<int*>(sdist + blockDim.x * K);

    for (int k = 0; k < K; ++k) {
        int off = threadIdx.x * K + k;
        sdist[off] = local_best[k];
        sidx[off] = local_idx[k];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = blockDim.x * K;
        for (int r = 0; r < K; ++r) {
            int best = r;
            for (int t = r + 1; t < total; ++t) {
                if (sdist[t] < sdist[best]) best = t;
            }
            if (best != r) {
                float td = sdist[best];
                sdist[best] = sdist[r];
                sdist[r] = td;
                int ti = sidx[best];
                sidx[best] = sidx[r];
                sidx[r] = ti;
            }
        }

        for (int r = 0; r < K; ++r) {
            if (sidx[r] < 0) continue;
            topk_insert_linear(cur_best, cur_idx, K, sdist[r], sidx[r]);
        }

        for (int k = 0; k < K; ++k) {
            best_dist[(size_t)qi * topk + k] = cur_best[k];
            best_idx[(size_t)qi * topk + k] = cur_idx[k];
        }
        for (int k = K; k < topk; ++k) {
            best_dist[(size_t)qi * topk + k] = FLT_MAX;
            best_idx[(size_t)qi * topk + k] = -1;
        }
    }
}
