#include "fixed_pool_graph.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>

namespace ANNS {
namespace {

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err)); \
    } \
} while (0)

int read_env_int(const char* name, int default_value, int lo, int hi) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    char* end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v) return default_value;
    if (parsed < lo) parsed = lo;
    if (parsed > hi) parsed = hi;
    return static_cast<int>(parsed);
}

uint64_t read_env_u64(const char* name, uint64_t default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(v, &end, 10);
    if (end == v) return default_value;
    return static_cast<uint64_t>(parsed);
}

__device__ __forceinline__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ float l2_dist(const float* x, const float* y, int dim) {
    float acc = 0.0f;
    for (int d = 0; d < dim; ++d) {
        const float diff = x[d] - y[d];
        acc += diff * diff;
    }
    return acc;
}

__global__ void init_pool_kernel(const float* vecs, int n, int dim, int pool_l,
                                 uint32_t seed, uint32_t* pool, float* dist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * pool_l;
    if (idx >= total) return;

    int i = idx / pool_l;
    int slot = idx - i * pool_l;
    uint32_t h = mix32(seed ^ (uint32_t)(i * 0x9e3779b9u) ^ (uint32_t)(slot * 0x85ebca6bu));
    uint32_t cand = (n <= 1) ? 0u : (h % (uint32_t)(n - 1));
    if ((int)cand >= i) ++cand;
    pool[idx] = cand;
    dist[idx] = (n <= 1) ? INFINITY : l2_dist(vecs + (size_t)i * dim, vecs + (size_t)cand * dim, dim);
}

__device__ void insert_candidate_serial(uint32_t* ids, float* dists, int pool_l,
                                        uint32_t cand, float cand_dist) {
    if (cand_dist >= dists[pool_l - 1]) return;
    for (int k = 0; k < pool_l; ++k) {
        if (ids[k] == cand) {
            if (cand_dist < dists[k]) dists[k] = cand_dist;
            return;
        }
    }
    int pos = pool_l - 1;
    while (pos > 0 && cand_dist < dists[pos - 1]) {
        ids[pos] = ids[pos - 1];
        dists[pos] = dists[pos - 1];
        --pos;
    }
    ids[pos] = cand;
    dists[pos] = cand_dist;
}

// One block owns one point. This first implementation uses warp/block parallelism
// to compute candidate distances, then a serial per-node merge for simplicity.
__global__ void refine_pool_kernel(const float* vecs, int n, int dim, int pool_l,
                                   uint32_t* pool, float* dist) {
    int i = blockIdx.x;
    if (i >= n) return;

    extern __shared__ unsigned char smem[];
    uint32_t* cand_ids = reinterpret_cast<uint32_t*>(smem);
    float* cand_dists = reinterpret_cast<float*>(cand_ids + pool_l * pool_l);

    const float* qi = vecs + (size_t)i * dim;
    const int total = pool_l * pool_l;
    for (int t = threadIdx.x; t < total; t += blockDim.x) {
        int a = t / pool_l;
        int b = t - a * pool_l;
        uint32_t nb = pool[i * pool_l + a];
        uint32_t cand = pool[(int)nb * pool_l + b];
        cand_ids[t] = cand;
        cand_dists[t] = ((int)cand == i) ? INFINITY : l2_dist(qi, vecs + (size_t)cand * dim, dim);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t local_ids[128];
        float local_dists[128];
        for (int k = 0; k < pool_l; ++k) {
            local_ids[k] = pool[i * pool_l + k];
            local_dists[k] = dist[i * pool_l + k];
        }
        // Keep local pool sorted. Random initialization is not sorted.
        for (int a = 1; a < pool_l; ++a) {
            uint32_t id = local_ids[a];
            float dd = local_dists[a];
            int b = a - 1;
            while (b >= 0 && dd < local_dists[b]) {
                local_ids[b + 1] = local_ids[b];
                local_dists[b + 1] = local_dists[b];
                --b;
            }
            local_ids[b + 1] = id;
            local_dists[b + 1] = dd;
        }
        for (int t = 0; t < total; ++t) {
            insert_candidate_serial(local_ids, local_dists, pool_l, cand_ids[t], cand_dists[t]);
        }
        for (int k = 0; k < pool_l; ++k) {
            pool[i * pool_l + k] = local_ids[k];
            dist[i * pool_l + k] = local_dists[k];
        }
    }
}

__global__ void prune_pool_kernel(const float* vecs, int n, int dim, int pool_l, int max_degree,
                                  float alpha, const uint32_t* pool, const float* dist,
                                  uint32_t* edges, uint32_t* degrees) {
    int i = blockIdx.x;
    if (i >= n) return;
    if (threadIdx.x != 0) return;

    uint32_t selected[128];
    uint32_t selected_count = 0;
    for (int c = 0; c < pool_l && selected_count < (uint32_t)max_degree; ++c) {
        uint32_t cand = pool[i * pool_l + c];
        if ((int)cand == i || !isfinite(dist[i * pool_l + c])) continue;

        bool duplicate = false;
        for (uint32_t s = 0; s < selected_count; ++s) duplicate |= (selected[s] == cand);
        if (duplicate) continue;

        bool occluded = false;
        const float dist_i_c = dist[i * pool_l + c];
        const float* xc = vecs + (size_t)cand * dim;
        for (uint32_t s = 0; s < selected_count; ++s) {
            const float dist_s_c = l2_dist(vecs + (size_t)selected[s] * dim, xc, dim);
            if (alpha * dist_s_c <= dist_i_c) {
                occluded = true;
                break;
            }
        }
        if (!occluded) selected[selected_count++] = cand;
    }

    degrees[i] = selected_count;
    for (int k = 0; k < max_degree; ++k) {
        edges[i * max_degree + k] = (k < (int)selected_count) ? selected[k] : UINT32_MAX;
    }
}

} // namespace

FixedPoolBuildStats build_fixed_pool_graph_gpu(
    std::shared_ptr<IStorage> storage,
    std::shared_ptr<DistanceHandler> /*distance_handler*/,
    std::shared_ptr<Graph> graph,
    IdxType max_degree,
    IdxType requested_pool_size,
    float alpha,
    uint32_t iterations,
    uint64_t seed) {

    FixedPoolBuildStats stats;
    const IdxType n = storage->get_num_points();
    const IdxType dim = storage->get_dim();
    stats.num_points = n;
    stats.dim = dim;

    if (storage->get_data_type() != FLOAT) {
        throw std::runtime_error("FixedPoolGPU currently supports only float vectors.");
    }
    if (max_degree > 128) {
        throw std::runtime_error("FixedPoolGPU currently supports max_degree <= 128.");
    }
    if (max_degree == 0 || n == 0) return stats;

    int pool_l = static_cast<int>(std::max<IdxType>(max_degree, requested_pool_size));
    pool_l = read_env_int("UNG_FIXED_POOL_L", pool_l, (int)max_degree, 128);
    pool_l = std::min<int>(pool_l, 128);
    if (pool_l < (int)max_degree) pool_l = (int)max_degree;
    iterations = (uint32_t)read_env_int("UNG_FIXED_POOL_ITERS", (int)iterations, 1, 64);
    seed = read_env_u64("UNG_FIXED_POOL_SEED", seed);

    stats.pool_size = (IdxType)pool_l;
    stats.iterations = iterations;

    auto wall0 = std::chrono::high_resolution_clock::now();
    const float* h_vecs = reinterpret_cast<const float*>(storage->get_vector(0));
    const size_t vec_bytes = (size_t)n * (size_t)dim * sizeof(float);
    const size_t pool_bytes = (size_t)n * (size_t)pool_l * sizeof(uint32_t);
    const size_t dist_bytes = (size_t)n * (size_t)pool_l * sizeof(float);
    const size_t edge_bytes = (size_t)n * (size_t)max_degree * sizeof(uint32_t);

    float *d_vecs = nullptr;
    uint32_t *d_pool = nullptr, *d_edges = nullptr, *d_degrees = nullptr;
    float* d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vecs, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_pool, pool_bytes));
    CUDA_CHECK(cudaMalloc(&d_dist, dist_bytes));
    CUDA_CHECK(cudaMalloc(&d_edges, edge_bytes));
    CUDA_CHECK(cudaMalloc(&d_degrees, (size_t)n * sizeof(uint32_t)));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d_vecs, h_vecs, vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    stats.h2d_ms += ms;

    const int init_threads = 256;
    const int init_blocks = (int)(((size_t)n * (size_t)pool_l + init_threads - 1) / init_threads);
    CUDA_CHECK(cudaEventRecord(e0));
    init_pool_kernel<<<init_blocks, init_threads>>>(d_vecs, (int)n, (int)dim, pool_l,
                                                    (uint32_t)seed, d_pool, d_dist);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    stats.init_ms += ms;

    const int refine_threads = read_env_int("UNG_FIXED_POOL_THREADS", 128, 32, 256);
    const size_t refine_smem = (size_t)pool_l * (size_t)pool_l * (sizeof(uint32_t) + sizeof(float));
    CUDA_CHECK(cudaEventRecord(e0));
    for (uint32_t it = 0; it < iterations; ++it) {
        refine_pool_kernel<<<(int)n, refine_threads, refine_smem>>>(d_vecs, (int)n, (int)dim, pool_l, d_pool, d_dist);
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    stats.refine_ms += ms;

    CUDA_CHECK(cudaEventRecord(e0));
    prune_pool_kernel<<<(int)n, 32>>>(d_vecs, (int)n, (int)dim, pool_l, (int)max_degree,
                                     alpha, d_pool, d_dist, d_edges, d_degrees);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    stats.prune_ms += ms;

    std::vector<uint32_t> h_edges((size_t)n * (size_t)max_degree);
    std::vector<uint32_t> h_degrees(n);
    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(h_edges.data(), d_edges, edge_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_degrees.data(), d_degrees, (size_t)n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    stats.d2h_ms += ms;

    for (IdxType i = 0; i < n; ++i) {
        auto& out = graph->neighbors[i];
        out.clear();
        out.reserve(std::min<IdxType>(max_degree, h_degrees[i]));
        for (IdxType k = 0; k < max_degree && k < h_degrees[i]; ++k) {
            uint32_t nb = h_edges[(size_t)i * (size_t)max_degree + k];
            if (nb != UINT32_MAX && nb != i) out.push_back(nb);
        }
        if (out.empty() && n > 1) out.push_back((i + 1) % n);
    }

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    cudaFree(d_vecs);
    cudaFree(d_pool);
    cudaFree(d_dist);
    cudaFree(d_edges);
    cudaFree(d_degrees);

    stats.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - wall0).count();

    std::cout << "[FixedPoolGPU] n=" << n
              << " dim=" << dim
              << " R=" << max_degree
              << " L=" << pool_l
              << " iters=" << iterations
              << " h2d_ms=" << stats.h2d_ms
              << " init_ms=" << stats.init_ms
              << " refine_ms=" << stats.refine_ms
              << " prune_ms=" << stats.prune_ms
              << " d2h_ms=" << stats.d2h_ms
              << " total_ms=" << stats.total_ms
              << std::endl;
    return stats;
}

} // namespace ANNS
