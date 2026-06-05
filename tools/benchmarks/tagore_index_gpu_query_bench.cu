#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#define WARP 32
#define KMAX 16
#define DEGMAX 128
#define ENTRYMAX 512
#define ETOPMAX 16

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

static bool read_fvecs(const std::string& path, int expected_dim, std::vector<float>& out, int& n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    std::streamoff bytes = in.tellg();
    in.seekg(0, std::ios::beg);
    const size_t rec = sizeof(int) + (size_t)expected_dim * sizeof(float);
    if (bytes <= 0 || (size_t)bytes % rec != 0) return false;
    n = (int)((size_t)bytes / rec);
    out.resize((size_t)n * expected_dim);
    for (int i = 0; i < n; ++i) {
        int dim = 0;
        in.read((char*)&dim, sizeof(int));
        if (dim != expected_dim) return false;
        in.read((char*)&out[(size_t)i * expected_dim], (size_t)expected_dim * sizeof(float));
    }
    return true;
}

static bool read_vamana_index(const std::string& path, int n, int degree, std::vector<int>& graph, uint32_t& ep, uint32_t& file_max_degree) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint64_t index_size = 0;
    size_t frozen = 0;
    in.read((char*)&index_size, sizeof(uint64_t));
    in.read((char*)&file_max_degree, sizeof(uint32_t));
    in.read((char*)&ep, sizeof(uint32_t));
    in.read((char*)&frozen, sizeof(size_t));
    graph.assign((size_t)n * degree, -1);
    for (int i = 0; i < n; ++i) {
        uint32_t GK = 0;
        in.read((char*)&GK, sizeof(uint32_t));
        std::vector<uint32_t> tmp(GK);
        if (GK) in.read((char*)tmp.data(), (size_t)GK * sizeof(uint32_t));
        int w = 0;
        for (uint32_t j = 0; j < GK && w < degree; ++j) {
            if (tmp[j] < (uint32_t)n && tmp[j] != (uint32_t)i) graph[(size_t)i * degree + w++] = (int)tmp[j];
        }
    }
    return true;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float l2_warp(const float* a, const float* b, int dim, int lane) {
    float s = 0.f;
    for (int d = lane; d < dim; d += WARP) {
        float x = a[d] - b[d];
        s += x * x;
    }
    return warp_sum(s);
}

__device__ __forceinline__ bool topk_has(const int* ids, int K, int id) {
    for (int i = 0; i < KMAX; ++i) if (i < K && ids[i] == id) return true;
    return false;
}

__device__ __forceinline__ void top_insert_unique(float* bd, int* bi, int K, float d, int id) {
    if (id < 0 || d >= bd[K - 1] || topk_has(bi, K, id)) return;
    int pos = K - 1;
    while (pos > 0 && d < bd[pos - 1]) {
        bd[pos] = bd[pos - 1];
        bi[pos] = bi[pos - 1];
        --pos;
    }
    bd[pos] = d;
    bi[pos] = id;
}

__global__ void exact_search_var_kernel(const float* Q, const float* X, const int* x_offsets, const int* q_offsets,
                                        int groups, int dim, int topk, int* out_i, float* out_d) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int w = tid >> 5;
    int lane = threadIdx.x & 31;
    int total_q = q_offsets[groups];
    if (w >= total_q) return;
    int g = 0;
    while (g + 1 < groups && q_offsets[g + 1] <= w) ++g;
    int q_local = w - q_offsets[g];
    int nx = x_offsets[g + 1] - x_offsets[g];
    const float* qp = Q + (size_t)w * dim;
    const float* xb = X + (size_t)x_offsets[g] * dim;
    float bd[KMAX]; int bi[KMAX];
    for (int k = 0; k < KMAX; ++k) { bd[k] = FLT_MAX; bi[k] = -1; }
    for (int x = 0; x < nx; ++x) {
        float d = l2_warp(qp, xb + (size_t)x * dim, dim, lane);
        if (lane == 0) top_insert_unique(bd, bi, topk, d, x);
    }
    if (lane == 0) {
        size_t base = (size_t)w * topk;
        for (int k = 0; k < topk; ++k) { out_i[base + k] = bi[k]; out_d[base + k] = bd[k]; }
    }
    (void)q_local;
}

__global__ void graph_cfs_search_var_kernel(const float* Q, const float* X, const int* graph, const int* x_offsets, const int* q_offsets,
                                            int groups, int dim, int degree, int topk, int entry_count, int entry_top,
                                            int* out_i, float* out_d) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int w = tid >> 5;
    int lane = threadIdx.x & 31;
    int total_q = q_offsets[groups];
    if (w >= total_q) return;
    int g = 0;
    while (g + 1 < groups && q_offsets[g + 1] <= w) ++g;
    int nx = x_offsets[g + 1] - x_offsets[g];
    const float* qp = Q + (size_t)w * dim;
    const float* xb = X + (size_t)x_offsets[g] * dim;
    const int* gg = graph + (size_t)x_offsets[g] * degree;

    float ed[ETOPMAX]; int ei[ETOPMAX];
    for (int k = 0; k < ETOPMAX; ++k) { ed[k] = FLT_MAX; ei[k] = -1; }
    for (int e = 0; e < entry_count; ++e) {
        int id = (int)(((long long)e * nx) / entry_count);
        float d = l2_warp(qp, xb + (size_t)id * dim, dim, lane);
        if (lane == 0) top_insert_unique(ed, ei, entry_top, d, id);
    }

    float bd[KMAX]; int bi[KMAX];
    for (int k = 0; k < KMAX; ++k) { bd[k] = FLT_MAX; bi[k] = -1; }
    for (int t = 0; t < entry_top; ++t) {
        int entry = __shfl_sync(0xffffffff, ei[t], 0);
        float de = (entry >= 0) ? l2_warp(qp, xb + (size_t)entry * dim, dim, lane) : FLT_MAX;
        if (lane == 0) top_insert_unique(bd, bi, topk, de, entry);
        if (entry < 0) continue;
        for (int j = 0; j < degree; ++j) {
            int id = gg[(size_t)entry * degree + j];
            float d = (id >= 0) ? l2_warp(qp, xb + (size_t)id * dim, dim, lane) : FLT_MAX;
            if (lane == 0) top_insert_unique(bd, bi, topk, d, id);
        }
    }
    if (lane == 0) {
        size_t base = (size_t)w * topk;
        for (int k = 0; k < topk; ++k) { out_i[base + k] = bi[k]; out_d[base + k] = bd[k]; }
    }
}

static float elapsed(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.f;
    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    return ms;
}

int main(int argc, char** argv) {
    if (argc < 5 || ((argc - 5) % 3) != 0) {
        std::fprintf(stderr, "usage: %s dim nq_per_group topk degree [data.fvecs index.vamana build_ms]...\n", argv[0]);
        return 1;
    }
    int dim = std::atoi(argv[1]);
    int nq_per_group = std::atoi(argv[2]);
    int topk = std::atoi(argv[3]);
    int degree = std::atoi(argv[4]);
    int groups = (argc - 5) / 3;
    if (topk > KMAX || degree > DEGMAX) return 2;

    std::vector<int> x_offsets(groups + 1, 0), q_offsets(groups + 1, 0);
    std::vector<float> hX, hQ;
    std::vector<int> hGraph;
    double build_ms = 0.0;
    std::mt19937 rng(20260519);
    std::normal_distribution<float> noise(0.f, 0.05f);

    for (int g = 0; g < groups; ++g) {
        std::string data_path = argv[5 + g * 3];
        std::string index_path = argv[6 + g * 3];
        double bm = std::atof(argv[7 + g * 3]);
        build_ms += bm;
        std::vector<float> oneX;
        int n = 0;
        if (!read_fvecs(data_path, dim, oneX, n)) { std::fprintf(stderr, "failed to read %s\n", data_path.c_str()); return 3; }
        uint32_t ep = 0, file_max = 0;
        std::vector<int> oneG;
        if (!read_vamana_index(index_path, n, degree, oneG, ep, file_max)) { std::fprintf(stderr, "failed to read %s\n", index_path.c_str()); return 4; }
        x_offsets[g + 1] = x_offsets[g] + n;
        q_offsets[g + 1] = q_offsets[g] + nq_per_group;
        hX.insert(hX.end(), oneX.begin(), oneX.end());
        hGraph.insert(hGraph.end(), oneG.begin(), oneG.end());
        std::uniform_int_distribution<int> pick(0, n - 1);
        for (int q = 0; q < nq_per_group; ++q) {
            int a = pick(rng);
            for (int d = 0; d < dim; ++d) hQ.push_back(oneX[(size_t)a * dim + d] + noise(rng));
        }
    }
    int total_x = x_offsets[groups], total_q = q_offsets[groups];

    float *dX, *dQ, *dExactD, *dAnnD;
    int *dGraph, *dExactI, *dAnnI, *dXoff, *dQoff;
    ck(cudaMalloc(&dX, (size_t)hX.size() * sizeof(float)), "malloc X");
    ck(cudaMalloc(&dQ, (size_t)hQ.size() * sizeof(float)), "malloc Q");
    ck(cudaMalloc(&dGraph, (size_t)hGraph.size() * sizeof(int)), "malloc graph");
    ck(cudaMalloc(&dXoff, (size_t)x_offsets.size() * sizeof(int)), "malloc xoff");
    ck(cudaMalloc(&dQoff, (size_t)q_offsets.size() * sizeof(int)), "malloc qoff");
    ck(cudaMalloc(&dExactI, (size_t)total_q * topk * sizeof(int)), "malloc exactI");
    ck(cudaMalloc(&dExactD, (size_t)total_q * topk * sizeof(float)), "malloc exactD");
    ck(cudaMalloc(&dAnnI, (size_t)total_q * topk * sizeof(int)), "malloc annI");
    ck(cudaMalloc(&dAnnD, (size_t)total_q * topk * sizeof(float)), "malloc annD");

    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "event a"); ck(cudaEventCreate(&b), "event b");
    ck(cudaEventRecord(a), "h2d a");
    ck(cudaMemcpy(dX, hX.data(), (size_t)hX.size() * sizeof(float), cudaMemcpyHostToDevice), "copy X");
    ck(cudaMemcpy(dQ, hQ.data(), (size_t)hQ.size() * sizeof(float), cudaMemcpyHostToDevice), "copy Q");
    ck(cudaMemcpy(dGraph, hGraph.data(), (size_t)hGraph.size() * sizeof(int), cudaMemcpyHostToDevice), "copy graph");
    ck(cudaMemcpy(dXoff, x_offsets.data(), (size_t)x_offsets.size() * sizeof(int), cudaMemcpyHostToDevice), "copy xoff");
    ck(cudaMemcpy(dQoff, q_offsets.data(), (size_t)q_offsets.size() * sizeof(int), cudaMemcpyHostToDevice), "copy qoff");
    ck(cudaEventRecord(b), "h2d b"); ck(cudaEventSynchronize(b), "h2d sync");
    float h2d = elapsed(a, b);

    int wpb = 8;
    dim3 block(wpb * 32), grid_q((total_q + wpb - 1) / wpb);
    exact_search_var_kernel<<<grid_q, block>>>(dQ, dX, dXoff, dQoff, groups, dim, topk, dExactI, dExactD);
    graph_cfs_search_var_kernel<<<grid_q, block>>>(dQ, dX, dGraph, dXoff, dQoff, groups, dim, degree, topk, 128, 4, dAnnI, dAnnD);
    ck(cudaDeviceSynchronize(), "warmup");

    ck(cudaEventRecord(a), "exact a");
    exact_search_var_kernel<<<grid_q, block>>>(dQ, dX, dXoff, dQoff, groups, dim, topk, dExactI, dExactD);
    ck(cudaEventRecord(b), "exact b"); ck(cudaEventSynchronize(b), "exact sync");
    float exact = elapsed(a, b);

    const int configs[][2] = {{64,4},{128,4},{128,8},{256,4},{256,8}};
    printf("groups,total_x,nq_per_group,total_q,dim,topk,degree,build_ms,h2d_ms,exact_search_ms,entry_count,entry_top,ann_search_ms,d2h_ms,exact_e2e_ms,ann_e2e_once_ms,ann_e2e_reuse_ms,search_speedup,e2e_once_speedup,e2e_reuse_speedup,recall_at_k\n");
    for (auto& cfg : configs) {
        int entry_count = cfg[0], entry_top = cfg[1];
        ck(cudaEventRecord(a), "ann a");
        graph_cfs_search_var_kernel<<<grid_q, block>>>(dQ, dX, dGraph, dXoff, dQoff, groups, dim, degree, topk, entry_count, entry_top, dAnnI, dAnnD);
        ck(cudaEventRecord(b), "ann b"); ck(cudaEventSynchronize(b), "ann sync");
        float ann = elapsed(a, b);
        std::vector<int> hExactI((size_t)total_q * topk), hAnnI((size_t)total_q * topk);
        ck(cudaEventRecord(a), "d2h a");
        ck(cudaMemcpy(hExactI.data(), dExactI, hExactI.size() * sizeof(int), cudaMemcpyDeviceToHost), "copy exact");
        ck(cudaMemcpy(hAnnI.data(), dAnnI, hAnnI.size() * sizeof(int), cudaMemcpyDeviceToHost), "copy ann");
        ck(cudaEventRecord(b), "d2h b"); ck(cudaEventSynchronize(b), "d2h sync");
        float d2h = elapsed(a, b);
        long long hits = 0;
        for (int q = 0; q < total_q; ++q) {
            for (int i = 0; i < topk; ++i) {
                int id = hAnnI[(size_t)q * topk + i];
                for (int j = 0; j < topk; ++j) if (id == hExactI[(size_t)q * topk + j]) { ++hits; break; }
            }
        }
        double recall = (double)hits / ((long long)total_q * topk);
        double exact_e2e = h2d + exact + d2h;
        double ann_once = build_ms + h2d + ann + d2h;
        double ann_reuse = h2d + ann + d2h;
        printf("%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
               groups,total_x,nq_per_group,total_q,dim,topk,degree,build_ms,h2d,exact,entry_count,entry_top,ann,d2h,exact_e2e,ann_once,ann_reuse,exact/ann,exact_e2e/ann_once,exact_e2e/ann_reuse,recall);
    }
    return 0;
}
