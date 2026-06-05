#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define WARP 32
#define KMAX 16

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

static void cb(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error at %s: %d\n", what, (int)s);
        std::exit(3);
    }
}

static float elapsed(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, a, b), "event elapsed");
    return ms;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

__device__ __forceinline__ bool topk_has(const int* ids, int topk, int id) {
    for (int i = 0; i < KMAX; ++i) {
        if (i < topk && ids[i] == id) return true;
    }
    return false;
}

__device__ __forceinline__ void top_insert(float* best_d, int* best_i, int topk, float d, int id) {
    if (id < 0 || d >= best_d[topk - 1] || topk_has(best_i, topk, id)) return;
    int pos = topk - 1;
    while (pos > 0 && d < best_d[pos - 1]) {
        best_d[pos] = best_d[pos - 1];
        best_i[pos] = best_i[pos - 1];
        --pos;
    }
    best_d[pos] = d;
    best_i[pos] = id;
}

__global__ void norms_kernel(const float* data, float* norms, int n, int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = data[(size_t)row * dim + d];
        sum += v * v;
    }
    __shared__ float smem[256];
    smem[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) norms[row] = smem[0];
}

__global__ void topk_from_dot_kernel(
    const float* dot,
    const float* q_norms,
    const float* x_norms,
    int groups,
    int nq,
    int nx,
    int topk,
    int* out_i,
    float* out_d) {
    int q_global = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = groups * nq;
    if (q_global >= total_q) return;
    int group = q_global / nq;
    int q_local = q_global - group * nq;
    const float* dot_row = dot + ((size_t)group * nq + q_local) * nx;
    const float* x_norm = x_norms + (size_t)group * nx;
    float qn = q_norms[q_global];
    float best_d[KMAX];
    int best_i[KMAX];
    for (int k = 0; k < KMAX; ++k) {
        best_d[k] = FLT_MAX;
        best_i[k] = -1;
    }
    for (int x = 0; x < nx; ++x) {
        float dist = qn + x_norm[x] - 2.0f * dot_row[x];
        top_insert(best_d, best_i, topk, dist, x);
    }
    size_t base = (size_t)q_global * topk;
    for (int k = 0; k < topk; ++k) {
        out_i[base + k] = best_i[k];
        out_d[base + k] = best_d[k];
    }
}

__global__ void fused_l2_topk_kernel(
    const float* q,
    const float* x,
    int groups,
    int nq,
    int nx,
    int dim,
    int topk,
    int* out_i,
    float* out_d) {
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_q = groups * nq;
    if (warp_global >= total_q) return;
    int group = warp_global / nq;
    int q_local = warp_global - group * nq;
    const float* qp = q + ((size_t)group * nq + q_local) * dim;
    const float* xp = x + (size_t)group * nx * dim;
    float best_d[KMAX];
    int best_i[KMAX];
    for (int k = 0; k < KMAX; ++k) {
        best_d[k] = FLT_MAX;
        best_i[k] = -1;
    }
    for (int xi = 0; xi < nx; ++xi) {
        float sum = 0.0f;
        const float* xv = xp + (size_t)xi * dim;
        for (int d = lane; d < dim; d += WARP) {
            float diff = qp[d] - xv[d];
            sum += diff * diff;
        }
        sum = warp_sum(sum);
        if (lane == 0) top_insert(best_d, best_i, topk, sum, xi);
    }
    if (lane == 0) {
        size_t base = (size_t)warp_global * topk;
        for (int k = 0; k < topk; ++k) {
            out_i[base + k] = best_i[k];
            out_d[base + k] = best_d[k];
        }
    }
}

struct Payload {
    float* d_q = nullptr;
    float* d_x = nullptr;
    float* d_dot = nullptr;
    float* d_q_norm = nullptr;
    float* d_x_norm = nullptr;
    int* d_sgemm_i = nullptr;
    float* d_sgemm_d = nullptr;
    int* d_fused_i = nullptr;
    float* d_fused_d = nullptr;
};

static void fill_data(std::vector<float>& q, std::vector<float>& x, int groups, int nq, int nx, int dim) {
    std::mt19937 rng(20260521);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.03f);
    for (float& v : x) v = normal(rng);
    for (int g = 0; g < groups; ++g) {
        for (int qi = 0; qi < nq; ++qi) {
            int picked = qi % nx;
            const float* src = x.data() + ((size_t)g * nx + picked) * dim;
            float* dst = q.data() + ((size_t)g * nq + qi) * dim;
            for (int d = 0; d < dim; ++d) dst[d] = src[d] + noise(rng);
        }
    }
}

static void alloc_payload(Payload& p, int groups, int nq, int nx, int dim, int topk) {
    size_t q_count = (size_t)groups * nq * dim;
    size_t x_count = (size_t)groups * nx * dim;
    size_t dot_count = (size_t)groups * nq * nx;
    size_t out_count = (size_t)groups * nq * topk;
    ck(cudaMalloc(&p.d_q, q_count * sizeof(float)), "malloc q");
    ck(cudaMalloc(&p.d_x, x_count * sizeof(float)), "malloc x");
    ck(cudaMalloc(&p.d_dot, dot_count * sizeof(float)), "malloc dot");
    ck(cudaMalloc(&p.d_q_norm, (size_t)groups * nq * sizeof(float)), "malloc q norms");
    ck(cudaMalloc(&p.d_x_norm, (size_t)groups * nx * sizeof(float)), "malloc x norms");
    ck(cudaMalloc(&p.d_sgemm_i, out_count * sizeof(int)), "malloc sgemm ids");
    ck(cudaMalloc(&p.d_sgemm_d, out_count * sizeof(float)), "malloc sgemm dists");
    ck(cudaMalloc(&p.d_fused_i, out_count * sizeof(int)), "malloc fused ids");
    ck(cudaMalloc(&p.d_fused_d, out_count * sizeof(float)), "malloc fused dists");
}

static void free_payload(Payload& p) {
    cudaFree(p.d_q);
    cudaFree(p.d_x);
    cudaFree(p.d_dot);
    cudaFree(p.d_q_norm);
    cudaFree(p.d_x_norm);
    cudaFree(p.d_sgemm_i);
    cudaFree(p.d_sgemm_d);
    cudaFree(p.d_fused_i);
    cudaFree(p.d_fused_d);
}

static void run_sgemm_topk(cublasHandle_t handle, Payload& p, int groups, int nq, int nx, int dim, int topk) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cb(cublasSgemmStridedBatched(
           handle,
           CUBLAS_OP_T,
           CUBLAS_OP_N,
           nx,
           nq,
           dim,
           &alpha,
           p.d_x,
           dim,
           (long long)nx * dim,
           p.d_q,
           dim,
           (long long)nq * dim,
           &beta,
           p.d_dot,
           nx,
           (long long)nq * nx,
           groups),
       "cublasSgemmStridedBatched");
    int total_q = groups * nq;
    topk_from_dot_kernel<<<(total_q + 127) / 128, 128>>>(p.d_dot, p.d_q_norm, p.d_x_norm, groups, nq, nx, topk, p.d_sgemm_i, p.d_sgemm_d);
}

static void run_fused(Payload& p, int groups, int nq, int nx, int dim, int topk) {
    int total_q = groups * nq;
    int warps_per_block = 8;
    fused_l2_topk_kernel<<<(total_q + warps_per_block - 1) / warps_per_block, warps_per_block * 32>>>(
        p.d_q, p.d_x, groups, nq, nx, dim, topk, p.d_fused_i, p.d_fused_d);
}

int main(int argc, char** argv) {
    int groups = argc > 1 ? std::atoi(argv[1]) : 100;
    int nx = argc > 2 ? std::atoi(argv[2]) : 256;
    int nq = argc > 3 ? std::atoi(argv[3]) : 1024;
    int dim = argc > 4 ? std::atoi(argv[4]) : 128;
    int topk = argc > 5 ? std::atoi(argv[5]) : 6;
    int payloads = argc > 6 ? std::atoi(argv[6]) : 2;
    int replays = argc > 7 ? std::atoi(argv[7]) : 4;
    if (topk > KMAX || nx <= 0 || nq <= 0 || dim <= 0 || groups <= 0) {
        std::fprintf(stderr, "usage: %s groups nx nq_per_group dim topk [payloads=2] [replays=4]\n", argv[0]);
        return 1;
    }
    payloads = std::max(1, payloads);
    replays = std::max(1, replays);

    std::vector<float> h_q((size_t)groups * nq * dim);
    std::vector<float> h_x((size_t)groups * nx * dim);
    fill_data(h_q, h_x, groups, nq, nx, dim);

    std::vector<Payload> ps(payloads);
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "event a");
    ck(cudaEventCreate(&b), "event b");
    float h2d_ms = 0.0f;
    for (int i = 0; i < payloads; ++i) {
        alloc_payload(ps[i], groups, nq, nx, dim, topk);
        ck(cudaEventRecord(a), "h2d start");
        ck(cudaMemcpy(ps[i].d_q, h_q.data(), h_q.size() * sizeof(float), cudaMemcpyHostToDevice), "copy q");
        ck(cudaMemcpy(ps[i].d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x");
        ck(cudaEventRecord(b), "h2d stop");
        ck(cudaEventSynchronize(b), "h2d sync");
        h2d_ms += elapsed(a, b);
        norms_kernel<<<groups * nq, 256>>>(ps[i].d_q, ps[i].d_q_norm, groups * nq, dim);
        norms_kernel<<<groups * nx, 256>>>(ps[i].d_x, ps[i].d_x_norm, groups * nx, dim);
    }
    h2d_ms /= payloads;

    cublasHandle_t handle;
    cb(cublasCreate(&handle), "cublasCreate");
    cb(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");

    for (int i = 0; i < 2 * payloads; ++i) {
        Payload& p = ps[i % payloads];
        run_sgemm_topk(handle, p, groups, nq, nx, dim, topk);
        run_fused(p, groups, nq, nx, dim, topk);
    }
    ck(cudaDeviceSynchronize(), "warmup sync");

    float sgemm_ms = 0.0f;
    for (int r = 0; r < replays; ++r) {
        for (int i = 0; i < payloads; ++i) {
            ck(cudaEventRecord(a), "sgemm start");
            run_sgemm_topk(handle, ps[i], groups, nq, nx, dim, topk);
            ck(cudaEventRecord(b), "sgemm stop");
            ck(cudaEventSynchronize(b), "sgemm sync");
            sgemm_ms += elapsed(a, b);
        }
    }
    sgemm_ms /= (payloads * replays);

    float fused_ms = 0.0f;
    for (int r = 0; r < replays; ++r) {
        for (int i = 0; i < payloads; ++i) {
            ck(cudaEventRecord(a), "fused start");
            run_fused(ps[i], groups, nq, nx, dim, topk);
            ck(cudaEventRecord(b), "fused stop");
            ck(cudaEventSynchronize(b), "fused sync");
            fused_ms += elapsed(a, b);
        }
    }
    fused_ms /= (payloads * replays);

    std::vector<int> sgemm_i((size_t)groups * nq * topk);
    std::vector<int> fused_i((size_t)groups * nq * topk);
    ck(cudaMemcpy(sgemm_i.data(), ps[0].d_sgemm_i, sgemm_i.size() * sizeof(int), cudaMemcpyDeviceToHost), "copy sgemm ids");
    ck(cudaMemcpy(fused_i.data(), ps[0].d_fused_i, fused_i.size() * sizeof(int), cudaMemcpyDeviceToHost), "copy fused ids");
    long long exact_match = 0;
    long long total = (long long)groups * nq * topk;
    for (long long i = 0; i < total; ++i) {
        if (sgemm_i[i] == fused_i[i]) ++exact_match;
    }
    double id_match = (double)exact_match / (double)total;
    double total_q = (double)groups * nq;
    double work_pairs = total_q * nx;

    std::printf("method,groups,nx,nq_per_group,total_q,dim,topk,payloads,replays,h2d_ms,search_ms,e2e_reuse_ms,work_pairs,id_match_vs_sgemm,exact\n");
    std::printf("nvidia_sgemm_plus_topk,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.0f,1.000000,1\n",
                groups, nx, nq, groups * nq, dim, topk, payloads, replays, h2d_ms, sgemm_ms, h2d_ms + sgemm_ms, work_pairs);
    std::printf("our_fused_l2_topk,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.0f,%.6f,1\n",
                groups, nx, nq, groups * nq, dim, topk, payloads, replays, h2d_ms, fused_ms, h2d_ms + fused_ms, work_pairs, id_match);
    std::printf("our_hybrid_sgemm_for_heavy,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.0f,1.000000,1\n",
                groups, nx, nq, groups * nq, dim, topk, payloads, replays, h2d_ms, sgemm_ms, h2d_ms + sgemm_ms, work_pairs);

    for (Payload& p : ps) free_payload(p);
    cublasDestroy(handle);
    return 0;
}
