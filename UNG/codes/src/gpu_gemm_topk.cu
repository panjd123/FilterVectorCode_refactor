#include "uni_nav_graph.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cassert>
#include <fstream>
#include <mutex>
#include <cstdarg>
#include <string>
#include <cfloat>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <omp.h>
#include <unordered_map>

struct UngGroupQueryDesc {
    uint32_t qi;
    uint32_t x_off;
    uint32_t nx;
};

// ========== GEMM 相关 ==========
#include <cublas_v2.h>
#include <cublasLt.h>

namespace {

// ============================================================
// 1) 计时/日志
// ============================================================
static constexpr const char* kProfLogPath =
    "/home/graphdb/Codes/FilterVectorResultsCUDA/prof/ung_prof.log";
static std::mutex g_prof_mtx;

[[maybe_unused]] inline void prof_log_line(const std::string& line) {
    std::lock_guard<std::mutex> lk(g_prof_mtx);
    std::ofstream ofs(kProfLogPath, std::ios::app);
    ofs << line << '\n';
}

[[maybe_unused]] inline void prof_logf(const char* fmt, ...) {
    char buf[1024];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    prof_log_line(buf);
}

inline int read_env_int(const char* key, int fallback, int min_v, int max_v) {
    const char* s = std::getenv(key);
    if (!s || !*s) return fallback;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return fallback;
    if (v < (long)min_v) v = (long)min_v;
    if (v > (long)max_v) v = (long)max_v;
    return (int)v;
}

// ============================================================
// 2) Q 与输出的复用缓存（Pinned Host + Device）
//    为什么要复用：避免每次调用都 malloc/free，尤其在循环里会非常慢
//
//    g_h_* 是 pinned host buffer：H2D / D2H 更快
//    g_d_* 是 device buffer：GPU 上真正计算用
// ============================================================
static float* g_h_Q   = nullptr;  // pinned：存放所有 query 向量（flatten 后）
static int*   g_h_idx = nullptr;  // pinned：topk 的索引输出（回传）
static float* g_h_dis = nullptr;  // pinned：topk 的距离输出（回传）

static float* g_d_Q   = nullptr;  // device：Q 的 GPU 拷贝
static int*   g_d_idx = nullptr;  // device：topk idx
static float* g_d_dis = nullptr;  // device：topk dist

// host/device 分开记录容量，避免 host 扩容掩盖 device 扩容需求
static size_t g_host_cap_nq = 0;
static int    g_host_cap_k = 0;
static int    g_host_dim_cap = 0;

static size_t g_dev_cap_nq = 0;
static int    g_dev_cap_k = 0;
static int    g_dev_dim_cap = 0;
static uint32_t* g_d_q_ids = nullptr; // device：flatten 后 query 对应的全局向量 id
static size_t g_qid_cap_nq = 0;
static uint32_t* g_d_singleton_qi = nullptr;   // device：singleton query 的 flatten 索引
static uint32_t* g_d_singleton_xoff = nullptr; // device：singleton 对应的全量 X 偏移
static size_t g_singleton_cap_nq = 0;
static UngGroupQueryDesc* g_d_group_desc = nullptr; // device：小/中组批处理描述符
static size_t g_group_desc_cap = 0;

// 申请/扩容 pinned host buffer：Q、idx、dist
inline void ensure_host_q_buffers(size_t need_nq, int need_k, int dim, bool need_q) {
    bool need = (need_nq > g_host_cap_nq) || (need_k > g_host_cap_k) || (dim != g_host_dim_cap)
             || (need_q && g_h_Q == nullptr);
    if (!need) return;

    // 释放旧 buffer
    if (g_h_Q)   { cudaFreeHost(g_h_Q); g_h_Q = nullptr; }
    if (g_h_idx) cudaFreeHost(g_h_idx);
    if (g_h_dis) cudaFreeHost(g_h_dis);

    // 扩容策略：每次至少翻倍 +1，减少频繁重分配
    g_host_cap_nq = std::max(need_nq, g_host_cap_nq * 2 + 1);
    g_host_cap_k = std::max(need_k, g_host_cap_k * 2 + 1);
    g_host_dim_cap = dim;

    // pinned host alloc
    if (need_q) {
        cudaHostAlloc(&g_h_Q, (size_t)g_host_cap_nq * dim * sizeof(float), cudaHostAllocDefault);
    }
    cudaHostAlloc(&g_h_idx, (size_t)g_host_cap_nq * g_host_cap_k * sizeof(int),   cudaHostAllocDefault);
    cudaHostAlloc(&g_h_dis, (size_t)g_host_cap_nq * g_host_cap_k * sizeof(float), cudaHostAllocDefault);
}

// 申请/扩容 device buffer：Q、idx、dist
inline void ensure_device_q_buffers(size_t need_nq, int need_k, int dim) {
    bool need = (need_nq > g_dev_cap_nq) || (need_k > g_dev_cap_k) || (dim != g_dev_dim_cap)
             || (g_d_Q == nullptr) || (g_d_idx == nullptr) || (g_d_dis == nullptr);
    if (!need) return;

    g_dev_cap_nq = std::max(need_nq, g_dev_cap_nq * 2 + 1);
    g_dev_cap_k = std::max(need_k, g_dev_cap_k * 2 + 1);
    g_dev_dim_cap = dim;

    // 释放旧 device buffer
    if (g_d_Q)   cudaFree(g_d_Q);
    if (g_d_idx) cudaFree(g_d_idx);
    if (g_d_dis) cudaFree(g_d_dis);

    cudaMalloc(&g_d_Q,   (size_t)g_dev_cap_nq * dim * sizeof(float));
    cudaMalloc(&g_d_idx, (size_t)g_dev_cap_nq * g_dev_cap_k * sizeof(int));
    cudaMalloc(&g_d_dis, (size_t)g_dev_cap_nq * g_dev_cap_k * sizeof(float));
}

inline void ensure_device_qid_buffer(size_t need_nq) {
    if (need_nq <= g_qid_cap_nq && g_d_q_ids) return;
    if (g_d_q_ids) cudaFree(g_d_q_ids);
    g_qid_cap_nq = std::max(need_nq, g_qid_cap_nq * 2 + 1);
    cudaMalloc(&g_d_q_ids, (size_t)g_qid_cap_nq * sizeof(uint32_t));
}

inline void ensure_device_singleton_buffers(size_t need_nq) {
    if (need_nq <= g_singleton_cap_nq && g_d_singleton_qi && g_d_singleton_xoff) return;
    if (g_d_singleton_qi) cudaFree(g_d_singleton_qi);
    if (g_d_singleton_xoff) cudaFree(g_d_singleton_xoff);
    g_singleton_cap_nq = std::max(need_nq, g_singleton_cap_nq * 2 + 1);
    cudaMalloc(&g_d_singleton_qi, (size_t)g_singleton_cap_nq * sizeof(uint32_t));
    cudaMalloc(&g_d_singleton_xoff, (size_t)g_singleton_cap_nq * sizeof(uint32_t));
}

inline void ensure_device_group_desc_buffer(size_t need_desc) {
    if (need_desc <= g_group_desc_cap && g_d_group_desc) return;
    if (g_d_group_desc) cudaFree(g_d_group_desc);
    g_group_desc_cap = std::max(need_desc, g_group_desc_cap * 2 + 1);
    cudaMalloc(&g_d_group_desc, (size_t)g_group_desc_cap * sizeof(UngGroupQueryDesc));
}

// ============================================================
// 3) 全量数据集 X 的复用缓存（Pinned Host + Device）
//    g_h_all_X：CPU pinned 版全量向量
//    g_d_all_X：GPU 上的全量向量
//    目的：build_cross_group_edges 会多次用到 base vectors
// ============================================================
static float* g_h_all_X = nullptr;  // pinned host：全量向量
static float* g_d_all_X = nullptr;  // device：全量向量
static size_t g_all_cap_n = 0;      // 能容纳的最大 total_points
static const float* g_hostreg_all_x_ptr = nullptr; // 直拷路径：已注册的 host 基地址
static size_t g_hostreg_all_x_bytes = 0;

// ============================================================
// [新增补充] 3.5) 全量数据集 X 的 L2 norm^2（Device 常驻）
// 目的：避免每个 target group 都重新计算 xnorm，11万组会有巨大 kernel 启动开销。
// 只要全量向量不变，这个 norm^2 也不变，所以应当跟 g_d_all_X 一样缓存复用。
// ============================================================
static float* g_d_all_norm = nullptr;   // device：全量向量的 norm^2
static size_t g_all_norm_cap_n = 0;     // 能容纳的最大 total_points（与 g_all_cap_n 类似）

inline void ensure_all_norm(size_t need_n) {
    if (need_n <= g_all_norm_cap_n && g_d_all_norm) return;
    if (g_d_all_norm) cudaFree(g_d_all_norm);
    g_all_norm_cap_n = std::max(need_n, g_all_norm_cap_n * 2 + 1);
    cudaMalloc(&g_d_all_norm, g_all_norm_cap_n * sizeof(float));
}

// ============================================================
// 4) GEMM 需要的额外缓冲区（device）
//    使用公式：||q-x||^2 = ||q||^2 + ||x||^2 - 2 * (q·x)
//
//    g_d_q_norm：每个 query 的 L2 范数平方
//    g_d_x_norm：每个 X（目标组）向量的 L2 范数平方
//    g_d_dot   ：保存 dot = Q * X_tile^T 的结果（分块流式）
// ============================================================
static float* g_d_q_norm = nullptr;   // [max nq]
static size_t g_qnorm_cap = 0;

static float* g_d_x_norm = nullptr;   // [max nx]
static size_t g_xnorm_cap = 0;

static float* g_d_dot = nullptr;      // [max nq * tile_nx]
static size_t g_dot_cap = 0;

// cublasLt matmul workspace（device，复用）
static void*  g_d_lt_workspace = nullptr;
static size_t g_lt_workspace_cap = 0;

inline void ensure_qnorm(size_t need) {
    if (need <= g_qnorm_cap) return;
    if (g_d_q_norm) cudaFree(g_d_q_norm);
    g_qnorm_cap = std::max(need, g_qnorm_cap * 2 + 1);
    cudaMalloc(&g_d_q_norm, g_qnorm_cap * sizeof(float));
}
[[maybe_unused]] inline void ensure_xnorm(size_t need) {
    if (need <= g_xnorm_cap) return;
    if (g_d_x_norm) cudaFree(g_d_x_norm);
    g_xnorm_cap = std::max(need, g_xnorm_cap * 2 + 1);
    cudaMalloc(&g_d_x_norm, g_xnorm_cap * sizeof(float));
}
inline void ensure_dot(size_t need_floats) {
    if (need_floats <= g_dot_cap) return;
    if (g_d_dot) cudaFree(g_d_dot);
    g_dot_cap = std::max(need_floats, g_dot_cap * 2 + 1);
    cudaMalloc(&g_d_dot, g_dot_cap * sizeof(float));
}
inline void ensure_lt_workspace(size_t need_bytes) {
    if (need_bytes <= g_lt_workspace_cap && g_d_lt_workspace) return;
    if (g_d_lt_workspace) cudaFree(g_d_lt_workspace);
    g_lt_workspace_cap = std::max(need_bytes, g_lt_workspace_cap * 2 + 1);
    cudaMalloc(&g_d_lt_workspace, g_lt_workspace_cap);
}

// ============================================================
// 5) cuBLAS / cuBLASLt 句柄（懒初始化，只创建一次）
//    - g_cublas 用来设置 math mode（TF32）
//    - g_cublasLt 用来调用 cublasLtMatmul（更灵活）
// ============================================================
static cublasHandle_t   g_cublas = nullptr;
static cublasLtHandle_t g_cublasLt = nullptr;

inline void ensure_cublas() {
    static bool inited = false;
    if (inited) return;

    cublasCreate(&g_cublas);
    cublasLtCreate(&g_cublasLt);

    // 为了速度：允许 TF32（Ampere 及以后通常更快）
    cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);

    inited = true;
}

} // namespace

// Global-scope naive CUDA kernel.
// Keeping this outside anonymous namespace avoids CUDA runtime symbol lookup issues
// observed with internal-linkage kernels under this build system.
extern "C" __global__ void ung_dot_batched_naive_global_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ Xtiles,
    int nq_chunk,
    int tile,
    int dim,
    int num_tiles,
    int q_base,
    int nq_total,
    int stride_x,
    int use_shared_x,
    int use_vec4,
    int x_cols_per_block,
    float* __restrict__ dot_batches)
{
    // Warp-specialized mapping:
    // - blockDim.x 固定 32（一个 warp）
    // - blockDim.y 表示同一 CTA 内并行处理多少个 query 行
    // - 每个 warp 负责一个 (qi, j)，lane 维度并行累加 dim，再做 warp reduce
    const int lane = threadIdx.x;  // 0..31
    const int qi_local = blockIdx.y * blockDim.y + threadIdx.y;
    const int j_base = blockIdx.x * x_cols_per_block;
    const int tb = blockIdx.z;

    if (tb >= num_tiles || j_base >= tile) return;
    int cols = x_cols_per_block;
    if (j_base + cols > tile) cols = tile - j_base;

    const bool qi_valid = (qi_local < nq_chunk);
    const int qi = q_base + qi_local;
    const float* x0 = Xtiles + (size_t)tb * (size_t)stride_x + (size_t)j_base * dim;
    const float* x1 = (cols > 1) ? (x0 + (size_t)dim) : nullptr;
    const float* x2 = (cols > 2) ? (x1 + (size_t)dim) : nullptr;
    const float* x3 = (cols > 3) ? (x2 + (size_t)dim) : nullptr;

    extern __shared__ float s_x[];
    if (use_shared_x) {
        const int t = threadIdx.y * blockDim.x + lane;
        const int tstride = blockDim.x * blockDim.y;
        if (cols > 0) {
            float* sx = s_x;
            for (int d = t; d < dim; d += tstride) sx[d] = x0[d];
        }
        if (cols > 1) {
            float* sx = s_x + (size_t)dim;
            for (int d = t; d < dim; d += tstride) sx[d] = x1[d];
        }
        if (cols > 2) {
            float* sx = s_x + (size_t)2 * (size_t)dim;
            for (int d = t; d < dim; d += tstride) sx[d] = x2[d];
        }
        if (cols > 3) {
            float* sx = s_x + (size_t)3 * (size_t)dim;
            for (int d = t; d < dim; d += tstride) sx[d] = x3[d];
        }
        __syncthreads();
    }

    if (!qi_valid || qi >= nq_total) return;
    const float* q = Q + (size_t)qi * dim;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    if (use_vec4 && !use_shared_x && (dim % 4 == 0) && cols == 1) {
        const int dim4 = dim >> 2;
        const float4* q4 = reinterpret_cast<const float4*>(q);
        const float4* x4 = reinterpret_cast<const float4*>(x0);
        for (int d4 = lane; d4 < dim4; d4 += 32) {
            float4 qv = q4[d4];
            float4 xv = x4[d4];
            acc0 += qv.x * xv.x + qv.y * xv.y + qv.z * xv.z + qv.w * xv.w;
        }
    } else {
        for (int d = lane; d < dim; d += 32) {
            const float qv = q[d];
            if (cols > 0) {
                const float xv = use_shared_x ? s_x[d] : x0[d];
                acc0 += qv * xv;
            }
            if (cols > 1) {
                const float xv = use_shared_x ? s_x[(size_t)dim + d] : x1[d];
                acc1 += qv * xv;
            }
            if (cols > 2) {
                const float xv = use_shared_x ? s_x[(size_t)2 * (size_t)dim + d] : x2[d];
                acc2 += qv * xv;
            }
            if (cols > 3) {
                const float xv = use_shared_x ? s_x[(size_t)3 * (size_t)dim + d] : x3[d];
                acc3 += qv * xv;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffff, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffff, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffff, acc3, offset);
    }

    if (lane == 0) {
        const size_t base = (size_t)tb * (size_t)nq_total * (size_t)tile + (size_t)qi * (size_t)tile + (size_t)j_base;
        dot_batches[base] = acc0;
        if (cols > 1) dot_batches[base + 1] = acc1;
        if (cols > 2) dot_batches[base + 2] = acc2;
        if (cols > 3) dot_batches[base + 3] = acc3;
    }
}

extern "C" __global__ void ung_gather_q_by_id_global_kernel(
    const float* __restrict__ all_x,
    const uint32_t* __restrict__ q_ids,
    int nq,
    int dim,
    float* __restrict__ out_q)
{
    const long long total = (long long)nq * (long long)dim;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        int qi = (int)(idx / dim);
        int d = (int)(idx - (long long)qi * dim);
        uint32_t src = q_ids[qi];
        out_q[idx] = all_x[(size_t)src * (size_t)dim + (size_t)d];
    }
}

extern "C" __global__ void ung_singleton_top1_global_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ q_norm,
    const float* __restrict__ all_x,
    const float* __restrict__ all_norm,
    const uint32_t* __restrict__ singleton_qi,
    const uint32_t* __restrict__ singleton_xoff,
    int singleton_nq,
    int dim,
    int topk,
    int* __restrict__ out_idx,
    float* __restrict__ out_dist)
{
    int sid = blockIdx.x;
    if (sid >= singleton_nq) return;
    uint32_t qi = singleton_qi[sid];
    uint32_t xoff = singleton_xoff[sid];

    const float* q = Q + (size_t)qi * (size_t)dim;
    const float* x = all_x + (size_t)xoff * (size_t)dim;
    float acc = 0.f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        acc += q[d] * x[d];
    }

    __shared__ float sm[256];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float dist = q_norm[qi] + all_norm[xoff] - 2.f * sm[0];
        size_t base = (size_t)qi * (size_t)topk;
        out_idx[base] = 0;
        out_dist[base] = dist;
    }
}

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

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;

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
            s_dist[j] = q_norm[qi] + x_norm[j] - 2.f * acc;
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

extern "C" __global__ void ung_group_desc_topk_fused_global_kernel(
    const UngGroupQueryDesc* __restrict__ descs, // [num_desc]
    int num_desc,
    const float* __restrict__ Q,                 // [total_queries, dim]
    const float* __restrict__ q_norm,            // [total_queries]
    const float* __restrict__ all_x,             // [total_points, dim]
    const float* __restrict__ all_norm,          // [total_points]
    int dim,
    int topk,
    int max_nx,
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

    const float* q = Q + (size_t)qi * (size_t)dim;
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
            s_dist[j] = q_norm[qi] + x_norm[j] - 2.f * acc;
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

// ============================================================
// 6) 小 kernel：计算范数、初始化 topk、从 dot tile 更新 topk
// ============================================================
namespace {

// 计算 A[i] 的 L2 范数平方：sum_d A[i,d]^2
// A: [n, dim] 行主序
// out: [n]
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

    // 每个 warp 负责一个向量，降低 block 启动数量与同步开销。
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

// 初始化 topk 输出数组为：idx=-1, dist=INF
// out_idx/out_dist: [nq, topk] 行主序
__global__ void init_topk_kernel(int nq, int topk, int* out_idx, float* out_dist) {
    const long long total = (long long)nq * (long long)topk;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)blockDim.x * gridDim.x) {
        out_idx[(size_t)idx] = -1;
        out_dist[(size_t)idx] = FLT_MAX;
    }
}

// 把一个候选 (dist, idx) 插入到“已排序”的 topK 数组里（线性插入）
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

// 从 GEMM 得到的 dot tile 更新 topK：
// 输入：dot[qi, j] = q · x_tile[j]
// 根据公式：dist = ||q||^2 + ||x||^2 - 2*dot
//
// dot:      [nq, tile] 行主序（tile 为当前 X 分块列数）
// qn:       [nq] query norm^2（注意这里传的是 g_d_q_norm + q_start）
// xn_tile:  [tile] 当前 X 分块的 norm^2（注意这里传的是 g_d_x_norm + xb）
// best_idx/best_dist：全程维护的 topK（running topK）
//
// 注意：这个 kernel 的实现思路：
// - 每个 block 负责一个 query（qi）
// - block 内每个 thread 扫自己负责的一部分列，先维护线程私有 topK
// - 然后把所有线程的候选写到 shared memory，thread0 做一次“全局 topK选择”
// - 再把这些候选 merge 进 running topK，写回 global memory
__global__ void update_topk_from_dot_tile_kernel(
    const float* __restrict__ dot,      // [nq, tile]
    const float* __restrict__ qn,       // [nq]
    const float* __restrict__ xn_tile,  // [tile]
    int nq,
    int tile,
    int topk,
    int x_base,                         // 当前 tile 在 target group 内的起始下标（用于变成全局 local idx）
    int* __restrict__ best_idx,         // [nq, topk] running
    float* __restrict__ best_dist)      // [nq, topk] running
{
    int qi = blockIdx.x;
    if (qi >= nq) return;

    // 你说 topk<=32，一般这里非常快；保留 KMAX=64 只是防御
    constexpr int KMAX = 64;
    int K = topk;
    if (K > KMAX) K = KMAX;

    // thread0 把已有 best 读到寄存器（running topK）
    float cur_best[KMAX];
    int   cur_idx[KMAX];
    if (threadIdx.x == 0) {
        for (int k = 0; k < K; ++k) {
            cur_best[k] = best_dist[(size_t)qi * topk + k];
            cur_idx[k]  = best_idx[(size_t)qi * topk + k];
        }
        for (int k = K; k < KMAX; ++k) {
            cur_best[k] = FLT_MAX; cur_idx[k] = -1;
        }
    }
    __syncthreads();

    // 每个 thread 自己维护一个 local topK（来自它扫描到的列）
    float local_best[KMAX];
    int   local_idx[KMAX];
    for (int k = 0; k < K; ++k) { local_best[k] = FLT_MAX; local_idx[k] = -1; }
    for (int k = K; k < KMAX; ++k) { local_best[k] = FLT_MAX; local_idx[k] = -1; }

    float qnorm = qn[qi];
    const float* row = dot + (size_t)qi * tile;

    // 扫描 tile 的列（分配给不同线程）
    for (int j = threadIdx.x; j < tile; j += blockDim.x) {
        float d = qnorm + xn_tile[j] - 2.0f * row[j];
        topk_insert_linear(local_best, local_idx, K, d, x_base + j);
    }

    // 把每个线程的 local topK 写到 shared，便于 thread0 再选一次全局 topK
    extern __shared__ unsigned char smem[];
    float* sdist = reinterpret_cast<float*>(smem);
    int*   sidx  = reinterpret_cast<int*>(sdist + blockDim.x * K);

    for (int k = 0; k < K; ++k) {
        int off = threadIdx.x * K + k;
        sdist[off] = local_best[k];
        sidx[off]  = local_idx[k];
    }
    __syncthreads();

    // thread0：从 blockDim*K 个候选里选出最小的 K 个，然后 merge 到 running topK
    if (threadIdx.x == 0) {
        int total = blockDim.x * K;

        // 简单 selection：把最小 K 个放到前 K 位
        for (int r = 0; r < K; ++r) {
            int best = r;
            for (int t = r + 1; t < total; ++t) {
                if (sdist[t] < sdist[best]) best = t;
            }
            if (best != r) {
                float td = sdist[best]; sdist[best] = sdist[r]; sdist[r] = td;
                int   ti = sidx[best];  sidx[best]  = sidx[r];  sidx[r]  = ti;
            }
        }

        // 把本 tile 的候选 merge 到 running topK
        for (int r = 0; r < K; ++r) {
            if (sidx[r] < 0) continue;
            topk_insert_linear(cur_best, cur_idx, K, sdist[r], sidx[r]);
        }

        // 写回 global memory
        for (int k = 0; k < K; ++k) {
            best_dist[(size_t)qi * topk + k] = cur_best[k];
            best_idx[(size_t)qi * topk + k]  = cur_idx[k];
        }
        // topk>64 的极端情况，补齐
        for (int k = K; k < topk; ++k) {
            best_dist[(size_t)qi * topk + k] = FLT_MAX;
            best_idx[(size_t)qi * topk + k]  = -1;
        }
    }
}

// 与 update_topk_from_dot_tile_kernel 语义一致，但一次处理多个同形状 tile。
// dot_batches: [num_tiles, nq, tile]，每个 tile 的 dot 子矩阵连续存放。
// xn_group:    [num_tiles * tile]，对应这些 tile 的连续 xnorm 段。
__global__ void update_topk_from_dot_grouped_tiles_kernel(
    const float* __restrict__ dot_batches,   // [num_tiles, nq, tile]
    const float* __restrict__ qn,            // [nq]
    const float* __restrict__ xn_group,      // [num_tiles * tile]
    int nq,
    int tile,
    int num_tiles,
    int topk,
    int x_base,                              // 这一批 tile 在目标组内的起点
    int* __restrict__ best_idx,              // [nq, topk] running
    float* __restrict__ best_dist)           // [nq, topk] running
{
    int qi = blockIdx.x;
    if (qi >= nq) return;

    constexpr int KMAX = 64;
    int K = topk;
    if (K > KMAX) K = KMAX;

    float cur_best[KMAX];
    int   cur_idx[KMAX];
    if (threadIdx.x == 0) {
        for (int k = 0; k < K; ++k) {
            cur_best[k] = best_dist[(size_t)qi * topk + k];
            cur_idx[k]  = best_idx[(size_t)qi * topk + k];
        }
        for (int k = K; k < KMAX; ++k) {
            cur_best[k] = FLT_MAX;
            cur_idx[k]  = -1;
        }
    }
    __syncthreads();

    float local_best[KMAX];
    int   local_idx[KMAX];
    for (int k = 0; k < K; ++k) { local_best[k] = FLT_MAX; local_idx[k] = -1; }
    for (int k = K; k < KMAX; ++k) { local_best[k] = FLT_MAX; local_idx[k] = -1; }

    const float qnorm = qn[qi];
    const int total_cols = tile * num_tiles;
    const size_t dot_stride = (size_t)nq * (size_t)tile;

    // 线性展开 (tile_id, j) -> p，减少 host 侧逐 tile launch。
    for (int p = threadIdx.x; p < total_cols; p += blockDim.x) {
        int tile_id = p / tile;
        int j = p - tile_id * tile;
        const float* row = dot_batches + (size_t)tile_id * dot_stride + (size_t)qi * tile;
        float d = qnorm + xn_group[p] - 2.0f * row[j];
        topk_insert_linear(local_best, local_idx, K, d, x_base + p);
    }

    extern __shared__ unsigned char smem[];
    float* sdist = reinterpret_cast<float*>(smem);
    int*   sidx  = reinterpret_cast<int*>(sdist + blockDim.x * K);

    for (int k = 0; k < K; ++k) {
        int off = threadIdx.x * K + k;
        sdist[off] = local_best[k];
        sidx[off]  = local_idx[k];
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
                float td = sdist[best]; sdist[best] = sdist[r]; sdist[r] = td;
                int   ti = sidx[best];  sidx[best]  = sidx[r];  sidx[r]  = ti;
            }
        }

        for (int r = 0; r < K; ++r) {
            if (sidx[r] < 0) continue;
            topk_insert_linear(cur_best, cur_idx, K, sdist[r], sidx[r]);
        }

        for (int k = 0; k < K; ++k) {
            best_dist[(size_t)qi * topk + k] = cur_best[k];
            best_idx[(size_t)qi * topk + k]  = cur_idx[k];
        }
        for (int k = K; k < topk; ++k) {
            best_dist[(size_t)qi * topk + k] = FLT_MAX;
            best_idx[(size_t)qi * topk + k]  = -1;
        }
    }
}

} // namespace

// ============================================================
// 7) 把全量向量一次性上传到 GPU（复用）
//    注意：这里 CPU 会把每个向量 memcpy 到 pinned buffer
//    如果你未来想进一步优化，可以考虑直接从原始连续内存读取（避免循环 memcpy）
// ============================================================
void ANNS::UniNavGraph::gpu_prepare_all_vectors_on_device(double* h2d_ms, double* /*d2h_ms*/) {
    const IdxType total_points = (IdxType)_base_storage->get_num_points();
    const int dim = _base_storage->get_dim();
    if (total_points <= 0) return;
    const auto t_prepare_start = std::chrono::high_resolution_clock::now();
    const size_t total_bytes = (size_t)total_points * (size_t)dim * sizeof(float);
    const size_t stride_bytes = (size_t)dim * sizeof(float);

    // 优先尝试“连续内存直拷”：避免 host 侧再做一次 400MB 级 memcpy。
    bool use_direct_hostreg = false;
    const float* direct_src = nullptr;
    if (total_points > 0) {
        const float* v0 = reinterpret_cast<const float*>(_base_storage->get_vector(0));
        bool contiguous = true;
        if (total_points > 1) {
            const float* v1 = reinterpret_cast<const float*>(_base_storage->get_vector(1));
            contiguous = (v1 == (v0 + (size_t)dim));
            if (contiguous) {
                // 多点抽样校验：避免“仅前两个向量连续”的误判。
                static const int kProbeCount = 6;
                for (int p = 0; p < kProbeCount; ++p) {
                    IdxType idx = (IdxType)(((uint64_t)(total_points - 1) * (uint64_t)(p + 1)) / (uint64_t)(kProbeCount + 1));
                    if (idx <= 1) continue;
                    const float* vp = reinterpret_cast<const float*>(_base_storage->get_vector(idx));
                    const float* v_expect = reinterpret_cast<const float*>(reinterpret_cast<const char*>(v0) + (size_t)idx * stride_bytes);
                    if (vp != v_expect) {
                        contiguous = false;
                        break;
                    }
                }
                if (contiguous && total_points > 2) {
                    const float* vlast = reinterpret_cast<const float*>(_base_storage->get_vector(total_points - 1));
                    const float* v_expect_last = reinterpret_cast<const float*>(reinterpret_cast<const char*>(v0) + (size_t)(total_points - 1) * stride_bytes);
                    contiguous = (vlast == v_expect_last);
                }
            }
        }
        if (contiguous) {
            if (g_hostreg_all_x_ptr && (g_hostreg_all_x_ptr != v0 || g_hostreg_all_x_bytes != total_bytes)) {
                cudaHostUnregister(const_cast<float*>(g_hostreg_all_x_ptr));
                g_hostreg_all_x_ptr = nullptr;
                g_hostreg_all_x_bytes = 0;
            }
            if (!g_hostreg_all_x_ptr) {
                cudaError_t reg_st = cudaHostRegister(const_cast<float*>(v0), total_bytes, cudaHostRegisterDefault);
                if (reg_st == cudaSuccess) {
                    g_hostreg_all_x_ptr = v0;
                    g_hostreg_all_x_bytes = total_bytes;
                    use_direct_hostreg = true;
                    direct_src = v0;
                } else {
                    // 注册失败不影响功能，回退到 pinned staging 路径。
                    cudaGetLastError();
                }
            } else {
                use_direct_hostreg = true;
                direct_src = v0;
            }
        }
    }

    double host_pack_ms = 0.0;
    if (!use_direct_hostreg) {
        const auto t_pack_start = std::chrono::high_resolution_clock::now();
        // fallback：先拷到 pinned host buffer，再 H2D。
        if (!g_h_all_X || g_all_cap_n < (size_t)total_points) {
            if (g_h_all_X) cudaFreeHost(g_h_all_X);
            g_all_cap_n = std::max((size_t)total_points, g_all_cap_n * 2 + 1);
            cudaHostAlloc(&g_h_all_X, g_all_cap_n * (size_t)dim * sizeof(float), cudaHostAllocDefault);
        }

        #pragma omp parallel for schedule(static, 1024)
        for (IdxType i = 0; i < total_points; ++i) {
            const float* v = reinterpret_cast<const float*>(_base_storage->get_vector(i));
            std::memcpy(g_h_all_X + (size_t)i * dim, v, (size_t)dim * sizeof(float));
        }
        const auto t_pack_end = std::chrono::high_resolution_clock::now();
        host_pack_ms = std::chrono::duration<double, std::milli>(t_pack_end - t_pack_start).count();
    }

    // 3) device buffer：不够就扩容
    if (!g_d_all_X || g_all_cap_n < (size_t)total_points) {
        size_t new_cap_n = std::max((size_t)total_points, g_all_cap_n * 2 + 1);
        if (g_d_all_X) cudaFree(g_d_all_X);
        cudaMalloc(&g_d_all_X, new_cap_n * (size_t)dim * sizeof(float));
        g_all_cap_n = new_cap_n;
    }

    // 4) H2D：只拷贝真实 total_points 部分
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    const float* h_src = use_direct_hostreg ? direct_src : g_h_all_X;
    cudaMemcpy(g_d_all_X, h_src, total_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    if (h2d_ms) *h2d_ms += ms;

    cudaEventDestroy(e0); cudaEventDestroy(e1);

    // ============================================================
    // [新增补充] 7.5) 一次性计算“全量向量”的 norm^2，并缓存到 g_d_all_norm
    // 这样后续每个 target group 就不需要再对它的 X 重复跑 l2_norm_sq_kernel<<<nx>>> 了
    // ============================================================
    ensure_all_norm((size_t)total_points);
    {
        // 这里是 GPU kernel 启动一次（总点数级别），比 11万组每组启动一次 xnorm 要划算得多
        int threads = 256;
        const int warps_per_block = threads / 32;
        dim3 grid((unsigned)(((int)total_points + warps_per_block - 1) / warps_per_block), 1u, 1u);
        dim3 block(threads, 1, 1);
        l2_norm_sq_kernel<<<grid, block>>>(g_d_all_X, (int)total_points, dim, g_d_all_norm);
        // 注意：这里不强制同步，让外部第一次真正用到它时再整体同步即可
        // 如果你希望更稳妥，也可以在这里 cudaDeviceSynchronize();
    }
    const auto t_prepare_end = std::chrono::high_resolution_clock::now();
    const double prepare_ms = std::chrono::duration<double, std::milli>(t_prepare_end - t_prepare_start).count();
    prof_logf("[PROF] cross_edges.prepare_all path=%s host_pack_ms=%.3f h2d_ms=%.3f total_ms=%.3f bytes=%zu",
              use_direct_hostreg ? "direct_hostregister" : "pinned_staging",
              host_pack_ms, (double)ms, prepare_ms, total_bytes);
}

// 释放全量向量缓存（如果你想整个构建结束后释放）
void ANNS::UniNavGraph::gpu_release_all_vectors_on_device() {
    if (g_d_all_X) { cudaFree(g_d_all_X); g_d_all_X = nullptr; }
    if (g_h_all_X) { cudaFreeHost(g_h_all_X); g_h_all_X = nullptr; }
    // cap 不清零，方便下次 reuse（你也可以清零）

    // [新增补充] 同时释放全量 norm 缓存
    if (g_d_all_norm) { cudaFree(g_d_all_norm); g_d_all_norm = nullptr; }
    // 这里也不清零 cap，保持和 all_X 同步的复用思路
    if (g_d_q_ids) { cudaFree(g_d_q_ids); g_d_q_ids = nullptr; }
    if (g_d_singleton_qi) { cudaFree(g_d_singleton_qi); g_d_singleton_qi = nullptr; }
    if (g_d_singleton_xoff) { cudaFree(g_d_singleton_xoff); g_d_singleton_xoff = nullptr; }
    if (g_d_group_desc) { cudaFree(g_d_group_desc); g_d_group_desc = nullptr; }
    if (g_hostreg_all_x_ptr) {
        cudaHostUnregister(const_cast<float*>(g_hostreg_all_x_ptr));
        g_hostreg_all_x_ptr = nullptr;
        g_hostreg_all_x_bytes = 0;
    }
}

// ============================================================
// 8) 主入口：基于 GEMM（分块流式）计算 cross-group neighbors
//
// 目标：对每个 group（target group）
//   - 把它所有 in-neighbors 组中的向量当作 Query（Q）
//   - 在 target group 的向量集合 X 中做 brute-force topK
//   - 结果写回 cross_group_neighbors[qid].insert(target_vid, dist)
//
// 关键优化点：
//   - 距离计算用 GEMM：dot = Q * X^T
//   - 分块 tile X，避免一次性 dot 矩阵太大
//   - topk<=32 的情况下，用“running topK + 每 tile 更新”很合适
// ============================================================
void ANNS::UniNavGraph::gpu_cross_groups_search_all_batched(
    const std::vector<IdxType>& target_group_ids,
    int dim,
    int topk,
    std::vector<SearchQueue>& cross_group_neighbors,
    double* h2d_ms, double* kernel_ms, double* d2h_ms)
{
    using namespace std;
    if (target_group_ids.empty()) return;
    const auto t_func_start = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = [](const std::chrono::high_resolution_clock::time_point& t0,
                         const std::chrono::high_resolution_clock::time_point& t1) -> double {
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    // ------------------------------------------------------------
    // 第一步：统计每个 target group 需要处理多少个 query，并做分流
    // - target_counts_raw: 原始 query 数
    // - target_counts:     真正走 GPU 的 query 数（CPU tiny 组会被置 0）
    // ------------------------------------------------------------
    vector<size_t> target_counts_raw(target_group_ids.size(), 0);
    vector<size_t> target_counts(target_group_ids.size(), 0);
    vector<int> target_group_nx(target_group_ids.size(), 0);
    vector<int> cpu_tiny_group_indices;
    const auto t_count_start = std::chrono::high_resolution_clock::now();

    for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
        IdxType tgt = target_group_ids[gi];
        const auto& tgt_rng = _group_id_to_range[tgt];
        target_group_nx[gi] = (int)(tgt_rng.second - tgt_rng.first);
        size_t cnt = 0;

        // 规则：Q 来自 LNG 的 in_neighbors[tgt] 中所有向量
        for (auto in_gid : _label_nav_graph->in_neighbors[tgt]) {
            const auto& rng = _group_id_to_range[in_gid];
            cnt += (size_t)(rng.second - rng.first);
        }

        target_counts_raw[gi] = cnt;
    }
    const auto t_count_end = std::chrono::high_resolution_clock::now();

    // 轻量组 CPU 路径（可选）：避免为极小工作量组走 GPU launch/copy
    const bool cpu_tiny_groups = (read_env_int("UNG_CPU_TINY_GROUPS", 0, 0, 1) == 1);
    const int cpu_tiny_nx_max = read_env_int("UNG_CPU_TINY_NX_MAX", 8, 2, 64);
    const int cpu_tiny_nq_max = read_env_int("UNG_CPU_TINY_NQ_MAX", 64, 1, 4096);
    const long long cpu_tiny_ops_thresh = (long long)read_env_int("UNG_CPU_TINY_OPS", 65536, 1024, 1073741824);
    size_t cpu_tiny_group_count = 0;
    size_t cpu_tiny_query_count = 0;
    size_t total_queries_sz = 0;
    cpu_tiny_group_indices.reserve(target_group_ids.size() / 8 + 1);

    for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
        const size_t cnt = target_counts_raw[gi];
        const int nx = target_group_nx[gi];
        const unsigned long long est_work = (unsigned long long)cnt * (unsigned long long)std::max(nx, 0) * (unsigned long long)dim;
        const bool route_cpu_tiny =
            cpu_tiny_groups &&
            nx > 1 &&
            nx <= cpu_tiny_nx_max &&
            cnt > 0 &&
            (int)cnt <= cpu_tiny_nq_max &&
            est_work <= (unsigned long long)cpu_tiny_ops_thresh;

        if (route_cpu_tiny) {
            cpu_tiny_group_indices.push_back((int)gi);
            cpu_tiny_group_count += 1;
            cpu_tiny_query_count += cnt;
            target_counts[gi] = 0;
        } else {
            target_counts[gi] = cnt;
            total_queries_sz += cnt;
        }
    }

    int total_queries = (int)total_queries_sz;

    // ------------------------------------------------------------
    // 第二步：为每个 target group 分配 Q 的连续区间（prefix offsets）
    // target_offsets[gi]：这个 group 的 Q 在 flatten 后数组中的起点
    // ------------------------------------------------------------
    vector<size_t> target_offsets(target_group_ids.size(), 0);
    size_t acc = 0;
    const auto t_offset_start = std::chrono::high_resolution_clock::now();
    for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
        target_offsets[gi] = acc;
        acc += target_counts[gi];
    }
    const auto t_offset_end = std::chrono::high_resolution_clock::now();

    // 轻量组 CPU 回退：用于把超小工作量组从 GPU 路径分流
    double cpu_tiny_ms = 0.0;
    auto process_cpu_tiny_groups = [&]() {
        if (!cpu_tiny_groups || cpu_tiny_group_indices.empty()) return;
        const auto t0 = std::chrono::high_resolution_clock::now();

        auto host_topk_insert = [](float* best_dist, int* best_idx, int K, float dist, int idx) {
            if (dist >= best_dist[K - 1]) return;
            int pos = K - 1;
            while (pos > 0 && dist < best_dist[pos - 1]) {
                best_dist[pos] = best_dist[pos - 1];
                best_idx[pos] = best_idx[pos - 1];
                --pos;
            }
            best_dist[pos] = dist;
            best_idx[pos] = idx;
        };

        for (int gi_i : cpu_tiny_group_indices) {
            const size_t gi = (size_t)gi_i;
            const IdxType tgt_gid = target_group_ids[gi];
            const auto& tgt_rng = _group_id_to_range[tgt_gid];
            const int nx = target_group_nx[gi];
            if (nx <= 0) continue;
            const int valid_k = std::min<int>(topk, nx);
            if (valid_k <= 0) continue;

            std::vector<const float*> x_ptrs((size_t)nx);
            for (int j = 0; j < nx; ++j) {
                x_ptrs[(size_t)j] = reinterpret_cast<const float*>(_base_storage->get_vector(tgt_rng.first + (IdxType)j));
            }

            for (auto in_gid : _label_nav_graph->in_neighbors[tgt_gid]) {
                const auto& qrng = _group_id_to_range[in_gid];
                for (IdxType qid = qrng.first; qid < qrng.second; ++qid) {
                    const float* q = reinterpret_cast<const float*>(_base_storage->get_vector(qid));

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
                        host_topk_insert(best_dist, best_idx, K, dist, j);
                    }

                    for (int r = 0; r < K; ++r) {
                        const int j = best_idx[r];
                        if (j < 0) continue;
                        cross_group_neighbors[qid].insert(tgt_rng.first + (IdxType)j, best_dist[r]);
                    }
                }
            }
        }

        const auto t1 = std::chrono::high_resolution_clock::now();
        cpu_tiny_ms = elapsed_ms(t0, t1);
    };

    if (total_queries == 0) {
        process_cpu_tiny_groups();
        prof_logf("[PROF] cross_edges.cpu_tiny groups=%zu queries=%zu ms=%.3f enabled=%d nx_max=%d nq_max=%d ops_thresh=%lld",
                  cpu_tiny_group_count, cpu_tiny_query_count, cpu_tiny_ms, cpu_tiny_groups ? 1 : 0,
                  cpu_tiny_nx_max, cpu_tiny_nq_max, cpu_tiny_ops_thresh);
        return;
    }

    // query 上传模式：0=老路径(Host memcpy + H2D Q)，1=新路径(H2D qid + device gather Q)。
    const bool use_device_query_gather = (read_env_int("UNG_Q_UPLOAD_MODE", 1, 0, 1) == 1);
    const bool singleton_fastpath = (read_env_int("UNG_SINGLETON_FASTPATH", 1, 0, 1) == 1);
    const int gather_threads = read_env_int("UNG_GATHER_THREADS", 256, 64, 512);
    const int gather_blocks = read_env_int("UNG_GATHER_BLOCKS", 4096, 1, 65535);
    size_t singleton_query_count = 0;
    std::vector<uint32_t> singleton_qi;
    std::vector<uint32_t> singleton_xoff;
    if (singleton_fastpath) {
        singleton_qi.reserve((size_t)total_queries / 2 + 1);
        singleton_xoff.reserve((size_t)total_queries / 2 + 1);
        for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
            if (target_group_nx[gi] != 1 || target_counts[gi] == 0) continue;
            const uint32_t xoff = (uint32_t)_group_id_to_range[target_group_ids[gi]].first;
            size_t q0 = target_offsets[gi], q1 = q0 + target_counts[gi];
            for (size_t qi = q0; qi < q1; ++qi) {
                singleton_qi.push_back((uint32_t)qi);
                singleton_xoff.push_back(xoff);
            }
        }
        singleton_query_count = singleton_qi.size();
    }

    // ------------------------------------------------------------
    // 第三步：确保缓存足够大（Q/idx/dist + qnorm）
    // ------------------------------------------------------------
    ensure_host_q_buffers((size_t)total_queries, topk, dim, !use_device_query_gather);
    ensure_device_q_buffers((size_t)total_queries, topk, dim);
    ensure_qnorm((size_t)total_queries);
    if (use_device_query_gather) {
        ensure_device_qid_buffer((size_t)total_queries);
    }
    if (singleton_fastpath && singleton_query_count > 0) {
        ensure_device_singleton_buffers(singleton_query_count);
    }

    // ------------------------------------------------------------
    // 第四步：建立映射（用于写回结果）
    // query_global_ids[qi] = 这个 flatten query 对应原始全局向量 id
    // query_target_index[qi] = 它属于哪个 target group (gi)
    // ------------------------------------------------------------
    vector<IdxType> query_global_ids(total_queries);
    vector<int> query_target_index(total_queries);

    // ------------------------------------------------------------
    // 第五步：填充 pinned Q（flatten）并记录映射
    // 注意：这里是 CPU memcpy 成本（但比逐组多次 H2D 更好）
    // ------------------------------------------------------------
    const auto t_fill_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic, 8)
    for (int gi = 0; gi < (int)target_group_ids.size(); ++gi) {
        if (target_counts[(size_t)gi] == 0) continue;
        size_t write_pos = target_offsets[(size_t)gi];
        IdxType tgt = target_group_ids[(size_t)gi];

        for (auto in_gid : _label_nav_graph->in_neighbors[tgt]) {
            const auto& rng = _group_id_to_range[in_gid];
            for (IdxType vid = rng.first; vid < rng.second; ++vid) {
                if (!use_device_query_gather) {
                    const float* v = reinterpret_cast<const float*>(_base_storage->get_vector(vid));
                    std::memcpy(g_h_Q + (size_t)write_pos * dim, v, (size_t)dim * sizeof(float));
                }

                query_global_ids[(int)write_pos] = vid;
                query_target_index[(int)write_pos] = gi;
                ++write_pos;
            }
        }
    }
    const auto t_fill_end = std::chrono::high_resolution_clock::now();

    // ------------------------------------------------------------
    // 第六步：一次性 H2D 拷贝所有 Q
    // ------------------------------------------------------------
    cudaEvent_t e0,e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    cudaEventRecord(e0);
    if (use_device_query_gather && g_d_all_X != nullptr) {
        cudaMemcpy(g_d_q_ids,
                   query_global_ids.data(),
                   (size_t)total_queries * sizeof(uint32_t),
                   cudaMemcpyHostToDevice);
        ung_gather_q_by_id_global_kernel<<<gather_blocks, gather_threads>>>(
            g_d_all_X, g_d_q_ids, total_queries, dim, g_d_Q);
    } else {
        cudaMemcpy(g_d_Q, g_h_Q, (size_t)total_queries * dim * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float th2d = 0.f;
    cudaEventElapsedTime(&th2d, e0, e1);
    if (h2d_ms) *h2d_ms += th2d;

    cudaEventDestroy(e0); cudaEventDestroy(e1);

    // ------------------------------------------------------------
    // 第七步：GPU 计算阶段计时（norm + GEMM + topK 更新）
    // ------------------------------------------------------------
    cudaEvent_t k0,k1;
    cudaEventCreate(&k0); cudaEventCreate(&k1);
    cudaEventRecord(k0);

    // X 的 tile 大小：越大 GEMM 越高效，但 dot buffer 也越大
    // 这里先设 1024，再根据 nq_g 自动收缩到可接受显存占用
    const int TILE_NX_DEFAULT = 1024;
    // 每个 tile 的 dot 输出容量上限（影响 tile_nx 自适应）
    const size_t DOT_TILE_CAP_BYTES = (size_t)256 * 1024 * 1024;
    // grouped GEMM 的总 dot 容量上限（限制一次批量 tile 数，避免占用过高显存）
    const size_t DOT_GROUP_CAP_BYTES = (size_t)768 * 1024 * 1024;
    const int GEMM_GROUP_TILES_MAX = 16;
    const size_t LT_WORKSPACE_BYTES = (size_t)64 * 1024 * 1024;
    // UNG_GEMM_IMPL: 0=cublasLt, 1=cublasSgemmStridedBatched, 2=naive CUDA core.
    // 默认按当前工作负载做启发式选择：高维且小 topk 场景优先 naive。
    const int gemm_impl_default = (dim >= 256 && topk <= 16) ? 2 : 1;
    const bool force_custom_kernel = (read_env_int("UNG_FORCE_CUSTOM_KERNEL", 1, 0, 1) == 1);
    const int gemm_impl_req = read_env_int("UNG_GEMM_IMPL", gemm_impl_default, 0, 2);
    const int gemm_impl = force_custom_kernel ? 2 : gemm_impl_req;
    const bool use_sgemm_strided = (gemm_impl == 1);
    const bool use_naive_cuda = (gemm_impl == 2);
    const bool use_cublas_lt = (!use_sgemm_strided && !use_naive_cuda);
    // 默认参数使用当前已验证更稳的配置，仍可通过环境变量覆盖。
    const int naive_warps_per_block = read_env_int("UNG_NAIVE_WARPS_PER_BLOCK", 4, 1, 16);
    const bool naive_shared_x = (read_env_int("UNG_NAIVE_SHARED_X", 0, 0, 1) == 1);
    const bool naive_vec4 = (read_env_int("UNG_NAIVE_VEC4", 1, 0, 1) == 1);
    const int naive_x_cols_per_block_req = read_env_int("UNG_NAIVE_X_COLS_PER_BLOCK", 1, 1, 4);
    const int naive_shared_kb = read_env_int("UNG_NAIVE_SHARED_KB", 48, 16, 164);
    const bool naive_heavy_sgemm = (read_env_int("UNG_NAIVE_HEAVY_SGEMM", 0, 0, 1) == 1);
    const int naive_heavy_nx = read_env_int("UNG_NAIVE_HEAVY_NX", 96, 16, 1048576);
    const int naive_heavy_nq = read_env_int("UNG_NAIVE_HEAVY_NQ", 256, 16, 1048576);
    const long long naive_heavy_work_thresh = (long long)read_env_int("UNG_NAIVE_HEAVY_WORK_M", 96, 1, 2000000000) * 1000000LL;
    const int topk_block_threads = read_env_int("UNG_TOPK_BLOCK_THREADS", (topk <= 8 ? 32 : (topk <= 16 ? 64 : 128)), 32, 256);
    const bool naive_group_fused = (read_env_int("UNG_NAIVE_GROUP_FUSED", 0, 0, 1) == 1);
    const int naive_group_threads = read_env_int("UNG_NAIVE_GROUP_THREADS", 128, 64, 256);
    const int singleton_threads = read_env_int("UNG_SINGLETON_THREADS", 256, 64, 512);
    const bool small_group_fused = (read_env_int("UNG_SMALL_GROUP_FUSED", 1, 0, 1) == 1);
    const int small_group_max_nx = read_env_int("UNG_SMALL_GROUP_MAX_NX", 8, 2, 32);
    const int small_group_warps = read_env_int("UNG_SMALL_GROUP_WARPS", 8, 1, 16);
    const bool medium_group_fused = (read_env_int("UNG_MEDIUM_GROUP_FUSED", 1, 0, 1) == 1);
    const int medium_group_max_nx = read_env_int("UNG_MEDIUM_GROUP_MAX_NX", 64, 9, 256);
    const int medium_group_warps = read_env_int("UNG_MEDIUM_GROUP_WARPS", 8, 1, 16);
    const bool bucket_group_fused = (read_env_int("UNG_BUCKET_GROUP_FUSED", 1, 0, 1) == 1);
    const char* gemm_mode = use_sgemm_strided ? "sgemm_strided_batched" : (use_naive_cuda ? "naive_cuda_core" : "cublasLt");
    const bool naive_debug_dot = (read_env_int("UNG_NAIVE_DEBUG_DOT", 0, 0, 1) == 1);
    bool naive_debug_logged = false;
    prof_logf("[PROF] cross_edges.gemm_impl mode=%s(%d) force_custom=%d req_impl=%d", gemm_mode, gemm_impl, force_custom_kernel ? 1 : 0, gemm_impl_req);
    prof_logf("[PROF] cross_edges.topk_kernel_threads value=%d", topk_block_threads);
    if (use_naive_cuda) {
        prof_logf("[PROF] cross_edges.naive_cfg warps_per_block=%d shared_x=%d vec4=%d x_cols=%d shared_kb=%d",
                  naive_warps_per_block, naive_shared_x ? 1 : 0, naive_vec4 ? 1 : 0, naive_x_cols_per_block_req, naive_shared_kb);
        prof_logf("[PROF] cross_edges.naive_heavy_sgemm enabled=%d nx_th=%d nq_th=%d work_th=%lld",
                  naive_heavy_sgemm ? 1 : 0, naive_heavy_nx, naive_heavy_nq, naive_heavy_work_thresh);
    }

    if (!use_naive_cuda || naive_heavy_sgemm) {
        // cuBLAS 路径，或 naive 路径里的重负载分流都需要句柄
        ensure_cublas();
    }
    if (use_cublas_lt) {
        ensure_lt_workspace(LT_WORKSPACE_BYTES);
    }
    size_t tail_gemv_count = 0;
    size_t small_group_fused_group_count = 0;
    size_t small_group_fused_query_count = 0;
    size_t medium_group_fused_group_count = 0;
    size_t medium_group_fused_query_count = 0;
    size_t heavy_sgemm_group_count = 0;
    size_t heavy_sgemm_query_count = 0;
    std::vector<uint8_t> bucket_group_kind(target_group_ids.size(), (uint8_t)0); // 0=none,1=small,2=medium
    std::vector<UngGroupQueryDesc> small_group_descs;
    std::vector<UngGroupQueryDesc> medium_group_descs;

    // qnorm 与 topk 初始化都只和 query 有关，提前全量做一次，避免每个 group 启动一次 kernel
    {
        int threads = 256;
        const int warps_per_block = threads / 32;
        dim3 qnorm_grid((unsigned)((total_queries + warps_per_block - 1) / warps_per_block), 1u, 1u), qnorm_block(threads, 1, 1);
        l2_norm_sq_kernel<<<qnorm_grid, qnorm_block>>>(g_d_Q, total_queries, dim, g_d_q_norm);

        const long long init_total = (long long)total_queries * (long long)topk;
        int init_grid_x = (int)((init_total + threads - 1) / threads);
        if (init_grid_x > 65535) init_grid_x = 65535;
        dim3 init_grid((unsigned)init_grid_x, 1u, 1u), init_block(threads, 1, 1);
        init_topk_kernel<<<init_grid, init_block>>>(total_queries, topk, g_d_idx, g_d_dis);

        if (singleton_fastpath && singleton_query_count > 0 && g_d_all_X != nullptr && g_d_all_norm != nullptr) {
            cudaMemcpy(g_d_singleton_qi,
                       singleton_qi.data(),
                       (size_t)singleton_query_count * sizeof(uint32_t),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(g_d_singleton_xoff,
                       singleton_xoff.data(),
                       (size_t)singleton_query_count * sizeof(uint32_t),
                       cudaMemcpyHostToDevice);
            dim3 sg_grid((unsigned)singleton_query_count, 1u, 1u), sg_block(singleton_threads, 1u, 1u);
            ung_singleton_top1_global_kernel<<<sg_grid, sg_block>>>(
                g_d_Q, g_d_q_norm, g_d_all_X, g_d_all_norm,
                g_d_singleton_qi, g_d_singleton_xoff, (int)singleton_query_count,
                dim, topk, g_d_idx, g_d_dis);
        }
    }

    if (bucket_group_fused &&
        use_naive_cuda &&
        topk <= 32 &&
        g_d_all_norm != nullptr &&
        !naive_group_fused) {
        std::vector<size_t> small_desc_offsets(target_group_ids.size() + 1, 0);
        std::vector<size_t> medium_desc_offsets(target_group_ids.size() + 1, 0);

        for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
            const int nq_g = (int)target_counts[gi];
            if (nq_g <= 0) continue;
            const int nx = target_group_nx[gi];
            if (nx <= 0) continue;
            if (singleton_fastpath && nx == 1) continue;

            bool heavy_route = false;
            if (naive_heavy_sgemm) {
                const unsigned long long work = (unsigned long long)nq_g * (unsigned long long)nx * (unsigned long long)dim;
                heavy_route = (nx >= naive_heavy_nx &&
                               nq_g >= naive_heavy_nq &&
                               work >= (unsigned long long)naive_heavy_work_thresh);
            }
            if (heavy_route) continue;

            if (small_group_fused && nx <= small_group_max_nx) {
                bucket_group_kind[gi] = (uint8_t)1;
                small_desc_offsets[gi + 1] = (size_t)nq_g;
            } else if (medium_group_fused && nx > small_group_max_nx && nx <= medium_group_max_nx) {
                bucket_group_kind[gi] = (uint8_t)2;
                medium_desc_offsets[gi + 1] = (size_t)nq_g;
            }
        }

        for (size_t i = 1; i < small_desc_offsets.size(); ++i) small_desc_offsets[i] += small_desc_offsets[i - 1];
        for (size_t i = 1; i < medium_desc_offsets.size(); ++i) medium_desc_offsets[i] += medium_desc_offsets[i - 1];
        small_group_descs.resize(small_desc_offsets.back());
        medium_group_descs.resize(medium_desc_offsets.back());

        #pragma omp parallel for schedule(static, 256)
        for (int gi_i = 0; gi_i < (int)target_group_ids.size(); ++gi_i) {
            size_t gi = (size_t)gi_i;
            const uint8_t kind = bucket_group_kind[gi];
            if (kind == 0) continue;

            const size_t q_start = target_offsets[gi];
            const uint32_t x_off = (uint32_t)_group_id_to_range[target_group_ids[gi]].first;
            const uint32_t nx = (uint32_t)target_group_nx[gi];
            const int nq_g = (int)target_counts[gi];

            if (kind == 1) {
                const size_t base = small_desc_offsets[gi];
                for (int q_local = 0; q_local < nq_g; ++q_local) {
                    UngGroupQueryDesc desc;
                    desc.qi = (uint32_t)(q_start + (size_t)q_local);
                    desc.x_off = x_off;
                    desc.nx = nx;
                    small_group_descs[base + (size_t)q_local] = desc;
                }
            } else {
                const size_t base = medium_desc_offsets[gi];
                for (int q_local = 0; q_local < nq_g; ++q_local) {
                    UngGroupQueryDesc desc;
                    desc.qi = (uint32_t)(q_start + (size_t)q_local);
                    desc.x_off = x_off;
                    desc.nx = nx;
                    medium_group_descs[base + (size_t)q_local] = desc;
                }
            }
        }
    }

    // ------------------------------------------------------------
    // 逐个 target group 处理：
    //   - 拿到这个 group 的 Q 子区间
    //   - 拿到这个 group 的 X（目标向量集合）
    //   - qnorm、xnorm
    //   - 初始化 running topK
    //   - tile X：GEMM + update topK
    // ------------------------------------------------------------
    for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
        int nq_g = (int)target_counts[gi];
        if (nq_g <= 0) continue;

        size_t q_start = target_offsets[gi];

        // 目标组 X 在全局向量中的范围
        IdxType tgt_gid = target_group_ids[gi];
        const auto& rng = _group_id_to_range[tgt_gid];
        int x_off = (int)rng.first;
        int nx = target_group_nx[gi];
        if (nx <= 0) continue;
        if (singleton_fastpath && nx == 1) continue;

        bool group_use_sgemm = use_sgemm_strided;
        bool group_use_naive = use_naive_cuda;
        bool group_use_lt = use_cublas_lt;
        if (use_naive_cuda && naive_heavy_sgemm) {
            const unsigned long long work = (unsigned long long)nq_g * (unsigned long long)nx * (unsigned long long)dim;
            if (nx >= naive_heavy_nx &&
                nq_g >= naive_heavy_nq &&
                work >= (unsigned long long)naive_heavy_work_thresh) {
                group_use_sgemm = true;
                group_use_naive = false;
                group_use_lt = false;
                heavy_sgemm_group_count += 1;
                heavy_sgemm_query_count += (size_t)nq_g;
            }
        }

        // dQ 指向本组 Q 子矩阵：[nq_g, dim]
        const float* dQ = g_d_Q + (size_t)q_start * dim;

        // dX 指向目标组 X：[nx, dim]
        const float* dX = g_d_all_X + (size_t)x_off * dim;
        const float* dXnorm_group = (g_d_all_norm ? (g_d_all_norm + (size_t)x_off) : nullptr);

        // 本组 topK 输出在全局输出数组中的位置（也是一个连续子区间）
        int*   dBestI = g_d_idx + (size_t)q_start * topk;
        float* dBestD = g_d_dis + (size_t)q_start * topk;

        // 1) 目标组 X 的 norm^2（写到 g_d_x_norm[0..nx-1]）
        // [改动点] 这里不再对每个 group 重算 xnorm，而是直接使用全量缓存 g_d_all_norm 的对应切片
        // 原来的 per-group xnorm kernel 被移除，避免 11 万组重复 kernel 启动开销

        if (group_use_naive && naive_group_fused && topk <= 32 && nx <= 1024) {
            size_t fused_smem = (size_t)dim * sizeof(float)
                              + (size_t)naive_group_threads * (size_t)topk * (sizeof(float) + sizeof(int));
            ung_group_fused_topk_global_kernel<<<(unsigned)nq_g, (unsigned)naive_group_threads, fused_smem>>>(
                dQ,
                g_d_q_norm + q_start,
                dX,
                dXnorm_group,
                nq_g,
                nx,
                dim,
                topk,
                dBestI,
                dBestD);
            continue;
        }

        if (group_use_naive && small_group_fused && nx <= small_group_max_nx && topk <= 32 && dXnorm_group != nullptr) {
            if (bucket_group_fused && bucket_group_kind[gi] == 1) {
                small_group_fused_group_count += 1;
                small_group_fused_query_count += (size_t)nq_g;
                continue;
            }
            int launch_warps = std::max(nx, small_group_warps);
            if (launch_warps > 16) launch_warps = 16;
            dim3 sg_grid((unsigned)nq_g, 1u, 1u);
            dim3 sg_block((unsigned)(launch_warps * 32), 1u, 1u);
            size_t sg_smem = (size_t)dim * sizeof(float)
                           + (size_t)small_group_max_nx * (sizeof(float) + sizeof(int));
            ung_small_group_topk_fused_global_kernel<<<sg_grid, sg_block, sg_smem>>>(
                dQ,
                g_d_q_norm + q_start,
                dX,
                dXnorm_group,
                nq_g,
                nx,
                dim,
                topk,
                small_group_max_nx,
                dBestI,
                dBestD);
            small_group_fused_group_count += 1;
            small_group_fused_query_count += (size_t)nq_g;
            continue;
        }

        if (group_use_naive && medium_group_fused && nx > small_group_max_nx &&
            nx <= medium_group_max_nx && topk <= 32 && dXnorm_group != nullptr) {
            if (bucket_group_fused && bucket_group_kind[gi] == 2) {
                medium_group_fused_group_count += 1;
                medium_group_fused_query_count += (size_t)nq_g;
                continue;
            }
            int launch_warps = std::min(medium_group_warps, 16);
            if (launch_warps < 1) launch_warps = 1;
            dim3 mg_grid((unsigned)nq_g, 1u, 1u);
            dim3 mg_block((unsigned)(launch_warps * 32), 1u, 1u);
            size_t mg_smem = (size_t)dim * sizeof(float)
                           + (size_t)medium_group_max_nx * (sizeof(float) + sizeof(int));
            ung_medium_group_topk_fused_global_kernel<<<mg_grid, mg_block, mg_smem>>>(
                dQ,
                g_d_q_norm + q_start,
                dX,
                dXnorm_group,
                nq_g,
                nx,
                dim,
                topk,
                medium_group_max_nx,
                dBestI,
                dBestD);
            medium_group_fused_group_count += 1;
            medium_group_fused_query_count += (size_t)nq_g;
            continue;
        }

        // 2) 决定 tile_nx（避免 dot = nq_g*tile_nx 太大）
        int tile_nx = TILE_NX_DEFAULT;

        // 默认 cap dot buffer 约 256MB（你可以按显存改）
        {
            size_t bytes = (size_t)nq_g * (size_t)tile_nx * sizeof(float);
            while (bytes > DOT_TILE_CAP_BYTES && tile_nx > 128) {
                tile_nx /= 2;
                bytes = (size_t)nq_g * (size_t)tile_nx * sizeof(float);
            }
        }
        size_t tile_bytes = (size_t)nq_g * (size_t)tile_nx * sizeof(float);
        int gemm_group_tiles = 1;
        if (tile_bytes > 0) {
            size_t by_cap = DOT_GROUP_CAP_BYTES / tile_bytes;
            if (by_cap == 0) by_cap = 1;
            gemm_group_tiles = (int)std::min<size_t>((size_t)GEMM_GROUP_TILES_MAX, by_cap);
        }
        if (gemm_group_tiles < 1) gemm_group_tiles = 1;
        ensure_dot((size_t)nq_g * (size_t)tile_nx * (size_t)gemm_group_tiles);

        cublasLtMatmulDesc_t opDesc = nullptr;
        cublasLtMatrixLayout_t qDesc = nullptr;
        int32_t order = CUBLASLT_ORDER_ROW;
        void* workspace = nullptr;
        size_t workspaceSize = 0;
        if (group_use_lt) {
            // ------------------------------------------------------------
            // cuBLASLt 描述符（描述矩阵形状、转置、布局等）
            // 我们要计算：
            //   dot = Q * X_tile^T
            // 其中：
            //   Q:      [nq_g, dim]  row-major
            //   X_tile: [cur,  dim]  row-major
            //   dot:    [nq_g, cur]  row-major
            //
            // 注意：这里用 cublasLt 的 row-major 支持（ORDER_ROW）
            // ld（leading dimension）对 row-major 来说就是“列数”
            // ------------------------------------------------------------
            cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

            cublasOperation_t opA = CUBLAS_OP_N; // Q 不转置
            cublasOperation_t opB = CUBLAS_OP_T; // X_tile 做转置（变成 [dim, cur]）
            cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

            // Q layout： [nq_g, dim]
            cublasLtMatrixLayoutCreate(&qDesc, CUDA_R_32F, nq_g, dim, dim);
            cublasLtMatrixLayoutSetAttribute(qDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

            // cublasLt 使用固定 workspace；algo 仍走默认路径，避免 per-group heuristic 查询开销
            workspace = g_d_lt_workspace;
            workspaceSize = std::min(g_lt_workspace_cap, LT_WORKSPACE_BYTES);
        }

        const float alpha = 1.0f;
        const float beta  = 0.0f;
        int threads = topk_block_threads;
        size_t smem = (size_t)threads * (size_t)std::min(topk, 64) * (sizeof(float) + sizeof(int));
        dim3 grid(nq_g,1,1), block(threads,1,1);

        // [新增补充] 直接指向“全量 norm 缓存”中，目标组对应的那一段
        // g_d_all_norm 是 [total_points]，所以目标组范围 [x_off, x_off+nx) 就是这组的 xnorm
        // ------------------------------------------------------------
        // 分块处理 X：xb 是 tile 起点，cur 是 tile 实际大小
        // 每个 tile：
        //   - GEMM 得 dot[nq_g, cur]
        //   - 根据 dot + norm 转成 L2，并更新 running topK
        // ------------------------------------------------------------
        int full_tile_count = nx / tile_nx;
        int rem = nx - full_tile_count * tile_nx;
        int xb = 0;
        // 先处理完整 tile（shape 一致）：分组 batch GEMM
        if (full_tile_count > 0) {
            int remaining_tiles = full_tile_count;
            while (remaining_tiles > 0) {
                int batched_tiles = std::min(remaining_tiles, gemm_group_tiles);
                int cur = tile_nx;
                const float* dXtile = dX + (size_t)xb * dim;

                if (group_use_sgemm) {
                    // 点积矩阵按“列主序 tile_nx x nq_g”写入 g_d_dot，内存与 row-major [nq_g, tile_nx] 等价。
                    const long long strideA = (long long)tile_nx * (long long)dim;
                    const long long strideB = 0;
                    const long long strideC = (long long)tile_nx * (long long)nq_g;

                    cublasStatus_t st = cublasSgemmStridedBatched(
                        g_cublas,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        tile_nx, nq_g, dim,
                        &alpha,
                        dXtile, dim, strideA,
                        dQ, dim, strideB,
                        &beta,
                        g_d_dot, tile_nx, strideC,
                        batched_tiles);
                    if (st != CUBLAS_STATUS_SUCCESS) {
                        prof_logf("[ERROR] cublasSgemmStridedBatched failed status=%d (nq=%d dim=%d tile=%d batch=%d).",
                                  (int)st, nq_g, dim, cur, batched_tiles);
                        throw std::runtime_error("cublasSgemmStridedBatched failed in full-tile path.");
                    }
                } else if (group_use_naive) {
                    constexpr int NAIVE_WARP = 32;
                    dim3 bdot(NAIVE_WARP, (unsigned)naive_warps_per_block, 1);
                    constexpr int NAIVE_MAX_GRID_Y = 65535;
                    const int q_chunk_cap = NAIVE_MAX_GRID_Y * (int)bdot.y;
                    bool use_shared_x = naive_shared_x && (dim <= 4096);
                    int x_cols_per_block = std::min(naive_x_cols_per_block_req, cur);
                    if (x_cols_per_block < 1) x_cols_per_block = 1;
                    if (use_shared_x) {
                        const size_t shared_cap = (size_t)naive_shared_kb * 1024;
                        while (x_cols_per_block > 1 &&
                               (size_t)x_cols_per_block * (size_t)dim * sizeof(float) > shared_cap) {
                            x_cols_per_block /= 2;
                        }
                        if ((size_t)x_cols_per_block * (size_t)dim * sizeof(float) > shared_cap) {
                            use_shared_x = false;
                            x_cols_per_block = 1;
                        }
                    }
                    size_t naive_smem = use_shared_x ? ((size_t)x_cols_per_block * (size_t)dim * sizeof(float)) : 0;
                    for (int q_base = 0; q_base < nq_g; q_base += q_chunk_cap) {
                        int q_chunk = std::min(q_chunk_cap, nq_g - q_base);
                        dim3 gdot((unsigned)((cur + x_cols_per_block - 1) / x_cols_per_block),
                                  (unsigned)((q_chunk + bdot.y - 1) / bdot.y),
                                  (unsigned)batched_tiles);
                        ung_dot_batched_naive_global_kernel<<<gdot, bdot, naive_smem>>>(
                            dQ,
                            dXtile,
                            q_chunk,
                            cur,
                            dim,
                            batched_tiles,
                            q_base,
                            nq_g,
                            tile_nx * dim,
                            use_shared_x ? 1 : 0,
                            naive_vec4 ? 1 : 0,
                            x_cols_per_block,
                            g_d_dot);
                    }
                    cudaError_t naive_err = cudaGetLastError();
                    if (naive_err != cudaSuccess) {
                        prof_logf("[ERROR] naive dot kernel failed: %s (nq=%d dim=%d tile=%d batch=%d xcols=%d shared=%d)",
                                  cudaGetErrorString(naive_err), nq_g, dim, cur, batched_tiles, x_cols_per_block, use_shared_x ? 1 : 0);
                        throw std::runtime_error("naive dot kernel launch failed in full-tile path.");
                    }
                    if (naive_debug_dot && !naive_debug_logged) {
                        int dbg_q = std::min(nq_g, 2);
                        int dbg_j = std::min(cur, 3);
                        std::vector<float> dbg_dot((size_t)dbg_q * (size_t)dbg_j, 0.f);
                        cudaMemcpy(dbg_dot.data(), g_d_dot, dbg_dot.size() * sizeof(float), cudaMemcpyDeviceToHost);
                        for (int qq = 0; qq < dbg_q; ++qq) {
                            for (int jj = 0; jj < dbg_j; ++jj) {
                                float cpu_dot = 0.f;
                                const float* qv = g_h_Q + ((size_t)q_start + (size_t)qq) * (size_t)dim;
                                const float* xv = g_h_all_X + ((size_t)x_off + (size_t)xb + (size_t)jj) * (size_t)dim;
                                for (int d = 0; d < dim; ++d) cpu_dot += qv[d] * xv[d];
                                float gpu_dot = dbg_dot[(size_t)qq * (size_t)dbg_j + (size_t)jj];
                                prof_logf("[PROF] naive_dot_debug full gi=%zu q=%d j=%d gpu=%.6f cpu=%.6f",
                                          gi, qq, jj, gpu_dot, cpu_dot);
                            }
                        }
                        naive_debug_logged = true;
                    }
                } else {
                    cublasLtMatrixLayout_t xFullDesc, cFullDesc;
                    cublasLtMatrixLayoutCreate(&xFullDesc, CUDA_R_32F, tile_nx, dim, dim);
                    cublasLtMatrixLayoutSetAttribute(xFullDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
                    cublasLtMatrixLayoutCreate(&cFullDesc, CUDA_R_32F, nq_g, tile_nx, tile_nx);
                    cublasLtMatrixLayoutSetAttribute(cFullDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

                    int64_t stride_a = 0;
                    int64_t stride_b = (int64_t)tile_nx * dim;
                    int64_t stride_c = (int64_t)nq_g * tile_nx;
                    cublasLtMatrixLayoutSetAttribute(qDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a));
                    cublasLtMatrixLayoutSetAttribute(xFullDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b));
                    cublasLtMatrixLayoutSetAttribute(cFullDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c));

                    int32_t batch_count = batched_tiles;
                    cublasLtMatrixLayoutSetAttribute(qDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
                    cublasLtMatrixLayoutSetAttribute(xFullDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
                    cublasLtMatrixLayoutSetAttribute(cFullDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));

                    cublasStatus_t st = cublasLtMatmul(
                        g_cublasLt,
                        opDesc,
                        &alpha,
                        dQ, qDesc,
                        dXtile, xFullDesc,
                        &beta,
                        g_d_dot, cFullDesc,
                        g_d_dot, cFullDesc,
                        nullptr,
                        workspace, workspaceSize,
                        0);
                    cublasLtMatrixLayoutDestroy(xFullDesc);
                    cublasLtMatrixLayoutDestroy(cFullDesc);

                    if (st != CUBLAS_STATUS_SUCCESS) {
                        prof_logf("[ERROR] cublasLtMatmul(batch) failed status=%d (nq=%d dim=%d tile=%d batch=%d).",
                                  (int)st, nq_g, dim, cur, batched_tiles);
                        throw std::runtime_error("cublasLtMatmul(batch) failed in full-tile grouped path.");
                    }
                }

                const float* xn_group_chunk = (dXnorm_group ? (dXnorm_group + xb) : nullptr);
                update_topk_from_dot_grouped_tiles_kernel<<<grid, block, smem>>>(
                    g_d_dot,
                    g_d_q_norm + q_start,
                    xn_group_chunk,
                    nq_g, cur, batched_tiles, topk, xb,
                    dBestI, dBestD);

                xb += batched_tiles * tile_nx;
                remaining_tiles -= batched_tiles;
            }
        }

        // 尾块（不足一个 tile）单次 GEMM
        if (rem > 0) {
            const float* dXtile = dX + (size_t)xb * dim;
            bool tail_ok = false;

            // tile=1 时优先使用 GEMV，规避 cublasLt 对极小矩阵的不稳定
            if (!group_use_naive && rem == 1) {
                // 优先尝试 SGEMM(m=1)；失败再降级到 GEMV
                cublasStatus_t st_sgemm = cublasSgemm(
                    g_cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    1, nq_g, dim,
                    &alpha,
                    dXtile, dim,
                    dQ, dim,
                    &beta,
                    g_d_dot, 1);
                if (st_sgemm == CUBLAS_STATUS_SUCCESS) {
                    tail_ok = true;
                } else {
                    prof_logf("[ERROR] cublasSgemm(tile=1) failed status=%d (nq=%d dim=%d), fallback to gemv.",
                              (int)st_sgemm, nq_g, dim);
                }
            }

            if (!group_use_naive && rem == 1 && !tail_ok) {
                cublasStatus_t st_gemv = cublasSgemv(
                    g_cublas,
                    CUBLAS_OP_T,
                    dim, nq_g,
                    &alpha,
                    dQ, dim,
                    dXtile, 1,
                    &beta,
                    g_d_dot, 1);
                if (st_gemv == CUBLAS_STATUS_SUCCESS) {
                    tail_ok = true;
                    ++tail_gemv_count;
                } else {
                    prof_logf("[ERROR] cublasSgemv fallback failed status=%d (nq=%d dim=%d tile=1).",
                              (int)st_gemv, nq_g, dim);
                }
            }

            if (!tail_ok) {
                if (group_use_naive) {
                    constexpr int NAIVE_WARP = 32;
                    dim3 bdot(NAIVE_WARP, (unsigned)naive_warps_per_block, 1);
                    constexpr int NAIVE_MAX_GRID_Y = 65535;
                    const int q_chunk_cap = NAIVE_MAX_GRID_Y * (int)bdot.y;
                    bool use_shared_x = naive_shared_x && (dim <= 4096);
                    int x_cols_per_block = std::min(naive_x_cols_per_block_req, rem);
                    if (x_cols_per_block < 1) x_cols_per_block = 1;
                    if (use_shared_x) {
                        const size_t shared_cap = (size_t)naive_shared_kb * 1024;
                        while (x_cols_per_block > 1 &&
                               (size_t)x_cols_per_block * (size_t)dim * sizeof(float) > shared_cap) {
                            x_cols_per_block /= 2;
                        }
                        if ((size_t)x_cols_per_block * (size_t)dim * sizeof(float) > shared_cap) {
                            use_shared_x = false;
                            x_cols_per_block = 1;
                        }
                    }
                    size_t naive_smem = use_shared_x ? ((size_t)x_cols_per_block * (size_t)dim * sizeof(float)) : 0;
                    for (int q_base = 0; q_base < nq_g; q_base += q_chunk_cap) {
                        int q_chunk = std::min(q_chunk_cap, nq_g - q_base);
                        dim3 gdot((unsigned)((rem + x_cols_per_block - 1) / x_cols_per_block),
                                  (unsigned)((q_chunk + bdot.y - 1) / bdot.y),
                                  1u);
                        ung_dot_batched_naive_global_kernel<<<gdot, bdot, naive_smem>>>(
                            dQ,
                            dXtile,
                            q_chunk,
                            rem,
                            dim,
                            1,
                            q_base,
                            nq_g,
                            rem * dim,
                            use_shared_x ? 1 : 0,
                            naive_vec4 ? 1 : 0,
                            x_cols_per_block,
                            g_d_dot);
                    }
                    cudaError_t naive_err = cudaGetLastError();
                    if (naive_err != cudaSuccess) {
                        prof_logf("[ERROR] naive dot kernel(tail) failed: %s (nq=%d dim=%d tile=%d xcols=%d shared=%d)",
                                  cudaGetErrorString(naive_err), nq_g, dim, rem, x_cols_per_block, use_shared_x ? 1 : 0);
                        throw std::runtime_error("naive dot kernel launch failed in tail-tile path.");
                    }
                    if (naive_debug_dot && !naive_debug_logged) {
                        int dbg_q = std::min(nq_g, 2);
                        int dbg_j = std::min(rem, 3);
                        std::vector<float> dbg_dot((size_t)dbg_q * (size_t)dbg_j, 0.f);
                        cudaMemcpy(dbg_dot.data(), g_d_dot, dbg_dot.size() * sizeof(float), cudaMemcpyDeviceToHost);
                        for (int qq = 0; qq < dbg_q; ++qq) {
                            for (int jj = 0; jj < dbg_j; ++jj) {
                                float cpu_dot = 0.f;
                                const float* qv = g_h_Q + ((size_t)q_start + (size_t)qq) * (size_t)dim;
                                const float* xv = g_h_all_X + ((size_t)x_off + (size_t)xb + (size_t)jj) * (size_t)dim;
                                for (int d = 0; d < dim; ++d) cpu_dot += qv[d] * xv[d];
                                float gpu_dot = dbg_dot[(size_t)qq * (size_t)dbg_j + (size_t)jj];
                                prof_logf("[PROF] naive_dot_debug tail gi=%zu q=%d j=%d gpu=%.6f cpu=%.6f",
                                          gi, qq, jj, gpu_dot, cpu_dot);
                            }
                        }
                        naive_debug_logged = true;
                    }
                    tail_ok = true;
                } else if (group_use_sgemm) {
                    cublasStatus_t st = cublasSgemm(
                        g_cublas,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        rem, nq_g, dim,
                        &alpha,
                        dXtile, dim,
                        dQ, dim,
                        &beta,
                        g_d_dot, rem);
                    if (st != CUBLAS_STATUS_SUCCESS) {
                        prof_logf("[ERROR] cublasSgemm(tail) failed status=%d (nq=%d dim=%d tile=%d).",
                                  (int)st, nq_g, dim, rem);
                        throw std::runtime_error("cublasSgemm failed in tail-tile path.");
                    }
                } else {
                    cublasLtMatrixLayout_t xTailDesc, cTailDesc;
                    cublasLtMatrixLayoutCreate(&xTailDesc, CUDA_R_32F, rem, dim, dim);
                    cublasLtMatrixLayoutSetAttribute(xTailDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
                    cublasLtMatrixLayoutCreate(&cTailDesc, CUDA_R_32F, nq_g, rem, rem);
                    cublasLtMatrixLayoutSetAttribute(cTailDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

                    int32_t one = 1;
                    int64_t stride_zero = 0;
                    cublasLtMatrixLayoutSetAttribute(qDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &one, sizeof(one));
                    cublasLtMatrixLayoutSetAttribute(qDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_zero, sizeof(stride_zero));
                    cublasLtMatrixLayoutSetAttribute(xTailDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &one, sizeof(one));
                    cublasLtMatrixLayoutSetAttribute(xTailDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_zero, sizeof(stride_zero));
                    cublasLtMatrixLayoutSetAttribute(cTailDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &one, sizeof(one));
                    cublasLtMatrixLayoutSetAttribute(cTailDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_zero, sizeof(stride_zero));

                    cublasStatus_t st = cublasLtMatmul(
                        g_cublasLt,
                        opDesc,
                        &alpha,
                        dQ, qDesc,
                        dXtile, xTailDesc,
                        &beta,
                        g_d_dot, cTailDesc,
                        g_d_dot, cTailDesc,
                        nullptr,
                        workspace, workspaceSize,
                        0);

                    cublasLtMatrixLayoutDestroy(xTailDesc);
                    cublasLtMatrixLayoutDestroy(cTailDesc);

                    if (st != CUBLAS_STATUS_SUCCESS) {
                        prof_logf("[ERROR] cublasLtMatmul failed status=%d (nq=%d dim=%d tile=%d).",
                                  (int)st, nq_g, dim, rem);
                        throw std::runtime_error("cublasLtMatmul failed in tail-tile path.");
                    }
                }
            }

            const float* xn_tile = (dXnorm_group ? (dXnorm_group + xb) : nullptr);
            update_topk_from_dot_tile_kernel<<<grid, block, smem>>>(
                g_d_dot,
                g_d_q_norm + q_start,
                xn_tile,
                nq_g, rem, topk, xb,
                dBestI, dBestD);
        }

        if (group_use_lt) {
            cublasLtMatrixLayoutDestroy(qDesc);
            cublasLtMatmulDescDestroy(opDesc);
        }
    }

    if (bucket_group_fused && g_d_all_norm != nullptr) {
        if (!small_group_descs.empty()) {
            ensure_device_group_desc_buffer(small_group_descs.size());
            cudaMemcpy(g_d_group_desc,
                       small_group_descs.data(),
                       small_group_descs.size() * sizeof(UngGroupQueryDesc),
                       cudaMemcpyHostToDevice);
            int launch_warps = std::max(small_group_warps, std::min(small_group_max_nx, 16));
            if (launch_warps > 16) launch_warps = 16;
            if (launch_warps < 1) launch_warps = 1;
            dim3 sg_grid((unsigned)small_group_descs.size(), 1u, 1u);
            dim3 sg_block((unsigned)(launch_warps * 32), 1u, 1u);
            size_t sg_smem = (size_t)dim * sizeof(float)
                           + (size_t)small_group_max_nx * (sizeof(float) + sizeof(int));
            ung_group_desc_topk_fused_global_kernel<<<sg_grid, sg_block, sg_smem>>>(
                g_d_group_desc,
                (int)small_group_descs.size(),
                g_d_Q,
                g_d_q_norm,
                g_d_all_X,
                g_d_all_norm,
                dim,
                topk,
                small_group_max_nx,
                g_d_idx,
                g_d_dis);
        }
        if (!medium_group_descs.empty()) {
            ensure_device_group_desc_buffer(medium_group_descs.size());
            cudaMemcpy(g_d_group_desc,
                       medium_group_descs.data(),
                       medium_group_descs.size() * sizeof(UngGroupQueryDesc),
                       cudaMemcpyHostToDevice);
            int launch_warps = std::min(medium_group_warps, 16);
            if (launch_warps < 1) launch_warps = 1;
            dim3 mg_grid((unsigned)medium_group_descs.size(), 1u, 1u);
            dim3 mg_block((unsigned)(launch_warps * 32), 1u, 1u);
            size_t mg_smem = (size_t)dim * sizeof(float)
                           + (size_t)medium_group_max_nx * (sizeof(float) + sizeof(int));
            ung_group_desc_topk_fused_global_kernel<<<mg_grid, mg_block, mg_smem>>>(
                g_d_group_desc,
                (int)medium_group_descs.size(),
                g_d_Q,
                g_d_q_norm,
                g_d_all_X,
                g_d_all_norm,
                dim,
                topk,
                medium_group_max_nx,
                g_d_idx,
                g_d_dis);
        }
    }

    // 与 GPU kernel 队列并行执行 CPU tiny 组，尽量把 CPU 时间隐藏在 GPU 计算后半段
    process_cpu_tiny_groups();

    cudaDeviceSynchronize();
    cudaEventRecord(k1);
    cudaEventSynchronize(k1);

    float tker = 0.f;
    cudaEventElapsedTime(&tker, k0, k1);
    if (kernel_ms) *kernel_ms += tker;

    cudaEventDestroy(k0); cudaEventDestroy(k1);

    // ------------------------------------------------------------
    // 第八步：一次性 D2H 把 topk idx/dist 全部拷回 pinned host
    // ------------------------------------------------------------
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    cudaEventRecord(e0);
    cudaMemcpy(g_h_idx, g_d_idx, (size_t)total_queries * topk * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(g_h_dis, g_d_dis, (size_t)total_queries * topk * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float td2h = 0.f;
    cudaEventElapsedTime(&td2h, e0, e1);
    if (d2h_ms) *d2h_ms += td2h;

    cudaEventDestroy(e0); cudaEventDestroy(e1);

    // ------------------------------------------------------------
    // 第九步：把结果写回 cross_group_neighbors（保持你原语义）
    // 注意：best_idx 里存的是“目标组内的 local 下标”
    // 所以需要加 tgt_offset（rng.first）变成全局向量 id
    // ------------------------------------------------------------
    const auto t_writeback_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static, 1024)
    for (int qi = 0; qi < total_queries; ++qi) {
        IdxType qid = query_global_ids[qi];
        int gi = query_target_index[qi];

        IdxType tgt_gid = target_group_ids[(size_t)gi];
        const auto& rng = _group_id_to_range[tgt_gid];
        IdxType tgt_offset = rng.first;
        const int valid_k = std::min<int>(topk, (int)(rng.second - rng.first));

        for (int r = 0; r < valid_k; ++r) {
            int j = g_h_idx[(size_t)qi * topk + r];
            if (j < 0) continue;
            float dist = g_h_dis[(size_t)qi * topk + r];
            cross_group_neighbors[qid].insert(tgt_offset + (IdxType)j, dist);
        }
    }
    const auto t_writeback_end = std::chrono::high_resolution_clock::now();

    size_t valid_topk_pairs = 0;
    for (int qi = 0; qi < total_queries; ++qi) {
        for (int r = 0; r < topk; ++r) {
            if (g_h_idx[(size_t)qi * topk + r] >= 0) ++valid_topk_pairs;
        }
    }
    prof_logf("[PROF] cross_edges.topk_valid_pairs total=%zu", valid_topk_pairs);
    prof_logf("[PROF] cross_edges.tail_gemv_calls total=%zu", tail_gemv_count);
    prof_logf("[PROF] cross_edges.small_group_fused groups=%zu queries=%zu max_nx=%d",
              small_group_fused_group_count, small_group_fused_query_count, small_group_max_nx);
    prof_logf("[PROF] cross_edges.medium_group_fused groups=%zu queries=%zu max_nx=%d",
              medium_group_fused_group_count, medium_group_fused_query_count, medium_group_max_nx);
    prof_logf("[PROF] cross_edges.bucket_group_fused enabled=%d small_desc=%zu medium_desc=%zu",
              bucket_group_fused ? 1 : 0, small_group_descs.size(), medium_group_descs.size());
    prof_logf("[PROF] cross_edges.heavy_sgemm groups=%zu queries=%zu",
              heavy_sgemm_group_count, heavy_sgemm_query_count);
    prof_logf("[PROF] cross_edges.cpu_tiny groups=%zu queries=%zu ms=%.3f enabled=%d nx_max=%d nq_max=%d ops_thresh=%lld",
              cpu_tiny_group_count, cpu_tiny_query_count, cpu_tiny_ms, cpu_tiny_groups ? 1 : 0,
              cpu_tiny_nx_max, cpu_tiny_nq_max, cpu_tiny_ops_thresh);
    prof_logf("[PROF] cross_edges.stage_ms count=%.3f offset=%.3f fill_q=%.3f writeback=%.3f total_queries=%d gather_q=%d singleton_fastpath=%d singleton_q=%zu cpu_tiny_groups=%zu",
              elapsed_ms(t_count_start, t_count_end),
              elapsed_ms(t_offset_start, t_offset_end),
              elapsed_ms(t_fill_start, t_fill_end),
              elapsed_ms(t_writeback_start, t_writeback_end),
              total_queries,
              use_device_query_gather ? 1 : 0,
              singleton_fastpath ? 1 : 0,
              singleton_query_count,
              cpu_tiny_group_count);

    const int verify_samples = read_env_int("UNG_GEMM_VERIFY_SAMPLES", 0, 0, 20000);
    if (verify_samples > 0) {
        const bool verify_strict = read_env_int("UNG_GEMM_VERIFY_STRICT", 0, 0, 1) == 1;
        const int checked = std::min(verify_samples, total_queries);
        int mismatch = 0;
        int mismatch_logged = 0;
        const int stride = std::max(1, total_queries / std::max(1, checked));

        auto l2_cpu = [&](IdxType qid, IdxType xid) -> float {
            const float* q = reinterpret_cast<const float*>(_base_storage->get_vector(qid));
            const float* x = reinterpret_cast<const float*>(_base_storage->get_vector(xid));
            float s = 0.f;
            for (int d = 0; d < dim; ++d) {
                float diff = q[d] - x[d];
                s += diff * diff;
            }
            return s;
        };

        for (int t = 0, qi = 0; t < checked && qi < total_queries; ++t, qi += stride) {
            IdxType qid = query_global_ids[qi];
            int gi = query_target_index[qi];
            IdxType tgt_gid = target_group_ids[(size_t)gi];
            const auto& rng = _group_id_to_range[tgt_gid];
            const IdxType x0 = rng.first, x1 = rng.second;
            if (x1 <= x0) continue;

            std::vector<std::pair<float, int>> cpu;
            cpu.reserve((size_t)std::min<IdxType>((IdxType)topk + 8, x1 - x0));
            for (IdxType xid = x0; xid < x1; ++xid) {
                float d = l2_cpu(qid, xid);
                int local = (int)(xid - x0);
                if ((int)cpu.size() < topk) {
                    cpu.emplace_back(d, local);
                    if ((int)cpu.size() == topk) {
                        std::sort(cpu.begin(), cpu.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
                    }
                } else if (d < cpu.back().first) {
                    cpu.back() = {d, local};
                    for (int i = topk - 1; i > 0 && cpu[i].first < cpu[i - 1].first; --i) std::swap(cpu[i], cpu[i - 1]);
                }
            }
            if (!cpu.empty()) {
                std::sort(cpu.begin(), cpu.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
            }

            std::vector<std::pair<int,float>> gpu;
            gpu.reserve(topk);
            for (int r = 0; r < topk; ++r) {
                int idx = g_h_idx[(size_t)qi * topk + r];
                float dis = g_h_dis[(size_t)qi * topk + r];
                if (idx >= 0) gpu.emplace_back(idx, dis);
            }
            if ((int)gpu.size() != (int)cpu.size()) {
                if (mismatch_logged < 3) {
                    prof_logf("[PROF] cross_edges.verify_mismatch qid=%lld qi=%d target_gid=%lld reason=size gpu=%d cpu=%d x_size=%lld",
                              (long long)qid, qi, (long long)tgt_gid, (int)gpu.size(), (int)cpu.size(), (long long)(x1 - x0));
                    mismatch_logged++;
                }
                ++mismatch;
                continue;
            }
            for (size_t i = 0; i < gpu.size(); ++i) {
                int gi_idx = gpu[i].first;
                int ci_idx = cpu[i].second;
                float gd = gpu[i].second;
                float cd = cpu[i].first;
                if (gi_idx != ci_idx || std::fabs(gd - cd) > 1e-3f * std::max(1.0f, std::fabs(cd))) {
                    if (mismatch_logged < 3) {
                        prof_logf("[PROF] cross_edges.verify_mismatch qid=%lld qi=%d target_gid=%lld rank=%d gpu_idx=%d gpu_dist=%.6f cpu_idx=%d cpu_dist=%.6f x_size=%lld",
                                  (long long)qid, qi, (long long)tgt_gid, (int)i, gi_idx, gd, ci_idx, cd, (long long)(x1 - x0));
                        for (int rr = 0; rr < std::min(topk, 6); ++rr) {
                            int gidx = (rr < (int)gpu.size()) ? gpu[rr].first : -1;
                            float gdis = (rr < (int)gpu.size()) ? gpu[rr].second : -1.f;
                            int cidx = (rr < (int)cpu.size()) ? cpu[rr].second : -1;
                            float cdis = (rr < (int)cpu.size()) ? cpu[rr].first : -1.f;
                            prof_logf("[PROF] cross_edges.verify_pair qid=%lld rank=%d gpu=(%d,%.6f) cpu=(%d,%.6f)",
                                      (long long)qid, rr, gidx, gdis, cidx, cdis);
                        }
                        mismatch_logged++;
                    }
                    ++mismatch;
                    break;
                }
            }
        }

        prof_logf("[PROF] cross_edges.verify sampled=%d mismatches=%d strict=%d", checked, mismatch, verify_strict ? 1 : 0);
        if (verify_strict && mismatch > 0) {
            throw std::runtime_error("UNG_GEMM verification failed: mismatch detected.");
        }
    }
}
