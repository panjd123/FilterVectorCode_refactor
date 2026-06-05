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
#include <limits>
#include <omp.h>
#include <unordered_map>
#include <mma.h>

#ifdef ANNS_HAVE_CUVS
// ACORN/FAISS headers pulled in by uni_nav_graph.h define a global debug(...)
// macro. RAPIDS logger has a debug() member function, so isolate cuVS includes
// from that macro pollution.
#ifdef debug
#pragma push_macro("debug")
#undef debug
#define ANNS_RESTORE_DEBUG_MACRO_AFTER_CUVS
#endif
#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#ifdef ANNS_RESTORE_DEBUG_MACRO_AFTER_CUVS
#pragma pop_macro("debug")
#undef ANNS_RESTORE_DEBUG_MACRO_AFTER_CUVS
#endif
#endif

struct UngGroupQueryDesc {
    uint32_t qi;
    uint32_t x_off;
    uint32_t nx;
};

struct UngGroupTileDesc {
    uint32_t q_start;
    uint32_t q_count;
    uint32_t x_off;
    uint32_t nx;
};

struct UngSourceQueryDesc {
    uint32_t qid;
    uint32_t edge_start;
    uint32_t edge_count;
};

struct UngSourceTileDesc {
    uint32_t q_start;
    uint32_t q_count;
    uint32_t out_start;
    uint32_t edge_start;
    uint32_t edge_count;
};

struct UngTargetSegmentDesc {
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
static uint32_t* g_d_q_target_offsets = nullptr; // device：flatten query 对应 target group 的全局起点
static size_t g_q_target_offsets_cap_nq = 0;
static int* g_d_global_idx = nullptr;            // device：按全局 qid merge 后的 topk id
static float* g_d_global_dis = nullptr;          // device：按全局 qid merge 后的 topk dist
static int* g_d_global_locks = nullptr;          // device：每个全局 qid 一个轻量锁
static size_t g_global_topk_cap_n = 0;
static int g_global_topk_cap_k = 0;
static uint32_t* g_d_singleton_qi = nullptr;   // device：singleton query 的 flatten 索引
static uint32_t* g_d_singleton_xoff = nullptr; // device：singleton 对应的全量 X 偏移
static size_t g_singleton_cap_nq = 0;
#ifdef ANNS_HAVE_CUVS
static int64_t* g_d_cuvs_idx = nullptr;
static int64_t* g_h_cuvs_idx = nullptr;
static size_t g_cuvs_idx_cap = 0;
#endif
static UngGroupQueryDesc* g_d_group_desc = nullptr; // device：小/中组批处理描述符
static size_t g_group_desc_cap = 0;
static UngGroupTileDesc* g_d_group_tile_desc = nullptr; // device：TF32 批处理 group tile 描述符
static size_t g_group_tile_desc_cap = 0;
static uint32_t* g_d_tf32_group_tile_offsets = nullptr; // device：每个 TF32 group 的 16-row tile prefix
static size_t g_tf32_group_tile_offsets_cap = 0;

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

    // Device Q can be tens of GB on large split chunks. Do not use geometric
    // growth here: a tiny chunk-size increase can otherwise double the buffer
    // and exhaust GPU memory on x400-scale runs.
    g_dev_cap_nq = need_nq;
    g_dev_cap_k = need_k;
    g_dev_dim_cap = dim;

    // 释放旧 device buffer
    if (g_d_Q)   cudaFree(g_d_Q);
    if (g_d_idx) cudaFree(g_d_idx);
    if (g_d_dis) cudaFree(g_d_dis);

    cudaError_t st = cudaMalloc(&g_d_Q, (size_t)g_dev_cap_nq * dim * sizeof(float));
    if (st != cudaSuccess) {
        g_d_Q = nullptr;
        throw std::runtime_error(std::string("cudaMalloc g_d_Q failed: ") + cudaGetErrorString(st));
    }
    st = cudaMalloc(&g_d_idx, (size_t)g_dev_cap_nq * g_dev_cap_k * sizeof(int));
    if (st != cudaSuccess) {
        g_d_idx = nullptr;
        throw std::runtime_error(std::string("cudaMalloc g_d_idx failed: ") + cudaGetErrorString(st));
    }
    st = cudaMalloc(&g_d_dis, (size_t)g_dev_cap_nq * g_dev_cap_k * sizeof(float));
    if (st != cudaSuccess) {
        g_d_dis = nullptr;
        throw std::runtime_error(std::string("cudaMalloc g_d_dis failed: ") + cudaGetErrorString(st));
    }
}

inline void ensure_device_output_buffers(size_t need_nq, int need_k) {
    const bool need = (need_nq > g_dev_cap_nq) || (need_k > g_dev_cap_k) ||
                      (g_d_idx == nullptr) || (g_d_dis == nullptr);
    if (!need) return;

    g_dev_cap_nq = need_nq;
    g_dev_cap_k = need_k;
    g_dev_dim_cap = 0;

    if (g_d_Q) { cudaFree(g_d_Q); g_d_Q = nullptr; }
    if (g_d_idx) { cudaFree(g_d_idx); g_d_idx = nullptr; }
    if (g_d_dis) { cudaFree(g_d_dis); g_d_dis = nullptr; }

    cudaError_t st = cudaMalloc(&g_d_idx, (size_t)g_dev_cap_nq * g_dev_cap_k * sizeof(int));
    if (st != cudaSuccess) {
        g_d_idx = nullptr;
        throw std::runtime_error(std::string("cudaMalloc g_d_idx failed: ") + cudaGetErrorString(st));
    }
    st = cudaMalloc(&g_d_dis, (size_t)g_dev_cap_nq * g_dev_cap_k * sizeof(float));
    if (st != cudaSuccess) {
        g_d_dis = nullptr;
        throw std::runtime_error(std::string("cudaMalloc g_d_dis failed: ") + cudaGetErrorString(st));
    }
}

inline void ensure_device_qid_buffer(size_t need_nq) {
    if (need_nq <= g_qid_cap_nq && g_d_q_ids) return;
    if (g_d_q_ids) cudaFree(g_d_q_ids);
    g_qid_cap_nq = std::max(need_nq, g_qid_cap_nq * 2 + 1);
    cudaMalloc(&g_d_q_ids, (size_t)g_qid_cap_nq * sizeof(uint32_t));
}

inline void ensure_device_q_target_offsets_buffer(size_t need_nq) {
    if (need_nq <= g_q_target_offsets_cap_nq && g_d_q_target_offsets) return;
    if (g_d_q_target_offsets) cudaFree(g_d_q_target_offsets);
    g_q_target_offsets_cap_nq = std::max(need_nq, g_q_target_offsets_cap_nq * 2 + 1);
    cudaMalloc(&g_d_q_target_offsets, (size_t)g_q_target_offsets_cap_nq * sizeof(uint32_t));
}

inline void ensure_device_global_topk_buffers(size_t need_n, int need_k) {
    bool need = (need_n > g_global_topk_cap_n) || (need_k > g_global_topk_cap_k)
             || !g_d_global_idx || !g_d_global_dis || !g_d_global_locks;
    if (!need) return;
    if (g_d_global_idx) cudaFree(g_d_global_idx);
    if (g_d_global_dis) cudaFree(g_d_global_dis);
    if (g_d_global_locks) cudaFree(g_d_global_locks);
    g_global_topk_cap_n = std::max(need_n, g_global_topk_cap_n * 2 + 1);
    g_global_topk_cap_k = std::max(need_k, g_global_topk_cap_k * 2 + 1);
    cudaMalloc(&g_d_global_idx, (size_t)g_global_topk_cap_n * (size_t)g_global_topk_cap_k * sizeof(int));
    cudaMalloc(&g_d_global_dis, (size_t)g_global_topk_cap_n * (size_t)g_global_topk_cap_k * sizeof(float));
    cudaMalloc(&g_d_global_locks, (size_t)g_global_topk_cap_n * sizeof(int));
}

inline void ensure_device_singleton_buffers(size_t need_nq) {
    if (need_nq <= g_singleton_cap_nq && g_d_singleton_qi && g_d_singleton_xoff) return;
    if (g_d_singleton_qi) cudaFree(g_d_singleton_qi);
    if (g_d_singleton_xoff) cudaFree(g_d_singleton_xoff);
    g_singleton_cap_nq = std::max(need_nq, g_singleton_cap_nq * 2 + 1);
    cudaMalloc(&g_d_singleton_qi, (size_t)g_singleton_cap_nq * sizeof(uint32_t));
    cudaMalloc(&g_d_singleton_xoff, (size_t)g_singleton_cap_nq * sizeof(uint32_t));
}

#ifdef ANNS_HAVE_CUVS
inline void ensure_cuvs_result_buffers(size_t need_items) {
    if (need_items <= g_cuvs_idx_cap && g_d_cuvs_idx && g_h_cuvs_idx) return;
    if (g_d_cuvs_idx) cudaFree(g_d_cuvs_idx);
    if (g_h_cuvs_idx) cudaFreeHost(g_h_cuvs_idx);
    g_cuvs_idx_cap = need_items;
    cudaError_t st = cudaMalloc(&g_d_cuvs_idx, g_cuvs_idx_cap * sizeof(int64_t));
    if (st != cudaSuccess) {
        g_d_cuvs_idx = nullptr;
        throw std::runtime_error(std::string("cudaMalloc g_d_cuvs_idx failed: ") + cudaGetErrorString(st));
    }
    st = cudaHostAlloc(&g_h_cuvs_idx, g_cuvs_idx_cap * sizeof(int64_t), cudaHostAllocDefault);
    if (st != cudaSuccess) {
        g_h_cuvs_idx = nullptr;
        throw std::runtime_error(std::string("cudaHostAlloc g_h_cuvs_idx failed: ") + cudaGetErrorString(st));
    }
}
#endif

inline void ensure_device_group_desc_buffer(size_t need_desc) {
    if (need_desc <= g_group_desc_cap && g_d_group_desc) return;
    if (g_d_group_desc) cudaFree(g_d_group_desc);
    g_group_desc_cap = std::max(need_desc, g_group_desc_cap * 2 + 1);
    cudaMalloc(&g_d_group_desc, (size_t)g_group_desc_cap * sizeof(UngGroupQueryDesc));
}

inline void ensure_device_group_tile_desc_buffer(size_t need_desc) {
    if (need_desc <= g_group_tile_desc_cap && g_d_group_tile_desc) return;
    if (g_d_group_tile_desc) cudaFree(g_d_group_tile_desc);
    g_group_tile_desc_cap = std::max(need_desc, g_group_tile_desc_cap * 2 + 1);
    cudaMalloc(&g_d_group_tile_desc, (size_t)g_group_tile_desc_cap * sizeof(UngGroupTileDesc));
}

inline void ensure_device_tf32_group_tile_offsets(size_t need_count) {
    if (need_count <= g_tf32_group_tile_offsets_cap && g_d_tf32_group_tile_offsets) return;
    if (g_d_tf32_group_tile_offsets) cudaFree(g_d_tf32_group_tile_offsets);
    g_tf32_group_tile_offsets_cap = std::max(need_count, g_tf32_group_tile_offsets_cap * 2 + 1);
    cudaMalloc(&g_d_tf32_group_tile_offsets,
               (size_t)g_tf32_group_tile_offsets_cap * sizeof(uint32_t));
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

__global__ void ung_merge_local_topk_rows_to_global_kernel(
    const int* __restrict__ local_idx,
    const float* __restrict__ local_dist,
    const uint32_t* __restrict__ query_ids,
    const uint32_t* __restrict__ target_offsets,
    int total_queries,
    int topk,
    int* __restrict__ global_idx,
    float* __restrict__ global_dist,
    int* __restrict__ locks)
{
    for (int qi = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         qi < total_queries;
         qi += (int)(gridDim.x * blockDim.x)) {
        const uint32_t qid = query_ids[qi];
        int* lock = locks + qid;
        while (atomicCAS(lock, 0, 1) != 0) { }

        int* out_i = global_idx + (size_t)qid * (size_t)topk;
        float* out_d = global_dist + (size_t)qid * (size_t)topk;
        const uint32_t target_off = target_offsets[qi];

        for (int r = 0; r < topk; ++r) {
            const size_t in_pos = (size_t)qi * (size_t)topk + (size_t)r;
            const int local = local_idx[in_pos];
            if (local < 0) continue;
            const float dist = local_dist[in_pos];
            const int gid = (int)(target_off + (uint32_t)local);

            bool exists = false;
            for (int k = 0; k < topk; ++k) {
                if (out_i[k] == gid) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                const int last = topk - 1;
                const bool better_than_tail =
                    (dist < out_d[last]) || (dist == out_d[last] && gid < out_i[last]);
                if (better_than_tail) {
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
            }
        }

        __threadfence();
        atomicExch(lock, 0);
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

extern "C" __global__ void ung_singleton_top1_global_merge_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ q_norm,
    const float* __restrict__ all_x,
    const float* __restrict__ all_norm,
    const uint32_t* __restrict__ singleton_qi,
    const uint32_t* __restrict__ singleton_xoff,
    const uint32_t* __restrict__ q_ids,
    int singleton_nq,
    int dim,
    int topk,
    int* __restrict__ global_idx,
    float* __restrict__ global_dist,
    int* __restrict__ locks)
{
    const int sid = blockIdx.x;
    if (sid >= singleton_nq) return;
    const uint32_t qi = singleton_qi[sid];
    const uint32_t xoff = singleton_xoff[sid];

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
        const uint32_t qid = q_ids[qi];
        const float dist = q_norm[qi] + all_norm[xoff] - 2.f * sm[0];
        int* lock = locks + qid;
        while (atomicCAS(lock, 0, 1) != 0) { }
        int* out_i = global_idx + (size_t)qid * (size_t)topk;
        float* out_d = global_dist + (size_t)qid * (size_t)topk;
        ung_global_topk_insert_locked(out_i, out_d, topk, (int)xoff, dist);
        __threadfence();
        atomicExch(lock, 0);
    }
}

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
    auto elapsed_ms = [](const std::chrono::high_resolution_clock::time_point& t0,
                         const std::chrono::high_resolution_clock::time_point& t1) -> double {
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };
    const size_t total_bytes = (size_t)total_points * (size_t)dim * sizeof(float);
    const size_t stride_bytes = (size_t)dim * sizeof(float);
    const bool allow_direct_hostreg = (read_env_int("UNG_GPU_PREPARE_DIRECT_HOSTREG", 1, 0, 1) == 1);
    const bool allow_direct_pageable_env = (read_env_int("UNG_GPU_PREPARE_DIRECT_PAGEABLE", 0, 0, 1) == 1);
    const int hostreg_max_mb = read_env_int("UNG_GPU_PREPARE_HOSTREG_MAX_MB", 4096, 0, 1048576);
    const size_t hostreg_max_bytes = (size_t)hostreg_max_mb * 1024ull * 1024ull;
    const bool hostreg_size_allowed = hostreg_max_mb == 0 || total_bytes <= hostreg_max_bytes;
    double contig_probe_ms = 0.0;
    double host_register_ms = 0.0;
    double pinned_alloc_ms = 0.0;
    double device_alloc_ms = 0.0;
    double norm_launch_ms = 0.0;
    std::cout << "[cross_edges][prepare_all] points=" << total_points
              << " dim=" << dim
              << " bytes=" << total_bytes
              << " hostreg_allowed=" << (allow_direct_hostreg && hostreg_size_allowed ? 1 : 0)
              << " direct_pageable=" << (allow_direct_pageable_env || !hostreg_size_allowed ? 1 : 0)
              << " hostreg_max_mb=" << hostreg_max_mb
              << std::endl;

    // 优先尝试“连续内存直拷”：避免 host 侧再做一次全量 memcpy。
    // 注意：当数据太大不适合 cudaHostRegister 时，仍然应该探测连续性并走
    // pageable cudaMemcpyAsync；之前这里被 hostreg_size_allowed 短路，会错误回退
    // 到 pinned staging，10%x40 上会多出 7.4GB 的 CPU repack。
    bool use_direct_hostreg = false;
    const float* direct_src = nullptr;
    const bool want_direct_pageable = allow_direct_pageable_env || !hostreg_size_allowed;
    if ((allow_direct_hostreg || want_direct_pageable) && total_points > 0) {
        const auto t_probe_start = std::chrono::high_resolution_clock::now();
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
        const auto t_probe_end = std::chrono::high_resolution_clock::now();
        contig_probe_ms = elapsed_ms(t_probe_start, t_probe_end);
        if (contiguous && want_direct_pageable) {
            use_direct_hostreg = true;
            direct_src = v0;
        } else if (contiguous && allow_direct_hostreg && hostreg_size_allowed) {
            if (g_hostreg_all_x_ptr && (g_hostreg_all_x_ptr != v0 || g_hostreg_all_x_bytes != total_bytes)) {
                const auto t_unreg_start = std::chrono::high_resolution_clock::now();
                cudaHostUnregister(const_cast<float*>(g_hostreg_all_x_ptr));
                const auto t_unreg_end = std::chrono::high_resolution_clock::now();
                host_register_ms += elapsed_ms(t_unreg_start, t_unreg_end);
                g_hostreg_all_x_ptr = nullptr;
                g_hostreg_all_x_bytes = 0;
            }
            if (!g_hostreg_all_x_ptr) {
                const auto t_reg_start = std::chrono::high_resolution_clock::now();
                std::cout << "[cross_edges][prepare_all] cudaHostRegister begin bytes="
                          << total_bytes << std::endl;
                cudaError_t reg_st = cudaHostRegister(const_cast<float*>(v0), total_bytes, cudaHostRegisterDefault);
                const auto t_reg_end = std::chrono::high_resolution_clock::now();
                host_register_ms += elapsed_ms(t_reg_start, t_reg_end);
                std::cout << "[cross_edges][prepare_all] cudaHostRegister end status="
                          << cudaGetErrorString(reg_st)
                          << " ms=" << host_register_ms << std::endl;
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
            const auto t_alloc_start = std::chrono::high_resolution_clock::now();
            cudaHostAlloc(&g_h_all_X, g_all_cap_n * (size_t)dim * sizeof(float), cudaHostAllocDefault);
            const auto t_alloc_end = std::chrono::high_resolution_clock::now();
            pinned_alloc_ms += elapsed_ms(t_alloc_start, t_alloc_end);
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
        const auto t_alloc_start = std::chrono::high_resolution_clock::now();
        cudaMalloc(&g_d_all_X, new_cap_n * (size_t)dim * sizeof(float));
        const auto t_alloc_end = std::chrono::high_resolution_clock::now();
        device_alloc_ms += elapsed_ms(t_alloc_start, t_alloc_end);
        g_all_cap_n = new_cap_n;
    }

    // 4) H2D：只拷贝真实 total_points 部分
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, stream);
    const float* h_src = use_direct_hostreg ? direct_src : g_h_all_X;
    cudaMemcpyAsync(g_d_all_X, h_src, total_bytes, cudaMemcpyHostToDevice, stream);
    cudaEventRecord(e1, stream);
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
        const auto t_norm_start = std::chrono::high_resolution_clock::now();
        int threads = 256;
        const int warps_per_block = threads / 32;
        dim3 grid((unsigned)(((int)total_points + warps_per_block - 1) / warps_per_block), 1u, 1u);
        dim3 block(threads, 1, 1);
        l2_norm_sq_kernel<<<grid, block, 0, stream>>>(g_d_all_X, (int)total_points, dim, g_d_all_norm);
        const auto t_norm_end = std::chrono::high_resolution_clock::now();
        norm_launch_ms = elapsed_ms(t_norm_start, t_norm_end);
        // 注意：这里不强制同步，让外部第一次真正用到它时再整体同步即可
        // 如果你希望更稳妥，也可以在这里 cudaDeviceSynchronize();
    }
    const auto t_prepare_end = std::chrono::high_resolution_clock::now();
    const double prepare_ms = std::chrono::duration<double, std::milli>(t_prepare_end - t_prepare_start).count();
    prof_logf("[PROF] cross_edges.prepare_all path=%s allow_direct=%d direct_pageable=%d contig_probe_ms=%.3f host_register_ms=%.3f pinned_alloc_ms=%.3f host_pack_ms=%.3f device_alloc_ms=%.3f h2d_ms=%.3f norm_launch_ms=%.3f total_ms=%.3f bytes=%zu",
              (use_direct_hostreg ? (want_direct_pageable ? "direct_pageable" : "direct_hostregister") : "pinned_staging"),
              allow_direct_hostreg ? 1 : 0, want_direct_pageable ? 1 : 0, contig_probe_ms, host_register_ms, pinned_alloc_ms,
              host_pack_ms, device_alloc_ms, (double)ms, norm_launch_ms, prepare_ms, total_bytes);
    std::cout << "[cross_edges][prepare_all] done path="
              << (use_direct_hostreg ? (want_direct_pageable ? "direct_pageable" : "direct_hostregister") : "pinned_staging")
              << " host_pack_ms=" << host_pack_ms
              << " h2d_ms=" << (double)ms
              << " total_ms=" << prepare_ms
              << std::endl;
    cudaStreamDestroy(stream);
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
    if (g_d_q_target_offsets) { cudaFree(g_d_q_target_offsets); g_d_q_target_offsets = nullptr; }
    if (g_d_global_idx) { cudaFree(g_d_global_idx); g_d_global_idx = nullptr; }
    if (g_d_global_dis) { cudaFree(g_d_global_dis); g_d_global_dis = nullptr; }
    if (g_d_global_locks) { cudaFree(g_d_global_locks); g_d_global_locks = nullptr; }
    if (g_d_singleton_qi) { cudaFree(g_d_singleton_qi); g_d_singleton_qi = nullptr; }
    if (g_d_singleton_xoff) { cudaFree(g_d_singleton_xoff); g_d_singleton_xoff = nullptr; }
#ifdef ANNS_HAVE_CUVS
    if (g_d_cuvs_idx) { cudaFree(g_d_cuvs_idx); g_d_cuvs_idx = nullptr; }
    if (g_h_cuvs_idx) { cudaFreeHost(g_h_cuvs_idx); g_h_cuvs_idx = nullptr; }
#endif
    if (g_d_group_desc) { cudaFree(g_d_group_desc); g_d_group_desc = nullptr; }
    if (g_d_group_tile_desc) { cudaFree(g_d_group_tile_desc); g_d_group_tile_desc = nullptr; }
    if (g_hostreg_all_x_ptr) {
        cudaHostUnregister(const_cast<float*>(g_hostreg_all_x_ptr));
        g_hostreg_all_x_ptr = nullptr;
        g_hostreg_all_x_bytes = 0;
    }
}

bool ANNS::UniNavGraph::build_cross_edges_generate_gpu_source_exact(
    std::vector<SearchQueue>& cross_group_neighbors,
    std::vector<std::vector<IdxType>>* cross_group_neighbor_ids,
    std::vector<IdxType>* cross_group_neighbor_flat_ids,
    double* h2d_ms_sum,
    double* kernel_ms_sum,
    double* d2h_ms_sum)
{
    const int dim = _base_storage->get_dim();
    const int topk = static_cast<int>(_num_cross_edges);
    if (dim <= 0 || topk <= 0) return true;
    if (topk > 32) {
        std::cerr << "[cross_edges][source_exact] topk > 32 is not supported by this experimental path." << std::endl;
        return false;
    }

    auto elapsed_ms = [](const std::chrono::high_resolution_clock::time_point& t0,
                         const std::chrono::high_resolution_clock::time_point& t1) -> double {
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    const auto t_total0 = std::chrono::high_resolution_clock::now();
    double prepare_h2d_ms = 0.0;
    bool prepared = false;
    try {
        gpu_prepare_all_vectors_on_device(&prepare_h2d_ms, nullptr);
        prepared = true;
        if (h2d_ms_sum) *h2d_ms_sum += prepare_h2d_ms;

        const int warps = read_env_int("UNG_GPU_SOURCE_EXACT_WARPS", 8, 1, 16);
        const int threads = warps * 32;
        const int source_mode = read_env_int("UNG_GPU_SOURCE_EXACT_MODE", 0, 0, 1);

        std::vector<UngTargetSegmentDesc> h_segments;
        std::vector<UngSourceQueryDesc> h_queries;
        std::vector<UngSourceTileDesc> h_tiles;
        h_segments.reserve(_label_nav_graph ? _num_groups * 2u : 0u);
        h_queries.reserve(_num_points);
        h_tiles.reserve((_num_points + 15u) / 16u + _num_groups);

        size_t source_groups = 0;
        size_t padded_tiles = 0;
        unsigned long long dim_ops = 0;
        const auto t_pack0 = std::chrono::high_resolution_clock::now();
        for (IdxType src_gid = 1; src_gid <= _num_groups; ++src_gid) {
            const auto& outs = _label_nav_graph->out_neighbors[src_gid];
            const auto& q_range = _group_id_to_range[src_gid];
            if (outs.empty() || q_range.first >= q_range.second) continue;

            const uint32_t edge_start = static_cast<uint32_t>(h_segments.size());
            uint32_t edge_count = 0;
            size_t candidate_count = 0;
            for (IdxType tgt_gid : outs) {
                const auto& x_range = _group_id_to_range[tgt_gid];
                const IdxType nx = x_range.second - x_range.first;
                if (nx <= 0) continue;
                UngTargetSegmentDesc seg;
                seg.x_off = static_cast<uint32_t>(x_range.first);
                seg.nx = static_cast<uint32_t>(nx);
                h_segments.push_back(seg);
                ++edge_count;
                candidate_count += static_cast<size_t>(nx);
            }
            if (edge_count == 0 || candidate_count == 0) {
                h_segments.resize(edge_start);
                continue;
            }

            ++source_groups;
            const uint32_t out_start = static_cast<uint32_t>(h_queries.size());
            for (IdxType qid = q_range.first; qid < q_range.second; ++qid) {
                UngSourceQueryDesc qd;
                qd.qid = static_cast<uint32_t>(qid);
                qd.edge_start = edge_start;
                qd.edge_count = edge_count;
                h_queries.push_back(qd);
            }
            const size_t tile_group_begin = h_tiles.size();
            for (IdxType qid = q_range.first; qid < q_range.second; qid += 16) {
                UngSourceTileDesc td;
                td.q_start = static_cast<uint32_t>(qid);
                td.q_count = static_cast<uint32_t>(std::min<IdxType>(16, q_range.second - qid));
                td.out_start = out_start + static_cast<uint32_t>(qid - q_range.first);
                td.edge_start = edge_start;
                td.edge_count = edge_count;
                h_tiles.push_back(td);
            }
            const bool pad_source_group_tiles =
                read_env_int("UNG_GPU_SOURCE_EXACT_PAD_GROUPS", 1, 0, 1) == 1;
            if (pad_source_group_tiles && source_mode == 1 && topk <= 16) {
                const size_t group_tiles = h_tiles.size() - tile_group_begin;
                const size_t rem = group_tiles % static_cast<size_t>(warps);
                if (rem != 0) {
                    const size_t pad = static_cast<size_t>(warps) - rem;
                    UngSourceTileDesc inactive;
                    inactive.q_start = static_cast<uint32_t>(q_range.first);
                    inactive.q_count = 0;
                    inactive.out_start = out_start;
                    inactive.edge_start = edge_start;
                    inactive.edge_count = edge_count;
                    for (size_t p = 0; p < pad; ++p) h_tiles.push_back(inactive);
                    padded_tiles += pad;
                }
            }
            dim_ops += static_cast<unsigned long long>(q_range.second - q_range.first) *
                       static_cast<unsigned long long>(candidate_count) *
                       static_cast<unsigned long long>(dim);
        }
        const auto t_pack1 = std::chrono::high_resolution_clock::now();
        const double host_plan_ms = elapsed_ms(t_pack0, t_pack1);

        if (h_queries.empty()) {
            gpu_release_all_vectors_on_device();
            prof_logf("[PROF] cross_edges.source_exact empty=1 host_plan_ms=%.3f", host_plan_ms);
            return true;
        }
        if (h_queries.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("source_exact has more than INT_MAX query descriptors.");
        }

        const int nqueries = static_cast<int>(h_queries.size());
        const int nsegments = static_cast<int>(h_segments.size());
        UngSourceQueryDesc* d_queries = nullptr;
        UngSourceTileDesc* d_tiles = nullptr;
        UngTargetSegmentDesc* d_segments = nullptr;
        int* d_idx = nullptr;
        float* d_dis = nullptr;
        int* h_idx = nullptr;
        float* h_dis = nullptr;
        cudaStream_t stream = nullptr;
        cudaEvent_t e_h0 = nullptr, e_h1 = nullptr, e_k0 = nullptr, e_k1 = nullptr, e_d0 = nullptr, e_d1 = nullptr;

        const bool id_only_output = (cross_group_neighbor_ids || cross_group_neighbor_flat_ids);
        const size_t result_items = static_cast<size_t>(nqueries) * static_cast<size_t>(topk);
        cudaMalloc(&d_queries, h_queries.size() * sizeof(UngSourceQueryDesc));
        cudaMalloc(&d_tiles, h_tiles.size() * sizeof(UngSourceTileDesc));
        cudaMalloc(&d_segments, h_segments.size() * sizeof(UngTargetSegmentDesc));
        cudaMalloc(&d_idx, result_items * sizeof(int));
        if (!id_only_output) cudaMalloc(&d_dis, result_items * sizeof(float));
        cudaHostAlloc(&h_idx, result_items * sizeof(int), cudaHostAllocDefault);
        if (!id_only_output) cudaHostAlloc(&h_dis, result_items * sizeof(float), cudaHostAllocDefault);
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cudaEventCreate(&e_h0); cudaEventCreate(&e_h1);
        cudaEventCreate(&e_k0); cudaEventCreate(&e_k1);
        cudaEventCreate(&e_d0); cudaEventCreate(&e_d1);

        cudaEventRecord(e_h0, stream);
        cudaMemcpyAsync(d_queries, h_queries.data(), h_queries.size() * sizeof(UngSourceQueryDesc),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_tiles, h_tiles.data(), h_tiles.size() * sizeof(UngSourceTileDesc),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_segments, h_segments.data(), h_segments.size() * sizeof(UngTargetSegmentDesc),
                        cudaMemcpyHostToDevice, stream);
        cudaEventRecord(e_h1, stream);

        cudaEventRecord(e_k0, stream);
        if (source_mode == 1 && topk <= 16) {
            const size_t wmma_smem =
                ((size_t)8 * 16 + (size_t)warps * ((size_t)16 * 8 + (size_t)16 * 16)) *
                sizeof(float);
            const int ntile = static_cast<int>(h_tiles.size());
            const int grid_x = (ntile + warps - 1) / warps;
            ung_source_tf32_wmma_topk_global_kernel<<<dim3(static_cast<unsigned>(grid_x), 1u, 1u),
                                                       dim3(static_cast<unsigned>(threads), 1u, 1u),
                                                       wmma_smem, stream>>>(
                d_tiles, ntile, d_segments, g_d_all_X, g_d_all_norm, dim, topk, d_idx, d_dis);
        } else {
            const size_t exact_smem = static_cast<size_t>(warps) * static_cast<size_t>(topk) *
                                      (sizeof(float) + sizeof(int));
            ung_source_exact_topk_global_kernel<<<dim3(static_cast<unsigned>(nqueries), 1u, 1u),
                                                  dim3(static_cast<unsigned>(threads), 1u, 1u),
                                                  exact_smem, stream>>>(
                d_queries, nqueries, d_segments, g_d_all_X, g_d_all_norm, dim, topk, d_idx, d_dis);
        }
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            throw std::runtime_error(std::string("source_exact kernel launch failed: ") +
                                     cudaGetErrorString(launch_err));
        }
        cudaEventRecord(e_k1, stream);

        cudaEventRecord(e_d0, stream);
        cudaMemcpyAsync(h_idx, d_idx, result_items * sizeof(int), cudaMemcpyDeviceToHost, stream);
        if (!id_only_output) {
            cudaMemcpyAsync(h_dis, d_dis, result_items * sizeof(float), cudaMemcpyDeviceToHost, stream);
        }
        cudaEventRecord(e_d1, stream);
        cudaError_t sync_err = cudaEventSynchronize(e_d1);
        if (sync_err != cudaSuccess) {
            throw std::runtime_error(std::string("source_exact stream failed: ") +
                                     cudaGetErrorString(sync_err));
        }

        float h2d_ms = 0.f, kernel_ms = 0.f, d2h_ms = 0.f;
        cudaEventElapsedTime(&h2d_ms, e_h0, e_h1);
        cudaEventElapsedTime(&kernel_ms, e_k0, e_k1);
        cudaEventElapsedTime(&d2h_ms, e_d0, e_d1);
        if (h2d_ms_sum) *h2d_ms_sum += h2d_ms;
        if (kernel_ms_sum) *kernel_ms_sum += kernel_ms;
        if (d2h_ms_sum) *d2h_ms_sum += d2h_ms;

        const auto t_wb0 = std::chrono::high_resolution_clock::now();
        if (cross_group_neighbor_flat_ids) {
            cross_group_neighbor_flat_ids->assign(
                static_cast<size_t>(_num_points) * static_cast<size_t>(topk),
                std::numeric_limits<IdxType>::max());
        }
#pragma omp parallel for schedule(static, 4096)
        for (int qi = 0; qi < nqueries; ++qi) {
            const IdxType qid = static_cast<IdxType>(h_queries[static_cast<size_t>(qi)].qid);
            if (cross_group_neighbor_flat_ids) {
                const size_t out_base = static_cast<size_t>(qid) * static_cast<size_t>(topk);
                const size_t in_base = static_cast<size_t>(qi) * static_cast<size_t>(topk);
                for (int k = 0; k < topk; ++k) {
                    const int gid = h_idx[in_base + static_cast<size_t>(k)];
                    if (gid >= 0)
                        (*cross_group_neighbor_flat_ids)[out_base + static_cast<size_t>(k)] = static_cast<IdxType>(gid);
                }
            } else if (cross_group_neighbor_ids) {
                auto& ids = (*cross_group_neighbor_ids)[qid];
                for (int k = 0; k < topk; ++k) {
                    const int gid = h_idx[static_cast<size_t>(qi) * static_cast<size_t>(topk) + static_cast<size_t>(k)];
                    if (gid >= 0) ids.emplace_back(static_cast<IdxType>(gid));
                }
            } else {
                for (int k = 0; k < topk; ++k) {
                    const size_t pos = static_cast<size_t>(qi) * static_cast<size_t>(topk) + static_cast<size_t>(k);
                    const int gid = h_idx[pos];
                    if (gid < 0) continue;
                    cross_group_neighbors[qid].insert(static_cast<IdxType>(gid), h_dis[pos]);
                }
            }
        }
        const auto t_wb1 = std::chrono::high_resolution_clock::now();
        const double writeback_ms = elapsed_ms(t_wb0, t_wb1);
        const double total_ms = elapsed_ms(t_total0, std::chrono::high_resolution_clock::now());

        std::cout << "[cross_edges][source_exact] source_groups=" << source_groups
                  << " queries=" << nqueries
                  << " segments=" << nsegments
                  << " tiles=" << h_tiles.size()
                  << " padded_tiles=" << padded_tiles
                  << " mode=" << (source_mode == 1 && topk <= 16 ? "tf32_wmma" : "cuda_core")
                  << " warps=" << warps
                  << " prepare_h2d(ms)=" << prepare_h2d_ms
                  << " desc_h2d(ms)=" << h2d_ms
                  << " kernel(ms)=" << kernel_ms
                  << " d2h(ms)=" << d2h_ms
                  << " writeback(ms)=" << writeback_ms
                  << " id_only=" << (id_only_output ? 1 : 0)
                  << " total(ms)=" << total_ms
                  << std::endl;
        prof_logf("[PROF] cross_edges.source_exact source_groups=%zu queries=%d segments=%d tiles=%zu padded_tiles=%zu mode=%s dim_ops=%llu warps=%d host_plan_ms=%.3f prepare_h2d_ms=%.3f desc_h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f writeback_ms=%.3f total_ms=%.3f id_only=%d no_dist_output=%d",
                  source_groups, nqueries, nsegments, h_tiles.size(), padded_tiles,
                  (source_mode == 1 && topk <= 16 ? "tf32_wmma" : "cuda_core"),
                  dim_ops, warps, host_plan_ms,
                  prepare_h2d_ms, static_cast<double>(h2d_ms), static_cast<double>(kernel_ms),
                  static_cast<double>(d2h_ms), writeback_ms, total_ms,
                  id_only_output ? 1 : 0,
                  id_only_output ? 1 : 0);

        cudaEventDestroy(e_h0); cudaEventDestroy(e_h1);
        cudaEventDestroy(e_k0); cudaEventDestroy(e_k1);
        cudaEventDestroy(e_d0); cudaEventDestroy(e_d1);
        cudaStreamDestroy(stream);
        cudaFree(d_queries);
        cudaFree(d_tiles);
        cudaFree(d_segments);
        cudaFree(d_idx);
        if (d_dis) cudaFree(d_dis);
        cudaFreeHost(h_idx);
        if (h_dis) cudaFreeHost(h_dis);
        gpu_release_all_vectors_on_device();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[cross_edges][source_exact] failed: " << e.what() << std::endl;
        prof_logf("[PROF] cross_edges.source_exact_exception what=%s", e.what());
        if (prepared) gpu_release_all_vectors_on_device();
        return false;
    } catch (...) {
        std::cerr << "[cross_edges][source_exact] failed: unknown exception" << std::endl;
        prof_logf("[PROF] cross_edges.source_exact_exception what=unknown");
        if (prepared) gpu_release_all_vectors_on_device();
        return false;
    }
}

bool ANNS::UniNavGraph::build_cross_edges_generate_cuvs_bruteforce(
    std::vector<SearchQueue>& cross_group_neighbors,
    double* h2d_ms_sum,
    double* kernel_ms_sum,
    double* d2h_ms_sum)
{
#ifndef ANNS_HAVE_CUVS
    (void)cross_group_neighbors;
    (void)h2d_ms_sum;
    (void)kernel_ms_sum;
    (void)d2h_ms_sum;
    std::cerr << "[cross_edges][cuVS] build was not compiled with UNG_ENABLE_CUVS=ON." << std::endl;
    return false;
#else
    using namespace cuvs::neighbors;
    const int dim = _base_storage->get_dim();
    const int topk = static_cast<int>(_num_cross_edges);
    if (dim <= 0 || topk <= 0) return true;

    std::vector<IdxType> groups_to_process;
    groups_to_process.reserve(_num_groups);
    for (IdxType group_id = 1; group_id <= _num_groups; ++group_id) {
        if (!_label_nav_graph->in_neighbors[group_id].empty()) {
            groups_to_process.push_back(group_id);
        }
    }
    std::cout << "[cross_edges] backend=GPU cuVS brute_force per_group, target_groups="
              << groups_to_process.size() << std::endl;
    if (groups_to_process.empty()) return true;

    auto elapsed_ms = [](const std::chrono::high_resolution_clock::time_point& t0,
                         const std::chrono::high_resolution_clock::time_point& t1) -> double {
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    double prepare_h2d = 0.0;
    gpu_prepare_all_vectors_on_device(&prepare_h2d, nullptr);
    if (h2d_ms_sum) *h2d_ms_sum += prepare_h2d;

    raft::device_resources res;
    auto stream = raft::resource::get_cuda_stream(res);
    brute_force::index_params index_params;
    index_params.metric = cuvs::distance::DistanceType::L2Expanded;
    brute_force::search_params search_params;

    size_t target_groups = 0;
    size_t query_visits = 0;
    size_t valid_pairs = 0;
    unsigned long long dim_ops = 0;
    double pack_ms = 0.0;
    double q_h2d_ms = 0.0;
    double build_search_ms = 0.0;
    double result_d2h_ms = 0.0;
    double writeback_ms = 0.0;

    std::vector<IdxType> qids;
    for (IdxType group_id : groups_to_process) {
        const auto& target_range = _group_id_to_range[group_id];
        const int64_t nx = static_cast<int64_t>(target_range.second - target_range.first);
        if (nx <= 0) continue;

        size_t nq_sz = 0;
        for (auto in_gid : _label_nav_graph->in_neighbors[group_id]) {
            const auto& qrng = _group_id_to_range[in_gid];
            nq_sz += static_cast<size_t>(qrng.second - qrng.first);
        }
        if (nq_sz == 0) continue;
        const int64_t nq = static_cast<int64_t>(nq_sz);
        const int valid_k = std::min<int>(topk, static_cast<int>(nx));
        if (valid_k <= 0) continue;

        ++target_groups;
        query_visits += nq_sz;
        dim_ops += static_cast<unsigned long long>(nq) *
                   static_cast<unsigned long long>(nx) *
                   static_cast<unsigned long long>(dim);

        ensure_host_q_buffers(nq_sz, topk, dim, true);
        ensure_device_q_buffers(nq_sz, topk, dim);
        ensure_cuvs_result_buffers(static_cast<size_t>(nq) * static_cast<size_t>(topk));

        qids.clear();
        qids.reserve(nq_sz);
        const auto t_pack0 = std::chrono::high_resolution_clock::now();
        size_t qi = 0;
        for (auto in_gid : _label_nav_graph->in_neighbors[group_id]) {
            const auto& qrng = _group_id_to_range[in_gid];
            for (IdxType qid = qrng.first; qid < qrng.second; ++qid) {
                const float* q = reinterpret_cast<const float*>(_base_storage->get_vector(qid));
                std::memcpy(g_h_Q + qi * static_cast<size_t>(dim),
                            q,
                            static_cast<size_t>(dim) * sizeof(float));
                qids.push_back(qid);
                ++qi;
            }
        }
        const auto t_pack1 = std::chrono::high_resolution_clock::now();
        pack_ms += elapsed_ms(t_pack0, t_pack1);

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);

        cudaEventRecord(e0, stream);
        cudaMemcpyAsync(g_d_Q,
                        g_h_Q,
                        static_cast<size_t>(nq) * static_cast<size_t>(dim) * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream);
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        q_h2d_ms += ms;

        auto dataset_view = raft::make_device_matrix_view<const float, int64_t>(
            g_d_all_X + static_cast<size_t>(target_range.first) * static_cast<size_t>(dim),
            nx,
            static_cast<int64_t>(dim));
        auto query_view = raft::make_device_matrix_view<const float, int64_t>(
            g_d_Q,
            nq,
            static_cast<int64_t>(dim));
        auto neighbors_view = raft::make_device_matrix_view<int64_t, int64_t>(
            g_d_cuvs_idx,
            nq,
            static_cast<int64_t>(topk));
        auto distances_view = raft::make_device_matrix_view<float, int64_t>(
            g_d_dis,
            nq,
            static_cast<int64_t>(topk));

        cudaEventRecord(e0, stream);
        auto index = brute_force::build(res, index_params, dataset_view);
        brute_force::search(res, search_params, index, query_view, neighbors_view, distances_view);
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1);
        build_search_ms += ms;

        cudaEventRecord(e0, stream);
        cudaMemcpyAsync(g_h_cuvs_idx,
                        g_d_cuvs_idx,
                        static_cast<size_t>(nq) * static_cast<size_t>(topk) * sizeof(int64_t),
                        cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(g_h_dis,
                        g_d_dis,
                        static_cast<size_t>(nq) * static_cast<size_t>(topk) * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream);
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1);
        result_d2h_ms += ms;

        cudaEventDestroy(e0);
        cudaEventDestroy(e1);

        const auto t_wb0 = std::chrono::high_resolution_clock::now();
        for (int64_t row = 0; row < nq; ++row) {
            const IdxType qid = qids[static_cast<size_t>(row)];
            for (int r = 0; r < valid_k; ++r) {
                const int64_t local = g_h_cuvs_idx[static_cast<size_t>(row) * static_cast<size_t>(topk) + static_cast<size_t>(r)];
                if (local < 0 || local >= nx) continue;
                const float dist = g_h_dis[static_cast<size_t>(row) * static_cast<size_t>(topk) + static_cast<size_t>(r)];
                cross_group_neighbors[qid].insert(target_range.first + static_cast<IdxType>(local), dist);
                ++valid_pairs;
            }
        }
        const auto t_wb1 = std::chrono::high_resolution_clock::now();
        writeback_ms += elapsed_ms(t_wb0, t_wb1);
    }

    if (h2d_ms_sum) *h2d_ms_sum += q_h2d_ms;
    if (kernel_ms_sum) *kernel_ms_sum += build_search_ms;
    if (d2h_ms_sum) *d2h_ms_sum += result_d2h_ms;

    prof_logf("[PROF] cross_edges.cuvs_bruteforce target_groups=%zu query_visits=%zu dim_ops=%llu valid_pairs=%zu",
              target_groups, query_visits, dim_ops, valid_pairs);
    prof_logf("[PROF] cross_edges.cuvs_bruteforce_ms prepare_h2d=%.3f pack_q=%.3f q_h2d=%.3f build_search=%.3f d2h=%.3f writeback=%.3f",
              prepare_h2d, pack_ms, q_h2d_ms, build_search_ms, result_d2h_ms, writeback_ms);
    std::cout << "[cross_edges][cuVS] target_groups=" << target_groups
              << " query_visits=" << query_visits
              << " build_search_ms=" << build_search_ms
              << " q_h2d_ms=" << q_h2d_ms
              << " d2h_ms=" << result_d2h_ms
              << " writeback_ms=" << writeback_ms << std::endl;
    gpu_release_all_vectors_on_device();
    return true;
#endif
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
    std::vector<std::vector<IdxType>>* cross_group_neighbor_ids,
    std::vector<IdxType>* cross_group_neighbor_flat_ids,
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
    const int bench_min_nx = read_env_int("UNG_BENCH_MIN_NX", 0, 0, 1048576);
    const int bench_max_nx = read_env_int("UNG_BENCH_MAX_NX", 1048576, 0, 1048576);
    const long long bench_min_work =
        (long long)read_env_int("UNG_BENCH_MIN_WORK_M", 0, 0, 2000000000) * 1000000LL;
    size_t bench_skipped_groups = 0;

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
        const bool skip_for_bench =
            nx < bench_min_nx ||
            nx > bench_max_nx ||
            est_work < (unsigned long long)bench_min_work;
        if (skip_for_bench) {
            target_counts[gi] = 0;
            bench_skipped_groups += 1;
            continue;
        }
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

    const int flat_q_cap_mb = read_env_int("UNG_GPU_FLAT_Q_CAP_MB", 4096, 64, 131072);
    const int flat_out_cap_mb = read_env_int("UNG_GPU_FLAT_OUT_CAP_MB", 4096, 64, 131072);
    const size_t flat_q_cap_bytes = (size_t)flat_q_cap_mb * 1024ull * 1024ull;
    const size_t flat_out_cap_bytes = (size_t)flat_out_cap_mb * 1024ull * 1024ull;
    size_t max_flat_queries = flat_q_cap_bytes / ((size_t)std::max(dim, 1) * sizeof(float));
    const size_t out_bytes_per_query = (size_t)std::max(topk, 1) * (sizeof(int) + sizeof(float));
    max_flat_queries = std::min(max_flat_queries, flat_out_cap_bytes / out_bytes_per_query);
    if (max_flat_queries < 1) max_flat_queries = 1;

    const bool universal_gpu_route = (read_env_int("UNG_UNIVERSAL_GPU", 0, 0, 1) == 1);
    const bool request_double_buffer_early = (read_env_int("UNG_GPU_DOUBLE_BUFFER", 1, 0, 1) == 1);
    const bool db_nosplit = (read_env_int("UNG_GPU_DB_NOSPLIT", universal_gpu_route ? 1 : 0, 0, 1) == 1);
    if (!(request_double_buffer_early && db_nosplit) &&
        cross_group_neighbor_ids == nullptr &&
        cross_group_neighbor_flat_ids == nullptr &&
        total_queries_sz > max_flat_queries && target_group_ids.size() > 1) {
        prof_logf("[PROF] cross_edges.stream_split active_queries=%zu cap_queries=%zu q_cap_mb=%d out_cap_mb=%d groups=%zu",
                  total_queries_sz, max_flat_queries, flat_q_cap_mb, flat_out_cap_mb, target_group_ids.size());

        std::vector<IdxType> chunk_group_ids;
        chunk_group_ids.reserve(std::min(target_group_ids.size(), (size_t)4096));
        size_t chunk_queries = 0;
        size_t chunk_count = 0;

        auto flush_chunk = [&]() {
            if (chunk_group_ids.empty()) return;
            prof_logf("[PROF] cross_edges.stream_chunk index=%zu groups=%zu approx_queries=%zu",
                      chunk_count, chunk_group_ids.size(), chunk_queries);
            gpu_cross_groups_search_all_batched(chunk_group_ids, dim, topk, cross_group_neighbors, nullptr, nullptr,
                                                h2d_ms, kernel_ms, d2h_ms);
            chunk_group_ids.clear();
            chunk_queries = 0;
            ++chunk_count;
        };

        for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
            const size_t cnt = target_counts[gi];
            if (!chunk_group_ids.empty() && chunk_queries > 0 &&
                cnt > 0 && chunk_queries + cnt > max_flat_queries) {
                flush_chunk();
            }
            chunk_group_ids.push_back(target_group_ids[gi]);
            chunk_queries += cnt;
            if (cnt >= max_flat_queries) {
                flush_chunk();
            }
        }
        flush_chunk();
        prof_logf("[PROF] cross_edges.stream_split_done chunks=%zu", chunk_count);
        return;
    }

    if (total_queries_sz > (size_t)std::numeric_limits<int>::max()) {
        throw std::runtime_error("GPU cross-edge batch has more than INT_MAX queries; lower UNG_GPU_FLAT_Q_CAP_MB or split the input groups.");
    }

    int total_queries = (int)total_queries_sz;
    prof_logf("[PROF] cross_edges.bench_filter min_nx=%d max_nx=%d min_work=%lld skipped_groups=%zu active_queries=%d",
              bench_min_nx, bench_max_nx, bench_min_work, bench_skipped_groups, total_queries);

    const bool request_double_buffer = request_double_buffer_early;
    const bool use_device_query_gather_for_db = (read_env_int("UNG_Q_UPLOAD_MODE", 1, 0, 1) == 1);
    const int db_chunk_queries = read_env_int("UNG_GPU_DB_CHUNK_QUERIES", 262144, 4096, 1 << 28);
    const int db_medium_max_nx = read_env_int("UNG_GPU_DB_MEDIUM_MAX_NX", 128, 8, 256);
    const int db_large_max_nx = read_env_int("UNG_GPU_DB_LARGE_MAX_NX", universal_gpu_route ? 1048576 : 1023, 129, 1048576);
    const int db_gather_threads = read_env_int("UNG_GATHER_THREADS", 256, 64, 512);
    const int db_gather_blocks = read_env_int("UNG_GATHER_BLOCKS", 4096, 1, 65535);
    const int db_medium_warps = read_env_int("UNG_MEDIUM_GROUP_WARPS", 8, 1, 16);
    const int db_large_warps = read_env_int("UNG_LARGE_GROUP_WARPS", 2, 1, 16);
    if (request_double_buffer &&
        use_device_query_gather_for_db &&
        g_d_all_X != nullptr &&
        g_d_all_norm != nullptr &&
        cpu_tiny_group_indices.empty() &&
        topk <= 32 &&
        dim <= 4096) {
        bool supported = true;
        size_t unsupported_groups = 0;
        for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
            if (target_counts[gi] == 0) continue;
            const int nx = target_group_nx[gi];
            if (nx <= 0 || nx > db_large_max_nx) {
                supported = false;
                ++unsupported_groups;
            }
        }

        if (supported) {
            struct DbSlot {
                cudaStream_t stream = nullptr;
                cudaEvent_t h2d_start = nullptr, h2d_end = nullptr;
                cudaEvent_t kernel_start = nullptr, kernel_end = nullptr;
                cudaEvent_t d2h_start = nullptr, d2h_end = nullptr;
                cudaEvent_t done = nullptr;
                uint32_t *h_qids = nullptr;
                uint32_t *h_target_offsets = nullptr;
                int *h_idx = nullptr;
                float *h_dis = nullptr;
                uint32_t *d_qids = nullptr;
                uint32_t *d_target_offsets = nullptr;
                int *d_idx = nullptr;
                float *d_dis = nullptr;
                UngGroupQueryDesc *d_group_desc = nullptr;
                UngGroupTileDesc *d_tile_desc = nullptr;
                size_t cap_q = 0;
                size_t cap_group_desc = 0;
                size_t cap_tile_desc = 0;
                int active_queries = 0;
                std::vector<IdxType> query_global_ids;
                std::vector<IdxType> query_target_gids;
            };

            auto init_slot = [](DbSlot &s) {
                cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking);
                cudaEventCreate(&s.h2d_start);
                cudaEventCreate(&s.h2d_end);
                cudaEventCreate(&s.kernel_start);
                cudaEventCreate(&s.kernel_end);
                cudaEventCreate(&s.d2h_start);
                cudaEventCreate(&s.d2h_end);
                cudaEventCreate(&s.done);
            };
            auto free_slot = [](DbSlot &s) {
                if (s.stream) cudaStreamDestroy(s.stream);
                if (s.h2d_start) cudaEventDestroy(s.h2d_start);
                if (s.h2d_end) cudaEventDestroy(s.h2d_end);
                if (s.kernel_start) cudaEventDestroy(s.kernel_start);
                if (s.kernel_end) cudaEventDestroy(s.kernel_end);
                if (s.d2h_start) cudaEventDestroy(s.d2h_start);
                if (s.d2h_end) cudaEventDestroy(s.d2h_end);
                if (s.done) cudaEventDestroy(s.done);
                if (s.h_qids) cudaFreeHost(s.h_qids);
                if (s.h_target_offsets) cudaFreeHost(s.h_target_offsets);
                if (s.h_idx) cudaFreeHost(s.h_idx);
                if (s.h_dis) cudaFreeHost(s.h_dis);
                if (s.d_qids) cudaFree(s.d_qids);
                if (s.d_target_offsets) cudaFree(s.d_target_offsets);
                if (s.d_idx) cudaFree(s.d_idx);
                if (s.d_dis) cudaFree(s.d_dis);
                if (s.d_group_desc) cudaFree(s.d_group_desc);
                if (s.d_tile_desc) cudaFree(s.d_tile_desc);
            };
            auto ensure_slot_q = [&](DbSlot &s, size_t need_q) {
                if (need_q <= s.cap_q && s.h_qids && s.h_target_offsets && s.h_idx && s.h_dis &&
                    s.d_qids && s.d_target_offsets && s.d_idx && s.d_dis) return;
                if (s.h_qids) cudaFreeHost(s.h_qids);
                if (s.h_target_offsets) cudaFreeHost(s.h_target_offsets);
                if (s.h_idx) cudaFreeHost(s.h_idx);
                if (s.h_dis) cudaFreeHost(s.h_dis);
                if (s.d_qids) cudaFree(s.d_qids);
                if (s.d_target_offsets) cudaFree(s.d_target_offsets);
                if (s.d_idx) cudaFree(s.d_idx);
                if (s.d_dis) cudaFree(s.d_dis);
                s.cap_q = std::max(need_q, s.cap_q * 2 + 1);
                cudaHostAlloc(&s.h_qids, s.cap_q * sizeof(uint32_t), cudaHostAllocDefault);
                cudaHostAlloc(&s.h_target_offsets, s.cap_q * sizeof(uint32_t), cudaHostAllocDefault);
                cudaHostAlloc(&s.h_idx, s.cap_q * (size_t)topk * sizeof(int), cudaHostAllocDefault);
                cudaHostAlloc(&s.h_dis, s.cap_q * (size_t)topk * sizeof(float), cudaHostAllocDefault);
                cudaMalloc(&s.d_qids, s.cap_q * sizeof(uint32_t));
                cudaMalloc(&s.d_target_offsets, s.cap_q * sizeof(uint32_t));
                cudaMalloc(&s.d_idx, s.cap_q * (size_t)topk * sizeof(int));
                cudaMalloc(&s.d_dis, s.cap_q * (size_t)topk * sizeof(float));
            };
            auto ensure_slot_group_desc = [](DbSlot &s, size_t need) {
                if (need <= s.cap_group_desc && s.d_group_desc) return;
                if (s.d_group_desc) cudaFree(s.d_group_desc);
                s.cap_group_desc = std::max(need, s.cap_group_desc * 2 + 1);
                cudaMalloc(&s.d_group_desc, s.cap_group_desc * sizeof(UngGroupQueryDesc));
            };
            auto ensure_slot_tile_desc = [](DbSlot &s, size_t need) {
                if (need <= s.cap_tile_desc && s.d_tile_desc) return;
                if (s.d_tile_desc) cudaFree(s.d_tile_desc);
                s.cap_tile_desc = std::max(need, s.cap_tile_desc * 2 + 1);
                cudaMalloc(&s.d_tile_desc, s.cap_tile_desc * sizeof(UngGroupTileDesc));
            };

            DbSlot slots[2];
            init_slot(slots[0]);
            init_slot(slots[1]);
            bool slot_active[2] = {false, false};
            const bool db_global_merge_requested = read_env_int("UNG_GPU_DB_GLOBAL_MERGE", 0, 0, 1) == 1;
            const bool db_global_merge =
                db_global_merge_requested ||
                cross_group_neighbor_ids != nullptr ||
                cross_group_neighbor_flat_ids != nullptr;
            const int total_points_for_merge = (int)_base_storage->get_num_points();
            if (db_global_merge) {
                ensure_device_global_topk_buffers((size_t)total_points_for_merge, topk);
                ensure_host_q_buffers((size_t)total_points_for_merge, topk, dim, false);
                int threads = 256;
                const long long init_total = (long long)total_points_for_merge * (long long)topk;
                int init_grid_x = (int)((init_total + threads - 1) / threads);
                if (init_grid_x > 65535) init_grid_x = 65535;
                init_topk_kernel<<<dim3((unsigned)init_grid_x, 1u, 1u), dim3(threads, 1, 1)>>>(
                    total_points_for_merge, topk, g_d_global_idx, g_d_global_dis);
                cudaMemset(g_d_global_locks, 0, (size_t)total_points_for_merge * sizeof(int));
                cudaDeviceSynchronize();
            }
            double db_h2d_ms = 0.0, db_kernel_ms = 0.0, db_d2h_ms = 0.0, db_writeback_ms = 0.0;
            size_t db_chunks = 0, db_groups = 0, db_queries = 0, db_group_desc_count = 0, db_tile_desc_count = 0;

            auto finish_slot = [&](int si) {
                if (!slot_active[si]) return;
                DbSlot &s = slots[si];
                cudaEventSynchronize(s.done);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, s.h2d_start, s.h2d_end);
                db_h2d_ms += ms;
                cudaEventElapsedTime(&ms, s.kernel_start, s.kernel_end);
                db_kernel_ms += ms;
                cudaEventElapsedTime(&ms, s.d2h_start, s.d2h_end);
                db_d2h_ms += ms;

                const auto wb0 = std::chrono::high_resolution_clock::now();
                if (!db_global_merge) {
                    for (int qi = 0; qi < s.active_queries; ++qi) {
                        const IdxType qid = s.query_global_ids[(size_t)qi];
                        const IdxType tgt_gid = s.query_target_gids[(size_t)qi];
                        const IdxType tgt_off = _group_id_to_range[tgt_gid].first;
                        for (int k = 0; k < topk; ++k) {
                            const int local_id = s.h_idx[(size_t)qi * (size_t)topk + (size_t)k];
                            if (local_id < 0) continue;
                            const float dist = s.h_dis[(size_t)qi * (size_t)topk + (size_t)k];
                            cross_group_neighbors[qid].insert(tgt_off + (IdxType)local_id, dist);
                        }
                    }
                }
                const auto wb1 = std::chrono::high_resolution_clock::now();
                db_writeback_ms += std::chrono::duration<double, std::milli>(wb1 - wb0).count();
                slot_active[si] = false;
            };

            auto enqueue_chunk = [&](int si, size_t group_begin, size_t group_end) {
                DbSlot &s = slots[si];
                size_t chunk_queries = 0;
                size_t group_desc_need = 0;
                size_t tile_desc_need = 0;
                for (size_t gi = group_begin; gi < group_end; ++gi) {
                    const size_t nq = target_counts[gi];
                    if (nq == 0) continue;
                    chunk_queries += nq;
                    const int nx = target_group_nx[gi];
                    if (nx <= db_medium_max_nx) {
                        group_desc_need += nq;
                    } else {
                        tile_desc_need += (nq + 15u) / 16u;
                    }
                }
                ensure_slot_q(s, chunk_queries);
                if (group_desc_need > 0) ensure_slot_group_desc(s, group_desc_need);
                if (tile_desc_need > 0) ensure_slot_tile_desc(s, tile_desc_need);
                s.query_global_ids.clear();
                s.query_target_gids.clear();
                s.query_global_ids.reserve(chunk_queries);
                s.query_target_gids.reserve(chunk_queries);
                std::vector<UngGroupQueryDesc> group_descs;
                std::vector<UngGroupTileDesc> tile_descs;
                group_descs.reserve(group_desc_need);
                tile_descs.reserve(tile_desc_need);

                size_t qpos = 0;
                for (size_t gi = group_begin; gi < group_end; ++gi) {
                    const size_t q_start = qpos;
                    const int nx = target_group_nx[gi];
                    if (target_counts[gi] == 0 || nx <= 0) continue;
                    const IdxType tgt_gid = target_group_ids[gi];
                    for (auto in_gid : _label_nav_graph->in_neighbors[tgt_gid]) {
                        const auto &qrng = _group_id_to_range[in_gid];
                        for (IdxType vid = qrng.first; vid < qrng.second; ++vid) {
                            s.h_qids[qpos] = (uint32_t)vid;
                            s.h_target_offsets[qpos] = (uint32_t)_group_id_to_range[tgt_gid].first;
                            s.query_global_ids.push_back(vid);
                            s.query_target_gids.push_back(tgt_gid);
                            ++qpos;
                        }
                    }
                    const uint32_t x_off = (uint32_t)_group_id_to_range[tgt_gid].first;
                    if (nx <= db_medium_max_nx) {
                        for (size_t qi = q_start; qi < qpos; ++qi) {
                            UngGroupQueryDesc desc;
                            desc.qi = (uint32_t)qi;
                            desc.x_off = x_off;
                            desc.nx = (uint32_t)nx;
                            group_descs.push_back(desc);
                        }
                    } else {
                        for (size_t qi = q_start; qi < qpos; qi += 16) {
                            UngGroupTileDesc desc;
                            desc.q_start = (uint32_t)qi;
                            desc.q_count = (uint32_t)std::min<size_t>(16, qpos - qi);
                            desc.x_off = x_off;
                            desc.nx = (uint32_t)nx;
                            tile_descs.push_back(desc);
                        }
                    }
                }
                s.active_queries = (int)qpos;
                db_chunks += 1;
                db_groups += group_end - group_begin;
                db_queries += qpos;
                db_group_desc_count += group_descs.size();
                db_tile_desc_count += tile_descs.size();

                cudaEventRecord(s.h2d_start, s.stream);
                cudaMemcpyAsync(s.d_qids, s.h_qids, qpos * sizeof(uint32_t), cudaMemcpyHostToDevice, s.stream);
                cudaMemcpyAsync(s.d_target_offsets, s.h_target_offsets, qpos * sizeof(uint32_t),
                                cudaMemcpyHostToDevice, s.stream);
                if (!group_descs.empty()) {
                    cudaMemcpyAsync(s.d_group_desc, group_descs.data(),
                                    group_descs.size() * sizeof(UngGroupQueryDesc),
                                    cudaMemcpyHostToDevice, s.stream);
                }
                if (!tile_descs.empty()) {
                    cudaMemcpyAsync(s.d_tile_desc, tile_descs.data(),
                                    tile_descs.size() * sizeof(UngGroupTileDesc),
                                    cudaMemcpyHostToDevice, s.stream);
                }
                cudaEventRecord(s.h2d_end, s.stream);

                cudaEventRecord(s.kernel_start, s.stream);
                int threads = 256;
                const long long init_total = (long long)qpos * (long long)topk;
                int init_grid_x = (int)((init_total + threads - 1) / threads);
                if (init_grid_x > 65535) init_grid_x = 65535;
                init_topk_kernel<<<dim3((unsigned)init_grid_x, 1u, 1u), dim3(threads, 1, 1), 0, s.stream>>>(
                    (int)qpos, topk, s.d_idx, s.d_dis);

                if (!group_descs.empty()) {
                    int launch_warps = std::min(db_medium_warps, 16);
                    if (launch_warps < 1) launch_warps = 1;
                    dim3 mg_grid((unsigned)group_descs.size(), 1u, 1u);
                    dim3 mg_block((unsigned)(launch_warps * 32), 1u, 1u);
                    size_t mg_smem = (size_t)dim * sizeof(float)
                                   + (size_t)db_medium_max_nx * (sizeof(float) + sizeof(int));
                    ung_group_desc_topk_fused_global_kernel<<<mg_grid, mg_block, mg_smem, s.stream>>>(
                        s.d_group_desc, (int)group_descs.size(), nullptr, nullptr,
                        g_d_all_X, g_d_all_norm, s.d_qids, dim, topk, db_medium_max_nx, 1, s.d_idx, s.d_dis);
                }
                if (!tile_descs.empty()) {
                    int launch_warps = db_large_warps;
                    if (launch_warps < 1) launch_warps = 1;
                    if (launch_warps > 16) launch_warps = 16;
                    dim3 tg_grid((unsigned)((tile_descs.size() + (size_t)launch_warps - 1) / (size_t)launch_warps), 1u, 1u);
                    dim3 tg_block((unsigned)(launch_warps * 32), 1u, 1u);
                    size_t tg_smem = ((size_t)8 * 16
                                    + (size_t)launch_warps * ((size_t)16 * 8 + (size_t)8 * 16 + (size_t)16 * 16)) * sizeof(float);
                    ung_group_tile_tf32_wmma_topk_global_kernel<<<tg_grid, tg_block, tg_smem, s.stream>>>(
                        s.d_tile_desc, (int)tile_descs.size(), nullptr, nullptr,
                        g_d_all_X, g_d_all_norm, s.d_qids, dim, topk, 1, s.d_idx, s.d_dis);
                }
                if (db_global_merge) {
                    int merge_threads = 256;
                    int merge_grid_x = (int)((qpos + (size_t)merge_threads - 1) / (size_t)merge_threads);
                    if (merge_grid_x > 65535) merge_grid_x = 65535;
                    ung_merge_local_topk_rows_to_global_kernel<<<dim3((unsigned)merge_grid_x, 1u, 1u),
                                                                 dim3((unsigned)merge_threads, 1u, 1u),
                                                                 0, s.stream>>>(
                        s.d_idx,
                        s.d_dis,
                        s.d_qids,
                        s.d_target_offsets,
                        (int)qpos,
                        topk,
                        g_d_global_idx,
                        g_d_global_dis,
                        g_d_global_locks);
                }
                cudaEventRecord(s.kernel_end, s.stream);

                cudaEventRecord(s.d2h_start, s.stream);
                if (!db_global_merge) {
                    cudaMemcpyAsync(s.h_idx, s.d_idx, qpos * (size_t)topk * sizeof(int), cudaMemcpyDeviceToHost, s.stream);
                    cudaMemcpyAsync(s.h_dis, s.d_dis, qpos * (size_t)topk * sizeof(float), cudaMemcpyDeviceToHost, s.stream);
                }
                cudaEventRecord(s.d2h_end, s.stream);
                cudaEventRecord(s.done, s.stream);
                slot_active[si] = true;
            };

            size_t group_begin = 0;
            size_t chunk_index = 0;
            while (group_begin < target_group_ids.size()) {
                size_t group_end = group_begin;
                size_t chunk_q = 0;
                while (group_end < target_group_ids.size()) {
                    const size_t nq = target_counts[group_end];
                    if (group_end > group_begin && chunk_q > 0 && nq > 0 && chunk_q + nq > (size_t)db_chunk_queries) break;
                    chunk_q += nq;
                    ++group_end;
                    if (chunk_q >= (size_t)db_chunk_queries) break;
                }
                if (group_end == group_begin) ++group_end;
                const int si = (int)(chunk_index & 1u);
                finish_slot(si);
                enqueue_chunk(si, group_begin, group_end);
                group_begin = group_end;
                ++chunk_index;
            }
            finish_slot(0);
            finish_slot(1);
            if (db_global_merge) {
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                cudaEventRecord(e0);
                cudaMemcpyAsync(g_h_idx, g_d_global_idx,
                                (size_t)total_points_for_merge * (size_t)topk * sizeof(int),
                                cudaMemcpyDeviceToHost);
                if (!cross_group_neighbor_ids && !cross_group_neighbor_flat_ids) {
                    cudaMemcpyAsync(g_h_dis, g_d_global_dis,
                                    (size_t)total_points_for_merge * (size_t)topk * sizeof(float),
                                    cudaMemcpyDeviceToHost);
                }
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                db_d2h_ms += ms;
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);

                const auto wb0 = std::chrono::high_resolution_clock::now();
                if (cross_group_neighbor_flat_ids) {
                    cross_group_neighbor_flat_ids->assign(
                        (size_t)total_points_for_merge * (size_t)topk,
                        std::numeric_limits<IdxType>::max());
                }
                #pragma omp parallel for schedule(static, 1024)
                for (int qid_i = 0; qid_i < total_points_for_merge; ++qid_i) {
                    if (cross_group_neighbor_flat_ids) {
                        const size_t out_base = (size_t)qid_i * (size_t)topk;
                        for (int k = 0; k < topk; ++k) {
                            const int gid = g_h_idx[out_base + (size_t)k];
                            if (gid >= 0)
                                (*cross_group_neighbor_flat_ids)[out_base + (size_t)k] = (IdxType)gid;
                        }
                    } else if (cross_group_neighbor_ids) {
                        auto &ids = (*cross_group_neighbor_ids)[(IdxType)qid_i];
                        for (int k = 0; k < topk; ++k) {
                            const int gid = g_h_idx[(size_t)qid_i * (size_t)topk + (size_t)k];
                            if (gid >= 0) ids.emplace_back((IdxType)gid);
                        }
                    } else {
                        for (int k = 0; k < topk; ++k) {
                            const int gid = g_h_idx[(size_t)qid_i * (size_t)topk + (size_t)k];
                            if (gid < 0) continue;
                            const float dist = g_h_dis[(size_t)qid_i * (size_t)topk + (size_t)k];
                            cross_group_neighbors[(IdxType)qid_i].insert((IdxType)gid, dist);
                        }
                    }
                }
                const auto wb1 = std::chrono::high_resolution_clock::now();
                db_writeback_ms += std::chrono::duration<double, std::milli>(wb1 - wb0).count();
            }
            if (h2d_ms) *h2d_ms += db_h2d_ms;
            if (kernel_ms) *kernel_ms += db_kernel_ms;
            if (d2h_ms) *d2h_ms += db_d2h_ms;
            std::cout << "[cross_edges] double_buffer chunks=" << db_chunks
                      << " groups=" << db_groups
                      << " queries=" << db_queries
                      << " chunk_q=" << db_chunk_queries
                      << " group_desc=" << db_group_desc_count
                      << " tile_desc=" << db_tile_desc_count
                      << " H2D(ms)=" << db_h2d_ms
                      << " Kernel(ms)=" << db_kernel_ms
                      << " D2H(ms)=" << db_d2h_ms
                      << " writeback(ms)=" << db_writeback_ms
                      << std::endl;
            prof_logf("[PROF] cross_edges.double_buffer chunks=%zu groups=%zu queries=%zu chunk_q=%d group_desc=%zu tile_desc=%zu h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f writeback_ms=%.3f",
                      db_chunks, db_groups, db_queries, db_chunk_queries, db_group_desc_count, db_tile_desc_count,
                      db_h2d_ms, db_kernel_ms, db_d2h_ms, db_writeback_ms);
            prof_logf("[PROF] cross_edges.id_vector_writeback enabled=%d double_buffer=1",
                      (cross_group_neighbor_ids && db_global_merge) ? 1 : 0);
            prof_logf("[PROF] cross_edges.flat_id_writeback enabled=%d double_buffer=1",
                      (cross_group_neighbor_flat_ids && db_global_merge) ? 1 : 0);
            prof_logf("[PROF] cross_edges.db_global_merge enabled=%d", db_global_merge ? 1 : 0);
            prof_logf("[PROF] cross_edges.stage_ms count=%.3f offset=%.3f fill_q=%.3f writeback=%.3f total_queries=%d gather_q=1 singleton_fastpath=0 singleton_q=0 cpu_tiny_groups=%zu",
                      elapsed_ms(t_count_start, t_count_end), 0.0, 0.0,
                      db_writeback_ms, total_queries, cpu_tiny_group_count);
            free_slot(slots[0]);
            free_slot(slots[1]);
            return;
        }
        const bool split_unsupported =
            read_env_int("UNG_GPU_DB_SPLIT_UNSUPPORTED", 1, 0, 1) == 1;
        if (split_unsupported &&
            cross_group_neighbor_ids == nullptr &&
            cross_group_neighbor_flat_ids == nullptr &&
            unsupported_groups > 0 &&
            unsupported_groups < target_group_ids.size()) {
            std::vector<IdxType> db_group_ids;
            std::vector<IdxType> fallback_group_ids;
            db_group_ids.reserve(target_group_ids.size() - unsupported_groups);
            fallback_group_ids.reserve(unsupported_groups);
            for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
                if (target_counts[gi] == 0) {
                    continue;
                }
                const int nx = target_group_nx[gi];
                if (nx > 0 && nx <= db_large_max_nx) {
                    db_group_ids.push_back(target_group_ids[gi]);
                } else {
                    fallback_group_ids.push_back(target_group_ids[gi]);
                }
            }
            prof_logf("[PROF] cross_edges.double_buffer_split supported_groups=%zu fallback_groups=%zu max_supported_nx=%d",
                      db_group_ids.size(), fallback_group_ids.size(), db_large_max_nx);
            std::cout << "[cross_edges] double_buffer split supported_groups=" << db_group_ids.size()
                      << " fallback_groups=" << fallback_group_ids.size()
                      << " max_supported_nx=" << db_large_max_nx << std::endl;
            if (!db_group_ids.empty()) {
                gpu_cross_groups_search_all_batched(db_group_ids, dim, topk, cross_group_neighbors, nullptr, nullptr,
                                                    h2d_ms, kernel_ms, d2h_ms);
            }
            if (!fallback_group_ids.empty()) {
                gpu_cross_groups_search_all_batched(fallback_group_ids, dim, topk, cross_group_neighbors, nullptr, nullptr,
                                                    h2d_ms, kernel_ms, d2h_ms);
            }
            return;
        }
        prof_logf("[PROF] cross_edges.double_buffer_fallback unsupported_groups=%zu max_supported_nx=%d",
                  unsupported_groups, db_large_max_nx);
    }

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

    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // query 上传模式：0=老路径(Host memcpy + H2D Q)，1=新路径(H2D qid + device gather Q)。
    const bool use_device_query_gather = (read_env_int("UNG_Q_UPLOAD_MODE", 1, 0, 1) == 1);
    const bool singleton_fastpath = (read_env_int("UNG_SINGLETON_FASTPATH", 1, 0, 1) == 1);
    const bool direct_qid_fused = (read_env_int("UNG_DIRECT_QID_FUSED", 0, 0, 1) == 1);
    const bool direct_qid_all_fused = (read_env_int("UNG_DIRECT_QID_ALL_FUSED", 0, 0, 1) == 1);
    const bool id_only_writeback_requested = (read_env_int("UNG_GPU_ID_ONLY_WRITEBACK", 0, 0, 1) == 1);
    const bool gpu_global_merge_requested = (read_env_int("UNG_GPU_GLOBAL_MERGE", 0, 0, 1) == 1);
    const bool gpu_global_merge =
        gpu_global_merge_requested ||
        cross_group_neighbor_ids != nullptr ||
        cross_group_neighbor_flat_ids != nullptr;
    const bool id_vector_writeback = cross_group_neighbor_ids != nullptr && gpu_global_merge;
    const bool flat_id_writeback = cross_group_neighbor_flat_ids != nullptr && gpu_global_merge;
    if ((cross_group_neighbor_ids != nullptr || cross_group_neighbor_flat_ids != nullptr) && !gpu_global_merge) {
        prof_logf("[PROF] cross_edges.id_vector_writeback disabled_for_host_merge requested=1");
    }
    const bool id_only_writeback = (id_only_writeback_requested || flat_id_writeback) && gpu_global_merge;
    if (id_only_writeback_requested && !gpu_global_merge) {
        prof_logf("[PROF] cross_edges.id_only_writeback disabled_for_host_merge requested=1");
    }
    const bool gpu_global_merge_direct = gpu_global_merge &&
        (read_env_int("UNG_GPU_GLOBAL_MERGE_DIRECT", 1, 0, 1) == 1);
    const int gpu_global_merge_direct_max_nx =
        read_env_int("UNG_GPU_GLOBAL_MERGE_DIRECT_MAX_NX", 256, 1, 1048576);
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

    bool direct_qid_all_effective = direct_qid_all_fused;
    if (direct_qid_all_effective && singleton_query_count > 0) {
        direct_qid_all_effective = false;
        prof_logf("[PROF] cross_edges.direct_qid_all_fused disabled_for_singleton queries=%zu",
                  singleton_query_count);
    }
    if (direct_qid_all_effective) {
        const bool bucket_group_fused_probe = (read_env_int("UNG_BUCKET_GROUP_FUSED", 0, 0, 1) == 1);
        const bool small_group_fused_probe = (read_env_int("UNG_SMALL_GROUP_FUSED", 1, 0, 1) == 1);
        const int small_group_max_nx_probe = read_env_int("UNG_SMALL_GROUP_MAX_NX", 8, 2, 32);
        const bool medium_group_fused_probe = (read_env_int("UNG_MEDIUM_GROUP_FUSED", 1, 0, 1) == 1);
        const int medium_group_max_nx_probe = read_env_int("UNG_MEDIUM_GROUP_MAX_NX", 4096, 9, 8192);
        const bool large_group_fused_probe = (read_env_int("UNG_LARGE_GROUP_FUSED", 0, 0, 1) == 1);
        const int large_group_min_nx_probe = read_env_int("UNG_LARGE_GROUP_MIN_NX", 128, 16, 1048576);
        const int large_group_max_nx_probe = read_env_int("UNG_LARGE_GROUP_MAX_NX", 1023, 16, 1048576);
        const int large_group_fused_mode_probe = read_env_int("UNG_LARGE_GROUP_FUSED_MODE", 2, 0, 2);

        size_t unsupported_groups = 0;
        size_t unsupported_queries = 0;
        int first_unsupported_nx = 0;
        size_t first_unsupported_nq = 0;
        for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
            const size_t nq = target_counts[gi];
            if (nq == 0) continue;
            const int nx = target_group_nx[gi];
            if (nx <= 0) continue;
            if (singleton_fastpath && nx == 1) continue;

            const bool covered_by_bucket =
                bucket_group_fused_probe &&
                ((small_group_fused_probe && nx <= small_group_max_nx_probe) ||
                 (medium_group_fused_probe && nx > small_group_max_nx_probe && nx <= medium_group_max_nx_probe) ||
                 (large_group_fused_probe && large_group_fused_mode_probe == 2 && topk <= 16 &&
                  nx >= large_group_min_nx_probe && nx <= large_group_max_nx_probe));
            const bool covered_by_direct_large =
                large_group_fused_probe && topk <= 32 &&
                nx >= large_group_min_nx_probe && nx <= large_group_max_nx_probe;

            if (!covered_by_bucket && !covered_by_direct_large) {
                if (unsupported_groups == 0) {
                    first_unsupported_nx = nx;
                    first_unsupported_nq = nq;
                }
                ++unsupported_groups;
                unsupported_queries += nq;
            }
        }

        if (unsupported_groups > 0) {
            direct_qid_all_effective = false;
            prof_logf("[PROF] cross_edges.direct_qid_all_fused disabled_for_fallback groups=%zu queries=%zu first_nq=%zu first_nx=%d",
                      unsupported_groups, unsupported_queries, first_unsupported_nq, first_unsupported_nx);
        }
    }

    // ------------------------------------------------------------
    // 第三步：确保缓存足够大（Q/idx/dist + qnorm）
    // ------------------------------------------------------------
    const size_t host_out_rows = gpu_global_merge
        ? std::max<size_t>((size_t)total_queries, (size_t)_base_storage->get_num_points())
        : (size_t)total_queries;
    ensure_host_q_buffers(host_out_rows, topk, dim, !use_device_query_gather);
    if (!direct_qid_all_effective) {
        ensure_device_q_buffers((size_t)total_queries, topk, dim);
        ensure_qnorm((size_t)total_queries);
    } else {
        ensure_device_output_buffers((size_t)total_queries, topk);
    }
    if (use_device_query_gather) {
        ensure_device_qid_buffer((size_t)total_queries);
    }
    if (gpu_global_merge) {
        ensure_device_qid_buffer((size_t)total_queries);
        ensure_device_q_target_offsets_buffer((size_t)total_queries);
        ensure_device_global_topk_buffers((size_t)_base_storage->get_num_points(), topk);
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
    vector<uint32_t> query_target_offsets;
    if (gpu_global_merge) {
        query_target_offsets.resize((size_t)total_queries);
    }

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
        const uint32_t tgt_offset = (uint32_t)_group_id_to_range[tgt].first;

        for (auto in_gid : _label_nav_graph->in_neighbors[tgt]) {
            const auto& rng = _group_id_to_range[in_gid];
            for (IdxType vid = rng.first; vid < rng.second; ++vid) {
                if (!use_device_query_gather) {
                    const float* v = reinterpret_cast<const float*>(_base_storage->get_vector(vid));
                    std::memcpy(g_h_Q + (size_t)write_pos * dim, v, (size_t)dim * sizeof(float));
                }

                query_global_ids[(int)write_pos] = vid;
                query_target_index[(int)write_pos] = gi;
                if (gpu_global_merge) query_target_offsets[(size_t)write_pos] = tgt_offset;
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

    cudaEventRecord(e0, stream);
    if (use_device_query_gather && g_d_all_X != nullptr) {
        cudaMemcpyAsync(g_d_q_ids,
                        query_global_ids.data(),
                        (size_t)total_queries * sizeof(uint32_t),
                        cudaMemcpyHostToDevice,
                        stream);
        if (gpu_global_merge) {
            cudaMemcpyAsync(g_d_q_target_offsets,
                            query_target_offsets.data(),
                            (size_t)total_queries * sizeof(uint32_t),
                            cudaMemcpyHostToDevice,
                            stream);
        }
        if (!direct_qid_all_effective) {
            ung_gather_q_by_id_global_kernel<<<gather_blocks, gather_threads, 0, stream>>>(
                g_d_all_X, g_d_q_ids, total_queries, dim, g_d_Q);
        }
    } else {
        if (!direct_qid_all_effective) {
            cudaMemcpyAsync(g_d_Q, g_h_Q, (size_t)total_queries * dim * sizeof(float), cudaMemcpyHostToDevice, stream);
        }
        if (gpu_global_merge) {
            cudaMemcpyAsync(g_d_q_ids,
                            query_global_ids.data(),
                            (size_t)total_queries * sizeof(uint32_t),
                            cudaMemcpyHostToDevice,
                            stream);
            cudaMemcpyAsync(g_d_q_target_offsets,
                            query_target_offsets.data(),
                            (size_t)total_queries * sizeof(uint32_t),
                            cudaMemcpyHostToDevice,
                            stream);
        }
    }
    cudaEventRecord(e1, stream);
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
    cudaEventRecord(k0, stream);

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
    // Large groups are compute-heavy enough for cuBLAS to outperform the custom
    // CUDA-core dot kernel. Keep this as an env-tunable split point so small and
    // medium groups can still use the fused topK kernels below.
    const bool naive_heavy_sgemm = (read_env_int("UNG_NAIVE_HEAVY_SGEMM", 1, 0, 1) == 1);
    const int naive_heavy_nx = read_env_int("UNG_NAIVE_HEAVY_NX", 256, 16, 1048576);
    const int naive_heavy_nq = read_env_int("UNG_NAIVE_HEAVY_NQ", 256, 16, 1048576);
    const long long naive_heavy_work_thresh = (long long)read_env_int("UNG_NAIVE_HEAVY_WORK_M", 1, 1, 2000000000) * 1000000LL;
    const int topk_block_threads = read_env_int("UNG_TOPK_BLOCK_THREADS", (topk <= 8 ? 32 : (topk <= 16 ? 64 : 128)), 32, 256);
    const bool naive_group_fused = (read_env_int("UNG_NAIVE_GROUP_FUSED", 0, 0, 1) == 1);
    const int naive_group_threads = read_env_int("UNG_NAIVE_GROUP_THREADS", 128, 64, 256);
    const int singleton_threads = read_env_int("UNG_SINGLETON_THREADS", 256, 64, 512);
    const bool small_group_fused = (read_env_int("UNG_SMALL_GROUP_FUSED", 1, 0, 1) == 1);
    const int small_group_max_nx = read_env_int("UNG_SMALL_GROUP_MAX_NX", 8, 2, 32);
    const int small_group_warps = read_env_int("UNG_SMALL_GROUP_WARPS", 8, 1, 16);
    const bool medium_group_fused = (read_env_int("UNG_MEDIUM_GROUP_FUSED", 1, 0, 1) == 1);
    const int medium_group_max_nx = read_env_int("UNG_MEDIUM_GROUP_MAX_NX", 4096, 9, 8192);
    const int medium_group_warps = read_env_int("UNG_MEDIUM_GROUP_WARPS", 8, 1, 16);
    const bool medium_group_sync_debug = (read_env_int("UNG_MEDIUM_GROUP_SYNC_DEBUG", 0, 0, 1) == 1);
    // Descriptor batching is selected by the higher-level build profile for
    // the nx range where our WMMA groupGEMM + fused topK path wins; leave the
    // raw helper default off so direct callers opt into it explicitly.
    const bool bucket_group_fused = (read_env_int("UNG_BUCKET_GROUP_FUSED", 0, 0, 1) == 1);
    const bool large_group_fused = (read_env_int("UNG_LARGE_GROUP_FUSED", 0, 0, 1) == 1);
    const int large_group_min_nx = read_env_int("UNG_LARGE_GROUP_MIN_NX", 128, 16, 1048576);
    const int large_group_max_nx = read_env_int("UNG_LARGE_GROUP_MAX_NX", 1023, 16, 1048576);
    const int large_group_warps = read_env_int("UNG_LARGE_GROUP_WARPS", 2, 1, 16);
    const bool tf32_group_prefix =
        (read_env_int("UNG_TF32_GROUP_PREFIX", 0, 0, 1) == 1);
    const bool tf32_group_2d =
        (read_env_int("UNG_TF32_GROUP_2D", 0, 0, 1) == 1);
    // 0: one CTA/query CUDA-core fused, 1: one warp/query CUDA-core fused,
    // 2: TF32 Tensor Core 16x16 WMMA fused topK.
    const int large_group_fused_mode = read_env_int("UNG_LARGE_GROUP_FUSED_MODE", 2, 0, 2);
    const char* gemm_mode = use_sgemm_strided ? "sgemm_strided_batched" : (use_naive_cuda ? "naive_cuda_core" : "cublasLt");
    const bool naive_debug_dot = (read_env_int("UNG_NAIVE_DEBUG_DOT", 0, 0, 1) == 1);
    bool naive_debug_logged = false;
    prof_logf("[PROF] cross_edges.gemm_impl mode=%s(%d) force_custom=%d req_impl=%d", gemm_mode, gemm_impl, force_custom_kernel ? 1 : 0, gemm_impl_req);
    prof_logf("[PROF] cross_edges.topk_kernel_threads value=%d", topk_block_threads);
    prof_logf("[PROF] cross_edges.large_group_fused_cfg enabled=%d mode=%d min_nx=%d max_nx=%d warps=%d",
              large_group_fused ? 1 : 0, large_group_fused_mode,
              large_group_min_nx, large_group_max_nx, large_group_warps);
    prof_logf("[PROF] cross_edges.tf32_group_prefix enabled=%d group_2d=%d",
              tf32_group_prefix ? 1 : 0, tf32_group_2d ? 1 : 0);
    prof_logf("[PROF] cross_edges.medium_group_cfg max_nx=%d warps=%d regtopk=1",
              medium_group_max_nx, medium_group_warps);
    if (use_naive_cuda) {
        prof_logf("[PROF] cross_edges.naive_cfg warps_per_block=%d shared_x=%d vec4=%d x_cols=%d shared_kb=%d",
                  naive_warps_per_block, naive_shared_x ? 1 : 0, naive_vec4 ? 1 : 0, naive_x_cols_per_block_req, naive_shared_kb);
        prof_logf("[PROF] cross_edges.naive_heavy_sgemm enabled=%d nx_th=%d nq_th=%d work_th=%lld",
                  naive_heavy_sgemm ? 1 : 0, naive_heavy_nx, naive_heavy_nq, naive_heavy_work_thresh);
    }

    if (!use_naive_cuda || naive_heavy_sgemm) {
        // cuBLAS 路径，或 naive 路径里的重负载分流都需要句柄
        ensure_cublas();
        cublasSetStream(g_cublas, stream);
    }
    if (use_cublas_lt) {
        ensure_lt_workspace(LT_WORKSPACE_BYTES);
    }
    size_t tail_gemv_count = 0;
    size_t small_group_fused_group_count = 0;
    size_t small_group_fused_query_count = 0;
    size_t medium_group_fused_group_count = 0;
    size_t medium_group_fused_query_count = 0;
    size_t large_group_fused_group_count = 0;
    size_t large_group_fused_query_count = 0;
    size_t heavy_sgemm_group_count = 0;
    size_t heavy_sgemm_query_count = 0;
    size_t global_merge_direct_query_count = 0;
    size_t global_merge_direct_group_count = 0;
    std::vector<uint8_t> bucket_group_kind(target_group_ids.size(), (uint8_t)0); // 0=none,1=small,2=medium,3=tf32 tile
    std::vector<UngGroupQueryDesc> small_group_descs;
    std::vector<UngGroupQueryDesc> medium_group_descs;
    std::vector<UngGroupTileDesc> tf32_tile_descs;
    std::vector<UngGroupTileDesc> tf32_group_descs;
    std::vector<uint32_t> tf32_group_tile_offsets;

    // qnorm 与 topk 初始化都只和 query 有关，提前全量做一次，避免每个 group 启动一次 kernel
    {
        int threads = 256;
        const int warps_per_block = threads / 32;
        dim3 qnorm_grid((unsigned)((total_queries + warps_per_block - 1) / warps_per_block), 1u, 1u), qnorm_block(threads, 1, 1);
        if (!direct_qid_all_effective) {
            l2_norm_sq_kernel<<<qnorm_grid, qnorm_block, 0, stream>>>(g_d_Q, total_queries, dim, g_d_q_norm);
        }

        const long long init_total = (long long)total_queries * (long long)topk;
        int init_grid_x = (int)((init_total + threads - 1) / threads);
        if (init_grid_x > 65535) init_grid_x = 65535;
        dim3 init_grid((unsigned)init_grid_x, 1u, 1u), init_block(threads, 1, 1);
        init_topk_kernel<<<init_grid, init_block, 0, stream>>>(total_queries, topk, g_d_idx, g_d_dis);
        if (gpu_global_merge) {
            const int total_points = (int)_base_storage->get_num_points();
            const long long global_init_total = (long long)total_points * (long long)topk;
            int global_init_grid_x = (int)((global_init_total + threads - 1) / threads);
            if (global_init_grid_x > 65535) global_init_grid_x = 65535;
            init_topk_kernel<<<dim3((unsigned)global_init_grid_x, 1u, 1u), init_block, 0, stream>>>(
                total_points, topk, g_d_global_idx, g_d_global_dis);
            cudaMemsetAsync(g_d_global_locks, 0, (size_t)total_points * sizeof(int), stream);
            cudaError_t global_init_err = cudaGetLastError();
            if (global_init_err != cudaSuccess) {
                prof_logf("[ERROR] gpu_global_merge init failed: %s", cudaGetErrorString(global_init_err));
                throw std::runtime_error("gpu_global_merge init failed.");
            }
        }

        if (singleton_fastpath && singleton_query_count > 0 && g_d_all_X != nullptr && g_d_all_norm != nullptr) {
            cudaMemcpyAsync(g_d_singleton_qi,
                            singleton_qi.data(),
                            (size_t)singleton_query_count * sizeof(uint32_t),
                            cudaMemcpyHostToDevice,
                            stream);
            cudaMemcpyAsync(g_d_singleton_xoff,
                            singleton_xoff.data(),
                            (size_t)singleton_query_count * sizeof(uint32_t),
                            cudaMemcpyHostToDevice,
                            stream);
            dim3 sg_grid((unsigned)singleton_query_count, 1u, 1u), sg_block(singleton_threads, 1u, 1u);
            if (gpu_global_merge_direct && use_device_query_gather) {
                ung_singleton_top1_global_merge_kernel<<<sg_grid, sg_block, 0, stream>>>(
                    g_d_Q, g_d_q_norm, g_d_all_X, g_d_all_norm,
                    g_d_singleton_qi, g_d_singleton_xoff, g_d_q_ids,
                    (int)singleton_query_count, dim, topk,
                    g_d_global_idx, g_d_global_dis, g_d_global_locks);
                global_merge_direct_query_count += singleton_query_count;
            } else {
                ung_singleton_top1_global_kernel<<<sg_grid, sg_block, 0, stream>>>(
                    g_d_Q, g_d_q_norm, g_d_all_X, g_d_all_norm,
                    g_d_singleton_qi, g_d_singleton_xoff, (int)singleton_query_count,
                    dim, topk, g_d_idx, g_d_dis);
            }
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

            const bool tf32_tile_route =
                large_group_fused &&
                large_group_fused_mode == 2 &&
                topk <= 16 &&
                nx >= large_group_min_nx &&
                nx <= large_group_max_nx;

            bool heavy_route = false;
            if (naive_heavy_sgemm) {
                const unsigned long long work = (unsigned long long)nq_g * (unsigned long long)nx * (unsigned long long)dim;
                heavy_route = (nx >= naive_heavy_nx &&
                               nq_g >= naive_heavy_nq &&
                               work >= (unsigned long long)naive_heavy_work_thresh);
            }
            if (tf32_tile_route) {
                bucket_group_kind[gi] = (uint8_t)3;
            } else if (heavy_route) {
                continue;
            }

            if (bucket_group_kind[gi] == 3) {
                // Batched TF32 path builds tile descriptors below.
            } else if (small_group_fused && nx <= small_group_max_nx) {
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
        size_t tf32_tile_count = 0;
        size_t tf32_group_count = 0;
        for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
            if (bucket_group_kind[gi] == 3) {
                tf32_tile_count += ((size_t)target_counts[gi] + 15u) / 16u;
                tf32_group_count += 1;
            }
        }
        if (tf32_group_prefix || tf32_group_2d) {
            tf32_group_descs.reserve(tf32_group_count);
            tf32_group_tile_offsets.reserve(tf32_group_count + 1);
        } else {
            tf32_tile_descs.reserve(tf32_tile_count);
        }

        #pragma omp parallel for schedule(static, 256)
        for (int gi_i = 0; gi_i < (int)target_group_ids.size(); ++gi_i) {
            size_t gi = (size_t)gi_i;
            const uint8_t kind = bucket_group_kind[gi];
            if (kind == 0) continue;
            if (kind == 3) continue;

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

        for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
            if (bucket_group_kind[gi] != 3) continue;
            const size_t q_start = target_offsets[gi];
            const uint32_t x_off = (uint32_t)_group_id_to_range[target_group_ids[gi]].first;
            const uint32_t nx = (uint32_t)target_group_nx[gi];
            const int nq_g = (int)target_counts[gi];
            if (tf32_group_prefix || tf32_group_2d) {
                if (tf32_group_tile_offsets.empty()) tf32_group_tile_offsets.push_back(0);
                UngGroupTileDesc desc;
                desc.q_start = (uint32_t)q_start;
                desc.q_count = (uint32_t)nq_g;
                desc.x_off = x_off;
                desc.nx = nx;
                tf32_group_descs.push_back(desc);
                const uint32_t tiles = (uint32_t)(((size_t)nq_g + 15u) / 16u);
                tf32_group_tile_offsets.push_back(tf32_group_tile_offsets.back() + tiles);
            } else {
                for (int q_local = 0; q_local < nq_g; q_local += 16) {
                    UngGroupTileDesc desc;
                    desc.q_start = (uint32_t)(q_start + (size_t)q_local);
                    desc.q_count = (uint32_t)std::min(16, nq_g - q_local);
                    desc.x_off = x_off;
                    desc.nx = nx;
                    tf32_tile_descs.push_back(desc);
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

        // dQ 指向本组 Q 子矩阵：[nq_g, dim]
        const float* dQ = g_d_Q + (size_t)q_start * dim;

        // dX 指向目标组 X：[nx, dim]
        const float* dX = g_d_all_X + (size_t)x_off * dim;
        const float* dXnorm_group = (g_d_all_norm ? (g_d_all_norm + (size_t)x_off) : nullptr);
        const bool allow_group_global_merge_direct =
            gpu_global_merge_direct && use_device_query_gather &&
            nx <= gpu_global_merge_direct_max_nx;
        const bool use_large_group_fused_for_group =
            large_group_fused && topk <= 32 && dXnorm_group != nullptr &&
            nx >= large_group_min_nx && nx <= large_group_max_nx;

        if (use_naive_cuda && naive_heavy_sgemm && !use_large_group_fused_for_group) {
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

        // 本组 topK 输出在全局输出数组中的位置（也是一个连续子区间）
        int*   dBestI = g_d_idx + (size_t)q_start * topk;
        float* dBestD = g_d_dis + (size_t)q_start * topk;

        // 1) 目标组 X 的 norm^2（写到 g_d_x_norm[0..nx-1]）
        // [改动点] 这里不再对每个 group 重算 xnorm，而是直接使用全量缓存 g_d_all_norm 的对应切片
        // 原来的 per-group xnorm kernel 被移除，避免 11 万组重复 kernel 启动开销

        if (bucket_group_fused && bucket_group_kind[gi] == 3) {
            large_group_fused_group_count += 1;
            large_group_fused_query_count += (size_t)nq_g;
            continue;
        }

        if (use_large_group_fused_for_group) {
            int launch_warps = large_group_warps;
            if (launch_warps < 1) launch_warps = 1;
            if (launch_warps > 16) launch_warps = 16;
            dim3 lg_block((unsigned)(launch_warps * 32), 1u, 1u);
            if (large_group_fused_mode == 2 && topk <= 16) {
                dim3 lg_grid((unsigned)((nq_g + launch_warps * 16 - 1) / (launch_warps * 16)), 1u, 1u);
                size_t lg_smem = ((size_t)8 * 16
                                + (size_t)launch_warps * ((size_t)16 * 8 + (size_t)16 * 16)) * sizeof(float);
                if (allow_group_global_merge_direct) {
                    ung_large_group_tf32_wmma_topk_global_merge_kernel<<<lg_grid, lg_block, lg_smem, stream>>>(
                        dQ,
                        g_d_q_norm + q_start,
                        dX,
                        dXnorm_group,
                        g_d_q_ids,
                        (int)q_start,
                        x_off,
                        nq_g,
                        nx,
                        dim,
                        topk,
                        g_d_global_idx,
                        g_d_global_dis,
                        g_d_global_locks);
                    global_merge_direct_group_count += 1;
                    global_merge_direct_query_count += (size_t)nq_g;
                } else {
                    ung_large_group_tf32_wmma_topk_global_kernel<<<lg_grid, lg_block, lg_smem, stream>>>(
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
                }
            } else if (large_group_fused_mode == 1) {
                dim3 lg_grid((unsigned)((nq_g + launch_warps - 1) / launch_warps), 1u, 1u);
                if (allow_group_global_merge_direct) {
                    ung_large_group_warp_query_topk_global_merge_kernel<<<lg_grid, lg_block, 0, stream>>>(
                        dQ,
                        g_d_q_norm + q_start,
                        dX,
                        dXnorm_group,
                        g_d_q_ids,
                        (int)q_start,
                        x_off,
                        nq_g,
                        nx,
                        dim,
                        topk,
                        g_d_global_idx,
                        g_d_global_dis,
                        g_d_global_locks);
                    global_merge_direct_group_count += 1;
                    global_merge_direct_query_count += (size_t)nq_g;
                } else {
                    ung_large_group_warp_query_topk_global_kernel<<<lg_grid, lg_block, 0, stream>>>(
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
                }
            } else {
                dim3 lg_grid((unsigned)nq_g, 1u, 1u);
                size_t lg_smem = (size_t)launch_warps * (size_t)topk * (sizeof(float) + sizeof(int));
                if (allow_group_global_merge_direct) {
                    ung_large_group_warp_fused_topk_global_merge_kernel<<<lg_grid, lg_block, lg_smem, stream>>>(
                        dQ,
                        g_d_q_norm + q_start,
                        dX,
                        dXnorm_group,
                        g_d_q_ids,
                        (int)q_start,
                        x_off,
                        nq_g,
                        nx,
                        dim,
                        topk,
                        g_d_global_idx,
                        g_d_global_dis,
                        g_d_global_locks);
                    global_merge_direct_group_count += 1;
                    global_merge_direct_query_count += (size_t)nq_g;
                } else {
                    ung_large_group_warp_fused_topk_global_kernel<<<lg_grid, lg_block, lg_smem, stream>>>(
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
                }
            }
            cudaError_t large_fused_err = cudaGetLastError();
            if (large_fused_err != cudaSuccess) {
                prof_logf("[ERROR] large_group_fused launch failed: %s (nq=%d nx=%d dim=%d topk=%d warps=%d mode=%d)",
                          cudaGetErrorString(large_fused_err), nq_g, nx, dim, topk, launch_warps, large_group_fused_mode);
                throw std::runtime_error("large_group_fused launch failed.");
            }
            large_group_fused_group_count += 1;
            large_group_fused_query_count += (size_t)nq_g;
            continue;
        }

        if (group_use_naive && naive_group_fused && topk <= 32 && nx <= 1024) {
            size_t fused_smem = (size_t)dim * sizeof(float)
                              + (size_t)naive_group_threads * (size_t)topk * (sizeof(float) + sizeof(int));
            ung_group_fused_topk_global_kernel<<<(unsigned)nq_g, (unsigned)naive_group_threads, fused_smem, stream>>>(
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
            if (allow_group_global_merge_direct) {
                ung_small_group_topk_fused_global_merge_kernel<<<sg_grid, sg_block, sg_smem, stream>>>(
                    dQ,
                    g_d_q_norm + q_start,
                    dX,
                    dXnorm_group,
                    g_d_q_ids,
                    (int)q_start,
                    x_off,
                    nq_g,
                    nx,
                    dim,
                    topk,
                    small_group_max_nx,
                    g_d_global_idx,
                    g_d_global_dis,
                    g_d_global_locks);
                global_merge_direct_group_count += 1;
                global_merge_direct_query_count += (size_t)nq_g;
            } else {
                ung_small_group_topk_fused_global_kernel<<<sg_grid, sg_block, sg_smem, stream>>>(
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
            }
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
            const int group_max_nx = nx;
            size_t mg_smem = (size_t)dim * sizeof(float)
                           + (size_t)group_max_nx * (sizeof(float) + sizeof(int));
            if (allow_group_global_merge_direct) {
                ung_medium_group_topk_fused_global_merge_kernel<<<mg_grid, mg_block, mg_smem, stream>>>(
                    dQ,
                    g_d_q_norm + q_start,
                    dX,
                    dXnorm_group,
                    g_d_q_ids,
                    (int)q_start,
                    x_off,
                    nq_g,
                    nx,
                    dim,
                    topk,
                    group_max_nx,
                    g_d_global_idx,
                    g_d_global_dis,
                    g_d_global_locks);
                global_merge_direct_group_count += 1;
                global_merge_direct_query_count += (size_t)nq_g;
            } else {
                ung_medium_group_topk_fused_global_kernel<<<mg_grid, mg_block, mg_smem, stream>>>(
                    dQ,
                    g_d_q_norm + q_start,
                    dX,
                    dXnorm_group,
                    nq_g,
                    nx,
                    dim,
                    topk,
                    group_max_nx,
                    dBestI,
                    dBestD);
            }
            cudaError_t medium_fused_err = cudaGetLastError();
            if (medium_fused_err != cudaSuccess) {
                prof_logf("[ERROR] medium_group_fused launch failed: %s (gi=%zu gid=%u nq=%d nx=%d dim=%d topk=%d max_nx=%d warps=%d smem=%zu)",
                          cudaGetErrorString(medium_fused_err), gi, (unsigned)tgt_gid, nq_g, nx, dim, topk,
                          group_max_nx, launch_warps, mg_smem);
                throw std::runtime_error("medium_group_fused launch failed.");
            }
            if (medium_group_sync_debug) {
                cudaError_t medium_sync_err = cudaStreamSynchronize(stream);
                if (medium_sync_err != cudaSuccess) {
                    prof_logf("[ERROR] medium_group_fused sync failed: %s (gi=%zu gid=%u nq=%d nx=%d dim=%d topk=%d max_nx=%d warps=%d smem=%zu)",
                              cudaGetErrorString(medium_sync_err), gi, (unsigned)tgt_gid, nq_g, nx, dim, topk,
                              group_max_nx, launch_warps, mg_smem);
                    throw std::runtime_error("medium_group_fused sync failed.");
                }
            }
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
            // Keep inputs/results in float, but explicitly allow Tensor Core
            // TF32 execution for the cuBLASLt GEMM path. The cuBLAS SGEMM path
            // is already covered by cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH).
            cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F);

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
                        ung_dot_batched_naive_global_kernel<<<gdot, bdot, naive_smem, stream>>>(
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
                        prof_logf("[ERROR] naive dot kernel failed: %s (nq=%d nx=%d dim=%d tile=%d batch=%d xcols=%d shared=%d)",
                                  cudaGetErrorString(naive_err), nq_g, nx, dim, cur, batched_tiles, x_cols_per_block, use_shared_x ? 1 : 0);
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
                update_topk_from_dot_grouped_tiles_kernel<<<grid, block, smem, stream>>>(
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
                        ung_dot_batched_naive_global_kernel<<<gdot, bdot, naive_smem, stream>>>(
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
                        prof_logf("[ERROR] naive dot kernel(tail) failed: %s (nq=%d nx=%d dim=%d tile=%d xcols=%d shared=%d)",
                                  cudaGetErrorString(naive_err), nq_g, nx, dim, rem, x_cols_per_block, use_shared_x ? 1 : 0);
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
            update_topk_from_dot_tile_kernel<<<grid, block, smem, stream>>>(
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
        if (!tf32_group_descs.empty()) {
            const uint32_t total_prefix_tiles = tf32_group_tile_offsets.empty()
                ? 0u
                : tf32_group_tile_offsets.back();
            ensure_device_group_tile_desc_buffer(tf32_group_descs.size());
            cudaMemcpyAsync(g_d_group_tile_desc,
                            tf32_group_descs.data(),
                            tf32_group_descs.size() * sizeof(UngGroupTileDesc),
                            cudaMemcpyHostToDevice,
                            stream);
            int launch_warps = large_group_warps;
            if (launch_warps < 1) launch_warps = 1;
            if (launch_warps > 16) launch_warps = 16;
            dim3 pg_block((unsigned)(launch_warps * 32), 1u, 1u);
            const int tf32_group_kt = tf32_group_2d ? 4 : 1;
            size_t pg_smem = ((size_t)tf32_group_kt * 8 * 16
                            + (size_t)launch_warps * ((size_t)tf32_group_kt * 16 * 8 + (size_t)16 * 16)) * sizeof(float);
            if (tf32_group_2d) {
                uint32_t max_group_tiles = 0;
                for (size_t i = 1; i < tf32_group_tile_offsets.size(); ++i) {
                    max_group_tiles = std::max(max_group_tiles,
                                               tf32_group_tile_offsets[i] - tf32_group_tile_offsets[i - 1]);
                }
                dim3 pg_grid((unsigned)(((size_t)max_group_tiles + (size_t)launch_warps - 1) / (size_t)launch_warps),
                             (unsigned)tf32_group_descs.size(),
                             1u);
                ung_group_2d_tf32_wmma_topk_global_kernel<<<pg_grid, pg_block, pg_smem, stream>>>(
                    g_d_group_tile_desc,
                    (int)tf32_group_descs.size(),
                    g_d_Q,
                    g_d_q_norm,
                    g_d_all_X,
                    g_d_all_norm,
                    g_d_q_ids,
                    dim,
                    topk,
                    direct_qid_all_fused ? 1 : 0,
                    g_d_idx,
                    g_d_dis);
            } else {
                ensure_device_tf32_group_tile_offsets(tf32_group_tile_offsets.size());
                cudaMemcpyAsync(g_d_tf32_group_tile_offsets,
                                tf32_group_tile_offsets.data(),
                                tf32_group_tile_offsets.size() * sizeof(uint32_t),
                                cudaMemcpyHostToDevice,
                                stream);
                dim3 pg_grid((unsigned)(((size_t)total_prefix_tiles + (size_t)launch_warps - 1) / (size_t)launch_warps), 1u, 1u);
                ung_group_prefix_tf32_wmma_topk_global_kernel<<<pg_grid, pg_block, pg_smem, stream>>>(
                    g_d_group_tile_desc,
                    g_d_tf32_group_tile_offsets,
                    (int)tf32_group_descs.size(),
                    total_prefix_tiles,
                    g_d_Q,
                    g_d_q_norm,
                    g_d_all_X,
                    g_d_all_norm,
                    dim,
                    topk,
                    g_d_idx,
                    g_d_dis);
            }
            cudaError_t tf32_prefix_err = cudaGetLastError();
            if (tf32_prefix_err != cudaSuccess) {
                prof_logf("[ERROR] tf32_group_prefix launch failed: %s (groups=%zu tiles=%u dim=%d topk=%d warps=%d 2d=%d)",
                          cudaGetErrorString(tf32_prefix_err), tf32_group_descs.size(),
                          total_prefix_tiles, dim, topk, launch_warps, tf32_group_2d ? 1 : 0);
                throw std::runtime_error("tf32_group_prefix launch failed.");
            }
        }
        if (!tf32_tile_descs.empty()) {
            ensure_device_group_tile_desc_buffer(tf32_tile_descs.size());
            cudaMemcpyAsync(g_d_group_tile_desc,
                            tf32_tile_descs.data(),
                            tf32_tile_descs.size() * sizeof(UngGroupTileDesc),
                            cudaMemcpyHostToDevice,
                            stream);
            int launch_warps = large_group_warps;
            if (launch_warps < 1) launch_warps = 1;
            if (launch_warps > 16) launch_warps = 16;
            dim3 tg_grid((unsigned)((tf32_tile_descs.size() + (size_t)launch_warps - 1) / (size_t)launch_warps), 1u, 1u);
            dim3 tg_block((unsigned)(launch_warps * 32), 1u, 1u);
            size_t tg_smem = ((size_t)8 * 16
                            + (size_t)launch_warps * ((size_t)16 * 8 + (size_t)8 * 16 + (size_t)16 * 16)) * sizeof(float);
            ung_group_tile_tf32_wmma_topk_global_kernel<<<tg_grid, tg_block, tg_smem, stream>>>(
                g_d_group_tile_desc,
                (int)tf32_tile_descs.size(),
                g_d_Q,
                g_d_q_norm,
                g_d_all_X,
                g_d_all_norm,
                nullptr,
                dim,
                topk,
                0,
                g_d_idx,
                g_d_dis);
            cudaError_t tf32_batch_err = cudaGetLastError();
            if (tf32_batch_err != cudaSuccess) {
                prof_logf("[ERROR] tf32_tile_fused launch failed: %s (tiles=%zu dim=%d topk=%d warps=%d)",
                          cudaGetErrorString(tf32_batch_err), tf32_tile_descs.size(), dim, topk, launch_warps);
                throw std::runtime_error("tf32_tile_fused launch failed.");
            }
        }
        const size_t small_desc_count = small_group_descs.size();
        if (small_desc_count > 0) {
            ensure_device_group_desc_buffer(small_group_descs.size());
            cudaMemcpyAsync(g_d_group_desc,
                            small_group_descs.data(),
                            small_group_descs.size() * sizeof(UngGroupQueryDesc),
                            cudaMemcpyHostToDevice,
                            stream);
            int launch_warps = std::max(small_group_warps, std::min(small_group_max_nx, 16));
            if (launch_warps > 16) launch_warps = 16;
            if (launch_warps < 1) launch_warps = 1;
            dim3 sg_grid((unsigned)small_desc_count, 1u, 1u);
            dim3 sg_block((unsigned)(launch_warps * 32), 1u, 1u);
            size_t sg_smem = (size_t)dim * sizeof(float)
                           + (size_t)small_group_max_nx * (sizeof(float) + sizeof(int));
            ung_group_desc_topk_fused_global_kernel<<<sg_grid, sg_block, sg_smem, stream>>>(
                g_d_group_desc,
                (int)small_group_descs.size(),
                g_d_Q,
                g_d_q_norm,
                g_d_all_X,
                g_d_all_norm,
                g_d_q_ids,
                dim,
                topk,
                small_group_max_nx,
                direct_qid_fused ? 1 : 0,
                g_d_idx,
                g_d_dis);
            cudaError_t sg_err = cudaGetLastError();
            if (sg_err != cudaSuccess) {
                prof_logf("[ERROR] small group_desc launch failed: %s (desc=%zu direct_qid=%d)",
                          cudaGetErrorString(sg_err), small_group_descs.size(), direct_qid_fused ? 1 : 0);
                throw std::runtime_error("small group_desc launch failed.");
            }
        }
        const size_t medium_desc_count = medium_group_descs.size();
        if (medium_desc_count > 0) {
            ensure_device_group_desc_buffer(medium_group_descs.size());
            cudaMemcpyAsync(g_d_group_desc,
                            medium_group_descs.data(),
                            medium_group_descs.size() * sizeof(UngGroupQueryDesc),
                            cudaMemcpyHostToDevice,
                            stream);
            int launch_warps = std::min(medium_group_warps, 16);
            if (launch_warps < 1) launch_warps = 1;
            dim3 mg_grid((unsigned)medium_desc_count, 1u, 1u);
            dim3 mg_block((unsigned)(launch_warps * 32), 1u, 1u);
            size_t mg_smem = (size_t)dim * sizeof(float)
                           + (size_t)medium_group_max_nx * (sizeof(float) + sizeof(int));
            ung_group_desc_topk_fused_global_kernel<<<mg_grid, mg_block, mg_smem, stream>>>(
                g_d_group_desc,
                (int)medium_group_descs.size(),
                g_d_Q,
                g_d_q_norm,
                g_d_all_X,
                g_d_all_norm,
                g_d_q_ids,
                dim,
                topk,
                medium_group_max_nx,
                direct_qid_fused ? 1 : 0,
                g_d_idx,
                g_d_dis);
            cudaError_t mg_err = cudaGetLastError();
            if (mg_err != cudaSuccess) {
                prof_logf("[ERROR] medium group_desc launch failed: %s (desc=%zu direct_qid=%d)",
                          cudaGetErrorString(mg_err), medium_group_descs.size(), direct_qid_fused ? 1 : 0);
                throw std::runtime_error("medium group_desc launch failed.");
            }
        }
    }

    if (gpu_global_merge && global_merge_direct_query_count < (size_t)total_queries) {
        cudaError_t pre_merge_err = cudaGetLastError();
        if (pre_merge_err != cudaSuccess) {
            prof_logf("[ERROR] gpu_global_merge pre-existing CUDA error: %s", cudaGetErrorString(pre_merge_err));
            throw std::runtime_error("gpu_global_merge pre-existing CUDA error.");
        }
        const int merge_threads = 256;
        int merge_grid_x = (int)(((long long)total_queries + merge_threads - 1) / merge_threads);
        if (merge_grid_x > 65535) merge_grid_x = 65535;
        ung_merge_local_topk_rows_to_global_kernel<<<dim3((unsigned)merge_grid_x, 1u, 1u),
                                                      dim3((unsigned)merge_threads, 1u, 1u),
                                                      0, stream>>>(
            g_d_idx,
            g_d_dis,
            g_d_q_ids,
            g_d_q_target_offsets,
            total_queries,
            topk,
            g_d_global_idx,
            g_d_global_dis,
            g_d_global_locks);
        cudaError_t merge_err = cudaGetLastError();
        if (merge_err != cudaSuccess) {
            prof_logf("[ERROR] gpu_global_merge launch failed: %s", cudaGetErrorString(merge_err));
            throw std::runtime_error("gpu_global_merge launch failed.");
        }
    }

    // 与 GPU kernel 队列并行执行 CPU tiny 组，尽量把 CPU 时间隐藏在 GPU 计算后半段
    process_cpu_tiny_groups();

    cudaEventRecord(k1, stream);
    cudaError_t ksync_err = cudaEventSynchronize(k1);
    if (ksync_err != cudaSuccess) {
        prof_logf("[ERROR] cross_edges kernel stream failed before D2H: %s", cudaGetErrorString(ksync_err));
        throw std::runtime_error("cross_edges GPU kernel stream failed.");
    }

    float tker = 0.f;
    cudaEventElapsedTime(&tker, k0, k1);
    if (kernel_ms) *kernel_ms += tker;

    cudaEventDestroy(k0); cudaEventDestroy(k1);

    // ------------------------------------------------------------
    // 第八步：一次性 D2H 把 topk idx/dist 全部拷回 pinned host
    // ------------------------------------------------------------
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    cudaEventRecord(e0, stream);
    const int d2h_rows = gpu_global_merge ? (int)_base_storage->get_num_points() : total_queries;
    int* d2h_idx_src = gpu_global_merge ? g_d_global_idx : g_d_idx;
    float* d2h_dis_src = gpu_global_merge ? g_d_global_dis : g_d_dis;
    cudaError_t copy_idx_err = cudaMemcpyAsync(g_h_idx, d2h_idx_src, (size_t)d2h_rows * topk * sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (copy_idx_err != cudaSuccess) {
        prof_logf("[ERROR] cross_edges D2H idx enqueue failed: %s", cudaGetErrorString(copy_idx_err));
        throw std::runtime_error("cross_edges D2H idx enqueue failed.");
    }
    if (!id_only_writeback) {
        cudaError_t copy_dis_err = cudaMemcpyAsync(g_h_dis, d2h_dis_src, (size_t)d2h_rows * topk * sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (copy_dis_err != cudaSuccess) {
            prof_logf("[ERROR] cross_edges D2H dist enqueue failed: %s", cudaGetErrorString(copy_dis_err));
            throw std::runtime_error("cross_edges D2H dist enqueue failed.");
        }
    }
    cudaEventRecord(e1, stream);
    cudaError_t d2h_sync_err = cudaEventSynchronize(e1);
    if (d2h_sync_err != cudaSuccess) {
        prof_logf("[ERROR] cross_edges D2H stream failed: %s", cudaGetErrorString(d2h_sync_err));
        throw std::runtime_error("cross_edges D2H stream failed.");
    }

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
    if (flat_id_writeback) {
        const int total_points = (int)_base_storage->get_num_points();
        cross_group_neighbor_flat_ids->assign(
            (size_t)total_points * (size_t)topk,
            std::numeric_limits<IdxType>::max());
        #pragma omp parallel for schedule(static, 1024)
        for (int qid_i = 0; qid_i < total_points; ++qid_i) {
            const size_t base = (size_t)qid_i * (size_t)topk;
            for (int r = 0; r < topk; ++r) {
                int gid = g_h_idx[base + (size_t)r];
                if (gid >= 0)
                    (*cross_group_neighbor_flat_ids)[base + (size_t)r] = (IdxType)gid;
            }
        }
    } else if (id_vector_writeback) {
        const int total_points = (int)_base_storage->get_num_points();
        #pragma omp parallel for schedule(static, 1024)
        for (int qid_i = 0; qid_i < total_points; ++qid_i) {
            auto& ids = (*cross_group_neighbor_ids)[(IdxType)qid_i];
            for (int r = 0; r < topk; ++r) {
                int gid = g_h_idx[(size_t)qid_i * topk + r];
                if (gid >= 0) ids.emplace_back((IdxType)gid);
            }
        }
    } else if (gpu_global_merge) {
        const int total_points = (int)_base_storage->get_num_points();
        #pragma omp parallel for schedule(static, 1024)
        for (int qid_i = 0; qid_i < total_points; ++qid_i) {
            const IdxType qid = (IdxType)qid_i;
            for (int r = 0; r < topk; ++r) {
                int gid = g_h_idx[(size_t)qid_i * topk + r];
                if (gid < 0) continue;
                float dist = id_only_writeback ? (float)r : g_h_dis[(size_t)qid_i * topk + r];
                cross_group_neighbors[qid].insert((IdxType)gid, dist);
            }
        }
    } else {
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
                float dist = id_only_writeback ? (float)r : g_h_dis[(size_t)qi * topk + r];
                cross_group_neighbors[qid].insert(tgt_offset + (IdxType)j, dist);
            }
        }
    }
    const auto t_writeback_end = std::chrono::high_resolution_clock::now();

    const bool count_valid_pairs = (read_env_int("UNG_COUNT_VALID_PAIRS", 0, 0, 1) == 1);
    size_t valid_topk_pairs = 0;
    if (count_valid_pairs) {
        for (int qi = 0; qi < d2h_rows; ++qi) {
            for (int r = 0; r < topk; ++r) {
                if (g_h_idx[(size_t)qi * topk + r] >= 0) ++valid_topk_pairs;
            }
        }
    }
    prof_logf("[PROF] cross_edges.topk_valid_pairs total=%zu", valid_topk_pairs);
    prof_logf("[PROF] cross_edges.tail_gemv_calls total=%zu", tail_gemv_count);
    prof_logf("[PROF] cross_edges.small_group_fused groups=%zu queries=%zu max_nx=%d",
              small_group_fused_group_count, small_group_fused_query_count, small_group_max_nx);
    prof_logf("[PROF] cross_edges.medium_group_fused groups=%zu queries=%zu max_nx=%d",
              medium_group_fused_group_count, medium_group_fused_query_count, medium_group_max_nx);
    prof_logf("[PROF] cross_edges.large_group_fused groups=%zu queries=%zu min_nx=%d max_nx=%d warps=%d",
              large_group_fused_group_count, large_group_fused_query_count,
              large_group_min_nx, large_group_max_nx, large_group_warps);
    prof_logf("[PROF] cross_edges.bucket_group_fused enabled=%d small_desc=%zu medium_desc=%zu tf32_tile_desc=%zu",
              bucket_group_fused ? 1 : 0,
              small_group_descs.size(),
              medium_group_descs.size(),
              tf32_tile_descs.size());
    prof_logf("[PROF] cross_edges.tf32_group_prefix groups=%zu tiles=%zu",
              tf32_group_descs.size(),
              tf32_group_tile_offsets.empty() ? 0zu : (size_t)tf32_group_tile_offsets.back());
    prof_logf("[PROF] cross_edges.direct_qid_fused enabled=%d all=%d",
              direct_qid_fused ? 1 : 0, direct_qid_all_effective ? 1 : 0);
    prof_logf("[PROF] cross_edges.id_only_writeback enabled=%d", id_only_writeback ? 1 : 0);
    prof_logf("[PROF] cross_edges.id_vector_writeback enabled=%d", id_vector_writeback ? 1 : 0);
    prof_logf("[PROF] cross_edges.flat_id_writeback enabled=%d", flat_id_writeback ? 1 : 0);
    prof_logf("[PROF] cross_edges.gpu_global_merge enabled=%d rows=%d", gpu_global_merge ? 1 : 0, d2h_rows);
    prof_logf("[PROF] cross_edges.gpu_global_merge_direct enabled=%d groups=%zu queries=%zu",
              gpu_global_merge_direct ? 1 : 0,
              global_merge_direct_group_count,
              global_merge_direct_query_count);
    prof_logf("[PROF] cross_edges.gpu_global_merge_direct_max_nx value=%d",
              gpu_global_merge_direct_max_nx);
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
        if (gpu_global_merge) {
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
            auto insert_cpu = [&](std::vector<std::pair<float, int>>& cpu, float dist, int gid) {
                for (const auto& p : cpu) {
                    if (p.second == gid) return;
                }
                if ((int)cpu.size() < topk) {
                    cpu.emplace_back(dist, gid);
                    std::sort(cpu.begin(), cpu.end(), [](const auto& a, const auto& b) {
                        return a.first < b.first || (a.first == b.first && a.second < b.second);
                    });
                } else if (dist < cpu.back().first || (dist == cpu.back().first && gid < cpu.back().second)) {
                    cpu.back() = {dist, gid};
                    std::sort(cpu.begin(), cpu.end(), [](const auto& a, const auto& b) {
                        return a.first < b.first || (a.first == b.first && a.second < b.second);
                    });
                }
            };

            for (int t = 0, qi = 0; t < checked && qi < total_queries; ++t, qi += stride) {
                const IdxType qid = query_global_ids[qi];
                if ((size_t)qid >= _new_vec_id_to_group_id.size()) continue;
                const IdxType q_group = _new_vec_id_to_group_id[(size_t)qid];
                std::vector<std::pair<float, int>> cpu;
                cpu.reserve((size_t)topk);

                for (IdxType tgt_gid : target_group_ids) {
                    const auto& ins = _label_nav_graph->in_neighbors[tgt_gid];
                    if (std::find(ins.begin(), ins.end(), q_group) == ins.end()) continue;
                    const auto& rng = _group_id_to_range[tgt_gid];
                    for (IdxType xid = rng.first; xid < rng.second; ++xid) {
                        insert_cpu(cpu, l2_cpu(qid, xid), (int)xid);
                    }
                }

                std::vector<std::pair<int, float>> gpu;
                gpu.reserve((size_t)topk);
                for (int r = 0; r < topk; ++r) {
                    int idx = g_h_idx[(size_t)qid * (size_t)topk + (size_t)r];
                    float dis = g_h_dis[(size_t)qid * (size_t)topk + (size_t)r];
                    if (idx >= 0) gpu.emplace_back(idx, dis);
                }
                if ((int)gpu.size() != (int)cpu.size()) {
                    if (mismatch_logged < 3) {
                        prof_logf("[PROF] cross_edges.verify_mismatch_global qid=%lld qi=%d q_group=%lld reason=size gpu=%d cpu=%d",
                                  (long long)qid, qi, (long long)q_group, (int)gpu.size(), (int)cpu.size());
                        ++mismatch_logged;
                    }
                    ++mismatch;
                    continue;
                }
                for (size_t i = 0; i < gpu.size(); ++i) {
                    const float gd = gpu[i].second;
                    const float cd = cpu[i].first;
                    const float tol = 1e-3f * std::max(1.0f, std::fabs(cd));
                    if (gpu[i].first != cpu[i].second && std::fabs(gd - cd) > tol) {
                        if (mismatch_logged < 3) {
                            prof_logf("[PROF] cross_edges.verify_mismatch_global qid=%lld qi=%d rank=%d gpu=(%d,%.6f) cpu=(%d,%.6f)",
                                      (long long)qid, qi, (int)i, gpu[i].first, gd, cpu[i].second, cd);
                            ++mismatch_logged;
                        }
                        ++mismatch;
                        break;
                    }
                }
            }
            prof_logf("[PROF] cross_edges.verify_global sampled=%d mismatches=%d strict=%d", checked, mismatch, verify_strict ? 1 : 0);
            if (verify_strict && mismatch > 0) {
                throw std::runtime_error("UNG_GEMM global-merge verification failed: mismatch detected.");
            }
            return;
        }
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
                const float tol = 1e-3f * std::max(1.0f, std::fabs(cd));
                // TopK order is not stable when multiple target vectors have
                // nearly identical distances. Treat those as equivalent; the
                // downstream graph only needs a valid nearest-neighbor set,
                // not a deterministic tie order.
                if (std::fabs(gd - cd) > tol) {
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
