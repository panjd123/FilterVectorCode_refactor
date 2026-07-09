// Cross-edge GPU buffer/cache state.
//
// This file is intentionally included inside gpu_gemm_topk.cu's anonymous
// namespace. It keeps the existing internal-linkage globals and lazy allocation
// semantics, while separating memory/cache ownership from route orchestration
// and kernel definitions.

// ============================================================
// Q/output reusable buffers (Pinned Host + Device)
// ============================================================
static float* g_h_Q   = nullptr;  // pinned: flattened query vectors
static int*   g_h_idx = nullptr;  // pinned: topK index output
static float* g_h_dis = nullptr;  // pinned: topK distance output

static float* g_d_Q   = nullptr;  // device: query vectors
static int*   g_d_idx = nullptr;  // device: topK index output
static float* g_d_dis = nullptr;  // device: topK distance output

static size_t g_host_cap_nq = 0;
static int    g_host_cap_k = 0;
static int    g_host_dim_cap = 0;

static size_t g_dev_cap_nq = 0;
static int    g_dev_cap_k = 0;
static int    g_dev_dim_cap = 0;
static uint32_t* g_d_q_ids = nullptr;
static size_t g_qid_cap_nq = 0;
static uint32_t* g_d_q_target_offsets = nullptr;
static size_t g_q_target_offsets_cap_nq = 0;
static int* g_d_global_idx = nullptr;
static float* g_d_global_dis = nullptr;
static int* g_d_global_locks = nullptr;
static size_t g_global_topk_cap_n = 0;
static int g_global_topk_cap_k = 0;
static uint32_t* g_d_singleton_qi = nullptr;
static uint32_t* g_d_singleton_xoff = nullptr;
static size_t g_singleton_cap_nq = 0;
#ifdef ANNS_HAVE_CUVS
static int64_t* g_d_cuvs_idx = nullptr;
static int64_t* g_h_cuvs_idx = nullptr;
static size_t g_cuvs_idx_cap = 0;
#endif
static UngGroupQueryDesc* g_d_group_desc = nullptr;
static size_t g_group_desc_cap = 0;
static UngGroupTileDesc* g_d_group_tile_desc = nullptr;
static size_t g_group_tile_desc_cap = 0;
static uint32_t* g_d_tf32_group_tile_offsets = nullptr;
static size_t g_tf32_group_tile_offsets_cap = 0;

inline void ensure_host_q_buffers(size_t need_nq, int need_k, int dim, bool need_q) {
    const bool need = (need_nq > g_host_cap_nq) || (need_k > g_host_cap_k) || (dim != g_host_dim_cap)
                   || (need_q && g_h_Q == nullptr);
    if (!need) return;

    if (g_h_Q) { cudaFreeHost(g_h_Q); g_h_Q = nullptr; }
    if (g_h_idx) cudaFreeHost(g_h_idx);
    if (g_h_dis) cudaFreeHost(g_h_dis);

    g_host_cap_nq = std::max(need_nq, g_host_cap_nq * 2 + 1);
    g_host_cap_k = std::max(need_k, g_host_cap_k * 2 + 1);
    g_host_dim_cap = dim;

    if (need_q) {
        cudaHostAlloc(&g_h_Q, (size_t)g_host_cap_nq * dim * sizeof(float), cudaHostAllocDefault);
    }
    cudaHostAlloc(&g_h_idx, (size_t)g_host_cap_nq * g_host_cap_k * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&g_h_dis, (size_t)g_host_cap_nq * g_host_cap_k * sizeof(float), cudaHostAllocDefault);
}

inline void ensure_device_q_buffers(size_t need_nq, int need_k, int dim) {
    const bool need = (need_nq > g_dev_cap_nq) || (need_k > g_dev_cap_k) || (dim != g_dev_dim_cap)
                   || (g_d_Q == nullptr) || (g_d_idx == nullptr) || (g_d_dis == nullptr);
    if (!need) return;

    // Device Q can be tens of GB on large split chunks. Do not use geometric
    // growth here: a tiny chunk-size increase can otherwise double the buffer
    // and exhaust GPU memory on x400-scale runs.
    g_dev_cap_nq = need_nq;
    g_dev_cap_k = need_k;
    g_dev_dim_cap = dim;

    if (g_d_Q) cudaFree(g_d_Q);
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
    const bool need = (need_n > g_global_topk_cap_n) || (need_k > g_global_topk_cap_k)
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
// Resident all-X cache and norm cache.
// ============================================================
static float* g_h_all_X = nullptr;
static float* g_d_all_X = nullptr;
static size_t g_all_cap_n = 0;
static const float* g_hostreg_all_x_ptr = nullptr;
static size_t g_hostreg_all_x_bytes = 0;

static float* g_d_all_norm = nullptr;
static size_t g_all_norm_cap_n = 0;

inline void ensure_all_norm(size_t need_n) {
    if (need_n <= g_all_norm_cap_n && g_d_all_norm) return;
    if (g_d_all_norm) cudaFree(g_d_all_norm);
    g_all_norm_cap_n = std::max(need_n, g_all_norm_cap_n * 2 + 1);
    cudaMalloc(&g_d_all_norm, g_all_norm_cap_n * sizeof(float));
}

// ============================================================
// GEMM/topK scratch buffers and cuBLAS handles.
// ============================================================
static float* g_d_q_norm = nullptr;
static size_t g_qnorm_cap = 0;

static float* g_d_x_norm = nullptr;
static size_t g_xnorm_cap = 0;

static float* g_d_dot = nullptr;
static size_t g_dot_cap = 0;

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

static cublasHandle_t   g_cublas = nullptr;
static cublasLtHandle_t g_cublasLt = nullptr;

inline void ensure_cublas() {
    static bool inited = false;
    if (inited) return;

    cublasCreate(&g_cublas);
    cublasLtCreate(&g_cublasLt);
    cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);

    inited = true;
}
