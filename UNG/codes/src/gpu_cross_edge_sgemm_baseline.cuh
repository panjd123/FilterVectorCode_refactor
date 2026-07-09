// SGEMM/cuBLASLt/naive-dot + separate topK baseline for cross-edge build.
//
// This file is included from gpu_gemm_topk.cu after the dot/topK kernels and
// shared CUDA buffers are defined. It deliberately remains a helper fragment:
// the public backend boundary is still gpu_cross_groups_search_all_batched().

struct CrossEdgeSeparateTopkConfig {
    int tile_nx_default = 1024;
    size_t dot_tile_cap_bytes = (size_t)256 * 1024 * 1024;
    size_t dot_group_cap_bytes = (size_t)768 * 1024 * 1024;
    int gemm_group_tiles_max = 16;
    size_t lt_workspace_bytes = (size_t)64 * 1024 * 1024;

    int topk_block_threads = 64;
    int naive_warps_per_block = 4;
    bool naive_shared_x = false;
    bool naive_vec4 = true;
    int naive_x_cols_per_block = 1;
    int naive_shared_kb = 48;
    bool naive_debug_dot = false;
};

inline void run_cross_edge_separate_topk_group(
    cudaStream_t stream,
    const CrossEdgeSeparateTopkConfig& cfg,
    int dim,
    int topk,
    int nq_g,
    int nx,
    int x_off,
    size_t q_start,
    size_t gi,
    const float* dQ,
    const float* dX,
    const float* dXnorm_group,
    const float* dQnorm_group,
    int* dBestI,
    float* dBestD,
    bool group_use_sgemm,
    bool group_use_naive,
    bool group_use_lt,
    const float* h_Q,
    const float* h_all_X,
    size_t& tail_gemv_count,
    bool& naive_debug_logged) {
    int tile_nx = cfg.tile_nx_default;
    size_t bytes = (size_t)nq_g * (size_t)tile_nx * sizeof(float);
    while (bytes > cfg.dot_tile_cap_bytes && tile_nx > 128) {
        tile_nx /= 2;
        bytes = (size_t)nq_g * (size_t)tile_nx * sizeof(float);
    }

    size_t tile_bytes = (size_t)nq_g * (size_t)tile_nx * sizeof(float);
    int gemm_group_tiles = 1;
    if (tile_bytes > 0) {
        size_t by_cap = cfg.dot_group_cap_bytes / tile_bytes;
        if (by_cap == 0) by_cap = 1;
        gemm_group_tiles = (int)std::min<size_t>((size_t)cfg.gemm_group_tiles_max, by_cap);
    }
    if (gemm_group_tiles < 1) gemm_group_tiles = 1;
    ensure_dot((size_t)nq_g * (size_t)tile_nx * (size_t)gemm_group_tiles);

    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t qDesc = nullptr;
    int32_t order = CUBLASLT_ORDER_ROW;
    void* workspace = nullptr;
    size_t workspaceSize = 0;
    if (group_use_lt) {
        cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F);

        cublasOperation_t opA = CUBLAS_OP_N;
        cublasOperation_t opB = CUBLAS_OP_T;
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

        cublasLtMatrixLayoutCreate(&qDesc, CUDA_R_32F, nq_g, dim, dim);
        cublasLtMatrixLayoutSetAttribute(qDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

        workspace = g_d_lt_workspace;
        workspaceSize = std::min(g_lt_workspace_cap, cfg.lt_workspace_bytes);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    int threads = cfg.topk_block_threads;
    size_t smem = (size_t)threads * (size_t)std::min(topk, 64) * (sizeof(float) + sizeof(int));
    dim3 grid(nq_g, 1, 1), block(threads, 1, 1);

    const auto launch_naive_dot = [&](const float* dXtile,
                                      int cur,
                                      int num_tiles,
                                      int q_base_stride,
                                      const char* phase) {
        constexpr int NAIVE_WARP = 32;
        dim3 bdot(NAIVE_WARP, (unsigned)cfg.naive_warps_per_block, 1);
        constexpr int NAIVE_MAX_GRID_Y = 65535;
        const int q_chunk_cap = NAIVE_MAX_GRID_Y * (int)bdot.y;
        bool use_shared_x = cfg.naive_shared_x && (dim <= 4096);
        int x_cols_per_block = std::min(cfg.naive_x_cols_per_block, cur);
        if (x_cols_per_block < 1) x_cols_per_block = 1;
        if (use_shared_x) {
            const size_t shared_cap = (size_t)cfg.naive_shared_kb * 1024;
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
                      (unsigned)num_tiles);
            ung_dot_batched_naive_global_kernel<<<gdot, bdot, naive_smem, stream>>>(
                dQ,
                dXtile,
                q_chunk,
                cur,
                dim,
                num_tiles,
                q_base,
                nq_g,
                q_base_stride,
                use_shared_x ? 1 : 0,
                cfg.naive_vec4 ? 1 : 0,
                x_cols_per_block,
                g_d_dot);
        }
        cudaError_t naive_err = cudaGetLastError();
        if (naive_err != cudaSuccess) {
            prof_logf("[ERROR] naive dot kernel%s failed: %s (nq=%d nx=%d dim=%d tile=%d batch=%d xcols=%d shared=%d)",
                      phase, cudaGetErrorString(naive_err), nq_g, nx, dim, cur, num_tiles,
                      x_cols_per_block, use_shared_x ? 1 : 0);
            throw std::runtime_error("naive dot kernel launch failed in separate-topk path.");
        }
    };

    const auto debug_naive_dot = [&](int cur, int xb, const char* phase) {
        if (!cfg.naive_debug_dot || naive_debug_logged) return;
        int dbg_q = std::min(nq_g, 2);
        int dbg_j = std::min(cur, 3);
        std::vector<float> dbg_dot((size_t)dbg_q * (size_t)dbg_j, 0.f);
        cudaMemcpy(dbg_dot.data(), g_d_dot, dbg_dot.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (int qq = 0; qq < dbg_q; ++qq) {
            for (int jj = 0; jj < dbg_j; ++jj) {
                float cpu_dot = 0.f;
                const float* qv = h_Q + ((size_t)q_start + (size_t)qq) * (size_t)dim;
                const float* xv = h_all_X + ((size_t)x_off + (size_t)xb + (size_t)jj) * (size_t)dim;
                for (int d = 0; d < dim; ++d) cpu_dot += qv[d] * xv[d];
                float gpu_dot = dbg_dot[(size_t)qq * (size_t)dbg_j + (size_t)jj];
                prof_logf("[PROF] naive_dot_debug %s gi=%zu q=%d j=%d gpu=%.6f cpu=%.6f",
                          phase, gi, qq, jj, gpu_dot, cpu_dot);
            }
        }
        naive_debug_logged = true;
    };

    int full_tile_count = nx / tile_nx;
    int rem = nx - full_tile_count * tile_nx;
    int xb = 0;
    if (full_tile_count > 0) {
        int remaining_tiles = full_tile_count;
        while (remaining_tiles > 0) {
            int batched_tiles = std::min(remaining_tiles, gemm_group_tiles);
            int cur = tile_nx;
            const float* dXtile = dX + (size_t)xb * dim;

            if (group_use_sgemm) {
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
                    throw std::runtime_error("cublasSgemmStridedBatched failed in separate-topk full-tile path.");
                }
            } else if (group_use_naive) {
                launch_naive_dot(dXtile, cur, batched_tiles, tile_nx * dim, " full");
                debug_naive_dot(cur, xb, "full");
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
                    throw std::runtime_error("cublasLtMatmul(batch) failed in separate-topk full-tile path.");
                }
            }

            const float* xn_group_chunk = dXnorm_group ? (dXnorm_group + xb) : nullptr;
            update_topk_from_dot_grouped_tiles_kernel<<<grid, block, smem, stream>>>(
                g_d_dot,
                dQnorm_group,
                xn_group_chunk,
                nq_g, cur, batched_tiles, topk, xb,
                dBestI, dBestD);

            xb += batched_tiles * tile_nx;
            remaining_tiles -= batched_tiles;
        }
    }

    if (rem > 0) {
        const float* dXtile = dX + (size_t)xb * dim;
        bool tail_ok = false;

        if (!group_use_naive && rem == 1) {
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
                launch_naive_dot(dXtile, rem, 1, rem * dim, "(tail)");
                debug_naive_dot(rem, xb, "tail");
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
                    throw std::runtime_error("cublasSgemm failed in separate-topk tail-tile path.");
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
                    throw std::runtime_error("cublasLtMatmul failed in separate-topk tail-tile path.");
                }
            }
        }

        const float* xn_tile = dXnorm_group ? (dXnorm_group + xb) : nullptr;
        update_topk_from_dot_tile_kernel<<<grid, block, smem, stream>>>(
            g_d_dot,
            dQnorm_group,
            xn_tile,
            nq_g, rem, topk, xb,
            dBestI, dBestD);
    }

    if (group_use_lt) {
        cublasLtMatrixLayoutDestroy(qDesc);
        cublasLtMatmulDescDestroy(opDesc);
    }
}
