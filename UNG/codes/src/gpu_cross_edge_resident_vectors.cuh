#ifndef ANNS_GPU_CROSS_EDGE_RESIDENT_VECTORS_CUH
#define ANNS_GPU_CROSS_EDGE_RESIDENT_VECTORS_CUH

// Resident all-vector upload/cache helpers for cross-edge GPU backends.
//
// This file is included from gpu_gemm_topk.cu after shared CUDA buffers and
// update kernels have been declared. It owns the UniNavGraph member functions
// that materialize base vectors as resident device arrays plus cached norms.

void ANNS::UniNavGraph::gpu_prepare_all_vectors_on_device(
    const CrossEdgeVectorUploadConfig& upload_config,
    double* h2d_ms,
    double* /*d2h_ms*/) {
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
    const bool allow_direct_hostreg = upload_config.direct_hostreg;
    const bool allow_direct_pageable_env = upload_config.direct_pageable;
    const int hostreg_max_mb = upload_config.hostreg_max_mb;
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
                static const int kProbeCount = 6;
                for (int p = 0; p < kProbeCount; ++p) {
                    IdxType idx = (IdxType)(((uint64_t)(total_points - 1) * (uint64_t)(p + 1)) /
                                            (uint64_t)(kProbeCount + 1));
                    if (idx <= 1) continue;
                    const float* vp = reinterpret_cast<const float*>(_base_storage->get_vector(idx));
                    const float* v_expect =
                        reinterpret_cast<const float*>(reinterpret_cast<const char*>(v0) + (size_t)idx * stride_bytes);
                    if (vp != v_expect) {
                        contiguous = false;
                        break;
                    }
                }
                if (contiguous && total_points > 2) {
                    const float* vlast = reinterpret_cast<const float*>(_base_storage->get_vector(total_points - 1));
                    const float* v_expect_last =
                        reinterpret_cast<const float*>(reinterpret_cast<const char*>(v0) +
                                                       (size_t)(total_points - 1) * stride_bytes);
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

    if (!g_d_all_X || g_all_cap_n < (size_t)total_points) {
        size_t new_cap_n = std::max((size_t)total_points, g_all_cap_n * 2 + 1);
        if (g_d_all_X) cudaFree(g_d_all_X);
        g_d_all_X = nullptr;
        const auto t_alloc_start = std::chrono::high_resolution_clock::now();
        cudaError_t alloc_st = cudaMalloc(&g_d_all_X, new_cap_n * (size_t)dim * sizeof(float));
        const auto t_alloc_end = std::chrono::high_resolution_clock::now();
        device_alloc_ms += elapsed_ms(t_alloc_start, t_alloc_end);
        if (alloc_st != cudaSuccess) {
            g_all_cap_n = 0;
            g_d_all_X = nullptr;
            throw std::runtime_error(std::string("cudaMalloc g_d_all_X failed in prepare_all: ") +
                                     cudaGetErrorString(alloc_st) +
                                     " bytes=" + std::to_string(new_cap_n * (size_t)dim * sizeof(float)));
        }
        g_all_cap_n = new_cap_n;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, stream);
    const float* h_src = use_direct_hostreg ? direct_src : g_h_all_X;
    cudaError_t copy_st = cudaMemcpyAsync(g_d_all_X, h_src, total_bytes, cudaMemcpyHostToDevice, stream);
    if (copy_st != cudaSuccess) {
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaStreamDestroy(stream);
        throw std::runtime_error(std::string("cudaMemcpyAsync g_d_all_X failed in prepare_all: ") +
                                 cudaGetErrorString(copy_st));
    }
    cudaEventRecord(e1, stream);
    cudaError_t sync_st = cudaEventSynchronize(e1);
    if (sync_st != cudaSuccess) {
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaStreamDestroy(stream);
        throw std::runtime_error(std::string("cudaMemcpyAsync g_d_all_X sync failed in prepare_all: ") +
                                 cudaGetErrorString(sync_st));
    }

    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    if (h2d_ms) *h2d_ms += ms;

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    ensure_all_norm((size_t)total_points);
    {
        const auto t_norm_start = std::chrono::high_resolution_clock::now();
        int threads = 256;
        const int warps_per_block = threads / 32;
        dim3 grid((unsigned)(((int)total_points + warps_per_block - 1) / warps_per_block), 1u, 1u);
        dim3 block(threads, 1, 1);
        l2_norm_sq_kernel<<<grid, block, 0, stream>>>(g_d_all_X, (int)total_points, dim, g_d_all_norm);
        cudaError_t norm_st = cudaGetLastError();
        if (norm_st != cudaSuccess) {
            cudaStreamDestroy(stream);
            throw std::runtime_error(std::string("l2_norm_sq_kernel launch failed in prepare_all: ") +
                                     cudaGetErrorString(norm_st));
        }
        const auto t_norm_end = std::chrono::high_resolution_clock::now();
        norm_launch_ms = elapsed_ms(t_norm_start, t_norm_end);
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

void ANNS::UniNavGraph::gpu_prepare_all_vectors_for_cross_edge(
    CrossEdgeBuildTiming& timing,
    const CrossEdgeGpuRuntimeConfig& gpu_route) {
    double h2d_ms = 0.0;
    gpu_prepare_all_vectors_on_device(gpu_route.vector_upload, &h2d_ms, nullptr);
    timing.gpu_h2d_ms += h2d_ms;
}

void ANNS::UniNavGraph::gpu_release_all_vectors_on_device() {
    if (g_d_all_X) { cudaFree(g_d_all_X); g_d_all_X = nullptr; }
    if (g_h_all_X) { cudaFreeHost(g_h_all_X); g_h_all_X = nullptr; }
    if (g_d_all_norm) { cudaFree(g_d_all_norm); g_d_all_norm = nullptr; }
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

#endif
