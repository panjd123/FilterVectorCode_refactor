#ifndef ANNS_GPU_CROSS_EDGE_X_STREAMING_CUH
#define ANNS_GPU_CROSS_EDGE_X_STREAMING_CUH

// X-side streaming boundary route for cross-edge search.
//
// This route avoids resident all-X device storage by streaming target vectors
// in X chunks, then processing query subchunks against each streamed X chunk.
// It is a scalability/OOM boundary path rather than the default performance
// route. Include from gpu_gemm_topk.cu after the CUDA kernels, shared global
// buffers, and buffer-management helpers it calls have been declared.

void ANNS::UniNavGraph::gpu_cross_groups_search_x_streaming(
    const std::vector<IdxType>& target_group_ids,
    int dim,
    int topk,
    std::vector<SearchQueue>& cross_group_neighbors,
    CrossEdgeBuildTiming* timing,
    const CrossEdgeGpuRuntimeConfig& gpu_route)
{
    using namespace std;
    if (target_group_ids.empty() || dim <= 0 || topk <= 0) return;
    if (topk > 16) {
        throw std::runtime_error("X-streaming GPU cross-edge currently supports topk <= 16.");
    }

    auto elapsed_ms = [](const std::chrono::high_resolution_clock::time_point& t0,
                         const std::chrono::high_resolution_clock::time_point& t1) -> double {
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    const int x_chunk_mb = gpu_route.x_stream_x_chunk_mb;
    const int q_chunk_mb = gpu_route.x_stream_q_chunk_mb;
    const int out_chunk_mb = gpu_route.x_stream_out_chunk_mb;
    const int medium_max_nx = gpu_route.db_medium_max_nx;
    const int medium_warps = gpu_route.db_medium_group_warps;
    const int large_warps = gpu_route.db_large_group_warps;
    const size_t x_cap_vecs = std::max<size_t>(1, ((size_t)x_chunk_mb * 1024ull * 1024ull) /
                                                   ((size_t)dim * sizeof(float)));
    const size_t q_cap_by_q = std::max<size_t>(1, ((size_t)q_chunk_mb * 1024ull * 1024ull) /
                                                  ((size_t)dim * sizeof(float)));
    const size_t q_cap_by_out = std::max<size_t>(1, ((size_t)out_chunk_mb * 1024ull * 1024ull) /
                                                    ((size_t)std::max(topk, 1) * (sizeof(int) + sizeof(float))));
    const size_t q_cap_vecs = std::max<size_t>(1, std::min(q_cap_by_q, q_cap_by_out));

    struct StreamGroup {
        IdxType gid = 0;
        IdxType x_global = 0;
        uint32_t x_local = 0;
        uint32_t nx = 0;
        size_t nq = 0;
    };

    vector<size_t> target_counts(target_group_ids.size(), 0);
    vector<uint32_t> target_nx(target_group_ids.size(), 0);
    const auto t_count0 = std::chrono::high_resolution_clock::now();
    for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
        const IdxType tgt_gid = target_group_ids[gi];
        const auto& xrng = _group_id_to_range[tgt_gid];
        const size_t nx = (size_t)(xrng.second - xrng.first);
        if (nx > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("X-streaming GPU cross-edge group nx exceeds uint32 descriptor range.");
        }
        target_nx[gi] = (uint32_t)nx;
        size_t nq = 0;
        for (IdxType in_gid : _label_nav_graph->in_neighbors[tgt_gid]) {
            const auto& qrng = _group_id_to_range[in_gid];
            nq += (size_t)(qrng.second - qrng.first);
        }
        target_counts[gi] = nq;
    }
    const auto t_count1 = std::chrono::high_resolution_clock::now();

    float* d_chunk_X = nullptr;
    float* d_chunk_norm = nullptr;
    size_t d_chunk_cap_vecs = 0;
    auto ensure_x_chunk = [&](size_t need_vecs) {
        if (need_vecs <= d_chunk_cap_vecs && d_chunk_X && d_chunk_norm) return;
        if (d_chunk_X) cudaFree(d_chunk_X);
        if (d_chunk_norm) cudaFree(d_chunk_norm);
        d_chunk_X = nullptr;
        d_chunk_norm = nullptr;
        d_chunk_cap_vecs = need_vecs;
        cudaError_t st = cudaMalloc(&d_chunk_X, d_chunk_cap_vecs * (size_t)dim * sizeof(float));
        if (st != cudaSuccess) {
            d_chunk_cap_vecs = 0;
            throw std::runtime_error(std::string("cudaMalloc X-stream chunk X failed: ") + cudaGetErrorString(st));
        }
        st = cudaMalloc(&d_chunk_norm, d_chunk_cap_vecs * sizeof(float));
        if (st != cudaSuccess) {
            cudaFree(d_chunk_X);
            d_chunk_X = nullptr;
            d_chunk_cap_vecs = 0;
            throw std::runtime_error(std::string("cudaMalloc X-stream chunk norm failed: ") + cudaGetErrorString(st));
        }
    };

    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    double pack_x_ms = 0.0, pack_q_ms = 0.0, h2d_total_ms = 0.0;
    double kernel_total_ms = 0.0, d2h_total_ms = 0.0, writeback_ms = 0.0;
    size_t chunk_count = 0, subchunk_count = 0, streamed_groups = 0;
    size_t streamed_queries = 0, streamed_x_vecs = 0, desc_count = 0, tile_desc_count = 0;

    auto process_chunk = [&](size_t begin, size_t end) {
        vector<StreamGroup> groups;
        groups.reserve(end - begin);
        size_t chunk_x_vecs = 0;
        for (size_t gi = begin; gi < end; ++gi) {
            if (target_counts[gi] == 0 || target_nx[gi] == 0) continue;
            const IdxType gid = target_group_ids[gi];
            const auto& xrng = _group_id_to_range[gid];
            StreamGroup g;
            g.gid = gid;
            g.x_global = xrng.first;
            g.x_local = (uint32_t)chunk_x_vecs;
            g.nx = target_nx[gi];
            g.nq = target_counts[gi];
            groups.push_back(g);
            chunk_x_vecs += (size_t)g.nx;
        }
        if (groups.empty()) return;

        ensure_host_q_buffers(std::max<size_t>(chunk_x_vecs, q_cap_vecs), topk, dim, true);
        ensure_x_chunk(chunk_x_vecs);
        const auto px0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 256)
        for (int gii = 0; gii < (int)groups.size(); ++gii) {
            const StreamGroup& g = groups[(size_t)gii];
            for (uint32_t j = 0; j < g.nx; ++j) {
                const float* src = reinterpret_cast<const float*>(
                    _base_storage->get_vector(g.x_global + (IdxType)j));
                std::memcpy(g_h_Q + ((size_t)g.x_local + (size_t)j) * (size_t)dim,
                            src,
                            (size_t)dim * sizeof(float));
            }
        }
        const auto px1 = std::chrono::high_resolution_clock::now();
        pack_x_ms += elapsed_ms(px0, px1);

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0, stream);
        cudaMemcpyAsync(d_chunk_X, g_h_Q, chunk_x_vecs * (size_t)dim * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
        const int norm_threads = 256;
        const int norm_warps = norm_threads / 32;
        l2_norm_sq_kernel<<<dim3((unsigned)((chunk_x_vecs + (size_t)norm_warps - 1) / (size_t)norm_warps), 1u, 1u),
                            dim3(norm_threads, 1, 1), 0, stream>>>(
            d_chunk_X, (int)chunk_x_vecs, dim, d_chunk_norm);
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        h2d_total_ms += ms;
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);

        for (size_t base_group = 0; base_group < groups.size();) {
            size_t q_need = 0;
            size_t group_end = base_group;
            for (; group_end < groups.size(); ++group_end) {
                const size_t nq = groups[group_end].nq;
                if (group_end > base_group && q_need > 0 && q_need + nq > q_cap_vecs) break;
                q_need += nq;
                if (q_need >= q_cap_vecs) {
                    ++group_end;
                    break;
                }
            }
            if (group_end == base_group) group_end = base_group + 1;
            q_need = 0;
            size_t group_desc_need = 0;
            size_t tile_need = 0;
            for (size_t gii = base_group; gii < group_end; ++gii) {
                q_need += groups[gii].nq;
                if ((int)groups[gii].nx <= medium_max_nx) group_desc_need += groups[gii].nq;
                else tile_need += (groups[gii].nq + 15u) / 16u;
            }
            if (q_need == 0) {
                base_group = group_end;
                continue;
            }
            if (q_need > (size_t)std::numeric_limits<int>::max()) {
                throw std::runtime_error("X-streaming GPU cross-edge q subchunk exceeds INT_MAX.");
            }
            ensure_host_q_buffers(q_need, topk, dim, true);
            ensure_device_q_buffers(q_need, topk, dim);
            ensure_qnorm(q_need);
            if (group_desc_need > 0) ensure_device_group_desc_buffer(group_desc_need);
            if (tile_need > 0) ensure_device_group_tile_desc_buffer(tile_need);

            vector<IdxType> query_global_ids(q_need);
            vector<IdxType> query_target_global(q_need);
            vector<UngGroupQueryDesc> group_descs;
            vector<UngGroupTileDesc> tile_descs;
            group_descs.reserve(group_desc_need);
            tile_descs.reserve(tile_need);

            const auto pq0 = std::chrono::high_resolution_clock::now();
            size_t qpos = 0;
            for (size_t gii = base_group; gii < group_end; ++gii) {
                const StreamGroup& g = groups[gii];
                const size_t q_start = qpos;
                for (IdxType in_gid : _label_nav_graph->in_neighbors[g.gid]) {
                    const auto& qrng = _group_id_to_range[in_gid];
                    for (IdxType qid = qrng.first; qid < qrng.second; ++qid) {
                        const float* src = reinterpret_cast<const float*>(_base_storage->get_vector(qid));
                        std::memcpy(g_h_Q + qpos * (size_t)dim, src, (size_t)dim * sizeof(float));
                        query_global_ids[qpos] = qid;
                        query_target_global[qpos] = g.x_global;
                        ++qpos;
                    }
                }
                if ((int)g.nx <= medium_max_nx) {
                    for (size_t qi = q_start; qi < qpos; ++qi) {
                        UngGroupQueryDesc desc;
                        desc.qi = (uint32_t)qi;
                        desc.x_off = g.x_local;
                        desc.nx = g.nx;
                        group_descs.push_back(desc);
                    }
                } else {
                    for (size_t qi = q_start; qi < qpos; qi += 16) {
                        UngGroupTileDesc desc;
                        desc.q_start = (uint32_t)qi;
                        desc.q_count = (uint32_t)std::min<size_t>(16, qpos - qi);
                        desc.x_off = g.x_local;
                        desc.nx = g.nx;
                        tile_descs.push_back(desc);
                    }
                }
            }
            const auto pq1 = std::chrono::high_resolution_clock::now();
            pack_q_ms += elapsed_ms(pq0, pq1);

            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0, stream);
            cudaMemcpyAsync(g_d_Q, g_h_Q, qpos * (size_t)dim * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            if (!group_descs.empty()) {
                cudaMemcpyAsync(g_d_group_desc, group_descs.data(),
                                group_descs.size() * sizeof(UngGroupQueryDesc),
                                cudaMemcpyHostToDevice, stream);
            }
            if (!tile_descs.empty()) {
                cudaMemcpyAsync(g_d_group_tile_desc, tile_descs.data(),
                                tile_descs.size() * sizeof(UngGroupTileDesc),
                                cudaMemcpyHostToDevice, stream);
            }
            cudaEventRecord(e1, stream);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms, e0, e1);
            h2d_total_ms += ms;
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);

            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0, stream);
            const int threads = 256;
            const int qnorm_warps = threads / 32;
            l2_norm_sq_kernel<<<dim3((unsigned)((qpos + (size_t)qnorm_warps - 1) / (size_t)qnorm_warps), 1u, 1u),
                                dim3(threads, 1, 1), 0, stream>>>(
                g_d_Q, (int)qpos, dim, g_d_q_norm);
            const long long init_total = (long long)qpos * (long long)topk;
            int init_grid_x = (int)((init_total + threads - 1) / threads);
            if (init_grid_x > 65535) init_grid_x = 65535;
            init_topk_kernel<<<dim3((unsigned)init_grid_x, 1u, 1u), dim3(threads, 1, 1), 0, stream>>>(
                (int)qpos, topk, g_d_idx, g_d_dis);
            if (!group_descs.empty()) {
                int launch_warps = std::min(medium_warps, 16);
                if (launch_warps < 1) launch_warps = 1;
                size_t smem = (size_t)dim * sizeof(float)
                            + (size_t)medium_max_nx * (sizeof(float) + sizeof(int));
                ung_group_desc_topk_fused_global_kernel<<<dim3((unsigned)group_descs.size(), 1u, 1u),
                                                          dim3((unsigned)(launch_warps * 32), 1u, 1u),
                                                          smem,
                                                          stream>>>(
                    g_d_group_desc, (int)group_descs.size(), g_d_Q, g_d_q_norm,
                    d_chunk_X, d_chunk_norm, nullptr, dim, topk, medium_max_nx, 0,
                    g_d_idx, g_d_dis);
            }
            if (!tile_descs.empty()) {
                int launch_warps = large_warps;
                if (launch_warps < 1) launch_warps = 1;
                if (launch_warps > 16) launch_warps = 16;
                size_t smem = ((size_t)8 * 16
                             + (size_t)launch_warps * ((size_t)16 * 8 + (size_t)8 * 16 + (size_t)16 * 16)) * sizeof(float);
                ung_group_tile_tf32_wmma_topk_global_kernel<<<
                    dim3((unsigned)((tile_descs.size() + (size_t)launch_warps - 1) / (size_t)launch_warps), 1u, 1u),
                    dim3((unsigned)(launch_warps * 32), 1u, 1u),
                    smem,
                    stream>>>(
                    g_d_group_tile_desc, (int)tile_descs.size(), g_d_Q, g_d_q_norm,
                    d_chunk_X, d_chunk_norm, nullptr, dim, topk, 0, g_d_idx, g_d_dis);
            }
            cudaEventRecord(e1, stream);
            cudaEventSynchronize(e1);
            cudaError_t kerr = cudaGetLastError();
            if (kerr != cudaSuccess) {
                throw std::runtime_error(std::string("X-streaming GPU cross-edge kernel failed: ") +
                                         cudaGetErrorString(kerr));
            }
            cudaEventElapsedTime(&ms, e0, e1);
            kernel_total_ms += ms;
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);

            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0, stream);
            cudaMemcpyAsync(g_h_idx, g_d_idx, qpos * (size_t)topk * sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(g_h_dis, g_d_dis, qpos * (size_t)topk * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(e1, stream);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms, e0, e1);
            d2h_total_ms += ms;
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);

            const auto wb0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static, 1024)
            for (int qi = 0; qi < (int)qpos; ++qi) {
                const IdxType qid = query_global_ids[(size_t)qi];
                const IdxType tgt_global = query_target_global[(size_t)qi];
                for (int k = 0; k < topk; ++k) {
                    const int local = g_h_idx[(size_t)qi * (size_t)topk + (size_t)k];
                    if (local < 0) continue;
                    cross_group_neighbors[qid].insert(tgt_global + (IdxType)local,
                                                      g_h_dis[(size_t)qi * (size_t)topk + (size_t)k]);
                }
            }
            const auto wb1 = std::chrono::high_resolution_clock::now();
            writeback_ms += elapsed_ms(wb0, wb1);

            ++subchunk_count;
            streamed_queries += qpos;
            desc_count += group_descs.size();
            tile_desc_count += tile_descs.size();
            base_group = group_end;
        }

        ++chunk_count;
        streamed_groups += groups.size();
        streamed_x_vecs += chunk_x_vecs;
    };

    size_t begin = 0;
    while (begin < target_group_ids.size()) {
        size_t end = begin;
        size_t chunk_x = 0;
        while (end < target_group_ids.size()) {
            const size_t nx = target_nx[end];
            if (target_counts[end] == 0 || nx == 0) {
                ++end;
                continue;
            }
            if (end > begin && chunk_x > 0 && chunk_x + nx > x_cap_vecs) break;
            chunk_x += nx;
            ++end;
            if (chunk_x >= x_cap_vecs) break;
        }
        if (end == begin) ++end;
        process_chunk(begin, end);
        begin = end;
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    if (d_chunk_X) cudaFree(d_chunk_X);
    if (d_chunk_norm) cudaFree(d_chunk_norm);

    if (timing) {
        timing->gpu_h2d_ms += h2d_total_ms;
        timing->gpu_kernel_ms += kernel_total_ms;
        timing->gpu_d2h_ms += d2h_total_ms;
    }

    const double count_ms = elapsed_ms(t_count0, t_count1);
    std::cout << "[cross_edges] x_stream chunks=" << chunk_count
              << " subchunks=" << subchunk_count
              << " groups=" << streamed_groups
              << " x_vecs=" << streamed_x_vecs
              << " queries=" << streamed_queries
              << " x_cap_mb=" << x_chunk_mb
              << " q_cap_mb=" << q_chunk_mb
              << " count(ms)=" << count_ms
              << " pack_x(ms)=" << pack_x_ms
              << " pack_q(ms)=" << pack_q_ms
              << " H2D+Xnorm(ms)=" << h2d_total_ms
              << " Kernel(ms)=" << kernel_total_ms
              << " D2H(ms)=" << d2h_total_ms
              << " writeback(ms)=" << writeback_ms
              << std::endl;
    prof_logf("[PROF] cross_edges.x_stream chunks=%zu subchunks=%zu groups=%zu x_vecs=%zu queries=%zu x_cap_mb=%d q_cap_mb=%d out_cap_mb=%d desc=%zu tile_desc=%zu count_ms=%.3f pack_x_ms=%.3f pack_q_ms=%.3f h2d_xnorm_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f writeback_ms=%.3f",
              chunk_count, subchunk_count, streamed_groups, streamed_x_vecs, streamed_queries,
              x_chunk_mb, q_chunk_mb, out_chunk_mb, desc_count, tile_desc_count,
              count_ms, pack_x_ms, pack_q_ms, h2d_total_ms, kernel_total_ms, d2h_total_ms, writeback_ms);
}

#endif // ANNS_GPU_CROSS_EDGE_X_STREAMING_CUH
