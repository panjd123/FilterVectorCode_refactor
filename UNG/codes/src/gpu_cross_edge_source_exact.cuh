#ifndef ANNS_GPU_CROSS_EDGE_SOURCE_EXACT_CUH
#define ANNS_GPU_CROSS_EDGE_SOURCE_EXACT_CUH

// Source-centric exact experimental host route for cross-edge search.
//
// This route is retained as a checked negative ablation/future design anchor,
// not as the default performance path. It consumes the kernels in
// gpu_cross_edge_source_exact_kernels.cuh and the shared resident-all-X cache.

bool ANNS::UniNavGraph::build_cross_edges_generate_gpu_source_exact(
    std::vector<SearchQueue>& cross_group_neighbors,
    std::vector<std::vector<IdxType>>* cross_group_neighbor_ids,
    std::vector<IdxType>* cross_group_neighbor_flat_ids,
    CrossEdgeBuildTiming& timing,
    const CrossEdgeGpuRuntimeConfig& gpu_route)
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
        const double before_prepare_h2d = timing.gpu_h2d_ms;
        gpu_prepare_all_vectors_for_cross_edge(timing, gpu_route);
        prepared = true;
        prepare_h2d_ms = timing.gpu_h2d_ms - before_prepare_h2d;

        const int warps = gpu_route.source_exact_warps;
        const int threads = warps * 32;
        const int source_mode = gpu_route.source_exact_mode;

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
            if (gpu_route.source_exact_pad_groups && source_mode == 1 && topk <= 16) {
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
        timing.gpu_h2d_ms += h2d_ms;
        timing.gpu_kernel_ms += kernel_ms;
        timing.gpu_d2h_ms += d2h_ms;

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

#endif // ANNS_GPU_CROSS_EDGE_SOURCE_EXACT_CUH
