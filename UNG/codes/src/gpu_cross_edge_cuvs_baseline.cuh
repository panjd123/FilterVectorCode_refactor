#ifndef ANNS_GPU_CROSS_EDGE_CUVS_BASELINE_CUH
#define ANNS_GPU_CROSS_EDGE_CUVS_BASELINE_CUH

// cuVS per-group brute-force baseline for cross-edge search.
//
// This is a library-call baseline used to measure API/packing/writeback
// overhead against the fused UNG route. It intentionally writes SearchQueue
// output and does not participate in flat-id/id-vector output-boundary
// optimizations.

bool ANNS::UniNavGraph::build_cross_edges_generate_cuvs_bruteforce(
    std::vector<SearchQueue>& cross_group_neighbors,
    CrossEdgeBuildTiming& timing,
    const CrossEdgeGpuRuntimeConfig& gpu_route)
{
#ifndef ANNS_HAVE_CUVS
    (void)cross_group_neighbors;
    (void)timing;
    (void)gpu_route;
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

    const double before_prepare_h2d = timing.gpu_h2d_ms;
    gpu_prepare_all_vectors_for_cross_edge(timing, gpu_route);
    const double prepare_h2d = timing.gpu_h2d_ms - before_prepare_h2d;

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

    timing.gpu_h2d_ms += q_h2d_ms;
    timing.gpu_kernel_ms += build_search_ms;
    timing.gpu_d2h_ms += result_d2h_ms;

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

#endif // ANNS_GPU_CROSS_EDGE_CUVS_BASELINE_CUH
