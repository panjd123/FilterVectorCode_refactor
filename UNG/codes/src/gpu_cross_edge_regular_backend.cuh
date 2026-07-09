// Backend adapter helpers for the regular all-batched cross-edge route.
//
// This file keeps the regular backend lifecycle out of gpu_gemm_topk.cu:
// singleton prepass planning, device state initialization, singleton kernel
// launch, and final local-to-global merge. Dispatch and output policy remain
// in their dedicated helpers.

#ifndef ANNS_GPU_CROSS_EDGE_REGULAR_BACKEND_CUH
#define ANNS_GPU_CROSS_EDGE_REGULAR_BACKEND_CUH

struct CrossEdgeSingletonPrepass {
    std::vector<uint32_t> query_rows;
    std::vector<uint32_t> target_offsets;

    size_t size() const { return query_rows.size(); }
};

inline CrossEdgeSingletonPrepass collect_cross_edge_singleton_prepass(
    bool enabled,
    const std::vector<ANNS::IdxType>& target_group_ids,
    const std::vector<size_t>& target_counts,
    const std::vector<int>& target_group_nx,
    const std::vector<size_t>& target_offsets,
    const std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>>& group_id_to_range,
    int total_queries)
{
    CrossEdgeSingletonPrepass prepass;
    if (!enabled) {
        return prepass;
    }

    prepass.query_rows.reserve((size_t)total_queries / 2 + 1);
    prepass.target_offsets.reserve((size_t)total_queries / 2 + 1);
    for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
        if (target_group_nx[gi] != 1 || target_counts[gi] == 0) continue;
        const uint32_t xoff = (uint32_t)group_id_to_range[target_group_ids[gi]].first;
        const size_t q0 = target_offsets[gi];
        const size_t q1 = q0 + target_counts[gi];
        for (size_t qi = q0; qi < q1; ++qi) {
            prepass.query_rows.push_back((uint32_t)qi);
            prepass.target_offsets.push_back(xoff);
        }
    }
    return prepass;
}

inline void ensure_cross_edge_regular_backend_buffers(
    size_t host_out_rows,
    int total_queries,
    int total_points,
    int topk,
    int dim,
    bool use_device_query_gather,
    bool direct_qid_all_effective,
    bool gpu_global_merge,
    size_t singleton_query_count)
{
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
        ensure_device_global_topk_buffers((size_t)total_points, topk);
    }
    if (singleton_query_count > 0) {
        ensure_device_singleton_buffers(singleton_query_count);
    }
}

inline void initialize_cross_edge_regular_topk_state(
    cudaStream_t stream,
    int total_queries,
    int total_points,
    int dim,
    int topk,
    bool direct_qid_all_effective,
    bool gpu_global_merge)
{
    const int threads = 256;
    const int warps_per_block = threads / 32;
    dim3 qnorm_grid((unsigned)((total_queries + warps_per_block - 1) / warps_per_block), 1u, 1u);
    dim3 qnorm_block(threads, 1, 1);
    if (!direct_qid_all_effective) {
        l2_norm_sq_kernel<<<qnorm_grid, qnorm_block, 0, stream>>>(g_d_Q, total_queries, dim, g_d_q_norm);
    }

    const long long init_total = (long long)total_queries * (long long)topk;
    int init_grid_x = (int)((init_total + threads - 1) / threads);
    if (init_grid_x > 65535) init_grid_x = 65535;
    dim3 init_block(threads, 1, 1);
    init_topk_kernel<<<dim3((unsigned)init_grid_x, 1u, 1u), init_block, 0, stream>>>(
        total_queries, topk, g_d_idx, g_d_dis);

    if (!gpu_global_merge) {
        return;
    }

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

inline void launch_cross_edge_singleton_prepass(
    cudaStream_t stream,
    const CrossEdgeSingletonPrepass& prepass,
    int dim,
    int topk,
    const CrossEdgeRegularRouteConfig& route_cfg,
    CrossEdgeRegularDispatchCounters& dispatch_counters)
{
    const size_t singleton_query_count = prepass.size();
    if (singleton_query_count == 0 || g_d_all_X == nullptr || g_d_all_norm == nullptr) {
        return;
    }

    cudaMemcpyAsync(g_d_singleton_qi,
                    prepass.query_rows.data(),
                    singleton_query_count * sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(g_d_singleton_xoff,
                    prepass.target_offsets.data(),
                    singleton_query_count * sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    stream);

    dim3 sg_grid((unsigned)singleton_query_count, 1u, 1u);
    dim3 sg_block(route_cfg.singleton_threads, 1u, 1u);
    if (route_cfg.gpu_global_merge_direct && route_cfg.use_device_query_gather) {
        ung_singleton_top1_global_merge_kernel<<<sg_grid, sg_block, 0, stream>>>(
            g_d_Q, g_d_q_norm, g_d_all_X, g_d_all_norm,
            g_d_singleton_qi, g_d_singleton_xoff, g_d_q_ids,
            (int)singleton_query_count, dim, topk,
            g_d_global_idx, g_d_global_dis, g_d_global_locks);
        dispatch_counters.global_merge_direct_query_count += singleton_query_count;
    } else {
        ung_singleton_top1_global_kernel<<<sg_grid, sg_block, 0, stream>>>(
            g_d_Q, g_d_q_norm, g_d_all_X, g_d_all_norm,
            g_d_singleton_qi, g_d_singleton_xoff, (int)singleton_query_count,
            dim, topk, g_d_idx, g_d_dis);
    }
}

inline CrossEdgeBucketDescriptors build_cross_edge_regular_bucket_descriptors(
    const std::vector<ANNS::IdxType>& target_group_ids,
    const std::vector<size_t>& target_counts,
    const std::vector<int>& target_group_nx,
    const std::vector<size_t>& target_offsets,
    const std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>>& group_id_to_range,
    int dim,
    int topk,
    const CrossEdgeRegularRouteConfig& route_cfg)
{
    return build_cross_edge_bucket_descriptors(target_group_ids,
                                               target_counts,
                                               target_group_nx,
                                               target_offsets,
                                               group_id_to_range,
                                               dim,
                                               topk,
                                               route_cfg.bucket_group_fused,
                                               route_cfg.use_naive_cuda,
                                               g_d_all_norm != nullptr,
                                               route_cfg.naive_group_fused,
                                               route_cfg.singleton_fastpath,
                                               route_cfg.large_group_fused,
                                               route_cfg.large_group_fused_mode,
                                               route_cfg.large_group_min_nx,
                                               route_cfg.large_group_max_nx,
                                               route_cfg.naive_heavy_sgemm,
                                               route_cfg.naive_heavy_nx,
                                               route_cfg.naive_heavy_nq,
                                               route_cfg.naive_heavy_work_thresh,
                                               route_cfg.small_group_fused,
                                               route_cfg.small_group_max_nx,
                                               route_cfg.medium_group_fused,
                                               route_cfg.medium_group_max_nx,
                                               route_cfg.tf32_group_prefix,
                                               route_cfg.tf32_group_2d);
}

inline void launch_cross_edge_regular_bucket_fused_groups(
    cudaStream_t stream,
    int dim,
    int topk,
    const CrossEdgeRegularRouteConfig& route_cfg,
    const CrossEdgeBucketDescriptors& bucket_descs)
{
    launch_cross_edge_bucket_fused_groups(stream,
                                          route_cfg.bucket_group_fused,
                                          dim,
                                          topk,
                                          route_cfg.large_group_warps,
                                          route_cfg.tf32_group_2d,
                                          route_cfg.small_group_warps,
                                          route_cfg.small_group_max_nx,
                                          route_cfg.medium_group_warps,
                                          route_cfg.medium_group_max_nx,
                                          route_cfg.direct_qid_all_fused,
                                          route_cfg.direct_qid_fused,
                                          bucket_descs.small_group_descs,
                                          bucket_descs.medium_group_descs,
                                          bucket_descs.tf32_tile_descs,
                                          bucket_descs.tf32_group_descs,
                                          bucket_descs.tf32_group_tile_offsets,
                                          g_d_Q,
                                          g_d_q_norm,
                                          g_d_all_X,
                                          g_d_all_norm,
                                          g_d_q_ids,
                                          g_d_idx,
                                          g_d_dis);
}

inline void finalize_cross_edge_regular_global_merge(
    cudaStream_t stream,
    bool gpu_global_merge,
    size_t global_merge_direct_query_count,
    int total_queries,
    int topk)
{
    if (!gpu_global_merge || global_merge_direct_query_count >= (size_t)total_queries) {
        return;
    }

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

inline void log_cross_edge_regular_backend_profile(
    const CrossEdgeRegularRouteConfig& route_cfg,
    const CrossEdgeRegularDispatchCounters& dispatch_counters,
    const CrossEdgeBucketDescriptors& bucket_descs,
    const CrossEdgeTopkOutputFinalizeResult& output_result,
    bool direct_qid_all_effective,
    bool gpu_global_merge,
    size_t singleton_query_count,
    size_t cpu_tiny_group_count,
    size_t cpu_tiny_query_count,
    double cpu_tiny_ms,
    bool cpu_tiny_enabled,
    int cpu_tiny_nx_max,
    int cpu_tiny_nq_max,
    long long cpu_tiny_ops_thresh,
    double count_ms,
    double offset_ms,
    double fill_q_ms,
    int total_queries)
{
    CrossEdgeRegularProfileSummary profile_summary;
    profile_summary.valid_topk_pairs = output_result.valid_topk_pairs;
    profile_summary.tail_gemv_count = dispatch_counters.tail_gemv_count;
    profile_summary.small_group_fused_groups = dispatch_counters.small_group_fused_group_count;
    profile_summary.small_group_fused_queries = dispatch_counters.small_group_fused_query_count;
    profile_summary.small_group_max_nx = route_cfg.small_group_max_nx;
    profile_summary.medium_group_fused_groups = dispatch_counters.medium_group_fused_group_count;
    profile_summary.medium_group_fused_queries = dispatch_counters.medium_group_fused_query_count;
    profile_summary.medium_group_max_nx = route_cfg.medium_group_max_nx;
    profile_summary.large_group_fused_groups = dispatch_counters.large_group_fused_group_count;
    profile_summary.large_group_fused_queries = dispatch_counters.large_group_fused_query_count;
    profile_summary.large_group_min_nx = route_cfg.large_group_min_nx;
    profile_summary.large_group_max_nx = route_cfg.large_group_max_nx;
    profile_summary.large_group_warps = route_cfg.large_group_warps;
    profile_summary.bucket_group_fused = route_cfg.bucket_group_fused;
    profile_summary.small_desc_count = bucket_descs.small_group_descs.size();
    profile_summary.medium_desc_count = bucket_descs.medium_group_descs.size();
    profile_summary.tf32_tile_desc_count = bucket_descs.tf32_tile_descs.size();
    profile_summary.tf32_group_count = bucket_descs.tf32_group_descs.size();
    profile_summary.tf32_group_tile_count =
        bucket_descs.tf32_group_tile_offsets.empty()
            ? 0zu
            : (size_t)bucket_descs.tf32_group_tile_offsets.back();
    profile_summary.direct_qid_fused = route_cfg.direct_qid_fused;
    profile_summary.direct_qid_all_effective = direct_qid_all_effective;
    profile_summary.id_only_writeback = route_cfg.id_only_writeback;
    profile_summary.id_vector_writeback = route_cfg.id_vector_writeback;
    profile_summary.flat_id_writeback = route_cfg.flat_id_writeback;
    profile_summary.gpu_global_merge = gpu_global_merge;
    profile_summary.d2h_rows = output_result.d2h_rows;
    profile_summary.gpu_global_merge_direct = route_cfg.gpu_global_merge_direct;
    profile_summary.global_merge_direct_groups = dispatch_counters.global_merge_direct_group_count;
    profile_summary.global_merge_direct_queries = dispatch_counters.global_merge_direct_query_count;
    profile_summary.gpu_global_merge_direct_max_nx = route_cfg.gpu_global_merge_direct_max_nx;
    profile_summary.heavy_sgemm_groups = dispatch_counters.heavy_sgemm_group_count;
    profile_summary.heavy_sgemm_queries = dispatch_counters.heavy_sgemm_query_count;
    profile_summary.cpu_tiny_groups = cpu_tiny_group_count;
    profile_summary.cpu_tiny_queries = cpu_tiny_query_count;
    profile_summary.cpu_tiny_ms = cpu_tiny_ms;
    profile_summary.cpu_tiny_enabled = cpu_tiny_enabled;
    profile_summary.cpu_tiny_nx_max = cpu_tiny_nx_max;
    profile_summary.cpu_tiny_nq_max = cpu_tiny_nq_max;
    profile_summary.cpu_tiny_ops_thresh = cpu_tiny_ops_thresh;
    profile_summary.count_ms = count_ms;
    profile_summary.offset_ms = offset_ms;
    profile_summary.fill_q_ms = fill_q_ms;
    profile_summary.writeback_ms = output_result.writeback_ms;
    profile_summary.total_queries = total_queries;
    profile_summary.gather_q = route_cfg.use_device_query_gather;
    profile_summary.singleton_fastpath = route_cfg.singleton_fastpath;
    profile_summary.singleton_queries = singleton_query_count;
    log_cross_edge_regular_profile_summary(profile_summary);
}

#endif // ANNS_GPU_CROSS_EDGE_REGULAR_BACKEND_CUH
