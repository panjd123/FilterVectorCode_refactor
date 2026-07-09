#ifndef ANNS_GPU_CROSS_EDGE_REGULAR_ROUTE_CUH
#define ANNS_GPU_CROSS_EDGE_REGULAR_ROUTE_CUH

// Route/config assembly for the regular all-batched cross-edge path.
//
// This helper is intentionally host-only. It turns the public
// CrossEdgeGpuRuntimeConfig into the smaller execution configs consumed by
// packing, dispatch, and output.

struct CrossEdgeRegularRouteConfig {
    bool use_device_query_gather = true;
    bool singleton_fastpath = true;
    bool direct_qid_fused = false;
    bool direct_qid_all_fused = false;
    bool id_only_writeback_requested = false;
    bool gpu_global_merge_requested = false;
    bool gpu_global_merge = false;
    bool id_vector_writeback = false;
    bool flat_id_writeback = false;
    bool id_only_writeback = false;
    bool gpu_global_merge_direct = false;
    int gpu_global_merge_direct_max_nx = 256;
    int gather_threads = 256;
    int gather_blocks = 4096;

    bool force_custom_kernel = true;
    int gemm_impl_req = 2;
    int gemm_impl = 2;
    bool use_sgemm_strided = false;
    bool use_naive_cuda = true;
    bool use_cublas_lt = false;

    int naive_warps_per_block = 4;
    bool naive_shared_x = false;
    bool naive_vec4 = true;
    int naive_x_cols_per_block = 1;
    int naive_shared_kb = 48;
    bool naive_heavy_sgemm = true;
    int naive_heavy_nx = 256;
    int naive_heavy_nq = 256;
    long long naive_heavy_work_thresh = 1000000LL;
    int topk_block_threads = 64;
    bool naive_group_fused = false;
    int naive_group_threads = 128;
    int singleton_threads = 256;

    bool small_group_fused = true;
    int small_group_max_nx = 8;
    int small_group_warps = 8;
    bool medium_group_fused = true;
    int medium_group_max_nx = 4096;
    int medium_group_warps = 8;
    bool medium_group_sync_debug = false;
    bool bucket_group_fused = false;
    bool large_group_fused = false;
    int large_group_min_nx = 128;
    int large_group_max_nx = 1023;
    int large_group_warps = 2;
    int large_group_fused_mode = 2;
    bool tf32_group_prefix = false;
    bool tf32_group_2d = false;
    bool naive_debug_dot = false;

    const char* gemm_mode_name() const
    {
        return use_sgemm_strided ? "sgemm_strided_batched" :
               (use_naive_cuda ? "naive_cuda_core" : "cublasLt");
    }
};

inline CrossEdgeRegularRouteConfig make_cross_edge_regular_route_config(
    const ANNS::CrossEdgeGpuRuntimeConfig& gpu_route,
    int dim,
    int topk,
    bool wants_id_vector_output,
    bool wants_flat_id_output)
{
    CrossEdgeRegularRouteConfig cfg;

    cfg.use_device_query_gather = gpu_route.query_upload_device_gather;
    cfg.singleton_fastpath = gpu_route.singleton_fastpath;
    cfg.direct_qid_fused = gpu_route.direct_qid_fused;
    cfg.direct_qid_all_fused = gpu_route.direct_qid_all_fused;
    cfg.id_only_writeback_requested = gpu_route.id_only_writeback_requested;
    cfg.gpu_global_merge_requested = gpu_route.global_merge;
    cfg.gpu_global_merge = cfg.gpu_global_merge_requested ||
                           wants_id_vector_output ||
                           wants_flat_id_output;
    cfg.id_vector_writeback = wants_id_vector_output && cfg.gpu_global_merge;
    cfg.flat_id_writeback = wants_flat_id_output && cfg.gpu_global_merge;
    cfg.id_only_writeback = (cfg.id_only_writeback_requested || cfg.flat_id_writeback) &&
                            cfg.gpu_global_merge;
    cfg.gpu_global_merge_direct =
        cfg.gpu_global_merge &&
        gpu_route.global_merge_direct;
    cfg.gpu_global_merge_direct_max_nx = gpu_route.global_merge_direct_max_nx;
    cfg.gather_threads = gpu_route.gather_threads;
    cfg.gather_blocks = gpu_route.gather_blocks;

    const int gemm_impl_default = (dim >= 256 && topk <= 16) ? 2 : 1;
    cfg.force_custom_kernel = gpu_route.force_custom_kernel;
    cfg.gemm_impl_req =
        gpu_route.gemm_impl_request >= 0 ? gpu_route.gemm_impl_request : gemm_impl_default;
    cfg.gemm_impl = cfg.force_custom_kernel ? 2 : cfg.gemm_impl_req;
    cfg.use_sgemm_strided = (cfg.gemm_impl == 1);
    cfg.use_naive_cuda = (cfg.gemm_impl == 2);
    cfg.use_cublas_lt = (!cfg.use_sgemm_strided && !cfg.use_naive_cuda);

    cfg.naive_warps_per_block = gpu_route.naive_warps_per_block;
    cfg.naive_shared_x = gpu_route.naive_shared_x;
    cfg.naive_vec4 = gpu_route.naive_vec4;
    cfg.naive_x_cols_per_block = gpu_route.naive_x_cols_per_block;
    cfg.naive_shared_kb = gpu_route.naive_shared_kb;
    cfg.naive_heavy_sgemm = gpu_route.naive_heavy_sgemm;
    cfg.naive_heavy_nx = gpu_route.naive_heavy_nx;
    cfg.naive_heavy_nq = gpu_route.naive_heavy_nq;
    cfg.naive_heavy_work_thresh = (long long)gpu_route.naive_heavy_work_m * 1000000LL;
    const int topk_block_threads_default = (topk <= 8 ? 32 : (topk <= 16 ? 64 : 128));
    cfg.topk_block_threads =
        gpu_route.topk_block_threads > 0 ? gpu_route.topk_block_threads : topk_block_threads_default;
    cfg.naive_group_fused = gpu_route.naive_group_fused;
    cfg.naive_group_threads = gpu_route.naive_group_threads;
    cfg.singleton_threads = gpu_route.singleton_threads;

    cfg.small_group_fused = gpu_route.small_group_fused;
    cfg.small_group_max_nx = gpu_route.small_group_max_nx;
    cfg.small_group_warps = gpu_route.small_group_warps;
    cfg.medium_group_fused = gpu_route.medium_group_fused;
    cfg.medium_group_max_nx = gpu_route.medium_group_max_nx;
    cfg.medium_group_warps = gpu_route.medium_group_warps;
    cfg.medium_group_sync_debug = gpu_route.debug.medium_group_sync_debug;
    cfg.bucket_group_fused = gpu_route.bucket_group_fused;
    cfg.large_group_fused = gpu_route.large_group_fused;
    cfg.large_group_min_nx = gpu_route.large_group_min_nx;
    cfg.large_group_max_nx = gpu_route.large_group_max_nx;
    cfg.large_group_warps = gpu_route.large_group_warps;
    cfg.tf32_group_prefix = gpu_route.tf32_group_prefix;
    cfg.tf32_group_2d = gpu_route.tf32_group_2d;
    cfg.large_group_fused_mode = gpu_route.large_group_fused_mode;
    cfg.naive_debug_dot = gpu_route.debug.naive_debug_dot;
    return cfg;
}

inline bool compute_cross_edge_direct_qid_all_effective(
    const CrossEdgeRegularRouteConfig& cfg,
    const std::vector<ANNS::IdxType>& target_group_ids,
    const std::vector<size_t>& target_counts,
    const std::vector<int>& target_group_nx,
    size_t singleton_query_count,
    int topk)
{
    bool effective = cfg.direct_qid_all_fused;
    if (effective && singleton_query_count > 0) {
        effective = false;
        prof_logf("[PROF] cross_edges.direct_qid_all_fused disabled_for_singleton queries=%zu",
                  singleton_query_count);
    }
    if (!effective) {
        return false;
    }

    size_t unsupported_groups = 0;
    size_t unsupported_queries = 0;
    int first_unsupported_nx = 0;
    size_t first_unsupported_nq = 0;
    for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
        const size_t nq = target_counts[gi];
        if (nq == 0) continue;
        const int nx = target_group_nx[gi];
        if (nx <= 0) continue;
        if (cfg.singleton_fastpath && nx == 1) continue;

        const bool covered_by_bucket =
            cfg.bucket_group_fused &&
            ((cfg.small_group_fused && nx <= cfg.small_group_max_nx) ||
             (cfg.medium_group_fused && nx > cfg.small_group_max_nx && nx <= cfg.medium_group_max_nx) ||
             (cfg.large_group_fused && cfg.large_group_fused_mode == 2 && topk <= 16 &&
              nx >= cfg.large_group_min_nx && nx <= cfg.large_group_max_nx));
        const bool covered_by_direct_large =
            cfg.large_group_fused && topk <= 32 &&
            nx >= cfg.large_group_min_nx && nx <= cfg.large_group_max_nx;

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
        prof_logf("[PROF] cross_edges.direct_qid_all_fused disabled_for_fallback groups=%zu queries=%zu first_nq=%zu first_nx=%d",
                  unsupported_groups, unsupported_queries, first_unsupported_nq, first_unsupported_nx);
        return false;
    }
    return true;
}

inline CrossEdgeSeparateTopkConfig make_cross_edge_separate_topk_config(
    const CrossEdgeRegularRouteConfig& route_cfg)
{
    CrossEdgeSeparateTopkConfig cfg;
    cfg.tile_nx_default = 1024;
    cfg.dot_tile_cap_bytes = (size_t)256 * 1024 * 1024;
    cfg.dot_group_cap_bytes = (size_t)768 * 1024 * 1024;
    cfg.gemm_group_tiles_max = 16;
    cfg.lt_workspace_bytes = (size_t)64 * 1024 * 1024;
    cfg.topk_block_threads = route_cfg.topk_block_threads;
    cfg.naive_warps_per_block = route_cfg.naive_warps_per_block;
    cfg.naive_shared_x = route_cfg.naive_shared_x;
    cfg.naive_vec4 = route_cfg.naive_vec4;
    cfg.naive_x_cols_per_block = route_cfg.naive_x_cols_per_block;
    cfg.naive_shared_kb = route_cfg.naive_shared_kb;
    cfg.naive_debug_dot = route_cfg.naive_debug_dot;
    return cfg;
}

inline CrossEdgeRegularDispatchConfig make_cross_edge_regular_dispatch_config(
    const CrossEdgeRegularRouteConfig& route_cfg,
    int dim,
    int topk)
{
    CrossEdgeRegularDispatchConfig cfg;
    cfg.dim = dim;
    cfg.topk = topk;
    cfg.singleton_fastpath = route_cfg.singleton_fastpath;
    cfg.use_sgemm_strided = route_cfg.use_sgemm_strided;
    cfg.use_naive_cuda = route_cfg.use_naive_cuda;
    cfg.use_cublas_lt = route_cfg.use_cublas_lt;
    cfg.naive_heavy_sgemm = route_cfg.naive_heavy_sgemm;
    cfg.naive_heavy_nx = route_cfg.naive_heavy_nx;
    cfg.naive_heavy_nq = route_cfg.naive_heavy_nq;
    cfg.naive_heavy_work_thresh = route_cfg.naive_heavy_work_thresh;
    cfg.naive_group_fused = route_cfg.naive_group_fused;
    cfg.naive_group_threads = route_cfg.naive_group_threads;
    cfg.small_group_fused = route_cfg.small_group_fused;
    cfg.small_group_max_nx = route_cfg.small_group_max_nx;
    cfg.small_group_warps = route_cfg.small_group_warps;
    cfg.medium_group_fused = route_cfg.medium_group_fused;
    cfg.medium_group_max_nx = route_cfg.medium_group_max_nx;
    cfg.medium_group_warps = route_cfg.medium_group_warps;
    cfg.medium_group_sync_debug = route_cfg.medium_group_sync_debug;
    cfg.large_group_fused = route_cfg.large_group_fused;
    cfg.large_group_min_nx = route_cfg.large_group_min_nx;
    cfg.large_group_max_nx = route_cfg.large_group_max_nx;
    cfg.large_group_warps = route_cfg.large_group_warps;
    cfg.large_group_fused_mode = route_cfg.large_group_fused_mode;
    cfg.bucket_group_fused = route_cfg.bucket_group_fused;
    cfg.gpu_global_merge_direct = route_cfg.gpu_global_merge_direct;
    cfg.use_device_query_gather = route_cfg.use_device_query_gather;
    cfg.gpu_global_merge_direct_max_nx = route_cfg.gpu_global_merge_direct_max_nx;
    return cfg;
}

inline void log_cross_edge_regular_route_config(const CrossEdgeRegularRouteConfig& cfg)
{
    prof_logf("[PROF] cross_edges.gemm_impl mode=%s(%d) force_custom=%d req_impl=%d",
              cfg.gemm_mode_name(), cfg.gemm_impl, cfg.force_custom_kernel ? 1 : 0, cfg.gemm_impl_req);
    prof_logf("[PROF] cross_edges.topk_kernel_threads value=%d", cfg.topk_block_threads);
    prof_logf("[PROF] cross_edges.large_group_fused_cfg enabled=%d mode=%d min_nx=%d max_nx=%d warps=%d",
              cfg.large_group_fused ? 1 : 0, cfg.large_group_fused_mode,
              cfg.large_group_min_nx, cfg.large_group_max_nx, cfg.large_group_warps);
    prof_logf("[PROF] cross_edges.tf32_group_prefix enabled=%d group_2d=%d",
              cfg.tf32_group_prefix ? 1 : 0, cfg.tf32_group_2d ? 1 : 0);
    prof_logf("[PROF] cross_edges.medium_group_cfg max_nx=%d warps=%d regtopk=1",
              cfg.medium_group_max_nx, cfg.medium_group_warps);
    if (cfg.use_naive_cuda) {
        prof_logf("[PROF] cross_edges.naive_cfg warps_per_block=%d shared_x=%d vec4=%d x_cols=%d shared_kb=%d",
                  cfg.naive_warps_per_block, cfg.naive_shared_x ? 1 : 0, cfg.naive_vec4 ? 1 : 0,
                  cfg.naive_x_cols_per_block, cfg.naive_shared_kb);
        prof_logf("[PROF] cross_edges.naive_heavy_sgemm enabled=%d nx_th=%d nq_th=%d work_th=%lld",
                  cfg.naive_heavy_sgemm ? 1 : 0, cfg.naive_heavy_nx, cfg.naive_heavy_nq,
                  cfg.naive_heavy_work_thresh);
    }
}

#endif // ANNS_GPU_CROSS_EDGE_REGULAR_ROUTE_CUH
