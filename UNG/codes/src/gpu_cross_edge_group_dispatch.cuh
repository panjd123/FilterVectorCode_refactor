#ifndef ANNS_GPU_CROSS_EDGE_GROUP_DISPATCH_CUH
#define ANNS_GPU_CROSS_EDGE_GROUP_DISPATCH_CUH

// Per-target-group dispatch for the regular all-batched cross-edge route.
//
// The caller owns high-level route config and profile counters. This helper
// executes each target group by trying immediate fused kernels first, then
// falling back to the separate dot + topK baseline.

struct CrossEdgeRegularWorkloadView {
    const std::vector<ANNS::IdxType>* target_group_ids = nullptr;
    const std::vector<size_t>* target_counts = nullptr;
    const std::vector<int>* target_group_nx = nullptr;
    const std::vector<size_t>* target_offsets = nullptr;
    const std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>>* group_id_to_range = nullptr;
    const std::vector<uint8_t>* bucket_group_kind = nullptr;
};

struct CrossEdgeRegularDispatchConfig {
    int dim = 0;
    int topk = 0;
    bool singleton_fastpath = true;

    bool use_sgemm_strided = false;
    bool use_naive_cuda = false;
    bool use_cublas_lt = false;
    bool naive_heavy_sgemm = true;
    int naive_heavy_nx = 256;
    int naive_heavy_nq = 256;
    long long naive_heavy_work_thresh = 1000000LL;

    bool naive_group_fused = false;
    int naive_group_threads = 128;
    bool small_group_fused = true;
    int small_group_max_nx = 8;
    int small_group_warps = 8;
    bool medium_group_fused = true;
    int medium_group_max_nx = 4096;
    int medium_group_warps = 8;
    bool medium_group_sync_debug = false;
    bool large_group_fused = false;
    int large_group_min_nx = 128;
    int large_group_max_nx = 1023;
    int large_group_warps = 2;
    int large_group_fused_mode = 2;
    bool bucket_group_fused = false;

    bool gpu_global_merge_direct = false;
    bool use_device_query_gather = true;
    int gpu_global_merge_direct_max_nx = 256;
};

struct CrossEdgeRegularDeviceBuffers {
    const float* d_Q = nullptr;
    const float* d_q_norm = nullptr;
    const uint32_t* d_q_ids = nullptr;
    const float* d_all_X = nullptr;
    const float* d_all_norm = nullptr;
    int* d_idx = nullptr;
    float* d_dis = nullptr;
    int* d_global_idx = nullptr;
    float* d_global_dis = nullptr;
    int* d_global_locks = nullptr;
    const float* h_Q = nullptr;
    const float* h_all_X = nullptr;
};

struct CrossEdgeRegularDispatchCounters {
    size_t tail_gemv_count = 0;
    size_t small_group_fused_group_count = 0;
    size_t small_group_fused_query_count = 0;
    size_t medium_group_fused_group_count = 0;
    size_t medium_group_fused_query_count = 0;
    size_t large_group_fused_group_count = 0;
    size_t large_group_fused_query_count = 0;
    size_t heavy_sgemm_group_count = 0;
    size_t heavy_sgemm_query_count = 0;
    size_t global_merge_direct_group_count = 0;
    size_t global_merge_direct_query_count = 0;
    bool naive_debug_logged = false;
};

inline void dispatch_cross_edge_regular_groups(
    cudaStream_t stream,
    const CrossEdgeRegularWorkloadView& workload,
    const CrossEdgeSeparateTopkConfig& separate_topk_cfg,
    const CrossEdgeRegularDispatchConfig& cfg,
    const CrossEdgeRegularDeviceBuffers& buffers,
    CrossEdgeRegularDispatchCounters& counters)
{
    const auto& target_group_ids = *workload.target_group_ids;
    const auto& target_counts = *workload.target_counts;
    const auto& target_group_nx = *workload.target_group_nx;
    const auto& target_offsets = *workload.target_offsets;
    const auto& group_id_to_range = *workload.group_id_to_range;
    const auto& bucket_group_kind = *workload.bucket_group_kind;
    const int dim = cfg.dim;
    const int topk = cfg.topk;

    for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
        int nq_g = (int)target_counts[gi];
        if (nq_g <= 0) continue;

        const size_t q_start = target_offsets[gi];
        const ANNS::IdxType tgt_gid = target_group_ids[gi];
        const auto& rng = group_id_to_range[tgt_gid];
        const int x_off = (int)rng.first;
        const int nx = target_group_nx[gi];
        if (nx <= 0) continue;
        if (cfg.singleton_fastpath && nx == 1) continue;

        bool group_use_sgemm = cfg.use_sgemm_strided;
        bool group_use_naive = cfg.use_naive_cuda;
        bool group_use_lt = cfg.use_cublas_lt;

        const float* dQ = buffers.d_Q + (size_t)q_start * (size_t)dim;
        const float* dX = buffers.d_all_X + (size_t)x_off * (size_t)dim;
        const float* dXnorm_group = buffers.d_all_norm ? (buffers.d_all_norm + (size_t)x_off) : nullptr;
        const bool allow_group_global_merge_direct =
            cfg.gpu_global_merge_direct && cfg.use_device_query_gather &&
            nx <= cfg.gpu_global_merge_direct_max_nx;
        const bool use_large_group_fused_for_group =
            cfg.large_group_fused && topk <= 32 && dXnorm_group != nullptr &&
            nx >= cfg.large_group_min_nx && nx <= cfg.large_group_max_nx;

        if (cfg.use_naive_cuda && cfg.naive_heavy_sgemm && !use_large_group_fused_for_group) {
            const unsigned long long work =
                (unsigned long long)nq_g * (unsigned long long)nx * (unsigned long long)dim;
            if (nx >= cfg.naive_heavy_nx &&
                nq_g >= cfg.naive_heavy_nq &&
                work >= (unsigned long long)cfg.naive_heavy_work_thresh) {
                group_use_sgemm = true;
                group_use_naive = false;
                group_use_lt = false;
                counters.heavy_sgemm_group_count += 1;
                counters.heavy_sgemm_query_count += (size_t)nq_g;
            }
        }

        int* dBestI = buffers.d_idx + (size_t)q_start * (size_t)topk;
        float* dBestD = buffers.d_dis + (size_t)q_start * (size_t)topk;
        const uint8_t group_bucket_kind = gi < bucket_group_kind.size() ? bucket_group_kind[gi] : 0;

        if (try_launch_cross_edge_per_group_fused(
                stream,
                gi,
                tgt_gid,
                nq_g,
                nx,
                x_off,
                q_start,
                dim,
                topk,
                dQ,
                dX,
                dXnorm_group,
                buffers.d_q_norm,
                buffers.d_q_ids,
                dBestI,
                dBestD,
                buffers.d_global_idx,
                buffers.d_global_dis,
                buffers.d_global_locks,
                group_use_naive,
                cfg.naive_group_fused,
                cfg.naive_group_threads,
                cfg.small_group_fused,
                cfg.small_group_max_nx,
                cfg.small_group_warps,
                cfg.medium_group_fused,
                cfg.medium_group_max_nx,
                cfg.medium_group_warps,
                cfg.medium_group_sync_debug,
                cfg.large_group_fused,
                cfg.large_group_min_nx,
                cfg.large_group_max_nx,
                cfg.large_group_warps,
                cfg.large_group_fused_mode,
                cfg.bucket_group_fused,
                group_bucket_kind,
                allow_group_global_merge_direct,
                counters.small_group_fused_group_count,
                counters.small_group_fused_query_count,
                counters.medium_group_fused_group_count,
                counters.medium_group_fused_query_count,
                counters.large_group_fused_group_count,
                counters.large_group_fused_query_count,
                counters.global_merge_direct_group_count,
                counters.global_merge_direct_query_count)) {
            continue;
        }

        run_cross_edge_separate_topk_group(
            stream,
            separate_topk_cfg,
            dim,
            topk,
            nq_g,
            nx,
            x_off,
            q_start,
            gi,
            dQ,
            dX,
            dXnorm_group,
            buffers.d_q_norm + q_start,
            dBestI,
            dBestD,
            group_use_sgemm,
            group_use_naive,
            group_use_lt,
            buffers.h_Q,
            buffers.h_all_X,
            counters.tail_gemv_count,
            counters.naive_debug_logged);
    }
}

#endif // ANNS_GPU_CROSS_EDGE_GROUP_DISPATCH_CUH
