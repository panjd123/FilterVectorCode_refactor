#ifndef ANNS_GPU_CROSS_EDGE_PLANNING_CUH
#define ANNS_GPU_CROSS_EDGE_PLANNING_CUH

// Common planning helpers for batched cross-edge GPU routes.
//
// These helpers are intentionally route-neutral: they compute LNG in-neighbor
// query counts, target group sizes, CPU-tiny fallback selection, bench filters,
// and flat-batch capacity. CUDA execution paths consume this plan but should
// not duplicate the workload semantics.

struct CrossEdgeTargetWorkload {
    std::vector<size_t> raw_query_counts;
    std::vector<size_t> gpu_query_counts;
    std::vector<int> group_nx;
    std::vector<int> cpu_tiny_group_indices;
    size_t bench_skipped_groups = 0;
    size_t cpu_tiny_group_count = 0;
    size_t cpu_tiny_query_count = 0;
    size_t total_gpu_queries = 0;
    double count_ms = 0.0;
    int bench_min_nx = 0;
    int bench_max_nx = 1048576;
    long long bench_min_work = 0;
    bool cpu_tiny_groups = false;
    int cpu_tiny_nx_max = 8;
    int cpu_tiny_nq_max = 64;
    long long cpu_tiny_ops_thresh = 65536;

    explicit CrossEdgeTargetWorkload(size_t group_count)
        : raw_query_counts(group_count, 0),
          gpu_query_counts(group_count, 0),
          group_nx(group_count, 0)
    {
        cpu_tiny_group_indices.reserve(group_count / 8 + 1);
    }
};

struct CrossEdgeBatchCapacity {
    int flat_q_cap_mb = 4096;
    int flat_out_cap_mb = 4096;
    size_t max_flat_queries = 1;
};

CrossEdgeBatchCapacity make_cross_edge_batch_capacity(
    int dim,
    int topk,
    const ANNS::CrossEdgeGpuRuntimeConfig& gpu_route)
{
    CrossEdgeBatchCapacity cap;
    cap.flat_q_cap_mb = gpu_route.flat_q_cap_mb;
    cap.flat_out_cap_mb = gpu_route.flat_out_cap_mb;

    const size_t flat_q_cap_bytes = (size_t)cap.flat_q_cap_mb * 1024ull * 1024ull;
    const size_t flat_out_cap_bytes = (size_t)cap.flat_out_cap_mb * 1024ull * 1024ull;
    const size_t q_bytes_per_query = (size_t)std::max(dim, 1) * sizeof(float);
    const size_t out_bytes_per_query = (size_t)std::max(topk, 1) * (sizeof(int) + sizeof(float));

    cap.max_flat_queries = flat_q_cap_bytes / q_bytes_per_query;
    cap.max_flat_queries = std::min(cap.max_flat_queries, flat_out_cap_bytes / out_bytes_per_query);
    if (cap.max_flat_queries < 1) {
        cap.max_flat_queries = 1;
    }
    return cap;
}

CrossEdgeTargetWorkload plan_cross_edge_target_workload(
    const std::vector<ANNS::IdxType>& target_group_ids,
    int dim,
    const std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>>& group_id_to_range,
    const ANNS::LabelNavGraph& label_nav_graph,
    const ANNS::CrossEdgeGpuRuntimeConfig& gpu_route,
    const std::vector<uint8_t>* skip_source_groups = nullptr)
{
    CrossEdgeTargetWorkload workload(target_group_ids.size());
    workload.bench_min_nx = gpu_route.benchmark_filter.min_nx;
    workload.bench_max_nx = gpu_route.benchmark_filter.max_nx;
    workload.bench_min_work = gpu_route.benchmark_filter.min_work;
    workload.cpu_tiny_groups = gpu_route.cpu_tiny_groups;
    workload.cpu_tiny_nx_max = gpu_route.cpu_tiny_nx_max;
    workload.cpu_tiny_nq_max = gpu_route.cpu_tiny_nq_max;
    workload.cpu_tiny_ops_thresh = (long long)gpu_route.cpu_tiny_ops;

    const auto t_count_start = std::chrono::high_resolution_clock::now();
    for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
        const ANNS::IdxType tgt = target_group_ids[gi];
        const auto& tgt_rng = group_id_to_range[tgt];
        workload.group_nx[gi] = (int)(tgt_rng.second - tgt_rng.first);
        size_t cnt = 0;

        // Cross edges use all vectors from LNG in-neighbor groups as queries.
        for (auto in_gid : label_nav_graph.in_neighbors[tgt]) {
            if (skip_source_groups &&
                (size_t)in_gid < skip_source_groups->size() &&
                (*skip_source_groups)[(size_t)in_gid]) {
                continue;
            }
            const auto& rng = group_id_to_range[in_gid];
            cnt += (size_t)(rng.second - rng.first);
        }

        workload.raw_query_counts[gi] = cnt;
    }
    const auto t_count_end = std::chrono::high_resolution_clock::now();
    workload.count_ms = std::chrono::duration<double, std::milli>(t_count_end - t_count_start).count();

    for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
        const size_t cnt = workload.raw_query_counts[gi];
        const int nx = workload.group_nx[gi];
        const unsigned long long est_work =
            (unsigned long long)cnt * (unsigned long long)std::max(nx, 0) * (unsigned long long)dim;
        const bool skip_for_bench =
            nx < workload.bench_min_nx ||
            nx > workload.bench_max_nx ||
            est_work < (unsigned long long)workload.bench_min_work;
        if (skip_for_bench) {
            workload.gpu_query_counts[gi] = 0;
            workload.bench_skipped_groups += 1;
            continue;
        }

        const bool route_cpu_tiny =
            workload.cpu_tiny_groups &&
            nx > 1 &&
            nx <= workload.cpu_tiny_nx_max &&
            cnt > 0 &&
            (int)cnt <= workload.cpu_tiny_nq_max &&
            est_work <= (unsigned long long)workload.cpu_tiny_ops_thresh;

        if (route_cpu_tiny) {
            workload.cpu_tiny_group_indices.push_back((int)gi);
            workload.cpu_tiny_group_count += 1;
            workload.cpu_tiny_query_count += cnt;
            workload.gpu_query_counts[gi] = 0;
        } else {
            workload.gpu_query_counts[gi] = cnt;
            workload.total_gpu_queries += cnt;
        }
    }

    return workload;
}

#endif // ANNS_GPU_CROSS_EDGE_PLANNING_CUH
