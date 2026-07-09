#ifndef ANNS_GPU_CROSS_EDGE_QUERY_PACK_CUH
#define ANNS_GPU_CROSS_EDGE_QUERY_PACK_CUH

// Query packing/upload helper for the regular all-batched cross-edge route.
//
// This helper flattens LNG in-neighbor vectors into query rows, builds the
// host-side writeback maps, and uploads either Q vectors or qids for device
// gather. It intentionally preserves the original copy/gather semantics.

struct CrossEdgeQueryUploadResult {
    std::vector<ANNS::IdxType> query_global_ids;
    std::vector<int> query_target_index;
    std::vector<uint32_t> query_target_offsets;
    double fill_ms = 0.0;
    double h2d_ms = 0.0;
};

CrossEdgeQueryUploadResult pack_and_upload_cross_edge_queries(
    cudaStream_t stream,
    const std::vector<ANNS::IdxType>& target_group_ids,
    const std::vector<size_t>& target_counts,
    const std::vector<size_t>& target_offsets,
    int total_queries,
    int dim,
    bool use_device_query_gather,
    bool gpu_global_merge,
    bool direct_qid_all_effective,
    int gather_blocks,
    int gather_threads,
    ANNS::IStorage* base_storage,
    const std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>>& group_id_to_range,
    const ANNS::LabelNavGraph& label_nav_graph,
    const std::vector<uint8_t>* skip_source_groups = nullptr)
{
    CrossEdgeQueryUploadResult result;
    result.query_global_ids.resize((size_t)total_queries);
    result.query_target_index.resize((size_t)total_queries);
    if (gpu_global_merge) {
        result.query_target_offsets.resize((size_t)total_queries);
    }

    const auto t_fill_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic, 8)
    for (int gi = 0; gi < (int)target_group_ids.size(); ++gi) {
        if (target_counts[(size_t)gi] == 0) continue;
        size_t write_pos = target_offsets[(size_t)gi];
        ANNS::IdxType tgt = target_group_ids[(size_t)gi];
        const uint32_t tgt_offset = (uint32_t)group_id_to_range[tgt].first;

        for (auto in_gid : label_nav_graph.in_neighbors[tgt]) {
            if (skip_source_groups &&
                (size_t)in_gid < skip_source_groups->size() &&
                (*skip_source_groups)[(size_t)in_gid]) {
                continue;
            }
            const auto& rng = group_id_to_range[in_gid];
            for (ANNS::IdxType vid = rng.first; vid < rng.second; ++vid) {
                if (!use_device_query_gather) {
                    const float* v = reinterpret_cast<const float*>(base_storage->get_vector(vid));
                    std::memcpy(g_h_Q + (size_t)write_pos * (size_t)dim, v, (size_t)dim * sizeof(float));
                }

                result.query_global_ids[(size_t)write_pos] = vid;
                result.query_target_index[(size_t)write_pos] = gi;
                if (gpu_global_merge) result.query_target_offsets[(size_t)write_pos] = tgt_offset;
                ++write_pos;
            }
        }
    }
    const auto t_fill_end = std::chrono::high_resolution_clock::now();
    result.fill_ms = std::chrono::duration<double, std::milli>(t_fill_end - t_fill_start).count();

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, stream);
    if (use_device_query_gather && g_d_all_X != nullptr) {
        cudaMemcpyAsync(g_d_q_ids,
                        result.query_global_ids.data(),
                        (size_t)total_queries * sizeof(uint32_t),
                        cudaMemcpyHostToDevice,
                        stream);
        if (gpu_global_merge) {
            cudaMemcpyAsync(g_d_q_target_offsets,
                            result.query_target_offsets.data(),
                            (size_t)total_queries * sizeof(uint32_t),
                            cudaMemcpyHostToDevice,
                            stream);
        }
        if (!direct_qid_all_effective) {
            ung_gather_q_by_id_global_kernel<<<gather_blocks, gather_threads, 0, stream>>>(
                g_d_all_X, g_d_q_ids, total_queries, dim, g_d_Q);
        }
    } else {
        if (!direct_qid_all_effective) {
            cudaMemcpyAsync(g_d_Q,
                            g_h_Q,
                            (size_t)total_queries * (size_t)dim * sizeof(float),
                            cudaMemcpyHostToDevice,
                            stream);
        }
        if (gpu_global_merge) {
            cudaMemcpyAsync(g_d_q_ids,
                            result.query_global_ids.data(),
                            (size_t)total_queries * sizeof(uint32_t),
                            cudaMemcpyHostToDevice,
                            stream);
            cudaMemcpyAsync(g_d_q_target_offsets,
                            result.query_target_offsets.data(),
                            (size_t)total_queries * sizeof(uint32_t),
                            cudaMemcpyHostToDevice,
                            stream);
        }
    }
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);

    float h2d_ms = 0.f;
    cudaEventElapsedTime(&h2d_ms, e0, e1);
    result.h2d_ms = (double)h2d_ms;

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return result;
}

#endif // ANNS_GPU_CROSS_EDGE_QUERY_PACK_CUH
