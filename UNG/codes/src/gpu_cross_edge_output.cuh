// Output finalization helpers for the non-double-buffer batched cross-edge path.
//
// This file is included from gpu_gemm_topk.cu after the shared CUDA buffers and
// logging helpers are available. It owns D2H and host writeback only; kernel
// route selection and graph semantics stay in the caller.

struct CrossEdgeTopkOutputFinalizeResult {
    int d2h_rows = 0;
    double d2h_ms = 0.0;
    double writeback_ms = 0.0;
    size_t valid_topk_pairs = 0;
};

inline CrossEdgeTopkOutputFinalizeResult finalize_cross_edge_topk_output(
    cudaStream_t stream,
    bool gpu_global_merge,
    const CrossEdgeTopkOutputWriter& writer,
    bool count_valid_pairs,
    int total_queries,
    int total_points,
    int topk,
    const int* d_idx,
    const float* d_dis,
    const int* d_global_idx,
    const float* d_global_dis,
    int* h_idx,
    float* h_dis,
    const std::vector<ANNS::IdxType>& query_global_ids,
    const std::vector<int>& query_target_index,
    const std::vector<ANNS::IdxType>& target_group_ids,
    const std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>>& group_id_to_range) {
    CrossEdgeTopkOutputFinalizeResult result;
    result.d2h_rows = gpu_global_merge ? total_points : total_queries;

    const int* d2h_idx_src = gpu_global_merge ? d_global_idx : d_idx;
    const float* d2h_dis_src = gpu_global_merge ? d_global_dis : d_dis;

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, stream);

    cudaError_t copy_idx_err = cudaMemcpyAsync(
        h_idx,
        d2h_idx_src,
        (size_t)result.d2h_rows * (size_t)topk * sizeof(int),
        cudaMemcpyDeviceToHost,
        stream);
    if (copy_idx_err != cudaSuccess) {
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        prof_logf("[ERROR] cross_edges D2H idx enqueue failed: %s", cudaGetErrorString(copy_idx_err));
        throw std::runtime_error("cross_edges D2H idx enqueue failed.");
    }

    if (!writer.id_only_writeback) {
        cudaError_t copy_dis_err = cudaMemcpyAsync(
            h_dis,
            d2h_dis_src,
            (size_t)result.d2h_rows * (size_t)topk * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream);
        if (copy_dis_err != cudaSuccess) {
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            prof_logf("[ERROR] cross_edges D2H dist enqueue failed: %s", cudaGetErrorString(copy_dis_err));
            throw std::runtime_error("cross_edges D2H dist enqueue failed.");
        }
    }

    cudaEventRecord(e1, stream);
    cudaError_t d2h_sync_err = cudaEventSynchronize(e1);
    if (d2h_sync_err != cudaSuccess) {
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        prof_logf("[ERROR] cross_edges D2H stream failed: %s", cudaGetErrorString(d2h_sync_err));
        throw std::runtime_error("cross_edges D2H stream failed.");
    }

    float d2h_ms = 0.0f;
    cudaEventElapsedTime(&d2h_ms, e0, e1);
    result.d2h_ms = (double)d2h_ms;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    const auto t_writeback_start = std::chrono::high_resolution_clock::now();
    if (writer.writes_flat_ids()) {
        if (writer.flat_ids == nullptr) {
            throw std::runtime_error("cross-edge flat-id output writer requires flat_ids storage.");
        }
        writer.flat_ids->assign(
            (size_t)total_points * (size_t)topk,
            std::numeric_limits<ANNS::IdxType>::max());
#pragma omp parallel for schedule(static, 1024)
        for (int qid_i = 0; qid_i < total_points; ++qid_i) {
            const size_t base = (size_t)qid_i * (size_t)topk;
            for (int r = 0; r < topk; ++r) {
                const int gid = h_idx[base + (size_t)r];
                if (gid >= 0) {
                    (*writer.flat_ids)[base + (size_t)r] = (ANNS::IdxType)gid;
                }
            }
        }
    } else if (writer.writes_id_vectors()) {
        if (writer.id_vectors == nullptr) {
            throw std::runtime_error("cross-edge id-vector output writer requires id_vectors storage.");
        }
#pragma omp parallel for schedule(static, 1024)
        for (int qid_i = 0; qid_i < total_points; ++qid_i) {
            auto& ids = (*writer.id_vectors)[(ANNS::IdxType)qid_i];
            for (int r = 0; r < topk; ++r) {
                const int gid = h_idx[(size_t)qid_i * (size_t)topk + (size_t)r];
                if (gid >= 0) ids.emplace_back((ANNS::IdxType)gid);
            }
        }
    } else if (gpu_global_merge) {
        if (writer.search_queues == nullptr) {
            throw std::runtime_error("cross-edge SearchQueue output writer requires search queue storage.");
        }
#pragma omp parallel for schedule(static, 1024)
        for (int qid_i = 0; qid_i < total_points; ++qid_i) {
            const ANNS::IdxType qid = (ANNS::IdxType)qid_i;
            auto& queue = (*writer.search_queues)[qid];
            if (queue.capacity() < topk) {
                queue.reserve(topk);
            }
            for (int r = 0; r < topk; ++r) {
                const int gid = h_idx[(size_t)qid_i * (size_t)topk + (size_t)r];
                if (gid < 0) continue;
                const float dist = writer.id_only_writeback ? (float)r : h_dis[(size_t)qid_i * (size_t)topk + (size_t)r];
                queue.insert((ANNS::IdxType)gid, dist);
            }
        }
    } else {
        if (writer.search_queues == nullptr) {
            throw std::runtime_error("cross-edge SearchQueue output writer requires search queue storage.");
        }
#pragma omp parallel for schedule(static, 1024)
        for (int qi = 0; qi < total_queries; ++qi) {
            const ANNS::IdxType qid = query_global_ids[(size_t)qi];
            const int gi = query_target_index[(size_t)qi];
            const ANNS::IdxType tgt_gid = target_group_ids[(size_t)gi];
            const auto& rng = group_id_to_range[(size_t)tgt_gid];
            const ANNS::IdxType tgt_offset = rng.first;
            const int valid_k = std::min<int>(topk, (int)(rng.second - rng.first));
            auto& queue = (*writer.search_queues)[qid];
            if (queue.capacity() < topk) {
                queue.reserve(topk);
            }

            for (int r = 0; r < valid_k; ++r) {
                const int j = h_idx[(size_t)qi * (size_t)topk + (size_t)r];
                if (j < 0) continue;
                const float dist = writer.id_only_writeback ? (float)r : h_dis[(size_t)qi * (size_t)topk + (size_t)r];
                queue.insert(tgt_offset + (ANNS::IdxType)j, dist);
            }
        }
    }
    const auto t_writeback_end = std::chrono::high_resolution_clock::now();
    result.writeback_ms = std::chrono::duration<double, std::milli>(
        t_writeback_end - t_writeback_start).count();

    if (count_valid_pairs) {
        size_t valid_pairs = 0;
        for (int qi = 0; qi < result.d2h_rows; ++qi) {
            for (int r = 0; r < topk; ++r) {
                if (h_idx[(size_t)qi * (size_t)topk + (size_t)r] >= 0) ++valid_pairs;
            }
        }
        result.valid_topk_pairs = valid_pairs;
    }

    return result;
}
