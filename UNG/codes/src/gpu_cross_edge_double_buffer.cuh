#ifndef ANNS_GPU_CROSS_EDGE_DOUBLE_BUFFER_CUH
#define ANNS_GPU_CROSS_EDGE_DOUBLE_BUFFER_CUH

// Cross-edge double-buffer route policy, per-slot resource management,
// chunk packing, launch enqueue, and final writeback helpers.
//
// This file is intentionally included from gpu_gemm_topk.cu after the CUDA
// kernels and global buffers it calls have been declared in the anonymous
// namespace. Keep it free of namespace blocks and standard-library includes.

struct CrossEdgeDoubleBufferRoute {
    bool request = true;
    bool db_nosplit = false;
    bool query_upload_device_gather = true;
    bool split_unsupported = true;
    bool global_merge_requested = false;
    int chunk_queries = 262144;
    int medium_max_nx = 128;
    int large_max_nx = 1023;
    int medium_warps = 8;
    int large_warps = 2;
};

CrossEdgeDoubleBufferRoute make_cross_edge_double_buffer_route(
    const ANNS::CrossEdgeGpuRuntimeConfig& gpu_route)
{
    CrossEdgeDoubleBufferRoute route;
    route.request = gpu_route.double_buffer;
    route.db_nosplit = gpu_route.db_nosplit;
    route.query_upload_device_gather = gpu_route.query_upload_device_gather;
    route.split_unsupported = gpu_route.db_split_unsupported;
    route.global_merge_requested = gpu_route.db_global_merge;
    route.chunk_queries = gpu_route.db_chunk_queries;
    route.medium_max_nx = gpu_route.db_medium_max_nx;
    route.large_max_nx = gpu_route.db_large_max_nx;
    route.medium_warps = gpu_route.db_medium_group_warps;
    route.large_warps = gpu_route.db_large_group_warps;
    return route;
}

struct CrossEdgeDoubleBufferEligibility {
    bool can_enter = false;
    bool all_groups_supported = false;
    size_t unsupported_groups = 0;
};

CrossEdgeDoubleBufferEligibility evaluate_cross_edge_double_buffer_eligibility(
    const CrossEdgeDoubleBufferRoute& route,
    const std::vector<size_t>& target_counts,
    const std::vector<int>& target_group_nx,
    bool all_vectors_ready,
    bool has_cpu_tiny_groups,
    int topk,
    int dim)
{
    CrossEdgeDoubleBufferEligibility eligibility;
    eligibility.can_enter =
        route.request &&
        route.query_upload_device_gather &&
        all_vectors_ready &&
        !has_cpu_tiny_groups &&
        topk <= 32 &&
        dim <= 4096;

    if (!eligibility.can_enter) {
        return eligibility;
    }

    eligibility.all_groups_supported = true;
    for (size_t gi = 0; gi < target_counts.size(); ++gi) {
        if (target_counts[gi] == 0) {
            continue;
        }
        const int nx = target_group_nx[gi];
        if (nx <= 0 || nx > route.large_max_nx) {
            eligibility.all_groups_supported = false;
            ++eligibility.unsupported_groups;
        }
    }
    return eligibility;
}

struct CrossEdgeDoubleBufferSlot {
    cudaStream_t stream = nullptr;
    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_end = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_end = nullptr;
    cudaEvent_t d2h_start = nullptr;
    cudaEvent_t d2h_end = nullptr;
    cudaEvent_t done = nullptr;
    uint32_t* h_qids = nullptr;
    uint32_t* h_target_offsets = nullptr;
    int* h_idx = nullptr;
    float* h_dis = nullptr;
    uint32_t* d_qids = nullptr;
    uint32_t* d_target_offsets = nullptr;
    int* d_idx = nullptr;
    float* d_dis = nullptr;
    UngGroupQueryDesc* d_group_desc = nullptr;
    UngGroupTileDesc* d_tile_desc = nullptr;
    size_t cap_q = 0;
    size_t cap_group_desc = 0;
    size_t cap_tile_desc = 0;
    int active_queries = 0;
    std::vector<ANNS::IdxType> query_global_ids;
    std::vector<ANNS::IdxType> query_target_gids;
};

void init_cross_edge_double_buffer_slot(CrossEdgeDoubleBufferSlot& s)
{
    cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking);
    cudaEventCreate(&s.h2d_start);
    cudaEventCreate(&s.h2d_end);
    cudaEventCreate(&s.kernel_start);
    cudaEventCreate(&s.kernel_end);
    cudaEventCreate(&s.d2h_start);
    cudaEventCreate(&s.d2h_end);
    cudaEventCreate(&s.done);
}

void free_cross_edge_double_buffer_slot(CrossEdgeDoubleBufferSlot& s)
{
    if (s.stream) cudaStreamDestroy(s.stream);
    if (s.h2d_start) cudaEventDestroy(s.h2d_start);
    if (s.h2d_end) cudaEventDestroy(s.h2d_end);
    if (s.kernel_start) cudaEventDestroy(s.kernel_start);
    if (s.kernel_end) cudaEventDestroy(s.kernel_end);
    if (s.d2h_start) cudaEventDestroy(s.d2h_start);
    if (s.d2h_end) cudaEventDestroy(s.d2h_end);
    if (s.done) cudaEventDestroy(s.done);
    if (s.h_qids) cudaFreeHost(s.h_qids);
    if (s.h_target_offsets) cudaFreeHost(s.h_target_offsets);
    if (s.h_idx) cudaFreeHost(s.h_idx);
    if (s.h_dis) cudaFreeHost(s.h_dis);
    if (s.d_qids) cudaFree(s.d_qids);
    if (s.d_target_offsets) cudaFree(s.d_target_offsets);
    if (s.d_idx) cudaFree(s.d_idx);
    if (s.d_dis) cudaFree(s.d_dis);
    if (s.d_group_desc) cudaFree(s.d_group_desc);
    if (s.d_tile_desc) cudaFree(s.d_tile_desc);
}

void ensure_cross_edge_double_buffer_slot_q(CrossEdgeDoubleBufferSlot& s, size_t need_q, int topk)
{
    if (need_q <= s.cap_q && s.h_qids && s.h_target_offsets && s.h_idx && s.h_dis &&
        s.d_qids && s.d_target_offsets && s.d_idx && s.d_dis) {
        return;
    }
    if (s.h_qids) cudaFreeHost(s.h_qids);
    if (s.h_target_offsets) cudaFreeHost(s.h_target_offsets);
    if (s.h_idx) cudaFreeHost(s.h_idx);
    if (s.h_dis) cudaFreeHost(s.h_dis);
    if (s.d_qids) cudaFree(s.d_qids);
    if (s.d_target_offsets) cudaFree(s.d_target_offsets);
    if (s.d_idx) cudaFree(s.d_idx);
    if (s.d_dis) cudaFree(s.d_dis);
    s.cap_q = std::max(need_q, s.cap_q * 2 + 1);
    cudaHostAlloc(&s.h_qids, s.cap_q * sizeof(uint32_t), cudaHostAllocDefault);
    cudaHostAlloc(&s.h_target_offsets, s.cap_q * sizeof(uint32_t), cudaHostAllocDefault);
    cudaHostAlloc(&s.h_idx, s.cap_q * (size_t)topk * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&s.h_dis, s.cap_q * (size_t)topk * sizeof(float), cudaHostAllocDefault);
    cudaMalloc(&s.d_qids, s.cap_q * sizeof(uint32_t));
    cudaMalloc(&s.d_target_offsets, s.cap_q * sizeof(uint32_t));
    cudaMalloc(&s.d_idx, s.cap_q * (size_t)topk * sizeof(int));
    cudaMalloc(&s.d_dis, s.cap_q * (size_t)topk * sizeof(float));
}

void ensure_cross_edge_double_buffer_slot_group_desc(CrossEdgeDoubleBufferSlot& s, size_t need)
{
    if (need <= s.cap_group_desc && s.d_group_desc) {
        return;
    }
    if (s.d_group_desc) cudaFree(s.d_group_desc);
    s.cap_group_desc = std::max(need, s.cap_group_desc * 2 + 1);
    cudaMalloc(&s.d_group_desc, s.cap_group_desc * sizeof(UngGroupQueryDesc));
}

void ensure_cross_edge_double_buffer_slot_tile_desc(CrossEdgeDoubleBufferSlot& s, size_t need)
{
    if (need <= s.cap_tile_desc && s.d_tile_desc) {
        return;
    }
    if (s.d_tile_desc) cudaFree(s.d_tile_desc);
    s.cap_tile_desc = std::max(need, s.cap_tile_desc * 2 + 1);
    cudaMalloc(&s.d_tile_desc, s.cap_tile_desc * sizeof(UngGroupTileDesc));
}

void finish_cross_edge_double_buffer_slot(
    CrossEdgeDoubleBufferSlot& s,
    bool& slot_active,
    bool db_global_merge,
    int topk,
    const std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>>& group_id_to_range,
    std::vector<ANNS::SearchQueue>& cross_group_neighbors,
    double& db_h2d_ms,
    double& db_kernel_ms,
    double& db_d2h_ms,
    double& db_writeback_ms)
{
    if (!slot_active) {
        return;
    }

    cudaEventSynchronize(s.done);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s.h2d_start, s.h2d_end);
    db_h2d_ms += ms;
    cudaEventElapsedTime(&ms, s.kernel_start, s.kernel_end);
    db_kernel_ms += ms;
    cudaEventElapsedTime(&ms, s.d2h_start, s.d2h_end);
    db_d2h_ms += ms;

    const auto wb0 = std::chrono::high_resolution_clock::now();
    if (!db_global_merge) {
        for (int qi = 0; qi < s.active_queries; ++qi) {
            const ANNS::IdxType qid = s.query_global_ids[(size_t)qi];
            const ANNS::IdxType tgt_gid = s.query_target_gids[(size_t)qi];
            const ANNS::IdxType tgt_off = group_id_to_range[tgt_gid].first;
            auto& queue = cross_group_neighbors[qid];
            if (queue.capacity() < topk) {
                queue.reserve(topk);
            }
            for (int k = 0; k < topk; ++k) {
                const int local_id = s.h_idx[(size_t)qi * (size_t)topk + (size_t)k];
                if (local_id < 0) {
                    continue;
                }
                const float dist = s.h_dis[(size_t)qi * (size_t)topk + (size_t)k];
                queue.insert(tgt_off + (ANNS::IdxType)local_id, dist);
            }
        }
    }
    const auto wb1 = std::chrono::high_resolution_clock::now();
    db_writeback_ms += std::chrono::duration<double, std::milli>(wb1 - wb0).count();
    slot_active = false;
}

struct CrossEdgeDoubleBufferChunkPack {
    size_t group_count = 0;
    size_t query_count = 0;
    std::vector<UngGroupQueryDesc> group_descs;
    std::vector<UngGroupTileDesc> tile_descs;
};

CrossEdgeDoubleBufferChunkPack pack_cross_edge_double_buffer_chunk(
    CrossEdgeDoubleBufferSlot& slot,
    const CrossEdgeDoubleBufferRoute& route,
    const std::vector<ANNS::IdxType>& target_group_ids,
    const std::vector<size_t>& target_counts,
    const std::vector<int>& target_group_nx,
    const std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>>& group_id_to_range,
    const ANNS::LabelNavGraph& label_nav_graph,
    const std::vector<uint8_t>* skip_source_groups,
    size_t group_begin,
    size_t group_end,
    int topk)
{
    CrossEdgeDoubleBufferChunkPack pack;
    pack.group_count = group_end - group_begin;

    size_t chunk_queries = 0;
    size_t group_desc_need = 0;
    size_t tile_desc_need = 0;
    for (size_t gi = group_begin; gi < group_end; ++gi) {
        const size_t nq = target_counts[gi];
        if (nq == 0) {
            continue;
        }
        chunk_queries += nq;
        const int nx = target_group_nx[gi];
        if (nx <= route.medium_max_nx) {
            group_desc_need += nq;
        } else {
            tile_desc_need += (nq + 15u) / 16u;
        }
    }

    ensure_cross_edge_double_buffer_slot_q(slot, chunk_queries, topk);
    if (group_desc_need > 0) {
        ensure_cross_edge_double_buffer_slot_group_desc(slot, group_desc_need);
    }
    if (tile_desc_need > 0) {
        ensure_cross_edge_double_buffer_slot_tile_desc(slot, tile_desc_need);
    }

    slot.query_global_ids.clear();
    slot.query_target_gids.clear();
    slot.query_global_ids.reserve(chunk_queries);
    slot.query_target_gids.reserve(chunk_queries);
    pack.group_descs.reserve(group_desc_need);
    pack.tile_descs.reserve(tile_desc_need);

    size_t qpos = 0;
    for (size_t gi = group_begin; gi < group_end; ++gi) {
        const size_t q_start = qpos;
        const int nx = target_group_nx[gi];
        if (target_counts[gi] == 0 || nx <= 0) {
            continue;
        }
        const ANNS::IdxType tgt_gid = target_group_ids[gi];
        for (auto in_gid : label_nav_graph.in_neighbors[tgt_gid]) {
            if (skip_source_groups &&
                (size_t)in_gid < skip_source_groups->size() &&
                (*skip_source_groups)[(size_t)in_gid]) {
                continue;
            }
            const auto& qrng = group_id_to_range[in_gid];
            for (ANNS::IdxType vid = qrng.first; vid < qrng.second; ++vid) {
                slot.h_qids[qpos] = (uint32_t)vid;
                slot.h_target_offsets[qpos] = (uint32_t)group_id_to_range[tgt_gid].first;
                slot.query_global_ids.push_back(vid);
                slot.query_target_gids.push_back(tgt_gid);
                ++qpos;
            }
        }

        const uint32_t x_off = (uint32_t)group_id_to_range[tgt_gid].first;
        if (nx <= route.medium_max_nx) {
            for (size_t qi = q_start; qi < qpos; ++qi) {
                UngGroupQueryDesc desc;
                desc.qi = (uint32_t)qi;
                desc.x_off = x_off;
                desc.nx = (uint32_t)nx;
                pack.group_descs.push_back(desc);
            }
        } else {
            for (size_t qi = q_start; qi < qpos; qi += 16) {
                UngGroupTileDesc desc;
                desc.q_start = (uint32_t)qi;
                desc.q_count = (uint32_t)std::min<size_t>(16, qpos - qi);
                desc.x_off = x_off;
                desc.nx = (uint32_t)nx;
                pack.tile_descs.push_back(desc);
            }
        }
    }

    slot.active_queries = (int)qpos;
    pack.query_count = qpos;
    return pack;
}

void enqueue_cross_edge_double_buffer_chunk(
    CrossEdgeDoubleBufferSlot& slot,
    bool& slot_active,
    const CrossEdgeDoubleBufferRoute& route,
    const CrossEdgeDoubleBufferChunkPack& pack,
    bool db_global_merge,
    int dim,
    int topk)
{
    const size_t qpos = pack.query_count;

    cudaEventRecord(slot.h2d_start, slot.stream);
    cudaMemcpyAsync(slot.d_qids, slot.h_qids, qpos * sizeof(uint32_t), cudaMemcpyHostToDevice, slot.stream);
    cudaMemcpyAsync(slot.d_target_offsets, slot.h_target_offsets, qpos * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, slot.stream);
    if (!pack.group_descs.empty()) {
        cudaMemcpyAsync(slot.d_group_desc, pack.group_descs.data(),
                        pack.group_descs.size() * sizeof(UngGroupQueryDesc),
                        cudaMemcpyHostToDevice, slot.stream);
    }
    if (!pack.tile_descs.empty()) {
        cudaMemcpyAsync(slot.d_tile_desc, pack.tile_descs.data(),
                        pack.tile_descs.size() * sizeof(UngGroupTileDesc),
                        cudaMemcpyHostToDevice, slot.stream);
    }
    cudaEventRecord(slot.h2d_end, slot.stream);

    cudaEventRecord(slot.kernel_start, slot.stream);
    int threads = 256;
    const long long init_total = (long long)qpos * (long long)topk;
    int init_grid_x = (int)((init_total + threads - 1) / threads);
    if (init_grid_x > 65535) init_grid_x = 65535;
    init_topk_kernel<<<dim3((unsigned)init_grid_x, 1u, 1u), dim3(threads, 1, 1), 0, slot.stream>>>(
        (int)qpos, topk, slot.d_idx, slot.d_dis);

    if (!pack.group_descs.empty()) {
        int launch_warps = std::min(route.medium_warps, 16);
        if (launch_warps < 1) launch_warps = 1;
        dim3 mg_grid((unsigned)pack.group_descs.size(), 1u, 1u);
        dim3 mg_block((unsigned)(launch_warps * 32), 1u, 1u);
        size_t mg_smem = (size_t)dim * sizeof(float)
                       + (size_t)route.medium_max_nx * (sizeof(float) + sizeof(int));
        ung_group_desc_topk_fused_global_kernel<<<mg_grid, mg_block, mg_smem, slot.stream>>>(
            slot.d_group_desc, (int)pack.group_descs.size(), nullptr, nullptr,
            g_d_all_X, g_d_all_norm, slot.d_qids, dim, topk, route.medium_max_nx, 1, slot.d_idx, slot.d_dis);
    }
    if (!pack.tile_descs.empty()) {
        int launch_warps = route.large_warps;
        if (launch_warps < 1) launch_warps = 1;
        if (launch_warps > 16) launch_warps = 16;
        dim3 tg_grid((unsigned)((pack.tile_descs.size() + (size_t)launch_warps - 1) / (size_t)launch_warps), 1u, 1u);
        dim3 tg_block((unsigned)(launch_warps * 32), 1u, 1u);
        size_t tg_smem = ((size_t)8 * 16
                        + (size_t)launch_warps * ((size_t)16 * 8 + (size_t)8 * 16 + (size_t)16 * 16)) * sizeof(float);
        ung_group_tile_tf32_wmma_topk_global_kernel<<<tg_grid, tg_block, tg_smem, slot.stream>>>(
            slot.d_tile_desc, (int)pack.tile_descs.size(), nullptr, nullptr,
            g_d_all_X, g_d_all_norm, slot.d_qids, dim, topk, 1, slot.d_idx, slot.d_dis);
    }
    if (db_global_merge) {
        int merge_threads = 256;
        int merge_grid_x = (int)((qpos + (size_t)merge_threads - 1) / (size_t)merge_threads);
        if (merge_grid_x > 65535) merge_grid_x = 65535;
        ung_merge_local_topk_rows_to_global_kernel<<<dim3((unsigned)merge_grid_x, 1u, 1u),
                                                     dim3((unsigned)merge_threads, 1u, 1u),
                                                     0, slot.stream>>>(
            slot.d_idx,
            slot.d_dis,
            slot.d_qids,
            slot.d_target_offsets,
            (int)qpos,
            topk,
            g_d_global_idx,
            g_d_global_dis,
            g_d_global_locks);
    }
    cudaEventRecord(slot.kernel_end, slot.stream);

    cudaEventRecord(slot.d2h_start, slot.stream);
    if (!db_global_merge) {
        cudaMemcpyAsync(slot.h_idx, slot.d_idx, qpos * (size_t)topk * sizeof(int), cudaMemcpyDeviceToHost, slot.stream);
        cudaMemcpyAsync(slot.h_dis, slot.d_dis, qpos * (size_t)topk * sizeof(float), cudaMemcpyDeviceToHost, slot.stream);
    }
    cudaEventRecord(slot.d2h_end, slot.stream);
    cudaEventRecord(slot.done, slot.stream);
    slot_active = true;
}

void finalize_cross_edge_double_buffer_global_merge(
    int total_points_for_merge,
    int topk,
    const CrossEdgeTopkOutputWriter& writer,
    double& db_d2h_ms,
    double& db_writeback_ms)
{
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    cudaMemcpyAsync(g_h_idx, g_d_global_idx,
                    (size_t)total_points_for_merge * (size_t)topk * sizeof(int),
                    cudaMemcpyDeviceToHost);
    if (!writer.id_only_writeback) {
        cudaMemcpyAsync(g_h_dis, g_d_global_dis,
                        (size_t)total_points_for_merge * (size_t)topk * sizeof(float),
                        cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    db_d2h_ms += ms;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);

    const auto wb0 = std::chrono::high_resolution_clock::now();
    if (writer.writes_flat_ids()) {
        if (writer.flat_ids == nullptr) {
            throw std::runtime_error("cross-edge double-buffer flat-id writer requires flat_ids storage.");
        }
        writer.flat_ids->assign(
            (size_t)total_points_for_merge * (size_t)topk,
            std::numeric_limits<ANNS::IdxType>::max());
    } else if (writer.writes_id_vectors()) {
        if (writer.id_vectors == nullptr) {
            throw std::runtime_error("cross-edge double-buffer id-vector writer requires id_vectors storage.");
        }
    } else if (writer.search_queues == nullptr) {
        throw std::runtime_error("cross-edge double-buffer SearchQueue writer requires search queue storage.");
    }
    #pragma omp parallel for schedule(static, 1024)
    for (int qid_i = 0; qid_i < total_points_for_merge; ++qid_i) {
        if (writer.writes_flat_ids()) {
            const size_t out_base = (size_t)qid_i * (size_t)topk;
            for (int k = 0; k < topk; ++k) {
                const int gid = g_h_idx[out_base + (size_t)k];
                if (gid >= 0) {
                    (*writer.flat_ids)[out_base + (size_t)k] = (ANNS::IdxType)gid;
                }
            }
        } else if (writer.writes_id_vectors()) {
            auto& ids = (*writer.id_vectors)[(ANNS::IdxType)qid_i];
            for (int k = 0; k < topk; ++k) {
                const int gid = g_h_idx[(size_t)qid_i * (size_t)topk + (size_t)k];
                if (gid >= 0) {
                    ids.emplace_back((ANNS::IdxType)gid);
                }
            }
        } else {
            for (int k = 0; k < topk; ++k) {
                const int gid = g_h_idx[(size_t)qid_i * (size_t)topk + (size_t)k];
                if (gid < 0) {
                    continue;
                }
                const float dist = writer.id_only_writeback
                    ? (float)k
                    : g_h_dis[(size_t)qid_i * (size_t)topk + (size_t)k];
                auto& queue = (*writer.search_queues)[(ANNS::IdxType)qid_i];
                if (queue.capacity() < topk) {
                    queue.reserve(topk);
                }
                queue.insert((ANNS::IdxType)gid, dist);
            }
        }
    }
    const auto wb1 = std::chrono::high_resolution_clock::now();
    db_writeback_ms += std::chrono::duration<double, std::milli>(wb1 - wb0).count();
}

template <typename FinishSlotFn, typename EnqueueChunkFn>
void run_cross_edge_double_buffer_chunks(
    const std::vector<size_t>& target_counts,
    int chunk_queries,
    FinishSlotFn finish_slot,
    EnqueueChunkFn enqueue_chunk)
{
    size_t group_begin = 0;
    size_t chunk_index = 0;
    while (group_begin < target_counts.size()) {
        size_t group_end = group_begin;
        size_t chunk_q = 0;
        while (group_end < target_counts.size()) {
            const size_t nq = target_counts[group_end];
            if (group_end > group_begin && chunk_q > 0 &&
                nq > 0 && chunk_q + nq > (size_t)chunk_queries) {
                break;
            }
            chunk_q += nq;
            ++group_end;
            if (chunk_q >= (size_t)chunk_queries) {
                break;
            }
        }
        if (group_end == group_begin) {
            ++group_end;
        }
        const int si = (int)(chunk_index & 1u);
        finish_slot(si);
        enqueue_chunk(si, group_begin, group_end);
        group_begin = group_end;
        ++chunk_index;
    }
    finish_slot(0);
    finish_slot(1);
}

#endif // ANNS_GPU_CROSS_EDGE_DOUBLE_BUFFER_CUH
