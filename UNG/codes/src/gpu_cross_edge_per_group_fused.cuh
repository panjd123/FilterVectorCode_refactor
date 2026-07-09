// Immediate per-group fused kernel route for the regular batched cross-edge
// path. This helper handles only routes that launch a kernel immediately for a
// single target group. Descriptor-batched routes and separate-topK fallback stay
// in their own helpers.

inline bool try_launch_cross_edge_per_group_fused(
    cudaStream_t stream,
    size_t gi,
    ANNS::IdxType tgt_gid,
    int nq_g,
    int nx,
    int x_off,
    size_t q_start,
    int dim,
    int topk,
    const float* dQ,
    const float* dX,
    const float* dXnorm_group,
    const float* d_q_norm,
    const uint32_t* d_q_ids,
    int* dBestI,
    float* dBestD,
    int* d_global_idx,
    float* d_global_dis,
    int* d_global_locks,
    bool group_use_naive,
    bool naive_group_fused,
    int naive_group_threads,
    bool small_group_fused,
    int small_group_max_nx,
    int small_group_warps,
    bool medium_group_fused,
    int medium_group_max_nx,
    int medium_group_warps,
    bool medium_group_sync_debug,
    bool large_group_fused,
    int large_group_min_nx,
    int large_group_max_nx,
    int large_group_warps,
    int large_group_fused_mode,
    bool bucket_group_fused,
    uint8_t bucket_group_kind,
    bool allow_group_global_merge_direct,
    size_t& small_group_fused_group_count,
    size_t& small_group_fused_query_count,
    size_t& medium_group_fused_group_count,
    size_t& medium_group_fused_query_count,
    size_t& large_group_fused_group_count,
    size_t& large_group_fused_query_count,
    size_t& global_merge_direct_group_count,
    size_t& global_merge_direct_query_count) {
    if (bucket_group_fused && bucket_group_kind == 3) {
        large_group_fused_group_count += 1;
        large_group_fused_query_count += (size_t)nq_g;
        return true;
    }

    const bool use_large_group_fused_for_group =
        large_group_fused && topk <= 32 && dXnorm_group != nullptr &&
        nx >= large_group_min_nx && nx <= large_group_max_nx;

    if (use_large_group_fused_for_group) {
        int launch_warps = large_group_warps;
        if (launch_warps < 1) launch_warps = 1;
        if (launch_warps > 16) launch_warps = 16;
        dim3 lg_block((unsigned)(launch_warps * 32), 1u, 1u);
        if (large_group_fused_mode == 2 && topk <= 16) {
            dim3 lg_grid((unsigned)((nq_g + launch_warps * 16 - 1) / (launch_warps * 16)), 1u, 1u);
            size_t lg_smem = ((size_t)8 * 16
                            + (size_t)launch_warps * ((size_t)16 * 8 + (size_t)16 * 16)) * sizeof(float);
            if (allow_group_global_merge_direct) {
                ung_large_group_tf32_wmma_topk_global_merge_kernel<<<lg_grid, lg_block, lg_smem, stream>>>(
                    dQ,
                    d_q_norm + q_start,
                    dX,
                    dXnorm_group,
                    d_q_ids,
                    (int)q_start,
                    x_off,
                    nq_g,
                    nx,
                    dim,
                    topk,
                    d_global_idx,
                    d_global_dis,
                    d_global_locks);
                global_merge_direct_group_count += 1;
                global_merge_direct_query_count += (size_t)nq_g;
            } else {
                ung_large_group_tf32_wmma_topk_global_kernel<<<lg_grid, lg_block, lg_smem, stream>>>(
                    dQ,
                    d_q_norm + q_start,
                    dX,
                    dXnorm_group,
                    nq_g,
                    nx,
                    dim,
                    topk,
                    dBestI,
                    dBestD);
            }
        } else if (large_group_fused_mode == 1) {
            dim3 lg_grid((unsigned)((nq_g + launch_warps - 1) / launch_warps), 1u, 1u);
            if (allow_group_global_merge_direct) {
                ung_large_group_warp_query_topk_global_merge_kernel<<<lg_grid, lg_block, 0, stream>>>(
                    dQ,
                    d_q_norm + q_start,
                    dX,
                    dXnorm_group,
                    d_q_ids,
                    (int)q_start,
                    x_off,
                    nq_g,
                    nx,
                    dim,
                    topk,
                    d_global_idx,
                    d_global_dis,
                    d_global_locks);
                global_merge_direct_group_count += 1;
                global_merge_direct_query_count += (size_t)nq_g;
            } else {
                ung_large_group_warp_query_topk_global_kernel<<<lg_grid, lg_block, 0, stream>>>(
                    dQ,
                    d_q_norm + q_start,
                    dX,
                    dXnorm_group,
                    nq_g,
                    nx,
                    dim,
                    topk,
                    dBestI,
                    dBestD);
            }
        } else {
            dim3 lg_grid((unsigned)nq_g, 1u, 1u);
            size_t lg_smem = (size_t)launch_warps * (size_t)topk * (sizeof(float) + sizeof(int));
            if (allow_group_global_merge_direct) {
                ung_large_group_warp_fused_topk_global_merge_kernel<<<lg_grid, lg_block, lg_smem, stream>>>(
                    dQ,
                    d_q_norm + q_start,
                    dX,
                    dXnorm_group,
                    d_q_ids,
                    (int)q_start,
                    x_off,
                    nq_g,
                    nx,
                    dim,
                    topk,
                    d_global_idx,
                    d_global_dis,
                    d_global_locks);
                global_merge_direct_group_count += 1;
                global_merge_direct_query_count += (size_t)nq_g;
            } else {
                ung_large_group_warp_fused_topk_global_kernel<<<lg_grid, lg_block, lg_smem, stream>>>(
                    dQ,
                    d_q_norm + q_start,
                    dX,
                    dXnorm_group,
                    nq_g,
                    nx,
                    dim,
                    topk,
                    dBestI,
                    dBestD);
            }
        }
        cudaError_t large_fused_err = cudaGetLastError();
        if (large_fused_err != cudaSuccess) {
            prof_logf("[ERROR] large_group_fused launch failed: %s (nq=%d nx=%d dim=%d topk=%d warps=%d mode=%d)",
                      cudaGetErrorString(large_fused_err), nq_g, nx, dim, topk, launch_warps, large_group_fused_mode);
            throw std::runtime_error("large_group_fused launch failed.");
        }
        large_group_fused_group_count += 1;
        large_group_fused_query_count += (size_t)nq_g;
        return true;
    }

    if (group_use_naive && naive_group_fused && topk <= 32 && nx <= 1024) {
        size_t fused_smem = (size_t)dim * sizeof(float)
                          + (size_t)naive_group_threads * (size_t)topk * (sizeof(float) + sizeof(int));
        ung_group_fused_topk_global_kernel<<<(unsigned)nq_g, (unsigned)naive_group_threads, fused_smem, stream>>>(
            dQ,
            d_q_norm + q_start,
            dX,
            dXnorm_group,
            nq_g,
            nx,
            dim,
            topk,
            dBestI,
            dBestD);
        return true;
    }

    if (group_use_naive && small_group_fused && nx <= small_group_max_nx && topk <= 32 && dXnorm_group != nullptr) {
        if (bucket_group_fused && bucket_group_kind == 1) {
            small_group_fused_group_count += 1;
            small_group_fused_query_count += (size_t)nq_g;
            return true;
        }
        int launch_warps = std::max(nx, small_group_warps);
        if (launch_warps > 16) launch_warps = 16;
        dim3 sg_grid((unsigned)nq_g, 1u, 1u);
        dim3 sg_block((unsigned)(launch_warps * 32), 1u, 1u);
        size_t sg_smem = (size_t)dim * sizeof(float)
                       + (size_t)small_group_max_nx * (sizeof(float) + sizeof(int));
        if (allow_group_global_merge_direct) {
            ung_small_group_topk_fused_global_merge_kernel<<<sg_grid, sg_block, sg_smem, stream>>>(
                dQ,
                d_q_norm + q_start,
                dX,
                dXnorm_group,
                d_q_ids,
                (int)q_start,
                x_off,
                nq_g,
                nx,
                dim,
                topk,
                small_group_max_nx,
                d_global_idx,
                d_global_dis,
                d_global_locks);
            global_merge_direct_group_count += 1;
            global_merge_direct_query_count += (size_t)nq_g;
        } else {
            ung_small_group_topk_fused_global_kernel<<<sg_grid, sg_block, sg_smem, stream>>>(
                dQ,
                d_q_norm + q_start,
                dX,
                dXnorm_group,
                nq_g,
                nx,
                dim,
                topk,
                small_group_max_nx,
                dBestI,
                dBestD);
        }
        small_group_fused_group_count += 1;
        small_group_fused_query_count += (size_t)nq_g;
        return true;
    }

    if (group_use_naive && medium_group_fused && nx > small_group_max_nx &&
        nx <= medium_group_max_nx && topk <= 32 && dXnorm_group != nullptr) {
        if (bucket_group_fused && bucket_group_kind == 2) {
            medium_group_fused_group_count += 1;
            medium_group_fused_query_count += (size_t)nq_g;
            return true;
        }
        int launch_warps = std::min(medium_group_warps, 16);
        if (launch_warps < 1) launch_warps = 1;
        dim3 mg_grid((unsigned)nq_g, 1u, 1u);
        dim3 mg_block((unsigned)(launch_warps * 32), 1u, 1u);
        const int group_max_nx = nx;
        size_t mg_smem = (size_t)dim * sizeof(float)
                       + (size_t)group_max_nx * (sizeof(float) + sizeof(int));
        if (allow_group_global_merge_direct) {
            ung_medium_group_topk_fused_global_merge_kernel<<<mg_grid, mg_block, mg_smem, stream>>>(
                dQ,
                d_q_norm + q_start,
                dX,
                dXnorm_group,
                d_q_ids,
                (int)q_start,
                x_off,
                nq_g,
                nx,
                dim,
                topk,
                group_max_nx,
                d_global_idx,
                d_global_dis,
                d_global_locks);
            global_merge_direct_group_count += 1;
            global_merge_direct_query_count += (size_t)nq_g;
        } else {
            ung_medium_group_topk_fused_global_kernel<<<mg_grid, mg_block, mg_smem, stream>>>(
                dQ,
                d_q_norm + q_start,
                dX,
                dXnorm_group,
                nq_g,
                nx,
                dim,
                topk,
                group_max_nx,
                dBestI,
                dBestD);
        }
        cudaError_t medium_fused_err = cudaGetLastError();
        if (medium_fused_err != cudaSuccess) {
            prof_logf("[ERROR] medium_group_fused launch failed: %s (gi=%zu gid=%u nq=%d nx=%d dim=%d topk=%d max_nx=%d warps=%d smem=%zu)",
                      cudaGetErrorString(medium_fused_err), gi, (unsigned)tgt_gid, nq_g, nx, dim, topk,
                      group_max_nx, launch_warps, mg_smem);
            throw std::runtime_error("medium_group_fused launch failed.");
        }
        if (medium_group_sync_debug) {
            cudaError_t medium_sync_err = cudaStreamSynchronize(stream);
            if (medium_sync_err != cudaSuccess) {
                prof_logf("[ERROR] medium_group_fused sync failed: %s (gi=%zu gid=%u nq=%d nx=%d dim=%d topk=%d max_nx=%d warps=%d smem=%zu)",
                          cudaGetErrorString(medium_sync_err), gi, (unsigned)tgt_gid, nq_g, nx, dim, topk,
                          group_max_nx, launch_warps, mg_smem);
                throw std::runtime_error("medium_group_fused sync failed.");
            }
        }
        medium_group_fused_group_count += 1;
        medium_group_fused_query_count += (size_t)nq_g;
        return true;
    }

    return false;
}
