// Descriptor-batched small/medium/TF32 fused launches for the regular
// cross-edge batched path.
//
// Included from gpu_gemm_topk.cu after kernels, shared descriptor buffers, and
// ensure_* helpers are available. This helper owns launch mechanics only; the
// caller still owns routing, descriptor construction, counters, and output.

inline void launch_cross_edge_bucket_fused_groups(
    cudaStream_t stream,
    bool bucket_group_fused,
    int dim,
    int topk,
    int large_group_warps,
    bool tf32_group_2d,
    int small_group_warps,
    int small_group_max_nx,
    int medium_group_warps,
    int medium_group_max_nx,
    bool direct_qid_all_fused,
    bool direct_qid_fused,
    const std::vector<UngGroupQueryDesc>& small_group_descs,
    const std::vector<UngGroupQueryDesc>& medium_group_descs,
    const std::vector<UngGroupTileDesc>& tf32_tile_descs,
    const std::vector<UngGroupTileDesc>& tf32_group_descs,
    const std::vector<uint32_t>& tf32_group_tile_offsets,
    const float* d_Q,
    const float* d_q_norm,
    const float* d_all_X,
    const float* d_all_norm,
    const uint32_t* d_q_ids,
    int* d_idx,
    float* d_dis) {
    if (!bucket_group_fused || d_all_norm == nullptr) return;

    if (!tf32_group_descs.empty()) {
        const uint32_t total_prefix_tiles = tf32_group_tile_offsets.empty()
            ? 0u
            : tf32_group_tile_offsets.back();
        ensure_device_group_tile_desc_buffer(tf32_group_descs.size());
        cudaMemcpyAsync(g_d_group_tile_desc,
                        tf32_group_descs.data(),
                        tf32_group_descs.size() * sizeof(UngGroupTileDesc),
                        cudaMemcpyHostToDevice,
                        stream);
        int launch_warps = large_group_warps;
        if (launch_warps < 1) launch_warps = 1;
        if (launch_warps > 16) launch_warps = 16;
        dim3 pg_block((unsigned)(launch_warps * 32), 1u, 1u);
        const int tf32_group_kt = tf32_group_2d ? 4 : 1;
        size_t pg_smem = ((size_t)tf32_group_kt * 8 * 16
                        + (size_t)launch_warps * ((size_t)tf32_group_kt * 16 * 8 + (size_t)16 * 16)) * sizeof(float);
        if (tf32_group_2d) {
            uint32_t max_group_tiles = 0;
            for (size_t i = 1; i < tf32_group_tile_offsets.size(); ++i) {
                max_group_tiles = std::max(max_group_tiles,
                                           tf32_group_tile_offsets[i] - tf32_group_tile_offsets[i - 1]);
            }
            dim3 pg_grid((unsigned)(((size_t)max_group_tiles + (size_t)launch_warps - 1) / (size_t)launch_warps),
                         (unsigned)tf32_group_descs.size(),
                         1u);
            ung_group_2d_tf32_wmma_topk_global_kernel<<<pg_grid, pg_block, pg_smem, stream>>>(
                g_d_group_tile_desc,
                (int)tf32_group_descs.size(),
                d_Q,
                d_q_norm,
                d_all_X,
                d_all_norm,
                d_q_ids,
                dim,
                topk,
                direct_qid_all_fused ? 1 : 0,
                d_idx,
                d_dis);
        } else {
            ensure_device_tf32_group_tile_offsets(tf32_group_tile_offsets.size());
            cudaMemcpyAsync(g_d_tf32_group_tile_offsets,
                            tf32_group_tile_offsets.data(),
                            tf32_group_tile_offsets.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice,
                            stream);
            dim3 pg_grid((unsigned)(((size_t)total_prefix_tiles + (size_t)launch_warps - 1) / (size_t)launch_warps), 1u, 1u);
            ung_group_prefix_tf32_wmma_topk_global_kernel<<<pg_grid, pg_block, pg_smem, stream>>>(
                g_d_group_tile_desc,
                g_d_tf32_group_tile_offsets,
                (int)tf32_group_descs.size(),
                total_prefix_tiles,
                d_Q,
                d_q_norm,
                d_all_X,
                d_all_norm,
                dim,
                topk,
                d_idx,
                d_dis);
        }
        cudaError_t tf32_prefix_err = cudaGetLastError();
        if (tf32_prefix_err != cudaSuccess) {
            prof_logf("[ERROR] tf32_group_prefix launch failed: %s (groups=%zu tiles=%u dim=%d topk=%d warps=%d 2d=%d)",
                      cudaGetErrorString(tf32_prefix_err), tf32_group_descs.size(),
                      total_prefix_tiles, dim, topk, launch_warps, tf32_group_2d ? 1 : 0);
            throw std::runtime_error("tf32_group_prefix launch failed.");
        }
    }

    if (!tf32_tile_descs.empty()) {
        ensure_device_group_tile_desc_buffer(tf32_tile_descs.size());
        cudaMemcpyAsync(g_d_group_tile_desc,
                        tf32_tile_descs.data(),
                        tf32_tile_descs.size() * sizeof(UngGroupTileDesc),
                        cudaMemcpyHostToDevice,
                        stream);
        int launch_warps = large_group_warps;
        if (launch_warps < 1) launch_warps = 1;
        if (launch_warps > 16) launch_warps = 16;
        dim3 tg_grid((unsigned)((tf32_tile_descs.size() + (size_t)launch_warps - 1) / (size_t)launch_warps), 1u, 1u);
        dim3 tg_block((unsigned)(launch_warps * 32), 1u, 1u);
        size_t tg_smem = ((size_t)8 * 16
                        + (size_t)launch_warps * ((size_t)16 * 8 + (size_t)8 * 16 + (size_t)16 * 16)) * sizeof(float);
        ung_group_tile_tf32_wmma_topk_global_kernel<<<tg_grid, tg_block, tg_smem, stream>>>(
            g_d_group_tile_desc,
            (int)tf32_tile_descs.size(),
            d_Q,
            d_q_norm,
            d_all_X,
            d_all_norm,
            nullptr,
            dim,
            topk,
            0,
            d_idx,
            d_dis);
        cudaError_t tf32_batch_err = cudaGetLastError();
        if (tf32_batch_err != cudaSuccess) {
            prof_logf("[ERROR] tf32_tile_fused launch failed: %s (tiles=%zu dim=%d topk=%d warps=%d)",
                      cudaGetErrorString(tf32_batch_err), tf32_tile_descs.size(), dim, topk, launch_warps);
            throw std::runtime_error("tf32_tile_fused launch failed.");
        }
    }

    if (!small_group_descs.empty()) {
        ensure_device_group_desc_buffer(small_group_descs.size());
        cudaMemcpyAsync(g_d_group_desc,
                        small_group_descs.data(),
                        small_group_descs.size() * sizeof(UngGroupQueryDesc),
                        cudaMemcpyHostToDevice,
                        stream);
        int launch_warps = std::max(small_group_warps, std::min(small_group_max_nx, 16));
        if (launch_warps > 16) launch_warps = 16;
        if (launch_warps < 1) launch_warps = 1;
        dim3 sg_grid((unsigned)small_group_descs.size(), 1u, 1u);
        dim3 sg_block((unsigned)(launch_warps * 32), 1u, 1u);
        size_t sg_smem = (size_t)dim * sizeof(float)
                       + (size_t)small_group_max_nx * (sizeof(float) + sizeof(int));
        ung_group_desc_topk_fused_global_kernel<<<sg_grid, sg_block, sg_smem, stream>>>(
            g_d_group_desc,
            (int)small_group_descs.size(),
            d_Q,
            d_q_norm,
            d_all_X,
            d_all_norm,
            d_q_ids,
            dim,
            topk,
            small_group_max_nx,
            direct_qid_fused ? 1 : 0,
            d_idx,
            d_dis);
        cudaError_t sg_err = cudaGetLastError();
        if (sg_err != cudaSuccess) {
            prof_logf("[ERROR] small group_desc launch failed: %s (desc=%zu direct_qid=%d)",
                      cudaGetErrorString(sg_err), small_group_descs.size(), direct_qid_fused ? 1 : 0);
            throw std::runtime_error("small group_desc launch failed.");
        }
    }

    if (!medium_group_descs.empty()) {
        ensure_device_group_desc_buffer(medium_group_descs.size());
        cudaMemcpyAsync(g_d_group_desc,
                        medium_group_descs.data(),
                        medium_group_descs.size() * sizeof(UngGroupQueryDesc),
                        cudaMemcpyHostToDevice,
                        stream);
        int launch_warps = std::min(medium_group_warps, 16);
        if (launch_warps < 1) launch_warps = 1;
        dim3 mg_grid((unsigned)medium_group_descs.size(), 1u, 1u);
        dim3 mg_block((unsigned)(launch_warps * 32), 1u, 1u);
        size_t mg_smem = (size_t)dim * sizeof(float)
                       + (size_t)medium_group_max_nx * (sizeof(float) + sizeof(int));
        ung_group_desc_topk_fused_global_kernel<<<mg_grid, mg_block, mg_smem, stream>>>(
            g_d_group_desc,
            (int)medium_group_descs.size(),
            d_Q,
            d_q_norm,
            d_all_X,
            d_all_norm,
            d_q_ids,
            dim,
            topk,
            medium_group_max_nx,
            direct_qid_fused ? 1 : 0,
            d_idx,
            d_dis);
        cudaError_t mg_err = cudaGetLastError();
        if (mg_err != cudaSuccess) {
            prof_logf("[ERROR] medium group_desc launch failed: %s (desc=%zu direct_qid=%d)",
                      cudaGetErrorString(mg_err), medium_group_descs.size(), direct_qid_fused ? 1 : 0);
            throw std::runtime_error("medium group_desc launch failed.");
        }
    }
}
