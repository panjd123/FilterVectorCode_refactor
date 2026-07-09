#ifndef ANNS_GPU_CROSS_EDGE_BUCKET_DESCRIPTORS_CUH
#define ANNS_GPU_CROSS_EDGE_BUCKET_DESCRIPTORS_CUH

// Host-side descriptor assembly for descriptor-batched fused cross-edge routes.
//
// Launch mechanics live in gpu_cross_edge_bucket_fused.cuh. This file only
// decides descriptor buckets and fills POD descriptors consumed by those
// kernels.

struct CrossEdgeBucketDescriptors {
    // 0=none, 1=small descriptor, 2=medium descriptor, 3=TF32 tile/group descriptor.
    std::vector<uint8_t> group_kind;
    std::vector<UngGroupQueryDesc> small_group_descs;
    std::vector<UngGroupQueryDesc> medium_group_descs;
    std::vector<UngGroupTileDesc> tf32_tile_descs;
    std::vector<UngGroupTileDesc> tf32_group_descs;
    std::vector<uint32_t> tf32_group_tile_offsets;
};

CrossEdgeBucketDescriptors build_cross_edge_bucket_descriptors(
    const std::vector<ANNS::IdxType>& target_group_ids,
    const std::vector<size_t>& target_counts,
    const std::vector<int>& target_group_nx,
    const std::vector<size_t>& target_offsets,
    const std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>>& group_id_to_range,
    int dim,
    int topk,
    bool bucket_group_fused,
    bool use_naive_cuda,
    bool has_all_norm,
    bool naive_group_fused,
    bool singleton_fastpath,
    bool large_group_fused,
    int large_group_fused_mode,
    int large_group_min_nx,
    int large_group_max_nx,
    bool naive_heavy_sgemm,
    int naive_heavy_nx,
    int naive_heavy_nq,
    long long naive_heavy_work_thresh,
    bool small_group_fused,
    int small_group_max_nx,
    bool medium_group_fused,
    int medium_group_max_nx,
    bool tf32_group_prefix,
    bool tf32_group_2d)
{
    CrossEdgeBucketDescriptors out;
    out.group_kind.assign(target_group_ids.size(), (uint8_t)0);
    if (!bucket_group_fused || !use_naive_cuda || topk > 32 || !has_all_norm || naive_group_fused) {
        return out;
    }

    std::vector<size_t> small_desc_offsets(target_group_ids.size() + 1, 0);
    std::vector<size_t> medium_desc_offsets(target_group_ids.size() + 1, 0);

    for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
        const int nq_g = (int)target_counts[gi];
        if (nq_g <= 0) continue;
        const int nx = target_group_nx[gi];
        if (nx <= 0) continue;
        if (singleton_fastpath && nx == 1) continue;

        const bool tf32_tile_route =
            large_group_fused &&
            large_group_fused_mode == 2 &&
            topk <= 16 &&
            nx >= large_group_min_nx &&
            nx <= large_group_max_nx;

        bool heavy_route = false;
        if (naive_heavy_sgemm) {
            const unsigned long long work =
                (unsigned long long)nq_g * (unsigned long long)nx * (unsigned long long)dim;
            heavy_route = (nx >= naive_heavy_nx &&
                           nq_g >= naive_heavy_nq &&
                           work >= (unsigned long long)naive_heavy_work_thresh);
        }
        if (tf32_tile_route) {
            out.group_kind[gi] = (uint8_t)3;
        } else if (heavy_route) {
            continue;
        }

        if (out.group_kind[gi] == 3) {
            // Batched TF32 path builds tile descriptors below.
        } else if (small_group_fused && nx <= small_group_max_nx) {
            out.group_kind[gi] = (uint8_t)1;
            small_desc_offsets[gi + 1] = (size_t)nq_g;
        } else if (medium_group_fused && nx > small_group_max_nx && nx <= medium_group_max_nx) {
            out.group_kind[gi] = (uint8_t)2;
            medium_desc_offsets[gi + 1] = (size_t)nq_g;
        }
    }

    for (size_t i = 1; i < small_desc_offsets.size(); ++i) {
        small_desc_offsets[i] += small_desc_offsets[i - 1];
    }
    for (size_t i = 1; i < medium_desc_offsets.size(); ++i) {
        medium_desc_offsets[i] += medium_desc_offsets[i - 1];
    }
    out.small_group_descs.resize(small_desc_offsets.back());
    out.medium_group_descs.resize(medium_desc_offsets.back());

    size_t tf32_tile_count = 0;
    size_t tf32_group_count = 0;
    for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
        if (out.group_kind[gi] == 3) {
            tf32_tile_count += ((size_t)target_counts[gi] + 15u) / 16u;
            tf32_group_count += 1;
        }
    }
    if (tf32_group_prefix || tf32_group_2d) {
        out.tf32_group_descs.reserve(tf32_group_count);
        out.tf32_group_tile_offsets.reserve(tf32_group_count + 1);
    } else {
        out.tf32_tile_descs.reserve(tf32_tile_count);
    }

    #pragma omp parallel for schedule(static, 256)
    for (int gi_i = 0; gi_i < (int)target_group_ids.size(); ++gi_i) {
        size_t gi = (size_t)gi_i;
        const uint8_t kind = out.group_kind[gi];
        if (kind == 0) continue;
        if (kind == 3) continue;

        const size_t q_start = target_offsets[gi];
        const uint32_t x_off = (uint32_t)group_id_to_range[target_group_ids[gi]].first;
        const uint32_t nx = (uint32_t)target_group_nx[gi];
        const int nq_g = (int)target_counts[gi];

        if (kind == 1) {
            const size_t base = small_desc_offsets[gi];
            for (int q_local = 0; q_local < nq_g; ++q_local) {
                UngGroupQueryDesc desc;
                desc.qi = (uint32_t)(q_start + (size_t)q_local);
                desc.x_off = x_off;
                desc.nx = nx;
                out.small_group_descs[base + (size_t)q_local] = desc;
            }
        } else {
            const size_t base = medium_desc_offsets[gi];
            for (int q_local = 0; q_local < nq_g; ++q_local) {
                UngGroupQueryDesc desc;
                desc.qi = (uint32_t)(q_start + (size_t)q_local);
                desc.x_off = x_off;
                desc.nx = nx;
                out.medium_group_descs[base + (size_t)q_local] = desc;
            }
        }
    }

    for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
        if (out.group_kind[gi] != 3) continue;
        const size_t q_start = target_offsets[gi];
        const uint32_t x_off = (uint32_t)group_id_to_range[target_group_ids[gi]].first;
        const uint32_t nx = (uint32_t)target_group_nx[gi];
        const int nq_g = (int)target_counts[gi];
        if (tf32_group_prefix || tf32_group_2d) {
            if (out.tf32_group_tile_offsets.empty()) out.tf32_group_tile_offsets.push_back(0);
            UngGroupTileDesc desc;
            desc.q_start = (uint32_t)q_start;
            desc.q_count = (uint32_t)nq_g;
            desc.x_off = x_off;
            desc.nx = nx;
            out.tf32_group_descs.push_back(desc);
            const uint32_t tiles = (uint32_t)(((size_t)nq_g + 15u) / 16u);
            out.tf32_group_tile_offsets.push_back(out.tf32_group_tile_offsets.back() + tiles);
        } else {
            for (int q_local = 0; q_local < nq_g; q_local += 16) {
                UngGroupTileDesc desc;
                desc.q_start = (uint32_t)(q_start + (size_t)q_local);
                desc.q_count = (uint32_t)std::min(16, nq_g - q_local);
                desc.x_off = x_off;
                desc.nx = nx;
                out.tf32_tile_descs.push_back(desc);
            }
        }
    }

    return out;
}

#endif // ANNS_GPU_CROSS_EDGE_BUCKET_DESCRIPTORS_CUH
