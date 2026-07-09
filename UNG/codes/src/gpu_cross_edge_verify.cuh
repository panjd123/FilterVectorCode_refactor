// Optional CPU reference verification for the batched cross-edge GPU topK path.
//
// This diagnostic code is gated by CrossEdgeGpuRuntimeConfig::diagnostics and
// should not be part of the hot build path. Keeping it out of
// gpu_cross_groups_search_all_batched() makes the main route easier to read.

inline float cross_edge_verify_l2_cpu(ANNS::IStorage* storage, int dim, ANNS::IdxType qid, ANNS::IdxType xid) {
    const float* q = reinterpret_cast<const float*>(storage->get_vector(qid));
    const float* x = reinterpret_cast<const float*>(storage->get_vector(xid));
    float s = 0.f;
    for (int d = 0; d < dim; ++d) {
        const float diff = q[d] - x[d];
        s += diff * diff;
    }
    return s;
}

inline void cross_edge_verify_insert_global(std::vector<std::pair<float, int>>& cpu,
                                            int topk,
                                            float dist,
                                            int gid) {
    for (const auto& p : cpu) {
        if (p.second == gid) return;
    }
    if ((int)cpu.size() < topk) {
        cpu.emplace_back(dist, gid);
        std::sort(cpu.begin(), cpu.end(), [](const auto& a, const auto& b) {
            return a.first < b.first || (a.first == b.first && a.second < b.second);
        });
    } else if (dist < cpu.back().first || (dist == cpu.back().first && gid < cpu.back().second)) {
        cpu.back() = {dist, gid};
        std::sort(cpu.begin(), cpu.end(), [](const auto& a, const auto& b) {
            return a.first < b.first || (a.first == b.first && a.second < b.second);
        });
    }
}

inline void verify_cross_edge_topk_output(
    bool gpu_global_merge,
    bool verify_strict,
    int verify_samples,
    int total_queries,
    int dim,
    int topk,
    ANNS::IStorage* storage,
    const ANNS::LabelNavGraph& label_nav_graph,
    const std::vector<ANNS::IdxType>& new_vec_id_to_group_id,
    const std::vector<std::pair<ANNS::IdxType, ANNS::IdxType>>& group_id_to_range,
    const std::vector<ANNS::IdxType>& target_group_ids,
    const std::vector<ANNS::IdxType>& query_global_ids,
    const std::vector<int>& query_target_index,
    const int* h_idx,
    const float* h_dis) {
    if (verify_samples <= 0) return;

    if (gpu_global_merge) {
        const int checked = std::min(verify_samples, total_queries);
        int mismatch = 0;
        int mismatch_logged = 0;
        const int stride = std::max(1, total_queries / std::max(1, checked));

        for (int t = 0, qi = 0; t < checked && qi < total_queries; ++t, qi += stride) {
            const ANNS::IdxType qid = query_global_ids[(size_t)qi];
            if ((size_t)qid >= new_vec_id_to_group_id.size()) continue;
            const ANNS::IdxType q_group = new_vec_id_to_group_id[(size_t)qid];
            std::vector<std::pair<float, int>> cpu;
            cpu.reserve((size_t)topk);

            for (ANNS::IdxType tgt_gid : target_group_ids) {
                const auto& ins = label_nav_graph.in_neighbors[tgt_gid];
                if (std::find(ins.begin(), ins.end(), q_group) == ins.end()) continue;
                const auto& rng = group_id_to_range[(size_t)tgt_gid];
                for (ANNS::IdxType xid = rng.first; xid < rng.second; ++xid) {
                    cross_edge_verify_insert_global(
                        cpu,
                        topk,
                        cross_edge_verify_l2_cpu(storage, dim, qid, xid),
                        (int)xid);
                }
            }

            std::vector<std::pair<int, float>> gpu;
            gpu.reserve((size_t)topk);
            for (int r = 0; r < topk; ++r) {
                const int idx = h_idx[(size_t)qid * (size_t)topk + (size_t)r];
                const float dis = h_dis[(size_t)qid * (size_t)topk + (size_t)r];
                if (idx >= 0) gpu.emplace_back(idx, dis);
            }
            if ((int)gpu.size() != (int)cpu.size()) {
                if (mismatch_logged < 3) {
                    prof_logf("[PROF] cross_edges.verify_mismatch_global qid=%lld qi=%d q_group=%lld reason=size gpu=%d cpu=%d",
                              (long long)qid, qi, (long long)q_group, (int)gpu.size(), (int)cpu.size());
                    ++mismatch_logged;
                }
                ++mismatch;
                continue;
            }
            for (size_t i = 0; i < gpu.size(); ++i) {
                const float gd = gpu[i].second;
                const float cd = cpu[i].first;
                const float tol = 1e-3f * std::max(1.0f, std::fabs(cd));
                if (gpu[i].first != cpu[i].second && std::fabs(gd - cd) > tol) {
                    if (mismatch_logged < 3) {
                        prof_logf("[PROF] cross_edges.verify_mismatch_global qid=%lld qi=%d rank=%d gpu=(%d,%.6f) cpu=(%d,%.6f)",
                                  (long long)qid, qi, (int)i, gpu[i].first, gd, cpu[i].second, cd);
                        ++mismatch_logged;
                    }
                    ++mismatch;
                    break;
                }
            }
        }
        prof_logf("[PROF] cross_edges.verify_global sampled=%d mismatches=%d strict=%d",
                  checked, mismatch, verify_strict ? 1 : 0);
        if (verify_strict && mismatch > 0) {
            throw std::runtime_error("UNG_GEMM global-merge verification failed: mismatch detected.");
        }
        return;
    }

    const int checked = std::min(verify_samples, total_queries);
    int mismatch = 0;
    int mismatch_logged = 0;
    const int stride = std::max(1, total_queries / std::max(1, checked));

    for (int t = 0, qi = 0; t < checked && qi < total_queries; ++t, qi += stride) {
        const ANNS::IdxType qid = query_global_ids[(size_t)qi];
        const int target_index = query_target_index[(size_t)qi];
        const ANNS::IdxType tgt_gid = target_group_ids[(size_t)target_index];
        const auto& rng = group_id_to_range[(size_t)tgt_gid];
        const ANNS::IdxType x0 = rng.first;
        const ANNS::IdxType x1 = rng.second;
        if (x1 <= x0) continue;

        std::vector<std::pair<float, int>> cpu;
        cpu.reserve((size_t)std::min<ANNS::IdxType>((ANNS::IdxType)topk + 8, x1 - x0));
        for (ANNS::IdxType xid = x0; xid < x1; ++xid) {
            const float d = cross_edge_verify_l2_cpu(storage, dim, qid, xid);
            const int local = (int)(xid - x0);
            if ((int)cpu.size() < topk) {
                cpu.emplace_back(d, local);
                if ((int)cpu.size() == topk) {
                    std::sort(cpu.begin(), cpu.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                }
            } else if (d < cpu.back().first) {
                cpu.back() = {d, local};
                for (int i = topk - 1; i > 0 && cpu[(size_t)i].first < cpu[(size_t)i - 1].first; --i) {
                    std::swap(cpu[(size_t)i], cpu[(size_t)i - 1]);
                }
            }
        }
        if (!cpu.empty()) {
            std::sort(cpu.begin(), cpu.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        }

        std::vector<std::pair<int, float>> gpu;
        gpu.reserve((size_t)topk);
        for (int r = 0; r < topk; ++r) {
            const int idx = h_idx[(size_t)qi * (size_t)topk + (size_t)r];
            const float dis = h_dis[(size_t)qi * (size_t)topk + (size_t)r];
            if (idx >= 0) gpu.emplace_back(idx, dis);
        }
        if ((int)gpu.size() != (int)cpu.size()) {
            if (mismatch_logged < 3) {
                prof_logf("[PROF] cross_edges.verify_mismatch qid=%lld qi=%d target_gid=%lld reason=size gpu=%d cpu=%d x_size=%lld",
                          (long long)qid, qi, (long long)tgt_gid, (int)gpu.size(), (int)cpu.size(), (long long)(x1 - x0));
                ++mismatch_logged;
            }
            ++mismatch;
            continue;
        }
        for (size_t i = 0; i < gpu.size(); ++i) {
            const int gpu_idx = gpu[i].first;
            const int cpu_idx = cpu[i].second;
            const float gd = gpu[i].second;
            const float cd = cpu[i].first;
            const float tol = 1e-3f * std::max(1.0f, std::fabs(cd));
            if (std::fabs(gd - cd) > tol) {
                if (mismatch_logged < 3) {
                    prof_logf("[PROF] cross_edges.verify_mismatch qid=%lld qi=%d target_gid=%lld rank=%d gpu_idx=%d gpu_dist=%.6f cpu_idx=%d cpu_dist=%.6f x_size=%lld",
                              (long long)qid, qi, (long long)tgt_gid, (int)i, gpu_idx, gd, cpu_idx, cd, (long long)(x1 - x0));
                    for (int rr = 0; rr < std::min(topk, 6); ++rr) {
                        const int gidx = (rr < (int)gpu.size()) ? gpu[(size_t)rr].first : -1;
                        const float gdis = (rr < (int)gpu.size()) ? gpu[(size_t)rr].second : -1.f;
                        const int cidx = (rr < (int)cpu.size()) ? cpu[(size_t)rr].second : -1;
                        const float cdis = (rr < (int)cpu.size()) ? cpu[(size_t)rr].first : -1.f;
                        prof_logf("[PROF] cross_edges.verify_pair qid=%lld rank=%d gpu=(%d,%.6f) cpu=(%d,%.6f)",
                                  (long long)qid, rr, gidx, gdis, cidx, cdis);
                    }
                    ++mismatch_logged;
                }
                ++mismatch;
                break;
            }
        }
    }

    prof_logf("[PROF] cross_edges.verify sampled=%d mismatches=%d strict=%d",
              checked, mismatch, verify_strict ? 1 : 0);
    if (verify_strict && mismatch > 0) {
        throw std::runtime_error("UNG_GEMM verification failed: mismatch detected.");
    }
}
