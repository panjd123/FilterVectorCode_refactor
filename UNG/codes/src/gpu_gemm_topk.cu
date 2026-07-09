#include "uni_nav_graph.h"
#include "gpu_cross_edge_common.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cassert>
#include <fstream>
#include <mutex>
#include <cstdarg>
#include <string>
#include <cfloat>
#include <cstring>
#include <chrono>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <omp.h>
#include <unordered_map>
#include <mma.h>

#ifdef ANNS_HAVE_CUVS
// ACORN/FAISS headers pulled in by uni_nav_graph.h define a global debug(...)
// macro. RAPIDS logger has a debug() member function, so isolate cuVS includes
// from that macro pollution.
#ifdef debug
#pragma push_macro("debug")
#undef debug
#define ANNS_RESTORE_DEBUG_MACRO_AFTER_CUVS
#endif
#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#ifdef ANNS_RESTORE_DEBUG_MACRO_AFTER_CUVS
#pragma pop_macro("debug")
#undef ANNS_RESTORE_DEBUG_MACRO_AFTER_CUVS
#endif
#endif

// ========== GEMM 相关 ==========
#include <cublas_v2.h>
#include <cublasLt.h>

namespace {

// ============================================================
// 1) 计时/日志
// ============================================================
static constexpr const char* kProfLogPath =
    "/home/graphdb/Codes/FilterVectorResultsCUDA/prof/ung_prof.log";
static std::mutex g_prof_mtx;

[[maybe_unused]] inline void prof_log_line(const std::string& line) {
    std::lock_guard<std::mutex> lk(g_prof_mtx);
    std::ofstream ofs(kProfLogPath, std::ios::app);
    ofs << line << '\n';
}



[[maybe_unused]] inline void prof_logf(const char* fmt, ...) {
    char buf[1024];
    va_list ap; va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    prof_log_line(buf);
}

struct SpecialGpuInterSettings {
    unsigned long long pair_work_cap = 0;
    bool use_wmma = false;
    int warps_per_query = 2;
};

unsigned long long read_env_ull_local(const char* key, unsigned long long fallback)
{
    const char* value = std::getenv(key);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    return (end == value || *end != 0) ? fallback : parsed;
}

int read_env_int_clamped_local(const char* key, int fallback, int min_value, int max_value)
{
    const char* value = std::getenv(key);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != 0) return fallback;
    if (parsed < min_value) parsed = min_value;
    if (parsed > max_value) parsed = max_value;
    return static_cast<int>(parsed);
}

bool read_env_bool_local(const char* key, bool fallback)
{
    const char* value = std::getenv(key);
    if (!value || !*value) return fallback;
    return std::atoi(value) != 0;
}

SpecialGpuInterSettings make_special_gpu_inter_settings(int topk)
{
    SpecialGpuInterSettings settings;
    settings.pair_work_cap = read_env_ull_local("UNG_SPECIAL_INTER_MAX_PAIR_WORK", 0ull);
    settings.use_wmma = topk <= 16 && read_env_bool_local("UNG_SPECIAL_GPU_INTER_WMMA", false);
    settings.warps_per_query = read_env_int_clamped_local("UNG_SPECIAL_GPU_INTER_WARPS", 2, 1, 16);
    return settings;
}

#include "gpu_cross_edge_planning.cuh"

#include "gpu_cross_edge_buffers.cuh"

} // namespace

// Global-scope naive CUDA kernel.
// Keeping this outside anonymous namespace avoids CUDA runtime symbol lookup issues
// observed with internal-linkage kernels under this build system.
extern "C" __global__ void ung_dot_batched_naive_global_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ Xtiles,
    int nq_chunk,
    int tile,
    int dim,
    int num_tiles,
    int q_base,
    int nq_total,
    int stride_x,
    int use_shared_x,
    int use_vec4,
    int x_cols_per_block,
    float* __restrict__ dot_batches)
{
    // Warp-specialized mapping:
    // - blockDim.x 固定 32（一个 warp）
    // - blockDim.y 表示同一 CTA 内并行处理多少个 query 行
    // - 每个 warp 负责一个 (qi, j)，lane 维度并行累加 dim，再做 warp reduce
    const int lane = threadIdx.x;  // 0..31
    const int qi_local = blockIdx.y * blockDim.y + threadIdx.y;
    const int j_base = blockIdx.x * x_cols_per_block;
    const int tb = blockIdx.z;

    if (tb >= num_tiles || j_base >= tile) return;
    int cols = x_cols_per_block;
    if (j_base + cols > tile) cols = tile - j_base;

    const bool qi_valid = (qi_local < nq_chunk);
    const int qi = q_base + qi_local;
    const float* x0 = Xtiles + (size_t)tb * (size_t)stride_x + (size_t)j_base * dim;
    const float* x1 = (cols > 1) ? (x0 + (size_t)dim) : nullptr;
    const float* x2 = (cols > 2) ? (x1 + (size_t)dim) : nullptr;
    const float* x3 = (cols > 3) ? (x2 + (size_t)dim) : nullptr;

    extern __shared__ float s_x[];
    if (use_shared_x) {
        const int t = threadIdx.y * blockDim.x + lane;
        const int tstride = blockDim.x * blockDim.y;
        if (cols > 0) {
            float* sx = s_x;
            for (int d = t; d < dim; d += tstride) sx[d] = x0[d];
        }
        if (cols > 1) {
            float* sx = s_x + (size_t)dim;
            for (int d = t; d < dim; d += tstride) sx[d] = x1[d];
        }
        if (cols > 2) {
            float* sx = s_x + (size_t)2 * (size_t)dim;
            for (int d = t; d < dim; d += tstride) sx[d] = x2[d];
        }
        if (cols > 3) {
            float* sx = s_x + (size_t)3 * (size_t)dim;
            for (int d = t; d < dim; d += tstride) sx[d] = x3[d];
        }
        __syncthreads();
    }

    if (!qi_valid || qi >= nq_total) return;
    const float* q = Q + (size_t)qi * dim;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    if (use_vec4 && !use_shared_x && (dim % 4 == 0) && cols == 1) {
        const int dim4 = dim >> 2;
        const float4* q4 = reinterpret_cast<const float4*>(q);
        const float4* x4 = reinterpret_cast<const float4*>(x0);
        for (int d4 = lane; d4 < dim4; d4 += 32) {
            float4 qv = q4[d4];
            float4 xv = x4[d4];
            acc0 += qv.x * xv.x + qv.y * xv.y + qv.z * xv.z + qv.w * xv.w;
        }
    } else {
        for (int d = lane; d < dim; d += 32) {
            const float qv = q[d];
            if (cols > 0) {
                const float xv = use_shared_x ? s_x[d] : x0[d];
                acc0 += qv * xv;
            }
            if (cols > 1) {
                const float xv = use_shared_x ? s_x[(size_t)dim + d] : x1[d];
                acc1 += qv * xv;
            }
            if (cols > 2) {
                const float xv = use_shared_x ? s_x[(size_t)2 * (size_t)dim + d] : x2[d];
                acc2 += qv * xv;
            }
            if (cols > 3) {
                const float xv = use_shared_x ? s_x[(size_t)3 * (size_t)dim + d] : x3[d];
                acc3 += qv * xv;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffff, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffff, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffff, acc3, offset);
    }

    if (lane == 0) {
        const size_t base = (size_t)tb * (size_t)nq_total * (size_t)tile + (size_t)qi * (size_t)tile + (size_t)j_base;
        dot_batches[base] = acc0;
        if (cols > 1) dot_batches[base + 1] = acc1;
        if (cols > 2) dot_batches[base + 2] = acc2;
        if (cols > 3) dot_batches[base + 3] = acc3;
    }
}

extern "C" __global__ void ung_gather_q_by_id_global_kernel(
    const float* __restrict__ all_x,
    const uint32_t* __restrict__ q_ids,
    int nq,
    int dim,
    float* __restrict__ out_q)
{
    const long long total = (long long)nq * (long long)dim;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        int qi = (int)(idx / dim);
        int d = (int)(idx - (long long)qi * dim);
        uint32_t src = q_ids[qi];
        out_q[idx] = all_x[(size_t)src * (size_t)dim + (size_t)d];
    }
}

__global__ void ung_merge_local_topk_rows_to_global_kernel(
    const int* __restrict__ local_idx,
    const float* __restrict__ local_dist,
    const uint32_t* __restrict__ query_ids,
    const uint32_t* __restrict__ target_offsets,
    int total_queries,
    int topk,
    int* __restrict__ global_idx,
    float* __restrict__ global_dist,
    int* __restrict__ locks)
{
    for (int qi = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         qi < total_queries;
         qi += (int)(gridDim.x * blockDim.x)) {
        const uint32_t qid = query_ids[qi];
        int* lock = locks + qid;
        while (atomicCAS(lock, 0, 1) != 0) { }

        int* out_i = global_idx + (size_t)qid * (size_t)topk;
        float* out_d = global_dist + (size_t)qid * (size_t)topk;
        const uint32_t target_off = target_offsets[qi];

        for (int r = 0; r < topk; ++r) {
            const size_t in_pos = (size_t)qi * (size_t)topk + (size_t)r;
            const int local = local_idx[in_pos];
            if (local < 0) continue;
            const float dist = local_dist[in_pos];
            const int gid = (int)(target_off + (uint32_t)local);

            bool exists = false;
            for (int k = 0; k < topk; ++k) {
                if (out_i[k] == gid) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                const int last = topk - 1;
                const bool better_than_tail =
                    (dist < out_d[last]) || (dist == out_d[last] && gid < out_i[last]);
                if (better_than_tail) {
                    int pos = last;
                    while (pos > 0 &&
                           (dist < out_d[pos - 1] || (dist == out_d[pos - 1] && gid < out_i[pos - 1]))) {
                        out_d[pos] = out_d[pos - 1];
                        out_i[pos] = out_i[pos - 1];
                        --pos;
                    }
                    out_d[pos] = dist;
                    out_i[pos] = gid;
                }
            }
        }

        __threadfence();
        atomicExch(lock, 0);
    }
}

#include "gpu_cross_edge_topk_device.cuh"

#include "gpu_cross_edge_singleton_kernels.cuh"

#include "gpu_cross_edge_large_group_kernels.cuh"

#include "gpu_cross_edge_descriptor_kernels.cuh"

#include "gpu_cross_edge_group_fused_kernels.cuh"

#include "gpu_cross_edge_source_exact_kernels.cuh"

namespace {

#include "gpu_cross_edge_update_kernels.cuh"

#include "gpu_cross_edge_double_buffer.cuh"

#include "gpu_cross_edge_sgemm_baseline.cuh"

} // namespace

// Resident all-vector upload/cache helpers.
#include "gpu_cross_edge_resident_vectors.cuh"

#include "gpu_cross_edge_source_exact.cuh"

#include "gpu_cross_edge_cuvs_baseline.cuh"

#include "gpu_cross_edge_x_streaming.cuh"

#include "gpu_cross_edge_bucket_fused.cuh"

#include "gpu_cross_edge_output.cuh"

#include "gpu_cross_edge_profile.cuh"

#include "gpu_cross_edge_verify.cuh"

#include "gpu_cross_edge_per_group_fused.cuh"

#include "gpu_cross_edge_cpu_tiny.cuh"

#include "gpu_cross_edge_query_pack.cuh"

#include "gpu_cross_edge_bucket_descriptors.cuh"

#include "gpu_cross_edge_group_dispatch.cuh"

#include "gpu_cross_edge_regular_route.cuh"

#include "gpu_cross_edge_regular_backend.cuh"

// ============================================================
// 8) 主入口：基于 GEMM（分块流式）计算 cross-group neighbors
//
// 目标：对每个 group（target group）
//   - 把它所有 in-neighbors 组中的向量当作 Query（Q）
//   - 在 target group 的向量集合 X 中做 brute-force topK
//   - 结果写回 cross_group_neighbors[qid].insert(target_vid, dist)
//
// 关键优化点：
//   - 距离计算用 GEMM：dot = Q * X^T
//   - 分块 tile X，避免一次性 dot 矩阵太大
//   - topk<=32 的情况下，用“running topK + 每 tile 更新”很合适
// ============================================================
void ANNS::UniNavGraph::gpu_cross_groups_search_all_batched(
    const std::vector<IdxType>& target_group_ids,
    int dim,
    int topk,
    std::vector<SearchQueue>& cross_group_neighbors,
    std::vector<std::vector<IdxType>>* cross_group_neighbor_ids,
    std::vector<IdxType>* cross_group_neighbor_flat_ids,
    CrossEdgeBuildTiming* timing,
    const CrossEdgeGpuRuntimeConfig& gpu_route)
{
    using namespace std;
    if (target_group_ids.empty()) return;
    const auto t_func_start = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = [](const std::chrono::high_resolution_clock::time_point& t0,
                         const std::chrono::high_resolution_clock::time_point& t1) -> double {
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    // ------------------------------------------------------------
    // 第一步：统计每个 target group 的查询规模，并决定 CPU tiny /
    // bench filter / GPU 主路径分流。执行路径只消费 planning 结果。
    // ------------------------------------------------------------
    std::vector<uint8_t> skip_source_groups;
    const std::vector<uint8_t>* skip_source_groups_ptr = nullptr;
    if (_build_config.special_blocks_enabled && _build_config.special_block_skip_trivial) {
        skip_source_groups.assign((size_t)_num_groups + 1u, 0);
        size_t skip_group_count = 0;
        for (IdxType group_id = 1; group_id <= _num_groups; ++group_id) {
            if (is_trivial_special_block_root_group(group_id)) {
                skip_source_groups[(size_t)group_id] = 1;
                ++skip_group_count;
            }
        }
        skip_source_groups_ptr = &skip_source_groups;
        prof_logf("[PROF] cross_edges.gpu_source_skip trivial_source_groups=%zu", skip_group_count);
    }
    CrossEdgeTargetWorkload workload =
        plan_cross_edge_target_workload(target_group_ids,
                                        dim,
                                        _group_id_to_range,
                                        *_label_nav_graph,
                                        gpu_route,
                                        skip_source_groups_ptr);
    auto& target_counts = workload.gpu_query_counts;
    auto& target_group_nx = workload.group_nx;
    auto& cpu_tiny_group_indices = workload.cpu_tiny_group_indices;
    const bool cpu_tiny_groups = workload.cpu_tiny_groups;
    const int cpu_tiny_nx_max = workload.cpu_tiny_nx_max;
    const int cpu_tiny_nq_max = workload.cpu_tiny_nq_max;
    const long long cpu_tiny_ops_thresh = workload.cpu_tiny_ops_thresh;
    const size_t cpu_tiny_group_count = workload.cpu_tiny_group_count;
    const size_t cpu_tiny_query_count = workload.cpu_tiny_query_count;
    const size_t total_queries_sz = workload.total_gpu_queries;
    const size_t bench_skipped_groups = workload.bench_skipped_groups;

    const CrossEdgeBatchCapacity batch_capacity =
        make_cross_edge_batch_capacity(dim, topk, gpu_route);

    const CrossEdgeDoubleBufferRoute db_route =
        make_cross_edge_double_buffer_route(gpu_route);

    if (!(db_route.request && db_route.db_nosplit) &&
        cross_group_neighbor_ids == nullptr &&
        cross_group_neighbor_flat_ids == nullptr &&
        total_queries_sz > batch_capacity.max_flat_queries && target_group_ids.size() > 1) {
        prof_logf("[PROF] cross_edges.stream_split active_queries=%zu cap_queries=%zu q_cap_mb=%d out_cap_mb=%d groups=%zu",
                  total_queries_sz,
                  batch_capacity.max_flat_queries,
                  batch_capacity.flat_q_cap_mb,
                  batch_capacity.flat_out_cap_mb,
                  target_group_ids.size());

        std::vector<IdxType> chunk_group_ids;
        chunk_group_ids.reserve(std::min(target_group_ids.size(), (size_t)4096));
        size_t chunk_queries = 0;
        size_t chunk_count = 0;

        auto flush_chunk = [&]() {
            if (chunk_group_ids.empty()) return;
            prof_logf("[PROF] cross_edges.stream_chunk index=%zu groups=%zu approx_queries=%zu",
                      chunk_count, chunk_group_ids.size(), chunk_queries);
            gpu_cross_groups_search_all_batched(chunk_group_ids, dim, topk, cross_group_neighbors, nullptr, nullptr,
                                                timing, gpu_route);
            chunk_group_ids.clear();
            chunk_queries = 0;
            ++chunk_count;
        };

        for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
            const size_t cnt = target_counts[gi];
            if (!chunk_group_ids.empty() && chunk_queries > 0 &&
                cnt > 0 && chunk_queries + cnt > batch_capacity.max_flat_queries) {
                flush_chunk();
            }
            chunk_group_ids.push_back(target_group_ids[gi]);
            chunk_queries += cnt;
            if (cnt >= batch_capacity.max_flat_queries) {
                flush_chunk();
            }
        }
        flush_chunk();
        prof_logf("[PROF] cross_edges.stream_split_done chunks=%zu", chunk_count);
        return;
    }

    if (total_queries_sz > (size_t)std::numeric_limits<int>::max()) {
        throw std::runtime_error("GPU cross-edge batch has more than INT_MAX queries; lower UNG_GPU_FLAT_Q_CAP_MB or split the input groups.");
    }

    int total_queries = (int)total_queries_sz;
    prof_logf("[PROF] cross_edges.bench_filter min_nx=%d max_nx=%d min_work=%lld skipped_groups=%zu active_queries=%d",
              workload.bench_min_nx,
              workload.bench_max_nx,
              workload.bench_min_work,
              bench_skipped_groups,
              total_queries);

    const CrossEdgeDoubleBufferEligibility db_eligibility =
        evaluate_cross_edge_double_buffer_eligibility(db_route,
                                                      target_counts,
                                                      target_group_nx,
                                                      g_d_all_X != nullptr && g_d_all_norm != nullptr,
                                                      !cpu_tiny_group_indices.empty(),
                                                      topk,
                                                      dim);
    if (db_eligibility.can_enter) {
        const size_t unsupported_groups = db_eligibility.unsupported_groups;

        if (db_eligibility.all_groups_supported) {
            CrossEdgeDoubleBufferSlot slots[2];
            init_cross_edge_double_buffer_slot(slots[0]);
            init_cross_edge_double_buffer_slot(slots[1]);
            bool slot_active[2] = {false, false};
            const bool db_global_merge =
                db_route.global_merge_requested ||
                cross_group_neighbor_ids != nullptr ||
                cross_group_neighbor_flat_ids != nullptr;
            const int total_points_for_merge = (int)_base_storage->get_num_points();
            if (db_global_merge) {
                ensure_device_global_topk_buffers((size_t)total_points_for_merge, topk);
                ensure_host_q_buffers((size_t)total_points_for_merge, topk, dim, false);
                int threads = 256;
                const long long init_total = (long long)total_points_for_merge * (long long)topk;
                int init_grid_x = (int)((init_total + threads - 1) / threads);
                if (init_grid_x > 65535) init_grid_x = 65535;
                init_topk_kernel<<<dim3((unsigned)init_grid_x, 1u, 1u), dim3(threads, 1, 1)>>>(
                    total_points_for_merge, topk, g_d_global_idx, g_d_global_dis);
                cudaMemset(g_d_global_locks, 0, (size_t)total_points_for_merge * sizeof(int));
                cudaDeviceSynchronize();
            }
            double db_h2d_ms = 0.0, db_kernel_ms = 0.0, db_d2h_ms = 0.0, db_writeback_ms = 0.0;
            size_t db_chunks = 0, db_groups = 0, db_queries = 0, db_group_desc_count = 0, db_tile_desc_count = 0;

            auto finish_slot = [&](int si) {
                finish_cross_edge_double_buffer_slot(slots[si],
                                                     slot_active[si],
                                                     db_global_merge,
                                                     topk,
                                                     _group_id_to_range,
                                                     cross_group_neighbors,
                                                     db_h2d_ms,
                                                     db_kernel_ms,
                                                     db_d2h_ms,
                                                     db_writeback_ms);
            };

            auto enqueue_chunk = [&](int si, size_t group_begin, size_t group_end) {
                CrossEdgeDoubleBufferSlot &s = slots[si];
                CrossEdgeDoubleBufferChunkPack chunk_pack =
                    pack_cross_edge_double_buffer_chunk(s,
                                                        db_route,
                                                        target_group_ids,
                                                        target_counts,
                                                        target_group_nx,
                                                        _group_id_to_range,
                                                        *_label_nav_graph,
                                                        skip_source_groups_ptr,
                                                        group_begin,
                                                        group_end,
                                                        topk);
                const size_t qpos = chunk_pack.query_count;
                db_chunks += 1;
                db_groups += chunk_pack.group_count;
                db_queries += qpos;
                db_group_desc_count += chunk_pack.group_descs.size();
                db_tile_desc_count += chunk_pack.tile_descs.size();

                enqueue_cross_edge_double_buffer_chunk(s,
                                                       slot_active[si],
                                                       db_route,
                                                       chunk_pack,
                                                       db_global_merge,
                                                       dim,
                                                       topk);
            };

            run_cross_edge_double_buffer_chunks(target_counts,
                                                db_route.chunk_queries,
                                                finish_slot,
                                                enqueue_chunk);
            if (db_global_merge) {
                CrossEdgeTopkOutputWriter db_output_writer;
                db_output_writer.mode =
                    cross_group_neighbor_flat_ids
                        ? ANNS::CrossEdgeGpuWritebackMode::FlatId
                        : (cross_group_neighbor_ids
                               ? ANNS::CrossEdgeGpuWritebackMode::IdVector
                               : ANNS::CrossEdgeGpuWritebackMode::SearchQueue);
                db_output_writer.id_only_writeback =
                    cross_group_neighbor_ids != nullptr ||
                    cross_group_neighbor_flat_ids != nullptr;
                db_output_writer.search_queues = &cross_group_neighbors;
                db_output_writer.id_vectors = cross_group_neighbor_ids;
                db_output_writer.flat_ids = cross_group_neighbor_flat_ids;
                finalize_cross_edge_double_buffer_global_merge(total_points_for_merge,
                                                               topk,
                                                               db_output_writer,
                                                               db_d2h_ms,
                                                               db_writeback_ms);
            }
            if (timing) {
                timing->gpu_h2d_ms += db_h2d_ms;
                timing->gpu_kernel_ms += db_kernel_ms;
                timing->gpu_d2h_ms += db_d2h_ms;
            }
            std::cout << "[cross_edges] double_buffer chunks=" << db_chunks
                      << " groups=" << db_groups
                      << " queries=" << db_queries
                      << " chunk_q=" << db_route.chunk_queries
                      << " group_desc=" << db_group_desc_count
                      << " tile_desc=" << db_tile_desc_count
                      << " H2D(ms)=" << db_h2d_ms
                      << " Kernel(ms)=" << db_kernel_ms
                      << " D2H(ms)=" << db_d2h_ms
                      << " writeback(ms)=" << db_writeback_ms
                      << std::endl;
            prof_logf("[PROF] cross_edges.double_buffer chunks=%zu groups=%zu queries=%zu chunk_q=%d group_desc=%zu tile_desc=%zu h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f writeback_ms=%.3f",
                      db_chunks, db_groups, db_queries, db_route.chunk_queries, db_group_desc_count, db_tile_desc_count,
                      db_h2d_ms, db_kernel_ms, db_d2h_ms, db_writeback_ms);
            prof_logf("[PROF] cross_edges.id_vector_writeback enabled=%d double_buffer=1",
                      (cross_group_neighbor_ids && db_global_merge) ? 1 : 0);
            prof_logf("[PROF] cross_edges.flat_id_writeback enabled=%d double_buffer=1",
                      (cross_group_neighbor_flat_ids && db_global_merge) ? 1 : 0);
            prof_logf("[PROF] cross_edges.db_global_merge enabled=%d", db_global_merge ? 1 : 0);
            prof_logf("[PROF] cross_edges.stage_ms count=%.3f offset=%.3f fill_q=%.3f writeback=%.3f total_queries=%d gather_q=1 singleton_fastpath=0 singleton_q=0 cpu_tiny_groups=%zu",
                      workload.count_ms, 0.0, 0.0,
                      db_writeback_ms, total_queries, cpu_tiny_group_count);
            free_cross_edge_double_buffer_slot(slots[0]);
            free_cross_edge_double_buffer_slot(slots[1]);
            return;
        }
        if (db_route.split_unsupported &&
            cross_group_neighbor_ids == nullptr &&
            cross_group_neighbor_flat_ids == nullptr &&
            unsupported_groups > 0 &&
            unsupported_groups < target_group_ids.size()) {
            std::vector<IdxType> db_group_ids;
            std::vector<IdxType> fallback_group_ids;
            db_group_ids.reserve(target_group_ids.size() - unsupported_groups);
            fallback_group_ids.reserve(unsupported_groups);
            for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
                if (target_counts[gi] == 0) {
                    continue;
                }
                const int nx = target_group_nx[gi];
                if (nx > 0 && nx <= db_route.large_max_nx) {
                    db_group_ids.push_back(target_group_ids[gi]);
                } else {
                    fallback_group_ids.push_back(target_group_ids[gi]);
                }
            }
            prof_logf("[PROF] cross_edges.double_buffer_split supported_groups=%zu fallback_groups=%zu max_supported_nx=%d",
                      db_group_ids.size(), fallback_group_ids.size(), db_route.large_max_nx);
            std::cout << "[cross_edges] double_buffer split supported_groups=" << db_group_ids.size()
                      << " fallback_groups=" << fallback_group_ids.size()
                      << " max_supported_nx=" << db_route.large_max_nx << std::endl;
            if (!db_group_ids.empty()) {
                gpu_cross_groups_search_all_batched(db_group_ids, dim, topk, cross_group_neighbors, nullptr, nullptr,
                                                    timing, gpu_route);
            }
            if (!fallback_group_ids.empty()) {
                gpu_cross_groups_search_all_batched(fallback_group_ids, dim, topk, cross_group_neighbors, nullptr, nullptr,
                                                    timing, gpu_route);
            }
            return;
        }
        prof_logf("[PROF] cross_edges.double_buffer_fallback unsupported_groups=%zu max_supported_nx=%d",
                  unsupported_groups, db_route.large_max_nx);
    }

    // ------------------------------------------------------------
    // 第二步：为每个 target group 分配 Q 的连续区间（prefix offsets）
    // target_offsets[gi]：这个 group 的 Q 在 flatten 后数组中的起点
    // ------------------------------------------------------------
    vector<size_t> target_offsets(target_group_ids.size(), 0);
    size_t acc = 0;
    const auto t_offset_start = std::chrono::high_resolution_clock::now();
    for (size_t gi = 0; gi < target_group_ids.size(); ++gi) {
        target_offsets[gi] = acc;
        acc += target_counts[gi];
    }
    const auto t_offset_end = std::chrono::high_resolution_clock::now();

    // 轻量组 CPU 回退：用于把超小工作量组从 GPU 路径分流
    double cpu_tiny_ms = 0.0;
    auto process_cpu_tiny_groups = [&]() {
        if (!cpu_tiny_groups || cpu_tiny_group_indices.empty()) return;
        cpu_tiny_ms = run_cross_edge_cpu_tiny_groups(cpu_tiny_group_indices,
                                                     target_group_ids,
                                                     target_group_nx,
                                                     dim,
                                                     topk,
                                                     _base_storage.get(),
                                                     _group_id_to_range,
                                                     *_label_nav_graph,
                                                     cross_group_neighbors);
    };

    if (total_queries == 0) {
        process_cpu_tiny_groups();
        prof_logf("[PROF] cross_edges.cpu_tiny groups=%zu queries=%zu ms=%.3f enabled=%d nx_max=%d nq_max=%d ops_thresh=%lld",
                  cpu_tiny_group_count, cpu_tiny_query_count, cpu_tiny_ms, cpu_tiny_groups ? 1 : 0,
                  cpu_tiny_nx_max, cpu_tiny_nq_max, cpu_tiny_ops_thresh);
        return;
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    CrossEdgeRegularRouteConfig regular_route_cfg = make_cross_edge_regular_route_config(
        gpu_route,
        dim,
        topk,
        cross_group_neighbor_ids != nullptr,
        cross_group_neighbor_flat_ids != nullptr);

    const bool use_device_query_gather = regular_route_cfg.use_device_query_gather;
    const bool singleton_fastpath = regular_route_cfg.singleton_fastpath;
    const bool id_only_writeback_requested = regular_route_cfg.id_only_writeback_requested;
    const bool gpu_global_merge = regular_route_cfg.gpu_global_merge;
    const bool id_only_writeback = regular_route_cfg.id_only_writeback;
    const int gather_threads = regular_route_cfg.gather_threads;
    const int gather_blocks = regular_route_cfg.gather_blocks;
    const bool use_naive_cuda = regular_route_cfg.use_naive_cuda;
    const bool use_cublas_lt = regular_route_cfg.use_cublas_lt;
    const bool naive_heavy_sgemm = regular_route_cfg.naive_heavy_sgemm;

    if ((cross_group_neighbor_ids != nullptr || cross_group_neighbor_flat_ids != nullptr) && !gpu_global_merge) {
        prof_logf("[PROF] cross_edges.id_vector_writeback disabled_for_host_merge requested=1");
    }
    if (id_only_writeback_requested && !gpu_global_merge) {
        prof_logf("[PROF] cross_edges.id_only_writeback disabled_for_host_merge requested=1");
    }
    const CrossEdgeSingletonPrepass singleton_prepass =
        collect_cross_edge_singleton_prepass(singleton_fastpath,
                                             target_group_ids,
                                             target_counts,
                                             target_group_nx,
                                             target_offsets,
                                             _group_id_to_range,
                                             total_queries);
    const size_t singleton_query_count = singleton_prepass.size();

    const bool direct_qid_all_effective = compute_cross_edge_direct_qid_all_effective(
        regular_route_cfg,
        target_group_ids,
        target_counts,
        target_group_nx,
        singleton_query_count,
        topk);

    // ------------------------------------------------------------
    // 第三步：确保缓存足够大（Q/idx/dist + qnorm）
    // ------------------------------------------------------------
    const size_t host_out_rows = gpu_global_merge
        ? std::max<size_t>((size_t)total_queries, (size_t)_base_storage->get_num_points())
        : (size_t)total_queries;
    ensure_cross_edge_regular_backend_buffers(host_out_rows,
                                              total_queries,
                                              (int)_base_storage->get_num_points(),
                                              topk,
                                              dim,
                                              use_device_query_gather,
                                              direct_qid_all_effective,
                                              gpu_global_merge,
                                              singleton_query_count);

    // ------------------------------------------------------------
    // 第四/五/六步：建立写回映射，填充/上传 Q 或 qid。
    // 细节收在 helper 中，主函数只消费映射和 timing。
    // ------------------------------------------------------------
    CrossEdgeQueryUploadResult query_upload = pack_and_upload_cross_edge_queries(
        stream,
        target_group_ids,
        target_counts,
        target_offsets,
        total_queries,
        dim,
        use_device_query_gather,
        gpu_global_merge,
        direct_qid_all_effective,
        gather_blocks,
        gather_threads,
        _base_storage.get(),
        _group_id_to_range,
        *_label_nav_graph,
        skip_source_groups_ptr);
    auto& query_global_ids = query_upload.query_global_ids;
    auto& query_target_index = query_upload.query_target_index;
    if (timing) timing->gpu_h2d_ms += query_upload.h2d_ms;

    // ------------------------------------------------------------
    // 第七步：GPU 计算阶段计时（norm + GEMM + topK 更新）
    // ------------------------------------------------------------
    cudaEvent_t k0,k1;
    cudaEventCreate(&k0); cudaEventCreate(&k1);
    cudaEventRecord(k0, stream);

    CrossEdgeSeparateTopkConfig separate_topk_cfg =
        make_cross_edge_separate_topk_config(regular_route_cfg);
    log_cross_edge_regular_route_config(regular_route_cfg);

    if (!use_naive_cuda || naive_heavy_sgemm) {
        // cuBLAS 路径，或 naive 路径里的重负载分流都需要句柄
        ensure_cublas();
        cublasSetStream(g_cublas, stream);
    }
    if (use_cublas_lt) {
        ensure_lt_workspace(separate_topk_cfg.lt_workspace_bytes);
    }
    CrossEdgeRegularDispatchCounters dispatch_counters;
    // qnorm 与 topk 初始化都只和 query 有关，提前全量做一次，避免每个 group 启动一次 kernel
    initialize_cross_edge_regular_topk_state(stream,
                                             total_queries,
                                             (int)_base_storage->get_num_points(),
                                             dim,
                                             topk,
                                             direct_qid_all_effective,
                                             gpu_global_merge);
    launch_cross_edge_singleton_prepass(stream,
                                        singleton_prepass,
                                        dim,
                                        topk,
                                        regular_route_cfg,
                                        dispatch_counters);

    CrossEdgeBucketDescriptors bucket_descs = build_cross_edge_regular_bucket_descriptors(
        target_group_ids,
        target_counts,
        target_group_nx,
        target_offsets,
        _group_id_to_range,
        dim,
        topk,
        regular_route_cfg);
    auto& bucket_group_kind = bucket_descs.group_kind;

    CrossEdgeRegularWorkloadView regular_workload;
    regular_workload.target_group_ids = &target_group_ids;
    regular_workload.target_counts = &target_counts;
    regular_workload.target_group_nx = &target_group_nx;
    regular_workload.target_offsets = &target_offsets;
    regular_workload.group_id_to_range = &_group_id_to_range;
    regular_workload.bucket_group_kind = &bucket_group_kind;

    CrossEdgeRegularDispatchConfig dispatch_cfg =
        make_cross_edge_regular_dispatch_config(regular_route_cfg, dim, topk);

    CrossEdgeRegularDeviceBuffers dispatch_buffers;
    dispatch_buffers.d_Q = g_d_Q;
    dispatch_buffers.d_q_norm = g_d_q_norm;
    dispatch_buffers.d_q_ids = g_d_q_ids;
    dispatch_buffers.d_all_X = g_d_all_X;
    dispatch_buffers.d_all_norm = g_d_all_norm;
    dispatch_buffers.d_idx = g_d_idx;
    dispatch_buffers.d_dis = g_d_dis;
    dispatch_buffers.d_global_idx = g_d_global_idx;
    dispatch_buffers.d_global_dis = g_d_global_dis;
    dispatch_buffers.d_global_locks = g_d_global_locks;
    dispatch_buffers.h_Q = g_h_Q;
    dispatch_buffers.h_all_X = g_h_all_X;

    dispatch_cross_edge_regular_groups(
        stream,
        regular_workload,
        separate_topk_cfg,
        dispatch_cfg,
        dispatch_buffers,
        dispatch_counters);

    launch_cross_edge_regular_bucket_fused_groups(
        stream,
        dim,
        topk,
        regular_route_cfg,
        bucket_descs);

    finalize_cross_edge_regular_global_merge(stream,
                                             gpu_global_merge,
                                             dispatch_counters.global_merge_direct_query_count,
                                             total_queries,
                                             topk);

    // 与 GPU kernel 队列并行执行 CPU tiny 组，尽量把 CPU 时间隐藏在 GPU 计算后半段
    process_cpu_tiny_groups();

    cudaEventRecord(k1, stream);
    cudaError_t ksync_err = cudaEventSynchronize(k1);
    if (ksync_err != cudaSuccess) {
        prof_logf("[ERROR] cross_edges kernel stream failed before D2H: %s", cudaGetErrorString(ksync_err));
        throw std::runtime_error("cross_edges GPU kernel stream failed.");
    }

    float tker = 0.f;
    cudaEventElapsedTime(&tker, k0, k1);
    if (timing) timing->gpu_kernel_ms += tker;

    cudaEventDestroy(k0); cudaEventDestroy(k1);

    // ------------------------------------------------------------
    // 第八/九步：D2H + host writeback。
    // 输出策略统一收在 helper 中，主函数只保留 route 和 profiling。
    // ------------------------------------------------------------
    const CrossEdgeGpuDiagnosticsConfig& diagnostics = gpu_route.diagnostics;
    const bool count_valid_pairs = diagnostics.count_valid_pairs;
    CrossEdgeTopkOutputWriter output_writer;
    output_writer.mode =
        regular_route_cfg.flat_id_writeback
            ? ANNS::CrossEdgeGpuWritebackMode::FlatId
            : (regular_route_cfg.id_vector_writeback
                   ? ANNS::CrossEdgeGpuWritebackMode::IdVector
                   : ANNS::CrossEdgeGpuWritebackMode::SearchQueue);
    output_writer.id_only_writeback = id_only_writeback;
    output_writer.search_queues = &cross_group_neighbors;
    output_writer.id_vectors = cross_group_neighbor_ids;
    output_writer.flat_ids = cross_group_neighbor_flat_ids;
    const CrossEdgeTopkOutputFinalizeResult output_result = finalize_cross_edge_topk_output(
        stream,
        gpu_global_merge,
        output_writer,
        count_valid_pairs,
        total_queries,
        (int)_base_storage->get_num_points(),
        topk,
        g_d_idx,
        g_d_dis,
        g_d_global_idx,
        g_d_global_dis,
        g_h_idx,
        g_h_dis,
        query_global_ids,
        query_target_index,
        target_group_ids,
        _group_id_to_range);
    if (timing) timing->gpu_d2h_ms += output_result.d2h_ms;
    log_cross_edge_regular_backend_profile(regular_route_cfg,
                                           dispatch_counters,
                                           bucket_descs,
                                           output_result,
                                           direct_qid_all_effective,
                                           gpu_global_merge,
                                           singleton_query_count,
                                           cpu_tiny_group_count,
                                           cpu_tiny_query_count,
                                           cpu_tiny_ms,
                                           cpu_tiny_groups,
                                           cpu_tiny_nx_max,
                                           cpu_tiny_nq_max,
                                           cpu_tiny_ops_thresh,
                                           workload.count_ms,
                                           elapsed_ms(t_offset_start, t_offset_end),
                                           query_upload.fill_ms,
                                           total_queries);

    const int verify_samples = diagnostics.verify_samples;
    if (verify_samples > 0) {
        verify_cross_edge_topk_output(
            gpu_global_merge,
            diagnostics.verify_strict,
            verify_samples,
            total_queries,
            dim,
            topk,
            _base_storage.get(),
            *_label_nav_graph,
            _new_vec_id_to_group_id,
            _group_id_to_range,
            target_group_ids,
            query_global_ids,
            query_target_index,
            g_h_idx,
            g_h_dis);
    }
}

bool ANNS::UniNavGraph::gpu_build_special_inter_edges(
    const std::vector<SpecialBlock>& special_blocks,
    const std::vector<std::vector<IdxType>>& block_points,
    std::vector<std::vector<SpecialEdge>>& special_edges_by_point,
    IdxType& inter_edge_count,
    double& gpu_ms)
{
    const int dim = _base_storage->get_dim();
    const int topk = static_cast<int>(_num_cross_edges);
    if (special_blocks.empty() || topk <= 0 || dim <= 0) return true;
    if (topk > 32) return false;

    auto elapsed_ms = [](const std::chrono::high_resolution_clock::time_point& t0,
                         const std::chrono::high_resolution_clock::time_point& t1) -> double {
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    const auto total_start = std::chrono::high_resolution_clock::now();
    const SpecialGpuInterSettings inter_settings = make_special_gpu_inter_settings(topk);
    const unsigned long long inter_pair_work_cap = inter_settings.pair_work_cap;
    size_t cap_skipped_pairs = 0;
    unsigned long long cap_skipped_queries = 0;
    unsigned long long cap_skipped_work = 0;
    CrossEdgeGpuRuntimeConfig gpu_route = make_cross_edge_gpu_runtime_config(_build_config, true);
    CrossEdgeBuildTiming timing;
    gpu_prepare_all_vectors_for_cross_edge(timing, gpu_route);

    std::vector<UngTargetSegmentDesc> h_segments;
    std::vector<UngSourceQueryDesc> h_queries;
    std::vector<IdxType> query_source_points;
    std::vector<IdxType> query_parent_blocks;
    h_segments.reserve(special_blocks.size() * 2);
    h_queries.reserve(_num_points);
    query_source_points.reserve(_num_points);
    query_parent_blocks.reserve(_num_points);

    const auto pack_start = std::chrono::high_resolution_clock::now();
    for (const SpecialBlock& block : special_blocks) {
        if (block.block_id >= block_points.size()) continue;
        const auto& src_points = block_points[block.block_id];
        if (src_points.empty() || block.child_block_ids.empty()) continue;

        for (IdxType child_block_id : block.child_block_ids) {
            if (child_block_id == 0 || child_block_id > special_blocks.size()) continue;
            const auto& dst_points = block_points[child_block_id];
            const unsigned long long pair_work =
                static_cast<unsigned long long>(src_points.size()) *
                static_cast<unsigned long long>(dst_points.size());
            if (inter_pair_work_cap > 0 && pair_work > inter_pair_work_cap) {
                cap_skipped_pairs += 1;
                cap_skipped_queries += static_cast<unsigned long long>(src_points.size());
                cap_skipped_work += pair_work;
                continue;
            }
            const SpecialBlock& child = special_blocks[child_block_id - 1];
            const uint32_t edge_start = static_cast<uint32_t>(h_segments.size());
            uint32_t edge_count = 0;
            for (IdxType group_id : child.member_group_ids) {
                if (group_id >= _group_id_to_range.size()) continue;
                const auto& range = _group_id_to_range[group_id];
                if (range.second <= range.first) continue;
                h_segments.push_back({static_cast<uint32_t>(range.first),
                                      static_cast<uint32_t>(range.second - range.first)});
                ++edge_count;
            }
            if (edge_count == 0) {
                h_segments.resize(edge_start);
                continue;
            }
            for (IdxType source : src_points) {
                h_queries.push_back({static_cast<uint32_t>(source), edge_start, edge_count});
                query_source_points.push_back(source);
                query_parent_blocks.push_back(block.block_id);
            }
        }
    }
    const double pack_ms = elapsed_ms(pack_start, std::chrono::high_resolution_clock::now());
    const bool use_wmma_inter = inter_settings.use_wmma;
    std::vector<UngSourceTileDesc> h_tiles;
    if (use_wmma_inter) {
        h_tiles.reserve((h_queries.size() + 15u) / 16u);
        size_t row = 0;
        while (row < h_queries.size()) {
            const UngSourceQueryDesc first = h_queries[row];
            size_t count = 1;
            while (count < 16u && row + count < h_queries.size()) {
                const UngSourceQueryDesc cur = h_queries[row + count];
                if (cur.edge_start != first.edge_start || cur.edge_count != first.edge_count) break;
                if (cur.qid != first.qid + static_cast<uint32_t>(count)) break;
                ++count;
            }
            h_tiles.push_back({first.qid,
                               static_cast<uint32_t>(count),
                               static_cast<uint32_t>(row),
                               first.edge_start,
                               first.edge_count});
            row += count;
        }
    }
    if (inter_pair_work_cap > 0) {
        std::cout << "[special_edges][gpu_inter_pair_work_cap] max_pair_work=" << inter_pair_work_cap
                  << " skipped_pairs=" << cap_skipped_pairs
                  << " skipped_queries=" << cap_skipped_queries
                  << " skipped_pair_work=" << cap_skipped_work
                  << std::endl;
    }
    if (h_queries.empty()) {
        gpu_ms = elapsed_ms(total_start, std::chrono::high_resolution_clock::now());
        return true;
    }

    UngSourceQueryDesc* d_queries = nullptr;
    UngSourceTileDesc* d_tiles = nullptr;
    UngTargetSegmentDesc* d_segments = nullptr;
    int* d_idx = nullptr;
    float* d_dis = nullptr;
    int* h_idx = nullptr;
    float* h_dis = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t e_h0 = nullptr, e_h1 = nullptr, e_k0 = nullptr, e_k1 = nullptr, e_d0 = nullptr, e_d1 = nullptr;

    const size_t result_items = h_queries.size() * static_cast<size_t>(topk);
    cudaMalloc(&d_queries, h_queries.size() * sizeof(UngSourceQueryDesc));
    if (use_wmma_inter) cudaMalloc(&d_tiles, h_tiles.size() * sizeof(UngSourceTileDesc));
    cudaMalloc(&d_segments, h_segments.size() * sizeof(UngTargetSegmentDesc));
    cudaMalloc(&d_idx, result_items * sizeof(int));
    cudaMalloc(&d_dis, result_items * sizeof(float));
    cudaHostAlloc(&h_idx, result_items * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&h_dis, result_items * sizeof(float), cudaHostAllocDefault);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaEventCreate(&e_h0);
    cudaEventCreate(&e_h1);
    cudaEventCreate(&e_k0);
    cudaEventCreate(&e_k1);
    cudaEventCreate(&e_d0);
    cudaEventCreate(&e_d1);

    cudaEventRecord(e_h0, stream);
    cudaMemcpyAsync(d_queries, h_queries.data(), h_queries.size() * sizeof(UngSourceQueryDesc),
                    cudaMemcpyHostToDevice, stream);
    if (use_wmma_inter) {
        cudaMemcpyAsync(d_tiles, h_tiles.data(), h_tiles.size() * sizeof(UngSourceTileDesc),
                        cudaMemcpyHostToDevice, stream);
    }
    cudaMemcpyAsync(d_segments, h_segments.data(), h_segments.size() * sizeof(UngTargetSegmentDesc),
                    cudaMemcpyHostToDevice, stream);
    cudaEventRecord(e_h1, stream);

    const int warps = inter_settings.warps_per_query;
    const int threads = warps * 32;
    cudaEventRecord(e_k0, stream);
    if (use_wmma_inter) {
        const size_t wmma_smem =
            ((size_t)8 * 16 + (size_t)warps * ((size_t)16 * 8 + (size_t)16 * 16)) *
            sizeof(float);
        const int ntile = static_cast<int>(h_tiles.size());
        const int grid_x = (ntile + warps - 1) / warps;
        ung_source_tf32_wmma_topk_global_kernel<<<dim3(static_cast<unsigned>(grid_x), 1u, 1u),
                                                   dim3(static_cast<unsigned>(threads), 1u, 1u),
                                                   wmma_smem, stream>>>(
            d_tiles,
            ntile,
            d_segments,
            g_d_all_X,
            g_d_all_norm,
            dim,
            topk,
            d_idx,
            d_dis);
    } else {
        const size_t smem = static_cast<size_t>(warps) * static_cast<size_t>(topk) *
                            (sizeof(float) + sizeof(int));
        ung_source_exact_topk_global_kernel<<<dim3(static_cast<unsigned>(h_queries.size()), 1u, 1u),
                                              dim3(static_cast<unsigned>(threads), 1u, 1u),
                                              smem, stream>>>(
            d_queries,
            static_cast<int>(h_queries.size()),
            d_segments,
            g_d_all_X,
            g_d_all_norm,
            dim,
            topk,
            d_idx,
            d_dis);
    }
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        cudaFree(d_queries); if (d_tiles) cudaFree(d_tiles); cudaFree(d_segments); cudaFree(d_idx); cudaFree(d_dis);
        cudaFreeHost(h_idx); cudaFreeHost(h_dis);
        cudaEventDestroy(e_h0); cudaEventDestroy(e_h1); cudaEventDestroy(e_k0); cudaEventDestroy(e_k1);
        cudaEventDestroy(e_d0); cudaEventDestroy(e_d1); cudaStreamDestroy(stream);
        return false;
    }
    cudaEventRecord(e_k1, stream);

    cudaEventRecord(e_d0, stream);
    cudaMemcpyAsync(h_idx, d_idx, result_items * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_dis, d_dis, result_items * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(e_d1, stream);
    cudaError_t sync_err = cudaEventSynchronize(e_d1);
    if (sync_err != cudaSuccess) {
        cudaFree(d_queries); if (d_tiles) cudaFree(d_tiles); cudaFree(d_segments); cudaFree(d_idx); cudaFree(d_dis);
        cudaFreeHost(h_idx); cudaFreeHost(h_dis);
        cudaEventDestroy(e_h0); cudaEventDestroy(e_h1); cudaEventDestroy(e_k0); cudaEventDestroy(e_k1);
        cudaEventDestroy(e_d0); cudaEventDestroy(e_d1); cudaStreamDestroy(stream);
        return false;
    }

    float h2d_ms = 0.f, kernel_ms = 0.f, d2h_ms = 0.f;
    cudaEventElapsedTime(&h2d_ms, e_h0, e_h1);
    cudaEventElapsedTime(&kernel_ms, e_k0, e_k1);
    cudaEventElapsedTime(&d2h_ms, e_d0, e_d1);

    const auto write_start = std::chrono::high_resolution_clock::now();
    for (size_t qi = 0; qi < h_queries.size(); ++qi) {
        const IdxType source = query_source_points[qi];
        const IdxType parent_block = query_parent_blocks[qi];
        auto& special_edges = special_edges_by_point[source];
        const size_t base = qi * static_cast<size_t>(topk);
        for (int k = 0; k < topk; ++k) {
            const int gid = h_idx[base + static_cast<size_t>(k)];
            if (gid >= 0) {
                special_edges.push_back({static_cast<IdxType>(gid), parent_block, SpecialEdgeKind::InterBlock});
                inter_edge_count += 1;
            }
        }
    }
    const double write_ms = elapsed_ms(write_start, std::chrono::high_resolution_clock::now());

    cudaFree(d_queries);
    if (d_tiles) cudaFree(d_tiles);
    cudaFree(d_segments);
    cudaFree(d_idx);
    cudaFree(d_dis);
    cudaFreeHost(h_idx);
    cudaFreeHost(h_dis);
    cudaEventDestroy(e_h0);
    cudaEventDestroy(e_h1);
    cudaEventDestroy(e_k0);
    cudaEventDestroy(e_k1);
    cudaEventDestroy(e_d0);
    cudaEventDestroy(e_d1);
    cudaStreamDestroy(stream);

    gpu_ms = elapsed_ms(total_start, std::chrono::high_resolution_clock::now());
    std::cout << "[special_edges][gpu_inter] queries=" << h_queries.size()
              << " segments=" << h_segments.size()
              << " mode=" << (use_wmma_inter ? "tf32_wmma" : "cuda_core")
              << " tiles=" << h_tiles.size()
              << " pack_ms=" << pack_ms
              << " h2d_ms=" << h2d_ms
              << " kernel_ms=" << kernel_ms
              << " d2h_ms=" << d2h_ms
              << " write_ms=" << write_ms
              << " total_ms=" << gpu_ms
              << std::endl;
    return true;
}
