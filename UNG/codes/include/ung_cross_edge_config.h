#ifndef ANNS_UNG_CROSS_EDGE_CONFIG_H
#define ANNS_UNG_CROSS_EDGE_CONFIG_H

#include "ung_build_config.h"

#include <string>

namespace ANNS
{

enum class CrossEdgeBackend : int
{
   CPU = 0,
   GPU = 1,
};

enum class CrossEdgeGpuWritebackMode : int
{
   SearchQueue = 0,
   IdVector = 1,
   FlatId = 2,
};

const char *cross_edge_gpu_writeback_mode_name(CrossEdgeGpuWritebackMode mode);

struct CrossEdgeVectorUploadConfig
{
   bool direct_hostreg = true;
   bool direct_pageable = false;
   int hostreg_max_mb = 4096;
};

struct CrossEdgeGpuDiagnosticsConfig
{
   bool count_valid_pairs = false;
   int verify_samples = 0;
   bool verify_strict = false;
};

struct CrossEdgeGpuDebugConfig
{
   bool medium_group_sync_debug = false;
   bool naive_debug_dot = false;
};

struct CrossEdgeBenchmarkFilterConfig
{
   int min_nx = 0;
   int max_nx = 1048576;
   long long min_work = 0;
};

struct CrossEdgeGpuRuntimeConfig
{
   bool source_exact = false;
   bool source_exact_pad_groups = true;
   bool universal_route = false;
   bool db_nosplit = false;
   bool double_buffer = true;
   bool x_streaming = false;
   bool query_upload_device_gather = true;
   bool global_merge = false;
   bool global_merge_direct = true;
   bool db_global_merge = false;
   bool db_split_unsupported = true;
   bool cpu_tiny_groups = false;
   bool singleton_fastpath = true;
   bool direct_qid_fused = false;
   bool direct_qid_all_fused = false;
   bool id_only_writeback_requested = false;
   bool force_custom_kernel = true;
   bool naive_shared_x = false;
   bool naive_vec4 = true;
   bool naive_heavy_sgemm = true;
   bool naive_group_fused = false;
   bool tf32_group_prefix = false;
   bool tf32_group_2d = false;
   bool bucket_group_fused = false;
   bool small_group_fused = true;
   bool medium_group_fused = true;
   bool large_group_fused = false;
   bool id_vector_writeback = false;
   bool flat_id_writeback = false;
   bool skip_searchqueue_storage = false;
   bool auto_x_streaming = true;
   int cpu_tiny_nx_max = 8;
   int cpu_tiny_nq_max = 64;
   int cpu_tiny_ops = 65536;
   int source_exact_warps = 8;
   int source_exact_mode = 0;
   int gemm_impl_request = -1;
   int naive_warps_per_block = 4;
   int naive_x_cols_per_block = 1;
   int naive_shared_kb = 48;
   int naive_heavy_nx = 256;
   int naive_heavy_nq = 256;
   int naive_heavy_work_m = 1;
   int topk_block_threads = -1;
   int naive_group_threads = 128;
   int singleton_threads = 256;
   int small_group_max_nx = 8;
   int small_group_warps = 8;
   int medium_group_max_nx = 4096;
   int medium_group_warps = 8;
   int large_group_min_nx = 128;
   int large_group_max_nx = 1023;
   int large_group_warps = 2;
   int large_group_fused_mode = 2;
   int global_merge_direct_max_nx = 256;
   int gather_threads = 256;
   int gather_blocks = 4096;
   int flat_q_cap_mb = 4096;
   int flat_out_cap_mb = 4096;
   int db_chunk_queries = 262144;
   int db_medium_max_nx = 128;
   int db_large_max_nx = 1023;
   int db_medium_group_warps = 8;
   int db_large_group_warps = 2;
   int x_stream_x_chunk_mb = 8192;
   int x_stream_q_chunk_mb = 2048;
   int x_stream_out_chunk_mb = 2048;
   CrossEdgeVectorUploadConfig vector_upload;
   CrossEdgeGpuDiagnosticsConfig diagnostics;
   CrossEdgeGpuDebugConfig debug;
   CrossEdgeBenchmarkFilterConfig benchmark_filter;

   bool needs_searchqueue_storage() const;
   CrossEdgeGpuWritebackMode writeback_mode() const;
   bool uses_id_vector_writeback() const;
   bool uses_flat_id_writeback() const;
   bool requires_searchqueue_writeback() const;
   const char *writeback_mode_name() const;
   const char *route_name() const;
   const char *route_class_name() const;
   std::string validation_error() const;
   std::string summary() const;
};

void apply_cross_edge_gpu_topk_env_defaults(UngGpuTopkImpl impl);

CrossEdgeGpuRuntimeConfig make_cross_edge_gpu_runtime_config(const UngBuildConfig &build_config,
                                                             bool backend_is_gpu);

} // namespace ANNS

#endif // ANNS_UNG_CROSS_EDGE_CONFIG_H
