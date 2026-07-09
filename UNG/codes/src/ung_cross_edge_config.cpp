#include "include/ung_cross_edge_config.h"

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <sstream>

namespace ANNS
{
namespace
{

int env_int(const char *key, int fallback, int min_v, int max_v)
{
   const char *s = std::getenv(key);
   if (!s || !*s)
      return fallback;
   char *end = nullptr;
   long v = std::strtol(s, &end, 10);
   if (end == s || *end != '\0')
      return fallback;
   v = std::max<long>(min_v, std::min<long>(max_v, v));
   return static_cast<int>(v);
}

int env_int_optional(const char *key, int missing, int min_v, int max_v)
{
   const char *s = std::getenv(key);
   if (!s || !*s)
      return missing;
   char *end = nullptr;
   long v = std::strtol(s, &end, 10);
   if (end == s || *end != '\0')
      return missing;
   v = std::max<long>(min_v, std::min<long>(max_v, v));
   return static_cast<int>(v);
}

struct EnvDefault
{
   const char *key;
   const char *value;
};

void set_env_defaults(std::initializer_list<EnvDefault> defaults)
{
   for (const auto &item : defaults)
   {
      if (!std::getenv(item.key))
         setenv(item.key, item.value, 0);
   }
}

bool is_gpu_batched_cross_edge(const UngBuildConfig &build_config)
{
   return build_config.cross_edge_impl == UngCrossEdgeImpl::GpuBatched;
}

void read_cross_edge_route_flags(CrossEdgeGpuRuntimeConfig &cfg,
                                 const UngBuildConfig &build_config,
                                 bool backend_is_gpu)
{
   const bool gpu_batched = is_gpu_batched_cross_edge(build_config);
   cfg.source_exact = backend_is_gpu &&
                      env_int("UNG_GPU_SOURCE_EXACT", 0, 0, 1) == 1;
   cfg.source_exact_pad_groups = env_int("UNG_GPU_SOURCE_EXACT_PAD_GROUPS", 1, 0, 1) == 1;
   cfg.universal_route = gpu_batched &&
                         env_int("UNG_UNIVERSAL_GPU", 0, 0, 1) == 1;
   cfg.db_nosplit = env_int("UNG_GPU_DB_NOSPLIT", cfg.universal_route ? 1 : 0, 0, 1) == 1;
   cfg.double_buffer = env_int("UNG_GPU_DOUBLE_BUFFER", 1, 0, 1) == 1;
   cfg.x_streaming = gpu_batched &&
                     env_int("UNG_GPU_X_STREAMING", 0, 0, 1) == 1;
   cfg.db_global_merge = env_int("UNG_GPU_DB_GLOBAL_MERGE", 0, 0, 1) == 1;
   cfg.db_split_unsupported = env_int("UNG_GPU_DB_SPLIT_UNSUPPORTED", 1, 0, 1) == 1;
   cfg.auto_x_streaming = env_int("UNG_GPU_X_STREAMING_AUTO", 1, 0, 1) == 1;
}

void read_cross_edge_output_flags(CrossEdgeGpuRuntimeConfig &cfg)
{
   cfg.query_upload_device_gather = env_int("UNG_Q_UPLOAD_MODE", 1, 0, 1) == 1;
   cfg.global_merge = env_int("UNG_GPU_GLOBAL_MERGE", 0, 0, 1) == 1;
   cfg.global_merge_direct = env_int("UNG_GPU_GLOBAL_MERGE_DIRECT", 1, 0, 1) == 1;
   cfg.direct_qid_fused = env_int("UNG_DIRECT_QID_FUSED", 0, 0, 1) == 1;
   cfg.direct_qid_all_fused = env_int("UNG_DIRECT_QID_ALL_FUSED", 0, 0, 1) == 1;
   cfg.id_only_writeback_requested = env_int("UNG_GPU_ID_ONLY_WRITEBACK", 0, 0, 1) == 1;
   cfg.global_merge_direct_max_nx =
       env_int("UNG_GPU_GLOBAL_MERGE_DIRECT_MAX_NX", 256, 1, 1048576);
   cfg.gather_threads = env_int("UNG_GATHER_THREADS", 256, 64, 512);
   cfg.gather_blocks = env_int("UNG_GATHER_BLOCKS", 4096, 1, 65535);
}

void read_cross_edge_kernel_flags(CrossEdgeGpuRuntimeConfig &cfg)
{
   cfg.cpu_tiny_groups = env_int("UNG_CPU_TINY_GROUPS", 0, 0, 1) == 1;
   cfg.singleton_fastpath = env_int("UNG_SINGLETON_FASTPATH", 1, 0, 1) == 1;
   cfg.force_custom_kernel = env_int("UNG_FORCE_CUSTOM_KERNEL", 1, 0, 1) == 1;
   cfg.gemm_impl_request = env_int_optional("UNG_GEMM_IMPL", -1, 0, 2);
   cfg.naive_shared_x = env_int("UNG_NAIVE_SHARED_X", 0, 0, 1) == 1;
   cfg.naive_vec4 = env_int("UNG_NAIVE_VEC4", 1, 0, 1) == 1;
   cfg.naive_heavy_sgemm = env_int("UNG_NAIVE_HEAVY_SGEMM", 1, 0, 1) == 1;
   cfg.naive_group_fused = env_int("UNG_NAIVE_GROUP_FUSED", 0, 0, 1) == 1;
   cfg.tf32_group_prefix = env_int("UNG_TF32_GROUP_PREFIX", 0, 0, 1) == 1;
   cfg.tf32_group_2d = env_int("UNG_TF32_GROUP_2D", 0, 0, 1) == 1;
   cfg.bucket_group_fused = env_int("UNG_BUCKET_GROUP_FUSED", 0, 0, 1) == 1;
   cfg.small_group_fused = env_int("UNG_SMALL_GROUP_FUSED", 1, 0, 1) == 1;
   cfg.medium_group_fused = env_int("UNG_MEDIUM_GROUP_FUSED", 1, 0, 1) == 1;
   cfg.large_group_fused = env_int("UNG_LARGE_GROUP_FUSED", 0, 0, 1) == 1;
}

void read_cross_edge_kernel_limits(CrossEdgeGpuRuntimeConfig &cfg)
{
   cfg.cpu_tiny_nx_max = env_int("UNG_CPU_TINY_NX_MAX", 8, 2, 64);
   cfg.cpu_tiny_nq_max = env_int("UNG_CPU_TINY_NQ_MAX", 64, 1, 4096);
   cfg.cpu_tiny_ops = env_int("UNG_CPU_TINY_OPS", 65536, 1024, 1073741824);
   cfg.source_exact_warps = env_int("UNG_GPU_SOURCE_EXACT_WARPS", 8, 1, 16);
   cfg.source_exact_mode = env_int("UNG_GPU_SOURCE_EXACT_MODE", 0, 0, 1);
   cfg.naive_warps_per_block = env_int("UNG_NAIVE_WARPS_PER_BLOCK", 4, 1, 16);
   cfg.naive_x_cols_per_block = env_int("UNG_NAIVE_X_COLS_PER_BLOCK", 1, 1, 4);
   cfg.naive_shared_kb = env_int("UNG_NAIVE_SHARED_KB", 48, 16, 164);
   cfg.naive_heavy_nx = env_int("UNG_NAIVE_HEAVY_NX", 256, 16, 1048576);
   cfg.naive_heavy_nq = env_int("UNG_NAIVE_HEAVY_NQ", 256, 16, 1048576);
   cfg.naive_heavy_work_m = env_int("UNG_NAIVE_HEAVY_WORK_M", 1, 1, 2000000000);
   cfg.topk_block_threads = env_int_optional("UNG_TOPK_BLOCK_THREADS", -1, 32, 256);
   cfg.naive_group_threads = env_int("UNG_NAIVE_GROUP_THREADS", 128, 64, 256);
   cfg.singleton_threads = env_int("UNG_SINGLETON_THREADS", 256, 64, 512);
   cfg.small_group_max_nx = env_int("UNG_SMALL_GROUP_MAX_NX", 8, 2, 32);
   cfg.small_group_warps = env_int("UNG_SMALL_GROUP_WARPS", 8, 1, 16);
   cfg.medium_group_max_nx = env_int("UNG_MEDIUM_GROUP_MAX_NX", 4096, 9, 8192);
   cfg.medium_group_warps = env_int("UNG_MEDIUM_GROUP_WARPS", 8, 1, 16);
   cfg.large_group_min_nx = env_int("UNG_LARGE_GROUP_MIN_NX", 128, 16, 1048576);
   cfg.large_group_max_nx = env_int("UNG_LARGE_GROUP_MAX_NX", 1023, 16, 1048576);
   cfg.large_group_warps = env_int("UNG_LARGE_GROUP_WARPS", 2, 1, 16);
   cfg.large_group_fused_mode = env_int("UNG_LARGE_GROUP_FUSED_MODE", 2, 0, 2);
}

void read_cross_edge_batch_capacity(CrossEdgeGpuRuntimeConfig &cfg)
{
   cfg.flat_q_cap_mb = env_int("UNG_GPU_FLAT_Q_CAP_MB", 4096, 64, 131072);
   cfg.flat_out_cap_mb = env_int("UNG_GPU_FLAT_OUT_CAP_MB", 4096, 64, 131072);
   cfg.db_chunk_queries = env_int("UNG_GPU_DB_CHUNK_QUERIES", 262144, 4096, 1 << 28);
   cfg.db_medium_max_nx = env_int("UNG_GPU_DB_MEDIUM_MAX_NX", 128, 8, 256);
   cfg.db_large_max_nx =
       env_int("UNG_GPU_DB_LARGE_MAX_NX", cfg.universal_route ? 1048576 : 1023, 129, 1048576);
   cfg.db_medium_group_warps = cfg.medium_group_warps;
   cfg.db_large_group_warps = cfg.large_group_warps;
   cfg.x_stream_x_chunk_mb = env_int("UNG_GPU_X_CHUNK_MB", 8192, 256, 131072);
   cfg.x_stream_q_chunk_mb = env_int("UNG_GPU_X_STREAM_Q_MB", 2048, 64, 131072);
   cfg.x_stream_out_chunk_mb = env_int("UNG_GPU_X_STREAM_OUT_MB", 2048, 64, 131072);
}

void read_cross_edge_vector_upload(CrossEdgeGpuRuntimeConfig &cfg)
{
   cfg.vector_upload.direct_hostreg = env_int("UNG_GPU_PREPARE_DIRECT_HOSTREG", 1, 0, 1) == 1;
   cfg.vector_upload.direct_pageable = env_int("UNG_GPU_PREPARE_DIRECT_PAGEABLE", 0, 0, 1) == 1;
   cfg.vector_upload.hostreg_max_mb = env_int("UNG_GPU_PREPARE_HOSTREG_MAX_MB", 4096, 0, 1048576);
}

void read_cross_edge_diagnostics(CrossEdgeGpuRuntimeConfig &cfg)
{
   cfg.diagnostics.count_valid_pairs = env_int("UNG_COUNT_VALID_PAIRS", 0, 0, 1) == 1;
   cfg.diagnostics.verify_samples = env_int("UNG_GEMM_VERIFY_SAMPLES", 0, 0, 20000);
   cfg.diagnostics.verify_strict = env_int("UNG_GEMM_VERIFY_STRICT", 0, 0, 1) == 1;
}

void read_cross_edge_debug(CrossEdgeGpuRuntimeConfig &cfg)
{
   cfg.debug.medium_group_sync_debug = env_int("UNG_MEDIUM_GROUP_SYNC_DEBUG", 0, 0, 1) == 1;
   cfg.debug.naive_debug_dot = env_int("UNG_NAIVE_DEBUG_DOT", 0, 0, 1) == 1;
}

void read_cross_edge_benchmark_filter(CrossEdgeGpuRuntimeConfig &cfg)
{
   cfg.benchmark_filter.min_nx = env_int("UNG_BENCH_MIN_NX", 0, 0, 1048576);
   cfg.benchmark_filter.max_nx = env_int("UNG_BENCH_MAX_NX", 1048576, 0, 1048576);
   cfg.benchmark_filter.min_work =
       static_cast<long long>(env_int("UNG_BENCH_MIN_WORK_M", 0, 0, 2000000000)) * 1000000LL;
}

void derive_cross_edge_writeback(CrossEdgeGpuRuntimeConfig &cfg,
                                 const UngBuildConfig &build_config)
{
   const bool gpu_batched = is_gpu_batched_cross_edge(build_config);
   cfg.id_vector_writeback = gpu_batched &&
                             !cfg.x_streaming &&
                             env_int("UNG_GPU_ID_VECTOR_WRITEBACK", 0, 0, 1) == 1 &&
                             cfg.db_nosplit;
   cfg.flat_id_writeback = gpu_batched &&
                           !cfg.x_streaming &&
                           env_int("UNG_GPU_FLAT_ID_WRITEBACK", cfg.universal_route ? 1 : 0, 0, 1) == 1 &&
                           cfg.db_nosplit;
   cfg.skip_searchqueue_storage =
       (cfg.id_vector_writeback || cfg.flat_id_writeback) &&
       build_config.gpu_strict &&
       (cfg.source_exact || gpu_batched);
}

void append_cross_edge_route_summary(std::ostringstream &os,
                                     const CrossEdgeGpuRuntimeConfig &cfg)
{
   os << "route=" << cfg.route_name()
      << " class=" << cfg.route_class_name()
      << " source_exact=" << (cfg.source_exact ? 1 : 0)
      << " source_exact_pad_groups=" << (cfg.source_exact_pad_groups ? 1 : 0)
      << " universal=" << (cfg.universal_route ? 1 : 0)
      << " db_nosplit=" << (cfg.db_nosplit ? 1 : 0)
      << " double_buffer=" << (cfg.double_buffer ? 1 : 0)
      << " x_streaming=" << (cfg.x_streaming ? 1 : 0);
}

void append_cross_edge_output_summary(std::ostringstream &os,
                                      const CrossEdgeGpuRuntimeConfig &cfg)
{
   os << " q_upload_device_gather=" << (cfg.query_upload_device_gather ? 1 : 0)
      << " global_merge=" << (cfg.global_merge ? 1 : 0)
      << " global_merge_direct=" << (cfg.global_merge_direct ? 1 : 0);
}

void append_cross_edge_db_route_summary(std::ostringstream &os,
                                        const CrossEdgeGpuRuntimeConfig &cfg)
{
   os << " db_global_merge=" << (cfg.db_global_merge ? 1 : 0)
      << " db_split_unsupported=" << (cfg.db_split_unsupported ? 1 : 0);
}

void append_cross_edge_output_direct_summary(std::ostringstream &os,
                                             const CrossEdgeGpuRuntimeConfig &cfg)
{
   os << " direct_qid_fused=" << (cfg.direct_qid_fused ? 1 : 0)
      << " direct_qid_all_fused=" << (cfg.direct_qid_all_fused ? 1 : 0)
      << " id_only_writeback_requested=" << (cfg.id_only_writeback_requested ? 1 : 0);
}

void append_cross_edge_kernel_summary(std::ostringstream &os,
                                      const CrossEdgeGpuRuntimeConfig &cfg)
{
   os << " cpu_tiny_groups=" << (cfg.cpu_tiny_groups ? 1 : 0)
      << " singleton_fastpath=" << (cfg.singleton_fastpath ? 1 : 0);
}

void append_cross_edge_kernel_mode_summary(std::ostringstream &os,
                                           const CrossEdgeGpuRuntimeConfig &cfg)
{
   os << " force_custom_kernel=" << (cfg.force_custom_kernel ? 1 : 0)
      << " gemm_impl_request=" << cfg.gemm_impl_request
      << " naive_shared_x=" << (cfg.naive_shared_x ? 1 : 0)
      << " naive_vec4=" << (cfg.naive_vec4 ? 1 : 0)
      << " naive_heavy_sgemm=" << (cfg.naive_heavy_sgemm ? 1 : 0)
      << " naive_group_fused=" << (cfg.naive_group_fused ? 1 : 0)
      << " tf32_group_prefix=" << (cfg.tf32_group_prefix ? 1 : 0)
      << " tf32_group_2d=" << (cfg.tf32_group_2d ? 1 : 0)
      << " bucket_group_fused=" << (cfg.bucket_group_fused ? 1 : 0)
      << " small_group_fused=" << (cfg.small_group_fused ? 1 : 0)
      << " medium_group_fused=" << (cfg.medium_group_fused ? 1 : 0)
      << " large_group_fused=" << (cfg.large_group_fused ? 1 : 0);
}

void append_cross_edge_auto_route_summary(std::ostringstream &os,
                                          const CrossEdgeGpuRuntimeConfig &cfg)
{
   os << " auto_x_streaming=" << (cfg.auto_x_streaming ? 1 : 0);
}

void append_cross_edge_kernel_limit_summary(std::ostringstream &os,
                                            const CrossEdgeGpuRuntimeConfig &cfg)
{
   os << " cpu_tiny_nx_max=" << cfg.cpu_tiny_nx_max
      << " cpu_tiny_nq_max=" << cfg.cpu_tiny_nq_max
      << " cpu_tiny_ops=" << cfg.cpu_tiny_ops
      << " source_exact_warps=" << cfg.source_exact_warps
      << " source_exact_mode=" << cfg.source_exact_mode
      << " naive_warps_per_block=" << cfg.naive_warps_per_block
      << " naive_x_cols_per_block=" << cfg.naive_x_cols_per_block
      << " naive_shared_kb=" << cfg.naive_shared_kb
      << " naive_heavy_nx=" << cfg.naive_heavy_nx
      << " naive_heavy_nq=" << cfg.naive_heavy_nq
      << " naive_heavy_work_m=" << cfg.naive_heavy_work_m
      << " topk_block_threads=" << cfg.topk_block_threads
      << " naive_group_threads=" << cfg.naive_group_threads
      << " singleton_threads=" << cfg.singleton_threads
      << " small_group_max_nx=" << cfg.small_group_max_nx
      << " small_group_warps=" << cfg.small_group_warps
      << " medium_group_max_nx=" << cfg.medium_group_max_nx
      << " medium_group_warps=" << cfg.medium_group_warps
      << " large_group_min_nx=" << cfg.large_group_min_nx
      << " large_group_max_nx=" << cfg.large_group_max_nx
      << " large_group_warps=" << cfg.large_group_warps
      << " large_group_fused_mode=" << cfg.large_group_fused_mode;
}

void append_cross_edge_capacity_summary(std::ostringstream &os,
                                        const CrossEdgeGpuRuntimeConfig &cfg)
{
   os << " flat_q_cap_mb=" << cfg.flat_q_cap_mb
      << " flat_out_cap_mb=" << cfg.flat_out_cap_mb
      << " db_chunk_queries=" << cfg.db_chunk_queries
      << " db_medium_max_nx=" << cfg.db_medium_max_nx
      << " db_large_max_nx=" << cfg.db_large_max_nx
      << " db_medium_group_warps=" << cfg.db_medium_group_warps
      << " db_large_group_warps=" << cfg.db_large_group_warps
      << " x_stream_x_chunk_mb=" << cfg.x_stream_x_chunk_mb
      << " x_stream_q_chunk_mb=" << cfg.x_stream_q_chunk_mb
      << " x_stream_out_chunk_mb=" << cfg.x_stream_out_chunk_mb;
}

void append_cross_edge_output_tail_summary(std::ostringstream &os,
                                           const CrossEdgeGpuRuntimeConfig &cfg)
{
   os << " global_merge_direct_max_nx=" << cfg.global_merge_direct_max_nx
      << " gather_threads=" << cfg.gather_threads
      << " gather_blocks=" << cfg.gather_blocks;
}

void append_cross_edge_vector_upload_summary(std::ostringstream &os,
                                             const CrossEdgeGpuRuntimeConfig &cfg)
{
   os << " prepare_direct_hostreg=" << (cfg.vector_upload.direct_hostreg ? 1 : 0)
      << " prepare_direct_pageable=" << (cfg.vector_upload.direct_pageable ? 1 : 0)
      << " prepare_hostreg_max_mb=" << cfg.vector_upload.hostreg_max_mb;
}

void append_cross_edge_diagnostics_summary(std::ostringstream &os,
                                           const CrossEdgeGpuRuntimeConfig &cfg)
{
   os << " count_valid_pairs=" << (cfg.diagnostics.count_valid_pairs ? 1 : 0)
      << " verify_samples=" << cfg.diagnostics.verify_samples
      << " verify_strict=" << (cfg.diagnostics.verify_strict ? 1 : 0);
}

void append_cross_edge_debug_summary(std::ostringstream &os,
                                     const CrossEdgeGpuRuntimeConfig &cfg)
{
   os << " medium_group_sync_debug=" << (cfg.debug.medium_group_sync_debug ? 1 : 0)
      << " naive_debug_dot=" << (cfg.debug.naive_debug_dot ? 1 : 0);
}

void append_cross_edge_benchmark_filter_summary(std::ostringstream &os,
                                                const CrossEdgeGpuRuntimeConfig &cfg)
{
   os << " bench_min_nx=" << cfg.benchmark_filter.min_nx
      << " bench_max_nx=" << cfg.benchmark_filter.max_nx
      << " bench_min_work=" << cfg.benchmark_filter.min_work;
}

void append_cross_edge_writeback_summary(std::ostringstream &os,
                                         const CrossEdgeGpuRuntimeConfig &cfg)
{
   os << " writeback_mode=" << cfg.writeback_mode_name()
      << " searchqueue_storage=" << (cfg.needs_searchqueue_storage() ? 1 : 0);
}

void apply_custom_naive_topk_env_defaults()
{
   set_env_defaults({
       {"UNG_FORCE_CUSTOM_KERNEL", "1"},
       {"UNG_GEMM_IMPL", "2"},
       {"UNG_NAIVE_HEAVY_SGEMM", "0"},
       {"UNG_SMALL_GROUP_FUSED", "0"},
       {"UNG_MEDIUM_GROUP_FUSED", "0"},
       {"UNG_BUCKET_GROUP_FUSED", "0"},
       {"UNG_LARGE_GROUP_FUSED", "0"},
   });
}

void apply_sgemm_topk_env_defaults()
{
   set_env_defaults({
       {"UNG_FORCE_CUSTOM_KERNEL", "0"},
       {"UNG_GEMM_IMPL", "1"},
       {"UNG_NAIVE_HEAVY_SGEMM", "1"},
       {"UNG_NAIVE_HEAVY_NX", "1"},
       {"UNG_NAIVE_HEAVY_NQ", "1"},
       {"UNG_SMALL_GROUP_FUSED", "0"},
       {"UNG_MEDIUM_GROUP_FUSED", "0"},
       {"UNG_BUCKET_GROUP_FUSED", "0"},
       {"UNG_LARGE_GROUP_FUSED", "0"},
   });
}

void apply_fused_group_topk_env_defaults()
{
   set_env_defaults({
       {"UNG_FORCE_CUSTOM_KERNEL", "1"},
       {"UNG_SMALL_GROUP_FUSED", "1"},
       {"UNG_MEDIUM_GROUP_FUSED", "1"},
       {"UNG_MEDIUM_GROUP_MAX_NX", "4096"},
       {"UNG_DIRECT_QID_FUSED", "1"},
       {"UNG_DIRECT_QID_ALL_FUSED", "1"},
       {"UNG_GPU_ID_ONLY_WRITEBACK", "1"},
       {"UNG_GPU_GLOBAL_MERGE", "1"},
       {"UNG_GPU_GLOBAL_MERGE_DIRECT", "1"},
       {"UNG_NAIVE_HEAVY_SGEMM", "0"},
       {"UNG_NAIVE_HEAVY_NX", "16"},
       {"UNG_NAIVE_HEAVY_NQ", "16"},
       {"UNG_BUCKET_GROUP_FUSED", "1"},
       {"UNG_LARGE_GROUP_FUSED", "1"},
       {"UNG_LARGE_GROUP_FUSED_MODE", "2"},
       {"UNG_LARGE_GROUP_MIN_NX", "32"},
       {"UNG_LARGE_GROUP_MAX_NX", "128"},
       {"UNG_LARGE_GROUP_WARPS", "4"},
       {"UNG_TF32_GROUP_2D", "1"},
   });
}

} // namespace

const char *cross_edge_gpu_writeback_mode_name(CrossEdgeGpuWritebackMode mode)
{
   switch (mode)
   {
   case CrossEdgeGpuWritebackMode::SearchQueue:
      return "search_queue";
   case CrossEdgeGpuWritebackMode::IdVector:
      return "id_vector";
   case CrossEdgeGpuWritebackMode::FlatId:
      return "flat_id";
   }
   return "unknown";
}

bool CrossEdgeGpuRuntimeConfig::needs_searchqueue_storage() const
{
   return !skip_searchqueue_storage;
}

CrossEdgeGpuWritebackMode CrossEdgeGpuRuntimeConfig::writeback_mode() const
{
   if (flat_id_writeback)
      return CrossEdgeGpuWritebackMode::FlatId;
   if (id_vector_writeback)
      return CrossEdgeGpuWritebackMode::IdVector;
   return CrossEdgeGpuWritebackMode::SearchQueue;
}

bool CrossEdgeGpuRuntimeConfig::uses_id_vector_writeback() const
{
   return writeback_mode() == CrossEdgeGpuWritebackMode::IdVector;
}

bool CrossEdgeGpuRuntimeConfig::uses_flat_id_writeback() const
{
   return writeback_mode() == CrossEdgeGpuWritebackMode::FlatId;
}

bool CrossEdgeGpuRuntimeConfig::requires_searchqueue_writeback() const
{
   return x_streaming;
}

const char *CrossEdgeGpuRuntimeConfig::writeback_mode_name() const
{
   return cross_edge_gpu_writeback_mode_name(writeback_mode());
}

const char *CrossEdgeGpuRuntimeConfig::route_name() const
{
   if (source_exact)
      return "source_exact";
   if (x_streaming)
      return "x_streaming";
   if (universal_route)
      return "universal_batched";
   return "batched";
}

const char *CrossEdgeGpuRuntimeConfig::route_class_name() const
{
   if (source_exact)
      return "negative_experimental";
   if (x_streaming)
      return "boundary_scalability";
   if (universal_route)
      return "main_engineering";
   return "main_or_baseline";
}

std::string CrossEdgeGpuRuntimeConfig::validation_error() const
{
   if (requires_searchqueue_writeback() && writeback_mode() != CrossEdgeGpuWritebackMode::SearchQueue)
      return "x_streaming requires SearchQueue writeback; disable id/flat writeback.";
   if (db_chunk_queries <= 0)
      return "db_chunk_queries must be positive.";
   if (flat_q_cap_mb <= 0 || flat_out_cap_mb <= 0)
      return "flat batch caps must be positive.";
   if (x_stream_x_chunk_mb <= 0 || x_stream_q_chunk_mb <= 0 || x_stream_out_chunk_mb <= 0)
      return "X-streaming chunk sizes must be positive.";
   return "";
}

std::string CrossEdgeGpuRuntimeConfig::summary() const
{
   std::ostringstream os;
   append_cross_edge_route_summary(os, *this);
   append_cross_edge_output_summary(os, *this);
   append_cross_edge_db_route_summary(os, *this);
   append_cross_edge_kernel_summary(os, *this);
   append_cross_edge_output_direct_summary(os, *this);
   append_cross_edge_kernel_mode_summary(os, *this);
   append_cross_edge_auto_route_summary(os, *this);
   append_cross_edge_kernel_limit_summary(os, *this);
   append_cross_edge_output_tail_summary(os, *this);
   append_cross_edge_capacity_summary(os, *this);
   append_cross_edge_vector_upload_summary(os, *this);
   append_cross_edge_diagnostics_summary(os, *this);
   append_cross_edge_debug_summary(os, *this);
   append_cross_edge_benchmark_filter_summary(os, *this);
   append_cross_edge_writeback_summary(os, *this);
   return os.str();
}

void apply_cross_edge_gpu_topk_env_defaults(UngGpuTopkImpl impl)
{
   switch (impl)
   {
   case UngGpuTopkImpl::CustomNaive:
      apply_custom_naive_topk_env_defaults();
      break;
   case UngGpuTopkImpl::SgemmTopk:
      apply_sgemm_topk_env_defaults();
      break;
   case UngGpuTopkImpl::FusedGroupTopk:
      apply_fused_group_topk_env_defaults();
      break;
   case UngGpuTopkImpl::Auto:
      break;
   }
}

CrossEdgeGpuRuntimeConfig make_cross_edge_gpu_runtime_config(const UngBuildConfig &build_config,
                                                             bool backend_is_gpu)
{
   CrossEdgeGpuRuntimeConfig cfg;
   read_cross_edge_route_flags(cfg, build_config, backend_is_gpu);
   read_cross_edge_output_flags(cfg);
   read_cross_edge_kernel_flags(cfg);
   read_cross_edge_kernel_limits(cfg);
   read_cross_edge_batch_capacity(cfg);
   read_cross_edge_vector_upload(cfg);
   read_cross_edge_diagnostics(cfg);
   read_cross_edge_debug(cfg);
   read_cross_edge_benchmark_filter(cfg);
   derive_cross_edge_writeback(cfg, build_config);
   return cfg;
}

} // namespace ANNS
