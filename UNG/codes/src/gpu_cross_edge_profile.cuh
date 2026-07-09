// Profiling summary helpers for the regular batched cross-edge path.
//
// These helpers keep log formatting out of gpu_cross_groups_search_all_batched()
// without changing any counters, timing fields, or profile keys.

struct CrossEdgeRegularProfileSummary {
    size_t valid_topk_pairs = 0;
    size_t tail_gemv_count = 0;

    size_t small_group_fused_groups = 0;
    size_t small_group_fused_queries = 0;
    int small_group_max_nx = 0;

    size_t medium_group_fused_groups = 0;
    size_t medium_group_fused_queries = 0;
    int medium_group_max_nx = 0;

    size_t large_group_fused_groups = 0;
    size_t large_group_fused_queries = 0;
    int large_group_min_nx = 0;
    int large_group_max_nx = 0;
    int large_group_warps = 0;

    bool bucket_group_fused = false;
    size_t small_desc_count = 0;
    size_t medium_desc_count = 0;
    size_t tf32_tile_desc_count = 0;
    size_t tf32_group_count = 0;
    size_t tf32_group_tile_count = 0;

    bool direct_qid_fused = false;
    bool direct_qid_all_effective = false;
    bool id_only_writeback = false;
    bool id_vector_writeback = false;
    bool flat_id_writeback = false;

    bool gpu_global_merge = false;
    int d2h_rows = 0;
    bool gpu_global_merge_direct = false;
    size_t global_merge_direct_groups = 0;
    size_t global_merge_direct_queries = 0;
    int gpu_global_merge_direct_max_nx = 0;

    size_t heavy_sgemm_groups = 0;
    size_t heavy_sgemm_queries = 0;

    size_t cpu_tiny_groups = 0;
    size_t cpu_tiny_queries = 0;
    double cpu_tiny_ms = 0.0;
    bool cpu_tiny_enabled = false;
    int cpu_tiny_nx_max = 0;
    int cpu_tiny_nq_max = 0;
    long long cpu_tiny_ops_thresh = 0;

    double count_ms = 0.0;
    double offset_ms = 0.0;
    double fill_q_ms = 0.0;
    double writeback_ms = 0.0;
    int total_queries = 0;
    bool gather_q = false;
    bool singleton_fastpath = false;
    size_t singleton_queries = 0;
};

inline void log_cross_edge_regular_profile_summary(const CrossEdgeRegularProfileSummary& s) {
    prof_logf("[PROF] cross_edges.topk_valid_pairs total=%zu", s.valid_topk_pairs);
    prof_logf("[PROF] cross_edges.tail_gemv_calls total=%zu", s.tail_gemv_count);
    prof_logf("[PROF] cross_edges.small_group_fused groups=%zu queries=%zu max_nx=%d",
              s.small_group_fused_groups, s.small_group_fused_queries, s.small_group_max_nx);
    prof_logf("[PROF] cross_edges.medium_group_fused groups=%zu queries=%zu max_nx=%d",
              s.medium_group_fused_groups, s.medium_group_fused_queries, s.medium_group_max_nx);
    prof_logf("[PROF] cross_edges.large_group_fused groups=%zu queries=%zu min_nx=%d max_nx=%d warps=%d",
              s.large_group_fused_groups, s.large_group_fused_queries,
              s.large_group_min_nx, s.large_group_max_nx, s.large_group_warps);
    prof_logf("[PROF] cross_edges.bucket_group_fused enabled=%d small_desc=%zu medium_desc=%zu tf32_tile_desc=%zu",
              s.bucket_group_fused ? 1 : 0,
              s.small_desc_count,
              s.medium_desc_count,
              s.tf32_tile_desc_count);
    prof_logf("[PROF] cross_edges.tf32_group_prefix groups=%zu tiles=%zu",
              s.tf32_group_count,
              s.tf32_group_tile_count);
    prof_logf("[PROF] cross_edges.direct_qid_fused enabled=%d all=%d",
              s.direct_qid_fused ? 1 : 0, s.direct_qid_all_effective ? 1 : 0);
    prof_logf("[PROF] cross_edges.id_only_writeback enabled=%d", s.id_only_writeback ? 1 : 0);
    prof_logf("[PROF] cross_edges.id_vector_writeback enabled=%d", s.id_vector_writeback ? 1 : 0);
    prof_logf("[PROF] cross_edges.flat_id_writeback enabled=%d", s.flat_id_writeback ? 1 : 0);
    prof_logf("[PROF] cross_edges.gpu_global_merge enabled=%d rows=%d",
              s.gpu_global_merge ? 1 : 0, s.d2h_rows);
    prof_logf("[PROF] cross_edges.gpu_global_merge_direct enabled=%d groups=%zu queries=%zu",
              s.gpu_global_merge_direct ? 1 : 0,
              s.global_merge_direct_groups,
              s.global_merge_direct_queries);
    prof_logf("[PROF] cross_edges.gpu_global_merge_direct_max_nx value=%d",
              s.gpu_global_merge_direct_max_nx);
    prof_logf("[PROF] cross_edges.heavy_sgemm groups=%zu queries=%zu",
              s.heavy_sgemm_groups, s.heavy_sgemm_queries);
    prof_logf("[PROF] cross_edges.cpu_tiny groups=%zu queries=%zu ms=%.3f enabled=%d nx_max=%d nq_max=%d ops_thresh=%lld",
              s.cpu_tiny_groups, s.cpu_tiny_queries, s.cpu_tiny_ms, s.cpu_tiny_enabled ? 1 : 0,
              s.cpu_tiny_nx_max, s.cpu_tiny_nq_max, s.cpu_tiny_ops_thresh);
    prof_logf("[PROF] cross_edges.stage_ms count=%.3f offset=%.3f fill_q=%.3f writeback=%.3f total_queries=%d gather_q=%d singleton_fastpath=%d singleton_q=%zu cpu_tiny_groups=%zu",
              s.count_ms,
              s.offset_ms,
              s.fill_q_ms,
              s.writeback_ms,
              s.total_queries,
              s.gather_q ? 1 : 0,
              s.singleton_fastpath ? 1 : 0,
              s.singleton_queries,
              s.cpu_tiny_groups);
}
