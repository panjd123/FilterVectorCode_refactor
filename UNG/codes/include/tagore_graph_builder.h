#ifndef ANNS_TAGORE_GRAPH_BUILDER_H
#define ANNS_TAGORE_GRAPH_BUILDER_H

#include <cstdint>
#include <vector>

namespace ANNS
{

struct TagoreGroupRequest
{
   const float *data = nullptr;
   uint32_t num_points = 0;
};

struct TagoreBuildResult
{
   std::vector<uint32_t> graph;
   uint32_t graph_stride = 0;
   uint32_t entry_point = 0;
   double convert_ms = 0.0;
   double alloc_ms = 0.0;
   double h2d_ms = 0.0;
   double memset_ms = 0.0;
   double gnn_ms = 0.0;
   double prune_ms = 0.0;
   double grnnd_refine_ms = 0.0;
   double d2h_ms = 0.0;
   double free_ms = 0.0;
};

enum class TagorePruneMode : uint32_t
{
   TagoreVamana = 0,
   TagoreVamanaWithGrnndRefine = 1,
   FastGrnnd = 2,
};

struct TagoreFastExactBatchConfig
{
   uint32_t head_keep = 0;
   uint32_t anchor_tail = 1;
   uint32_t anchor_slots = 0; // 0 means use all remaining slots when anchor_tail is enabled.
   uint32_t bidir_anchor = 0;
   uint32_t reverse_cap = 0;
   uint32_t reverse_slots = 0;
   uint32_t reverse_forward_cap = 0;
   bool pinned_host = false;
   bool device_lookup = true;
   bool direct_h2d_requested = true;
   uint32_t direct_h2d_max_runs = 4096;
   uint32_t graph_stride = 0;
   bool use_warp_kernel = false;
};

struct TagoreFastGrnndPruneConfig
{
   uint32_t light_threshold = 256;
   uint32_t light_head_keep = 0;
   uint32_t light_reverse_cap = 0;
   uint32_t light_reverse_slots = 1;
   uint32_t light_reverse_forward_cap = 1;
   uint32_t repair_degree = 1;
   uint32_t forward_cap = 1;
   uint32_t reverse_cap = 0;
};

struct TagoreCudaRuntimeConfig
{
   TagoreFastExactBatchConfig fast_exact;
   TagoreFastGrnndPruneConfig fast_prune;
   uint32_t fast_exact_batch_threshold = 0;
   uint32_t requested_streams = 1;
   bool compact_d2h_requested = true;
};

struct TagoreBatchBuildResult
{
   std::vector<TagoreBuildResult> groups;
   std::vector<uint32_t> packed_graph;
   std::vector<uint32_t> packed_offsets;
   uint32_t packed_graph_stride = 0;
   double batch_pack_ms = 0.0;
   double batch_alloc_ms = 0.0;
   double batch_free_ms = 0.0;
};

TagoreBuildResult build_tagore_vamana_cuda(const float *data,
                                           uint32_t num_points,
                                           uint32_t dim,
                                           uint32_t k,
                                           uint32_t final_degree,
                                           uint32_t top_m,
                                           uint32_t iterations,
                                           float alpha);

TagoreBatchBuildResult build_tagore_vamana_cuda_batch(const std::vector<TagoreGroupRequest> &groups,
                                                      uint32_t dim,
                                                      uint32_t k,
                                                      uint32_t final_degree,
                                                      uint32_t top_m,
                                                      uint32_t iterations,
                                                      float alpha,
                                                      TagorePruneMode prune_mode);

TagoreCudaRuntimeConfig make_tagore_cuda_runtime_config(uint32_t k,
                                                        uint32_t final_degree,
                                                        bool allow_parallel);

TagoreBatchBuildResult build_tagore_vamana_cuda_batch(const std::vector<TagoreGroupRequest> &groups,
                                                      uint32_t dim,
                                                      uint32_t k,
                                                      uint32_t final_degree,
                                                      uint32_t top_m,
                                                      uint32_t iterations,
                                                      float alpha,
                                                      TagorePruneMode prune_mode,
                                                      const TagoreCudaRuntimeConfig &runtime_config);

} // namespace ANNS

#endif // ANNS_TAGORE_GRAPH_BUILDER_H
