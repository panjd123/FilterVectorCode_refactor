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
                                                      bool grnnd_like_refine = false);

TagoreBatchBuildResult build_tagore_vamana_cuda_batch(const std::vector<TagoreGroupRequest> &groups,
                                                      uint32_t dim,
                                                      uint32_t k,
                                                      uint32_t final_degree,
                                                      uint32_t top_m,
                                                      uint32_t iterations,
                                                      float alpha,
                                                      TagorePruneMode prune_mode);

} // namespace ANNS

#endif // ANNS_TAGORE_GRAPH_BUILDER_H
