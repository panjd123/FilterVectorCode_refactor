#include "tagore_graph_builder.h"
#include "ung_gpu_l2_batch.h"
#include "uni_nav_graph.h"

#include <stdexcept>

namespace ANNS
{
namespace
{
[[noreturn]] void throw_cuda_unavailable()
{
   throw std::runtime_error(
       "CUDA backend is unavailable in this CPU-only UNG build");
}
} // namespace

TagoreBuildResult build_tagore_vamana_cuda(const float *,
                                            uint32_t,
                                            uint32_t,
                                            uint32_t,
                                            uint32_t,
                                            uint32_t,
                                            uint32_t,
                                            float)
{
   throw_cuda_unavailable();
}

TagoreBatchBuildResult build_tagore_vamana_cuda_batch(
    const std::vector<TagoreGroupRequest> &,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    float,
    TagorePruneMode)
{
   throw_cuda_unavailable();
}

TagoreBatchBuildResult build_tagore_vamana_cuda_batch(
    const std::vector<TagoreGroupRequest> &,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    float,
    TagorePruneMode,
    const TagoreCudaRuntimeConfig &)
{
   throw_cuda_unavailable();
}

void UniNavGraph::gpu_prepare_all_vectors_on_device(
    const CrossEdgeVectorUploadConfig &,
    double *,
    double *)
{
   throw_cuda_unavailable();
}

void UniNavGraph::gpu_prepare_all_vectors_for_cross_edge(
    CrossEdgeBuildTiming &,
    const CrossEdgeGpuRuntimeConfig &)
{
   throw_cuda_unavailable();
}

void UniNavGraph::gpu_release_all_vectors_on_device()
{
   // Safe no-op: callers use this for cleanup after a failed GPU route.
}

void UniNavGraph::gpu_cross_groups_search_all_batched(
    const std::vector<IdxType> &,
    int,
    int,
    std::vector<SearchQueue> &,
    std::vector<std::vector<IdxType>> *,
    std::vector<IdxType> *,
    CrossEdgeBuildTiming *,
    const CrossEdgeGpuRuntimeConfig &)
{
   throw_cuda_unavailable();
}

void UniNavGraph::gpu_cross_groups_search_x_streaming(
    const std::vector<IdxType> &,
    int,
    int,
    std::vector<SearchQueue> &,
    CrossEdgeBuildTiming *,
    const CrossEdgeGpuRuntimeConfig &)
{
   throw_cuda_unavailable();
}

bool UniNavGraph::build_cross_edges_generate_cuvs_bruteforce(
    std::vector<SearchQueue> &,
    CrossEdgeBuildTiming &,
    const CrossEdgeGpuRuntimeConfig &)
{
   return false;
}

bool UniNavGraph::build_cross_edges_generate_gpu_source_exact(
    std::vector<SearchQueue> &,
    std::vector<std::vector<IdxType>> *,
    std::vector<IdxType> *,
    CrossEdgeBuildTiming &,
    const CrossEdgeGpuRuntimeConfig &)
{
   return false;
}

bool UniNavGraph::gpu_build_special_inter_edges(
    const std::vector<SpecialBlock> &,
    const std::vector<std::vector<IdxType>> &,
    std::vector<std::vector<SpecialEdge>> &,
    IdxType &,
    double &)
{
   return false;
}

bool gpu_l2_batch_compute_float(const float *,
                                uint32_t,
                                uint32_t,
                                const float *,
                                const uint32_t *,
                                size_t,
                                float *)
{
   return false;
}

bool gpu_l2_batch_compute_query_candidates_float(const float *,
                                                 uint32_t,
                                                 uint32_t,
                                                 const float *,
                                                 size_t,
                                                 const uint32_t *,
                                                 const uint32_t *,
                                                 size_t,
                                                 float *)
{
   return false;
}

} // namespace ANNS
