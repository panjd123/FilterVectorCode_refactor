#ifndef ANNS_UNG_GROUP_GRAPH_BUILD_TYPES_H
#define ANNS_UNG_GROUP_GRAPH_BUILD_TYPES_H

#include "config.h"
#include "tagore_graph_builder.h"
#include "ung_build_settings.h"

#include <cstdint>
#include <vector>

namespace ANNS
{

struct GroupGraphBuildContext
{
   GroupGraphRoute route;
   CpuGroupGraphSettings cpu_settings;
};

GroupGraphBuildContext make_group_graph_build_context(const UngBuildConfig &build_config,
                                                      IdxType max_degree,
                                                      uint32_t num_threads);

struct TagoreGroupBuildContext
{
   GroupGraphRoute route;
   TagoreGroupGraphSettings settings;
   TagoreCudaRuntimeConfig runtime_config;
   TagorePruneMode prune_mode = TagorePruneMode::TagoreVamana;
   uint32_t exact_batch_threshold = 0;
};

TagoreGroupBuildContext make_tagore_group_build_context(const UngBuildConfig &build_config,
                                                        IdxType max_degree,
                                                        uint32_t num_threads);

struct TagoreFallbackBuildStats
{
   double wall_ms = 0.0;
   size_t complete_groups = 0;
   size_t complete_points = 0;
   size_t cpu_groups = 0;
   size_t cpu_points = 0;
};

struct TagoreBatchBuildArtifacts
{
   std::vector<TagoreBuildResult> results;
   std::vector<size_t> exact_packed_rank;
   std::vector<uint32_t> exact_packed_graph;
   std::vector<uint32_t> exact_packed_offsets;
   uint32_t exact_packed_stride = 0;
   TagoreBatchBuildResult batch_timing{};
   double wall_ms = 0.0;
};

struct TagoreGroupPartition
{
   std::vector<IdxType> fallback_group_ids;
   std::vector<IdxType> tagore_group_ids;
   std::vector<TagoreGroupRequest> tagore_requests;
   uint64_t tagore_total_points = 0;
};

} // namespace ANNS

#endif // ANNS_UNG_GROUP_GRAPH_BUILD_TYPES_H
