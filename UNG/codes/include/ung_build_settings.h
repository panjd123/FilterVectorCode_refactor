#ifndef ANNS_UNG_BUILD_SETTINGS_H
#define ANNS_UNG_BUILD_SETTINGS_H

#include "config.h"
#include "ung_build_config.h"

#include <cstdint>
#include <string>

namespace ANNS
{

enum class TagorePruneMode : uint32_t;

struct GroupGraphRoute
{
   UngGroupGraphImpl impl = UngGroupGraphImpl::VamanaCpu;

   const char *name() const;
   bool uses_cuda_backend() const;
   bool is_adaptive_cuda() const;
   TagorePruneMode tagore_prune_mode() const;
};

GroupGraphRoute make_group_graph_route(const UngBuildConfig &build_config);

struct GraphReserveSettings
{
   IdxType complete_threshold = 0;
   IdxType additional_slack = 0;
   IdxType hard_cap = 0;
   IdxType auto_min_points = 0;
   IdxType auto_min_groups = 0;
   int auto_small_pct = 0;
   bool manual_override = false;
   bool manual_enabled = false;

   const char *mode_name() const;
   bool auto_enabled(IdxType num_points, IdxType num_groups, uint64_t small_points) const;
   bool enabled(IdxType num_points, IdxType num_groups, uint64_t small_points) const;
   std::string summary(IdxType num_points, IdxType num_groups, uint64_t small_points) const;
};

GraphReserveSettings make_graph_reserve_settings(IdxType max_degree, IdxType num_cross_edges);

struct CpuGroupGraphSettings
{
   IdxType complete_threshold = 0;
   bool bounded_complete = false;
   bool profile = false;
   int large_inner_threads = 1;
   IdxType large_inner_threshold = 1024;
   int large_outer_threads = 1;

   bool split_large_group_build() const;
   std::string summary() const;
};

CpuGroupGraphSettings make_cpu_group_graph_settings(IdxType max_degree, uint32_t num_threads);

struct TagoreGroupGraphSettings
{
   IdxType complete_threshold = 0;
   int fallback_impl = 1;
   bool bounded_complete = true;
   int fallback_threads = 1;
   bool overlap_fallback = true;
   uint32_t adaptive_exact_max_nx = 4096;
   uint32_t fast_grnnd_batch_exact_nx = 0;
   bool fast_fill = false;
   int fill_threads = 1;

   bool should_overlap_fallback(bool has_fallback_groups, bool has_gpu_requests) const;
   uint32_t exact_batch_threshold(TagorePruneMode prune_mode) const;
   std::string summary() const;
};

TagoreGroupGraphSettings make_tagore_group_graph_settings(IdxType max_degree,
                                                          uint32_t num_threads,
                                                          bool adaptive_group_graph);

struct IntraGroupIdSettings
{
   int requested = -1;
   bool auto_enable = false;

   bool enabled() const;
   std::string summary() const;
};

IntraGroupIdSettings make_intra_group_id_settings(const UngBuildConfig &build_config);

struct CpuHybridCrossSettings
{
   long long work_threshold = 10000;
   IdxType target_exact_max_nx = 1000;

   std::string summary() const;
};

CpuHybridCrossSettings make_cpu_hybrid_cross_settings();

struct GpuCrossLifecycleSettings
{
   bool release_after_cross = false;

   std::string summary() const;
};

GpuCrossLifecycleSettings make_gpu_cross_lifecycle_settings();

struct AdditionalEdgesSettings
{
   bool direct_append_requested = false;

   bool direct_append_enabled(UngAdditionalEdgesImpl impl) const;
   std::string summary(UngAdditionalEdgesImpl impl) const;
};

AdditionalEdgesSettings make_additional_edges_settings();

} // namespace ANNS

#endif // ANNS_UNG_BUILD_SETTINGS_H
