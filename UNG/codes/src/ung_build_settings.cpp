#include "include/ung_build_settings.h"
#include "include/tagore_graph_builder.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>

namespace
{

int read_env_int_local(const char *key, int fallback, int min_v, int max_v)
{
   const char *s = std::getenv(key);
   if (!s || !*s)
      return fallback;
   char *end = nullptr;
   long v = std::strtol(s, &end, 10);
   if (end == s || *end != '\0')
      return fallback;
   if (v < min_v)
      v = min_v;
   if (v > max_v)
      v = max_v;
   return static_cast<int>(v);
}

int read_env_int_clamped(const char *key, int fallback, int min_v, int max_v)
{
   const char *s = std::getenv(key);
   if (!s || !*s)
      return fallback;
   char *end = nullptr;
   long v = std::strtol(s, &end, 10);
   if (end == s || *end != '\0')
      return fallback;
   if (v < static_cast<long>(min_v))
      v = static_cast<long>(min_v);
   if (v > static_cast<long>(max_v))
      v = static_cast<long>(max_v);
   return static_cast<int>(v);
}

bool env_is_set_local(const char *key)
{
   const char *s = std::getenv(key);
   return s && *s;
}

} // namespace

namespace ANNS
{

const char *GroupGraphRoute::name() const
{
   return to_string(impl);
}

bool GroupGraphRoute::uses_cuda_backend() const
{
   return impl == UngGroupGraphImpl::TagoreCuda ||
          impl == UngGroupGraphImpl::GrnndLikeCuda ||
          impl == UngGroupGraphImpl::FastGrnndCuda ||
          impl == UngGroupGraphImpl::AdaptiveCuda;
}

bool GroupGraphRoute::is_adaptive_cuda() const
{
   return impl == UngGroupGraphImpl::AdaptiveCuda;
}

TagorePruneMode GroupGraphRoute::tagore_prune_mode() const
{
   if (impl == UngGroupGraphImpl::GrnndLikeCuda)
      return TagorePruneMode::TagoreVamanaWithGrnndRefine;
   if (impl == UngGroupGraphImpl::FastGrnndCuda ||
       impl == UngGroupGraphImpl::AdaptiveCuda)
      return TagorePruneMode::FastGrnnd;
   return TagorePruneMode::TagoreVamana;
}

GroupGraphRoute make_group_graph_route(const UngBuildConfig &build_config)
{
   GroupGraphRoute route;
   route.impl = build_config.group_graph_impl;
   return route;
}

const char *GraphReserveSettings::mode_name() const
{
   return manual_override ? "manual" : "auto";
}

bool GraphReserveSettings::auto_enabled(IdxType num_points,
                                        IdxType num_groups,
                                        uint64_t small_points) const
{
   return (num_points >= auto_min_points) ||
          (num_groups >= auto_min_groups) ||
          (num_points > 0 &&
           (static_cast<double>(small_points) / static_cast<double>(num_points)) >=
               static_cast<double>(auto_small_pct) / 100.0);
}

bool GraphReserveSettings::enabled(IdxType num_points,
                                   IdxType num_groups,
                                   uint64_t small_points) const
{
   return manual_override ? manual_enabled : auto_enabled(num_points, num_groups, small_points);
}

std::string GraphReserveSettings::summary(IdxType num_points,
                                          IdxType num_groups,
                                          uint64_t small_points) const
{
   std::ostringstream os;
   os << "mode=" << mode_name()
      << " enabled=" << (enabled(num_points, num_groups, small_points) ? 1 : 0)
      << " manual_enabled=" << (manual_enabled ? 1 : 0)
      << " complete_threshold=" << complete_threshold
      << " auto_min_points=" << auto_min_points
      << " auto_min_groups=" << auto_min_groups
      << " auto_small_pct=" << auto_small_pct
      << " additional_slack=" << additional_slack
      << " hard_cap=" << hard_cap;
   return os.str();
}

GraphReserveSettings make_graph_reserve_settings(IdxType max_degree, IdxType num_cross_edges)
{
   GraphReserveSettings cfg;
   cfg.complete_threshold =
       static_cast<IdxType>(read_env_int_local("UNG_GROUP_GRAPH_COMPLETE_NX",
                                               static_cast<int>(std::max<IdxType>(max_degree, 64)),
                                               static_cast<int>(max_degree),
                                               1 << 20));
   cfg.auto_min_points =
       static_cast<IdxType>(read_env_int_local("UNG_GRAPH_RESERVE_AUTO_MIN_POINTS",
                                               500000, 1, 1 << 30));
   cfg.auto_min_groups =
       static_cast<IdxType>(read_env_int_local("UNG_GRAPH_RESERVE_AUTO_MIN_GROUPS",
                                               20000, 1, 1 << 30));
   cfg.auto_small_pct =
       read_env_int_local("UNG_GRAPH_RESERVE_AUTO_SMALL_PCT", 50, 0, 100);
   cfg.manual_override = env_is_set_local("UNG_GRAPH_RESERVE_CAPACITY");
   cfg.manual_enabled =
       cfg.manual_override && read_env_int_local("UNG_GRAPH_RESERVE_CAPACITY", 0, 0, 1) == 1;
   cfg.additional_slack =
       static_cast<IdxType>(read_env_int_local("UNG_GRAPH_RESERVE_ADDITIONAL_SLACK",
                                               static_cast<int>(num_cross_edges),
                                               0, 1 << 20));
   cfg.hard_cap =
       static_cast<IdxType>(read_env_int_local("UNG_GRAPH_RESERVE_HARD_CAP",
                                               1 << 20,
                                               1, 1 << 30));
   return cfg;
}

bool CpuGroupGraphSettings::split_large_group_build() const
{
   return large_inner_threads > 1;
}

std::string CpuGroupGraphSettings::summary() const
{
   std::ostringstream os;
   os << "complete_threshold=" << complete_threshold
      << " bounded_complete=" << (bounded_complete ? 1 : 0)
      << " profile=" << (profile ? 1 : 0)
      << " large_inner_threads=" << large_inner_threads
      << " large_inner_nx=" << large_inner_threshold
      << " large_outer_threads=" << large_outer_threads
      << " split_large=" << (split_large_group_build() ? 1 : 0);
   return os.str();
}

CpuGroupGraphSettings make_cpu_group_graph_settings(IdxType max_degree, uint32_t num_threads)
{
   CpuGroupGraphSettings cfg;
   cfg.complete_threshold =
       static_cast<IdxType>(read_env_int_local("UNG_GROUP_GRAPH_COMPLETE_NX",
                                               static_cast<int>(std::max<IdxType>(max_degree, 64)),
                                               static_cast<int>(max_degree),
                                               1 << 20));
   cfg.bounded_complete =
       read_env_int_local("UNG_GROUP_GRAPH_BOUNDED_COMPLETE", 0, 0, 1) == 1;
   cfg.profile =
       read_env_int_local("UNG_GROUP_GRAPH_PROFILE", 0, 0, 1) == 1;
   cfg.large_inner_threads =
       read_env_int_local("UNG_GROUP_GRAPH_LARGE_INNER_THREADS", 1, 1, static_cast<int>(num_threads));
   cfg.large_inner_threshold =
       static_cast<IdxType>(read_env_int_local("UNG_GROUP_GRAPH_LARGE_INNER_NX", 1024, 1, 1 << 20));
   const int default_large_outer_threads =
       std::max(1, static_cast<int>(num_threads) / std::max(1, cfg.large_inner_threads));
   cfg.large_outer_threads =
       read_env_int_local("UNG_GROUP_GRAPH_LARGE_OUTER_THREADS", default_large_outer_threads,
                          1, static_cast<int>(num_threads));
   return cfg;
}

bool TagoreGroupGraphSettings::should_overlap_fallback(bool has_fallback_groups,
                                                       bool has_gpu_requests) const
{
   return overlap_fallback && has_fallback_groups && has_gpu_requests;
}

uint32_t TagoreGroupGraphSettings::exact_batch_threshold(TagorePruneMode prune_mode) const
{
   return prune_mode == TagorePruneMode::FastGrnnd ? fast_grnnd_batch_exact_nx : 0;
}

std::string TagoreGroupGraphSettings::summary() const
{
   std::ostringstream os;
   os << "complete_threshold=" << complete_threshold
      << " fallback_impl=" << fallback_impl
      << " bounded_complete=" << (bounded_complete ? 1 : 0)
      << " fallback_threads=" << fallback_threads
      << " overlap_fallback=" << (overlap_fallback ? 1 : 0)
      << " adaptive_exact_max_nx=" << adaptive_exact_max_nx
      << " fast_grnnd_batch_exact_nx=" << fast_grnnd_batch_exact_nx
      << " fast_fill=" << (fast_fill ? 1 : 0)
      << " fill_threads=" << fill_threads;
   return os.str();
}

TagoreGroupGraphSettings make_tagore_group_graph_settings(IdxType max_degree,
                                                          uint32_t num_threads,
                                                          bool adaptive_group_graph)
{
   TagoreGroupGraphSettings cfg;
   cfg.complete_threshold =
       static_cast<IdxType>(read_env_int_local("UNG_GROUP_GRAPH_COMPLETE_NX",
                                               static_cast<int>(std::max<IdxType>(max_degree, 64)),
                                               static_cast<int>(max_degree),
                                               1 << 20));
   cfg.fallback_impl = read_env_int_local("UNG_TAGORE_FALLBACK_IMPL", 1, 0, 1);
   cfg.bounded_complete =
       read_env_int_local("UNG_GROUP_GRAPH_BOUNDED_COMPLETE", adaptive_group_graph ? 1 : 0, 0, 1) == 1;
   cfg.fallback_threads =
       read_env_int_local("UNG_TAGORE_FALLBACK_THREADS",
                          static_cast<int>(num_threads),
                          1, static_cast<int>(num_threads));
   cfg.overlap_fallback =
       read_env_int_local("UNG_TAGORE_OVERLAP_FALLBACK", adaptive_group_graph ? 1 : 0, 0, 1) == 1;
   cfg.adaptive_exact_max_nx =
       static_cast<uint32_t>(read_env_int_local("UNG_ADAPTIVE_EXACT_MAX_NX",
                                                4096, 0, 1 << 20));
   cfg.fast_grnnd_batch_exact_nx =
       static_cast<uint32_t>(read_env_int_local("UNG_FAST_GRNND_BATCH_EXACT_NX",
                                                adaptive_group_graph ? static_cast<int>(cfg.adaptive_exact_max_nx) : 0,
                                                0, 1 << 20));
   cfg.fast_fill = read_env_int_local("UNG_TAGORE_FILL_FAST", 0, 0, 1) == 1;
   const int default_fill_threads =
       std::min<int>(static_cast<int>(num_threads), 16);
   cfg.fill_threads =
       read_env_int_local("UNG_TAGORE_FILL_THREADS", default_fill_threads, 1,
                          static_cast<int>(num_threads));
   return cfg;
}

bool IntraGroupIdSettings::enabled() const
{
   return requested >= 0 ? requested == 1 : auto_enable;
}

std::string IntraGroupIdSettings::summary() const
{
   std::ostringstream os;
   os << "requested=" << requested
      << " auto_enable=" << (auto_enable ? 1 : 0)
      << " enabled=" << (enabled() ? 1 : 0);
   return os.str();
}

IntraGroupIdSettings make_intra_group_id_settings(const UngBuildConfig &build_config)
{
   IntraGroupIdSettings cfg;
   const GroupGraphRoute group_route = make_group_graph_route(build_config);
   const bool gpu_group_graph = group_route.uses_cuda_backend();
   const bool cross_uses_group_vamana =
       build_config.cross_edge_impl == UngCrossEdgeImpl::CpuVamana ||
       build_config.cross_edge_impl == UngCrossEdgeImpl::OriginalCpu ||
       build_config.cross_edge_impl == UngCrossEdgeImpl::CpuHybridScanVamana;
   const bool additional_uses_group_vamana =
       build_config.additional_edges_impl == UngAdditionalEdgesImpl::CpuVamana;
   cfg.auto_enable = gpu_group_graph && !cross_uses_group_vamana && !additional_uses_group_vamana;
   cfg.requested = read_env_int_local("UNG_INTRA_GLOBAL_IDS", -1, -1, 1);
   return cfg;
}

std::string CpuHybridCrossSettings::summary() const
{
   std::ostringstream os;
   os << "threshold_nqnx=" << work_threshold
      << " target_exact_max_nx=" << target_exact_max_nx;
   return os.str();
}

CpuHybridCrossSettings make_cpu_hybrid_cross_settings()
{
   CpuHybridCrossSettings cfg;
   cfg.work_threshold =
       read_env_int_clamped("UNG_CPU_HYBRID_SCAN_MAX_WORK", 10000, 0, 1 << 30);
   cfg.target_exact_max_nx =
       static_cast<IdxType>(read_env_int_clamped("UNG_CPU_TARGET_EXACT_MAX_NX", 1000, 0, 1 << 30));
   return cfg;
}

std::string GpuCrossLifecycleSettings::summary() const
{
   std::ostringstream os;
   os << "release_after_cross=" << (release_after_cross ? 1 : 0);
   return os.str();
}

GpuCrossLifecycleSettings make_gpu_cross_lifecycle_settings()
{
   GpuCrossLifecycleSettings cfg;
   cfg.release_after_cross =
       read_env_int_local("UNG_GPU_RELEASE_AFTER_CROSS", 0, 0, 1) == 1;
   return cfg;
}

bool AdditionalEdgesSettings::direct_append_enabled(UngAdditionalEdgesImpl impl) const
{
   return direct_append_requested && impl == UngAdditionalEdgesImpl::CpuExactScan;
}

std::string AdditionalEdgesSettings::summary(UngAdditionalEdgesImpl impl) const
{
   std::ostringstream os;
   os << "direct_append_requested=" << (direct_append_requested ? 1 : 0)
      << " direct_append=" << (direct_append_enabled(impl) ? 1 : 0);
   return os.str();
}

AdditionalEdgesSettings make_additional_edges_settings()
{
   AdditionalEdgesSettings cfg;
   cfg.direct_append_requested =
       read_env_int_local("UNG_ADDITIONAL_DIRECT_APPEND", 0, 0, 1) == 1;
   return cfg;
}

} // namespace ANNS
