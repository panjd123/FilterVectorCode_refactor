#include "include/ung_build_config.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

namespace ANNS
{
namespace
{

int env_int(const char *key, int fallback, int lo, int hi)
{
   const char *s = std::getenv(key);
   if (!s || !*s)
      return fallback;
   char *end = nullptr;
   long v = std::strtol(s, &end, 10);
   if (end == s || *end != '\0')
      return fallback;
   v = std::max<long>(lo, std::min<long>(hi, v));
   return static_cast<int>(v);
}

bool env_bool(const char *key, bool fallback)
{
   return env_int(key, fallback ? 1 : 0, 0, 1) != 0;
}

std::string env_string(const char *key, const std::string &fallback)
{
   const char *s = std::getenv(key);
   return (s && *s) ? std::string(s) : fallback;
}

UngBuildProfile profile_from_env()
{
   std::string s = env_string("UNG_BUILD_PROFILE", "custom");
   if (s == "original_cpu")
      return UngBuildProfile::OriginalCpu;
   if (s == "current_cpu")
      return UngBuildProfile::CurrentCpu;
   if (s == "naive_gpu")
      return UngBuildProfile::NaiveGpu;
   if (s == "paper_fused")
      return UngBuildProfile::PaperFused;
   return UngBuildProfile::Custom;
}

} // namespace

const char *to_string(UngBuildProfile v)
{
   switch (v)
   {
   case UngBuildProfile::Custom:
      return "custom";
   case UngBuildProfile::OriginalCpu:
      return "original_cpu";
   case UngBuildProfile::CurrentCpu:
      return "current_cpu";
   case UngBuildProfile::NaiveGpu:
      return "naive_gpu";
   case UngBuildProfile::PaperFused:
      return "paper_fused";
   }
   return "unknown";
}

const char *to_string(UngGroupGraphImpl v)
{
   switch (v)
   {
   case UngGroupGraphImpl::VamanaCpu:
      return "vamana_cpu";
   case UngGroupGraphImpl::TagoreCuda:
      return "tagore_cuda";
   case UngGroupGraphImpl::GrnndLikeCuda:
      return "grnnd_like_cuda";
   case UngGroupGraphImpl::FastGrnndCuda:
      return "fast_grnnd_cuda";
   case UngGroupGraphImpl::AdaptiveCuda:
      return "adaptive_cuda";
   }
   return "unknown";
}

const char *to_string(UngGetMinSuperSetsImpl v)
{
   switch (v)
   {
   case UngGetMinSuperSetsImpl::OptimizedBucket:
      return "optimized_bucket";
   case UngGetMinSuperSetsImpl::OriginalSort:
      return "original_sort";
   }
   return "unknown";
}

const char *to_string(UngLngImpl v)
{
   switch (v)
   {
   case UngLngImpl::OptimizedPhase1:
      return "optimized_phase1";
   case UngLngImpl::LegacyAllocating:
      return "legacy_allocating";
   case UngLngImpl::OriginalCpu:
      return "original_cpu";
   }
   return "unknown";
}

const char *to_string(UngDescendantsImpl v)
{
   switch (v)
   {
   case UngDescendantsImpl::OptimizedEpochBfs:
      return "optimized_epoch_bfs";
   case UngDescendantsImpl::LegacyHashBfs:
      return "legacy_hash_bfs";
   }
   return "unknown";
}

const char *to_string(UngCoverageImpl v)
{
   switch (v)
   {
   case UngCoverageImpl::LegacyTopologicalMerge:
      return "legacy_topological_merge";
   case UngCoverageImpl::DescendantsDirect:
      return "descendants_direct";
   }
   return "unknown";
}

const char *to_string(UngCrossEdgeImpl v)
{
   switch (v)
   {
   case UngCrossEdgeImpl::CpuVamana:
      return "cpu_vamana";
   case UngCrossEdgeImpl::GpuBatched:
      return "gpu_batched";
   case UngCrossEdgeImpl::OriginalCpu:
      return "original_cpu";
   case UngCrossEdgeImpl::CpuExactScan:
      return "cpu_exact_scan";
   case UngCrossEdgeImpl::CpuHybridScanVamana:
      return "cpu_hybrid_scan_vamana";
   case UngCrossEdgeImpl::CuvsBruteForce:
      return "cuvs_brute_force";
   }
   return "unknown";
}

const char *to_string(UngAdditionalEdgesImpl v)
{
   switch (v)
   {
   case UngAdditionalEdgesImpl::CpuVamana:
      return "cpu_vamana";
   case UngAdditionalEdgesImpl::Skip:
      return "skip";
   case UngAdditionalEdgesImpl::CpuExactScan:
      return "cpu_exact_scan";
   }
   return "unknown";
}

const char *to_string(UngGpuTopkImpl v)
{
   switch (v)
   {
   case UngGpuTopkImpl::Auto:
      return "auto";
   case UngGpuTopkImpl::CustomNaive:
      return "custom_naive";
   case UngGpuTopkImpl::SgemmTopk:
      return "sgemm_topk";
   case UngGpuTopkImpl::FusedGroupTopk:
      return "fused_group_topk";
   }
   return "unknown";
}

UngBuildConfig UngBuildConfig::from_env(uint32_t build_threads, const std::string &index_name)
{
   UngBuildConfig cfg;
   cfg.profile = profile_from_env();

   if (cfg.profile == UngBuildProfile::OriginalCpu)
   {
      cfg.group_graph_impl = UngGroupGraphImpl::VamanaCpu;
      cfg.get_min_super_sets_impl = UngGetMinSuperSetsImpl::OriginalSort;
      cfg.lng_impl = UngLngImpl::OriginalCpu;
      cfg.descendants_impl = UngDescendantsImpl::LegacyHashBfs;
      cfg.coverage_impl = UngCoverageImpl::LegacyTopologicalMerge;
      cfg.cross_edge_impl = UngCrossEdgeImpl::OriginalCpu;
      cfg.additional_edges_impl = UngAdditionalEdgesImpl::CpuVamana;
      cfg.gpu_topk_impl = UngGpuTopkImpl::Auto;
   }
   else if (cfg.profile == UngBuildProfile::CurrentCpu)
   {
      cfg.group_graph_impl = UngGroupGraphImpl::VamanaCpu;
      cfg.get_min_super_sets_impl = UngGetMinSuperSetsImpl::OptimizedBucket;
      cfg.lng_impl = UngLngImpl::OptimizedPhase1;
      cfg.descendants_impl = UngDescendantsImpl::OptimizedEpochBfs;
      cfg.coverage_impl = UngCoverageImpl::DescendantsDirect;
      cfg.cross_edge_impl = UngCrossEdgeImpl::CpuVamana;
   }
   else if (cfg.profile == UngBuildProfile::NaiveGpu)
   {
      cfg.group_graph_impl = UngGroupGraphImpl::VamanaCpu;
      cfg.get_min_super_sets_impl = UngGetMinSuperSetsImpl::OptimizedBucket;
      cfg.lng_impl = UngLngImpl::OptimizedPhase1;
      cfg.descendants_impl = UngDescendantsImpl::OptimizedEpochBfs;
      cfg.coverage_impl = UngCoverageImpl::DescendantsDirect;
      cfg.cross_edge_impl = UngCrossEdgeImpl::GpuBatched;
      cfg.gpu_topk_impl = UngGpuTopkImpl::SgemmTopk;
   }
   else if (cfg.profile == UngBuildProfile::PaperFused)
   {
      cfg.group_graph_impl = UngGroupGraphImpl::VamanaCpu;
      cfg.get_min_super_sets_impl = UngGetMinSuperSetsImpl::OptimizedBucket;
      cfg.lng_impl = UngLngImpl::OptimizedPhase1;
      cfg.descendants_impl = UngDescendantsImpl::OptimizedEpochBfs;
      cfg.coverage_impl = UngCoverageImpl::DescendantsDirect;
      cfg.cross_edge_impl = UngCrossEdgeImpl::GpuBatched;
      cfg.gpu_topk_impl = UngGpuTopkImpl::FusedGroupTopk;
   }
   else
   {
      (void)index_name;
      int group_impl = env_int("UNG_GROUP_GRAPH_IMPL", 0, 0, 4);
      cfg.group_graph_impl = group_impl == 4 ? UngGroupGraphImpl::AdaptiveCuda
                                             : (group_impl == 3 ? UngGroupGraphImpl::FastGrnndCuda
                                                                : (group_impl == 2 ? UngGroupGraphImpl::GrnndLikeCuda
                                                                                   : (group_impl == 1 ? UngGroupGraphImpl::TagoreCuda
                                                                                                      : UngGroupGraphImpl::VamanaCpu)));
      cfg.get_min_super_sets_impl = env_int("UNG_GET_MIN_SUPER_SETS_IMPL", 0, 0, 1) == 1
                                        ? UngGetMinSuperSetsImpl::OriginalSort
                                        : UngGetMinSuperSetsImpl::OptimizedBucket;
      int lng_impl = env_int("UNG_LNG_IMPL", 0, 0, 2);
      cfg.lng_impl = lng_impl == 2 ? UngLngImpl::OriginalCpu
                                   : (lng_impl == 1 ? UngLngImpl::LegacyAllocating
                                                    : UngLngImpl::OptimizedPhase1);
      cfg.descendants_impl = env_int("UNG_DESCENDANTS_IMPL", 0, 0, 1) == 1 ? UngDescendantsImpl::LegacyHashBfs
                                                                           : UngDescendantsImpl::OptimizedEpochBfs;
      cfg.coverage_impl = env_int("UNG_COVERAGE_IMPL", 0, 0, 1) == 1 ? UngCoverageImpl::DescendantsDirect
                                                                     : UngCoverageImpl::LegacyTopologicalMerge;
      int cross_impl = env_int("UNG_CROSS_EDGE_IMPL", env_int("UNG_CROSS_EDGE_BACKEND", 1, 0, 1), 0, 5);
      cfg.cross_edge_impl = cross_impl == 2 ? UngCrossEdgeImpl::OriginalCpu
                                            : (cross_impl == 3 ? UngCrossEdgeImpl::CpuExactScan
                                                               : (cross_impl == 4 ? UngCrossEdgeImpl::CpuHybridScanVamana
                                                                                  : (cross_impl == 5 ? UngCrossEdgeImpl::CuvsBruteForce
                                                                                                     : (cross_impl == 1 ? UngCrossEdgeImpl::GpuBatched
                                                                                                                        : UngCrossEdgeImpl::CpuVamana))));
      int add_impl = env_int("UNG_ADDITIONAL_EDGES_IMPL", 0, 0, 2);
      cfg.additional_edges_impl = add_impl == 2 ? UngAdditionalEdgesImpl::CpuExactScan
                                                : (add_impl == 1 ? UngAdditionalEdgesImpl::Skip
                                                                 : UngAdditionalEdgesImpl::CpuVamana);
      cfg.gpu_topk_impl = static_cast<UngGpuTopkImpl>(env_int("UNG_GPU_TOPK_IMPL", 0, 0, 3));
   }

   cfg.gpu_strict = env_bool("UNG_CROSS_EDGE_GPU_STRICT", false);
   cfg.coverage_threads = static_cast<uint32_t>(env_int("UNG_COVERAGE_THREADS", static_cast<int>(build_threads), 1,
                                                        std::max<int>(1, static_cast<int>(build_threads))));
   const bool adaptive_group_graph = cfg.group_graph_impl == UngGroupGraphImpl::AdaptiveCuda;
   cfg.tagore_min_group_size = static_cast<uint32_t>(
       env_int("UNG_TAGORE_MIN_GROUP_SIZE", adaptive_group_graph ? 128 : 0, 0, 1 << 30));
   cfg.tagore_k = static_cast<uint32_t>(env_int("UNG_TAGORE_K", 64, 1, 1024));
   const bool grnnd_style = cfg.group_graph_impl == UngGroupGraphImpl::GrnndLikeCuda ||
                            cfg.group_graph_impl == UngGroupGraphImpl::FastGrnndCuda ||
                            cfg.group_graph_impl == UngGroupGraphImpl::AdaptiveCuda;
   const uint32_t default_tagore_iter = grnnd_style ? 4u : 10u;
   cfg.tagore_iter = static_cast<uint32_t>(env_int("UNG_TAGORE_ITER", static_cast<int>(default_tagore_iter), 1, 1000));
   cfg.tagore_m = static_cast<uint32_t>(env_int("UNG_TAGORE_M", 64, 1, 1024));
   return cfg;
}

void UngBuildConfig::print(std::ostream &os) const
{
   os << "[UNG config] profile=" << to_string(profile) << '\n'
      << "[UNG config] group_graph_impl=" << to_string(group_graph_impl) << '\n'
      << "[UNG config] get_min_super_sets_impl=" << to_string(get_min_super_sets_impl) << '\n'
      << "[UNG config] lng_impl=" << to_string(lng_impl) << '\n'
      << "[UNG config] descendants_impl=" << to_string(descendants_impl) << '\n'
      << "[UNG config] coverage_impl=" << to_string(coverage_impl) << '\n'
      << "[UNG config] cross_edge_impl=" << to_string(cross_edge_impl) << '\n'
      << "[UNG config] additional_edges_impl=" << to_string(additional_edges_impl) << '\n'
      << "[UNG config] gpu_topk_impl=" << to_string(gpu_topk_impl) << '\n'
      << "[UNG config] gpu_strict=" << (gpu_strict ? 1 : 0) << '\n'
      << "[UNG config] coverage_threads=" << coverage_threads << '\n'
      << "[UNG config] tagore_min_group_size=" << tagore_min_group_size << '\n'
      << "[UNG config] tagore_k=" << tagore_k << '\n'
      << "[UNG config] tagore_iter=" << tagore_iter << '\n'
      << "[UNG config] tagore_m=" << tagore_m << std::endl;
}

} // namespace ANNS
