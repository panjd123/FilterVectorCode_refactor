#include "include/ung_build_config.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
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

UngGroupGraphImpl group_graph_impl_from_int(int value)
{
   switch (value)
   {
   case 1:
      return UngGroupGraphImpl::TagoreCuda;
   case 2:
      return UngGroupGraphImpl::GrnndLikeCuda;
   case 3:
      return UngGroupGraphImpl::FastGrnndCuda;
   case 4:
      return UngGroupGraphImpl::AdaptiveCuda;
   default:
      return UngGroupGraphImpl::VamanaCpu;
   }
}

UngLngImpl lng_impl_from_int(int value)
{
   switch (value)
   {
   case 1:
      return UngLngImpl::LegacyAllocating;
   case 2:
      return UngLngImpl::OriginalCpu;
   default:
      return UngLngImpl::OptimizedPhase1;
   }
}

UngCrossEdgeImpl cross_edge_impl_from_int(int value)
{
   switch (value)
   {
   case 1:
      return UngCrossEdgeImpl::GpuBatched;
   case 2:
      return UngCrossEdgeImpl::OriginalCpu;
   case 3:
      return UngCrossEdgeImpl::CpuExactScan;
   case 4:
      return UngCrossEdgeImpl::CpuHybridScanVamana;
   case 5:
      return UngCrossEdgeImpl::CuvsBruteForce;
   default:
      return UngCrossEdgeImpl::CpuVamana;
   }
}

UngAdditionalEdgesImpl additional_edges_impl_from_int(int value)
{
   switch (value)
   {
   case 1:
      return UngAdditionalEdgesImpl::Skip;
   case 2:
      return UngAdditionalEdgesImpl::CpuExactScan;
   default:
      return UngAdditionalEdgesImpl::CpuVamana;
   }
}

UngGpuTopkImpl gpu_topk_impl_from_int(int value)
{
   switch (value)
   {
   case 1:
      return UngGpuTopkImpl::CustomNaive;
   case 2:
      return UngGpuTopkImpl::SgemmTopk;
   case 3:
      return UngGpuTopkImpl::FusedGroupTopk;
   default:
      return UngGpuTopkImpl::Auto;
   }
}

void apply_profile_defaults(UngBuildConfig &cfg)
{
   switch (cfg.profile)
   {
   case UngBuildProfile::OriginalCpu:
      cfg.group_graph_impl = UngGroupGraphImpl::VamanaCpu;
      cfg.get_min_super_sets_impl = UngGetMinSuperSetsImpl::OriginalSort;
      cfg.lng_impl = UngLngImpl::OriginalCpu;
      cfg.descendants_impl = UngDescendantsImpl::LegacyHashBfs;
      cfg.coverage_impl = UngCoverageImpl::LegacyTopologicalMerge;
      cfg.cross_edge_impl = UngCrossEdgeImpl::OriginalCpu;
      cfg.additional_edges_impl = UngAdditionalEdgesImpl::CpuVamana;
      cfg.gpu_topk_impl = UngGpuTopkImpl::Auto;
      break;
   case UngBuildProfile::CurrentCpu:
      cfg.group_graph_impl = UngGroupGraphImpl::VamanaCpu;
      cfg.get_min_super_sets_impl = UngGetMinSuperSetsImpl::OptimizedBucket;
      cfg.lng_impl = UngLngImpl::OptimizedPhase1;
      cfg.descendants_impl = UngDescendantsImpl::OptimizedEpochBfs;
      cfg.coverage_impl = UngCoverageImpl::DescendantsDirect;
      cfg.cross_edge_impl = UngCrossEdgeImpl::CpuVamana;
      break;
   case UngBuildProfile::NaiveGpu:
      cfg.group_graph_impl = UngGroupGraphImpl::VamanaCpu;
      cfg.get_min_super_sets_impl = UngGetMinSuperSetsImpl::OptimizedBucket;
      cfg.lng_impl = UngLngImpl::OptimizedPhase1;
      cfg.descendants_impl = UngDescendantsImpl::OptimizedEpochBfs;
      cfg.coverage_impl = UngCoverageImpl::DescendantsDirect;
      cfg.cross_edge_impl = UngCrossEdgeImpl::GpuBatched;
      cfg.gpu_topk_impl = UngGpuTopkImpl::SgemmTopk;
      break;
   case UngBuildProfile::PaperFused:
      cfg.group_graph_impl = UngGroupGraphImpl::VamanaCpu;
      cfg.get_min_super_sets_impl = UngGetMinSuperSetsImpl::OptimizedBucket;
      cfg.lng_impl = UngLngImpl::OptimizedPhase1;
      cfg.descendants_impl = UngDescendantsImpl::OptimizedEpochBfs;
      cfg.coverage_impl = UngCoverageImpl::DescendantsDirect;
      cfg.cross_edge_impl = UngCrossEdgeImpl::GpuBatched;
      cfg.gpu_topk_impl = UngGpuTopkImpl::FusedGroupTopk;
      break;
   case UngBuildProfile::Custom:
      break;
   }
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

UngBuildConfig UngBuildConfig::from_env(uint32_t build_threads)
{
   UngBuildConfig cfg;
   cfg.profile = profile_from_env();
   apply_profile_defaults(cfg);

   if (cfg.profile == UngBuildProfile::Custom)
   {
      cfg.group_graph_impl = group_graph_impl_from_int(env_int("UNG_GROUP_GRAPH_IMPL", 0, 0, 4));
      cfg.get_min_super_sets_impl = env_int("UNG_GET_MIN_SUPER_SETS_IMPL", 0, 0, 1) == 1
                                        ? UngGetMinSuperSetsImpl::OriginalSort
                                        : UngGetMinSuperSetsImpl::OptimizedBucket;
      cfg.lng_impl = lng_impl_from_int(env_int("UNG_LNG_IMPL", 0, 0, 2));
      cfg.descendants_impl = env_int("UNG_DESCENDANTS_IMPL", 0, 0, 1) == 1 ? UngDescendantsImpl::LegacyHashBfs
                                                                           : UngDescendantsImpl::OptimizedEpochBfs;
      cfg.coverage_impl = env_int("UNG_COVERAGE_IMPL", 0, 0, 1) == 1 ? UngCoverageImpl::DescendantsDirect
                                                                     : UngCoverageImpl::LegacyTopologicalMerge;
      cfg.cross_edge_impl =
          cross_edge_impl_from_int(env_int("UNG_CROSS_EDGE_IMPL", env_int("UNG_CROSS_EDGE_BACKEND", 1, 0, 1), 0, 5));
      cfg.additional_edges_impl = additional_edges_impl_from_int(env_int("UNG_ADDITIONAL_EDGES_IMPL", 0, 0, 2));
      cfg.gpu_topk_impl = gpu_topk_impl_from_int(env_int("UNG_GPU_TOPK_IMPL", 0, 0, 3));
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
   cfg.special_blocks_enabled = env_bool("UNG_SPECIAL_BLOCKS", false);
   cfg.special_block_min_points =
       static_cast<uint32_t>(env_int("UNG_SPECIAL_BLOCK_MIN_POINTS", 100, 1, 1 << 30));
   cfg.special_block_max_degree =
       static_cast<uint32_t>(env_int("UNG_SPECIAL_BLOCK_MAX_DEGREE", 0, 0, 1 << 20));
   cfg.special_block_num_cross_edges =
       static_cast<uint32_t>(env_int("UNG_SPECIAL_BLOCK_NUM_CROSS_EDGES", 0, 0, 1 << 20));
   cfg.special_block_data_mode = env_string("UNG_SPECIAL_BLOCK_DATA_MODE", "");
   cfg.special_block_skip_trivial = env_bool("UNG_SPECIAL_BLOCK_SKIP_TRIVIAL", true);
   if (cfg.special_blocks_enabled && cfg.special_block_data_mode != "x1")
   {
      std::cerr
          << "UNG_SPECIAL_BLOCKS=1 is only valid for explicit x1/original datasets. "
          << "Set UNG_SPECIAL_BLOCK_DATA_MODE=x1 after verifying the input is not a repeated xN dataset."
          << std::endl;
      throw std::runtime_error(
          "UNG_SPECIAL_BLOCKS=1 is only valid for explicit x1/original datasets. "
          "Set UNG_SPECIAL_BLOCK_DATA_MODE=x1 after verifying the input is not a repeated xN dataset.");
   }
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
      << "[UNG config] tagore_m=" << tagore_m << '\n'
      << "[UNG config] special_blocks_enabled=" << (special_blocks_enabled ? 1 : 0) << '\n'
      << "[UNG config] special_block_min_points=" << special_block_min_points << '\n'
      << "[UNG config] special_block_data_mode=" << special_block_data_mode << '\n'
      << "[UNG config] special_block_skip_trivial=" << (special_block_skip_trivial ? 1 : 0) << std::endl;
}

} // namespace ANNS
