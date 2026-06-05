#ifndef ANNS_UNG_BUILD_CONFIG_H
#define ANNS_UNG_BUILD_CONFIG_H

#include <cstdint>
#include <iosfwd>
#include <string>

namespace ANNS
{

enum class UngBuildProfile : int
{
   Custom = 0,
   OriginalCpu = 1,
   CurrentCpu = 2,
   NaiveGpu = 3,
   PaperFused = 4,
};

enum class UngGroupGraphImpl : int
{
   VamanaCpu = 0,
   TagoreCuda = 1,
   GrnndLikeCuda = 2,
   FastGrnndCuda = 3,
   AdaptiveCuda = 4,
};

enum class UngGetMinSuperSetsImpl : int
{
   OptimizedBucket = 0,
   OriginalSort = 1,
};

enum class UngLngImpl : int
{
   OptimizedPhase1 = 0,
   LegacyAllocating = 1,
   OriginalCpu = 2,
};

enum class UngDescendantsImpl : int
{
   OptimizedEpochBfs = 0,
   LegacyHashBfs = 1,
};

enum class UngCoverageImpl : int
{
   LegacyTopologicalMerge = 0,
   DescendantsDirect = 1,
};

enum class UngCrossEdgeImpl : int
{
   CpuVamana = 0,
   GpuBatched = 1,
   OriginalCpu = 2,
   CpuExactScan = 3,
   CpuHybridScanVamana = 4,
   CuvsBruteForce = 5,
};

enum class UngAdditionalEdgesImpl : int
{
   CpuVamana = 0,
   Skip = 1,
   CpuExactScan = 2,
};

enum class UngGpuTopkImpl : int
{
   Auto = 0,
   CustomNaive = 1,
   SgemmTopk = 2,
   FusedGroupTopk = 3,
};

struct UngBuildConfig
{
   UngBuildProfile profile = UngBuildProfile::Custom;
   UngGroupGraphImpl group_graph_impl = UngGroupGraphImpl::VamanaCpu;
   UngGetMinSuperSetsImpl get_min_super_sets_impl = UngGetMinSuperSetsImpl::OptimizedBucket;
   UngLngImpl lng_impl = UngLngImpl::OptimizedPhase1;
   UngDescendantsImpl descendants_impl = UngDescendantsImpl::OptimizedEpochBfs;
   UngCoverageImpl coverage_impl = UngCoverageImpl::LegacyTopologicalMerge;
   UngCrossEdgeImpl cross_edge_impl = UngCrossEdgeImpl::GpuBatched;
   UngAdditionalEdgesImpl additional_edges_impl = UngAdditionalEdgesImpl::CpuVamana;
   UngGpuTopkImpl gpu_topk_impl = UngGpuTopkImpl::Auto;

   bool gpu_strict = false;
   uint32_t coverage_threads = 0; // 0 means use build num_threads.
   uint32_t tagore_min_group_size = 0;
   uint32_t tagore_k = 64;
   uint32_t tagore_iter = 10; // GRNND-style paths default to 4 in from_env unless UNG_TAGORE_ITER is set.
   uint32_t tagore_m = 64;

   static UngBuildConfig from_env(uint32_t build_threads, const std::string &index_name);
   bool is_original_cpu_pipeline() const { return profile == UngBuildProfile::OriginalCpu; }
   void print(std::ostream &os) const;
};

const char *to_string(UngBuildProfile v);
const char *to_string(UngGroupGraphImpl v);
const char *to_string(UngGetMinSuperSetsImpl v);
const char *to_string(UngLngImpl v);
const char *to_string(UngDescendantsImpl v);
const char *to_string(UngCoverageImpl v);
const char *to_string(UngCrossEdgeImpl v);
const char *to_string(UngAdditionalEdgesImpl v);
const char *to_string(UngGpuTopkImpl v);

} // namespace ANNS

#endif // ANNS_UNG_BUILD_CONFIG_H
