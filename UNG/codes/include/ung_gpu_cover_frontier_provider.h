#ifndef ANNS_UNG_GPU_COVER_FRONTIER_PROVIDER_H
#define ANNS_UNG_GPU_COVER_FRONTIER_PROVIDER_H

#include "config.h"
#include "ung_entry_group.h"
#include "ung_query_stats.h"

#include <memory>
#include <vector>

namespace ANNS
{

struct QueryRouteDecision;

struct GpuCoverFrontierBuildInput
{
   const std::vector<std::vector<LabelType>> *group_labels = nullptr;
   const std::vector<std::vector<IdxType>> *lng_descendants = nullptr;
   IdxType num_groups = 0;
};

class GpuCoverFrontierProvider
{
public:
   explicit GpuCoverFrontierProvider(const GpuCoverFrontierBuildInput &input);
   ~GpuCoverFrontierProvider();

   GpuCoverFrontierProvider(const GpuCoverFrontierProvider &) = delete;
   GpuCoverFrontierProvider &operator=(const GpuCoverFrontierProvider &) = delete;

   void reserve_workspaces(size_t workspace_count, size_t max_query_labels);
   EntryGroupProviderResult run(const EntryGroupProviderRequest &request, QueryStats &stats) const;

private:
   struct Impl;
   std::unique_ptr<Impl> impl_;
};

} // namespace ANNS

#endif // ANNS_UNG_GPU_COVER_FRONTIER_PROVIDER_H
