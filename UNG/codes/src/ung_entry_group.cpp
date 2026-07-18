#include "include/ung_entry_group.h"

#include <stdexcept>
#include <utility>

namespace ANNS
{

const char *entry_group_provider_impl_name(EntryGroupProviderImpl impl)
{
   switch (impl)
   {
   case EntryGroupProviderImpl::CpuMinSuperSets:
      return "cpu_min_super_sets";
   case EntryGroupProviderImpl::GpuCoverFrontier:
      return "gpu_cover_frontier";
   case EntryGroupProviderImpl::CpuBruteForceEls:
      return "cpu_bruteforce_els";
   }
   return "unknown";
}

const char *entry_group_provider_kind_name(EntryGroupProviderKind kind)
{
   switch (kind)
   {
   case EntryGroupProviderKind::Passthrough:
      return "passthrough";
   case EntryGroupProviderKind::CpuMinSuperSets:
      return "cpu_min_super_sets";
   case EntryGroupProviderKind::CpuExpanded:
      return "cpu_expanded";
   case EntryGroupProviderKind::GpuCoverFrontier:
      return "gpu_cover_frontier";
   case EntryGroupProviderKind::CpuBruteForceEls:
      return "cpu_bruteforce_els";
   }
   return "unknown";
}

EntryGroupProviderImpl parse_entry_group_provider_impl(const std::string &value)
{
   if (value == "0" || value == "cpu" || value == "cpu_min_super_sets")
      return EntryGroupProviderImpl::CpuMinSuperSets;
   if (value == "1" || value == "gpu" || value == "gpu_cover_frontier")
      return EntryGroupProviderImpl::GpuCoverFrontier;
   if (value == "2" || value == "cpu_bruteforce" || value == "cpu_bruteforce_els" || value == "bruteforce_els")
      return EntryGroupProviderImpl::CpuBruteForceEls;
   throw std::invalid_argument(
       "Invalid entry_group_provider: " + value +
       " (expected cpu_min_super_sets/cpu/0, gpu_cover_frontier/gpu/1, or cpu_bruteforce_els/2)");
}

void EntryGroupProviderRequest::validate() const
{
   if (query_labels == nullptr)
      throw std::invalid_argument("EntryGroupProviderRequest requires query_labels");
   if (decision == nullptr)
      throw std::invalid_argument("EntryGroupProviderRequest requires decision");
}

bool EntryGroupProviderRequest::has_current_group_ids() const
{
   return current_group_ids != nullptr;
}

IdxType EntryGroupProviderRequest::true_group_id_or(IdxType fallback) const
{
   if (true_query_group_ids == nullptr || query_id >= true_query_group_ids->size())
      return fallback;
   return (*true_query_group_ids)[query_id];
}

void EntryGroupProviderResult::mark_fallback(EntryGroupProviderImpl requested, std::string reason)
{
   requested_impl = requested;
   fallback_used = true;
   fallback_reason = std::move(reason);
}

SearchEntryProvider::SearchEntryProvider(CpuProviderFn cpu_provider)
    : cpu_provider_(std::move(cpu_provider))
{
   if (!cpu_provider_)
      throw std::invalid_argument("SearchEntryProvider requires a CPU provider callback");
}

SearchEntryProvider::SearchEntryProvider(CpuProviderFn cpu_provider, GpuProviderFn gpu_provider)
    : cpu_provider_(std::move(cpu_provider)), gpu_provider_(std::move(gpu_provider))
{
   if (!cpu_provider_)
      throw std::invalid_argument("SearchEntryProvider requires a CPU provider callback");
}

EntryGroupProviderResult SearchEntryProvider::run(const EntryGroupProviderRequest &request,
                                                  QueryStats &stats) const
{
   request.validate();
   EntryGroupProviderResult result;
   result.requested_impl = request.impl;
   switch (request.impl)
   {
   case EntryGroupProviderImpl::CpuMinSuperSets:
      result = cpu_provider_(request, stats);
      result.requested_impl = request.impl;
      break;
   case EntryGroupProviderImpl::CpuBruteForceEls:
      result = cpu_provider_(request, stats);
      result.requested_impl = request.impl;
      break;
   case EntryGroupProviderImpl::GpuCoverFrontier:
      if (gpu_provider_)
      {
         result = gpu_provider_(request, stats);
         result.requested_impl = request.impl;
      }
      else
      {
         // Some test or legacy call sites may construct SearchEntryProvider
         // without a GPU callback. Preserve search semantics by using CPU.
         result = cpu_provider_(request, stats);
         result.mark_fallback(request.impl, "gpu_cover_frontier callback is not configured");
      }
      break;
   }
   return result;
}

} // namespace ANNS
