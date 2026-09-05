#ifndef ANNS_UNG_ENTRY_GROUP_H
#define ANNS_UNG_ENTRY_GROUP_H

#include "config.h"
#include "ung_entry_group_cache.h"

#include <functional>
#include <string>
#include <vector>

namespace ANNS
{

struct QueryRouteDecision;
struct QueryStats;

enum class SelectionMode
{
   SizeOnly,
   SizeAndDistance,
};

enum class EntryGroupProviderImpl : int
{
   CpuMinSuperSets = 0,
   GpuCoverFrontier = 1,
   CpuBruteForceEls = 2,
   CpuBruteForceElsScalar = 3,
   SpecialBlockTrie = 4,
};

enum class EntryGroupProviderKind : int
{
   Passthrough = 0,
   CpuMinSuperSets = 1,
   CpuExpanded = 2,
   GpuCoverFrontier = 3,
   CpuBruteForceEls = 4,
   CpuBruteForceElsScalar = 5,
   SpecialBlockTrie = 6,
};

const char *entry_group_provider_impl_name(EntryGroupProviderImpl impl);
const char *entry_group_provider_kind_name(EntryGroupProviderKind kind);
EntryGroupProviderImpl parse_entry_group_provider_impl(const std::string &value);
std::string make_entry_group_label_cache_key(EntryGroupProviderImpl impl,
                                             const std::vector<LabelType> &query_labels,
                                             bool recursive_more_start,
                                             bool ung_more_entry,
                                             size_t scalar_els_cap);

struct EntryGroupProviderRequest
{
   EntryGroupProviderImpl impl = EntryGroupProviderImpl::CpuMinSuperSets;
   const std::vector<LabelType> *query_labels = nullptr;
   IdxType query_id = 0;
   const QueryRouteDecision *decision = nullptr;
   bool recursive_more_start = false;
   bool ung_more_entry = false;
   const std::vector<IdxType> *true_query_group_ids = nullptr;
   const std::vector<IdxType> *current_group_ids = nullptr;
   size_t scalar_els_cap = 0;
   bool cache_query_label_results = false;

   void validate() const;
   bool has_current_group_ids() const;
   IdxType true_group_id_or(IdxType fallback = 0) const;
};

struct EntryGroupProviderResult
{
   EntryGroupProviderImpl requested_impl = EntryGroupProviderImpl::CpuMinSuperSets;
   EntryGroupProviderKind provider = EntryGroupProviderKind::Passthrough;
   std::vector<IdxType> group_ids;
   EntryGroupRouteStats route_stats;
   bool coverage_correct = true;
   bool exact_minimal = false;
   bool fallback_used = false;
   std::string fallback_reason;
   double elapsed_ms = 0.0;

   const char *provider_name() const
   {
      return entry_group_provider_kind_name(provider);
   }

   void mark_fallback(EntryGroupProviderImpl requested, std::string reason);
};

class SearchEntryProvider
{
public:
   using CpuProviderFn = std::function<EntryGroupProviderResult(const EntryGroupProviderRequest &request,
                                                                QueryStats &stats)>;
   using GpuProviderFn = std::function<EntryGroupProviderResult(const EntryGroupProviderRequest &request,
                                                                QueryStats &stats)>;

   explicit SearchEntryProvider(CpuProviderFn cpu_provider);
   SearchEntryProvider(CpuProviderFn cpu_provider, GpuProviderFn gpu_provider);

   EntryGroupProviderResult run(const EntryGroupProviderRequest &request, QueryStats &stats) const;

private:
   CpuProviderFn cpu_provider_;
   GpuProviderFn gpu_provider_;
};

} // namespace ANNS

#endif // ANNS_UNG_ENTRY_GROUP_H
