#ifndef ANNS_UNG_ENTRY_GROUP_H
#define ANNS_UNG_ENTRY_GROUP_H

#include "config.h"
#include "ung_entry_group_cache.h"

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

// Entry discovery is independent from hierarchy depth and group topology.
// Backend details such as bitsets or GPU kernels are implementation choices,
// not additional public strategies.
enum class EntryGroupStrategy : uint8_t
{
   Original = 0,
   OptimizedLng = 1,
   Trie = 2,
};

enum class EntryGroupProviderKind : int
{
   Passthrough = 0,
   Original = 1,
   OptimizedLng = 2,
   Trie = 3,
};

const char *entry_group_strategy_name(EntryGroupStrategy strategy);
const char *entry_group_provider_kind_name(EntryGroupProviderKind kind);
EntryGroupStrategy parse_entry_group_strategy(const std::string &value);
std::string make_entry_group_label_cache_key(EntryGroupStrategy strategy,
                                             const std::vector<LabelType> &query_labels,
                                             bool recursive_more_start,
                                             bool ung_more_entry);

struct EntryGroupProviderRequest
{
   EntryGroupStrategy strategy = EntryGroupStrategy::OptimizedLng;
   const std::vector<LabelType> *query_labels = nullptr;
   IdxType query_id = 0;
   const QueryRouteDecision *decision = nullptr;
   bool recursive_more_start = false;
   bool ung_more_entry = false;
   const std::vector<IdxType> *true_query_group_ids = nullptr;
   const std::vector<IdxType> *current_group_ids = nullptr;
   bool cache_query_label_results = false;

   void validate() const;
   bool has_current_group_ids() const;
   IdxType true_group_id_or(IdxType fallback = 0) const;
};

struct EntryGroupProviderResult
{
   EntryGroupStrategy requested_strategy = EntryGroupStrategy::OptimizedLng;
   EntryGroupProviderKind provider = EntryGroupProviderKind::Passthrough;
   std::vector<IdxType> group_ids;
   EntryGroupRouteStats route_stats;
   bool coverage_correct = true;
   bool exact_minimal = false;
   double elapsed_ms = 0.0;

   const char *provider_name() const
   {
      return entry_group_provider_kind_name(provider);
   }
};

} // namespace ANNS

#endif // ANNS_UNG_ENTRY_GROUP_H
