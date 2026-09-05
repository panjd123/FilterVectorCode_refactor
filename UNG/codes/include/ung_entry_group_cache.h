#ifndef ANNS_UNG_ENTRY_GROUP_CACHE_H
#define ANNS_UNG_ENTRY_GROUP_CACHE_H

#include "config.h"

#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ANNS
{

struct QueryStats;

struct EntryGroupRouteStats
{
   bool valid = false;
   size_t num_entry_points = 0;
   size_t num_lng_descendants = 0;
   size_t entry_group_matched_points = 0;
   float entry_group_total_coverage = 0.0f;
};

struct CachedEntryGroupResult
{
   std::vector<IdxType> group_ids;
   EntryGroupRouteStats route_stats;
};

void apply_entry_group_route_stats(const EntryGroupRouteStats &route_stats, QueryStats &stats);

class EntryGroupResultCache
{
public:
   using ComputeFn = std::function<CachedEntryGroupResult()>;

   CachedEntryGroupResult get_or_compute(const std::string &key, const ComputeFn &compute);
   size_t size() const;

private:
   mutable std::mutex mutex_;
   std::unordered_map<std::string, std::shared_future<CachedEntryGroupResult>> values_;
};

} // namespace ANNS

#endif // ANNS_UNG_ENTRY_GROUP_CACHE_H
