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

struct CachedEntryGroupResult
{
   std::vector<IdxType> group_ids;
};


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
