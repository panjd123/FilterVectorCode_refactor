#include "include/ung_entry_group_cache.h"

#include <memory>
#include <utility>

namespace ANNS
{



CachedEntryGroupResult EntryGroupResultCache::get_or_compute(const std::string &key,
                                                             const ComputeFn &compute)
{
   std::shared_future<CachedEntryGroupResult> future;
   std::shared_ptr<std::promise<CachedEntryGroupResult>> producer;
   {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto cached = values_.find(key);
      if (cached != values_.end())
      {
         future = cached->second;
      }
      else
      {
         producer = std::make_shared<std::promise<CachedEntryGroupResult>>();
         future = producer->get_future().share();
         values_.emplace(key, future);
      }
   }

   if (producer)
   {
      try
      {
         producer->set_value(compute());
      }
      catch (...)
      {
         producer->set_exception(std::current_exception());
         std::lock_guard<std::mutex> lock(mutex_);
         values_.erase(key);
      }
   }

   return future.get();
}

size_t EntryGroupResultCache::size() const
{
   std::lock_guard<std::mutex> lock(mutex_);
   return values_.size();
}

} // namespace ANNS
