#include "ung_entry_group_cache.h"
#include "ung_query_stats.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace
{
void expect(bool condition, const char *message)
{
   if (!condition)
   {
      std::cerr << "FAILED: " << message << '\n';
      std::exit(1);
   }
}
} // namespace

int main()
{
   ANNS::EntryGroupResultCache cache;
   std::atomic<int> computes{0};
   std::vector<ANNS::CachedEntryGroupResult> observed(8);
   std::vector<std::thread> workers;
   for (size_t i = 0; i < observed.size(); ++i)
   {
      workers.emplace_back([&, i] {
         observed[i] = cache.get_or_compute("cpu:1,2", [&] {
            computes.fetch_add(1);
            return ANNS::CachedEntryGroupResult{
                {3, 7}, ANNS::EntryGroupRouteStats{true, 2, 11, 29, 0.125f}};
         });
      });
   }
   for (auto &worker : workers)
      worker.join();

   expect(computes.load() == 1, "concurrent identical keys must compute once");
   expect(cache.size() == 1, "successful value must stay cached");
   for (const auto &value : observed)
   {
      expect(value.group_ids == std::vector<ANNS::IdxType>({3, 7}), "cached IDs must be exact");
      expect(value.route_stats.entry_group_matched_points == 29, "cached stats must be exact");
   }

   ANNS::EntryGroupResultCache retry_cache;
   std::atomic<int> attempts{0};
   bool saw_failure = false;
   try
   {
      (void)retry_cache.get_or_compute("retry", [&] {
         attempts.fetch_add(1);
         throw std::runtime_error("expected producer failure");
         return ANNS::CachedEntryGroupResult{};
      });
   }
   catch (const std::runtime_error &)
   {
      saw_failure = true;
   }
   expect(saw_failure, "producer exceptions must reach callers");
   expect(retry_cache.size() == 0, "failed values must not poison the cache");

   const auto retried = retry_cache.get_or_compute("retry", [&] {
      attempts.fetch_add(1);
      return ANNS::CachedEntryGroupResult{{5}, {true, 1, 4, 9, 0.25f}};
   });
   expect(attempts.load() == 2, "a later call must retry after failure");
   expect(retried.group_ids == std::vector<ANNS::IdxType>({5}), "retry result must be returned");
   ANNS::QueryStats stats;
   const ANNS::EntryGroupRouteStats snapshot{true, 2, 11, 29, 0.125f};
   ANNS::apply_entry_group_route_stats(snapshot, stats);
   expect(stats.num_entry_points == 2, "cached entry count must be restored");
   expect(stats.num_lng_descendants == 11, "cached descendant count must be restored");
   expect(stats.entry_group_matched_points == 29, "cached matched points must be restored");
   expect(stats.entry_group_total_coverage == 0.125f, "cached coverage must be restored");



   std::cout << "entry group cache checks passed\n";
   return 0;
}
