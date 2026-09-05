#include "ung_search_execution_context.h"

#include <cstdlib>
#include <iostream>

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
   ANNS::SearchExecutionContext context(2, 128, 64);

   auto first = context.search_cache_list.get_free_cache(8);
   expect(first->search_queue.capacity() == 8, "logical capacity must follow current Lsearch");
   first->search_queue.insert(7, 1.0f);
   first->search_queue.clear();
   auto spare = context.search_cache_list.get_free_cache(8);
   context.search_cache_list.release_cache(first);

   auto second = context.search_cache_list.get_free_cache(32);
   expect(second.get() == first.get(), "released cache should be reused");
   expect(second->search_queue.capacity() == 32, "reused cache must adopt the next Lsearch");
   expect(second->search_queue.size() == 0, "cleared cache must not retain queue state");
   context.search_cache_list.release_cache(second);
   context.search_cache_list.release_cache(spare);

   auto task = context.pool.enqueue([] { return 7; });
   expect(task.get() == 7, "persistent thread pool must execute later work");

   std::cout << "search execution context checks passed\n";
   return 0;
}
