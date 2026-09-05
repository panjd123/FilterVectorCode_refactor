#ifndef ANNS_UNG_SEARCH_EXECUTION_CONTEXT_H
#define ANNS_UNG_SEARCH_EXECUTION_CONTEXT_H

#include <immintrin.h>

#include "ThreadPool.h"
#include "search_cache.h"

#include <cstdint>

namespace ANNS
{

struct SearchExecutionContext
{
   SearchExecutionContext(uint32_t thread_count, IdxType num_points, IdxType max_lsearch)
       : num_threads(thread_count),
         max_lsearch(max_lsearch),
         search_cache_list(thread_count, num_points, max_lsearch),
         pool(thread_count)
   {
   }

   uint32_t num_threads;
   IdxType max_lsearch;
   SearchCacheList search_cache_list;
   ThreadPool pool;
};

} // namespace ANNS

#endif // ANNS_UNG_SEARCH_EXECUTION_CONTEXT_H
