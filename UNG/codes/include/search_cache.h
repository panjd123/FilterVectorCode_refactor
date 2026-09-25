#ifndef SEARCH_CACHE_H
#define SEARCH_CACHE_H

#include <mutex>
#include <deque>
#include <memory>
#include "visited_set.h"
#include "search_queue.h"
#include "ung_special_candidate_queue.h"

namespace ANNS
{

   struct SearchCache
   {
      SearchQueue search_queue;
      VisitedSet visited_set;
      VisitedSet special_visited_regular;
      std::vector<std::unique_ptr<VisitedSet>> special_visited_by_level;
      SpecialCandidateQueue special_candidate_queue;
      std::vector<Candidate> expanded_list;
      std::vector<float> occlude_factor;
      std::vector<uint8_t> favor_target_map;
      std::vector<IdxType> favor_target_touched;
      std::vector<uint8_t> special_free_state_cache;
      std::vector<IdxType> special_free_state_touched;
      std::vector<uint8_t> favor_selected_blocks;
      std::vector<uint32_t> favor_block_source_expanded;
      std::vector<uint8_t> favor_candidate_active;
      std::vector<IdxType> gpu_distance_ids;
      std::vector<float> gpu_distance_values;

      SearchCache(IdxType visited_set_size, int32_t search_queue_capacity)
      {
         search_queue.reserve(search_queue_capacity);
         visited_set.init(visited_set_size);
         special_visited_regular.init(visited_set_size);
      }

      VisitedSet &special_visited(size_t activation_level, IdxType visited_set_size)
      {
         if (activation_level == 0)
            return special_visited_regular;
         while (special_visited_by_level.size() < activation_level)
         {
            auto visited = std::make_unique<VisitedSet>();
            visited->init(visited_set_size);
            special_visited_by_level.push_back(std::move(visited));
         }
         return *special_visited_by_level[activation_level - 1];
      }
   };

   class SearchCacheList
   {
   public:
      SearchCacheList(uint32_t num_cache, IdxType visited_set_size, int32_t search_queue_capacity)
      {
         _visited_set_size = visited_set_size;
         _search_queue_capacity = search_queue_capacity;
         for (uint32_t i = 0; i < num_cache; i++)
            pool.emplace_back(std::make_shared<SearchCache>(visited_set_size, search_queue_capacity));
      }

      std::shared_ptr<SearchCache> get_free_cache(int32_t requested_capacity = -1)
      {
         std::unique_lock<std::mutex> lock(pool_guard);
         const int32_t capacity = requested_capacity >= 0 ? requested_capacity : _search_queue_capacity;
         if (pool.empty())
            return std::make_shared<SearchCache>(_visited_set_size, capacity);
         auto cache = pool.front();
         pool.pop_front();
         cache->search_queue.reserve(capacity);
         return cache;
      }

      void release_cache(std::shared_ptr<SearchCache> cache)
      {
         std::unique_lock<std::mutex> lock(pool_guard);
         pool.push_back(cache);
      }

      ~SearchCacheList() = default;
      int32_t _search_queue_capacity;

   private:
      std::deque<std::shared_ptr<SearchCache>> pool;
      std::mutex pool_guard;
      IdxType _visited_set_size;
      // int32_t _search_queue_capacity;
   };
}

#endif // SEARCH_CACHE_H
