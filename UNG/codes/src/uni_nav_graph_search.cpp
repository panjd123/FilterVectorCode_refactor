#include "include/uni_nav_graph.h"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <ThreadPool.h>
#include <omp.h>

namespace ANNS
{

   void UniNavGraph::thread_function(IdxType query_id,
                                     const SearchRuntimeConfig &runtime,
                                     const GraphSearchBackend &graph_backend,
                                     SearchCacheList &search_cache_list,
                                     std::pair<IdxType, float> *results,
                                     std::vector<float> &num_cmps,
                                     std::vector<QueryStats> &query_stats,
                                     const std::vector<IdxType> &true_query_group_ids)
   {
      omp_set_num_threads(1);

      const IdxType id = query_id;

      auto &stats = query_stats[id];
      auto total_search_start_time = std::chrono::high_resolution_clock::now();

      const char *query = _query_storage->get_vector(id);
      SearchQueue cur_result;
      cur_result.reserve(runtime.K);
      const auto &query_labels = _query_storage->get_label_set(id);

      std::vector<IdxType> entry_group_ids;
      QueryRouteDecision decision = decide_query_route(query_labels, runtime.idea2_available,
                                                       runtime.use_new_trie_method, runtime.recursive_more_start,
                                                       runtime.force_use_alg, runtime.bfs_filter,
                                                       entry_group_ids, stats);

      const EntryGroupProviderRequest entry_request{
          runtime.entry_group_provider,
          &query_labels,
          static_cast<IdxType>(id),
          &decision,
          runtime.recursive_more_start,
          runtime.ung_more_entry,
          &true_query_group_ids,
          &entry_group_ids};
      prepare_entry_groups_for_execution(entry_request, entry_group_ids, stats);
      populate_special_query_stats(query_labels, stats);

      if (decision.uses_acorn())
      {
         const bool use_old_bitmap_search = decision.use_old_bitmap_search(runtime.force_use_alg);
         if (!execute_acorn_query(query, query_labels, entry_group_ids, id, runtime.K, runtime.Lsearch,
                                  runtime.lsearch_start, runtime.lsearch_step, runtime.efs_start,
                                  runtime.efs_step_slow, runtime.efs_step_fast, runtime.lsearch_threshold,
                                  runtime.force_use_alg, decision.algorithm, runtime.bfs_filter,
                                  use_old_bitmap_search, results, num_cmps, cur_result, stats))
         {
            return;
         }
      }
      else
      {
         auto search_cache = search_cache_list.get_free_cache();
         const bool search_ok =
             (runtime.special_block_search && !_special_blocks.empty())
                 ? execute_special_block_ung_query(query, search_cache, runtime, graph_backend,
                                                   entry_group_ids, query_labels, id,
                                                   num_cmps, cur_result, stats)
                 : execute_ung_query(query, search_cache, graph_backend,
                                     entry_group_ids, id,
                                     runtime.num_entry_points, num_cmps,
                                     cur_result, stats);
         search_cache_list.release_cache(search_cache);
         if (!search_ok)
         {
            return;
         }
      }

      for (auto k = 0; k < runtime.K; ++k)
      {
         if (k < cur_result.size())
         {
            results[id * runtime.K + k].first = _new_to_old_vec_ids[cur_result[k].id];
            results[id * runtime.K + k].second = cur_result[k].distance;
         }
         else
         {
            results[id * runtime.K + k].first = -1;
         }
      }

      stats.time_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - total_search_start_time)
              .count();
   }

   void UniNavGraph::search_hybrid(std::shared_ptr<IStorage> &query_storage,
                                   std::shared_ptr<DistanceHandler> &distance_handler,
                                   const SearchRuntimeConfig &runtime,
                                   std::pair<IdxType, float> *results,
                                   std::vector<float> &num_cmps,
                                   std::vector<QueryStats> &query_stats,
                                   const std::vector<IdxType> &true_query_group_ids)
   {
      auto num_queries = query_storage->get_num_points();
      _query_storage = query_storage;
      _distance_handler = distance_handler;
      _scenario = runtime.scenario;
      query_stats.resize(num_queries);

      if (runtime.K > runtime.Lsearch)
      {
         std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
         exit(-1);
      }

      CrossEdgeCsrOutput csr_graph;
      const bool use_csr_graph_backend = runtime.graph_backend == SearchGraphBackendImpl::Csr;
      if (use_csr_graph_backend)
         csr_graph = build_search_graph_csr();
      const GraphSearchBackend graph_backend =
          use_csr_graph_backend ? GraphSearchBackend(csr_graph) : GraphSearchBackend(*_graph);

      SearchCacheList search_cache_list(runtime.num_threads, _num_points, runtime.Lsearch);
      ThreadPool pool(runtime.num_threads);
      std::vector<std::future<void>> tp_results;
      tp_results.reserve(num_queries);
      for (auto id = 0; id < num_queries; ++id)
      {
         tp_results.emplace_back(
             pool.enqueue([this, id, &runtime, &graph_backend, &search_cache_list,
                           results, &num_cmps, &query_stats, &true_query_group_ids] {
                this->thread_function(id, runtime, graph_backend, search_cache_list,
                                      results, num_cmps, query_stats, true_query_group_ids);
             }));
      }
      for (auto &&tp_result : tp_results)
         tp_result.get();
   }

   void UniNavGraph::search_hybrid(std::shared_ptr<IStorage> &query_storage,
                                   std::shared_ptr<DistanceHandler> &distance_handler,
                                   uint32_t num_threads, IdxType Lsearch,
                                   IdxType num_entry_points, std::string scenario,
                                   IdxType K, std::pair<IdxType, float> *results,
                                   std::vector<float> &num_cmps,
                                   std::vector<QueryStats> &query_stats,
                                   bool is_idea2_available,
                                   bool is_new_trie_method, bool is_rec_more_start,
                                   bool is_ung_more_entry,
                                   int lsearch_start, int lsearch_step,
                                   int efs_start, int efs_step_slow,int efs_step_fast,int lsearch_threshold,
                                   int force_use_alg,bool is_bfs_filter, const std::vector<IdxType> &true_query_group_ids)
   {
      SearchRuntimeConfig runtime = make_search_runtime_config(
          num_threads, Lsearch, num_entry_points, scenario, K,
          is_idea2_available, is_new_trie_method, is_rec_more_start,
          is_ung_more_entry, is_bfs_filter, lsearch_start, lsearch_step,
          efs_start, efs_step_slow, efs_step_fast, lsearch_threshold,
          force_use_alg);

      search_hybrid(query_storage, distance_handler, runtime, results, num_cmps, query_stats, true_query_group_ids);
   }

}
