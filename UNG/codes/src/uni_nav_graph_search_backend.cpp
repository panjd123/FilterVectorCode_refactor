#include "include/uni_nav_graph.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <limits>
#include <vector>

#include <roaring/roaring.hh>

#include "../../../ACORN/faiss/IndexACORN.h"

namespace ANNS
{

   bool UniNavGraph::execute_acorn_query(const char *query,
                                         const std::vector<LabelType> &query_labels,
                                         const std::vector<IdxType> &entry_group_ids,
                                         IdxType query_id,
                                         IdxType K,
                                         IdxType Lsearch,
                                         int lsearch_start,
                                         int lsearch_step,
                                         int efs_start,
                                         int efs_step_slow,
                                         int efs_step_fast,
                                         int lsearch_threshold,
                                         int force_use_alg,
                                         int algorithm,
                                         bool is_bfs_filter,
                                         bool use_old_bitmap_search,
                                         std::pair<IdxType, float> *results,
                                         std::vector<float> &num_cmps,
                                         SearchQueue &cur_result,
                                         QueryStats &stats)
   {
      auto search_time_start_ms = std::chrono::high_resolution_clock::now();
      std::shared_ptr<faiss::IndexACORNFlat> selected_acorn_index =
          (force_use_alg == 4) ? _acorn_1_index : _acorn_index;

      if (!selected_acorn_index)
      {
         std::cerr << "ERROR: ACORN index not loaded for query " << query_id << ". Skipping." << std::endl;
         for (auto k = 0; k < K; ++k)
            results[query_id * K + k].first = -1;
         return false;
      }

      int current_efs = efs_start;
      if (Lsearch <= lsearch_threshold)
      {
         current_efs = efs_start + ((Lsearch - lsearch_start) / lsearch_step) * efs_step_slow;
      }
      else
      {
         const int num_steps_in_slow_zone = (lsearch_threshold - lsearch_start) / lsearch_step;
         const int efs_at_threshold = efs_start + num_steps_in_slow_zone * efs_step_slow;
         const int num_steps_in_fast_zone = (Lsearch - lsearch_threshold) / lsearch_step;
         current_efs = efs_at_threshold + num_steps_in_fast_zone * efs_step_fast;
      }
      current_efs = std::max(current_efs, efs_start);

      stats.acorn_efs_used = current_efs;
      selected_acorn_index->acorn.efSearch = current_efs;

      const float *query_vector_float = reinterpret_cast<const float *>(query);
      std::vector<faiss::idx_t> result_original_ids(K);
      std::vector<float> result_dists(K);

      const bool current_use_bfs_filter = (force_use_alg == 0) ? (algorithm == 1) : is_bfs_filter;

      if (use_old_bitmap_search)
      {
         std::vector<std::vector<int>> query_attrs_for_acorn(1);
         query_attrs_for_acorn[0].assign(query_labels.begin(), query_labels.end());

         auto core_search_start_time = std::chrono::high_resolution_clock::now();
         selected_acorn_index->search_old_bitmap(
             1, query_vector_float, K, result_dists.data(), result_original_ids.data(),
             query_attrs_for_acorn, nullptr, nullptr, nullptr, current_use_bfs_filter);
         stats.core_search_time_ms =
             std::chrono::duration<double, std::milli>(
                 std::chrono::high_resolution_clock::now() - core_search_start_time)
                 .count();
      }
      else
      {
         roaring::Roaring current_roaring_bitmap = compute_bitmap_from_groups(entry_group_ids);
         auto bitmap_start_time = std::chrono::high_resolution_clock::now();
         std::vector<char> filter_map(_num_points, 0);
         for (uint32_t original_id : current_roaring_bitmap)
         {
            if (original_id < _num_points)
            {
               IdxType new_id = _old_to_new_vec_ids[original_id];
               filter_map[new_id] = 1;
            }
         }
         stats.bitmap_time_ms =
             std::chrono::duration<double, std::milli>(
                 std::chrono::high_resolution_clock::now() - bitmap_start_time)
                 .count();

         auto core_search_start_time = std::chrono::high_resolution_clock::now();
         selected_acorn_index->search(1, query_vector_float, K, result_dists.data(),
                                      result_original_ids.data(), filter_map.data(),
                                      nullptr, nullptr, nullptr, current_use_bfs_filter);
         stats.core_search_time_ms =
             std::chrono::duration<double, std::milli>(
                 std::chrono::high_resolution_clock::now() - core_search_start_time)
                 .count();
      }

      cur_result.clear();
      for (size_t i = 0; i < K; ++i)
      {
         if (result_original_ids[i] != -1)
            cur_result.insert(result_original_ids[i], result_dists[i]);
      }
      num_cmps[query_id] = 0;
      stats.search_time_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - search_time_start_ms)
              .count();
      return true;
   }

   bool UniNavGraph::execute_ung_query(const char *query,
                                       std::shared_ptr<SearchCache> search_cache,
                                       const GraphSearchBackend &graph_backend,
                                       const std::vector<IdxType> &entry_group_ids,
                                       IdxType query_id,
                                       IdxType num_entry_points,
                                       std::vector<float> &num_cmps,
                                       SearchQueue &cur_result,
                                       QueryStats &stats)
   {
      auto search_time_start_ms = std::chrono::high_resolution_clock::now();
      search_cache->visited_set.clear();
      std::vector<IdxType> entry_points;
      for (const auto &group_id : entry_group_ids)
         get_entry_points_given_group_id(num_entry_points, search_cache->visited_set, group_id, entry_points);

      if (entry_points.empty())
      {
         stats.num_distance_calcs = 0;
         return false;
      }

      auto core_search_start_time = std::chrono::high_resolution_clock::now();
      num_cmps[query_id] = iterate_to_fixed_point(query, search_cache, graph_backend,
                                                  query_id, entry_points, stats.num_nodes_visited);
      stats.core_search_time_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - core_search_start_time)
              .count();

      stats.num_distance_calcs = num_cmps[query_id];
      cur_result = search_cache->search_queue;
      stats.search_time_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - search_time_start_ms)
              .count();
      return true;
   }

   std::vector<IdxType> UniNavGraph::get_entry_points(const std::vector<LabelType> &query_label_set,
                                                      IdxType num_entry_points,
                                                      VisitedSet &visited_set)
   {
      std::vector<IdxType> entry_points;
      entry_points.reserve(num_entry_points);
      visited_set.clear();

      if (_scenario == "equality")
      {
         auto node = _trie_index.find_exact_match(query_label_set);
         if (node == nullptr)
            return entry_points;
         get_entry_points_given_group_id(num_entry_points, visited_set, node->group_id, entry_points);
      }
      else if (_scenario == "containment")
      {
         std::vector<IdxType> min_super_set_ids;
         get_min_super_sets(query_label_set, min_super_set_ids);
         for (auto group_id : min_super_set_ids)
            get_entry_points_given_group_id(num_entry_points, visited_set, group_id, entry_points);
      }
      else
      {
         std::cerr << "Error: invalid scenario " << _scenario << std::endl;
         exit(-1);
      }

      return entry_points;
   }

   void UniNavGraph::get_entry_points_given_group_id(IdxType num_entry_points,
                                                     VisitedSet &visited_set,
                                                     IdxType group_id,
                                                     std::vector<IdxType> &entry_points)
   {
      const auto &group_range = _group_id_to_range[group_id];

      if (group_range.second - group_range.first <= num_entry_points)
      {
         for (auto i = 0; i < group_range.second - group_range.first; ++i)
            entry_points.emplace_back(i + group_range.first);
         return;
      }

      const auto &group_entry_point = _group_entry_points[group_id];
      visited_set.set(group_entry_point);
      entry_points.emplace_back(group_entry_point);

      for (auto i = 1; i < num_entry_points; ++i)
      {
         auto entry_point = rand() % (group_range.second - group_range.first) + group_range.first;
         if (visited_set.check(entry_point) == false)
         {
            visited_set.set(entry_point);
            entry_points.emplace_back(i + group_range.first);
         }
      }
   }

   IdxType UniNavGraph::iterate_to_fixed_point(const char *query,
                                               std::shared_ptr<SearchCache> search_cache,
                                               IdxType target_id,
                                               const std::vector<IdxType> &entry_points,
                                               size_t &num_nodes_visited,
                                               bool clear_search_queue,
                                               bool clear_visited_set)
   {
      const GraphSearchBackend graph_backend(*_graph);
      return iterate_to_fixed_point(query, search_cache, graph_backend, target_id, entry_points,
                                    num_nodes_visited, clear_search_queue, clear_visited_set);
   }

   IdxType UniNavGraph::iterate_to_fixed_point(const char *query,
                                               std::shared_ptr<SearchCache> search_cache,
                                               const GraphSearchBackend &graph_backend,
                                               IdxType target_id,
                                               const std::vector<IdxType> &entry_points,
                                               size_t &num_nodes_visited,
                                               bool clear_search_queue,
                                               bool clear_visited_set)
   {
      (void)target_id;
      auto dim = _base_storage->get_dim();
      auto &search_queue = search_cache->search_queue;
      auto &visited_set = search_cache->visited_set;
      if (clear_search_queue)
         search_queue.clear();
      if (clear_visited_set)
         visited_set.clear();

      for (const auto &entry_point : entry_points)
         search_queue.insert(entry_point, _distance_handler->compute(query, _base_storage->get_vector(entry_point), dim));
      IdxType num_cmps = entry_points.size();

      while (search_queue.has_unexpanded_node())
      {
         const Candidate &cur = search_queue.get_closest_unexpanded();
         const GraphNeighborView neighbors = graph_backend.neighbors(cur.id);
         for (size_t i = 0; i < neighbors.size; ++i)
         {
            if (i + 1 < neighbors.size && visited_set.check(neighbors.ids[i + 1]) == false)
               _base_storage->prefetch_vec_by_id(neighbors.ids[i + 1]);

            const IdxType neighbor = neighbors.ids[i];
            if (visited_set.check(neighbor))
               continue;
            visited_set.set(neighbor);

            num_nodes_visited++;
            search_queue.insert(neighbor, _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim));
            num_cmps++;
         }
      }
      return num_cmps;
   }

   bool UniNavGraph::execute_special_block_ung_query(const char *query,
                                                     std::shared_ptr<SearchCache> search_cache,
                                                     const SearchRuntimeConfig &runtime,
                                                     const GraphSearchBackend &graph_backend,
                                                     const std::vector<IdxType> &entry_group_ids,
                                                     const std::vector<LabelType> &query_labels,
                                                     IdxType query_id,
                                                     std::vector<float> &num_cmps,
                                                     SearchQueue &cur_result,
                                                     QueryStats &stats)
   {
      struct SpecialCandidate
      {
         IdxType id = 0;
         float distance = 0.0f;
         bool expanded = false;
         bool free = false;
      };

      auto less_candidate = [](const SpecialCandidate &a, const SpecialCandidate &b) {
         if (a.distance != b.distance)
            return a.distance < b.distance;
         if (a.id != b.id)
            return a.id < b.id;
         return static_cast<int>(a.free) < static_cast<int>(b.free);
      };

      auto insert_candidate = [&](std::vector<SpecialCandidate> &queue,
                                  IdxType id,
                                  float distance,
                                  bool is_free,
                                  IdxType capacity,
                                  size_t &cur_unexpanded) {
         SpecialCandidate candidate{id, distance, false, is_free};
         if (queue.size() >= capacity && less_candidate(queue.back(), candidate))
            return;

         size_t lo = 0;
         size_t hi = queue.size();
         while (lo < hi)
         {
            const size_t mid = (lo + hi) >> 1;
            if (less_candidate(candidate, queue[mid]))
               hi = mid;
            else if (queue[mid].id == id && queue[mid].free == is_free)
               return;
            else
               lo = mid + 1;
         }
         queue.insert(queue.begin() + static_cast<std::ptrdiff_t>(lo), candidate);
         if (queue.size() > capacity)
            queue.pop_back();
         if (lo < cur_unexpanded)
            cur_unexpanded = lo;
      };

      auto search_time_start_ms = std::chrono::high_resolution_clock::now();
      stats.special_search_enabled = true;
      stats.special_free_use_regular = runtime.special_block_free_use_regular;
      const bool heavy_query_size_ok =
          runtime.special_heavy_edge_min_query_size == 0 ||
          stats.query_length >= runtime.special_heavy_edge_min_query_size;
      stats.special_heavy_edges_enabled =
          runtime.special_heavy_edge_search &&
          heavy_query_size_ok &&
          ((runtime.special_heavy_edge_min_matched_points > 0 &&
            stats.entry_group_matched_points >= runtime.special_heavy_edge_min_matched_points) ||
           (runtime.special_heavy_edge_min_entries > 0 &&
            stats.num_entry_points >= runtime.special_heavy_edge_min_entries));
      const bool profile_timing = std::getenv("UNG_SPECIAL_PROFILE_TIMING") != nullptr;
      auto elapsed_ms = [](const std::chrono::high_resolution_clock::time_point &start) {
         return std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - start)
             .count();
      };

      const IdxType dim = _base_storage->get_dim();
      const IdxType capacity = runtime.Lsearch;
      std::chrono::high_resolution_clock::time_point cover_time_start;
      if (profile_timing)
         cover_time_start = std::chrono::high_resolution_clock::now();
      std::vector<uint8_t> query_covers_block(_special_blocks.size() + 1, 0);
      if (!query_covers_block.empty())
      {
         std::vector<LabelType> sorted_query = query_labels;
         std::sort(sorted_query.begin(), sorted_query.end());
         for (size_t block_idx = 0; block_idx < _special_blocks.size(); ++block_idx)
         {
            std::vector<LabelType> sorted_root = _special_blocks[block_idx].root_labels;
            std::sort(sorted_root.begin(), sorted_root.end());
            if (std::includes(sorted_root.begin(), sorted_root.end(),
                              sorted_query.begin(), sorted_query.end()))
               query_covers_block[block_idx + 1] = 1;
         }
      }
      if (profile_timing)
         stats.special_cover_time_ms = elapsed_ms(cover_time_start);

      auto core_search_start_time = std::chrono::high_resolution_clock::now();
      const auto entry_time_start = core_search_start_time;
      auto &visited_regular = search_cache->special_visited_regular;
      auto &visited_free = search_cache->special_visited_free;
      visited_regular.clear();
      visited_free.clear();
      std::vector<SpecialCandidate> queue;
      constexpr size_t kReducedEntryGroupThreshold = 1000;
      constexpr IdxType kReducedEntryPointsPerLargeGroupSet = 4;
      const IdxType effective_num_entry_points =
          entry_group_ids.size() > kReducedEntryGroupThreshold
              ? std::min<IdxType>(runtime.num_entry_points, kReducedEntryPointsPerLargeGroupSet)
              : runtime.num_entry_points;
      const size_t entry_reserve = std::max<size_t>(
          static_cast<size_t>(capacity) + 1,
          entry_group_ids.size() * static_cast<size_t>(effective_num_entry_points));
      queue.reserve(entry_reserve);
      size_t cur_unexpanded = 0;

      std::vector<IdxType> entry_point_ids;
      std::vector<uint8_t> entry_is_free;
      entry_point_ids.reserve(entry_reserve);
      entry_is_free.reserve(entry_reserve);

      auto add_entry_point = [&](IdxType point_id, bool is_free) {
         if (visited_regular.check(point_id) || visited_free.check(point_id))
            return;

         entry_point_ids.push_back(point_id);
         entry_is_free.push_back(is_free ? 1 : 0);
         if (is_free)
         {
            visited_free.set(point_id);
            stats.special_entry_free_points++;
            stats.special_free_candidates_inserted++;
         }
         else
         {
            visited_regular.set(point_id);
            stats.special_entry_regular_points++;
            stats.special_regular_candidates_inserted++;
         }
      };
      for (IdxType group_id : entry_group_ids)
      {
         if (group_id >= _group_id_to_range.size())
            continue;
         const IdxType root_block_id = is_special_block_root_group(group_id) ? _group_id_to_special_block[group_id] : 0;
         const bool covered_root = root_block_id > 0 && root_block_id < query_covers_block.size() &&
                                   query_covers_block[root_block_id] != 0;
         const auto &range = _group_id_to_range[group_id];
         if (range.second <= range.first)
            continue;
         const IdxType take = std::min<IdxType>(effective_num_entry_points, range.second - range.first);
         for (IdxType local = 0; local < take; ++local)
         {
            const IdxType point_id =
                (local == 0 && group_id < _group_entry_points.size() &&
                 _group_entry_points[group_id] >= range.first && _group_entry_points[group_id] < range.second)
                    ? _group_entry_points[group_id]
                    : range.first + local;
            add_entry_point(point_id, covered_root);
         }
      }

      if (entry_point_ids.empty())
      {
         stats.num_distance_calcs = 0;
         return false;
      }

      queue.reserve(std::max(queue.capacity(), entry_point_ids.size()));
      for (size_t i = 0; i < entry_point_ids.size(); ++i)
      {
         const IdxType point_id = entry_point_ids[i];
         queue.push_back(SpecialCandidate{point_id,
                                          _distance_handler->compute(query, _base_storage->get_vector(point_id), dim),
                                          false, entry_is_free[i] != 0});
      }

      if (queue.size() > static_cast<size_t>(capacity))
      {
         auto keep_end = queue.begin() + static_cast<std::ptrdiff_t>(capacity);
         std::nth_element(queue.begin(), keep_end, queue.end(), less_candidate);
         queue.resize(static_cast<size_t>(capacity));
      }
      std::sort(queue.begin(), queue.end(), less_candidate);
      stats.special_entry_time_ms = elapsed_ms(entry_time_start);


      IdxType comparisons = static_cast<IdxType>(queue.size());
      stats.special_free_distance_calcs += stats.special_entry_free_points;
      stats.special_regular_distance_calcs += stats.special_entry_regular_points;
      while (cur_unexpanded < queue.size())
      {
         const size_t cur_idx = cur_unexpanded;
         queue[cur_idx].expanded = true;
         while (cur_unexpanded < queue.size() && queue[cur_unexpanded].expanded)
            ++cur_unexpanded;
         const SpecialCandidate cur = queue[cur_idx];

         auto visit_neighbor = [&](IdxType neighbor, bool next_free) {
            if (neighbor >= _num_points)
               return false;
            if (next_free)
            {
               if (visited_free.check(neighbor))
                  return false;
               visited_free.set(neighbor);
               stats.special_free_candidates_inserted++;
               if (!cur.free)
                  stats.special_free_upgrades++;
            }
            else
            {
               if (visited_regular.check(neighbor))
                  return false;
               visited_regular.set(neighbor);
               stats.special_regular_candidates_inserted++;
            }
            stats.num_nodes_visited++;
            insert_candidate(queue, neighbor,
                             _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim),
                             next_free, capacity, cur_unexpanded);
            if (next_free)
               stats.special_free_distance_calcs++;
            else
               stats.special_regular_distance_calcs++;
            comparisons++;
            return true;
         };

         if (cur.free)
            stats.special_free_nodes_expanded++;
         else
            stats.special_regular_nodes_expanded++;

         if (cur.free)
         {
            if (cur.id < _special_edges_by_point.size())
            {
               auto scan_special_edges = [&](const std::vector<SpecialEdge> &edges, bool heavy) {
                  std::chrono::high_resolution_clock::time_point special_edges_time_start;
                  if (profile_timing)
                     special_edges_time_start = std::chrono::high_resolution_clock::now();
                  for (const SpecialEdge &edge : edges)
                  {
                     stats.special_edges_scanned++;
                     if (heavy)
                        stats.special_heavy_edges_scanned++;
                     if (edge.kind == SpecialEdgeKind::InterBlock)
                        stats.special_inter_edges_scanned++;
                     else
                        stats.special_intra_edges_scanned++;
                     if (visit_neighbor(edge.target_point_id, true))
                     {
                        stats.special_edges_accepted++;
                        if (heavy)
                           stats.special_heavy_edges_accepted++;
                     }
                  }
                  if (profile_timing)
                     stats.special_special_edges_time_ms += elapsed_ms(special_edges_time_start);
               };
               scan_special_edges(_special_edges_by_point[cur.id], false);
               if (stats.special_heavy_edges_enabled && cur.id < _special_heavy_edges_by_point.size())
                  scan_special_edges(_special_heavy_edges_by_point[cur.id], true);
            }
            if (!runtime.special_block_free_use_regular)
               continue;
         }

         std::chrono::high_resolution_clock::time_point regular_edges_time_start;
         if (profile_timing)
            regular_edges_time_start = std::chrono::high_resolution_clock::now();
         const GraphNeighborView neighbors = graph_backend.neighbors(cur.id);
         for (size_t i = 0; i < neighbors.size; ++i)
         {
            stats.special_regular_edges_scanned++;
            const IdxType neighbor = neighbors.ids[i];
            bool next_free = cur.free;
            if (!next_free && is_special_block_root_point(neighbor))
            {
               const IdxType block_id = neighbor < _point_to_special_block.size() ? _point_to_special_block[neighbor] : 0;
               if (block_id > 0 && block_id < query_covers_block.size() &&
                   query_covers_block[block_id] != 0)
                  next_free = true;
            }
            if (visit_neighbor(neighbor, next_free))
               stats.special_regular_edges_accepted++;
         }
         if (profile_timing)
            stats.special_regular_edges_time_ms += elapsed_ms(regular_edges_time_start);
      }

      stats.core_search_time_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - core_search_start_time)
              .count();

      std::chrono::high_resolution_clock::time_point result_time_start;
      if (profile_timing)
         result_time_start = std::chrono::high_resolution_clock::now();
      cur_result.clear();
      for (const SpecialCandidate &candidate : queue)
         cur_result.insert(candidate.id, candidate.distance);
      if (profile_timing)
         stats.special_result_time_ms = elapsed_ms(result_time_start);
      num_cmps[query_id] = comparisons;
      stats.num_distance_calcs = comparisons;
      stats.search_time_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - search_time_start_ms)
              .count();
      return true;
   }

   CrossEdgeCsrOutput UniNavGraph::build_search_graph_csr() const
   {
      CrossEdgeCsrOutput csr;
      csr.row_offsets.resize(static_cast<size_t>(_num_points) + 1, 0);
      for (IdxType point_id = 0; point_id < _num_points; ++point_id)
      {
         csr.row_offsets[static_cast<size_t>(point_id) + 1] =
             csr.row_offsets[static_cast<size_t>(point_id)] +
             _graph->neighbors[point_id].size();
      }
      csr.col_indices.resize(csr.row_offsets.back());
#pragma omp parallel for schedule(dynamic, 4096)
      for (IdxType point_id = 0; point_id < _num_points; ++point_id)
      {
         const auto &neighbors = _graph->neighbors[point_id];
         const std::size_t base = csr.row_offsets[static_cast<size_t>(point_id)];
         for (std::size_t k = 0; k < neighbors.size(); ++k)
            csr.col_indices[base + k] = neighbors[k];
      }
      return csr;
   }

} // namespace ANNS
