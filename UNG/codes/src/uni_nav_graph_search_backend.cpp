#include "include/uni_nav_graph.h"
#include "include/ung_special_block_activation.h"
#include "include/ung_special_trie_regular_search.h"
#include "include/ung_favor_block_search.h"
#include "include/ung_gpu_l2_batch.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <limits>
#include <queue>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

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
      auto core_search_start_time = search_time_start_ms;
      auto entry_point_setup_start_time = core_search_start_time;
      search_cache->visited_set.clear();
      std::vector<IdxType> entry_points;
      for (const auto &group_id : entry_group_ids)
         get_entry_points_given_group_id(num_entry_points, search_cache->visited_set, group_id, entry_points);

      stats.entry_point_setup_time_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - entry_point_setup_start_time)
              .count();

      if (entry_points.empty())
      {
         stats.num_distance_calcs = 0;
         return false;
      }

      num_cmps[query_id] = iterate_to_fixed_point(query, search_cache, graph_backend,
                                                  query_id, entry_points, stats.num_nodes_visited,
                                                  true, true, &stats);
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
         {
            const IdxType entry_point = i + group_range.first;
            if (!visited_set.check(entry_point))
            {
               visited_set.set(entry_point);
               entry_points.emplace_back(entry_point);
            }
         }
         return;
      }

      const auto &group_entry_point = _group_entry_points[group_id];
      if (group_entry_point >= group_range.first && group_entry_point < group_range.second &&
          !visited_set.check(group_entry_point))
      {
         visited_set.set(group_entry_point);
         entry_points.emplace_back(group_entry_point);
      }

      for (auto i = 1; i < num_entry_points; ++i)
      {
         const IdxType entry_point = rand() % (group_range.second - group_range.first) + group_range.first;
         if (!visited_set.check(entry_point))
         {
            visited_set.set(entry_point);
            entry_points.emplace_back(entry_point);
         }
      }
   }

   IdxType UniNavGraph::iterate_to_fixed_point(const char *query,
                                               std::shared_ptr<SearchCache> search_cache,
                                               IdxType target_id,
                                               const std::vector<IdxType> &entry_points,
                                               size_t &num_nodes_visited,
                                               bool clear_search_queue,
                                               bool clear_visited_set,
                                               QueryStats *stats)
   {
      const GraphSearchBackend graph_backend(*_graph);
      return iterate_to_fixed_point(query, search_cache, graph_backend, target_id, entry_points,
                                    num_nodes_visited, clear_search_queue, clear_visited_set, stats);
   }

   IdxType UniNavGraph::iterate_to_fixed_point(const char *query,
                                               std::shared_ptr<SearchCache> search_cache,
                                               const GraphSearchBackend &graph_backend,
                                               IdxType target_id,
                                               const std::vector<IdxType> &entry_points,
                                               size_t &num_nodes_visited,
                                               bool clear_search_queue,
                                               bool clear_visited_set,
                                               QueryStats *stats)
   {
      (void)target_id;
      const auto entry_point_scoring_start_time = std::chrono::high_resolution_clock::now();
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
      if (stats != nullptr)
      {
         stats->entry_point_setup_time_ms +=
             std::chrono::duration<double, std::milli>(
                 std::chrono::high_resolution_clock::now() - entry_point_scoring_start_time)
                 .count();
      }

      while (search_queue.has_unexpanded_node())
      {
         const Candidate &cur = search_queue.get_closest_unexpanded();
         const GraphNeighborView neighbors = graph_backend.neighbors(cur.id);
         if (stats != nullptr)
            stats->regular_edges_scanned += neighbors.size;
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
      const bool profile_timing = std::getenv("UNG_SPECIAL_PROFILE_TIMING") != nullptr;
      const bool detail_stats = !runtime.special_light_stats;

      auto search_time_start_ms = std::chrono::high_resolution_clock::now();
      stats.special_search_enabled = true;
      stats.special_free_use_regular = runtime.special_block_free_use_regular;
      const bool trie_regular_search =
          runtime.entry_group_provider == EntryGroupProviderImpl::SpecialBlockTrie &&
          !ung_env_flag_enabled("UNG_SPECIAL_TRIE_LEGACY_LNG_REGULAR");
      const bool lazy_block_activation =
          trie_regular_search && ung_env_flag_enabled("UNG_SPECIAL_TRIE_LAZY_BLOCK_ACTIVATION");
      stats.special_trie_regular_search_enabled = trie_regular_search;
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
      auto elapsed_ms = [](const std::chrono::high_resolution_clock::time_point &start) {
         return std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - start)
             .count();
      };

      const IdxType dim = _base_storage->get_dim();
      const IdxType capacity = runtime.Lsearch;
      size_t free_node_expansions_per_block = std::numeric_limits<size_t>::max();
      if (const char *value = std::getenv("UNG_SPECIAL_FREE_NODE_EXPANSIONS_PER_BLOCK"))
      {
         const size_t configured_cap = static_cast<size_t>(std::strtoull(value, nullptr, 10));
         if (configured_cap > 0)
            free_node_expansions_per_block = configured_cap;
      }
      size_t free_node_cap_min_free_blocks = 0;
      if (const char *value = std::getenv("UNG_SPECIAL_FREE_NODE_CAP_MIN_FREE_BLOCKS"))
         free_node_cap_min_free_blocks =
             static_cast<size_t>(std::strtoull(value, nullptr, 10));
      size_t free_intra_edge_scan_cap = std::numeric_limits<size_t>::max();
      if (const char *value = std::getenv("UNG_SPECIAL_FREE_INTRA_EDGE_SCAN_CAP"))
      {
         const size_t configured_cap = static_cast<size_t>(std::strtoull(value, nullptr, 10));
         if (configured_cap > 0)
            free_intra_edge_scan_cap = configured_cap;
      }
      // Optional cap on inter-block edges examined from each free source
      // point.  The default is unlimited for backward compatibility.  This
      // is deliberately separate from the intra-block cap because a parent
      // block may have many child blocks and therefore a very large
      // inter-block fanout.
      size_t free_inter_edge_scan_cap = std::numeric_limits<size_t>::max();
      if (const char *value = std::getenv("UNG_SPECIAL_FREE_INTER_EDGE_SCAN_CAP"))
      {
         const size_t configured_cap = static_cast<size_t>(std::strtoull(value, nullptr, 10));
         if (configured_cap > 0)
            free_inter_edge_scan_cap = configured_cap;
      }
      size_t free_inter_edge_per_block_cap = std::numeric_limits<size_t>::max();
      if (const char *value = std::getenv("UNG_SPECIAL_FREE_INTER_EDGE_PER_BLOCK_CAP"))
      {
         const size_t configured_cap = static_cast<size_t>(std::strtoull(value, nullptr, 10));
         if (configured_cap > 0)
            free_inter_edge_per_block_cap = configured_cap;
      }
      size_t free_inter_edge_cap_min_child_blocks = 0;
      if (const char *value =
              std::getenv("UNG_SPECIAL_FREE_INTER_EDGE_CAP_MIN_CHILD_BLOCKS"))
         free_inter_edge_cap_min_child_blocks =
             static_cast<size_t>(std::strtoull(value, nullptr, 10));
      size_t free_block_stall_limit = 0;
      if (const char *value = std::getenv("UNG_SPECIAL_BLOCK_STALL_NO_UPDATE_LIMIT"))
         free_block_stall_limit = static_cast<size_t>(std::strtoull(value, nullptr, 10));
      size_t trie_block_portal_min_lsearch = 0;
      if (const char *value = std::getenv("UNG_SPECIAL_TRIE_BLOCK_PORTAL_MIN_LSEARCH"))
         trie_block_portal_min_lsearch = static_cast<size_t>(std::strtoull(value, nullptr, 10));
      const bool trie_block_portals =
          trie_regular_search &&
          !ung_env_flag_enabled("UNG_SPECIAL_TRIE_DISABLE_BLOCK_PORTALS") &&
          (lazy_block_activation ||
           static_cast<size_t>(capacity) >= trie_block_portal_min_lsearch);
      IdxType approx_dims = 0;
      if (const char *value = std::getenv("UNG_SPECIAL_APPROX_DIMS"))
         approx_dims = static_cast<IdxType>(std::strtoull(value, nullptr, 10));
      if (approx_dims >= dim)
         approx_dims = 0;
      const float approx_scale = approx_dims > 0 ? static_cast<float>(dim) / static_cast<float>(approx_dims) : 1.0f;
      bool gpu_free_distance = ung_env_flag_enabled("UNG_SPECIAL_FREE_GPU_DISTANCE") && approx_dims == 0 &&
                               _base_storage->get_data_type() == DataType::FLOAT;
      size_t gpu_free_distance_min = 4096;
      if (const char *value = std::getenv("UNG_SPECIAL_FREE_GPU_DISTANCE_MIN"))
         gpu_free_distance_min = static_cast<size_t>(std::strtoull(value, nullptr, 10));
      const float *gpu_base_vectors = gpu_free_distance
                                          ? reinterpret_cast<const float *>(_base_storage->get_vector(0))
                                          : nullptr;
      const float *gpu_query_vector = gpu_free_distance
                                         ? reinterpret_cast<const float *>(query)
                                         : nullptr;
      auto exact_distance = [&](IdxType point_id) {
         return _distance_handler->compute(query, _base_storage->get_vector(point_id), dim);
      };
      auto score_distance = [&](IdxType point_id) {
         if (approx_dims == 0)
            return exact_distance(point_id);
         const float *q = reinterpret_cast<const float *>(query);
         const float *v = reinterpret_cast<const float *>(_base_storage->get_vector(point_id));
         float sum = 0.0f;
         for (IdxType s = 0; s < approx_dims; ++s)
         {
            const IdxType d = (s * dim) / approx_dims;
            const float diff = q[d] - v[d];
            sum += diff * diff;
         }
         return sum * approx_scale;
      };
      std::chrono::high_resolution_clock::time_point cover_time_start;
      if (profile_timing)
         cover_time_start = std::chrono::high_resolution_clock::now();
      std::vector<uint8_t> query_covers_block(_special_blocks.size() + 1, 0);
      if (!query_covers_block.empty())
      {
         std::vector<LabelType> sorted_query = query_labels;
         std::sort(sorted_query.begin(), sorted_query.end());
         // A query covers a whole trie subtree iff its labels are contained
         // by the subtree root prefix. The historical common-label policy is
         // retained only for single-layer A/B compatibility; propagating it
         // across multiple layers could authorize an uncovered child block.
         const bool root_label_coverage =
             _special_block_summary.upper_blocks > 0 ||
             ung_env_flag_enabled("UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE");
         for (size_t block_idx = 0; block_idx < _special_blocks.size(); ++block_idx)
         {
            const std::vector<LabelType> &coverage_labels =
                root_label_coverage
                    ? _special_blocks[block_idx].root_labels
                    : _special_blocks[block_idx].common_labels;
            if (std::includes(coverage_labels.begin(), coverage_labels.end(),
                              sorted_query.begin(), sorted_query.end()))
               query_covers_block[block_idx + 1] = 1;
         }
      }
      std::vector<uint8_t> query_free_block = query_covers_block;
      // Free state is monotone down the Trie block hierarchy.
      if (runtime.scenario == "containment")
      {
         std::vector<IdxType> pending_blocks;
         pending_blocks.reserve(_special_blocks.size());
         for (IdxType block_id = 1; block_id < query_free_block.size(); ++block_id)
         {
            if (query_free_block[block_id] != 0)
               pending_blocks.push_back(block_id);
         }
         for (size_t pending_idx = 0; pending_idx < pending_blocks.size(); ++pending_idx)
         {
            const IdxType block_id = pending_blocks[pending_idx];
            for (IdxType child_block_id : _special_blocks[block_id - 1].child_block_ids)
            {
               if (child_block_id == 0 || child_block_id >= query_free_block.size() ||
                   query_free_block[child_block_id] != 0)
                  continue;
               query_free_block[child_block_id] = 1;
               pending_blocks.push_back(child_block_id);
            }
         }
      }
      const bool upper_gate_configured =
          runtime.special_max_activation_level < std::numeric_limits<uint8_t>::max() ||
          runtime.special_upper_min_covered_points > 0;
      if (upper_gate_configured)
      {
         const SpecialBlockLevelGateResult gate = special_block_apply_level_gate(
             query_free_block, _special_blocks,
             runtime.special_max_activation_level,
             runtime.special_upper_min_covered_points);
         stats.special_query_upper_block_count = gate.upper_covered_blocks;
         stats.special_query_upper_covered_points = gate.upper_covered_points;
         stats.special_query_upper_enabled = gate.upper_enabled;
      }

      std::vector<uint8_t> query_free_block_frontier(query_free_block.size(), 0);
      std::vector<IdxType> special_block_parent(query_free_block.size(), 0);
      for (IdxType block_id = 1;
           (detail_stats || lazy_block_activation) && block_id < query_free_block.size();
           ++block_id)
      {
         for (IdxType child_block_id : _special_blocks[block_id - 1].child_block_ids)
         {
            if (child_block_id > 0 && child_block_id < special_block_parent.size())
               special_block_parent[child_block_id] = block_id;
         }
      }
      for (IdxType block_id = 1;
           (detail_stats || lazy_block_activation) && block_id < query_free_block.size();
           ++block_id)
      {
         if (query_free_block[block_id] == 0)
            continue;
         const IdxType parent_block_id = special_block_parent[block_id];
         if (parent_block_id == 0 || query_free_block[parent_block_id] == 0)
         {
            query_free_block_frontier[block_id] = 1;
            if (detail_stats)
               stats.special_free_block_frontier_count++;
         }
      }
      stats.special_free_block_count = static_cast<size_t>(
          std::count(query_free_block.begin(), query_free_block.end(), uint8_t{1}));
      for (IdxType block_id = 1; detail_stats && block_id < query_free_block.size(); ++block_id)
      {
         if (query_free_block[block_id] == 0)
            continue;
         if (_special_blocks[block_id - 1].level == 0)
            stats.special_query_middle_block_count++;
         else
         {
            if (!upper_gate_configured)
            {
               stats.special_query_upper_block_count++;
               stats.special_query_upper_covered_points +=
                   static_cast<size_t>(_special_blocks[block_id - 1].point_count);
            }
         }
      }
      size_t lazy_seed_depth = 0;
      if (const char *value = std::getenv("UNG_SPECIAL_TRIE_LAZY_SEED_DEPTH"))
         lazy_seed_depth = static_cast<size_t>(std::strtoull(value, nullptr, 10));
      const std::vector<uint8_t> query_seed_block =
          lazy_block_activation
              ? special_block_lazy_seed_mask(query_free_block_frontier,
                                             _special_blocks, lazy_seed_depth)
              : query_free_block;
      if (profile_timing)
         stats.special_cover_time_ms = elapsed_ms(cover_time_start);
      std::vector<uint8_t> searched_blocks(_special_blocks.size() + 1, 0);
      std::vector<size_t> free_nodes_expanded_by_block(
          free_node_expansions_per_block != std::numeric_limits<size_t>::max() &&
                  stats.special_free_block_count >= free_node_cap_min_free_blocks
              ? _special_blocks.size() + 1
          : 0,
          0);
      std::vector<size_t> free_block_stall_counts(
          free_block_stall_limit > 0 ? _special_blocks.size() + 1 : 0, 0);
      std::vector<uint8_t> free_block_paused(
          free_block_stall_limit > 0 ? _special_blocks.size() + 1 : 0, 0);
      std::vector<size_t> inter_edges_seen_by_target_block(
          free_inter_edge_per_block_cap != std::numeric_limits<size_t>::max()
              ? _special_blocks.size() + 1
              : 0,
          0);
      std::vector<IdxType> inter_target_blocks_touched;
      if (!inter_edges_seen_by_target_block.empty())
         inter_target_blocks_touched.reserve(32);

      auto core_search_start_time = std::chrono::high_resolution_clock::now();
      const auto entry_time_start = core_search_start_time;
      auto &visited_regular = search_cache->special_visited_regular;
      auto &visited_free = search_cache->special_visited_free;
      auto &visited_upper = search_cache->special_visited_upper;
      visited_regular.clear();
      visited_free.clear();
      visited_upper.clear();
      auto &free_state_cache = search_cache->special_free_state_cache;
      auto &free_state_touched = search_cache->special_free_state_touched;
      if (free_state_cache.size() < _num_points)
         free_state_cache.assign(_num_points, 0);
      for (IdxType point_id : free_state_touched)
      {
         if (point_id < free_state_cache.size())
            free_state_cache[point_id] = 0;
      }
      free_state_touched.clear();

      auto cached_point_activation_level = [&](IdxType point_id) -> uint8_t {
         uint8_t &state = free_state_cache[point_id];
         if (state == 0)
         {
            const bool middle_is_covered = special_block_member_is_free(
                runtime.scenario, _point_to_special_block, point_id,
                query_free_block);
            // Cache uses 1 for ordinary and 2 for middle activation so zero
            // remains the uncached sentinel. Upper activation is deliberately
            // not reachable directly from ordinary graph traversal.
            state = middle_is_covered ? 2 : 1;
            free_state_touched.push_back(point_id);
         }
         return state == 2 ? uint8_t{1} : uint8_t{0};
      };
      auto point_upper_activation_level = [&](IdxType point_id,
                                              uint8_t current_level) -> uint8_t {
         if (current_level == 0 || runtime.scenario != "containment" ||
             point_id >= _point_to_upper_special_block.size())
            return current_level;
         const IdxType upper_block_id = _point_to_upper_special_block[point_id];
         if (upper_block_id == 0 || upper_block_id >= query_free_block.size() ||
             query_free_block[upper_block_id] == 0)
            return current_level;
         return std::max<uint8_t>(current_level, 2);
      };
      auto visited_for_level = [&](uint8_t level) -> VisitedSet & {
         if (level >= 2)
            return visited_upper;
         return level == 1 ? visited_free : visited_regular;
      };

      auto prefetch_point = [&](IdxType point_id) {
         if (!runtime.special_block_prefetch || point_id >= _num_points)
            return;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
         visited_regular.prefetch(point_id);
         visited_free.prefetch(point_id);
         visited_upper.prefetch(point_id);
         _mm_prefetch(_base_storage->get_vector(point_id), _MM_HINT_T0);
#endif
      };

      auto &candidate_queue = search_cache->special_candidate_queue;
      candidate_queue.reset(static_cast<size_t>(capacity), static_cast<size_t>(runtime.K),
                            ung_env_flag_enabled("UNG_SPECIAL_CANDIDATE_HEAP"));
      constexpr size_t kReducedEntryGroupThreshold = 1000;
      constexpr IdxType kReducedEntryPointsPerLargeGroupSet = 4;
      const IdxType effective_num_entry_points =
          entry_group_ids.size() > kReducedEntryGroupThreshold
              ? std::min<IdxType>(runtime.num_entry_points, kReducedEntryPointsPerLargeGroupSet)
              : runtime.num_entry_points;
      const size_t entry_reserve = std::max<size_t>(
          static_cast<size_t>(capacity) + 1,
          entry_group_ids.size() * static_cast<size_t>(effective_num_entry_points));
      std::vector<IdxType> entry_point_ids;
      std::vector<uint8_t> entry_activation_level;
      std::vector<uint8_t> entry_is_block_seed;
      entry_point_ids.reserve(entry_reserve);
      entry_activation_level.reserve(entry_reserve);
      entry_is_block_seed.reserve(entry_reserve);

      auto add_entry_point = [&](IdxType point_id, uint8_t activation_level,
                                 bool is_block_seed = false) {
         VisitedSet &visited = visited_for_level(activation_level);
         if (visited.check(point_id))
            return false;

         entry_point_ids.push_back(point_id);
         entry_activation_level.push_back(activation_level);
         entry_is_block_seed.push_back(is_block_seed ? 1 : 0);
         visited.set(point_id);
         if (activation_level > 0)
         {
            stats.special_entry_free_points++;
            if (detail_stats)
            {
               stats.special_free_candidates_inserted++;
            }
         }
         else
         {
            stats.special_entry_regular_points++;
            if (detail_stats)
            {
               stats.special_regular_candidates_inserted++;
            }
         }
         if (is_block_seed)
            stats.special_block_seed_points++;
         return true;
      };
      const bool seed_free_blocks =
          trie_regular_search && ung_env_flag_enabled("UNG_SPECIAL_TRIE_SEED_FREE_BLOCKS");
      size_t block_seed_cap = std::numeric_limits<size_t>::max();
      if (const char *value = std::getenv("UNG_SPECIAL_TRIE_BLOCK_SEED_CAP"))
      {
         const size_t configured_cap = static_cast<size_t>(std::strtoull(value, nullptr, 10));
         if (configured_cap > 0)
            block_seed_cap = configured_cap;
      }
      size_t block_seeds_per_block = 1;
      if (const char *value = std::getenv("UNG_SPECIAL_TRIE_BLOCK_SEEDS_PER_BLOCK"))
      {
         const size_t configured_count = static_cast<size_t>(std::strtoull(value, nullptr, 10));
         if (configured_count > 0)
            block_seeds_per_block = configured_count;
      }
      size_t block_seeds_retained_per_block = block_seeds_per_block;
      if (const char *value = std::getenv("UNG_SPECIAL_TRIE_BLOCK_SEEDS_RETAIN_PER_BLOCK"))
      {
         const size_t configured_count = static_cast<size_t>(std::strtoull(value, nullptr, 10));
         if (configured_count > 0)
            block_seeds_retained_per_block = configured_count;
      }
      size_t frontier_block_seeds_retained_per_block = block_seeds_retained_per_block;
      if (const char *value =
              std::getenv("UNG_SPECIAL_TRIE_FRONTIER_BLOCK_SEEDS_RETAIN_PER_BLOCK"))
      {
         const size_t configured_count = static_cast<size_t>(std::strtoull(value, nullptr, 10));
         if (configured_count > 0)
            frontier_block_seeds_retained_per_block = configured_count;
      }
      if (seed_free_blocks)
      {
         // Candidate landmarks are structurally spread across each block's direct members.
         for (IdxType block_id = 1; block_id < query_free_block.size(); ++block_id)
         {
            if (query_seed_block[block_id] == 0 ||
                _special_blocks[block_id - 1].level != 0)
               continue;
            const IdxType entry_point = _special_blocks[block_id - 1].entry_point_id;
            if (entry_point == SpecialBlock::kInvalidEntryPoint || entry_point >= _num_points)
               continue;
            add_entry_point(entry_point, 1, true);

            const std::vector<IdxType> &member_group_ids =
                _special_blocks[block_id - 1].member_group_ids;
            for (size_t seed_idx = 1;
                 seed_idx < block_seeds_per_block && !member_group_ids.empty();
                 ++seed_idx)
            {
               const size_t member_idx = std::min(
                   member_group_ids.size() - 1,
                   (seed_idx * member_group_ids.size()) / block_seeds_per_block);
               const IdxType group_id = member_group_ids[member_idx];
               if (group_id >= _group_id_to_range.size())
                  continue;
               const auto &range = _group_id_to_range[group_id];
               if (range.second <= range.first)
                  continue;
               const IdxType landmark =
                   group_id < _group_entry_points.size() &&
                           _group_entry_points[group_id] >= range.first &&
                           _group_entry_points[group_id] < range.second
                       ? _group_entry_points[group_id]
                       : range.first;
               add_entry_point(landmark, 1, true);
            }
         }
      }
      const char *free_group_entry_cap_value =
          std::getenv("UNG_SPECIAL_TRIE_FREE_GROUP_ENTRY_CAP");
      const bool free_group_entry_cap_enabled = free_group_entry_cap_value != nullptr;
      const size_t free_group_entry_cap = free_group_entry_cap_enabled
                                              ? static_cast<size_t>(std::strtoull(
                                                    free_group_entry_cap_value, nullptr, 10))
                                              : std::numeric_limits<size_t>::max();
      size_t exact_group_entry_cap = 16;
      if (const char *value = std::getenv("UNG_SPECIAL_TRIE_EXACT_GROUP_ENTRY_CAP"))
         exact_group_entry_cap = static_cast<size_t>(std::strtoull(value, nullptr, 10));
      std::vector<size_t> free_group_entries_by_block(
          free_group_entry_cap_enabled ? _special_blocks.size() + 1 : 0, 0);
      size_t exact_group_entries = 0;
      std::vector<uint8_t> entry_portal_seen(
          trie_block_portals ? _special_blocks.size() + 1 : 0, 0);
      for (IdxType group_id : entry_group_ids)
      {
         if (group_id >= _group_id_to_range.size())
            continue;
         const bool covered_root = special_block_member_is_free(runtime.scenario,
                                                                _group_id_to_special_block,
                                                                group_id,
                                                                query_free_block);
         const auto &range = _group_id_to_range[group_id];
         if (range.second <= range.first)
            continue;
         const bool exact_query_group =
             covered_root && group_id < _group_id_to_label_set.size() &&
             _group_id_to_label_set[group_id] == query_labels;
         const IdxType block_id =
             covered_root && group_id < _group_id_to_special_block.size()
                 ? _group_id_to_special_block[group_id]
                 : 0;
         const IdxType take = std::min<IdxType>(effective_num_entry_points, range.second - range.first);
         if (lazy_block_activation && covered_root && !exact_query_group &&
             (block_id == 0 || block_id >= query_seed_block.size() ||
              query_seed_block[block_id] == 0))
         {
            stats.special_group_entry_points_policy_skipped += take;
            continue;
         }
         for (IdxType local = 0; local < take; ++local)
         {
            size_t *policy_count = nullptr;
            size_t policy_cap = std::numeric_limits<size_t>::max();
            if (seed_free_blocks && exact_query_group)
            {
               policy_count = &exact_group_entries;
               policy_cap = exact_group_entry_cap;
            }
            else if (seed_free_blocks && covered_root && free_group_entry_cap_enabled &&
                     block_id > 0 && block_id < free_group_entries_by_block.size())
            {
               policy_count = &free_group_entries_by_block[block_id];
               policy_cap = free_group_entry_cap;
            }
            if (policy_count != nullptr && *policy_count >= policy_cap)
            {
               stats.special_group_entry_points_policy_skipped++;
               continue;
            }
            const IdxType point_id =
                (local == 0 && group_id < _group_entry_points.size() &&
                 _group_entry_points[group_id] >= range.first && _group_entry_points[group_id] < range.second)
                    ? _group_entry_points[group_id]
                    : range.first + local;
            if (add_entry_point(point_id, covered_root ? uint8_t{1} : uint8_t{0}))
            {
               if (policy_count != nullptr)
                  ++(*policy_count);
               if (covered_root)
                  stats.special_group_entry_free_points++;
               else
                  stats.special_group_entry_regular_points++;
            }
         }
         if (trie_block_portals && covered_root &&
             group_id < _group_id_to_special_block.size())
         {
            const IdxType block_id = _group_id_to_special_block[group_id];
            if (block_id > 0 && block_id <= _special_blocks.size() &&
                (!lazy_block_activation || query_seed_block[block_id] != 0) &&
                entry_portal_seen[block_id] == 0)
            {
               entry_portal_seen[block_id] = 1;
               stats.special_trie_block_portals_scanned++;
               const IdxType entry_point = _special_blocks[block_id - 1].entry_point_id;
               if (entry_point != SpecialBlock::kInvalidEntryPoint &&
                   entry_point < _num_points && add_entry_point(entry_point, 1))
                  stats.special_trie_block_portals_accepted++;
            }
         }
      }

      if (entry_point_ids.empty())
      {
         stats.num_distance_calcs = 0;
         for (IdxType point_id : free_state_touched)
         {
            if (point_id < free_state_cache.size())
               free_state_cache[point_id] = 0;
         }
         free_state_touched.clear();
         return false;
      }

      std::vector<SpecialSearchCandidate> block_seed_candidates;
      std::vector<SpecialSearchCandidate> other_entry_candidates;
      block_seed_candidates.reserve(stats.special_block_seed_points);
      other_entry_candidates.reserve(entry_point_ids.size() - stats.special_block_seed_points);
      for (size_t i = 0; i < entry_point_ids.size(); ++i)
      {
         const IdxType point_id = entry_point_ids[i];
         SpecialSearchCandidate candidate{point_id,
                                          score_distance(point_id),
                                          entry_activation_level[i]};
         if (entry_is_block_seed[i] != 0)
            block_seed_candidates.push_back(candidate);
         else
            other_entry_candidates.push_back(candidate);
      }

      auto retain_closest = [&](std::vector<SpecialSearchCandidate> &candidates, size_t keep) {
         if (candidates.size() <= keep)
            return;
         auto keep_end = candidates.begin() + static_cast<std::ptrdiff_t>(keep);
         std::nth_element(candidates.begin(), keep_end, candidates.end(), special_candidate_less);
         candidates.resize(keep);
      };
      if (block_seeds_retained_per_block < block_seeds_per_block &&
          !block_seed_candidates.empty())
      {
         // Route each block through its query-nearest landmark without widening the search beam.
         std::vector<std::vector<SpecialSearchCandidate>> candidates_by_block(
             _special_blocks.size() + 1);
         for (const SpecialSearchCandidate &candidate : block_seed_candidates)
         {
            const IdxType block_id =
                candidate.id < _point_to_special_block.size()
                    ? _point_to_special_block[candidate.id]
                    : 0;
            if (block_id > 0 && block_id < candidates_by_block.size())
               candidates_by_block[block_id].push_back(candidate);
         }
         block_seed_candidates.clear();
         for (IdxType block_id = 1; block_id < candidates_by_block.size(); ++block_id)
         {
            std::vector<SpecialSearchCandidate> &block_candidates = candidates_by_block[block_id];
            const size_t keep = query_free_block_frontier[block_id] != 0
                                    ? frontier_block_seeds_retained_per_block
                                    : block_seeds_retained_per_block;
            retain_closest(block_candidates, keep);
            block_seed_candidates.insert(block_seed_candidates.end(),
                                         block_candidates.begin(), block_candidates.end());
         }
      }
      const size_t entry_capacity = static_cast<size_t>(capacity);
      const size_t seed_capacity = std::min(entry_capacity, block_seed_cap);
      retain_closest(block_seed_candidates, seed_capacity);
      stats.special_block_seed_points_retained = block_seed_candidates.size();
      const size_t remaining_capacity = entry_capacity - block_seed_candidates.size();
      retain_closest(other_entry_candidates, remaining_capacity);

      std::vector<SpecialSearchCandidate> initial_candidates;
      initial_candidates.reserve(block_seed_candidates.size() + other_entry_candidates.size());
      initial_candidates.insert(initial_candidates.end(),
                                block_seed_candidates.begin(), block_seed_candidates.end());
      initial_candidates.insert(initial_candidates.end(),
                                other_entry_candidates.begin(), other_entry_candidates.end());
      candidate_queue.initialize(std::move(initial_candidates));
      const std::vector<SpecialSearchCandidate> retained_initial_candidates =
          candidate_queue.sorted_results();

      std::vector<uint8_t> entry_blocks(_special_blocks.size() + 1, 0);
      std::vector<uint8_t> retained_entry_blocks(_special_blocks.size() + 1, 0);
      for (IdxType point_id : entry_point_ids)
      {
         if (point_id >= _point_to_special_block.size())
            continue;
         const IdxType block_id = _point_to_special_block[point_id];
         if (block_id > 0 && block_id < entry_blocks.size())
            entry_blocks[block_id] = 1;
      }
      for (const SpecialSearchCandidate &candidate : retained_initial_candidates)
      {
         if (candidate.id >= _point_to_special_block.size())
            continue;
         const IdxType block_id = _point_to_special_block[candidate.id];
         if (block_id > 0 && block_id < retained_entry_blocks.size())
            retained_entry_blocks[block_id] = 1;
      }
      stats.special_entry_blocks = static_cast<size_t>(
          std::count(entry_blocks.begin(), entry_blocks.end(), uint8_t{1}));
      stats.special_retained_entry_blocks = static_cast<size_t>(
          std::count(retained_entry_blocks.begin(), retained_entry_blocks.end(), uint8_t{1}));
      stats.special_entry_time_ms = elapsed_ms(entry_time_start);
      stats.entry_point_setup_time_ms = stats.special_entry_time_ms;

      IdxType comparisons = static_cast<IdxType>(entry_point_ids.size());
      size_t queue_insertions_total = 0;
      if (detail_stats)
      {
         stats.special_free_distance_calcs += stats.special_entry_free_points;
         stats.special_regular_distance_calcs += stats.special_entry_regular_points;
      }
      auto insert_candidate = [&](IdxType id, float distance,
                                  uint8_t activation_level) {
         if (detail_stats)
            stats.special_queue_insert_attempts++;
         const SpecialCandidateInsertResult result =
             candidate_queue.insert(id, distance, activation_level);
         if (detail_stats)
         {
            if (result == SpecialCandidateInsertResult::Inserted)
            {
               stats.special_queue_insertions++;
               stats.special_queue_shifted_candidates +=
                   candidate_queue.last_shifted_candidates();
            }
            else
               stats.special_queue_bound_rejections++;
         }
         if (result == SpecialCandidateInsertResult::Inserted)
            ++queue_insertions_total;
      };
      auto visit_neighbor = [&](IdxType neighbor, uint8_t next_level,
                                uint8_t from_level) {
         if (neighbor >= _num_points)
            return false;
         VisitedSet &visited = visited_for_level(next_level);
         if (visited.check(neighbor))
            return false;
         visited.set(neighbor);
         if (next_level > 0)
         {
            if (detail_stats)
            {
               stats.special_free_candidates_inserted++;
               if (next_level > from_level)
               {
                  stats.special_free_upgrades++;
                  if (from_level == 0)
                     stats.special_middle_activations++;
                  if (from_level < 2 && next_level >= 2)
                     stats.special_upper_activations++;
               }
            }
         }
         else
         {
            if (detail_stats)
               stats.special_regular_candidates_inserted++;
         }
         stats.num_nodes_visited++;
         insert_candidate(neighbor, score_distance(neighbor), next_level);
         if (detail_stats)
         {
            if (next_level > 0)
               stats.special_free_distance_calcs++;
            else
               stats.special_regular_distance_calcs++;
         }
         comparisons++;
         return true;
      };

      const bool preexpand_block_seeds =
          seed_free_blocks && ung_env_flag_enabled("UNG_SPECIAL_TRIE_PREEXPAND_BLOCK_SEEDS");
      if (preexpand_block_seeds)
      {
         std::chrono::high_resolution_clock::time_point preexpand_time_start;
         if (profile_timing)
            preexpand_time_start = std::chrono::high_resolution_clock::now();
         const char *preexpand_per_block_value =
             std::getenv("UNG_SPECIAL_TRIE_BLOCK_SEEDS_PREEXPAND_PER_BLOCK");
         const bool preexpand_per_block_enabled = preexpand_per_block_value != nullptr;
         const size_t preexpand_per_block = preexpand_per_block_enabled
                                                ? static_cast<size_t>(std::strtoull(
                                                      preexpand_per_block_value, nullptr, 10))
                                                : std::numeric_limits<size_t>::max();
         // Give every routed block one navigation step before global distance competition.
         auto preexpand_edges = [&](SpecialEdgeView edges, bool heavy,
                                    size_t &inter_edges_seen,
                                    uint8_t source_level) {
            for (const SpecialEdge edge : edges)
            {
               if (edge.special_block_id == 0 ||
                   edge.special_block_id > _special_blocks.size())
                  continue;
               const SpecialBlock &edge_owner =
                   _special_blocks[edge.special_block_id - 1];
               const bool query_covers_owner =
                   edge.special_block_id < query_free_block.size() &&
                   query_free_block[edge.special_block_id] != 0;
               const SpecialBlockEdgeTransition transition =
                   special_block_edge_transition(source_level, edge_owner,
                                                 query_covers_owner);
               if (!transition.allowed)
                  continue;
               const uint8_t edge_level = special_block_activation_level(edge_owner);
               if (edge.kind == SpecialEdgeKind::InterBlock &&
                   inter_edges_seen >= free_inter_edge_scan_cap)
               {
                  if (detail_stats)
                     stats.special_free_inter_edges_cap_skipped++;
                  continue;
               }
               if (edge.kind == SpecialEdgeKind::InterBlock)
                  ++inter_edges_seen;
               if (detail_stats)
               {
                  stats.special_preexpand_edges_scanned++;
                  stats.special_edges_scanned++;
                  if (edge_level >= 2)
                     stats.special_upper_edges_scanned++;
                  else
                     stats.special_middle_edges_scanned++;
                  if (heavy)
                     stats.special_heavy_edges_scanned++;
                  if (edge.kind == SpecialEdgeKind::InterBlock)
                     stats.special_inter_edges_scanned++;
                  else
                     stats.special_intra_edges_scanned++;
               }
               if (visit_neighbor(edge.target_point_id,
                                  transition.successor_activation_level,
                                  source_level) && detail_stats)
               {
                  stats.special_preexpand_edges_accepted++;
                  stats.special_edges_accepted++;
                  if (heavy)
                     stats.special_heavy_edges_accepted++;
               }
            }
         };

         std::vector<IdxType> preexpanded_seed_ids;
         preexpanded_seed_ids.reserve(block_seed_candidates.size());
         std::vector<SpecialSearchCandidate> preexpand_candidates = block_seed_candidates;
         std::sort(preexpand_candidates.begin(), preexpand_candidates.end(), special_candidate_less);
         std::vector<size_t> preexpanded_by_block(
             preexpand_per_block_enabled ? _special_blocks.size() + 1 : 0, 0);
         for (const SpecialSearchCandidate &seed : preexpand_candidates)
         {
            const IdxType block_id =
                seed.id < _point_to_special_block.size()
                    ? _point_to_special_block[seed.id]
                    : 0;
            if (preexpand_per_block_enabled && block_id > 0 &&
                block_id < preexpanded_by_block.size())
            {
               if (preexpanded_by_block[block_id] >= preexpand_per_block)
                  continue;
               preexpanded_by_block[block_id]++;
            }
            preexpanded_seed_ids.push_back(seed.id);
            stats.special_block_seed_points_preexpanded++;
            if (block_id > 0)
            {
               if (block_id < searched_blocks.size() &&
                   searched_blocks[block_id] == 0)
               {
                  searched_blocks[block_id] = 1;
                  stats.special_blocks_searched++;
                  if (detail_stats)
                  {
                     if (_special_blocks[block_id - 1].level == 0)
                        stats.special_middle_blocks_searched++;
                     else
                        stats.special_upper_blocks_searched++;
                  }
               }
            }
            if (detail_stats)
            {
               stats.special_free_nodes_expanded++;
               if (seed.activation_level >= 2)
                  stats.special_upper_nodes_expanded++;
               else
                  stats.special_middle_nodes_expanded++;
            }
            size_t preexpand_inter_edges_seen = 0;
            preexpand_edges(special_edges_for_point(seed.id), false,
                            preexpand_inter_edges_seen, seed.activation_level);
            if (stats.special_heavy_edges_enabled)
               preexpand_edges(special_heavy_edges_for_point(seed.id), true,
                               preexpand_inter_edges_seen, seed.activation_level);
         }
         std::sort(preexpanded_seed_ids.begin(), preexpanded_seed_ids.end());
         candidate_queue.mark_expanded_ids(preexpanded_seed_ids);
         if (profile_timing)
            stats.special_preexpand_time_ms = elapsed_ms(preexpand_time_start);
      }
      const size_t early_stop_min_nodes = static_cast<size_t>(std::max<IdxType>(runtime.K, runtime.Lsearch));
      while (candidate_queue.has_unexpanded())
      {
         const bool can_consider_early_stop =
             runtime.K > 0 && candidate_queue.size() >= static_cast<size_t>(runtime.K);
         float best_unexpanded = 0.0f;
         float second_unexpanded = 0.0f;
         if (runtime.special_block_early_stop && can_consider_early_stop &&
             candidate_queue.peek_two_unexpanded(best_unexpanded, second_unexpanded) &&
             should_stop_special_block_search(runtime.special_block_early_stop,
                                              candidate_queue.size(),
                                              runtime.K,
                                              best_unexpanded,
                                              second_unexpanded,
                                              candidate_queue.kth_distance(),
                                              stats.num_nodes_visited,
                                              early_stop_min_nodes))
            break;

         SpecialSearchCandidate cur;
         if (!candidate_queue.pop_closest_unexpanded(cur))
            break;

         if (cur.free() && cur.id < _point_to_special_block.size())
         {
            const IdxType block_id = cur.activation_level >= 2 &&
                                             cur.id < _point_to_upper_special_block.size()
                                         ? _point_to_upper_special_block[cur.id]
                                         : _point_to_special_block[cur.id];
            if (block_id > 0 && block_id < searched_blocks.size() &&
                searched_blocks[block_id] == 0)
            {
               searched_blocks[block_id] = 1;
               stats.special_blocks_searched++;
               if (detail_stats)
               {
                  if (_special_blocks[block_id - 1].level == 0)
                     stats.special_middle_blocks_searched++;
                  else
                     stats.special_upper_blocks_searched++;
               }
            }
         }

         auto activate_trie_block_portal = [&](IdxType member_point) {
            if (!trie_block_portals || member_point >= _point_to_special_block.size())
               return;
            stats.special_trie_block_portals_scanned++;
            const IdxType block_id = _point_to_special_block[member_point];
            if (block_id == 0 || block_id > _special_blocks.size() ||
                block_id >= query_free_block.size() || query_free_block[block_id] == 0)
               return;
            const IdxType entry_point = _special_blocks[block_id - 1].entry_point_id;
            if (entry_point != SpecialBlock::kInvalidEntryPoint &&
                entry_point < _num_points &&
                visit_neighbor(entry_point,
                               std::max<uint8_t>(cur.activation_level, 1),
                               cur.activation_level))
               stats.special_trie_block_portals_accepted++;
         };

         if (detail_stats)
         {
            if (cur.free())
            {
               stats.special_free_nodes_expanded++;
               if (cur.activation_level >= 2)
                  stats.special_upper_nodes_expanded++;
               else
                  stats.special_middle_nodes_expanded++;
            }
            else
               stats.special_regular_nodes_expanded++;
         }

         if (cur.free())
         {
            bool scan_free_edges = true;
            IdxType current_free_block = 0;
            size_t block_insertions_before = queue_insertions_total;
            if (cur.activation_level >= 2 &&
                cur.id < _point_to_upper_special_block.size())
               current_free_block = _point_to_upper_special_block[cur.id];
            else if (cur.id < _point_to_special_block.size())
               current_free_block = _point_to_special_block[cur.id];
            if (free_block_stall_limit > 0 && current_free_block > 0 &&
                current_free_block < free_block_paused.size() &&
                free_block_paused[current_free_block] != 0)
               scan_free_edges = false;
            if (!free_nodes_expanded_by_block.empty() &&
                current_free_block > 0)
            {
               const IdxType block_id = current_free_block;
               if (block_id > 0 && block_id < free_nodes_expanded_by_block.size())
               {
                  if (free_nodes_expanded_by_block[block_id] >=
                      free_node_expansions_per_block)
                  {
                     scan_free_edges = false;
                     if (detail_stats)
                        stats.special_free_node_cap_skipped++;
                  }
                  else
                  {
                     free_nodes_expanded_by_block[block_id]++;
                  }
               }
            }
            if (scan_free_edges)
            {
               const bool cap_high_fanout_block =
                   free_inter_edge_cap_min_child_blocks == 0 ||
                   (current_free_block > 0 &&
                    current_free_block <= _special_blocks.size() &&
                    _special_blocks[current_free_block - 1].child_block_ids.size() >=
                        free_inter_edge_cap_min_child_blocks);
               const size_t active_inter_edge_scan_cap =
                   cap_high_fanout_block
                       ? free_inter_edge_scan_cap
                       : std::numeric_limits<size_t>::max();
               const size_t active_inter_edge_per_block_cap =
                   cap_high_fanout_block
                       ? free_inter_edge_per_block_cap
                       : std::numeric_limits<size_t>::max();
               size_t inter_edges_seen_for_node = 0;
               inter_target_blocks_touched.clear();
               auto scan_special_edges = [&](SpecialEdgeView edges, bool heavy) {
                  if (edges.empty())
                     return;
                  std::chrono::high_resolution_clock::time_point special_edges_time_start;
                  if (profile_timing)
                     special_edges_time_start = std::chrono::high_resolution_clock::now();
                  auto &gpu_ids = search_cache->gpu_distance_ids;
                  auto &gpu_distances = search_cache->gpu_distance_values;
                  gpu_ids.clear();
                  // The legacy batch scratch stores point ids only. Until it
                  // carries one activation level per edge, keep multi-level
                  // transitions on the scalar path so a level-2 edge cannot
                  // be silently downgraded to level 1.
                  const bool try_gpu_batch = gpu_free_distance &&
                                             _special_block_summary.upper_blocks == 0 &&
                                             edges.size() >= gpu_free_distance_min;
                  if (try_gpu_batch)
                     gpu_ids.reserve(edges.size());
                  auto process_edge_range = [&](size_t begin, size_t end) {
                     for (size_t edge_idx = begin; edge_idx < end; ++edge_idx)
                     {
                        const SpecialEdge edge = edges[edge_idx];
                        if (edge.special_block_id == 0 ||
                            edge.special_block_id > _special_blocks.size())
                           continue;
                        const SpecialBlock &edge_owner =
                            _special_blocks[edge.special_block_id - 1];
                        const bool query_covers_owner =
                            edge.special_block_id < query_free_block.size() &&
                            query_free_block[edge.special_block_id] != 0;
                        const SpecialBlockEdgeTransition transition =
                            special_block_edge_transition(cur.activation_level,
                                                         edge_owner,
                                                         query_covers_owner);
                        if (!transition.allowed)
                           continue;
                        const uint8_t edge_activation_level =
                            special_block_activation_level(edge_owner);
                        const uint8_t next_level =
                            transition.successor_activation_level;
                        if (edge.kind == SpecialEdgeKind::InterBlock &&
                            inter_edges_seen_for_node >= active_inter_edge_scan_cap)
                        {
                           if (detail_stats)
                              stats.special_free_inter_edges_cap_skipped++;
                           continue;
                        }
                        if (edge.kind == SpecialEdgeKind::InterBlock &&
                            !inter_edges_seen_by_target_block.empty())
                        {
                           const IdxType target_block = special_edge_target_owner(
                               edge, _special_blocks, _point_to_special_block,
                               _point_to_upper_special_block);
                           if (target_block > 0 &&
                               target_block < inter_edges_seen_by_target_block.size())
                           {
                              size_t &target_count =
                                  inter_edges_seen_by_target_block[target_block];
                              if (target_count >= active_inter_edge_per_block_cap)
                              {
                                 if (detail_stats)
                                    stats.special_free_inter_edges_cap_skipped++;
                                 continue;
                              }
                              if (target_count == 0)
                                 inter_target_blocks_touched.push_back(target_block);
                              ++target_count;
                           }
                        }
                        if (edge.kind == SpecialEdgeKind::InterBlock)
                           ++inter_edges_seen_for_node;
                        if (edge_idx + 1 < end)
                           prefetch_point(edges[edge_idx + 1].target_point_id);
                        if (detail_stats)
                        {
                           stats.special_edges_scanned++;
                           if (edge_activation_level >= 2)
                              stats.special_upper_edges_scanned++;
                           else
                              stats.special_middle_edges_scanned++;
                           if (heavy)
                              stats.special_heavy_edges_scanned++;
                           if (edge.kind == SpecialEdgeKind::InterBlock)
                              stats.special_inter_edges_scanned++;
                           else
                              stats.special_intra_edges_scanned++;
                        }
                        const IdxType neighbor = edge.target_point_id;
                        if (!try_gpu_batch)
                        {
                           if (visit_neighbor(
                                   neighbor,
                                   next_level, cur.activation_level))
                           {
                              if (detail_stats)
                              {
                                 stats.special_edges_accepted++;
                                 if (heavy)
                                    stats.special_heavy_edges_accepted++;
                              }
                           }
                           continue;
                        }
                        if (neighbor >= _num_points ||
                            visited_for_level(next_level).check(neighbor))
                           continue;
                        visited_for_level(next_level).set(neighbor);
                        gpu_ids.push_back(neighbor);
                     }
                  };

                  if (free_intra_edge_scan_cap == std::numeric_limits<size_t>::max())
                  {
                     process_edge_range(0, edges.size());
                  }
                  else
                  {
                     // Construction appends every block-local edge before its inter-block edges.
                     size_t intra_count = 0;
                     while (intra_count < edges.size() &&
                            edges[intra_count].kind == SpecialEdgeKind::IntraBlock)
                        ++intra_count;
                     const size_t intra_kept = std::min(intra_count, free_intra_edge_scan_cap);
                     if (detail_stats)
                        stats.special_free_edges_cap_skipped += intra_count - intra_kept;
                     process_edge_range(0, intra_kept);
                     process_edge_range(intra_count, edges.size());
                  }
                  if (try_gpu_batch && !gpu_ids.empty())
                  {
                     gpu_distances.resize(gpu_ids.size());
                     const bool gpu_ok = gpu_l2_batch_compute_float(gpu_base_vectors, _num_points, dim,
                                                                     gpu_query_vector, gpu_ids.data(),
                                                                     gpu_ids.size(), gpu_distances.data());
                     for (size_t batch_idx = 0; batch_idx < gpu_ids.size(); ++batch_idx)
                     {
                        const IdxType neighbor = gpu_ids[batch_idx];
                        if (detail_stats)
                           stats.special_free_candidates_inserted++;
                        stats.num_nodes_visited++;
                        insert_candidate(neighbor,
                                         gpu_ok ? gpu_distances[batch_idx] : exact_distance(neighbor),
                                         cur.activation_level);
                        if (detail_stats)
                           stats.special_free_distance_calcs++;
                        comparisons++;
                        if (detail_stats)
                        {
                           stats.special_edges_accepted++;
                           if (heavy)
                              stats.special_heavy_edges_accepted++;
                        }
                     }
                     if (!gpu_ok)
                        gpu_free_distance = false;
                  }
                  if (profile_timing)
                     stats.special_special_edges_time_ms += elapsed_ms(special_edges_time_start);
               };
               scan_special_edges(special_edges_for_point(cur.id), false);
               if (stats.special_heavy_edges_enabled)
                  scan_special_edges(special_heavy_edges_for_point(cur.id), true);
               for (IdxType target_block : inter_target_blocks_touched)
                  inter_edges_seen_by_target_block[target_block] = 0;
            }
            if (free_block_stall_limit > 0 && current_free_block > 0 &&
                current_free_block < free_block_stall_counts.size())
            {
               if (!scan_free_edges || queue_insertions_total == block_insertions_before)
               {
                  size_t &stall_count = free_block_stall_counts[current_free_block];
                  ++stall_count;
                  if (stall_count >= free_block_stall_limit)
                     free_block_paused[current_free_block] = 1;
               }
               else
               {
                  free_block_stall_counts[current_free_block] = 0;
               }
            }
            if (!runtime.special_block_free_use_regular)
               continue;
         }

         std::chrono::high_resolution_clock::time_point regular_edges_time_start;
         if (profile_timing)
            regular_edges_time_start = std::chrono::high_resolution_clock::now();
         const GraphNeighborView neighbors = graph_backend.neighbors(cur.id);
         auto process_regular_neighbor = [&](IdxType neighbor) {
            if (detail_stats)
               stats.special_regular_edges_scanned++;
            uint8_t next_level = cur.activation_level;
            if (next_level == 0)
               next_level = cached_point_activation_level(neighbor);
            else
               next_level = point_upper_activation_level(neighbor, next_level);
            const bool accepted = visit_neighbor(neighbor, next_level,
                                                 cur.activation_level);
            if (accepted && cur.activation_level == 0 && next_level > 0)
               activate_trie_block_portal(neighbor);
            if (accepted)
               if (detail_stats)
                  stats.special_regular_edges_accepted++;
            return accepted;
         };
         size_t i = 0;
         for (; i + 1 < neighbors.size; i += 2)
         {
            prefetch_point(neighbors.ids[i]);
            prefetch_point(neighbors.ids[i + 1]);
            if (i + 2 < neighbors.size)
               prefetch_point(neighbors.ids[i + 2]);
            const IdxType first = neighbors.ids[i];
            const IdxType second = neighbors.ids[i + 1];
            if (!trie_regular_search ||
                special_trie_regular_main_edge_allowed(_new_vec_id_to_group_id, cur.id, first))
               process_regular_neighbor(first);
            else if (detail_stats)
               stats.special_lng_cross_edges_skipped++;
            if (!trie_regular_search ||
                special_trie_regular_main_edge_allowed(_new_vec_id_to_group_id, cur.id, second))
               process_regular_neighbor(second);
            else if (detail_stats)
               stats.special_lng_cross_edges_skipped++;
         }
         for (; i < neighbors.size; ++i)
         {
            prefetch_point(neighbors.ids[i]);
            const IdxType neighbor = neighbors.ids[i];
            if (!trie_regular_search ||
                special_trie_regular_main_edge_allowed(_new_vec_id_to_group_id, cur.id, neighbor))
               process_regular_neighbor(neighbor);
            else if (detail_stats)
               stats.special_lng_cross_edges_skipped++;
         }
         if (trie_regular_search)
         {
            for (IdxType neighbor : special_regular_edges_for_point(cur.id))
            {
               if (detail_stats)
                  stats.special_trie_regular_edges_scanned++;
               if (process_regular_neighbor(neighbor) && detail_stats)
                  stats.special_trie_regular_edges_accepted++;
            }
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
      std::vector<SpecialSearchCandidate> final_candidates = candidate_queue.sorted_results();
      if (approx_dims == 0)
      {
         const size_t result_count = std::min<size_t>(static_cast<size_t>(runtime.K),
                                                      final_candidates.size());
         for (size_t i = 0; i < result_count; ++i)
            cur_result.insert(final_candidates[i].id, final_candidates[i].distance);
      }
      else
      {
         std::vector<SpecialSearchCandidate> exact_candidates;
         exact_candidates.reserve(final_candidates.size());
         for (const SpecialSearchCandidate &candidate : final_candidates)
            exact_candidates.push_back(
                SpecialSearchCandidate{candidate.id, exact_distance(candidate.id),
                                       candidate.activation_level});
         const size_t result_count = std::min<size_t>(static_cast<size_t>(runtime.K), exact_candidates.size());
         if (result_count < exact_candidates.size())
         {
            auto kth = exact_candidates.begin() + static_cast<std::ptrdiff_t>(result_count);
            std::nth_element(exact_candidates.begin(), kth, exact_candidates.end(),
                             special_candidate_less);
            exact_candidates.resize(result_count);
         }
         std::sort(exact_candidates.begin(), exact_candidates.end(), special_candidate_less);
         for (const SpecialSearchCandidate &candidate : exact_candidates)
            cur_result.insert(candidate.id, candidate.distance);
      }
      if (profile_timing)
         stats.special_result_time_ms = elapsed_ms(result_time_start);
      num_cmps[query_id] = comparisons;
      stats.num_distance_calcs = comparisons;
      stats.search_time_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - search_time_start_ms)
              .count();
      stats.regular_edges_scanned = stats.special_regular_edges_scanned;
      stats.free_edges_scanned = stats.special_edges_scanned;
      for (IdxType point_id : free_state_touched)
      {
         if (point_id < free_state_cache.size())
            free_state_cache[point_id] = 0;
      }
      free_state_touched.clear();
      return true;
   }

   bool UniNavGraph::execute_favor_block_ung_query(const char *query,
                                                           std::shared_ptr<SearchCache> search_cache,
                                                           const SearchRuntimeConfig &runtime,
                                                           const GraphSearchBackend &graph_backend,
                                                           const std::vector<IdxType> &entry_group_ids,
                                                           IdxType query_id,
                                                           std::vector<float> &num_cmps,
                                                           SearchQueue &cur_result,
                                                           QueryStats &stats)
   {
      struct FavorCandidate
      {
         IdxType id = 0;
         float distance = 0.0f;
         float raw_distance = 0.0f;
         bool target = false;
      };
      struct FavorCandidateMinHeap
      {
         bool operator()(const FavorCandidate &a, const FavorCandidate &b) const
         {
            if (a.distance != b.distance)
               return a.distance > b.distance;
            return a.id > b.id;
         }
      };
      struct FavorWorstCandidate
      {
         bool operator()(const FavorCandidate &a, const FavorCandidate &b) const
         {
            if (a.distance != b.distance)
               return a.distance < b.distance;
            return a.id < b.id;
         }
      };

      auto env_u32 = [](const char *name, uint32_t fallback) {
         if (const char *value = std::getenv(name))
            return static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
         return fallback;
      };
      auto env_float = [](const char *name, float fallback) {
         if (const char *value = std::getenv(name))
            return std::strtof(value, nullptr);
         return fallback;
      };

      auto search_time_start_ms = std::chrono::high_resolution_clock::now();
      stats.special_search_enabled = true;
      stats.favor_block_search_enabled = true;
      stats.special_free_use_regular = false;

      const IdxType dim = _base_storage->get_dim();
      const IdxType capacity = runtime.Lsearch;
      if (capacity == 0 || entry_group_ids.empty())
      {
         stats.num_distance_calcs = 0;
         return false;
      }

      roaring::Roaring target_bitmap = compute_bitmap_from_groups(entry_group_ids);
      stats.favor_target_points = target_bitmap.cardinality();
      stats.favor_selectivity = _num_points == 0 ? 0.0f :
          static_cast<float>(stats.favor_target_points) / static_cast<float>(_num_points);
      const float delta_d = env_float("UNG_FAVOR_DELTA_D", 1.0f);
      const float alpha = env_float("UNG_FAVOR_ALPHA", 1.0f);
      stats.favor_exclusion_distance = favor_exclusion_distance(
          stats.favor_selectivity, static_cast<float>(std::max<IdxType>(capacity, 1)), delta_d, alpha);

      auto &target_map = search_cache->favor_target_map;
      auto &target_touched = search_cache->favor_target_touched;
      if (target_map.size() < _num_points)
         target_map.assign(_num_points, 0);
      for (IdxType point_id : target_touched)
         if (point_id < target_map.size())
            target_map[point_id] = 0;
      target_touched.clear();
      for (uint32_t original_id : target_bitmap)
      {
         if (original_id < _old_to_new_vec_ids.size())
         {
            const IdxType new_id = _old_to_new_vec_ids[original_id];
            if (new_id < target_map.size() && target_map[new_id] == 0)
            {
               target_map[new_id] = 1;
               target_touched.push_back(new_id);
            }
         }
      }

      const uint32_t block_depth = env_u32("UNG_FAVOR_BLOCK_DEPTH", 1);
      const uint32_t max_blocks = env_u32("UNG_FAVOR_MAX_BLOCKS", 48);
      const uint32_t block_source_budget = env_u32("UNG_FAVOR_BLOCK_SOURCE_BUDGET", 16);
      auto &selected_block = search_cache->favor_selected_blocks;
      if (selected_block.size() < _special_blocks.size() + 1)
         selected_block.assign(_special_blocks.size() + 1, 0);
      else
         std::fill(selected_block.begin(), selected_block.end(), 0);
      std::vector<size_t> block_scores(_special_blocks.size() + 1, 0);
      std::vector<IdxType> frontier_blocks;
      auto add_block = [&](IdxType block_id, size_t score) {
         if (block_id == 0 || block_id >= selected_block.size())
            return;
         block_scores[block_id] += std::max<size_t>(score, 1);
         if (selected_block[block_id] != 0)
            return;
         selected_block[block_id] = 1;
         frontier_blocks.push_back(block_id);
      };
      for (IdxType group_id : entry_group_ids)
      {
         if (group_id < _group_id_to_special_block.size())
            add_block(_group_id_to_special_block[group_id], 1);
      }
      for (uint32_t depth = 0; depth < block_depth; ++depth)
      {
         const size_t level_end = frontier_blocks.size();
         for (size_t i = 0; i < level_end; ++i)
         {
            const IdxType block_id = frontier_blocks[i];
            if (block_id == 0 || block_id > _special_blocks.size())
               continue;
            const size_t parent_score = std::max<size_t>(block_scores[block_id], 1);
            for (IdxType child_id : _special_blocks[block_id - 1].child_block_ids)
               add_block(child_id, parent_score);
         }
      }

      std::vector<FavorBlockCandidate> block_candidates;
      block_candidates.reserve(_special_blocks.size());
      for (IdxType block_id = 1; block_id < selected_block.size(); ++block_id)
      {
         if (selected_block[block_id] == 0 || block_id > _special_blocks.size())
            continue;
         block_candidates.push_back(FavorBlockCandidate{block_id, block_scores[block_id],
                                                        _special_blocks[block_id - 1].point_count});
      }
      const std::vector<IdxType> selected_block_ids = favor_select_top_blocks(block_candidates, max_blocks);
      std::fill(selected_block.begin(), selected_block.end(), 0);
      for (IdxType block_id : selected_block_ids)
      {
         if (block_id < selected_block.size())
         {
            selected_block[block_id] = 1;
            stats.favor_block_count++;
            if (block_id > 0 && block_id <= _special_blocks.size())
               stats.favor_block_points += _special_blocks[block_id - 1].point_count;
         }
      }
      const bool has_block_space = stats.favor_block_count > 0;

      auto &block_source_expanded = search_cache->favor_block_source_expanded;
      if (block_source_expanded.size() < selected_block.size())
         block_source_expanded.assign(selected_block.size(), 0);
      else
         std::fill(block_source_expanded.begin(), block_source_expanded.end(), 0);

      auto &visited = search_cache->special_visited_regular;
      visited.clear();
      auto &candidate_active = search_cache->favor_candidate_active;
      if (candidate_active.size() < _num_points)
         candidate_active.assign(_num_points, 0);
      std::vector<FavorCandidate> candidate_heap_storage;
      candidate_heap_storage.reserve(std::max<size_t>(static_cast<size_t>(capacity) * 4, 4096));
      std::vector<FavorCandidate> best_heap_storage;
      best_heap_storage.reserve(static_cast<size_t>(capacity) + 1);
      std::priority_queue<FavorCandidate, std::vector<FavorCandidate>, FavorCandidateMinHeap> candidate_heap(
          FavorCandidateMinHeap{}, std::move(candidate_heap_storage));
      std::priority_queue<FavorCandidate, std::vector<FavorCandidate>, FavorWorstCandidate> best_heap(
          FavorWorstCandidate{}, std::move(best_heap_storage));
      IdxType comparisons = 0;

      auto point_in_selected_space = [&](IdxType id) {
         if (!has_block_space)
            return true;
         if (id >= _point_to_special_block.size())
            return target_map[id] != 0;
         const IdxType block_id = _point_to_special_block[id];
         return target_map[id] != 0 ||
                (block_id > 0 && block_id < selected_block.size() && selected_block[block_id] != 0);
      };

      auto insert_candidate = [&](IdxType id) {
         if (id >= _num_points || visited.check(id))
            return false;
         if (!point_in_selected_space(id))
            return false;
         const bool is_target = target_map[id] != 0;
         const float raw_distance = _distance_handler->compute(query, _base_storage->get_vector(id), dim);
         const float adjusted = favor_adjusted_distance(raw_distance, is_target, stats.favor_exclusion_distance);
         FavorCandidate candidate{id, adjusted, raw_distance, is_target};
         if (best_heap.size() >= static_cast<size_t>(capacity) &&
             best_heap.top().distance < candidate.distance)
            return false;
         visited.set(id);
         candidate_active[id] = 1;
         candidate_heap.push(candidate);
         best_heap.push(candidate);
         if (best_heap.size() > static_cast<size_t>(capacity))
         {
            const FavorCandidate removed = best_heap.top();
            best_heap.pop();
            if (removed.id < candidate_active.size())
               candidate_active[removed.id] = 0;
         }
         comparisons++;
         stats.num_nodes_visited++;
         if (is_target)
            stats.favor_td_candidates_inserted++;
         else
            stats.favor_ntd_candidates_inserted++;
         return true;
      };

      constexpr size_t kReducedEntryGroupThreshold = 1000;
      constexpr IdxType kReducedEntryPointsPerLargeGroupSet = 4;
      const IdxType effective_num_entry_points =
          entry_group_ids.size() > kReducedEntryGroupThreshold
              ? std::min<IdxType>(runtime.num_entry_points, kReducedEntryPointsPerLargeGroupSet)
              : runtime.num_entry_points;
      for (IdxType group_id : entry_group_ids)
      {
         if (group_id >= _group_id_to_range.size())
            continue;
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
            insert_candidate(point_id);
         }
      }

      if (candidate_heap.empty())
      {
         stats.num_distance_calcs = 0;
         return false;
      }

      while (!candidate_heap.empty())
      {
         const FavorCandidate cur = candidate_heap.top();
         candidate_heap.pop();
         if (cur.id >= candidate_active.size() || candidate_active[cur.id] == 0)
            continue;
         if (cur.target)
            stats.special_regular_nodes_expanded++;
         else
            stats.special_free_nodes_expanded++;

         const IdxType block_id = cur.id < _point_to_special_block.size() ? _point_to_special_block[cur.id] : 0;
         if (has_block_space && block_id > 0 && block_id < selected_block.size() &&
             selected_block[block_id] != 0)
         {
            const auto special_edges = special_edges_for_point(cur.id);
            const bool budget_exhausted =
                block_source_budget > 0 && block_source_expanded[block_id] >= block_source_budget;
            if (budget_exhausted)
            {
               stats.favor_block_edge_scans_skipped += special_edges.size();
            }
            else
            {
               block_source_expanded[block_id]++;
               stats.favor_blocks_expanded++;
               for (const SpecialEdge edge : special_edges)
               {
                  stats.special_edges_scanned++;
                  if (edge.kind == SpecialEdgeKind::InterBlock)
                     stats.special_inter_edges_scanned++;
                  else
                     stats.special_intra_edges_scanned++;
                  if (insert_candidate(edge.target_point_id))
                     stats.special_edges_accepted++;
               }
            }
         }

         const GraphNeighborView neighbors = graph_backend.neighbors(cur.id);
         for (size_t i = 0; i < neighbors.size; ++i)
         {
            stats.special_regular_edges_scanned++;
            if (insert_candidate(neighbors.ids[i]))
               stats.special_regular_edges_accepted++;
         }
      }

      cur_result.clear();
      std::vector<FavorCandidate> final_candidates;
      final_candidates.reserve(best_heap.size());
      while (!best_heap.empty())
      {
         final_candidates.push_back(best_heap.top());
         best_heap.pop();
      }
      std::sort(final_candidates.begin(), final_candidates.end(),
                [](const FavorCandidate &a, const FavorCandidate &b) {
                   if (a.raw_distance != b.raw_distance)
                      return a.raw_distance < b.raw_distance;
                   return a.id < b.id;
                });
      for (const FavorCandidate &candidate : final_candidates)
      {
         if (!candidate.target)
            continue;
         cur_result.insert(candidate.id, candidate.raw_distance);
         stats.favor_td_results++;
      }

      stats.special_regular_distance_calcs = stats.favor_td_candidates_inserted;
      stats.special_free_distance_calcs = stats.favor_ntd_candidates_inserted;
      stats.num_distance_calcs = comparisons;
      num_cmps[query_id] = comparisons;
      stats.core_search_time_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - search_time_start_ms)
              .count();
      stats.search_time_ms = stats.core_search_time_ms;
      return stats.favor_td_results > 0;
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
