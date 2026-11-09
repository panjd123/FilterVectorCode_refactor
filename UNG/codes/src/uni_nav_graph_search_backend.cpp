#include "include/uni_nav_graph.h"
#include "include/ung_special_block_activation.h"
#include "include/ung_special_block_router.h"
#include "include/ung_favor_block_search.h"
#include "include/ung_gpu_l2_batch.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <limits>
#include <queue>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

#include <roaring/roaring.hh>


namespace ANNS
{



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
      // Each candidate belongs to exactly one graph layer. Layer-zero
      // candidates use the persisted base graph; materialized-layer candidates
      // use only edges owned by that layer.
      stats.special_free_use_regular = false;
      const bool heavy_query_size_ok =
          runtime.special_heavy_edge_min_query_size == 0 ||
          stats.query_length >= runtime.special_heavy_edge_min_query_size;
      stats.special_heavy_edges_enabled =
          runtime.special_heavy_edge_search &&
          heavy_query_size_ok &&
          runtime.special_heavy_edge_min_entries > 0 &&
          stats.num_entry_points >= runtime.special_heavy_edge_min_entries;
      auto elapsed_ms = [](const std::chrono::high_resolution_clock::time_point &start) {
         return std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - start)
             .count();
      };

      const IdxType dim = _base_storage->get_dim();
      const IdxType capacity = runtime.Lsearch;
      auto read_bounded_router_size = [](const char *name,
                                         size_t default_value,
                                         size_t maximum,
                                         bool allow_zero) {
         const char *text = std::getenv(name);
         if (text == nullptr || text[0] == '\0' || text[0] == '-')
            return default_value;
         char *end = nullptr;
         errno = 0;
         const unsigned long long parsed = std::strtoull(text, &end, 10);
         if (errno != 0 || end == text || *end != '\0' ||
             parsed > maximum || (!allow_zero && parsed == 0))
            return default_value;
         return static_cast<size_t>(parsed);
      };
      const size_t block_router_top_r = read_bounded_router_size(
          "UNG_SPECIAL_BLOCK_ROUTER_TOP_R", 0, 1024, true);
      const size_t block_router_portals_per_block =
          read_bounded_router_size(
              "UNG_SPECIAL_BLOCK_ROUTER_PORTALS_PER_BLOCK", 8, 64,
              false);
      const size_t block_router_lb_shortlist_multiplier =
          read_bounded_router_size(
              "UNG_SPECIAL_BLOCK_ROUTER_LB_SHORTLIST_MULTIPLIER", 4,
              64, false);
      const size_t block_router_recheck_interval =
          read_bounded_router_size(
              "UNG_SPECIAL_BLOCK_ROUTER_RECHECK_INTERVAL", 32,
              1048576, false);
      const bool block_router_use_lower_bound =
          !ung_env_flag_enabled(
              "UNG_SPECIAL_BLOCK_ROUTER_DISABLE_LOWER_BOUND");
      const bool block_router_bounds_available =
          _special_block_router_radii.size() == _special_blocks.size() &&
          _special_block_router_centroids.size() ==
              _special_blocks.size() * static_cast<size_t>(dim);
      const bool block_router_metric_supported =
          _base_storage->get_data_type() == DataType::FLOAT &&
          dynamic_cast<FloatL2DistanceHandler *>(
              _distance_handler.get()) != nullptr;
      bool block_router_enabled =
          block_router_top_r > 0 &&
          runtime.scenario == "containment" &&
          (!block_router_use_lower_bound || block_router_metric_supported) &&
          (!block_router_use_lower_bound ||
           block_router_bounds_available);
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
      IdxType approx_dims = 0;
      if (const char *value = std::getenv("UNG_SPECIAL_APPROX_DIMS"))
         approx_dims = static_cast<IdxType>(std::strtoull(value, nullptr, 10));
      if (approx_dims >= dim)
         approx_dims = 0;
      if (block_router_use_lower_bound && approx_dims > 0)
         block_router_enabled = false;
      stats.special_block_router_enabled = block_router_enabled;
      stats.special_block_router_lower_bound_enabled =
          block_router_enabled && block_router_use_lower_bound;
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
      IdxType comparisons = 0;
      auto &distance_cache = search_cache->special_distance_cache;
      auto &distance_valid = search_cache->special_distance_valid;
      auto &distance_touched = search_cache->special_distance_touched;
      if (distance_cache.size() != static_cast<size_t>(_num_points))
      {
         distance_cache.resize(static_cast<size_t>(_num_points));
         distance_valid.assign(static_cast<size_t>(_num_points), 0);
         distance_touched.clear();
         distance_touched.reserve(static_cast<size_t>(capacity));
      }
      else
      {
         for (IdxType point_id : distance_touched)
            distance_valid[point_id] = 0;
         distance_touched.clear();
      }
      auto exact_distance = [&](IdxType point_id) {
         return _distance_handler->compute(query, _base_storage->get_vector(point_id), dim);
      };
      auto score_distance = [&](IdxType point_id, bool &computed) {
         float &cached = distance_cache[point_id];
         if (distance_valid[point_id] != 0)
         {
            computed = false;
            if (detail_stats)
               stats.special_distance_cache_hits++;
            return cached;
         }
         computed = true;
         if (approx_dims == 0)
         {
            cached = exact_distance(point_id);
         }
         else
         {
            const float *q = reinterpret_cast<const float *>(query);
            const float *v = reinterpret_cast<const float *>(_base_storage->get_vector(point_id));
            float sum = 0.0f;
            for (IdxType s = 0; s < approx_dims; ++s)
            {
               const IdxType d = (s * dim) / approx_dims;
               const float diff = q[d] - v[d];
               sum += diff * diff;
            }
            cached = sum * approx_scale;
         }
         distance_valid[point_id] = 1;
         distance_touched.push_back(point_id);
         ++comparisons;
         return cached;
      };
      // Block authorization is a distinct pre-search phase.  Measure it even
      // when the expensive fine-grained profiling counters are disabled so
      // the paper breakdown can remain mutually exclusive and auditable.
      const auto cover_time_start = std::chrono::high_resolution_clock::now();
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
      for (IdxType block_id = 1; block_id < query_free_block.size(); ++block_id)
      {
         for (IdxType child_block_id : _special_blocks[block_id - 1].child_block_ids)
         {
            if (child_block_id > 0 && child_block_id < special_block_parent.size())
               special_block_parent[child_block_id] = block_id;
         }
      }
      for (IdxType block_id = 1; block_id < query_free_block.size(); ++block_id)
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
      // The block router starts only from the highest authorized blocks. A
      // descendant becomes active after its parent selects it, rather than
      // receiving an unconditional query-entry seed.
      const std::vector<uint8_t> &query_seed_block =
          block_router_enabled ? query_free_block_frontier : query_free_block;
      // The exact upper-authorization router runs before this backend.  Add
      std::vector<uint8_t> active_free_block = query_seed_block;
      // coverage propagation to that pre-route time so the public block-
      // authorization stage remains complete for both routed and fallback
      // queries.
      stats.special_cover_time_ms += elapsed_ms(cover_time_start);
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
      auto &visited_regular = search_cache->special_visited(0, _num_points);
      visited_regular.clear();
      auto visited_for_level = [&](uint8_t level) -> VisitedSet & {
         return search_cache->special_visited(level, _num_points);
      };
      for (size_t level = 1;
           level <= _point_to_special_block_by_level.size(); ++level)
         visited_for_level(static_cast<uint8_t>(level)).clear();
      auto point_owner = [&](IdxType point_id, uint8_t activation_level) {
         if (activation_level == 0)
            return IdxType{0};
         const size_t level = static_cast<size_t>(activation_level - 1);
         if (level >= _point_to_special_block_by_level.size() ||
             point_id >= _point_to_special_block_by_level[level].size())
            return IdxType{0};
         return _point_to_special_block_by_level[level][point_id];
      };

      auto prefetch_point = [&](IdxType point_id) {
         if (!runtime.special_block_prefetch || point_id >= _num_points)
            return;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
         visited_regular.prefetch(point_id);
         _mm_prefetch(_base_storage->get_vector(point_id), _MM_HINT_T0);
#endif
      };

      auto &candidate_queue = search_cache->special_candidate_queue;
      const char *candidate_heap_env = std::getenv("UNG_SPECIAL_CANDIDATE_HEAP");
      const bool use_candidate_heap =
          candidate_heap_env == nullptr
              ? _point_to_special_block_by_level.size() > 1
              : ung_env_flag_enabled("UNG_SPECIAL_CANDIDATE_HEAP");
      candidate_queue.reset(static_cast<size_t>(capacity), static_cast<size_t>(runtime.K),
                            use_candidate_heap);

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
      std::vector<IdxType> entry_block_seed_owner;
      entry_point_ids.reserve(entry_reserve);
      entry_activation_level.reserve(entry_reserve);
      entry_is_block_seed.reserve(entry_reserve);
      entry_block_seed_owner.reserve(entry_reserve);

      auto add_entry_point = [&](IdxType point_id, uint8_t activation_level,
                                 IdxType block_seed_owner = 0) {
         const bool is_block_seed = block_seed_owner != 0;
         VisitedSet &visited = visited_for_level(activation_level);
         if (visited.check(point_id))
            return false;

         entry_point_ids.push_back(point_id);
         entry_activation_level.push_back(activation_level);
         entry_is_block_seed.push_back(is_block_seed ? 1 : 0);
         entry_block_seed_owner.push_back(block_seed_owner);
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
      size_t block_seed_cap = std::numeric_limits<size_t>::max();
      if (const char *value = std::getenv("UNG_SPECIAL_TRIE_BLOCK_SEED_CAP"))
      {
         const size_t configured_cap = static_cast<size_t>(std::strtoull(value, nullptr, 10));
         if (configured_cap > 0)
            block_seed_cap = configured_cap;
      }
      size_t block_seeds_per_block = 16;
      if (const char *value = std::getenv("UNG_SPECIAL_TRIE_BLOCK_SEEDS_PER_BLOCK"))
      {
         const size_t configured = static_cast<size_t>(std::strtoull(value, nullptr, 10));
         if (configured > 0)
            block_seeds_per_block = configured;
      }
      std::vector<IdxType> authorized_blocks_high_to_low;
      authorized_blocks_high_to_low.reserve(stats.special_free_block_count);
      for (IdxType block_id = 1; block_id < query_seed_block.size(); ++block_id)
      {
         if (query_seed_block[block_id] == 0 ||
             block_id > _special_blocks.size())
            continue;
         authorized_blocks_high_to_low.push_back(block_id);
      }
      std::sort(authorized_blocks_high_to_low.begin(),
                authorized_blocks_high_to_low.end(),
                [&](IdxType lhs, IdxType rhs) {
                   const SpecialBlock &left = _special_blocks[lhs - 1];
                   const SpecialBlock &right = _special_blocks[rhs - 1];
                   if (left.level != right.level)
                      return left.level > right.level;
                   return lhs < rhs;
                });

      // Generate the original block entry point followed by evenly sampled
      // group landmarks. Blocks are handled from coarse to fine so both
      // pre-expansion and the layer-priority frontier honor hierarchy order.
      for (IdxType block_id : authorized_blocks_high_to_low)
      {
         const SpecialBlock &block = _special_blocks[block_id - 1];
         const uint8_t activation_level = special_block_activation_level(block);
         const IdxType entry_point = block.entry_point_id;
         if (entry_point != SpecialBlock::kInvalidEntryPoint &&
             entry_point < _num_points)
            add_entry_point(entry_point, activation_level, block_id);

         const std::vector<IdxType> &member_groups = block.member_group_ids;
         for (size_t seed_index = 1;
              seed_index < block_seeds_per_block && !member_groups.empty();
              ++seed_index)
         {
            const size_t member_index = std::min(
                member_groups.size() - 1,
                (seed_index * member_groups.size()) / block_seeds_per_block);
            const IdxType group_id = member_groups[member_index];
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
            add_entry_point(landmark, activation_level, block_id);
         }
      }
      for (IdxType group_id : entry_group_ids)
      {
         if (group_id >= _group_id_to_range.size())
            continue;
         const auto &range = _group_id_to_range[group_id];
         if (range.second <= range.first)
            continue;
         std::vector<uint8_t> activation_levels;
         activation_levels.reserve(_group_to_special_block_by_level.size());
         for (size_t level_index = _group_to_special_block_by_level.size();
              level_index > 0; --level_index)
         {
            const auto &owners = _group_to_special_block_by_level[level_index - 1];
            if (group_id >= owners.size())
               continue;
            const IdxType candidate_block = owners[group_id];
            if (candidate_block > 0 && candidate_block < query_free_block.size() &&
                query_free_block[candidate_block] != 0)
            {
               if (!block_router_enabled ||
                   query_free_block_frontier[candidate_block] != 0)
                  activation_levels.push_back(static_cast<uint8_t>(level_index));
            }
         }
         // Entry-group points originate in the persisted base graph. Keep
         // that path in addition to every authorized upward layer.
         activation_levels.push_back(0);
         const bool covered_root = activation_levels.front() != 0;
         const IdxType take = std::min<IdxType>(effective_num_entry_points, range.second - range.first);
         for (IdxType local = 0; local < take; ++local)
         {
            const IdxType point_id =
                (local == 0 && group_id < _group_entry_points.size() &&
                 _group_entry_points[group_id] >= range.first && _group_entry_points[group_id] < range.second)
                    ? _group_entry_points[group_id]
                    : range.first + local;
            bool added_free_state = false;
            bool added_regular_state = false;
            for (uint8_t activation_level : activation_levels)
            {
               if (!add_entry_point(point_id, activation_level))
                  continue;
               if (activation_level > 0)
                  added_free_state = true;
               else
                  added_regular_state = true;
            }
            if (covered_root && added_free_state)
               stats.special_group_entry_free_points++;
            else if (added_regular_state)
               stats.special_group_entry_regular_points++;
         }
      }

      if (entry_point_ids.empty())
      {
         stats.num_distance_calcs = 0;
         return false;
      }

      std::vector<std::vector<SpecialSearchCandidate>> block_seed_candidates_by_block(
          _special_blocks.size() + 1);
      std::vector<SpecialSearchCandidate> other_entry_candidates;
      other_entry_candidates.reserve(entry_point_ids.size() - stats.special_block_seed_points);
      for (size_t i = 0; i < entry_point_ids.size(); ++i)
      {
         const IdxType point_id = entry_point_ids[i];
         bool computed = false;
         SpecialSearchCandidate candidate{point_id,
                                          score_distance(point_id, computed),
                                          entry_activation_level[i]};
         if (computed && detail_stats)
         {
            if (entry_activation_level[i] > 0)
               stats.special_free_distance_calcs++;
            else
               stats.special_regular_distance_calcs++;
         }
         if (entry_is_block_seed[i] != 0)
         {
            const IdxType block_id = entry_block_seed_owner[i];
            if (block_id > 0 &&
                block_id < block_seed_candidates_by_block.size())
               block_seed_candidates_by_block[block_id].push_back(candidate);
         }
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
      size_t block_seeds_retained_per_block = 1;
      if (const char *value =
              std::getenv("UNG_SPECIAL_TRIE_BLOCK_SEEDS_RETAIN_PER_BLOCK"))
      {
         const size_t configured = static_cast<size_t>(std::strtoull(value, nullptr, 10));
         if (configured > 0)
            block_seeds_retained_per_block = configured;
      }
      size_t frontier_block_seeds_retained_per_block = 8;
      if (const char *value = std::getenv(
              "UNG_SPECIAL_TRIE_FRONTIER_BLOCK_SEEDS_RETAIN_PER_BLOCK"))
      {
         const size_t configured = static_cast<size_t>(std::strtoull(value, nullptr, 10));
         if (configured > 0)
            frontier_block_seeds_retained_per_block = configured;
      }
      std::vector<size_t> block_seed_keep_per_block(
          _special_blocks.size() + 1, 0);
      for (IdxType block_id : authorized_blocks_high_to_low)
      {
         size_t keep = query_free_block_frontier[block_id] != 0
                           ? frontier_block_seeds_retained_per_block
                           : block_seeds_retained_per_block;
         block_seed_keep_per_block[block_id] = keep;
      }
      const size_t entry_capacity = static_cast<size_t>(capacity);
      const size_t seed_capacity =
          std::min(entry_capacity, block_seed_cap);
      std::vector<SpecialSearchCandidate> block_seed_candidates =
          select_diverse_block_seed_candidates(
              block_seed_candidates_by_block,
              authorized_blocks_high_to_low,
              block_seed_keep_per_block,
              seed_capacity);
      stats.special_block_seed_points_retained = block_seed_candidates.size();
      const size_t other_entry_capacity = entry_capacity;
      retain_closest(other_entry_candidates, other_entry_capacity);
      std::sort(other_entry_candidates.begin(), other_entry_candidates.end(),
                special_candidate_less);

      // Initialize only the reserved block landmarks. They are pre-expanded
      // before ordinary entries can compete for the unique-point beam.
      candidate_queue.initialize(block_seed_candidates);

      size_t queue_insertions_total = 0;
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
      std::vector<SpecialSearchCandidate> block_preexpand_candidates =
          block_seed_candidates;
      auto visit_neighbor = [&](IdxType neighbor, uint8_t next_level) {
         if (neighbor >= _num_points)
            return false;

         // Materialize the current state and every authorized state above it,
         // in coarse-to-current order. Layer movement is monotone: level 0
         // may promote to 1/2, level 1 may promote to 2, and level 2 never
         // creates a downward level-1/0 path. The vector score is shared by
         // every state of the point.
         const std::vector<uint8_t> activation_levels =
             special_block_monotone_levels_for_point(
                 neighbor, next_level, _point_to_special_block_by_level,
                 active_free_block);

         bool has_unvisited_state = false;
         for (uint8_t activation_level : activation_levels)
            if (!visited_for_level(activation_level).check(neighbor))
            {
               has_unvisited_state = true;
               break;
            }
         if (!has_unvisited_state)
            return false;

         bool computed = false;
         const float distance = score_distance(neighbor, computed);
         bool inserted_state = false;
         for (uint8_t activation_level : activation_levels)
         {
            VisitedSet &visited = visited_for_level(activation_level);
            if (visited.check(neighbor))
               continue;
            visited.set(neighbor);
            inserted_state = true;
            insert_candidate(neighbor, distance, activation_level);
            if (detail_stats)
            {
               if (activation_level != next_level)
                  stats.special_cross_layer_states_inserted++;
               if (activation_level > 0)
                  stats.special_free_candidates_inserted++;
               else
                  stats.special_regular_candidates_inserted++;
            }
         }
         if (!inserted_state)
            return false;
         if (computed)
         {
            stats.num_nodes_visited++;
            if (detail_stats)
            {
               if (activation_levels.front() > 0)
                  stats.special_free_distance_calcs++;
               else
                  stats.special_regular_distance_calcs++;
            }
         }
         return true;
      };

      auto block_router_portal_ids = [&](IdxType block_id) {
         std::vector<IdxType> portal_ids;
         if (block_id == 0 || block_id > _special_blocks.size())
            return portal_ids;
         portal_ids.reserve(block_router_portals_per_block);
         const SpecialBlock &block = _special_blocks[block_id - 1];
         auto append_portal = [&](IdxType point_id) {
            if (point_id >= _num_points ||
                std::find(portal_ids.begin(), portal_ids.end(), point_id) !=
                    portal_ids.end())
               return;
            portal_ids.push_back(point_id);
         };
         if (block.entry_point_id != SpecialBlock::kInvalidEntryPoint)
            append_portal(block.entry_point_id);
         const std::vector<IdxType> &member_groups = block.member_group_ids;
         for (size_t portal_index = 1;
              portal_index < block_router_portals_per_block &&
              !member_groups.empty();
              ++portal_index)
         {
            const size_t member_index = std::min(
                member_groups.size() - 1,
                (portal_index * member_groups.size()) /
                    block_router_portals_per_block);
            const IdxType group_id = member_groups[member_index];
            if (group_id >= _group_id_to_range.size())
               continue;
            const auto &range = _group_id_to_range[group_id];
            if (range.second <= range.first)
               continue;
            const IdxType point_id =
                group_id < _group_entry_points.size() &&
                        _group_entry_points[group_id] >= range.first &&
                        _group_entry_points[group_id] < range.second
                    ? _group_entry_points[group_id]
                    : range.first;
            append_portal(point_id);
         }
         return portal_ids;
      };

      struct ProgressiveBlockRouterState
      {
         bool initialized = false;
         bool complete = false;
         size_t next_lower_bound_child = 0;
         std::vector<SpecialBlockRouterChild> lower_bound_children;
         std::vector<SpecialBlockRouterChild> portal_children;
      };
      std::vector<ProgressiveBlockRouterState> block_router_states(
          _special_blocks.size() + 1);
      std::vector<std::vector<IdxType>> deferred_router_portals_by_block(
          _special_blocks.size() + 1);
      auto remember_deferred_router_portal =
          [&](IdxType block_id, IdxType point_id) {
             if (!block_router_enabled || block_id == 0 ||
                 block_id >= deferred_router_portals_by_block.size() ||
                 point_id >= _num_points)
                return;
             std::vector<IdxType> &deferred =
                 deferred_router_portals_by_block[block_id];
             if (deferred.size() >= block_router_portals_per_block ||
                 std::find(deferred.begin(), deferred.end(), point_id) !=
                     deferred.end())
                return;
             deferred.push_back(point_id);
             ++stats.special_block_router_deferred_portals_stored;
          };

      auto route_child_blocks = [&](IdxType parent_block_id,
                                    bool allow_next_batch) {
         if (!block_router_enabled || parent_block_id == 0 ||
             parent_block_id > _special_blocks.size())
            return false;

         ProgressiveBlockRouterState &state =
             block_router_states[parent_block_id];
         const bool initialize_now = !state.initialized;
         if (initialize_now)
         {
            state.initialized = true;
            const SpecialBlock &parent =
                _special_blocks[parent_block_id - 1];
            const auto lower_bound_start =
                std::chrono::high_resolution_clock::now();
            state.lower_bound_children.reserve(
                parent.child_block_ids.size());
            for (IdxType child_block_id : parent.child_block_ids)
            {
               if (child_block_id == 0 ||
                   child_block_id >= query_free_block.size() ||
                   child_block_id > _special_blocks.size() ||
                   query_free_block[child_block_id] == 0)
                  continue;
               const SpecialBlock &child =
                   _special_blocks[child_block_id - 1];
               if (child.level != parent.level)
                  continue;
               float lower_bound_distance = 0.0f;
               if (block_router_use_lower_bound)
               {
                  const float *centroid =
                      _special_block_router_centroids.data() +
                      static_cast<size_t>(child_block_id - 1) * dim;
                  const float query_centroid_distance_sq =
                      _distance_handler->compute(
                          query,
                          reinterpret_cast<const char *>(centroid), dim);
                  ++comparisons;
                  ++stats.special_block_router_lower_bound_calcs;
                  lower_bound_distance =
                      special_block_router_l2_lower_bound(
                          query_centroid_distance_sq,
                          _special_block_router_radii[child_block_id - 1]);
               }
               state.lower_bound_children.push_back(
                   SpecialBlockRouterChild{
                       child_block_id,
                       std::numeric_limits<float>::infinity(),
                       lower_bound_distance});
               ++stats.special_block_router_children_scored;
            }
            const size_t lower_bound_child_count =
                state.lower_bound_children.size();
            state.lower_bound_children =
                shortlist_special_block_router_children(
                    std::move(state.lower_bound_children),
                    lower_bound_child_count);
            stats.special_block_router_lower_bound_time_ms +=
                elapsed_ms(lower_bound_start);
            if (state.lower_bound_children.empty())
            {
               state.complete = true;
               return false;
            }
            ++stats.special_block_router_blocks_routed;
         }
         else if (!allow_next_batch || state.complete)
         {
            return false;
         }

         const float tau = candidate_queue.kth_distance();
         auto bound_can_improve = [&](const SpecialBlockRouterChild &child) {
            return special_block_router_finite_score(
                       child.lower_bound_distance) < tau;
         };

         state.portal_children.erase(
             std::remove_if(
                 state.portal_children.begin(),
                 state.portal_children.end(),
                 [&](const SpecialBlockRouterChild &child) {
                    return !bound_can_improve(child);
                 }),
             state.portal_children.end());

         while (state.portal_children.empty())
         {
            if (state.next_lower_bound_child >=
                    state.lower_bound_children.size() ||
                !bound_can_improve(
                    state.lower_bound_children[
                        state.next_lower_bound_child]))
            {
               state.complete = true;
               return false;
            }

            const size_t shortlist_window = std::max(
                block_router_top_r,
                block_router_top_r *
                    block_router_lb_shortlist_multiplier);
            const size_t window_end = std::min(
                state.lower_bound_children.size(),
                state.next_lower_bound_child + shortlist_window);
            const auto portal_score_start =
                std::chrono::high_resolution_clock::now();
            for (; state.next_lower_bound_child < window_end;
                 ++state.next_lower_bound_child)
            {
               SpecialBlockRouterChild candidate =
                   state.lower_bound_children[
                       state.next_lower_bound_child];
               if (!bound_can_improve(candidate))
                  break;
               ++stats.special_block_router_children_shortlisted;
               const std::vector<IdxType> portals =
                   block_router_portal_ids(candidate.block_id);
               if (portals.empty())
                  continue;
               for (IdxType portal_id : portals)
               {
                  bool computed = false;
                  const float portal_distance =
                      score_distance(portal_id, computed);
                  const float finite_portal_distance =
                      special_block_router_finite_score(portal_distance);
                  const float finite_best_distance =
                      special_block_router_finite_score(
                          candidate.portal_distance);
                  if (finite_portal_distance < finite_best_distance ||
                      (finite_portal_distance == finite_best_distance &&
                       portal_id < candidate.best_portal_id))
                  {
                     candidate.portal_distance = portal_distance;
                     candidate.best_portal_id = portal_id;
                  }
                  ++stats.special_block_router_portals_scored;
                  if (computed && detail_stats)
                     ++stats.special_free_distance_calcs;
               }
               state.portal_children.push_back(candidate);
               ++stats.special_block_router_portal_children_scored;
            }
            stats.special_block_router_portal_score_time_ms +=
                elapsed_ms(portal_score_start);
            std::sort(state.portal_children.begin(),
                      state.portal_children.end(),
                      special_block_router_child_less);
         }

         std::vector<SpecialBlockRouterChild> selected_children =
             pop_special_block_router_batch(
                 state.portal_children, block_router_top_r);
         stats.special_block_router_children_selected +=
             selected_children.size();
         for (const SpecialBlockRouterChild &selected : selected_children)
         {
            active_free_block[selected.block_id] = 1;
            const SpecialBlock &child =
                _special_blocks[selected.block_id - 1];
            const uint8_t child_activation_level =
                special_block_activation_level(child);
            if (selected.best_portal_id !=
                std::numeric_limits<IdxType>::max())
            {
               if (visit_neighbor(selected.best_portal_id,
                                  child_activation_level))
                  ++stats.special_block_router_portal_activations;
            }
            std::vector<IdxType> &deferred_portals =
                deferred_router_portals_by_block[selected.block_id];
            stats.special_block_router_deferred_portals_replayed +=
                deferred_portals.size();
            for (IdxType portal_id : deferred_portals)
            {
               if (portal_id == selected.best_portal_id)
                  continue;
               if (visit_neighbor(portal_id, child_activation_level))
                  ++stats.special_block_router_portal_activations;
            }
            deferred_portals.clear();
         }
         if (state.portal_children.empty() &&
             (state.next_lower_bound_child >=
                  state.lower_bound_children.size() ||
              !bound_can_improve(
                  state.lower_bound_children[
                      state.next_lower_bound_child])))
            state.complete = true;
         return !selected_children.empty();
      };
      auto advance_pending_block_router_batches = [&]() {
         bool activated = false;
         for (IdxType block_id = 1;
              block_id < block_router_states.size(); ++block_id)
         {
            if (!block_router_states[block_id].initialized ||
                block_router_states[block_id].complete)
               continue;
            if (route_child_blocks(block_id, true))
               activated = true;
         }
         return activated;
      };
      size_t block_seeds_preexpand_per_block = 2;
      if (const char *value = std::getenv(
              "UNG_SPECIAL_TRIE_BLOCK_SEEDS_PREEXPAND_PER_BLOCK"))
      {
         const size_t configured = static_cast<size_t>(std::strtoull(value, nullptr, 10));
         if (configured > 0)
            block_seeds_preexpand_per_block = configured;
      }
      const auto preexpand_time_start = std::chrono::high_resolution_clock::now();
      std::vector<size_t> preexpanded_by_block(_special_blocks.size() + 1, 0);
      std::vector<SpecialSearchCandidate> preexpanded_states;
      preexpanded_states.reserve(block_preexpand_candidates.size());
      for (const SpecialSearchCandidate &seed : block_preexpand_candidates)
      {
         const IdxType block_id = point_owner(seed.id, seed.activation_level);
         if (block_id == 0 || block_id > _special_blocks.size() ||
             preexpanded_by_block[block_id] >= block_seeds_preexpand_per_block)
            continue;
         ++preexpanded_by_block[block_id];
         preexpanded_states.push_back(seed);
         ++stats.special_block_seed_points_preexpanded;
         if (searched_blocks[block_id] == 0)
         {
            searched_blocks[block_id] = 1;
            ++stats.special_blocks_searched;
            if (detail_stats)
            {
               if (_special_blocks[block_id - 1].level == 0)
                  ++stats.special_middle_blocks_searched;
               else
                  ++stats.special_upper_blocks_searched;
            }
         }
         if (detail_stats)
         {
            ++stats.special_free_nodes_expanded;
            if (seed.activation_level >= 2)
               ++stats.special_upper_nodes_expanded;
            else
               ++stats.special_middle_nodes_expanded;
         }

         route_child_blocks(block_id, false);
         auto preexpand_state = [&](const SpecialSearchCandidate &source) {
            size_t intra_edges_seen = 0;
            size_t inter_edges_seen = 0;
            std::vector<size_t> inter_edges_by_target(
                free_inter_edge_per_block_cap !=
                        std::numeric_limits<size_t>::max()
                    ? _special_blocks.size() + 1
                    : 0,
                0);
            auto preexpand_edges = [&](SpecialEdgeView edges, bool heavy) {
            for (size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx)
            {
               const SpecialEdge edge = edges[edge_idx];
               if (edge.special_block_id == 0 ||
                   edge.special_block_id > _special_blocks.size())
                  continue;
               const SpecialBlock &edge_owner =
                   _special_blocks[edge.special_block_id - 1];
               const bool query_covers_owner =
                   edge.special_block_id < active_free_block.size() &&
                   active_free_block[edge.special_block_id] != 0;
               const SpecialBlockEdgeTransition transition =
                   special_block_edge_transition(source.activation_level,
                                                 edge_owner,
                                                 query_covers_owner);
               if (!transition.allowed)
               {
                  if (detail_stats && !query_covers_owner)
                     ++stats.special_inter_edges_coverage_rejected;
                  continue;
               }
               if (edge.kind == SpecialEdgeKind::IntraBlock)
               {
                  if (intra_edges_seen >= free_intra_edge_scan_cap)
                  {
                     if (detail_stats)
                        ++stats.special_free_edges_cap_skipped;
                     continue;
                  }
                  ++intra_edges_seen;
               }
               else
               {
                  if (inter_edges_seen >= free_inter_edge_scan_cap)
                  {
                     if (detail_stats)
                        ++stats.special_free_inter_edges_cap_skipped;
                     continue;
                  }
                  const IdxType target_block = special_edge_target_owner(
                      edge, _special_blocks,
                      _point_to_special_block_by_level);
                  if (block_router_enabled &&
                      (target_block == 0 ||
                       target_block >= active_free_block.size() ||
                       active_free_block[target_block] == 0))
                  {
                     ++stats.special_block_router_inter_edges_skipped;
                     remember_deferred_router_portal(
                         target_block, edge.target_point_id);
                     continue;
                  }
                  if (!inter_edges_by_target.empty() && target_block > 0 &&
                      target_block < inter_edges_by_target.size())
                  {
                     size_t &target_count = inter_edges_by_target[target_block];
                     if (target_count >= free_inter_edge_per_block_cap)
                     {
                        if (detail_stats)
                           ++stats.special_free_inter_edges_cap_skipped;
                        continue;
                     }
                     ++target_count;
                  }
                  ++inter_edges_seen;
               }

               ++stats.special_preexpand_edges_scanned;
               if (detail_stats)
               {
                  ++stats.special_edges_scanned;
                  if (special_block_activation_level(edge_owner) >= 2)
                     ++stats.special_upper_edges_scanned;
                  else
                     ++stats.special_middle_edges_scanned;
                  if (edge.kind == SpecialEdgeKind::InterBlock)
                     ++stats.special_inter_edges_scanned;
                  else
                     ++stats.special_intra_edges_scanned;
                  if (heavy)
                     ++stats.special_heavy_edges_scanned;
               }
               if (visit_neighbor(edge.target_point_id,
                                  transition.successor_activation_level))
               {
                  ++stats.special_preexpand_edges_accepted;
                  if (detail_stats)
                  {
                     ++stats.special_edges_accepted;
                     if (heavy)
                        ++stats.special_heavy_edges_accepted;
                  }
               }
            }
            };
            preexpand_edges(special_edges_for_point(source.id), false);
            if (stats.special_heavy_edges_enabled)
               preexpand_edges(special_heavy_edges_for_point(source.id), true);
         };

         preexpand_state(seed);
      }
      candidate_queue.mark_expanded_states(preexpanded_states);
      stats.special_preexpand_time_ms = elapsed_ms(preexpand_time_start);

      for (const SpecialSearchCandidate &candidate : other_entry_candidates)
      {
         candidate_queue.insert(candidate.id, candidate.distance,
                                candidate.activation_level);
      }

      std::vector<uint8_t> entry_blocks(_special_blocks.size() + 1, 0);
      std::vector<uint8_t> retained_entry_blocks(_special_blocks.size() + 1, 0);
      for (size_t entry_index = 0; entry_index < entry_point_ids.size(); ++entry_index)
      {
         const IdxType block_id = point_owner(
             entry_point_ids[entry_index], entry_activation_level[entry_index]);
         if (block_id > 0 && block_id < entry_blocks.size())
            entry_blocks[block_id] = 1;
      }
      for (const SpecialSearchCandidate &candidate : block_seed_candidates)
      {
         const IdxType block_id = point_owner(candidate.id,
                                              candidate.activation_level);
         if (block_id > 0 && block_id < retained_entry_blocks.size())
            retained_entry_blocks[block_id] = 1;
      }
      stats.special_entry_blocks = static_cast<size_t>(
          std::count(entry_blocks.begin(), entry_blocks.end(), uint8_t{1}));
      stats.special_retained_entry_blocks = static_cast<size_t>(
          std::count(retained_entry_blocks.begin(), retained_entry_blocks.end(), uint8_t{1}));
      stats.special_entry_time_ms = elapsed_ms(entry_time_start);
      stats.entry_point_setup_time_ms = stats.special_entry_time_ms;

      const size_t early_stop_min_nodes = static_cast<size_t>(std::max<IdxType>(runtime.K, runtime.Lsearch));
      size_t router_nodes_since_recheck = 0;
      while (true)
      {
         if (block_router_enabled &&
             router_nodes_since_recheck >=
                 block_router_recheck_interval)
         {
            advance_pending_block_router_batches();
            router_nodes_since_recheck = 0;
         }
         if (!candidate_queue.has_unexpanded())
         {
            if (block_router_enabled &&
                advance_pending_block_router_batches())
            {
               router_nodes_since_recheck = 0;
               continue;
            }
            break;
         }

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
         {
            if (block_router_enabled &&
                advance_pending_block_router_batches())
            {
               router_nodes_since_recheck = 0;
               continue;
            }
            break;
         }

         SpecialSearchCandidate cur;
         if (!candidate_queue.pop_closest_unexpanded(cur))
            break;
         if (block_router_enabled)
            ++router_nodes_since_recheck;

         if (cur.free())
         {
            const IdxType block_id = point_owner(cur.id, cur.activation_level);
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
            IdxType current_free_block = point_owner(cur.id, cur.activation_level);
            route_child_blocks(current_free_block, false);
            size_t block_insertions_before = queue_insertions_total;
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
                            edge.special_block_id < active_free_block.size() &&
                            active_free_block[edge.special_block_id] != 0;
                        const IdxType target_block =
                            edge.kind == SpecialEdgeKind::InterBlock
                                ? special_edge_target_owner(
                                      edge, _special_blocks,
                                      _point_to_special_block_by_level)
                                : 0;
                        if (block_router_enabled &&
                            edge.kind == SpecialEdgeKind::InterBlock &&
                            (target_block == 0 ||
                             target_block >= active_free_block.size() ||
                             active_free_block[target_block] == 0))
                        {
                           ++stats.special_block_router_inter_edges_skipped;
                           remember_deferred_router_portal(
                               target_block, edge.target_point_id);
                           continue;
                        }
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
                               edge, _special_blocks,
                               _point_to_special_block_by_level);
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
                           if (visit_neighbor(neighbor, next_level))
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

                  // Construction appends every block-local edge before its
                  // inter-block edges. Keep the ranges separate so the intra
                  // cap does not affect active-child inter-block navigation.
                  size_t intra_count = edges.size();
                  if (block_router_enabled ||
                      free_intra_edge_scan_cap !=
                          std::numeric_limits<size_t>::max())
                  {
                     intra_count = 0;
                     while (intra_count < edges.size() &&
                            edges[intra_count].kind ==
                                SpecialEdgeKind::IntraBlock)
                        ++intra_count;
                  }
                  const size_t intra_kept =
                      free_intra_edge_scan_cap ==
                              std::numeric_limits<size_t>::max()
                          ? intra_count
                          : std::min(intra_count,
                                     free_intra_edge_scan_cap);
                  if (detail_stats)
                     stats.special_free_edges_cap_skipped +=
                         intra_count - intra_kept;
                  process_edge_range(0, intra_kept);
                  process_edge_range(intra_count, edges.size());
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
            const bool accepted = visit_neighbor(neighbor, next_level);
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
            process_regular_neighbor(first);
            process_regular_neighbor(second);
         }
         for (; i < neighbors.size; ++i)
         {
            prefetch_point(neighbors.ids[i]);
            const IdxType neighbor = neighbors.ids[i];
            process_regular_neighbor(neighbor);
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
         for (const SpecialSearchCandidate &candidate : final_candidates)
         {
            cur_result.insert(candidate.id, candidate.distance);
            if (cur_result.size() >= static_cast<int32_t>(runtime.K))
               break;
         }
      }
      else
      {
         std::vector<SpecialSearchCandidate> exact_candidates;
         exact_candidates.reserve(final_candidates.size());
         for (const SpecialSearchCandidate &candidate : final_candidates)
            exact_candidates.push_back(
                SpecialSearchCandidate{candidate.id, exact_distance(candidate.id),
                                       candidate.activation_level});
         std::sort(exact_candidates.begin(), exact_candidates.end(), special_candidate_less);
         for (const SpecialSearchCandidate &candidate : exact_candidates)
         {
            cur_result.insert(candidate.id, candidate.distance);
            if (cur_result.size() >= static_cast<int32_t>(runtime.K))
               break;
         }
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

      roaring::Roaring target_bitmap = collect_target_points(entry_group_ids);
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
         if (!_group_to_special_block_by_level.empty() &&
             group_id < _group_to_special_block_by_level.front().size())
            add_block(_group_to_special_block_by_level.front()[group_id], 1);
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
         if (_point_to_special_block_by_level.empty() ||
             id >= _point_to_special_block_by_level.front().size())
            return target_map[id] != 0;
         const IdxType block_id = _point_to_special_block_by_level.front()[id];
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

         const IdxType block_id =
             !_point_to_special_block_by_level.empty() &&
                     cur.id < _point_to_special_block_by_level.front().size()
                 ? _point_to_special_block_by_level.front()[cur.id]
                 : 0;
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
