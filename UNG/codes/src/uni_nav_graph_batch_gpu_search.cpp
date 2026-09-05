#include "include/uni_nav_graph.h"
#include "include/ung_gpu_l2_batch.h"
#include "include/ung_special_block_activation.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <vector>

namespace ANNS
{
namespace
{
size_t env_size_t(const char *name, size_t fallback)
{
   if (const char *value = std::getenv(name))
   {
      char *end = nullptr;
      const unsigned long long parsed = std::strtoull(value, &end, 10);
      if (end != value && *end == 0)
         return static_cast<size_t>(parsed);
   }
   return fallback;
}

double elapsed_ms_since(const std::chrono::high_resolution_clock::time_point &start)
{
   return std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - start)
       .count();
}

struct BatchQueryState
{
   IdxType global_query_id = 0;
   std::chrono::high_resolution_clock::time_point total_start;
   std::chrono::high_resolution_clock::time_point core_start;
   size_t candidate_begin = 0;
   size_t candidate_end = 0;
};
} // namespace

bool UniNavGraph::execute_special_batch_gpu_candidate_search(std::shared_ptr<IStorage> &query_storage,
                                                             std::shared_ptr<DistanceHandler> &distance_handler,
                                                             const SearchRuntimeConfig &runtime,
                                                             std::pair<IdxType, float> *results,
                                                             std::vector<float> &num_cmps,
                                                             std::vector<QueryStats> &query_stats,
                                                             const std::vector<IdxType> &true_query_group_ids)
{
   if (_base_storage == nullptr || query_storage == nullptr || distance_handler == nullptr)
      return false;
   if (_base_storage->get_data_type() != DataType::FLOAT || query_storage->get_data_type() != DataType::FLOAT)
      return false;
   if (!has_special_edges() || _special_blocks.empty())
      return false;

   _query_storage = query_storage;
   _distance_handler = distance_handler;
   _scenario = runtime.scenario;

   const IdxType num_queries = query_storage->get_num_points();
   const IdxType dim = _base_storage->get_dim();
   if (query_storage->get_dim() != dim || num_queries == 0 || runtime.K == 0)
      return false;

   query_stats.resize(num_queries);
   const size_t batch_size = std::max<size_t>(1, env_size_t("UNG_SPECIAL_BATCH_GPU_SIZE", 128));
   const size_t candidate_factor = std::max<size_t>(1, env_size_t("UNG_SPECIAL_BATCH_GPU_CAND_FACTOR", 4));
   const size_t default_candidate_limit = std::max<size_t>(static_cast<size_t>(runtime.K),
                                                           static_cast<size_t>(runtime.Lsearch) * candidate_factor);
   const size_t candidate_limit = std::max<size_t>(static_cast<size_t>(runtime.K),
                                                   env_size_t("UNG_SPECIAL_BATCH_GPU_CAND_LIMIT", default_candidate_limit));
   const size_t reduced_entry_group_threshold = env_size_t("UNG_SPECIAL_BATCH_GPU_REDUCED_ENTRY_GROUP_THRESHOLD", 1000);
   const IdxType reduced_entry_points = static_cast<IdxType>(env_size_t("UNG_SPECIAL_BATCH_GPU_REDUCED_ENTRY_POINTS", 4));

   const float *base_vectors = reinterpret_cast<const float *>(_base_storage->get_vector(0));
   std::vector<float> batch_queries;
   std::vector<IdxType> flat_candidate_ids;
   std::vector<IdxType> flat_candidate_query_ids;
   std::vector<float> flat_distances;
   std::vector<BatchQueryState> batch_states;

   std::cerr << "[special_batch_gpu] enabled batch_size=" << batch_size
             << " candidate_limit=" << candidate_limit
             << " cand_factor=" << candidate_factor << std::endl;

   for (IdxType batch_begin = 0; batch_begin < num_queries; batch_begin += static_cast<IdxType>(batch_size))
   {
      const IdxType batch_end = std::min<IdxType>(num_queries, batch_begin + static_cast<IdxType>(batch_size));
      const size_t local_count = static_cast<size_t>(batch_end - batch_begin);
      batch_queries.resize(local_count * static_cast<size_t>(dim));
      flat_candidate_ids.clear();
      flat_candidate_query_ids.clear();
      batch_states.clear();
      batch_states.reserve(local_count);

      for (IdxType qid = batch_begin; qid < batch_end; ++qid)
      {
         const size_t local_q = static_cast<size_t>(qid - batch_begin);
         std::copy_n(reinterpret_cast<const float *>(query_storage->get_vector(qid)),
                     static_cast<size_t>(dim),
                     batch_queries.data() + local_q * static_cast<size_t>(dim));

         auto &stats = query_stats[qid];
         stats = QueryStats{};
         BatchQueryState state;
         state.global_query_id = qid;
         state.total_start = std::chrono::high_resolution_clock::now();

         const auto &query_labels = query_storage->get_label_set(qid);
         std::vector<IdxType> entry_group_ids;
         QueryRouteDecision decision = decide_query_route(query_labels, runtime.idea2_available,
                                                          runtime.use_new_trie_method, runtime.recursive_more_start,
                                                          runtime.force_use_alg, runtime.bfs_filter,
                                                          entry_group_ids, stats);
         if (decision.uses_acorn())
            return false;

         const EntryGroupProviderRequest entry_request{
             runtime.entry_group_provider,
             &query_labels,
             static_cast<IdxType>(qid),
             &decision,
             runtime.recursive_more_start,
             runtime.ung_more_entry,
             &true_query_group_ids,
             &entry_group_ids,
             runtime.scalar_els_cap,
             true};
         prepare_entry_groups_for_execution(entry_request, entry_group_ids, stats);
         populate_special_query_stats(query_labels, stats);

         state.core_start = std::chrono::high_resolution_clock::now();
         state.candidate_begin = flat_candidate_ids.size();

         std::vector<uint8_t> query_covers_block(_special_blocks.size() + 1, 0);
         if (!query_covers_block.empty())
         {
            std::vector<LabelType> sorted_query = query_labels;
            std::sort(sorted_query.begin(), sorted_query.end());
            const bool root_label_coverage =
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

         std::unordered_set<IdxType> seen;
         seen.reserve(std::min<size_t>(candidate_limit * 2, 1u << 20));
         auto add_candidate = [&](IdxType point_id) {
            if (point_id >= _num_points || flat_candidate_ids.size() - state.candidate_begin >= candidate_limit)
               return false;
            if (!seen.insert(point_id).second)
               return false;
            flat_candidate_ids.push_back(point_id);
            flat_candidate_query_ids.push_back(static_cast<IdxType>(local_q));
            return true;
         };

         const IdxType effective_num_entry_points =
             entry_group_ids.size() > reduced_entry_group_threshold
                 ? std::min<IdxType>(runtime.num_entry_points, reduced_entry_points)
                 : runtime.num_entry_points;

         for (IdxType group_id : entry_group_ids)
         {
            if (group_id >= _group_id_to_range.size())
               continue;
            const auto &range = _group_id_to_range[group_id];
            if (range.second <= range.first)
               continue;
            const bool covered_root = special_block_member_is_free(runtime.scenario,
                                                                   _group_id_to_special_block,
                                                                   group_id,
                                                                   query_covers_block);
            const IdxType take = std::min<IdxType>(effective_num_entry_points, range.second - range.first);
            for (IdxType local = 0; local < take; ++local)
            {
               const IdxType point_id =
                   (local == 0 && group_id < _group_entry_points.size() &&
                    _group_entry_points[group_id] >= range.first && _group_entry_points[group_id] < range.second)
                       ? _group_entry_points[group_id]
                       : range.first + local;
               add_candidate(point_id);
               if (!covered_root)
                  continue;
               for (const SpecialEdge edge : special_edges_for_point(point_id))
               {
                  add_candidate(edge.target_point_id);
                  if (flat_candidate_ids.size() - state.candidate_begin >= candidate_limit)
                     break;
               }
            }
            if (flat_candidate_ids.size() - state.candidate_begin >= candidate_limit)
               break;
         }

         state.candidate_end = flat_candidate_ids.size();
         stats.num_distance_calcs = state.candidate_end - state.candidate_begin;
         stats.num_nodes_visited = stats.num_distance_calcs;
         num_cmps[qid] = static_cast<float>(stats.num_distance_calcs);
         batch_states.push_back(state);
      }

      flat_distances.resize(flat_candidate_ids.size());
      bool gpu_ok = flat_candidate_ids.empty();
      auto batch_gpu_start = std::chrono::high_resolution_clock::now();
      if (!flat_candidate_ids.empty())
      {
         gpu_ok = gpu_l2_batch_compute_query_candidates_float(base_vectors,
                                                              _num_points,
                                                              dim,
                                                              batch_queries.data(),
                                                              local_count,
                                                              flat_candidate_ids.data(),
                                                              flat_candidate_query_ids.data(),
                                                              flat_candidate_ids.size(),
                                                              flat_distances.data());
      }
      const double batch_gpu_ms = elapsed_ms_since(batch_gpu_start);
      if (!gpu_ok)
      {
         for (size_t i = 0; i < flat_candidate_ids.size(); ++i)
         {
            const IdxType local_q = flat_candidate_query_ids[i];
            const char *query = reinterpret_cast<const char *>(batch_queries.data() + static_cast<size_t>(local_q) * dim);
            flat_distances[i] = distance_handler->compute(query,
                                                          _base_storage->get_vector(flat_candidate_ids[i]),
                                                          dim);
         }
      }

      for (const BatchQueryState &state : batch_states)
      {
         const IdxType qid = state.global_query_id;
         auto &stats = query_stats[qid];
         const size_t begin = state.candidate_begin;
         const size_t end = state.candidate_end;
         const size_t count = end - begin;
         std::vector<std::pair<IdxType, float>> candidates;
         candidates.reserve(count);
         for (size_t i = begin; i < end; ++i)
            candidates.emplace_back(flat_candidate_ids[i], flat_distances[i]);

         if (candidates.size() > static_cast<size_t>(runtime.K))
         {
            auto kth = candidates.begin() + static_cast<std::ptrdiff_t>(runtime.K);
            std::nth_element(candidates.begin(), kth, candidates.end(),
                             [](const auto &a, const auto &b) {
                                if (a.second != b.second)
                                   return a.second < b.second;
                                return a.first < b.first;
                             });
            candidates.resize(static_cast<size_t>(runtime.K));
         }
         std::sort(candidates.begin(), candidates.end(),
                   [](const auto &a, const auto &b) {
                      if (a.second != b.second)
                         return a.second < b.second;
                      return a.first < b.first;
                   });

         for (IdxType k = 0; k < runtime.K; ++k)
         {
            if (k < candidates.size())
            {
               const IdxType new_id = candidates[static_cast<size_t>(k)].first;
               results[static_cast<size_t>(qid) * runtime.K + k].first = _new_to_old_vec_ids[new_id];
               results[static_cast<size_t>(qid) * runtime.K + k].second = distance_handler->compute(
                   query_storage->get_vector(qid), _base_storage->get_vector(new_id), dim);
            }
            else
            {
               results[static_cast<size_t>(qid) * runtime.K + k].first = static_cast<IdxType>(-1);
               results[static_cast<size_t>(qid) * runtime.K + k].second = std::numeric_limits<float>::max();
            }
         }

         (void)batch_gpu_ms;
         stats.core_search_time_ms = elapsed_ms_since(state.core_start);
         stats.time_ms = elapsed_ms_since(state.total_start);
      }
   }

   std::cerr << "[special_batch_gpu] completed candidate-generation search" << std::endl;
   return true;
}

} // namespace ANNS
