#include "include/uni_nav_graph.h"

#include "include/ung_gpu_cover_frontier_provider.h"
#include "include/ung_cpu_bruteforce_els.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace
{
   // Route statistics are optional metadata (selector features/diagnostics).
   // The graph search only needs the materialized entry_group_ids, so allow
   // latency-focused runs to skip the expensive descendants/coverage bitmap
   // unions entirely.
   bool disable_entry_route_stats()
   {
      const char *value = std::getenv("UNG_DISABLE_ENTRY_ROUTE_STATS");
      if (value == nullptr)
         return false;
      return value[0] != '\0' && !(value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
               value[0] == 'n' || value[0] == 'N' || value[0] == 'o' ||
               value[0] == 'O');
   }

   bool reuse_entry_route_stats_cache()
   {
      static const bool enabled = std::getenv("UNG_DISABLE_ENTRY_ROUTE_STATS_CACHE") == nullptr;
      return enabled;
   }

   bool cpu_bruteforce_els_use_roaring()
   {
      const char *value = std::getenv("UNG_CPU_BRUTEFORCE_ELS_USE_ROARING");
      if (value == nullptr)
         return true;
      return !(value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
               value[0] == 'n' || value[0] == 'N' || value[0] == 'o' || value[0] == 'O');
   }

   bool reuse_els_results()
   {
      return !ANNS::ung_env_flag_enabled("UNG_DISABLE_ELS_REUSE");
   }
}

namespace ANNS
{
   void UniNavGraph::prepare_entry_groups_for_execution(const EntryGroupProviderRequest &request,
                                                        std::vector<IdxType> &entry_group_ids,
                                                        QueryStats &stats)
   {
      EntryGroupProviderResult result = run_entry_group_provider(request, stats);
      entry_group_ids = std::move(result.group_ids);
      // Keep the cheap entry count for CSV/debug output.  Do not materialize
      // descendants/coverage cardinalities when explicitly disabled.
      stats.num_entry_points = entry_group_ids.size();
      if (disable_entry_route_stats())
         return;
      if (reuse_entry_route_stats_cache() && result.route_stats.valid)
         apply_entry_group_route_stats(result.route_stats, stats);
      else
         populate_entry_group_route_stats(entry_group_ids, stats);
   }

   EntryGroupProviderResult UniNavGraph::run_entry_group_provider(const EntryGroupProviderRequest &request,
                                                                  QueryStats &stats)
   {
      const SearchEntryProvider provider(
          [this](const EntryGroupProviderRequest &provider_request, QueryStats &provider_stats) {
             if (provider_request.impl == EntryGroupProviderImpl::CpuBruteForceEls)
                return compute_cpu_bruteforce_entry_groups_for_execution(provider_request, provider_stats);
             if (provider_request.impl == EntryGroupProviderImpl::CpuBruteForceElsScalar)
                return compute_cpu_bruteforce_scalar_entry_groups_for_execution(provider_request, provider_stats);
             if (provider_request.impl == EntryGroupProviderImpl::SpecialBlockTrie)
                return compute_special_block_trie_entry_groups_for_execution(provider_request, provider_stats);
             return compute_cpu_entry_groups_for_execution(provider_request, provider_stats);
          },
          [this](const EntryGroupProviderRequest &provider_request, QueryStats &provider_stats) {
             return compute_gpu_entry_groups_for_execution(provider_request, provider_stats);
          });
      return provider.run(request, stats);
   }

   void UniNavGraph::initialize_gpu_cover_frontier_provider(size_t workspace_count,
                                                            size_t max_query_labels)
   {
      std::lock_guard<std::mutex> lock(_gpu_cover_frontier_provider_mutex);
      if (!_gpu_cover_frontier_provider)
      {
         const auto *descendants =
             (_label_nav_graph != nullptr) ? &_label_nav_graph->_lng_descendants : nullptr;
         GpuCoverFrontierBuildInput input;
         input.group_labels = &_group_id_to_label_set;
         input.lng_descendants = descendants;
         input.num_groups = _num_groups;
         _gpu_cover_frontier_provider = std::make_unique<GpuCoverFrontierProvider>(input);
      }
      if (workspace_count > 0)
         _gpu_cover_frontier_provider->reserve_workspaces(workspace_count, max_query_labels);
   }

   void UniNavGraph::warmup_gpu_cover_frontier_provider(const std::vector<LabelType> &query_labels)
   {
      const GpuCoverFrontierProvider *provider = nullptr;
      {
         std::lock_guard<std::mutex> lock(_gpu_cover_frontier_provider_mutex);
         if (!_gpu_cover_frontier_provider)
         {
            const auto *descendants =
                (_label_nav_graph != nullptr) ? &_label_nav_graph->_lng_descendants : nullptr;
            GpuCoverFrontierBuildInput input;
            input.group_labels = &_group_id_to_label_set;
            input.lng_descendants = descendants;
            input.num_groups = _num_groups;
            _gpu_cover_frontier_provider = std::make_unique<GpuCoverFrontierProvider>(input);
         }
         provider = _gpu_cover_frontier_provider.get();
      }

      QueryRouteDecision decision;
      QueryStats stats;
      const EntryGroupProviderRequest request{
          EntryGroupProviderImpl::GpuCoverFrontier,
          &query_labels,
          0,
          &decision,
          false,
          false,
          nullptr,
          nullptr};
      (void)provider->run(request, stats);
   }

   EntryGroupProviderResult UniNavGraph::compute_gpu_entry_groups_for_execution(
       const EntryGroupProviderRequest &request,
       QueryStats &stats)
   {
      try
      {
         const GpuCoverFrontierProvider *provider = nullptr;
         {
            std::lock_guard<std::mutex> lock(_gpu_cover_frontier_provider_mutex);
            if (!_gpu_cover_frontier_provider)
            {
               const auto *descendants =
                   (_label_nav_graph != nullptr) ? &_label_nav_graph->_lng_descendants : nullptr;
               GpuCoverFrontierBuildInput input;
               input.group_labels = &_group_id_to_label_set;
               input.lng_descendants = descendants;
               input.num_groups = _num_groups;
               _gpu_cover_frontier_provider = std::make_unique<GpuCoverFrontierProvider>(input);
            }
            provider = _gpu_cover_frontier_provider.get();
         }
         return provider->run(request, stats);
      }
      catch (const std::exception &ex)
      {
         EntryGroupProviderResult result = compute_cpu_entry_groups_for_execution(request, stats);
         result.mark_fallback(request.impl, std::string("gpu_cover_frontier failed: ") + ex.what());
         std::cerr << "[SearchEntryProvider] gpu_cover_frontier fallback: "
                   << result.fallback_reason << std::endl;
         return result;
      }
   }

   EntryGroupProviderResult UniNavGraph::compute_cpu_entry_groups_for_execution(
       const EntryGroupProviderRequest &request,
       QueryStats &stats)
   {
      const auto &query_labels = *request.query_labels;
      const QueryRouteDecision &decision = *request.decision;

      EntryGroupProviderResult result;
      result.requested_impl = request.impl;
      if (request.has_current_group_ids())
         result.group_ids = *request.current_group_ids;

      if (!decision.entry_groups_calculated && decision.algorithm == 0)
      {
         auto get_entry_group_start_time = std::chrono::high_resolution_clock::now();
         static std::atomic<int> counter{0};
         get_min_super_sets_debug(query_labels, result.group_ids, false, true, counter,
                                  decision.use_new_trie, request.recursive_more_start, stats, false);
         result.elapsed_ms =
             std::chrono::duration<double, std::milli>(
                 std::chrono::high_resolution_clock::now() - get_entry_group_start_time)
                 .count();
         stats.get_min_super_sets_time_ms = result.elapsed_ms;
         result.provider = EntryGroupProviderKind::CpuMinSuperSets;
         result.exact_minimal = true;
      }
      else if (decision.entry_groups_calculated)
      {
         result.provider = EntryGroupProviderKind::CpuMinSuperSets;
         result.exact_minimal = true;
      }
      else
      {
         result.provider = EntryGroupProviderKind::Passthrough;
      }

      if (request.ung_more_entry && decision.algorithm == 0)
      {
         const IdxType true_group_id = request.true_group_id_or(0);
         const size_t extra_k = result.group_ids.size() / 5;
         result.group_ids = select_entry_groups(result.group_ids, SelectionMode::SizeAndDistance,
                                                extra_k, 1.0, true_group_id);
         result.provider = EntryGroupProviderKind::CpuExpanded;
         result.exact_minimal = false;
      }
      return result;
   }

   EntryGroupProviderResult UniNavGraph::compute_special_block_trie_entry_groups_for_execution(
       const EntryGroupProviderRequest &request,
       QueryStats &stats)
   {
      const auto provider_start = std::chrono::high_resolution_clock::now();
      EntryGroupProviderResult result;
      result.requested_impl = request.impl;
      if (request.decision->algorithm != 0)
      {
         if (request.has_current_group_ids())
            result.group_ids = *request.current_group_ids;
         result.provider = EntryGroupProviderKind::Passthrough;
         return result;
      }
      if (_special_block_trie_index.empty())
         throw std::runtime_error(
             "special_block_trie provider requires a loaded trie-partitioned special-block index");

      const bool include_block_frontier =
          ung_env_flag_enabled("UNG_SPECIAL_TRIE_BLOCK_FRONTIER") &&
          !ung_env_flag_enabled("UNG_SPECIAL_TRIE_TERMINAL_ONLY");
      auto compute_frontier = [&](SpecialBlockTrieSearchStats &frontier_stats) {
         std::vector<IdxType> block_ids;
         // The terminal-only path can use the precomputed CRoaring label
         // postings and terminal-ancestor cache.  Keep the topology walk for
         // block-frontier mode because it must expose portals below the first
         // terminal on each matching branch.
         std::vector<IdxType> group_ids = include_block_frontier
                                              ? _special_block_trie_index.find_entry_groups(
                                                    *request.query_labels, &frontier_stats,
                                                    &block_ids, true)
                                              : _special_block_trie_index.find_entry_groups_bitset(
                                                    *request.query_labels, &frontier_stats);
         for (IdxType block_id : block_ids)
         {
            if (block_id == 0 || block_id > _special_blocks.size())
               continue;
            const IdxType anchor_group_id = _special_blocks[block_id - 1].root_group_id;
            if (anchor_group_id != 0)
               group_ids.push_back(anchor_group_id);
         }
         std::sort(group_ids.begin(), group_ids.end());
         group_ids.erase(std::unique(group_ids.begin(), group_ids.end()), group_ids.end());
         return group_ids;
      };

      SpecialBlockTrieSearchStats trie_stats;
      if (request.cache_query_label_results && reuse_els_results())
      {
         const std::string cache_key = make_entry_group_label_cache_key(
             request.impl, *request.query_labels, include_block_frontier, false, 0);
         const CachedEntryGroupResult cached =
             _special_block_trie_query_cache.get_or_compute(cache_key, [&] {
                SpecialBlockTrieSearchStats computed_stats;
                CachedEntryGroupResult computed;
                computed.group_ids = compute_frontier(computed_stats);
                if (!disable_entry_route_stats())
                   computed.route_stats = compute_entry_group_route_stats(computed.group_ids);
                {
                   std::lock_guard<std::mutex> lock(_special_block_trie_query_cache_mutex);
                   _special_block_trie_search_stats_cache[cache_key] = computed_stats;
                }
                return computed;
             });
         result.group_ids = cached.group_ids;
         result.route_stats = cached.route_stats;
         {
            std::lock_guard<std::mutex> lock(_special_block_trie_query_cache_mutex);
            const auto stats_it = _special_block_trie_search_stats_cache.find(cache_key);
            if (stats_it != _special_block_trie_search_stats_cache.end())
               trie_stats = stats_it->second;
         }
         result.elapsed_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::high_resolution_clock::now() - provider_start)
                                 .count();
      }
      else
      {
         result.group_ids = compute_frontier(trie_stats);
         result.elapsed_ms = trie_stats.elapsed_ms;
      }
      result.provider = EntryGroupProviderKind::SpecialBlockTrie;
      result.coverage_correct = true;
      result.exact_minimal = false;

      stats.get_min_super_sets_time_ms = result.elapsed_ms;
      stats.candidate_set_size = trie_stats.pivot_postings;
      stats.successful_checks = trie_stats.matching_pivots;
      stats.trie_nodes_traversed = static_cast<long long>(
          trie_stats.upward_nodes_visited + trie_stats.downward_nodes_visited);
      stats.special_trie_pivot_postings = trie_stats.pivot_postings;
      stats.special_trie_matching_pivots = trie_stats.matching_pivots;
      stats.special_trie_upward_nodes_visited = trie_stats.upward_nodes_visited;
      stats.special_trie_downward_nodes_visited = trie_stats.downward_nodes_visited;
      stats.special_trie_branches_pruned = trie_stats.branches_pruned;
      stats.special_trie_terminal_candidates = trie_stats.terminal_candidates;
      stats.special_trie_block_frontier_candidates = trie_stats.block_frontier_candidates;
      stats.special_trie_terminal_descendants_pruned =
          trie_stats.terminal_descendants_pruned;
      stats.special_trie_final_entries = result.group_ids.size();
      stats.special_trie_final_block_entries = trie_stats.final_block_entries;
      stats.special_trie_time_ms = result.elapsed_ms;
      return result;
   }

   SpecialBlockTrieWarmupStats UniNavGraph::warmup_special_block_trie(
       const std::shared_ptr<IStorage> &query_storage)
   {
      if (_special_block_trie_index.empty())
         throw std::runtime_error(
             "special-block trie warm-up requires a trie-partitioned index");
      const auto start = std::chrono::high_resolution_clock::now();
      std::unordered_map<std::string, std::vector<LabelType>> unique_queries;
      const bool include_block_frontier =
          ung_env_flag_enabled("UNG_SPECIAL_TRIE_BLOCK_FRONTIER") &&
          !ung_env_flag_enabled("UNG_SPECIAL_TRIE_TERMINAL_ONLY");
      for (IdxType query_id = 0; query_id < query_storage->get_num_points(); ++query_id)
      {
         const auto &labels = query_storage->get_label_set(query_id);
         const std::string key = make_entry_group_label_cache_key(
             EntryGroupProviderImpl::SpecialBlockTrie, labels,
             include_block_frontier, false, 0);
         unique_queries.emplace(key, labels);
      }

      QueryRouteDecision decision;
      for (const auto &item : unique_queries)
      {
         EntryGroupProviderRequest request;
         request.impl = EntryGroupProviderImpl::SpecialBlockTrie;
         request.query_labels = &item.second;
         request.decision = &decision;
         request.cache_query_label_results = true;
         QueryStats stats;
         (void)compute_special_block_trie_entry_groups_for_execution(request, stats);
      }

      SpecialBlockTrieWarmupStats warmup;
      warmup.unique_query_keys = unique_queries.size();
      warmup.elapsed_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - start)
                              .count();
      return warmup;
   }


   void UniNavGraph::ensure_cpu_bruteforce_els_cache()
   {
      std::lock_guard<std::mutex> lock(_cpu_bruteforce_els_cache_mutex);
      if (_cpu_bruteforce_els_cache.valid)
         return;

      CpuBruteForceElsCache cache;
      cache.num_groups_including_zero = _num_groups + 1;
      cache.words_per_query = (cache.num_groups_including_zero + 63) / 64;
      if (cache.words_per_query == 0)
      {
         cache.valid = true;
         _cpu_bruteforce_els_cache = std::move(cache);
         return;
      }

      for (IdxType gid = 1; gid <= _num_groups; ++gid)
      {
         cache.max_group_label_size = std::max<IdxType>(
             cache.max_group_label_size,
             static_cast<IdxType>(_group_id_to_label_set[gid].size()));
      }

      cache.size_group_bits.assign(
          static_cast<size_t>(cache.max_group_label_size + 1) * cache.words_per_query, 0);
      for (IdxType gid = 1; gid <= _num_groups; ++gid)
      {
         const IdxType row = static_cast<IdxType>(_group_id_to_label_set[gid].size());
         cache.size_group_bits[static_cast<size_t>(row) * cache.words_per_query + (gid >> 6)] |=
             uint64_t{1} << (gid & 63);
      }

      cache.valid = true;
      _cpu_bruteforce_els_cache = std::move(cache);
   }

   std::vector<std::shared_ptr<const std::vector<uint64_t>>>
   UniNavGraph::prepare_cpu_bruteforce_els_label_rows(
       const std::vector<LabelType> &query_labels)
   {
      ensure_cpu_bruteforce_els_cache();
      std::lock_guard<std::mutex> lock(_cpu_bruteforce_els_cache_mutex);

      std::vector<LabelType> missing_labels;
      missing_labels.reserve(query_labels.size());
      for (LabelType label : query_labels)
      {
         if (_cpu_bruteforce_els_cache.label_group_bits.find(label) ==
             _cpu_bruteforce_els_cache.label_group_bits.end() &&
             std::find(missing_labels.begin(), missing_labels.end(), label) == missing_labels.end())
         {
            missing_labels.push_back(label);
         }
      }

      if (!missing_labels.empty())
      {
         CpuBruteForceElsRows built = build_cpu_bruteforce_els_rows(
             _group_id_to_label_set, _num_groups, missing_labels);
         for (auto &item : built.label_group_bits)
         {
            _cpu_bruteforce_els_cache.label_group_bits.emplace(
                item.first,
                std::make_shared<const std::vector<uint64_t>>(std::move(item.second)));
         }
      }

      std::vector<std::shared_ptr<const std::vector<uint64_t>>> rows;
      rows.reserve(query_labels.size());
      for (LabelType label : query_labels)
         rows.push_back(_cpu_bruteforce_els_cache.label_group_bits.at(label));
      return rows;
   }

   CpuElsWarmupStats UniNavGraph::warmup_cpu_bruteforce_els(
       const std::shared_ptr<IStorage> &query_storage,
       bool recursive_more_start,
       bool ung_more_entry,
       size_t scalar_els_cap)
   {
      const auto start_time = std::chrono::high_resolution_clock::now();
      std::unordered_map<std::string, std::vector<LabelType>> unique_queries;
      std::unordered_set<LabelType> unique_labels;
      for (IdxType qid = 0; qid < query_storage->get_num_points(); ++qid)
      {
         const auto &labels = query_storage->get_label_set(qid);
         const std::string key = make_entry_group_label_cache_key(
             EntryGroupProviderImpl::CpuBruteForceEls, labels,
             recursive_more_start, ung_more_entry, scalar_els_cap);
         unique_queries.emplace(key, labels);
         unique_labels.insert(labels.begin(), labels.end());
      }

      std::vector<LabelType> workload_labels(unique_labels.begin(), unique_labels.end());
      std::sort(workload_labels.begin(), workload_labels.end());
      (void)prepare_cpu_bruteforce_els_label_rows(workload_labels);

      for (const auto &item : unique_queries)
      {
         QueryRouteDecision decision;
         EntryGroupProviderRequest request;
         request.impl = EntryGroupProviderImpl::CpuBruteForceEls;
         request.query_labels = &item.second;
         request.decision = &decision;
         request.recursive_more_start = recursive_more_start;
         request.ung_more_entry = ung_more_entry;
         request.scalar_els_cap = scalar_els_cap;
         request.cache_query_label_results = true;
         QueryStats stats;
         (void)compute_cpu_bruteforce_entry_groups_for_execution(request, stats);
      }

      CpuElsWarmupStats stats;
      stats.unique_query_keys = unique_queries.size();
      stats.prepared_labels = workload_labels.size();
      stats.elapsed_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::high_resolution_clock::now() - start_time)
                             .count();
      return stats;
   }

   EntryGroupProviderResult UniNavGraph::compute_cpu_bruteforce_entry_groups_for_execution(
       const EntryGroupProviderRequest &request,
       QueryStats &stats)
   {
      constexpr IdxType kFrontierDelta = 2;
      constexpr IdxType kFrontierCoverCap = 8192;
      const auto start_time = std::chrono::high_resolution_clock::now();
      const auto &query_labels = *request.query_labels;

      EntryGroupProviderResult result;
      result.requested_impl = request.impl;
      result.provider = EntryGroupProviderKind::CpuBruteForceEls;
      result.coverage_correct = true;
      result.exact_minimal = false;

      if (request.cache_query_label_results && reuse_els_results())
      {
         const std::string query_cache_key =
             make_entry_group_label_cache_key(request.impl, query_labels,
                                              request.recursive_more_start,
                                              request.ung_more_entry,
                                              request.scalar_els_cap);
         const CachedEntryGroupResult cached =
             _cpu_bruteforce_els_query_cache.get_or_compute(query_cache_key, [&] {
                EntryGroupProviderRequest uncached_request = request;
                uncached_request.cache_query_label_results = false;
                QueryStats uncached_stats;
                EntryGroupProviderResult computed =
                    compute_cpu_bruteforce_entry_groups_for_execution(uncached_request,
                                                                      uncached_stats);
                CachedEntryGroupResult value;
                value.group_ids = std::move(computed.group_ids);
                if (reuse_entry_route_stats_cache() && !disable_entry_route_stats())
                   value.route_stats = compute_entry_group_route_stats(value.group_ids);
                return value;
             });
         result.group_ids = cached.group_ids;
         result.route_stats = cached.route_stats;
         result.elapsed_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::high_resolution_clock::now() - start_time)
                                 .count();
         stats.get_min_super_sets_time_ms = result.elapsed_ms;
         return result;
      }

      const auto label_rows = prepare_cpu_bruteforce_els_label_rows(query_labels);
      const auto &cache = _cpu_bruteforce_els_cache;
      const IdxType num_groups_including_zero = cache.num_groups_including_zero;
      const IdxType words = cache.words_per_query;
      if (words == 0)
         return result;

      auto set_bit = [](std::vector<uint64_t> &bits, IdxType gid) {
         bits[gid >> 6] |= uint64_t{1} << (gid & 63);
      };

      std::vector<uint64_t> candidate_bits(words, ~uint64_t{0});
      if ((num_groups_including_zero & 63) != 0)
         candidate_bits.back() &= (uint64_t{1} << (num_groups_including_zero & 63)) - 1;
      candidate_bits[0] &= ~uint64_t{1};
      for (const auto &row : label_rows)
      {
         for (IdxType word = 0; word < words; ++word)
            candidate_bits[word] &= (*row)[word];
      }

      const IdxType q_len = static_cast<IdxType>(query_labels.size());
      const IdxType end_len = std::min<IdxType>(cache.max_group_label_size, q_len + kFrontierDelta);
      std::vector<uint64_t> frontier_bits(words, 0);
      for (IdxType len = q_len; len <= end_len && len <= cache.max_group_label_size; ++len)
      {
         const size_t row_offset = static_cast<size_t>(len) * words;
         for (IdxType word = 0; word < words; ++word)
            frontier_bits[word] |= candidate_bits[word] & cache.size_group_bits[row_offset + word];
      }

      std::vector<uint64_t> selected_frontier_bits(words, 0);
      std::vector<IdxType> frontier_ids;
      frontier_ids.reserve(std::min<IdxType>(kFrontierCoverCap, _num_groups));
      for (IdxType word = 0; word < words && frontier_ids.size() < kFrontierCoverCap; ++word)
      {
         uint64_t bits = frontier_bits[word];
         while (bits && frontier_ids.size() < kFrontierCoverCap)
         {
            const IdxType bit = static_cast<IdxType>(__builtin_ctzll(bits));
            const IdxType gid = (word << 6) + bit;
            if (gid > 0 && gid < num_groups_including_zero)
            {
               frontier_ids.push_back(gid);
               selected_frontier_bits[word] |= uint64_t{1} << bit;
            }
            bits &= bits - 1;
         }
      }

      if (cpu_bruteforce_els_use_roaring() && _lng_descendants_rb.size() > _num_groups)
      {
         result.group_ids = cpu_bruteforce_els_select_with_roaring(candidate_bits,
                                                                   selected_frontier_bits,
                                                                   frontier_ids,
                                                                   _lng_descendants_rb,
                                                                   num_groups_including_zero);
      }
      else
      {
         std::vector<uint64_t> covered_bits(words, 0);
         if (_label_nav_graph != nullptr && _label_nav_graph->_lng_descendants.size() > _num_groups)
         {
            for (IdxType gid : frontier_ids)
            {
               for (IdxType desc : _label_nav_graph->_lng_descendants[gid])
               {
                  if (desc > 0 && desc <= _num_groups)
                     set_bit(covered_bits, desc);
               }
            }
         }
         else
         {
            for (IdxType parent : frontier_ids)
            {
               const auto &parent_labels = _group_id_to_label_set[parent];
               for (IdxType child = 1; child <= _num_groups; ++child)
               {
                  if (child == parent)
                     continue;
                  const auto &child_labels = _group_id_to_label_set[child];
                  if (child_labels.size() > parent_labels.size() &&
                      std::includes(child_labels.begin(), child_labels.end(),
                                    parent_labels.begin(), parent_labels.end()))
                     set_bit(covered_bits, child);
               }
            }
         }

         for (IdxType word = 0; word < words; ++word)
         {
            uint64_t out = selected_frontier_bits[word] | (candidate_bits[word] & ~covered_bits[word]);
            while (out)
            {
               const IdxType bit = static_cast<IdxType>(__builtin_ctzll(out));
               const IdxType gid = (word << 6) + bit;
               if (gid > 0 && gid < num_groups_including_zero)
                  result.group_ids.push_back(gid);
               out &= out - 1;
            }
         }
      }

      result.elapsed_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - start_time)
                              .count();
      stats.get_min_super_sets_time_ms = result.elapsed_ms;
      return result;
   }

   EntryGroupProviderResult UniNavGraph::compute_cpu_bruteforce_scalar_entry_groups_for_execution(
       const EntryGroupProviderRequest &request,
       QueryStats &stats)
   {
      const auto start_time = std::chrono::high_resolution_clock::now();
      const auto &query_labels = *request.query_labels;

      EntryGroupProviderResult result;
      result.requested_impl = request.impl;
      result.provider = EntryGroupProviderKind::CpuBruteForceElsScalar;
      result.coverage_correct = true;
      result.exact_minimal = (request.scalar_els_cap == 0);
      result.group_ids = cpu_bruteforce_els_select_scalar_exact(_group_id_to_label_set,
                                                                query_labels,
                                                                _num_groups,
                                                                request.scalar_els_cap);

      result.elapsed_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - start_time)
                              .count();
      stats.get_min_super_sets_time_ms = result.elapsed_ms;
      return result;
   }

} // namespace ANNS
