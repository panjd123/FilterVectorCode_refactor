#include "include/uni_nav_graph.h"

#include "include/ung_gpu_cover_frontier_provider.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ANNS
{
   void UniNavGraph::prepare_entry_groups_for_execution(const EntryGroupProviderRequest &request,
                                                        std::vector<IdxType> &entry_group_ids,
                                                        QueryStats &stats)
   {
      EntryGroupProviderResult result = run_entry_group_provider(request, stats);
      entry_group_ids = std::move(result.group_ids);
      stats.num_entry_points = entry_group_ids.size();
      populate_entry_group_route_stats(entry_group_ids, stats);
   }

   EntryGroupProviderResult UniNavGraph::run_entry_group_provider(const EntryGroupProviderRequest &request,
                                                                  QueryStats &stats)
   {
      const SearchEntryProvider provider(
          [this](const EntryGroupProviderRequest &provider_request, QueryStats &provider_stats) {
             if (provider_request.impl == EntryGroupProviderImpl::CpuBruteForceEls)
                return compute_cpu_bruteforce_entry_groups_for_execution(provider_request, provider_stats);
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

      const IdxType num_groups_including_zero = _num_groups + 1;
      const IdxType words = (num_groups_including_zero + 63) / 64;
      if (words == 0)
         return result;

      std::unordered_map<LabelType, std::vector<uint64_t>> label_group_bits;
      label_group_bits.reserve(1024);
      IdxType max_group_label_size = 0;
      std::vector<std::vector<uint64_t>> size_group_bits(1);
      size_group_bits[0].assign(words, 0);
      auto set_bit = [](std::vector<uint64_t> &bits, IdxType gid) {
         bits[gid >> 6] |= uint64_t{1} << (gid & 63);
      };
      for (IdxType gid = 1; gid <= _num_groups; ++gid)
      {
         const auto &labels = _group_id_to_label_set[gid];
         max_group_label_size = std::max<IdxType>(max_group_label_size, static_cast<IdxType>(labels.size()));
         if (size_group_bits.size() <= labels.size())
            size_group_bits.resize(labels.size() + 1, std::vector<uint64_t>(words, 0));
         set_bit(size_group_bits[labels.size()], gid);
         for (LabelType label : labels)
         {
            auto &bits = label_group_bits[label];
            if (bits.empty())
               bits.assign(words, 0);
            set_bit(bits, gid);
         }
      }

      std::vector<uint64_t> candidate_bits(words, ~uint64_t{0});
      if ((num_groups_including_zero & 63) != 0)
         candidate_bits.back() &= (uint64_t{1} << (num_groups_including_zero & 63)) - 1;
      candidate_bits[0] &= ~uint64_t{1};
      for (LabelType label : query_labels)
      {
         auto it = label_group_bits.find(label);
         if (it == label_group_bits.end())
         {
            std::fill(candidate_bits.begin(), candidate_bits.end(), 0);
            break;
         }
         for (IdxType word = 0; word < words; ++word)
            candidate_bits[word] &= it->second[word];
      }

      const IdxType q_len = static_cast<IdxType>(query_labels.size());
      const IdxType end_len = std::min<IdxType>(max_group_label_size, q_len + kFrontierDelta);
      std::vector<uint64_t> frontier_bits(words, 0);
      for (IdxType len = q_len; len <= end_len && len < size_group_bits.size(); ++len)
         for (IdxType word = 0; word < words; ++word)
            frontier_bits[word] |= candidate_bits[word] & size_group_bits[len][word];

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

      result.elapsed_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - start_time)
                              .count();
      stats.get_min_super_sets_time_ms = result.elapsed_ms;
      return result;
   }

} // namespace ANNS
