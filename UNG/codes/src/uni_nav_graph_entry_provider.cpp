#include "include/uni_nav_graph.h"

#include "include/ung_gpu_cover_frontier_provider.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <stdexcept>
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
             return compute_cpu_entry_groups_for_execution(provider_request, provider_stats);
          },
          [this](const EntryGroupProviderRequest &provider_request, QueryStats &provider_stats) {
             return compute_gpu_entry_groups_for_execution(provider_request, provider_stats);
          });
      return provider.run(request, stats);
   }

   EntryGroupProviderResult UniNavGraph::compute_gpu_entry_groups_for_execution(
       const EntryGroupProviderRequest &request,
       QueryStats &stats)
   {
      std::lock_guard<std::mutex> lock(_gpu_cover_frontier_provider_mutex);
      try
      {
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
         return _gpu_cover_frontier_provider->run(request, stats);
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

} // namespace ANNS
