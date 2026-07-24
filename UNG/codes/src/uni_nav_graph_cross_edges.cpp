#include <omp.h>

#include "include/uni_nav_graph.h"
#include "include/ung_build_settings.h"
#include "include/ung_cross_edge_config.h"
#include "include/ung_cross_edge_output_writer.h"
#include "include/ung_cross_edge_result.h"
#include "include/ung_prof_log.h"
#include "vamana/vamana.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace ANNS
{
   CrossEdgeBackend UniNavGraph::resolve_cross_edge_backend() const
   {
      return (_build_config.cross_edge_impl == UngCrossEdgeImpl::CpuVamana ||
              _build_config.cross_edge_impl == UngCrossEdgeImpl::CpuExactScan ||
              _build_config.cross_edge_impl == UngCrossEdgeImpl::CpuHybridScanVamana)
                 ? CrossEdgeBackend::CPU
                 : CrossEdgeBackend::GPU;
   }

   void UniNavGraph::append_cross_edges_from_target_range(IdxType source_point_id,
                                                          IdxType target_first,
                                                          IdxType target_second,
                                                          std::shared_ptr<Vamana> target_index,
                                                          SearchCacheList *search_cache_list,
                                                          bool use_exact_scan,
                                                          SearchQueue &out_neighbors) const
   {
      if (target_second <= target_first)
         return;

      const char *query = _base_storage->get_vector(source_point_id);
      if (use_exact_scan || target_index == nullptr || search_cache_list == nullptr)
      {
         SearchQueue local_topk;
         local_topk.reserve(_num_cross_edges);
         const IdxType dim = _base_storage->get_dim();
         for (IdxType target_id = target_first; target_id < target_second; ++target_id)
         {
            const float dist = _distance_handler->compute(query, _base_storage->get_vector(target_id), dim);
            local_topk.insert(target_id, dist);
         }
         for (int k = 0; k < local_topk.size(); ++k)
            out_neighbors.insert(local_topk[k].id, local_topk[k].distance);
         return;
      }

      auto search_cache = search_cache_list->get_free_cache();
      target_index->iterate_to_fixed_point(query, search_cache);
      for (int k = 0; k < search_cache->search_queue.size(); ++k)
         out_neighbors.insert(search_cache->search_queue[k].id + target_first,
                              search_cache->search_queue[k].distance);
      search_cache_list->release_cache(search_cache);
   }

   void UniNavGraph::append_cross_edges_from_target_points(IdxType source_point_id,
                                                           const std::vector<IdxType> &target_point_ids,
                                                           std::shared_ptr<Vamana> target_index,
                                                           SearchCacheList *search_cache_list,
                                                           bool use_exact_scan,
                                                           SearchQueue &out_neighbors,
                                                           IdxType max_neighbors) const
   {
      if (target_point_ids.empty())
         return;

      const char *query = _base_storage->get_vector(source_point_id);
      if (use_exact_scan || target_index == nullptr || search_cache_list == nullptr)
      {
         SearchQueue local_topk;
         local_topk.reserve(max_neighbors);
         const IdxType dim = _base_storage->get_dim();
         for (IdxType target_id : target_point_ids)
         {
            const float dist = _distance_handler->compute(query, _base_storage->get_vector(target_id), dim);
            local_topk.insert(target_id, dist);
         }
         for (int k = 0; k < local_topk.size(); ++k)
            out_neighbors.insert(local_topk[k].id, local_topk[k].distance);
         return;
      }

      auto search_cache = search_cache_list->get_free_cache();
      target_index->iterate_to_fixed_point(query, search_cache);
      for (int k = 0; k < search_cache->search_queue.size(); ++k)
      {
         const IdxType local_id = search_cache->search_queue[k].id;
         if (local_id < target_point_ids.size())
            out_neighbors.insert(target_point_ids[local_id], search_cache->search_queue[k].distance);
      }
      search_cache_list->release_cache(search_cache);
   }

   void UniNavGraph::build_cross_edges_generate_cpu_baseline(std::vector<SearchQueue> &cross_group_neighbors,
                                                             SearchCacheList &search_cache_list)
   {
      size_t skipped_pairs = 0;
      size_t skipped_query_vectors = 0;
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         if (_label_nav_graph->in_neighbors[group_id].empty())
            continue;

         auto index = _vamana_instances[group_id];
         if (_num_cross_edges > _Lbuild)
         {
            std::cerr << "Error: num_cross_edges should be less than or equal to Lbuild" << std::endl;
            exit(-1);
         }

         for (auto in_group_id : _label_nav_graph->in_neighbors[group_id])
         {
            if (_build_config.special_blocks_enabled && _build_config.special_block_skip_trivial &&
                is_trivial_special_block_root_group(in_group_id))
            {
               const auto &range = _group_id_to_range[in_group_id];
               ++skipped_pairs;
               skipped_query_vectors += static_cast<size_t>(range.second - range.first);
               continue;
            }
            const auto &range = _group_id_to_range[in_group_id];

#pragma omp parallel for schedule(dynamic, 1)
            for (IdxType vec_id = range.first; vec_id < range.second; ++vec_id)
            {
               append_cross_edges_from_target_range(vec_id,
                                                    _group_id_to_range[group_id].first,
                                                    _group_id_to_range[group_id].second,
                                                    index,
                                                    &search_cache_list,
                                                    false,
                                                    cross_group_neighbors[vec_id]);
            }
         }
      }
      _special_block_summary.cross_trivial_skipped_pairs += static_cast<IdxType>(skipped_pairs);
      _special_block_summary.cross_trivial_skipped_query_vectors += static_cast<IdxType>(skipped_query_vectors);
      std::cout << "[cross_edges] cpu_vamana_skip skipped_pairs=" << skipped_pairs
                << " skipped_query_visits=" << skipped_query_vectors << std::endl;
      prof_logf("[PROF] cross_edges.cpu_vamana_skip skipped_pairs=%zu skipped_query_visits=%zu",
                skipped_pairs, skipped_query_vectors);
   }

   void UniNavGraph::build_cross_edges_generate_cpu_exact_scan(std::vector<SearchQueue> &cross_group_neighbors)
   {
      const IdxType dim = _base_storage->get_dim();
      size_t target_groups = 0;
      size_t pair_count = 0;
      size_t total_queries = 0;
      size_t skipped_pairs = 0;
      size_t skipped_query_vectors = 0;
      unsigned long long total_pair_ops = 0;

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         if (_label_nav_graph->in_neighbors[group_id].empty())
            continue;

         ++target_groups;
         const auto &target_range = _group_id_to_range[group_id];
         const IdxType nx = target_range.second - target_range.first;
         if (nx == 0)
            continue;

         for (auto in_group_id : _label_nav_graph->in_neighbors[group_id])
         {
            if (_build_config.special_blocks_enabled && _build_config.special_block_skip_trivial &&
                is_trivial_special_block_root_group(in_group_id))
            {
               const auto &query_range = _group_id_to_range[in_group_id];
               ++skipped_pairs;
               skipped_query_vectors += static_cast<size_t>(query_range.second - query_range.first);
               continue;
            }
            const auto &query_range = _group_id_to_range[in_group_id];
            const IdxType nq = query_range.second - query_range.first;
            if (nq == 0)
               continue;
            ++pair_count;
            total_queries += static_cast<size_t>(nq);
            total_pair_ops += static_cast<unsigned long long>(nq) *
                              static_cast<unsigned long long>(nx) *
                              static_cast<unsigned long long>(dim);

#pragma omp parallel for schedule(dynamic, 1)
            for (IdxType vec_id = query_range.first; vec_id < query_range.second; ++vec_id)
            {
               append_cross_edges_from_target_range(vec_id,
                                                    target_range.first,
                                                    target_range.second,
                                                    nullptr,
                                                    nullptr,
                                                    true,
                                                    cross_group_neighbors[vec_id]);
            }
         }
      }

      std::cout << "[cross_edges] cpu_exact_scan target_groups=" << target_groups
                << " pairs=" << pair_count
                << " skipped_pairs=" << skipped_pairs
                << " skipped_query_visits=" << skipped_query_vectors
                << " query_visits=" << total_queries
                << " dim_ops=" << total_pair_ops
                << std::endl;
      _special_block_summary.cross_trivial_skipped_pairs += static_cast<IdxType>(skipped_pairs);
      _special_block_summary.cross_trivial_skipped_query_vectors += static_cast<IdxType>(skipped_query_vectors);
      prof_logf("[PROF] cross_edges.cpu_exact_scan target_groups=%zu pairs=%zu skipped_pairs=%zu skipped_query_visits=%zu query_visits=%zu dim_ops=%llu",
                target_groups, pair_count, skipped_pairs, skipped_query_vectors, total_queries, total_pair_ops);
   }

   void UniNavGraph::build_cross_edges_generate_cpu_hybrid_scan_vamana(std::vector<SearchQueue> &cross_group_neighbors,
                                                                       SearchCacheList &search_cache_list)
   {
      const IdxType dim = _base_storage->get_dim();
      const CpuHybridCrossSettings hybrid_cfg = make_cpu_hybrid_cross_settings();
      const long long work_threshold = hybrid_cfg.work_threshold;
      std::cout << "[cross_edges] cpu_hybrid_config " << hybrid_cfg.summary() << std::endl;
      prof_logf("[PROF] cross_edges.cpu_hybrid_config %s", hybrid_cfg.summary().c_str());
      size_t scan_pairs = 0;
      size_t vamana_pairs = 0;
      size_t scan_queries = 0;
      size_t vamana_queries = 0;
      size_t skipped_pairs = 0;
      size_t skipped_query_vectors = 0;
      unsigned long long scan_dim_ops = 0;

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         if (_label_nav_graph->in_neighbors[group_id].empty())
            continue;

         const auto &target_range = _group_id_to_range[group_id];
         const IdxType nx = target_range.second - target_range.first;
         if (nx == 0)
            continue;

         auto index = _vamana_instances[group_id];
         if (_num_cross_edges > _Lbuild)
         {
            std::cerr << "Error: num_cross_edges should be less than or equal to Lbuild" << std::endl;
            exit(-1);
         }

         for (auto in_group_id : _label_nav_graph->in_neighbors[group_id])
         {
            if (_build_config.special_blocks_enabled && _build_config.special_block_skip_trivial &&
                is_trivial_special_block_root_group(in_group_id))
            {
               const auto &query_range = _group_id_to_range[in_group_id];
               ++skipped_pairs;
               skipped_query_vectors += static_cast<size_t>(query_range.second - query_range.first);
               continue;
            }
            const auto &query_range = _group_id_to_range[in_group_id];
            const IdxType nq = query_range.second - query_range.first;
            if (nq == 0)
               continue;

            const unsigned long long pair_work = static_cast<unsigned long long>(nq) *
                                                 static_cast<unsigned long long>(nx);
            const bool use_exact_scan =
                nx <= hybrid_cfg.target_exact_max_nx ||
                pair_work <= static_cast<unsigned long long>(work_threshold);
            if (!use_exact_scan)
            {
               ++vamana_pairs;
               vamana_queries += static_cast<size_t>(nq);
#pragma omp parallel for schedule(dynamic, 1)
               for (IdxType vec_id = query_range.first; vec_id < query_range.second; ++vec_id)
               {
                  append_cross_edges_from_target_range(vec_id,
                                                       target_range.first,
                                                       target_range.second,
                                                       index,
                                                       &search_cache_list,
                                                       false,
                                                       cross_group_neighbors[vec_id]);
               }
            }
            else
            {
               ++scan_pairs;
               scan_queries += static_cast<size_t>(nq);
               scan_dim_ops += pair_work * static_cast<unsigned long long>(dim);
#pragma omp parallel for schedule(dynamic, 1)
               for (IdxType vec_id = query_range.first; vec_id < query_range.second; ++vec_id)
               {
                  append_cross_edges_from_target_range(vec_id,
                                                       target_range.first,
                                                       target_range.second,
                                                       index,
                                                       &search_cache_list,
                                                       true,
                                                       cross_group_neighbors[vec_id]);
               }
            }
         }
      }

      std::cout << "[cross_edges] cpu_hybrid_scan_vamana threshold_nqnx=" << work_threshold
                << " scan_pairs=" << scan_pairs
                << " vamana_pairs=" << vamana_pairs
                << " skipped_pairs=" << skipped_pairs
                << " skipped_query_visits=" << skipped_query_vectors
                << " scan_queries=" << scan_queries
                << " vamana_queries=" << vamana_queries
                << " scan_dim_ops=" << scan_dim_ops
                << std::endl;
      _special_block_summary.cross_trivial_skipped_pairs += static_cast<IdxType>(skipped_pairs);
      _special_block_summary.cross_trivial_skipped_query_vectors += static_cast<IdxType>(skipped_query_vectors);
      prof_logf("[PROF] cross_edges.cpu_hybrid_scan_vamana threshold_nqnx=%lld scan_pairs=%zu vamana_pairs=%zu skipped_pairs=%zu skipped_query_visits=%zu scan_queries=%zu vamana_queries=%zu scan_dim_ops=%llu",
                work_threshold, scan_pairs, vamana_pairs, skipped_pairs, skipped_query_vectors,
                scan_queries, vamana_queries, scan_dim_ops);
   }

   // GPU cross-edge routing adapter and legacy env bridge.
   //
   // This function intentionally stays above individual CUDA backends: it maps
   // the typed CrossEdgeGpuRuntimeConfig plus legacy UngGpuTopkImpl defaults to
   // source-exact, X-streaming, cuVS, or regular all-batched execution. New
   // kernels should live behind gpu_cross_edge_*.cuh helpers instead of adding
   // env parsing or algorithm-specific policy here.
   bool UniNavGraph::build_cross_edges_generate_gpu_optimized(std::vector<SearchQueue> &cross_group_neighbors,
                                                              const CrossEdgeGpuRuntimeConfig &gpu_route,
                                                              std::vector<std::vector<IdxType>> *cross_group_neighbor_ids,
                                                              std::vector<IdxType> *cross_group_neighbor_flat_ids,
                                                              CrossEdgeBuildTiming &timing)
   {
      apply_cross_edge_gpu_topk_env_defaults(_build_config.gpu_topk_impl);
      std::cout << "[cross_edges] gpu_topk_impl=" << to_string(_build_config.gpu_topk_impl) << std::endl;
      std::cout << "[cross_edges] backend=GPU route=" << gpu_route.route_name()
                << " class=" << gpu_route.route_class_name()
                << " writeback=" << gpu_route.writeback_mode_name() << std::endl;

      if (gpu_route.source_exact)
      {
         return build_cross_edges_generate_gpu_source_exact(cross_group_neighbors,
                                                            cross_group_neighbor_ids,
                                                            cross_group_neighbor_flat_ids,
                                                            timing,
                                                            gpu_route);
      }

      std::vector<IdxType> groups_to_process;
      groups_to_process.reserve(_num_groups);
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         if (!_label_nav_graph->in_neighbors[group_id].empty())
            groups_to_process.push_back(group_id);

      std::cout << "[cross_edges] backend=GPU, target_groups=" << groups_to_process.size() << std::endl;
      if (groups_to_process.empty())
         return true;

      bool gpu_prepared = false;
      try
      {
         if (gpu_route.x_streaming)
         {
            std::cout << "[cross_edges] x_streaming begin" << std::endl;
            gpu_cross_groups_search_x_streaming(groups_to_process, _base_storage->get_dim(), _num_cross_edges,
                                                cross_group_neighbors,
                                                &timing, gpu_route);
            std::cout << "[cross_edges] x_streaming end" << std::endl;
            return true;
         }

         const auto prepare_start = std::chrono::high_resolution_clock::now();
         std::cout << "[cross_edges] prepare_all begin" << std::endl;
         gpu_prepare_all_vectors_for_cross_edge(timing, gpu_route);
         std::cout << "[cross_edges] prepare_all end wall_ms="
                   << std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - prepare_start)
                          .count()
                   << std::endl;
         gpu_prepared = true;
         std::cout << "[cross_edges] batched_search begin" << std::endl;
         gpu_cross_groups_search_all_batched(groups_to_process, _base_storage->get_dim(), _num_cross_edges,
                                             cross_group_neighbors, cross_group_neighbor_ids,
                                             cross_group_neighbor_flat_ids,
                                             &timing, gpu_route);
         std::cout << "[cross_edges] batched_search end" << std::endl;
      }
      catch (const std::exception &e)
      {
         std::cerr << "[cross_edges][GPU] failed: " << e.what() << std::endl;
         prof_logf("[PROF] cross_edges.gpu_exception what=%s", e.what());
         if (gpu_prepared)
            gpu_release_all_vectors_on_device();
         if (_build_config.special_blocks_enabled && _build_config.special_block_skip_trivial)
            return false;
         if (!gpu_prepared && gpu_route.auto_x_streaming && !cross_group_neighbor_ids && !cross_group_neighbor_flat_ids)
         {
            try
            {
               std::cout << "[cross_edges] retry with X-streaming GPU path" << std::endl;
               gpu_release_all_vectors_on_device();
               gpu_cross_groups_search_x_streaming(groups_to_process, _base_storage->get_dim(), _num_cross_edges,
                                                   cross_group_neighbors,
                                                   &timing, gpu_route);
               return true;
            }
            catch (const std::exception &se)
            {
               std::cerr << "[cross_edges][GPU][x_stream] failed: " << se.what() << std::endl;
               prof_logf("[PROF] cross_edges.x_stream_exception what=%s", se.what());
            }
         }
         return false;
      }
      catch (...)
      {
         std::cerr << "[cross_edges][GPU] failed: unknown exception" << std::endl;
         prof_logf("[PROF] cross_edges.gpu_exception what=unknown");
         if (gpu_prepared)
            gpu_release_all_vectors_on_device();
         return false;
      }

      if (gpu_prepared)
      {
         const GpuCrossLifecycleSettings lifecycle_cfg = make_gpu_cross_lifecycle_settings();
         prof_logf("[PROF] cross_edges.gpu_lifecycle_config %s", lifecycle_cfg.summary().c_str());
         if (lifecycle_cfg.release_after_cross)
            gpu_release_all_vectors_on_device();
         else
            prof_logf("[PROF] cross_edges.gpu_release_after_cross skipped=1");
      }
      return true;
   }

   void UniNavGraph::build_cross_edges_generate_additional(
       std::vector<std::vector<std::pair<IdxType, IdxType>>> *materialized,
       const CrossEdgeHostOutputView &cross_edge_outputs,
       SearchCacheList *search_cache_list,
       CrossEdgeBuildTiming &timing)
   {
      ScopedTimerMs t("cross_edges.additional_edges_ms", &timing.additional_ms);
      if (_build_config.additional_edges_impl == UngAdditionalEdgesImpl::Skip)
      {
         std::cout << "[cross_edges] additional_edges=skip" << std::endl;
         return;
      }
      if (_build_config.additional_edges_impl == UngAdditionalEdgesImpl::CpuVamana && search_cache_list == nullptr)
         throw std::runtime_error("CPU Vamana additional_edges requires a SearchCacheList.");

      auto append_additional_edge = [&](IdxType group_id, IdxType from_id, IdxType to_id) {
         if (materialized)
            (*materialized)[group_id].emplace_back(from_id, to_id);
         else
         {
            std::lock_guard<std::mutex> guard(_graph->neighbor_locks[from_id]);
            _graph->neighbors[from_id].emplace_back(to_id);
         }
      };

      unsigned long long additional_missing_group_edges = 0;
      unsigned long long additional_query_vectors = 0;
      unsigned long long additional_exact_dim_ops = 0;
      unsigned long long additional_appended_edges = 0;
      unsigned long long additional_skipped_groups = 0;
      unsigned long long additional_skipped_points = 0;

#pragma omp parallel reduction(+ : additional_missing_group_edges, additional_query_vectors, additional_exact_dim_ops, additional_appended_edges, additional_skipped_groups, additional_skipped_points)
      {
         std::vector<uint32_t> connected_epoch(static_cast<size_t>(_num_groups) + 1, 0);
         uint32_t epoch = 1;

#pragma omp for schedule(dynamic, 1)
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            if (_build_config.special_blocks_enabled && _build_config.special_block_skip_trivial &&
                is_trivial_special_block_root_group(group_id))
            {
               const auto &range = _group_id_to_range[group_id];
               additional_skipped_groups += 1;
               additional_skipped_points += static_cast<unsigned long long>(range.second - range.first);
               continue;
            }
            if (epoch == 0)
            {
               std::fill(connected_epoch.begin(), connected_epoch.end(), 0);
               epoch = 1;
            }
            const uint32_t cur_epoch = epoch++;
            const auto &cur_range = _group_id_to_range[group_id];

            for (IdxType i = cur_range.first; i < cur_range.second; ++i)
            {
               if (cross_edge_outputs.uses_flat_ids())
               {
                  const auto &flat_ids = *cross_edge_outputs.flat_ids;
                  const size_t base = static_cast<size_t>(i) * static_cast<size_t>(cross_edge_outputs.topk);
                  for (IdxType k = 0; k < _num_cross_edges; ++k)
                  {
                     const IdxType neighbor_id = flat_ids[base + static_cast<size_t>(k)];
                     if (neighbor_id == std::numeric_limits<IdxType>::max())
                        continue;
                     const IdxType neighbor_group = _new_vec_id_to_group_id[neighbor_id];
                     if (neighbor_group <= _num_groups)
                        connected_epoch[static_cast<size_t>(neighbor_group)] = cur_epoch;
                  }
               }
               else if (cross_edge_outputs.uses_id_vectors())
               {
                  for (IdxType neighbor_id : (*cross_edge_outputs.id_vectors)[i])
                  {
                     const IdxType neighbor_group = _new_vec_id_to_group_id[neighbor_id];
                     if (neighbor_group <= _num_groups)
                        connected_epoch[static_cast<size_t>(neighbor_group)] = cur_epoch;
                  }
               }
               else
               {
                  const auto &cross_group_neighbors = *cross_edge_outputs.search_queues;
                  for (IdxType j = 0; j < cross_group_neighbors[i].size(); ++j)
                  {
                     const IdxType neighbor_group = _new_vec_id_to_group_id[cross_group_neighbors[i][j].id];
                     if (neighbor_group <= _num_groups)
                        connected_epoch[static_cast<size_t>(neighbor_group)] = cur_epoch;
                  }
               }
            }

            for (IdxType out_group_id : _label_nav_graph->out_neighbors[group_id])
               if (out_group_id > _num_groups || connected_epoch[static_cast<size_t>(out_group_id)] != cur_epoch)
               {
                  additional_missing_group_edges += 1;
                  IdxType cnt = 0;
                  for (auto vec_id = cur_range.first; vec_id < cur_range.second && cnt < _num_cross_edges; ++vec_id)
                  {
                     additional_query_vectors += 1;
                     if (_build_config.additional_edges_impl == UngAdditionalEdgesImpl::CpuExactScan)
                     {
                        SearchQueue local_topk;
                        local_topk.reserve(_num_cross_edges);
                        const auto &out_range = _group_id_to_range[out_group_id];
                        const char *query = _base_storage->get_vector(vec_id);
                        const IdxType dim = _base_storage->get_dim();
                        additional_exact_dim_ops += static_cast<unsigned long long>(out_range.second - out_range.first) *
                                                    static_cast<unsigned long long>(dim);
                        for (IdxType target_id = out_range.first; target_id < out_range.second; ++target_id)
                        {
                           const float dist = _distance_handler->compute(query, _base_storage->get_vector(target_id), dim);
                           local_topk.insert(target_id, dist);
                        }
                        for (auto k = 0; k < local_topk.size() && k < _num_cross_edges / 2; ++k)
                        {
                           append_additional_edge(group_id, vec_id, local_topk[k].id);
                           cnt += 1;
                           additional_appended_edges += 1;
                        }
                     }
                     else
                     {
                        auto search_cache = search_cache_list->get_free_cache();
                        _vamana_instances[out_group_id]->iterate_to_fixed_point(_base_storage->get_vector(vec_id), search_cache);

                        for (auto k = 0; k < search_cache->search_queue.size() && k < _num_cross_edges / 2; ++k)
                        {
                           append_additional_edge(group_id, vec_id,
                                                  search_cache->search_queue[k].id + _group_id_to_range[out_group_id].first);
                           cnt += 1;
                           additional_appended_edges += 1;
                        }
                        search_cache_list->release_cache(search_cache);
                     }
                  }
               }
         }
      }
      std::cout << "[cross_edges] additional_work missing_group_edges=" << additional_missing_group_edges
                << " query_vectors=" << additional_query_vectors
                << " exact_dim_ops=" << additional_exact_dim_ops
                << " appended_edges=" << additional_appended_edges
                << " trivial_special_skipped_groups=" << additional_skipped_groups
                << " trivial_special_skipped_points=" << additional_skipped_points
                << " schedule=dynamic1"
                << std::endl;
      _special_block_summary.additional_trivial_skipped_groups += static_cast<IdxType>(additional_skipped_groups);
      _special_block_summary.additional_trivial_skipped_points += static_cast<IdxType>(additional_skipped_points);
      prof_logf("[PROF] cross_edges.additional_work missing_group_edges=%llu query_vectors=%llu exact_dim_ops=%llu appended_edges=%llu trivial_special_skipped_groups=%llu trivial_special_skipped_points=%llu schedule=dynamic1",
                additional_missing_group_edges, additional_query_vectors,
                additional_exact_dim_ops, additional_appended_edges,
                additional_skipped_groups, additional_skipped_points);
   }

   void UniNavGraph::merge_cross_edges_to_graph(const CrossEdgeHostOutputView &cross_edge_outputs,
                                                CrossEdgeBuildTiming &timing)
   {
      CrossEdgeGraphMaterializationWriter writer(*_graph, _num_cross_edges);
      writer.append_cross_edges(cross_edge_outputs, _num_points, timing);
   }

   void UniNavGraph::merge_additional_edges_to_graph(
       const std::vector<std::vector<std::pair<IdxType, IdxType>>> &additional_edges,
       CrossEdgeBuildTiming &timing)
   {
      CrossEdgeGraphMaterializationWriter writer(*_graph, _num_cross_edges);
      writer.append_materialized_additional_edges(additional_edges, _num_groups, timing);
   }

   void UniNavGraph::build_cross_group_edges()
   {
      if (_build_config.cross_edge_impl == UngCrossEdgeImpl::OriginalCpu)
      {
         build_cross_group_edges_original_cpu();
         return;
      }

      std::cout << "Building cross-group edges ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // UNG_CROSS_EDGE_BACKEND: 0=CPU baseline, 1=GPU optimized.
      const CrossEdgeBackend backend = resolve_cross_edge_backend();
      if (_build_config.special_blocks_enabled && _build_config.special_block_skip_trivial &&
          backend == CrossEdgeBackend::GPU)
      {
         size_t skipped_pairs = 0;
         size_t skipped_query_vectors = 0;
         for (IdxType target_group_id = 1; target_group_id <= _num_groups; ++target_group_id)
         {
            if (_label_nav_graph->in_neighbors[target_group_id].empty())
               continue;
            for (IdxType source_group_id : _label_nav_graph->in_neighbors[target_group_id])
            {
               if (!is_trivial_special_block_root_group(source_group_id))
                  continue;
               const auto &query_range = _group_id_to_range[source_group_id];
               skipped_pairs += 1;
               skipped_query_vectors += static_cast<size_t>(query_range.second - query_range.first);
            }
         }
         _special_block_summary.cross_trivial_skipped_pairs = static_cast<IdxType>(skipped_pairs);
         _special_block_summary.cross_trivial_skipped_query_vectors = static_cast<IdxType>(skipped_query_vectors);
         std::cout << "[cross_edges] special_blocks=1 uses GPU cross-edge path with trivial-source skip mask."
                   << " skipped_pairs=" << skipped_pairs
                   << " skipped_query_visits=" << skipped_query_vectors
                   << std::endl;
      }
      if (_intra_group_graph_ids_are_global)
      {
         const bool cross_uses_group_vamana =
             _build_config.cross_edge_impl == UngCrossEdgeImpl::CpuVamana ||
             _build_config.cross_edge_impl == UngCrossEdgeImpl::OriginalCpu ||
             _build_config.cross_edge_impl == UngCrossEdgeImpl::CpuHybridScanVamana;
         const bool additional_uses_group_vamana =
             _build_config.additional_edges_impl == UngAdditionalEdgesImpl::CpuVamana;
         if (cross_uses_group_vamana || additional_uses_group_vamana)
            throw std::runtime_error(
                "UNG_INTRA_GLOBAL_IDS is incompatible with CPU Vamana cross/additional edges; "
                "use GPU/CpuExact/Skip paths or disable UNG_INTRA_GLOBAL_IDS.");
      }
      // UNG_CROSS_EDGE_GPU_STRICT: 1=GPU失败直接报错; 0=失败自动回退CPU.
      const bool gpu_strict = _build_config.gpu_strict;
      const CrossEdgeGpuRuntimeConfig gpu_route =
          make_cross_edge_gpu_runtime_config(_build_config, backend == CrossEdgeBackend::GPU);
      const std::string gpu_route_error = gpu_route.validation_error();
      if (backend == CrossEdgeBackend::GPU && !gpu_route_error.empty())
         throw std::runtime_error("Invalid GPU cross-edge route: " + gpu_route_error);
      if (backend == CrossEdgeBackend::GPU &&
          _build_config.special_blocks_enabled && _build_config.special_block_skip_trivial)
      {
         const bool unsupported_skip_route =
             _build_config.cross_edge_impl == UngCrossEdgeImpl::CuvsBruteForce ||
             gpu_route.source_exact ||
             gpu_route.x_streaming ||
             gpu_route.cpu_tiny_groups ||
             gpu_route.universal_route;
         if (unsupported_skip_route)
         {
            const std::string msg =
                "GPU cross-edge trivial-source skip currently supports only the regular batched/double-buffer route; "
                "disable source_exact/x_streaming/cpu_tiny/universal/cuVS for special skip builds.";
            std::cerr << "[cross_edges][gpu_source_skip][unsupported_route] " << msg << std::endl;
            throw std::runtime_error(msg);
         }
      }
      const std::string gpu_route_summary = gpu_route.summary();
      if (backend == CrossEdgeBackend::GPU)
         std::cout << "[cross_edges] gpu_route " << gpu_route_summary << std::endl;
      const bool gpu_cuvs_searchqueue_only =
          backend == CrossEdgeBackend::GPU &&
          _build_config.cross_edge_impl == UngCrossEdgeImpl::CuvsBruteForce;
      CrossEdgeBuildResult cross_result;
      cross_result.route_summary = gpu_route_summary;
      cross_result.gpu_backend_used = (backend == CrossEdgeBackend::GPU);
      CrossEdgeBuildTiming &timing = cross_result.timing;
      CrossEdgeGpuWritebackMode &active_writeback_mode = cross_result.active_writeback_mode;
      // allocate memory for storaging cross-group neighbors
      std::vector<SearchQueue> cross_group_neighbors;
      auto ensure_cross_queue_storage = [&](bool reserve_per_point = true) {
         if (cross_group_neighbors.size() == static_cast<size_t>(_num_points))
            return;
         const auto storage_start = std::chrono::high_resolution_clock::now();
         cross_group_neighbors.clear();
         cross_group_neighbors.resize(_num_points);
         if (reserve_per_point)
         {
#pragma omp parallel for schedule(static, 4096)
            for (auto point_id = 0; point_id < _num_points; ++point_id)
               cross_group_neighbors[point_id].reserve(_num_cross_edges);
         }
         timing.output_storage_init_ms += std::chrono::duration<double, std::milli>(
                                             std::chrono::high_resolution_clock::now() - storage_start)
                                             .count();
      };
      if (gpu_route.needs_searchqueue_storage() || gpu_cuvs_searchqueue_only)
         ensure_cross_queue_storage(backend != CrossEdgeBackend::GPU);
      std::vector<std::vector<IdxType>> cross_group_neighbor_ids;
      std::vector<IdxType> cross_group_neighbor_flat_ids;
      if (gpu_route.uses_id_vector_writeback())
      {
         const auto storage_start = std::chrono::high_resolution_clock::now();
         cross_group_neighbor_ids.resize(_num_points);
#pragma omp parallel for schedule(static, 4096)
         for (auto point_id = 0; point_id < _num_points; ++point_id)
            cross_group_neighbor_ids[point_id].reserve(_num_cross_edges);
         timing.output_storage_init_ms += std::chrono::duration<double, std::milli>(
                                             std::chrono::high_resolution_clock::now() - storage_start)
                                             .count();
      }

      // allocate memory for search caches
      std::unique_ptr<SearchCacheList> search_cache_list;
      auto ensure_search_cache_list = [&]() -> SearchCacheList & {
         if (!search_cache_list)
         {
            const auto cache_start = std::chrono::high_resolution_clock::now();
            size_t max_group_size = 0;
            for (auto group_id = 1; group_id <= _num_groups; ++group_id)
               max_group_size = std::max(max_group_size, _group_id_to_vec_ids[group_id].size());
            search_cache_list.reset(new SearchCacheList(_num_threads, max_group_size, _Lbuild));
            timing.search_cache_init_ms += std::chrono::duration<double, std::milli>(
                                               std::chrono::high_resolution_clock::now() - cache_start)
                                               .count();
         }
         return *search_cache_list;
      };
      omp_set_num_threads(_num_threads);

   {
      ScopedTimerMs t("cross_edges.generate_ms", &timing.generate_ms);
      if (_index_name != "Vamana")
      {
         std::cerr << "Error: invalid index name " << _index_name << std::endl;
         exit(-1);
      }

      // 可切换主线：
      // - CPU baseline: build_cross_edges_generate_cpu_baseline
      // - GPU optimized: build_cross_edges_generate_gpu_optimized（失败可回退）
      if (backend == CrossEdgeBackend::GPU)
      {
         const bool gpu_ok =
             (_build_config.cross_edge_impl == UngCrossEdgeImpl::CuvsBruteForce)
                 ? build_cross_edges_generate_cuvs_bruteforce(cross_group_neighbors,
                                                              timing,
                                                              gpu_route)
                 : build_cross_edges_generate_gpu_optimized(cross_group_neighbors,
                                                            gpu_route,
                                                            gpu_route.uses_id_vector_writeback() ? &cross_group_neighbor_ids : nullptr,
                                                            gpu_route.uses_flat_id_writeback() ? &cross_group_neighbor_flat_ids : nullptr,
                                                            timing);
         if (!gpu_ok)
         {
            cross_result.gpu_fallback_used = true;
            if (gpu_strict)
               throw std::runtime_error("GPU cross-edge generation failed in strict mode.");
            if (_intra_group_graph_ids_are_global)
               std::cout << "[cross_edges] fallback to CPU exact scan path (global intra-group ids are incompatible with CPU Vamana fallback)." << std::endl;
            else
               std::cout << "[cross_edges] fallback to CPU baseline path." << std::endl;
            ensure_cross_queue_storage(true);
            if (_intra_group_graph_ids_are_global)
               build_cross_edges_generate_cpu_exact_scan(cross_group_neighbors);
            else
               build_cross_edges_generate_cpu_baseline(cross_group_neighbors, ensure_search_cache_list());
         }
         else
         {
            active_writeback_mode =
                gpu_cuvs_searchqueue_only
                    ? CrossEdgeGpuWritebackMode::SearchQueue
                    : gpu_route.writeback_mode();
         }
      }
      else
      {
         std::cout << "[cross_edges] backend=CPU impl=" << to_string(_build_config.cross_edge_impl) << std::endl;
         ensure_cross_queue_storage(true);
         if (_build_config.cross_edge_impl == UngCrossEdgeImpl::CpuExactScan)
            build_cross_edges_generate_cpu_exact_scan(cross_group_neighbors);
         else if (_build_config.cross_edge_impl == UngCrossEdgeImpl::CpuHybridScanVamana)
            build_cross_edges_generate_cpu_hybrid_scan_vamana(cross_group_neighbors, ensure_search_cache_list());
         else
            build_cross_edges_generate_cpu_baseline(cross_group_neighbors, ensure_search_cache_list());
      }

   }

      const AdditionalEdgesSettings additional_cfg = make_additional_edges_settings();
      const bool additional_direct_append_requested = additional_cfg.direct_append_requested;
      const bool additional_direct_append =
          additional_cfg.direct_append_enabled(_build_config.additional_edges_impl);
      std::cout << "[cross_edges] additional_config "
                << additional_cfg.summary(_build_config.additional_edges_impl) << std::endl;
      prof_logf("[PROF] cross_edges.additional_config %s",
                additional_cfg.summary(_build_config.additional_edges_impl).c_str());
      cross_result.additional_direct_append = additional_direct_append;
      if (additional_direct_append_requested && !additional_direct_append)
      {
         std::cout << "[cross_edges] additional_direct_append disabled: supported only for cpu_exact additional_edges"
                   << std::endl;
      }
      std::vector<std::vector<std::pair<IdxType, IdxType>>> additional_edges;
      if (!additional_direct_append)
      {
         const auto storage_start = std::chrono::high_resolution_clock::now();
         additional_edges.resize(_num_groups + 1);
         timing.additional_storage_init_ms += std::chrono::duration<double, std::milli>(
                                                std::chrono::high_resolution_clock::now() - storage_start)
                                                .count();
      }

      if (_build_config.additional_edges_impl == UngAdditionalEdgesImpl::CpuVamana)
         (void)ensure_search_cache_list();
      CrossEdgeHostOutputView cross_edge_outputs;
      cross_edge_outputs.mode = active_writeback_mode;
      cross_edge_outputs.search_queues = &cross_group_neighbors;
      cross_edge_outputs.id_vectors = &cross_group_neighbor_ids;
      cross_edge_outputs.flat_ids = &cross_group_neighbor_flat_ids;
      cross_edge_outputs.topk = _num_cross_edges;
      if (!additional_direct_append)
         build_cross_edges_generate_additional(&additional_edges,
                                               cross_edge_outputs,
                                               search_cache_list.get(),
                                               timing);
      // Add offsets after cross-edge generation so group-local intra edges and
      // global cross/additional edges share the same final id space.
   {
      ScopedTimerMs t("cross_edges.add_offset_ms", &timing.add_offset_ms);
      add_offset_for_uni_nav_graph();
   }

      merge_cross_edges_to_graph(cross_edge_outputs, timing);

      if (additional_direct_append)
      {
         build_cross_edges_generate_additional(nullptr,
                                               cross_edge_outputs,
                                               search_cache_list.get(),
                                               timing);
      }
      else
      {
         merge_additional_edges_to_graph(additional_edges, timing);
      }
      std::cout << "[cross_edges] additional_direct_append=" << (additional_direct_append ? 1 : 0) << std::endl;
      cross_result.flat_id_items = cross_group_neighbor_flat_ids.size();

         std::cout << "[GPU GEMM] H2D(ms)=" << timing.gpu_h2d_ms
          << "  Kernel(ms)=" << timing.gpu_kernel_ms
          << "  D2H(ms)=" << timing.gpu_d2h_ms
          << std::endl;
      std::cout << "[cross_edges] breakdown generate(ms)=" << timing.generate_ms
                << " additional(ms)=" << timing.additional_ms
                << " add_offset(ms)=" << timing.add_offset_ms
                << " merge_cross(ms)=" << timing.merge_cross_ms
                << " merge_add(ms)=" << timing.merge_additional_ms
                << " output_storage(ms)=" << timing.output_storage_init_ms
                << " additional_storage(ms)=" << timing.additional_storage_init_ms
                << " search_cache_init(ms)=" << timing.search_cache_init_ms
                << " route={" << cross_result.route_summary << "}"
                << " active_writeback=" << cross_result.active_writeback_mode_name()
                << " flat_items=" << cross_result.flat_id_items
                << std::endl;

      _build_cross_edges_time = std::chrono::duration<double, std::milli>(
                                    std::chrono::high_resolution_clock::now() - start_time)
                                    .count();
      std::cout << "\r- Finish in " << _build_cross_edges_time << " ms" << std::endl;

      prof_logf("[PROF] cross_edges.summary total_ms=%.3f gen_ms=%.3f add_ms=%.3f add_offset_ms=%.3f merge_cross_ms=%.3f merge_add_ms=%.3f "
      "output_storage_ms=%.3f additional_storage_ms=%.3f search_cache_init_ms=%.3f "
      "num_points=%u num_groups=%u num_cross_edges=%u Lbuild=%u threads=%u route={%s}",
      _build_cross_edges_time, timing.generate_ms, timing.additional_ms, timing.add_offset_ms,
      timing.merge_cross_ms, timing.merge_additional_ms,
      timing.output_storage_init_ms, timing.additional_storage_init_ms, timing.search_cache_init_ms,
      (unsigned)_num_points, (unsigned)_num_groups, (unsigned)_num_cross_edges, (unsigned)_Lbuild, (unsigned)_num_threads,
      cross_result.route_summary.c_str());
      prof_logf("[PROF] cross_edges.id_vector_writeback requested=%d active=%d",
                gpu_route.id_vector_writeback ? 1 : 0,
                active_writeback_mode == CrossEdgeGpuWritebackMode::IdVector ? 1 : 0);
      prof_logf("[PROF] cross_edges.flat_id_writeback requested=%d active=%d items=%zu",
                gpu_route.flat_id_writeback ? 1 : 0,
                active_writeback_mode == CrossEdgeGpuWritebackMode::FlatId ? 1 : 0,
                cross_result.flat_id_items);
      // GPU细分总览（H2D/Kernel/D2H 累计）
      prof_logf("[PROF] cross_edges.gpu_breakdown_sum h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f",
         timing.gpu_h2d_ms, timing.gpu_kernel_ms, timing.gpu_d2h_ms);

   }

   void UniNavGraph::build_cross_group_edges_original_cpu()
   {
      std::cout << "Building cross-group edges ..." << std::endl;
      std::cout << "- cross-edge impl: " << to_string(_build_config.cross_edge_impl) << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      std::vector<SearchQueue> cross_group_neighbors;
      cross_group_neighbors.resize(_num_points);
      for (auto point_id = 0; point_id < _num_points; ++point_id)
         cross_group_neighbors[point_id].reserve(_num_cross_edges);

      size_t max_group_size = 0;
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
         max_group_size = std::max(max_group_size, _group_id_to_vec_ids[group_id].size());
      SearchCacheList search_cache_list(_num_threads, max_group_size, _Lbuild);
      omp_set_num_threads(_num_threads);

      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         if (_label_nav_graph->in_neighbors[group_id].size() > 0)
         {
            if (group_id % 100 == 0)
               std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
            IdxType offset = _group_id_to_range[group_id].first;

            if (_index_name == "Vamana")
            {
               auto index = _vamana_instances[group_id];
               if (_num_cross_edges > _Lbuild)
               {
                  std::cerr << "Error: num_cross_edges should be less than or equal to Lbuild" << std::endl;
                  exit(-1);
               }

               for (auto in_group_id : _label_nav_graph->in_neighbors[group_id])
               {
                  if (_build_config.special_blocks_enabled && _build_config.special_block_skip_trivial &&
                      is_trivial_special_block_root_group(in_group_id))
                     continue;
                  const auto &range = _group_id_to_range[in_group_id];

#pragma omp parallel for schedule(dynamic, 1)
                  for (auto vec_id = range.first; vec_id < range.second; ++vec_id)
                  {
                     const char *query = _base_storage->get_vector(vec_id);
                     auto search_cache = search_cache_list.get_free_cache();
                     index->iterate_to_fixed_point(query, search_cache);

                     for (auto k = 0; k < search_cache->search_queue.size(); ++k)
                        cross_group_neighbors[vec_id].insert(search_cache->search_queue[k].id + offset,
                                                             search_cache->search_queue[k].distance);
                     search_cache_list.release_cache(search_cache);
                  }
               }
            }
            else
            {
               std::cerr << "Error: invalid index name " << _index_name << std::endl;
               exit(-1);
            }
         }
      }

      std::vector<std::vector<std::pair<IdxType, IdxType>>> additional_edges(_num_groups + 1);
#pragma omp parallel
      {
         std::vector<uint32_t> connected_epoch(static_cast<size_t>(_num_groups) + 1, 0);
         uint32_t epoch = 1;

#pragma omp for schedule(dynamic, 256)
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            if (_build_config.special_blocks_enabled && _build_config.special_block_skip_trivial &&
                is_trivial_special_block_root_group(group_id))
               continue;
            if (epoch == 0)
            {
               std::fill(connected_epoch.begin(), connected_epoch.end(), 0);
               epoch = 1;
            }
            const uint32_t cur_epoch = epoch++;
            const auto &cur_range = _group_id_to_range[group_id];

            for (IdxType i = cur_range.first; i < cur_range.second; ++i)
               for (IdxType j = 0; j < cross_group_neighbors[i].size(); ++j)
               {
                  const IdxType neighbor_group = _new_vec_id_to_group_id[cross_group_neighbors[i][j].id];
                  if (neighbor_group <= _num_groups)
                     connected_epoch[static_cast<size_t>(neighbor_group)] = cur_epoch;
               }

            for (IdxType out_group_id : _label_nav_graph->out_neighbors[group_id])
               if (out_group_id > _num_groups || connected_epoch[static_cast<size_t>(out_group_id)] != cur_epoch)
               {
                  IdxType cnt = 0;
                  for (auto vec_id = cur_range.first; vec_id < cur_range.second && cnt < _num_cross_edges; ++vec_id)
                  {
                     auto search_cache = search_cache_list.get_free_cache();
                     _vamana_instances[out_group_id]->iterate_to_fixed_point(_base_storage->get_vector(vec_id), search_cache);

                     for (auto k = 0; k < search_cache->search_queue.size() && k < _num_cross_edges / 2; ++k)
                     {
                        additional_edges[group_id].emplace_back(vec_id,
                                                                search_cache->search_queue[k].id + _group_id_to_range[out_group_id].first);
                        cnt += 1;
                     }
                     search_cache_list.release_cache(search_cache);
                  }
               }
         }
      }

      add_offset_for_uni_nav_graph();

#pragma omp parallel for schedule(dynamic, 4096)
      for (auto point_id = 0; point_id < _num_points; ++point_id)
         for (auto k = 0; k < cross_group_neighbors[point_id].size(); ++k)
            _graph->neighbors[point_id].emplace_back(cross_group_neighbors[point_id][k].id);

#pragma omp parallel for schedule(dynamic, 256)
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         for (const auto &[from_id, to_id] : additional_edges[group_id])
            _graph->neighbors[from_id].emplace_back(to_id);
      }

      _build_cross_edges_time = std::chrono::duration<double, std::milli>(
                                    std::chrono::high_resolution_clock::now() - start_time)
                                    .count();
      std::cout << "\r- Finish in " << _build_cross_edges_time << " ms" << std::endl;
   }

} // namespace ANNS
