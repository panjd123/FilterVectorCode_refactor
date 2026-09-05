#include "include/navix_hnsw.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <queue>
#include <unordered_set>

#include <omp.h>

namespace ANNS
{
namespace
{
struct CloserFirst
{
   bool operator()(const Candidate &a, const Candidate &b) const
   {
      if (a.distance != b.distance)
         return a.distance > b.distance;
      return a.id > b.id;
   }
};

bool closer_candidate(const Candidate &a, const Candidate &b)
{
   if (a.distance != b.distance)
      return a.distance < b.distance;
   return a.id < b.id;
}
} // namespace

void NavixHnsw::build(std::shared_ptr<IStorage> base_storage,
                      std::shared_ptr<DistanceHandler> distance_handler,
                      std::shared_ptr<Graph> graph,
                      IdxType max_degree,
                      IdxType ef_construction,
                      IdxType max_candidates,
                      uint32_t num_threads)
{
   _base_storage = std::move(base_storage);
   _distance_handler = std::move(distance_handler);
   _graph = std::move(graph);
   const IdxType num_points = _base_storage->get_num_points();
   if (num_points == 0)
      return;

   if (_verbose)
   {
      std::cout << "Building NaviX HNSW lower-layer graph ..." << std::endl;
      std::cout << "- max_degree: " << max_degree << std::endl;
      std::cout << "- ef_construction: " << ef_construction << std::endl;
      std::cout << "- max_candidates: " << max_candidates << std::endl;
      std::cout << "- num_threads: " << num_threads << std::endl;
   }
   auto start_time = std::chrono::high_resolution_clock::now();
   _entry_point = _base_storage->choose_medoid(num_threads, _distance_handler);

   VisitedSet visited;
   visited.init(num_points);
   for (IdxType id = 0; id < num_points; ++id)
   {
      if (id == _entry_point)
         continue;
      std::vector<Candidate> candidates =
          search_layer(_base_storage->get_vector(id),
                       _entry_point,
                       std::max<IdxType>(ef_construction, max_degree),
                       id,
                       visited);
      if (max_candidates > 0 && candidates.size() > max_candidates)
         candidates.resize(max_candidates);
      const std::vector<IdxType> selected = select_neighbors(id, std::move(candidates), max_degree);
      connect_bidirectional(id, selected, max_degree);
   }

   if (_verbose)
   {
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - start_time)
                            .count();
      std::cout << "- NaviX HNSW finished in " << ms << " ms" << std::endl;
   }
}

std::vector<Candidate> NavixHnsw::search_layer(const char *query,
                                               IdxType entry_point,
                                               IdxType ef,
                                               IdxType exclude_id,
                                               VisitedSet &visited) const
{
   const IdxType dim = _base_storage->get_dim();
   std::priority_queue<Candidate, std::vector<Candidate>, CloserFirst> candidates;
   SearchQueue results;
   results.reserve(static_cast<int32_t>(ef));

   visited.clear();
   const float entry_dist =
       _distance_handler->compute(query, _base_storage->get_vector(entry_point), dim);
   candidates.emplace(entry_point, entry_dist);
   if (entry_point != exclude_id)
      results.insert(entry_point, entry_dist);
   visited.set(entry_point);

   while (!candidates.empty())
   {
      const Candidate cur = candidates.top();
      if (results.size() > 0 && results.size() >= static_cast<int32_t>(ef) &&
          cur.distance > results[results.size() - 1].distance)
         break;
      candidates.pop();

      const auto &neighbors = _graph->neighbors[cur.id];
      for (size_t i = 0; i < neighbors.size(); ++i)
      {
         const IdxType neighbor = neighbors[i];
         if (neighbor >= _base_storage->get_num_points() || visited.check(neighbor))
            continue;
         visited.set(neighbor);
         const float dist =
             _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim);
         if (neighbor != exclude_id &&
             (results.size() < static_cast<int32_t>(ef) || dist < results[results.size() - 1].distance))
         {
            candidates.emplace(neighbor, dist);
            results.insert(neighbor, dist);
         }
      }
   }

   std::vector<Candidate> out;
   out.reserve(results.size());
   for (int32_t i = 0; i < results.size(); ++i)
      out.push_back(results[i]);
   std::sort(out.begin(), out.end(), closer_candidate);
   return out;
}

std::vector<IdxType> NavixHnsw::select_neighbors(IdxType src,
                                                 std::vector<Candidate> candidates,
                                                 IdxType max_degree) const
{
   candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                   [src](const Candidate &candidate) {
                                      return candidate.id == src;
                                   }),
                    candidates.end());
   std::sort(candidates.begin(), candidates.end(), closer_candidate);

   std::vector<IdxType> selected;
   selected.reserve(max_degree);
   std::unordered_set<IdxType> seen;
   for (const Candidate &candidate : candidates)
   {
      if (seen.insert(candidate.id).second)
         selected.push_back(candidate.id);
      if (selected.size() >= max_degree)
         break;
   }
   return selected;
}

void NavixHnsw::prune_node(IdxType node, IdxType max_degree)
{
   auto &neighbors = _graph->neighbors[node];
   if (neighbors.size() <= max_degree)
      return;

   std::vector<Candidate> candidates;
   candidates.reserve(neighbors.size());
   const IdxType dim = _base_storage->get_dim();
   for (IdxType neighbor : neighbors)
   {
      if (neighbor == node || neighbor >= _base_storage->get_num_points())
         continue;
      const float dist = _distance_handler->compute(_base_storage->get_vector(node),
                                                    _base_storage->get_vector(neighbor),
                                                    dim);
      candidates.emplace_back(neighbor, dist);
   }
   const std::vector<IdxType> selected = select_neighbors(node, std::move(candidates), max_degree);
   neighbors.clear();
   for (IdxType selected_id : selected)
      neighbors.push_back(selected_id);
}

void NavixHnsw::connect_bidirectional(IdxType src, const std::vector<IdxType> &neighbors, IdxType max_degree)
{
   auto &src_neighbors = _graph->neighbors[src];
   src_neighbors.clear();
   for (IdxType neighbor : neighbors)
      src_neighbors.push_back(neighbor);

   for (IdxType neighbor : neighbors)
   {
      auto &dst_neighbors = _graph->neighbors[neighbor];
      if (std::find(dst_neighbors.begin(), dst_neighbors.end(), src) == dst_neighbors.end())
         dst_neighbors.push_back(src);
      prune_node(neighbor, max_degree);
   }
   prune_node(src, max_degree);
}

} // namespace ANNS
