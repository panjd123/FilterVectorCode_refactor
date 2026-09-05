#include "include/ung_navix_search.h"

#include <algorithm>
#include <queue>
#include <vector>

namespace ANNS
{
namespace
{
bool is_filtered(const std::vector<uint8_t> &filter_map, IdxType id)
{
   return id < filter_map.size() && filter_map[id] != 0;
}

void collect_one_hop(const GraphNeighborView &neighbors,
                     const std::vector<uint8_t> &filter_map,
                     VisitedSet &visited_set,
                     std::vector<IdxType> &out)
{
   for (size_t i = 0; i < neighbors.size; ++i)
   {
      const IdxType neighbor = neighbors.ids[i];
      if (visited_set.check(neighbor) || !is_filtered(filter_map, neighbor))
         continue;
      visited_set.set(neighbor);
      out.push_back(neighbor);
   }
}

void collect_full_two_hop(const GraphSearchBackend &graph_backend,
                          const GraphNeighborView &neighbors,
                          const std::vector<uint8_t> &filter_map,
                          VisitedSet &visited_set,
                          std::vector<IdxType> &out,
                          NavixSearchStats &stats)
{
   std::queue<IdxType> first_hop;
   for (size_t i = 0; i < neighbors.size; ++i)
   {
      const IdxType neighbor = neighbors.ids[i];
      if (!visited_set.check(neighbor))
      {
         if (is_filtered(filter_map, neighbor))
            out.push_back(neighbor);
         first_hop.push(neighbor);
         visited_set.set(neighbor);
      }
   }

   while (!first_hop.empty())
   {
      const IdxType first = first_hop.front();
      first_hop.pop();
      const GraphNeighborView second_hop = graph_backend.neighbors(first);
      ++stats.nodes_visited;
      for (size_t i = 0; i < second_hop.size; ++i)
      {
         const IdxType second = second_hop.ids[i];
         if (visited_set.check(second) || !is_filtered(filter_map, second))
            continue;
         visited_set.set(second);
         out.push_back(second);
      }
   }
}

void collect_directed_two_hop(const char *query,
                              const std::shared_ptr<IStorage> &base_storage,
                              const std::shared_ptr<DistanceHandler> &distance_handler,
                              const GraphSearchBackend &graph_backend,
                              const GraphNeighborView &neighbors,
                              const std::vector<uint8_t> &filter_map,
                              VisitedSet &visited_set,
                              std::vector<IdxType> &out,
                              IdxType filter_neighbors_to_find,
                              NavixSearchStats &stats)
{
   const IdxType dim = base_storage->get_dim();
   std::vector<Candidate> first_hop;
   first_hop.reserve(neighbors.size);
   size_t filtered_seen = 0;

   for (size_t i = 0; i < neighbors.size; ++i)
   {
      const IdxType neighbor = neighbors.ids[i];
      const bool filtered = is_filtered(filter_map, neighbor);
      if (filtered)
         ++filtered_seen;
      if (visited_set.check(neighbor))
         continue;
      const float dist = distance_handler->compute(query, base_storage->get_vector(neighbor), dim);
      ++stats.distance_calcs;
      first_hop.emplace_back(neighbor, dist);
      if (filtered)
      {
         visited_set.set(neighbor);
         out.push_back(neighbor);
      }
   }

   std::sort(first_hop.begin(), first_hop.end());
   for (const Candidate &candidate : first_hop)
   {
      if (filtered_seen >= filter_neighbors_to_find)
         break;
      if (visited_set.check(candidate.id))
         continue;
      visited_set.set(candidate.id);
      const GraphNeighborView second_hop = graph_backend.neighbors(candidate.id);
      ++stats.nodes_visited;
      for (size_t i = 0; i < second_hop.size; ++i)
      {
         const IdxType second = second_hop.ids[i];
         const bool filtered = is_filtered(filter_map, second);
         if (filtered)
            ++filtered_seen;
         if (visited_set.check(second) || !filtered)
            continue;
         visited_set.set(second);
         out.push_back(second);
      }
   }
}

void insert_candidates(const char *query,
                       const std::shared_ptr<IStorage> &base_storage,
                       const std::shared_ptr<DistanceHandler> &distance_handler,
                       const std::vector<uint8_t> &filter_map,
                       const std::vector<IdxType> &candidate_ids,
                       SearchQueue &candidate_queue,
                       SearchQueue &result,
                       IdxType K,
                       NavixSearchStats &stats)
{
   const IdxType dim = base_storage->get_dim();
   for (IdxType candidate_id : candidate_ids)
   {
      if (candidate_id >= filter_map.size())
         continue;
      const float dist = distance_handler->compute(query, base_storage->get_vector(candidate_id), dim);
      ++stats.distance_calcs;
      candidate_queue.insert(candidate_id, dist);
      if (is_filtered(filter_map, candidate_id))
      {
         result.insert(candidate_id, dist);
         while (result.size() > static_cast<int32_t>(K))
            break;
      }
   }
}
} // namespace

NavixSearchStats navix_adaptive_local_search(const char *query,
                                             const std::shared_ptr<IStorage> &base_storage,
                                             const std::shared_ptr<DistanceHandler> &distance_handler,
                                             const GraphSearchBackend &graph_backend,
                                             const std::vector<uint8_t> &filter_map,
                                             const std::vector<IdxType> &entry_points,
                                             IdxType ef_search,
                                             IdxType K,
                                             SearchCache &search_cache,
                                             SearchQueue &result)
{
   NavixSearchStats stats;
   const IdxType dim = base_storage->get_dim();
   SearchQueue &candidate_queue = search_cache.search_queue;
   VisitedSet &visited_set = search_cache.visited_set;
   candidate_queue.clear();
   candidate_queue.reserve(static_cast<int32_t>(ef_search));
   result.clear();
   result.reserve(static_cast<int32_t>(K));
   visited_set.clear();

   for (IdxType entry_point : entry_points)
   {
      if (entry_point >= filter_map.size() || visited_set.check(entry_point))
         continue;
      visited_set.set(entry_point);
      const float dist = distance_handler->compute(query, base_storage->get_vector(entry_point), dim);
      ++stats.distance_calcs;
      candidate_queue.insert(entry_point, dist);
      if (is_filtered(filter_map, entry_point))
         result.insert(entry_point, dist);
   }

   IdxType seeded = 0;
   for (IdxType id = 0; id < static_cast<IdxType>(filter_map.size()) && seeded < K; ++id)
   {
      if (!is_filtered(filter_map, id) || visited_set.check(id))
         continue;
      visited_set.set(id);
      const float dist = distance_handler->compute(query, base_storage->get_vector(id), dim);
      ++stats.distance_calcs;
      candidate_queue.insert(id, dist);
      result.insert(id, dist);
      ++seeded;
   }

   while (candidate_queue.has_unexpanded_node())
   {
      const Candidate cur = candidate_queue.get_closest_unexpanded();
      const GraphNeighborView neighbors = graph_backend.neighbors(cur.id);
      ++stats.nodes_visited;
      if (neighbors.size == 0)
         continue;

      size_t filtered_neighbors = 0;
      for (size_t i = 0; i < neighbors.size; ++i)
         if (is_filtered(filter_map, neighbors.ids[i]))
            ++filtered_neighbors;

      const float local_selectivity =
          static_cast<float>(filtered_neighbors) / static_cast<float>(neighbors.size);
      const float estimated_full_two_hop_distance_comp =
          (static_cast<float>(neighbors.size) * static_cast<float>(filtered_neighbors) +
           static_cast<float>(filtered_neighbors)) *
          0.4f;
      const float estimated_directed_distance_comp =
          static_cast<float>(neighbors.size + (neighbors.size - filtered_neighbors));

      std::vector<IdxType> next_candidates;
      if (local_selectivity >= 0.4f)
      {
         collect_one_hop(neighbors, filter_map, visited_set, next_candidates);
         ++stats.one_hop_calls;
      }
      else if (estimated_full_two_hop_distance_comp > estimated_directed_distance_comp)
      {
         collect_directed_two_hop(query, base_storage, distance_handler, graph_backend,
                                  neighbors, filter_map, visited_set, next_candidates,
                                  static_cast<IdxType>(neighbors.size), stats);
         ++stats.directed_two_hop_calls;
      }
      else
      {
         collect_full_two_hop(graph_backend, neighbors, filter_map, visited_set,
                              next_candidates, stats);
         ++stats.full_two_hop_calls;
      }

      insert_candidates(query, base_storage, distance_handler, filter_map,
                        next_candidates, candidate_queue, result, K, stats);
   }

   return stats;
}

} // namespace ANNS
