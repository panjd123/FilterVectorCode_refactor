#include "include/navix_index.h"

#include "include/utils.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <stdexcept>

#include <boost/filesystem.hpp>
#include <omp.h>

namespace fs = boost::filesystem;

namespace ANNS
{

void StandaloneNavixIndex::build(std::shared_ptr<IStorage> base_storage,
                                 std::shared_ptr<DistanceHandler> distance_handler,
                                 IdxType max_degree,
                                 IdxType ef_construction,
                                 IdxType max_candidates,
                                 uint32_t num_threads)
{
   _base_storage = std::move(base_storage);
   _graph = std::make_shared<Graph>(_base_storage->get_num_points());
   _max_degree = max_degree;
   _ef_construction = ef_construction;
   _max_candidates = max_candidates;
   _num_threads = num_threads;

   NavixHnsw hnsw(true);
   hnsw.build(_base_storage, std::move(distance_handler), _graph,
              max_degree, ef_construction, max_candidates, num_threads);
   _entry_point = hnsw.get_entry_point();
}

void StandaloneNavixIndex::save(const std::string &index_path_prefix,
                                const std::string &result_path_prefix,
                                double build_ms) const
{
   fs::create_directories(index_path_prefix);
   fs::create_directories(result_path_prefix);

   std::map<std::string, std::string> meta;
   meta["index_name"] = "NaviX";
   meta["num_points"] = std::to_string(_base_storage->get_num_points());
   meta["dim"] = std::to_string(_base_storage->get_dim());
   meta["entry_point"] = std::to_string(_entry_point);
   meta["max_degree"] = std::to_string(_max_degree);
   meta["ef_construction"] = std::to_string(_ef_construction);
   meta["max_candidates"] = std::to_string(_max_candidates);
   meta["build_num_threads"] = std::to_string(_num_threads);
   meta["index_time(ms)"] = std::to_string(build_ms);
   write_kv_file(index_path_prefix + "meta", meta);

   std::string graph_file = index_path_prefix + "graph";
   _graph->save(graph_file);
   _base_storage->write_to_file(index_path_prefix + "vecs.bin",
                                index_path_prefix + "labels.txt");

   std::ofstream build_time(result_path_prefix + "build_time.csv");
   build_time << "Index Name,Build Time (ms)\n";
   build_time << "index_time," << build_ms << "\n";
}

void StandaloneNavixIndex::load(const std::string &index_path_prefix,
                                const std::string &data_type)
{
   const std::map<std::string, std::string> meta = parse_kv_file(index_path_prefix + "meta");
   auto find_required = [&](const std::string &key) -> std::string {
      const auto it = meta.find(key);
      if (it == meta.end())
         throw std::runtime_error("NaviX meta missing key: " + key);
      return it->second;
   };

   _entry_point = static_cast<IdxType>(std::stoull(find_required("entry_point")));
   _max_degree = static_cast<IdxType>(std::stoull(find_required("max_degree")));
   _ef_construction = static_cast<IdxType>(std::stoull(find_required("ef_construction")));
   _max_candidates = static_cast<IdxType>(std::stoull(find_required("max_candidates")));
   _num_threads = static_cast<uint32_t>(std::stoul(find_required("build_num_threads")));

   _base_storage = create_storage(data_type, false);
   _base_storage->load_from_file(index_path_prefix + "vecs.bin",
                                 index_path_prefix + "labels.txt");
   _graph = std::make_shared<Graph>(_base_storage->get_num_points());
   std::string graph_file = index_path_prefix + "graph";
   _graph->load(graph_file);
}

std::vector<uint8_t> StandaloneNavixIndex::make_filter_map(const std::vector<LabelType> &query_labels) const
{
   std::vector<uint8_t> filter_map(_base_storage->get_num_points(), 0);
   std::vector<LabelType> sorted_query = query_labels;
   std::sort(sorted_query.begin(), sorted_query.end());
   for (IdxType id = 0; id < _base_storage->get_num_points(); ++id)
   {
      std::vector<LabelType> base_labels = _base_storage->get_label_set(id);
      std::sort(base_labels.begin(), base_labels.end());
      if (std::includes(base_labels.begin(), base_labels.end(),
                        sorted_query.begin(), sorted_query.end()))
         filter_map[id] = 1;
   }
   return filter_map;
}

void StandaloneNavixIndex::search(std::shared_ptr<IStorage> query_storage,
                                  std::shared_ptr<DistanceHandler> distance_handler,
                                  uint32_t num_threads,
                                  IdxType Lsearch,
                                  IdxType K,
                                  std::pair<IdxType, float> *results,
                                  std::vector<float> &num_cmps,
                                  std::vector<QueryStats> &query_stats) const
{
   const IdxType num_queries = query_storage->get_num_points();
   query_stats.resize(num_queries);
   num_cmps.assign(num_queries, 0.0f);
   const GraphSearchBackend backend(*_graph);

   omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
   for (IdxType qid = 0; qid < num_queries; ++qid)
   {
      auto start = std::chrono::high_resolution_clock::now();
      const IdxType effective_Lsearch = std::max(Lsearch, K);
      SearchCache cache(_base_storage->get_num_points(), static_cast<int32_t>(effective_Lsearch));
      SearchQueue result_queue;
      const std::vector<LabelType> query_labels = query_storage->get_label_set(qid);
      const std::vector<uint8_t> filter_map = make_filter_map(query_labels);
      const std::vector<IdxType> entry_points{_entry_point};
      query_stats[qid].query_length = query_labels.size();

      auto core_start = std::chrono::high_resolution_clock::now();
      const NavixSearchStats stats = navix_adaptive_local_search(
          query_storage->get_vector(qid), _base_storage, distance_handler, backend,
          filter_map, entry_points, effective_Lsearch, K, cache, result_queue);
      query_stats[qid].core_search_time_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - core_start)
              .count();

      query_stats[qid].num_distance_calcs = stats.distance_calcs;
      query_stats[qid].num_nodes_visited = stats.nodes_visited;
      query_stats[qid].search_time_ms = query_stats[qid].core_search_time_ms;
      query_stats[qid].time_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - start)
              .count();
      num_cmps[qid] = static_cast<float>(stats.distance_calcs);

      for (IdxType k = 0; k < K; ++k)
      {
         if (k < static_cast<IdxType>(result_queue.size()))
         {
            results[qid * K + k].first = result_queue[k].id;
            results[qid * K + k].second = result_queue[k].distance;
         }
         else
         {
            results[qid * K + k].first = -1;
            results[qid * K + k].second = 0.0f;
         }
      }
   }
}

} // namespace ANNS
