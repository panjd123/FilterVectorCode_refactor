#ifndef ANNS_NAVIX_INDEX_H
#define ANNS_NAVIX_INDEX_H

#include "config.h"
#include "distance.h"
#include "graph.h"
#include "navix_hnsw.h"
#include "search_cache.h"
#include "ung_query_stats.h"
#include "storage.h"
#include "ung_graph_search_backend.h"
#include "ung_navix_search.h"

#include <memory>
#include <string>
#include <vector>

namespace ANNS
{

class StandaloneNavixIndex
{
public:
   void build(std::shared_ptr<IStorage> base_storage,
              std::shared_ptr<DistanceHandler> distance_handler,
              IdxType max_degree,
              IdxType ef_construction,
              IdxType max_candidates,
              uint32_t num_threads);

   void save(const std::string &index_path_prefix,
             const std::string &result_path_prefix,
             double build_ms) const;
   void load(const std::string &index_path_prefix,
             const std::string &data_type);

   void search(std::shared_ptr<IStorage> query_storage,
               std::shared_ptr<DistanceHandler> distance_handler,
               uint32_t num_threads,
               IdxType Lsearch,
               IdxType K,
               std::pair<IdxType, float> *results,
               std::vector<float> &num_cmps,
               std::vector<QueryStats> &query_stats) const;

private:
   std::vector<uint8_t> make_filter_map(const std::vector<LabelType> &query_labels) const;

   std::shared_ptr<IStorage> _base_storage;
   std::shared_ptr<Graph> _graph;
   IdxType _entry_point = 0;
   IdxType _max_degree = 0;
   IdxType _ef_construction = 0;
   IdxType _max_candidates = 0;
   uint32_t _num_threads = 1;
};

} // namespace ANNS

#endif // ANNS_NAVIX_INDEX_H
