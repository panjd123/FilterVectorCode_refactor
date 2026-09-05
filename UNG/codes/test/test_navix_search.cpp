#include "distance.h"
#include "graph.h"
#include "storage.h"
#include "ung_graph_search_backend.h"
#include "ung_navix_search.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{
void expect(bool condition, const char *message)
{
   if (!condition)
   {
      std::cerr << "FAILED: " << message << '\n';
      std::exit(1);
   }
}

void write_float_dataset(const std::string &bin_file,
                         const std::string &label_file,
                         const std::vector<float> &values,
                         ANNS::IdxType num_points,
                         ANNS::IdxType dim)
{
   std::ofstream bin(bin_file, std::ios::binary);
   bin.write(reinterpret_cast<const char *>(&num_points), sizeof(ANNS::IdxType));
   bin.write(reinterpret_cast<const char *>(&dim), sizeof(ANNS::IdxType));
   bin.write(reinterpret_cast<const char *>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
   bin.close();

   std::ofstream labels(label_file);
   for (ANNS::IdxType i = 0; i < num_points; ++i)
      labels << "1\n";
}
} // namespace

int main()
{
   const std::string bin_file = "/tmp/ung_navix_search_base.bin";
   const std::string label_file = "/tmp/ung_navix_search_base_labels.txt";
   write_float_dataset(bin_file,
                       label_file,
                       {0.0f, 0.0f,
                        1.0f, 0.0f,
                        2.0f, 0.0f},
                       3,
                       2);

   auto storage = ANNS::create_storage("float", false);
   storage->load_from_file(bin_file, label_file);
   std::shared_ptr<ANNS::DistanceHandler> distance = ANNS::get_distance_handler("float", "L2");
   ANNS::Graph graph(storage->get_num_points());
   graph.neighbors[0].push_back(1);
   graph.neighbors[1].push_back(2);

   ANNS::GraphSearchBackend backend(graph);
   ANNS::SearchCache cache(storage->get_num_points(), 4);
   ANNS::SearchQueue result;
   const std::vector<uint8_t> filter_map{0, 0, 1};
   const std::vector<ANNS::IdxType> entry_points{0};
   const float query_vec[2] = {2.0f, 0.0f};

   const ANNS::NavixSearchStats stats = ANNS::navix_adaptive_local_search(
       reinterpret_cast<const char *>(query_vec),
       storage,
       distance,
       backend,
       filter_map,
       entry_points,
       4,
       1,
       cache,
       result);

   expect(result.size() == 1, "NaviX search must return one filtered result");
   expect(result[0].id == 2, "NaviX search must return the filtered second-hop node");
   expect(stats.full_two_hop_calls > 0,
          "low local selectivity must use the full two-hop branch");

   storage->clean();
   std::cout << "navix adaptive-local search checks passed\n";
   return 0;
}
