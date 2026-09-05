#include "navix_hnsw.h"
#include "distance.h"
#include "storage.h"

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
   const std::string bin_file = "/tmp/ung_navix_hnsw_base.bin";
   const std::string label_file = "/tmp/ung_navix_hnsw_base_labels.txt";
   write_float_dataset(bin_file,
                       label_file,
                       {0.0f, 0.0f,
                        1.0f, 0.0f,
                        2.0f, 0.0f,
                        10.0f, 0.0f,
                        11.0f, 0.0f},
                       5,
                       2);

   auto storage = ANNS::create_storage("float", false);
   storage->load_from_file(bin_file, label_file);
   std::shared_ptr<ANNS::DistanceHandler> distance = ANNS::get_distance_handler("float", "L2");
   auto graph = std::make_shared<ANNS::Graph>(storage->get_num_points());

   ANNS::NavixHnsw index(false);
   index.build(storage, distance, graph, 2, 8, 8, 1);

   expect(index.get_entry_point() < storage->get_num_points(),
          "HNSW entry point must be a valid vector id");

   bool has_edge = false;
   for (ANNS::IdxType id = 0; id < storage->get_num_points(); ++id)
   {
      expect(graph->neighbors[id].size() <= 2,
             "HNSW lower-layer neighbor list must respect max degree");
      has_edge = has_edge || !graph->neighbors[id].empty();
   }
   expect(has_edge, "HNSW graph must contain at least one edge");

   storage->clean();
   std::cout << "navix hnsw build checks passed\n";
   return 0;
}
