#include "navix_index.h"
#include "distance.h"
#include "storage.h"

#include <boost/filesystem.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
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
                         ANNS::IdxType dim,
                         const std::vector<std::string> &labels_per_point)
{
   std::ofstream bin(bin_file, std::ios::binary);
   bin.write(reinterpret_cast<const char *>(&num_points), sizeof(ANNS::IdxType));
   bin.write(reinterpret_cast<const char *>(&dim), sizeof(ANNS::IdxType));
   bin.write(reinterpret_cast<const char *>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
   bin.close();

   std::ofstream labels(label_file);
   for (const auto &labels_for_point : labels_per_point)
      labels << labels_for_point << "\n";
}
} // namespace

int main()
{
   const std::string root = "/tmp/ung_navix_index_test/";
   boost::filesystem::remove_all(root);
   boost::filesystem::create_directories(root);

   const std::string base_bin = root + "base.bin";
   const std::string base_labels = root + "base_labels.txt";
   const std::string query_bin = root + "query.bin";
   const std::string query_labels = root + "query_labels.txt";
   write_float_dataset(base_bin,
                       base_labels,
                       {0.0f, 0.0f,
                        1.0f, 0.0f,
                        2.0f, 0.0f,
                        10.0f, 0.0f},
                       4,
                       2,
                       {"1", "1", "1,2", "3"});
   write_float_dataset(query_bin,
                       query_labels,
                       {2.0f, 0.0f},
                       1,
                       2,
                       {"2"});

   auto base_storage = ANNS::create_storage("float", false);
   base_storage->load_from_file(base_bin, base_labels);
   std::shared_ptr<ANNS::DistanceHandler> distance = ANNS::get_distance_handler("float", "L2");

   ANNS::StandaloneNavixIndex built;
   built.build(base_storage, distance, 2, 8, 8, 1);
   built.save(root + "index/", root + "results/", 12.5);

   ANNS::StandaloneNavixIndex loaded;
   loaded.load(root + "index/", "float");

   auto query_storage = ANNS::create_storage("float", false);
   query_storage->load_from_file(query_bin, query_labels);
   std::pair<ANNS::IdxType, float> result[1];
   std::vector<float> num_cmps;
   std::vector<ANNS::QueryStats> query_stats;
   loaded.search(query_storage, distance, 1, 4, 1, result, num_cmps, query_stats);

   expect(result[0].first == 2, "standalone NaviX must search the saved global index");
   expect(!num_cmps.empty() && num_cmps[0] > 0.0f, "standalone NaviX must report distance computations");
   expect(query_stats.size() == 1 && query_stats[0].query_length == 1,
          "standalone NaviX must preserve query stats");

   base_storage->clean();
   query_storage->clean();
   std::cout << "standalone navix index checks passed\n";
   return 0;
}
