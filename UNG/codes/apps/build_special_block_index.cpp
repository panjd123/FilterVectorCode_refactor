#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include <boost/program_options.hpp>

#include "special_block_index_builder.h"
#include "uni_nav_graph.h"

namespace po = boost::program_options;

int main(int argc, char **argv)
{
   std::string ung_index_path_prefix;
   std::string base_bin_file;
   std::string base_label_file;
   std::string block_index_path_prefix;
   std::string result_path_prefix;
   std::string data_type;
   std::string dist_fn;
   uint32_t num_threads = 1;
   ANNS::IdxType min_points = 100;
   ANNS::IdxType max_degree = ANNS::default_paras::MAX_DEGREE;
   ANNS::IdxType num_cross_edges = ANNS::default_paras::NUM_CROSS_EDGES;
   ANNS::IdxType Lbuild = ANNS::default_paras::L_BUILD;
   float alpha = ANNS::default_paras::ALPHA;

   try
   {
      po::options_description desc{"Arguments"};
      desc.add_options()
          ("help,h", "Print information on arguments")
          ("ung_index_path_prefix", po::value<std::string>(&ung_index_path_prefix)->required(),
           "Path prefix of the independently built UNG index; source data is used when required files are absent")
          ("base_bin_file", po::value<std::string>(&base_bin_file),
           "Fallback base vectors when the UNG index files are unavailable")
          ("base_label_file", po::value<std::string>(&base_label_file),
           "Fallback base labels when the UNG index files are unavailable")
          ("block_index_path_prefix", po::value<std::string>(&block_index_path_prefix)->required(),
           "Path prefix for the independent hierarchy index")
          ("result_path_prefix", po::value<std::string>(&result_path_prefix)->required(),
           "Path prefix for block build results")
          ("data_type", po::value<std::string>(&data_type)->required(),
           "Data type <int8/uint8/float>")
          ("dist_fn", po::value<std::string>(&dist_fn)->required(),
           "Distance function <L2/IP/cosine>")
          ("num_threads", po::value<uint32_t>(&num_threads)->default_value(1),
           "Number of build threads")
          ("min_points", po::value<ANNS::IdxType>(&min_points)->default_value(100),
           "Minimum uncovered point count for creating a block")
          ("max_degree", po::value<ANNS::IdxType>(&max_degree)->default_value(ANNS::default_paras::MAX_DEGREE),
           "Maximum block-local graph degree")
          ("num_cross_edges", po::value<ANNS::IdxType>(&num_cross_edges)->default_value(ANNS::default_paras::NUM_CROSS_EDGES),
           "Edges per same-layer topology relation")
          ("Lbuild", po::value<ANNS::IdxType>(&Lbuild)->default_value(ANNS::default_paras::L_BUILD),
           "Block-local Vamana build candidate size")
          ("alpha", po::value<float>(&alpha)->default_value(ANNS::default_paras::ALPHA),
           "Block-local Vamana pruning alpha");

      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, desc), vm);
      if (vm.count("help"))
      {
         std::cout << desc;
         return 0;
      }
      po::notify(vm);
   }
   catch (const std::exception &ex)
   {
      std::cerr << ex.what() << std::endl;
      return 1;
   }

   try
   {
      if (std::getenv("UNG_SPECIAL_EDGE_BINARY_ONLY") == nullptr &&
          std::getenv("UNG_SPECIAL_EDGE_BINARY") == nullptr)
         setenv("UNG_SPECIAL_EDGE_BINARY_ONLY", "1", 0);
      // This executable exists solely to build a hierarchy sidecar. Do not
      // require callers to repeat the legacy feature-enable environment flag.
      setenv("UNG_SPECIAL_BLOCKS", "1", 1);
      ANNS::SpecialBlockIndexBuildOptions options;
      options.ung_index_path_prefix = ung_index_path_prefix;
      options.base_bin_file = base_bin_file;
      options.base_label_file = base_label_file;
      options.block_index_path_prefix = block_index_path_prefix;
      options.result_path_prefix = result_path_prefix;
      options.data_type = data_type;
      options.distance_function = dist_fn;
      options.num_threads = num_threads;
      options.build_config = ANNS::UngBuildConfig::from_env(
          num_threads, static_cast<uint32_t>(min_points));
      options.max_degree = max_degree;
      options.num_cross_edges = num_cross_edges;
      options.Lbuild = Lbuild;
      options.alpha = alpha;
      ANNS::SpecialBlockIndexBuilder builder;
      const auto start = std::chrono::high_resolution_clock::now();
      builder.build(options);
      std::cout << "Independent hierarchy index built in "
                << std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - start)
                       .count()
                << " ms" << std::endl;
   }
   catch (const std::exception &ex)
   {
      std::cerr << "Failed to build independent hierarchy index: " << ex.what() << std::endl;
      return 1;
   }
   return 0;
}
