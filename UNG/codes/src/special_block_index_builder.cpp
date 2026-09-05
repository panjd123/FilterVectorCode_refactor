#include "include/special_block_index_builder.h"

#include "include/distance.h"
#include "include/uni_nav_graph.h"

#include <memory>

namespace ANNS
{

void SpecialBlockIndexBuilder::build(const SpecialBlockIndexBuildOptions &options) const
{
   std::shared_ptr<DistanceHandler> distance_handler =
       get_distance_handler(options.data_type, options.distance_function);
   UniNavGraph implementation;
   implementation.build_special_block_index(
       options.ung_index_path_prefix,
       options.base_bin_file,
       options.base_label_file,
       options.block_index_path_prefix,
       options.result_path_prefix,
       options.data_type,
       distance_handler,
       options.num_threads,
       options.min_points,
       options.max_degree,
       options.num_cross_edges,
       options.Lbuild,
       options.alpha);
}

} // namespace ANNS
