#ifndef ANNS_SPECIAL_BLOCK_INDEX_BUILDER_H
#define ANNS_SPECIAL_BLOCK_INDEX_BUILDER_H

#include "config.h"

#include <cstdint>
#include <string>

namespace ANNS
{

struct SpecialBlockIndexBuildOptions
{
   std::string ung_index_path_prefix;
   std::string base_bin_file;
   std::string base_label_file;
   std::string block_index_path_prefix;
   std::string result_path_prefix;
   std::string data_type;
   std::string distance_function;
   uint32_t num_threads = 1;
   IdxType min_points = 100;
   IdxType max_degree = 64;
   IdxType num_cross_edges = 6;
   IdxType Lbuild = 100;
   float alpha = 1.2f;
};

class SpecialBlockIndexBuilder
{
public:
   void build(const SpecialBlockIndexBuildOptions &options) const;
};

} // namespace ANNS

#endif // ANNS_SPECIAL_BLOCK_INDEX_BUILDER_H
