#ifndef ANNS_UNG_LNG_BLOCK_PARTITION_H
#define ANNS_UNG_LNG_BLOCK_PARTITION_H

#include "ung_special_blocks.h"

#include <cstdint>
#include <vector>

namespace ANNS
{

enum class LngBlockTreeMode : uint8_t
{
   Random = 0,
   Bfs = 1,
};

struct LngBlockPartitionInput
{
   const std::vector<std::vector<IdxType>> *out_neighbors = nullptr;
   const std::vector<IdxType> *group_points = nullptr;
   const std::vector<std::vector<LabelType>> *group_labels = nullptr;
   IdxType min_points = 0;
   LngBlockTreeMode tree_mode = LngBlockTreeMode::Random;
   uint64_t tree_seed = 1;
};

struct LngBlockPartitionResult
{
   std::vector<SpecialBlock> blocks;
   std::vector<IdxType> group_to_block;
};

LngBlockPartitionResult build_lng_special_blocks(const LngBlockPartitionInput &input);

} // namespace ANNS

#endif // ANNS_UNG_LNG_BLOCK_PARTITION_H
