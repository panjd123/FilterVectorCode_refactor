#ifndef ANNS_UNG_GROUP_TOPOLOGY_H
#define ANNS_UNG_GROUP_TOPOLOGY_H

#include "config.h"
#include "label_nav_graph.h"
#include "ung_hierarchy_config.h"
#include "ung_special_blocks.h"

#include <memory>
#include <vector>

namespace ANNS
{

struct GroupTopologyStats
{
   GroupTopologyKind kind = GroupTopologyKind::Lng;
   IdxType group_count = 0;
   uint64_t edge_count = 0;
};

std::shared_ptr<LabelNavGraph> build_group_topology(
    GroupTopologyKind kind,
    const std::vector<std::vector<LabelType>> &group_labels,
    IdxType num_groups,
    const std::shared_ptr<LabelNavGraph> &lng_graph,
    GroupTopologyStats *stats = nullptr);

void apply_layer_topology(GroupTopologyKind kind,
                          uint8_t level,
                          std::vector<SpecialBlock> &blocks);

} // namespace ANNS

#endif // ANNS_UNG_GROUP_TOPOLOGY_H
