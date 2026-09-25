#ifndef ANNS_UNG_HIERARCHY_CONFIG_H
#define ANNS_UNG_HIERARCHY_CONFIG_H

#include "config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ANNS
{

// The group-to-group topology is independent from both layer partitioning and
// entry-group discovery.  LNG connects every group to its minimal strict label
// supersets; Trie connects terminal groups to the nearest terminal descendants
// on the canonical label-prefix trie.
enum class GroupTopologyKind : uint8_t
{
   Lng = 0,
   Trie = 1,
};

struct HierarchyLayerSpec
{
   IdxType min_points = 0;
   GroupTopologyKind topology = GroupTopologyKind::Trie;
};

struct HierarchyPlan
{
   GroupTopologyKind base_topology = GroupTopologyKind::Lng;
   std::vector<HierarchyLayerSpec> layers;

   bool empty() const { return layers.empty(); }
   size_t layer_count() const { return layers.size(); }
   void validate() const;
   std::string encode() const;
};

const char *to_string(GroupTopologyKind value);
GroupTopologyKind parse_group_topology(const std::string &value);

// Grammar: comma-separated <min_points>:<topology> entries, ordered from the
// finest to the coarsest layer, e.g. "1000:trie,16000:lng,400000:trie".
HierarchyPlan parse_hierarchy_plan(const std::string &layers,
                                   GroupTopologyKind base_topology);

} // namespace ANNS

#endif // ANNS_UNG_HIERARCHY_CONFIG_H
