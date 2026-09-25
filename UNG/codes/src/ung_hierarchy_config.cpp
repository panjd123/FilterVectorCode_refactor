#include "include/ung_hierarchy_config.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ANNS
{
namespace
{
std::string lower(std::string value)
{
   std::transform(value.begin(), value.end(), value.begin(),
                  [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
   return value;
}
}

const char *to_string(GroupTopologyKind value)
{
   return value == GroupTopologyKind::Trie ? "trie" : "lng";
}

GroupTopologyKind parse_group_topology(const std::string &value)
{
   const std::string normalized = lower(value);
   if (normalized == "lng" || normalized == "0")
      return GroupTopologyKind::Lng;
   if (normalized == "trie" || normalized == "trie_tree" || normalized == "1")
      return GroupTopologyKind::Trie;
   throw std::invalid_argument("invalid group topology '" + value + "' (expected lng or trie)");
}

void HierarchyPlan::validate() const
{
   if (layers.size() >= static_cast<size_t>(std::numeric_limits<uint8_t>::max()))
      throw std::invalid_argument("hierarchy supports at most 254 materialized layers");
   IdxType previous = 0;
   for (size_t level = 0; level < layers.size(); ++level)
   {
      const HierarchyLayerSpec &layer = layers[level];
      if (layer.min_points == 0)
         throw std::invalid_argument("hierarchy layer threshold must be positive");
      if (level > 0 && layer.min_points <= previous)
         throw std::invalid_argument(
             "hierarchy layer thresholds must be strictly increasing from fine to coarse");
      previous = layer.min_points;
   }
}

std::string HierarchyPlan::encode() const
{
   std::ostringstream out;
   for (size_t level = 0; level < layers.size(); ++level)
   {
      if (level != 0)
         out << ',';
      out << layers[level].min_points << ':' << to_string(layers[level].topology);
   }
   return out.str();
}

HierarchyPlan parse_hierarchy_plan(const std::string &layers,
                                   GroupTopologyKind base_topology)
{
   HierarchyPlan plan;
   plan.base_topology = base_topology;
   if (layers.empty())
      return plan;

   std::istringstream input(layers);
   std::string item;
   while (std::getline(input, item, ','))
   {
      const size_t colon = item.find(':');
      if (colon == std::string::npos || item.find(':', colon + 1) != std::string::npos)
         throw std::invalid_argument(
             "invalid hierarchy layer '" + item + "' (expected min_points:topology)");
      const std::string threshold_text = item.substr(0, colon);
      const std::string topology_text = item.substr(colon + 1);
      if (threshold_text.empty() || topology_text.empty())
         throw std::invalid_argument("hierarchy layer fields cannot be empty");
      size_t consumed = 0;
      const unsigned long long threshold = std::stoull(threshold_text, &consumed);
      if (consumed != threshold_text.size() || threshold == 0 ||
          threshold > std::numeric_limits<IdxType>::max())
         throw std::invalid_argument("invalid hierarchy layer threshold '" + threshold_text + "'");
      plan.layers.push_back(HierarchyLayerSpec{
          static_cast<IdxType>(threshold), parse_group_topology(topology_text)});
   }
   plan.validate();
   return plan;
}

} // namespace ANNS
