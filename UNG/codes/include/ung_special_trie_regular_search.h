#ifndef ANNS_UNG_SPECIAL_TRIE_REGULAR_SEARCH_H
#define ANNS_UNG_SPECIAL_TRIE_REGULAR_SEARCH_H

#include "config.h"

#include <vector>

namespace ANNS
{

inline bool special_trie_regular_main_edge_allowed(
    const std::vector<IdxType> &point_to_group,
    IdxType source,
    IdxType target)
{
   if (source >= point_to_group.size() || target >= point_to_group.size())
      return false;
   const IdxType source_group = point_to_group[source];
   return source_group != 0 && point_to_group[target] == source_group;
}

} // namespace ANNS

#endif // ANNS_UNG_SPECIAL_TRIE_REGULAR_SEARCH_H
