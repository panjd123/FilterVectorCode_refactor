#include "include/ung_entry_group.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace ANNS
{

const char *entry_group_strategy_name(EntryGroupStrategy strategy)
{
   switch (strategy)
   {
   case EntryGroupStrategy::Original:
      return "original";
   case EntryGroupStrategy::OptimizedLng:
      return "optimized_lng";
   case EntryGroupStrategy::Trie:
      return "trie";
   }
   return "unknown";
}

const char *entry_group_provider_kind_name(EntryGroupProviderKind kind)
{
   switch (kind)
   {
   case EntryGroupProviderKind::Passthrough:
      return "passthrough";
   case EntryGroupProviderKind::Original:
      return "original";
   case EntryGroupProviderKind::OptimizedLng:
      return "optimized_lng";
   case EntryGroupProviderKind::Trie:
      return "trie";
   }
   return "unknown";
}

EntryGroupStrategy parse_entry_group_strategy(const std::string &value)
{
   if (value == "original" || value == "0" || value == "cpu" ||
       value == "cpu_min_super_sets")
      return EntryGroupStrategy::Original;
   if (value == "optimized_lng" || value == "1" || value == "optimized" ||
       value == "cpu_bruteforce_els" || value == "bruteforce_els" ||
       value == "gpu_cover_frontier" || value == "optimized_trie_exact")
      return EntryGroupStrategy::OptimizedLng;
   if (value == "trie" || value == "2" || value == "special_block_trie" ||
       value == "special_trie" || value == "trie_exact")
      return EntryGroupStrategy::Trie;
   throw std::invalid_argument(
       "Invalid entry_group_strategy: " + value +
       " (expected original/0, optimized_lng/1, or trie/2)");
}

std::string make_entry_group_label_cache_key(EntryGroupStrategy strategy,
                                             const std::vector<LabelType> &query_labels,
                                             bool recursive_more_start,
                                             bool ung_more_entry)
{
   std::vector<LabelType> labels = query_labels;
   std::sort(labels.begin(), labels.end());

   std::ostringstream out;
   out << static_cast<int>(strategy) << '|';
   out << (recursive_more_start ? '1' : '0') << '|';
   out << (ung_more_entry ? '1' : '0') << '|';
   for (LabelType label : labels)
      out << label << ',';
   return out.str();
}

void EntryGroupProviderRequest::validate() const
{
   if (query_labels == nullptr)
      throw std::invalid_argument("EntryGroupProviderRequest requires query_labels");
   if (decision == nullptr)
      throw std::invalid_argument("EntryGroupProviderRequest requires decision");
}

bool EntryGroupProviderRequest::has_current_group_ids() const
{
   return current_group_ids != nullptr;
}

IdxType EntryGroupProviderRequest::true_group_id_or(IdxType fallback) const
{
   if (true_query_group_ids == nullptr || query_id >= true_query_group_ids->size())
      return fallback;
   return (*true_query_group_ids)[query_id];
}

} // namespace ANNS
