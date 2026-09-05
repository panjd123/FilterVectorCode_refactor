#include "include/ung_cpu_bruteforce_els.h"

#include <algorithm>

namespace ANNS
{

CpuBruteForceElsRows build_cpu_bruteforce_els_rows(
    const std::vector<std::vector<LabelType>> &group_labels,
    IdxType num_groups,
    const std::vector<LabelType> &requested_labels)
{
   CpuBruteForceElsRows rows;
   rows.words_per_query = (num_groups + 1 + 63) / 64;
   for (LabelType label : requested_labels)
      rows.label_group_bits.try_emplace(label, rows.words_per_query, uint64_t{0});

   for (IdxType gid = 1; gid <= num_groups && gid < static_cast<IdxType>(group_labels.size()); ++gid)
   {
      for (LabelType label : group_labels[gid])
      {
         const auto row = rows.label_group_bits.find(label);
         if (row != rows.label_group_bits.end())
            row->second[gid >> 6] |= uint64_t{1} << (gid & 63);
      }
   }
   return rows;
}

roaring::Roaring cpu_bruteforce_els_roaring_from_bits(const std::vector<uint64_t> &bits,
                                                      IdxType num_groups_including_zero)
{
   roaring::Roaring rb;
   for (IdxType word = 0; word < static_cast<IdxType>(bits.size()); ++word)
   {
      uint64_t active = bits[word];
      while (active)
      {
         const IdxType bit = static_cast<IdxType>(__builtin_ctzll(active));
         const IdxType gid = (word << 6) + bit;
         if (gid > 0 && gid < num_groups_including_zero)
            rb.add(gid);
         active &= active - 1;
      }
   }
   return rb;
}

std::vector<IdxType> cpu_bruteforce_els_select_with_roaring(
    const std::vector<uint64_t> &candidate_bits,
    const std::vector<uint64_t> &selected_frontier_bits,
    const std::vector<IdxType> &frontier_ids,
    const std::vector<roaring::Roaring> &descendants,
    IdxType num_groups_including_zero)
{
   roaring::Roaring covered;
   for (IdxType gid : frontier_ids)
   {
      if (gid > 0 && gid < static_cast<IdxType>(descendants.size()))
         covered |= descendants[gid];
   }

   roaring::Roaring result = cpu_bruteforce_els_roaring_from_bits(selected_frontier_bits,
                                                                  num_groups_including_zero);
   roaring::Roaring uncovered = cpu_bruteforce_els_roaring_from_bits(candidate_bits,
                                                                     num_groups_including_zero);
   uncovered -= covered;
   result |= uncovered;

   std::vector<IdxType> group_ids;
   group_ids.reserve(result.cardinality());
   for (auto it = result.begin(); it != result.end(); ++it)
      group_ids.push_back(static_cast<IdxType>(*it));
   std::sort(group_ids.begin(), group_ids.end());
   return group_ids;
}

std::vector<IdxType> cpu_bruteforce_els_select_scalar_exact(
    const std::vector<std::vector<LabelType>> &group_labels,
    const std::vector<LabelType> &query_labels,
    IdxType num_groups,
    size_t max_results)
{
   std::vector<IdxType> candidates;
   candidates.reserve(num_groups);

   for (IdxType gid = 1; gid <= num_groups && gid < static_cast<IdxType>(group_labels.size()); ++gid)
   {
      const auto &labels = group_labels[gid];
      if (labels.size() < query_labels.size())
         continue;
      if (std::includes(labels.begin(), labels.end(),
                        query_labels.begin(), query_labels.end()))
         candidates.push_back(gid);
   }

   size_t max_label_size = 0;
   for (IdxType gid : candidates)
      max_label_size = std::max(max_label_size, group_labels[gid].size());

   std::vector<std::vector<size_t>> candidate_indices_by_size(max_label_size + 1);
   for (size_t i = 0; i < candidates.size(); ++i)
      candidate_indices_by_size[group_labels[candidates[i]].size()].push_back(i);

   std::vector<char> active(candidates.size(), 1);
   size_t remaining = candidates.size();
   const bool stop_at_cap = max_results > 0;

   for (size_t cover_size = query_labels.size(); cover_size < candidate_indices_by_size.size(); ++cover_size)
   {
      if (stop_at_cap && remaining <= max_results)
         break;

      for (size_t source_index : candidate_indices_by_size[cover_size])
      {
         if (stop_at_cap && remaining <= max_results)
            break;
         if (!active[source_index])
            continue;

         const IdxType source_gid = candidates[source_index];
         const auto &source_labels = group_labels[source_gid];
         for (size_t target_size = cover_size + 1; target_size < candidate_indices_by_size.size(); ++target_size)
         {
            for (size_t target_index : candidate_indices_by_size[target_size])
            {
               if (stop_at_cap && remaining <= max_results)
                  break;
               if (!active[target_index])
                  continue;

               const IdxType target_gid = candidates[target_index];
               const auto &target_labels = group_labels[target_gid];
               if (std::includes(target_labels.begin(), target_labels.end(),
                                 source_labels.begin(), source_labels.end()))
               {
                  active[target_index] = 0;
                  --remaining;
               }
            }
            if (stop_at_cap && remaining <= max_results)
               break;
         }
      }
   }

   std::vector<IdxType> result;
   result.reserve(remaining);
   for (size_t i = 0; i < candidates.size(); ++i)
   {
      if (active[i])
         result.push_back(candidates[i]);
   }
   return result;
}

} // namespace ANNS
