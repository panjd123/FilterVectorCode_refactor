#ifndef ANNS_UNG_CPU_BRUTEFORCE_ELS_H
#define ANNS_UNG_CPU_BRUTEFORCE_ELS_H

#include "config.h"

#include <cstdint>
#include <cstddef>
#include <vector>
#include <unordered_map>

#include <roaring/roaring.hh>

namespace ANNS
{

struct CpuBruteForceElsRows
{
   IdxType words_per_query = 0;
   std::unordered_map<LabelType, std::vector<uint64_t>> label_group_bits;
};

CpuBruteForceElsRows build_cpu_bruteforce_els_rows(
    const std::vector<std::vector<LabelType>> &group_labels,
    IdxType num_groups,
    const std::vector<LabelType> &requested_labels);

roaring::Roaring cpu_bruteforce_els_roaring_from_bits(const std::vector<uint64_t> &bits,
                                                      IdxType num_groups_including_zero);

std::vector<IdxType> cpu_bruteforce_els_select_with_roaring(
    const std::vector<uint64_t> &candidate_bits,
    const std::vector<uint64_t> &selected_frontier_bits,
    const std::vector<IdxType> &frontier_ids,
    const std::vector<roaring::Roaring> &descendants,
    IdxType num_groups_including_zero);

std::vector<IdxType> cpu_bruteforce_els_select_scalar_exact(
    const std::vector<std::vector<LabelType>> &group_labels,
    const std::vector<LabelType> &query_labels,
    IdxType num_groups,
    size_t max_results = 0);

} // namespace ANNS

#endif // ANNS_UNG_CPU_BRUTEFORCE_ELS_H
