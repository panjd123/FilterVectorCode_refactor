#ifndef ANNS_UNG_FAVOR_BLOCK_SEARCH_H
#define ANNS_UNG_FAVOR_BLOCK_SEARCH_H

#include "config.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace ANNS
{


struct FavorBlockCandidate
{
   IdxType block_id = 0;
   size_t score = 0;
   size_t point_count = 0;
};

inline std::vector<IdxType> favor_select_top_blocks(std::vector<FavorBlockCandidate> candidates,
                                                    size_t max_blocks)
{
   candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                   [](const FavorBlockCandidate &candidate) {
                                      return candidate.block_id == 0 || candidate.score == 0;
                                   }),
                    candidates.end());
   std::sort(candidates.begin(), candidates.end(),
             [](const FavorBlockCandidate &a, const FavorBlockCandidate &b) {
                if (a.score != b.score)
                   return a.score > b.score;
                if (a.point_count != b.point_count)
                   return a.point_count < b.point_count;
                return a.block_id < b.block_id;
             });
   if (max_blocks > 0 && candidates.size() > max_blocks)
      candidates.resize(max_blocks);
   std::vector<IdxType> selected;
   selected.reserve(candidates.size());
   for (const FavorBlockCandidate &candidate : candidates)
      selected.push_back(candidate.block_id);
   return selected;
}

inline float favor_exclusion_distance(float selectivity,
                                      float lsearch,
                                      float delta_d,
                                      float alpha)
{
   constexpr float kMinSelectivity = 1e-6f;
   const float p = std::max(selectivity, kMinSelectivity);
   const float safe_lsearch = std::max(lsearch, 1.0f);
   const float safe_delta = std::max(delta_d, 0.0f);
   const float safe_alpha = std::max(alpha, 0.0f);
   return safe_alpha * (1.0f - p) * (safe_lsearch - p) * safe_delta / (2.0f * p);
}

inline float favor_adjusted_distance(float base_distance,
                                     bool is_target,
                                     float exclusion_distance)
{
   return is_target ? base_distance : base_distance + exclusion_distance;
}

} // namespace ANNS

#endif // ANNS_UNG_FAVOR_BLOCK_SEARCH_H
