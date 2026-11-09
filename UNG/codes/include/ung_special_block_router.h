#ifndef ANNS_UNG_SPECIAL_BLOCK_ROUTER_H
#define ANNS_UNG_SPECIAL_BLOCK_ROUTER_H

#include "config.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace ANNS
{

struct SpecialBlockRouterChild
{
   IdxType block_id = 0;
   float portal_distance = std::numeric_limits<float>::infinity();
   float lower_bound_distance = std::numeric_limits<float>::infinity();
   IdxType best_portal_id = std::numeric_limits<IdxType>::max();
};

inline float special_block_router_finite_score(float score)
{
   return std::isfinite(score) ? score
                               : std::numeric_limits<float>::infinity();
}

// FLOAT/L2 handlers return squared distance. A centroid ball with radius r
// bounds every point in the represented block subtree by this value.
inline float special_block_router_l2_lower_bound(
    float query_centroid_distance_sq, float radius)
{
   if (!std::isfinite(query_centroid_distance_sq) ||
       !std::isfinite(radius) || radius < 0.0f)
      return std::numeric_limits<float>::infinity();
   const float outside = std::max(
       0.0f,
       std::sqrt(std::max(0.0f, query_centroid_distance_sq)) - radius);
   return outside * outside;
}

inline bool special_block_router_child_less(
    const SpecialBlockRouterChild &lhs,
    const SpecialBlockRouterChild &rhs)
{
   const float lhs_portal =
       special_block_router_finite_score(lhs.portal_distance);
   const float rhs_portal =
       special_block_router_finite_score(rhs.portal_distance);
   if (lhs_portal != rhs_portal)
      return lhs_portal < rhs_portal;
   const float lhs_bound =
       special_block_router_finite_score(lhs.lower_bound_distance);
   const float rhs_bound =
       special_block_router_finite_score(rhs.lower_bound_distance);
   if (lhs_bound != rhs_bound)
      return lhs_bound < rhs_bound;
   return lhs.block_id < rhs.block_id;
}

inline bool special_block_router_lower_bound_less(
    const SpecialBlockRouterChild &lhs,
    const SpecialBlockRouterChild &rhs)
{
   const float lhs_bound =
       special_block_router_finite_score(lhs.lower_bound_distance);
   const float rhs_bound =
       special_block_router_finite_score(rhs.lower_bound_distance);
   if (lhs_bound != rhs_bound)
      return lhs_bound < rhs_bound;
   return lhs.block_id < rhs.block_id;
}

template <typename Less>
inline std::vector<SpecialBlockRouterChild>
select_special_block_router_candidates(
    std::vector<SpecialBlockRouterChild> candidates,
    size_t keep_count, Less less)
{
   if (keep_count == 0 || candidates.empty())
      return {};
   const size_t keep = std::min(keep_count, candidates.size());
   if (keep < candidates.size())
   {
      auto keep_end =
          candidates.begin() + static_cast<std::ptrdiff_t>(keep);
      std::nth_element(candidates.begin(), keep_end,
                       candidates.end(), less);
      candidates.resize(keep);
   }
   std::sort(candidates.begin(), candidates.end(), less);
   return candidates;
}

inline std::vector<SpecialBlockRouterChild>
select_special_block_router_children(
    std::vector<SpecialBlockRouterChild> candidates, size_t top_r)
{
   return select_special_block_router_candidates(
       std::move(candidates), top_r,
       special_block_router_child_less);
}

// Consume only one activation batch. Unselected children remain pending so a
// later search checkpoint can reconsider them against the updated tau.
inline std::vector<SpecialBlockRouterChild>
pop_special_block_router_batch(
    std::vector<SpecialBlockRouterChild> &pending, size_t batch_size)
{
   if (batch_size == 0 || pending.empty())
      return {};
   std::sort(pending.begin(), pending.end(),
             special_block_router_child_less);
   const size_t count = std::min(batch_size, pending.size());
   std::vector<SpecialBlockRouterChild> batch(
       pending.begin(),
       pending.begin() + static_cast<std::ptrdiff_t>(count));
   pending.erase(
       pending.begin(),
       pending.begin() + static_cast<std::ptrdiff_t>(count));
   return batch;
}

inline std::vector<SpecialBlockRouterChild>
shortlist_special_block_router_children(
    std::vector<SpecialBlockRouterChild> candidates,
    size_t shortlist_size)
{
   return select_special_block_router_candidates(
       std::move(candidates), shortlist_size,
       special_block_router_lower_bound_less);
}

} // namespace ANNS

#endif // ANNS_UNG_SPECIAL_BLOCK_ROUTER_H
