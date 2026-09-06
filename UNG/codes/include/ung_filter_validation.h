#ifndef UNG_FILTER_VALIDATION_H
#define UNG_FILTER_VALIDATION_H

#include "config.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace ANNS
{
   struct ContainmentResultValidation
   {
      size_t checked_results = 0;
      size_t violations = 0;
   };

   // Validate filtered-search outputs without depending on UniNavGraph's
   // storage implementation. Both label ranges must be sorted because the
   // production containment index uses sorted label sets. Missing result ids
   // are unfilled K slots and therefore are not counted as returned results.
   template <typename QueryLabelsFor, typename PointLabelsFor>
   ContainmentResultValidation validate_containment_results(
       IdxType num_queries,
       const std::pair<IdxType, float> *results,
       IdxType K,
       const std::vector<IdxType> &old_to_new_ids,
       IdxType num_points,
       QueryLabelsFor &&query_labels_for,
       PointLabelsFor &&point_labels_for)
   {
      ContainmentResultValidation summary;
      const IdxType missing = std::numeric_limits<IdxType>::max();
      for (IdxType query_id = 0; query_id < num_queries; ++query_id)
      {
         const auto &query_labels = query_labels_for(query_id);
         for (IdxType rank = 0; rank < K; ++rank)
         {
            const IdxType original_id =
                results[static_cast<size_t>(query_id) * K + rank].first;
            if (original_id == missing)
               continue;

            ++summary.checked_results;
            if (original_id >= old_to_new_ids.size())
            {
               ++summary.violations;
               continue;
            }
            const IdxType reordered_id = old_to_new_ids[original_id];
            if (reordered_id >= num_points)
            {
               ++summary.violations;
               continue;
            }

            const auto &point_labels = point_labels_for(reordered_id);
            if (!std::includes(point_labels.begin(), point_labels.end(),
                               query_labels.begin(), query_labels.end()))
               ++summary.violations;
         }
      }
      return summary;
   }
}

#endif // UNG_FILTER_VALIDATION_H
