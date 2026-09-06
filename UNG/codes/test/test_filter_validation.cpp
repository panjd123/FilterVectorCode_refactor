#include "ung_filter_validation.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace
{
void expect(bool condition, const char *message)
{
   if (!condition)
   {
      std::cerr << "FAILED: " << message << '\n';
      std::exit(1);
   }
}
}

int main()
{
   using ANNS::IdxType;
   using ANNS::LabelType;

   // Mapping is deliberately non-identity: result ids are in original order,
   // while point labels are stored in reordered graph order.
   const std::vector<IdxType> old_to_new = {2, 0, 1};
   const std::vector<std::vector<LabelType>> point_labels = {
       {1, 2, 4}, // original id 1
       {2, 3},    // original id 2
       {1, 2, 3}  // original id 0
   };
   const std::vector<std::vector<LabelType>> query_labels = {
       {1, 2},
       {2, 3}};
   const IdxType missing = std::numeric_limits<IdxType>::max();
   const std::vector<std::pair<IdxType, float>> results = {
       {0, 0.0f},   // valid after original-to-reordered mapping
       {2, 0.0f},   // invalid: point lacks label 1
       {missing, 0}, // missing output slot: skipped
       {2, 0.0f},   // valid
       {1, 0.0f},   // invalid: point lacks label 3
       {99, 0.0f}   // invalid: original id out of range
   };

   const auto summary = ANNS::validate_containment_results(
       static_cast<IdxType>(query_labels.size()), results.data(), 3,
       old_to_new, static_cast<IdxType>(point_labels.size()),
       [&](IdxType query_id) -> const std::vector<LabelType> & {
          return query_labels[query_id];
       },
       [&](IdxType reordered_id) -> const std::vector<LabelType> & {
          return point_labels[reordered_id];
       });
   expect(summary.checked_results == 5,
          "missing result slots must not be counted as returned results");
   expect(summary.violations == 3,
          "valid, invalid, missing, and out-of-range results must be classified exactly");

   const std::vector<IdxType> invalid_reorder = {3};
   const std::pair<IdxType, float> invalid_mapping_result = {0, 0.0f};
   const auto invalid_mapping = ANNS::validate_containment_results(
       1, &invalid_mapping_result, 1, invalid_reorder,
       static_cast<IdxType>(point_labels.size()),
       [&](IdxType query_id) -> const std::vector<LabelType> & {
          return query_labels[query_id];
       },
       [&](IdxType reordered_id) -> const std::vector<LabelType> & {
          return point_labels[reordered_id];
       });
   expect(invalid_mapping.checked_results == 1 && invalid_mapping.violations == 1,
          "out-of-range reordered ids must be violations without storage access");

   std::cout << "filter validation tests passed\n";
   return 0;
}
