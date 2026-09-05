#include "ung_favor_block_search.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
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
} // namespace

int main()
{
   const float high_selectivity = ANNS::favor_exclusion_distance(0.5f, 100.0f, 2.0f, 1.0f);
   const float low_selectivity = ANNS::favor_exclusion_distance(0.1f, 100.0f, 2.0f, 1.0f);

   expect(high_selectivity > 0.0f, "exclusion distance must be positive for valid selectivity");
   expect(low_selectivity > high_selectivity, "lower selectivity must produce a larger NTD penalty");

   const float base = 12.5f;
   const float penalty = 3.25f;
   expect(std::fabs(ANNS::favor_adjusted_distance(base, true, penalty) - base) < 1e-6f,
          "TD vectors must keep the original distance");
   expect(std::fabs(ANNS::favor_adjusted_distance(base, false, penalty) - (base + penalty)) < 1e-6f,
          "NTD vectors must receive the exclusion distance penalty");

   const float clamped = ANNS::favor_exclusion_distance(0.0f, 100.0f, 2.0f, 1.0f);
   expect(std::isfinite(clamped), "zero selectivity must be clamped to a finite penalty");

   const std::vector<ANNS::FavorBlockCandidate> candidates{
       {1, 2, 120},
       {2, 5, 500},
       {3, 5, 100}};
   const std::vector<ANNS::IdxType> selected = ANNS::favor_select_top_blocks(candidates, 2);
   expect(selected.size() == 2, "max block cap must limit selected block count");
   expect(selected[0] == 3 && selected[1] == 2,
          "blocks must be ordered by score descending and point count ascending");

   std::cout << "favor block distance and selection checks passed\n";
   return 0;
}
