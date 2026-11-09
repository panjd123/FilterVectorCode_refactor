#include "ung_special_block_router.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
#include <vector>

using namespace ANNS;

int main()
{
   const std::vector<SpecialBlockRouterChild> candidates = {
       {7, 3.0f}, {2, 1.0f}, {5, 1.0f}, {9, 2.0f}};

   const auto top_two = select_special_block_router_children(candidates, 2);
   assert(top_two.size() == 2);
   assert(top_two[0].block_id == 2);
   assert(top_two[1].block_id == 5);

   const auto all = select_special_block_router_children(candidates, 99);
   assert(all.size() == candidates.size());
   assert(all[0].block_id == 2);
   assert(all[1].block_id == 5);
   assert(all[2].block_id == 9);
   assert(all[3].block_id == 7);

   assert(select_special_block_router_children(candidates, 0).empty());
   assert(select_special_block_router_children({}, 4).empty());

   assert(std::fabs(special_block_router_l2_lower_bound(25.0f, 2.0f) -
                    9.0f) < 1.0e-6f);
   assert(special_block_router_l2_lower_bound(1.0f, 2.0f) == 0.0f);
   assert(std::isinf(special_block_router_l2_lower_bound(
       std::numeric_limits<float>::quiet_NaN(), 1.0f)));

   std::vector<SpecialBlockRouterChild> pending = {
       {10, 4.0f, 0.1f},
       {11, 1.0f, 0.2f},
       {12, 3.0f, 0.3f},
       {13, 2.0f, 0.4f},
       {14, 5.0f, 0.5f}};
   const auto first_batch = pop_special_block_router_batch(pending, 2);
   assert(first_batch.size() == 2);
   assert(first_batch[0].block_id == 11);
   assert(first_batch[1].block_id == 13);
   assert(pending.size() == 3);
   const auto second_batch = pop_special_block_router_batch(pending, 2);
   assert(second_batch.size() == 2);
   assert(second_batch[0].block_id == 12);
   assert(second_batch[1].block_id == 10);
   assert(pending.size() == 1);
   assert(pending.front().block_id == 14);

   std::vector<SpecialBlockRouterChild> nan_scores = {
       {1, std::numeric_limits<float>::quiet_NaN(), 0.0f},
       {2, 1.0f, 0.0f}};
   assert(pop_special_block_router_batch(nan_scores, 1).front().block_id == 2);

   std::cout << "special block router tests passed\n";
   return 0;
}
