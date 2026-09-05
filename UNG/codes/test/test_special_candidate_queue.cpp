#include "ung_special_candidate_queue.h"

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
   ANNS::SpecialCandidateQueue queue;
   queue.reset(3, 2);
   queue.initialize({{9, 9.0f, false}, {3, 3.0f, false},
                     {5, 5.0f, true}, {1, 1.0f, false}});

   auto retained = queue.sorted_results();
   expect(retained.size() == 3, "initialization must retain capacity candidates");
   expect(retained[0].id == 1 && retained[1].id == 3 && retained[2].id == 5,
          "initialization must retain exact top-L order");
   expect(queue.kth_distance() == 3.0f, "top-K distance must be exact");

   ANNS::SpecialSearchCandidate current;
   expect(queue.pop_closest_unexpanded(current) && current.id == 1,
          "closest retained candidate must expand first");
   expect(queue.insert(2, 2.0f, false) == ANNS::SpecialCandidateInsertResult::Inserted,
          "closer candidate must enter retained pool");
   expect(queue.pop_closest_unexpanded(current) && current.id == 2,
          "new closer candidate must become the next expansion");

   retained = queue.sorted_results();
   expect(retained[0].id == 1 && retained[1].id == 2 && retained[2].id == 3,
          "expanded candidates must remain while the old worst candidate is evicted");
   expect(queue.pop_closest_unexpanded(current) && current.id == 3,
          "active unexpanded candidate must be returned");
   expect(!queue.pop_closest_unexpanded(current),
          "evicted candidates must be skipped in the expansion heap");

   queue.reset(4, 2);
   queue.initialize({{10, 4.0f, false}, {7, 4.0f, true},
                     {8, 4.0f, true}, {8, 4.0f, false}});
   retained = queue.sorted_results();
   expect(retained.size() == 3 && retained[0].id == 7 &&
              retained[1].id == 8 && retained[1].activation_level == 1 &&
              retained[2].id == 10,
          "initial candidates must be unique by point and retain the strongest activation");

   float first = 0.0f;
   float second = 0.0f;
   expect(queue.peek_two_unexpanded(first, second) && first == 4.0f && second == 4.0f,
          "two closest active distances must be available");

   queue.mark_expanded_ids({7, 8});
   expect(queue.pop_closest_unexpanded(current) && current.id == 10,
          "preexpanded IDs must be skipped without leaving the retained result set");
   expect(!queue.has_unexpanded(), "all marked candidates must remain expanded");

   queue.reset(4, 2);
   queue.initialize({{4, 1.0f, 2}, {4, 1.0f, 1}, {4, 1.0f, 0}});
   retained = queue.sorted_results();
   expect(retained.size() == 1 && retained[0].id == 4 &&
              retained[0].activation_level == 2,
          "candidate queue must collapse duplicate states to the strongest activation");
   expect(queue.pop_closest_unexpanded(current) && current.activation_level == 2,
          "the strongest initial state must remain expandable");
   expect(queue.insert(4, 1.0f, 2) == ANNS::SpecialCandidateInsertResult::BoundRejected,
          "an equal activation must not duplicate a point");

   queue.reset(4, 2);
   queue.initialize({{4, 1.0f, 1}, {9, 2.0f, 0}});
   expect(queue.pop_closest_unexpanded(current) && current.id == 4 &&
              current.activation_level == 1,
          "middle state must expand before an upgrade");
   expect(queue.insert(4, 1.0f, 2) == ANNS::SpecialCandidateInsertResult::Inserted,
          "a higher overlay must upgrade an existing point");
   expect(queue.pop_closest_unexpanded(current) && current.id == 4 &&
              current.activation_level == 2,
          "an upgraded point must become expandable again");
   retained = queue.sorted_results();
   expect(retained.size() == 2 && retained[0].id == 4 &&
              retained[0].activation_level == 2,
          "an activation upgrade must not consume another result slot");

   queue.reset(0, 0);
   expect(queue.insert(1, 1.0f, false) ==
              ANNS::SpecialCandidateInsertResult::BoundRejected &&
              queue.sorted_results().empty(),
          "zero capacity must remain empty");

   queue.reset(3, 2, false);
   queue.initialize({{3, 3.0f, false}, {1, 1.0f, false}, {5, 5.0f, true}});
   expect(queue.pop_closest_unexpanded(current) && current.id == 1,
          "ordered mode must expand the closest candidate first");
   expect(queue.insert(2, 2.0f, false) == ANNS::SpecialCandidateInsertResult::Inserted &&
              queue.last_shifted_candidates() == 2,
          "ordered mode must report contiguous candidate movement");
   retained = queue.sorted_results();
   expect(retained[0].id == 1 && retained[1].id == 2 && retained[2].id == 3,
          "ordered mode must preserve the same retained top-L set");
   std::cout << "special candidate queue checks passed\n";
   return 0;
}
