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
   queue.reset(3, 2, false);
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
              retained[1].id == 8 && retained[2].id == 10,
          "the retained result beam must contain each point only once");

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
   expect(retained.size() == 1 && retained[0].id == 4,
          "three layer states must consume only one unique-point beam slot");
   expect(queue.pop_closest_unexpanded(current) && current.activation_level == 2,
          "the highest authorized layer must expand first on a distance tie");
   expect(queue.pop_closest_unexpanded(current) && current.activation_level == 1,
          "the middle state must remain independently expandable");
   expect(queue.pop_closest_unexpanded(current) && current.activation_level == 0,
          "the base state must remain independently expandable");
   expect(queue.insert(4, 1.0f, 2) == ANNS::SpecialCandidateInsertResult::BoundRejected,
          "an identical point-layer state must not be duplicated");

   queue.reset(4, 2);
   queue.initialize({{4, 1.0f, 1}, {9, 2.0f, 0}});
   expect(queue.pop_closest_unexpanded(current) && current.id == 4 &&
              current.activation_level == 1,
          "middle state must expand first");
   expect(queue.insert(4, 1.0f, 2) == ANNS::SpecialCandidateInsertResult::Inserted,
          "a higher-layer state for the same point must remain independent");
   expect(queue.pop_closest_unexpanded(current) && current.id == 4 &&
              current.activation_level == 2,
          "the higher-layer state must be independently expandable");
   retained = queue.sorted_results();
   expect(retained.size() == 2 && retained[0].id == 4 && retained[1].id == 9,
          "different graph-layer states must not occupy distinct result slots");

   queue.reset(2, 1, true);
   queue.initialize({{1, 1.0f, 2}, {1, 1.0f, 1}, {2, 2.0f, 0}});
   expect(queue.size() == 2,
          "Lsearch capacity must count unique points rather than layer states");
   expect(queue.insert(1, 1.0f, 0) == ANNS::SpecialCandidateInsertResult::Inserted &&
              queue.size() == 2,
          "adding another state for an active point must not consume beam capacity");
   expect(queue.insert(3, 0.5f, 0) == ANNS::SpecialCandidateInsertResult::Inserted &&
              queue.size() == 2,
          "a closer new point must evict one whole point, not one layer state");

   queue.reset(2, 1, true);
   queue.initialize({{1, 10.0f, 2}, {1, 10.0f, 1}, {2, 2.0f, 0}});
   expect(queue.insert(3, 1.0f, 0) == ANNS::SpecialCandidateInsertResult::Inserted,
          "a closer point must enter a full unique-point beam");
   expect(queue.pop_closest_unexpanded(current) && current.id == 3,
          "the closest active point must expand after point eviction");
   expect(queue.pop_closest_unexpanded(current) && current.id == 2,
          "evicting a point must deactivate every queued layer state for that point");

   queue.reset(2, 1, true);
   queue.initialize({{4, 1.0f, 2}, {4, 1.0f, 1}});
   queue.mark_expanded_states({{4, 1.0f, 2}});
   expect(queue.pop_closest_unexpanded(current) && current.id == 4 &&
              current.activation_level == 1,
          "pre-expansion must mark only the selected point-layer state");


   queue.reset(4, 2, true);
   queue.initialize({{1, 1.0f, 0}, {2, 10.0f, 2}, {3, 2.0f, 1}});
   retained = queue.sorted_results();
   expect(retained[0].id == 1 && retained[1].id == 3 && retained[2].id == 2,
          "retained results must remain ordered by vector distance");
   expect(queue.pop_closest_unexpanded(current) && current.id == 2 &&
              current.activation_level == 2,
          "heap expansion must protect the highest layer from lower-layer eviction pressure");
   expect(queue.pop_closest_unexpanded(current) && current.id == 3 &&
              current.activation_level == 1,
          "heap expansion must descend through materialized levels before the base graph");

   std::vector<std::vector<ANNS::SpecialSearchCandidate>> block_candidates(4);
   block_candidates[1] = {{10, 10.0f, 1}, {11, 11.0f, 1}};
   block_candidates[2] = {{20, 1.0f, 1}, {21, 2.0f, 1}};
   block_candidates[3] = {{30, 5.0f, 1}, {31, 6.0f, 1}};
   const std::vector<ANNS::IdxType> block_order{1, 2, 3};
   const std::vector<size_t> keep_per_block{0, 2, 2, 2};
   auto selected = ANNS::select_diverse_block_seed_candidates(
       block_candidates, block_order, keep_per_block, 3);
   expect(selected.size() == 3 && selected[0].id == 20 &&
              selected[1].id == 30 && selected[2].id == 10,
          "capacity sufficient for all blocks must keep one primary per block");
   selected = ANNS::select_diverse_block_seed_candidates(
       block_candidates, block_order, keep_per_block, 2);
   expect(selected.size() == 2 && selected[0].id == 20 && selected[1].id == 30,
          "capacity below block count must choose globally closest block primaries");
   selected = ANNS::select_diverse_block_seed_candidates(
       block_candidates, block_order, keep_per_block, 5);
   expect(selected.size() == 5 && selected[0].id == 20 && selected[1].id == 21 &&
              selected[2].id == 30 && selected[3].id == 31 &&
              selected[4].id == 10,
          "extra capacity must be filled by globally closest secondary seeds");

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
