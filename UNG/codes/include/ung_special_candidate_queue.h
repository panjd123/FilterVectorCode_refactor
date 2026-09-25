#ifndef ANNS_UNG_SPECIAL_CANDIDATE_QUEUE_H
#define ANNS_UNG_SPECIAL_CANDIDATE_QUEUE_H

#include "config.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ANNS
{

struct SpecialSearchCandidate
{
   IdxType id = 0;
   float distance = 0.0f;
   // 0 is the base graph; level n > 0 is materialized hierarchy layer n.
   uint8_t activation_level = 0;

   bool free() const { return activation_level != 0; }
};

inline bool special_candidate_less(const SpecialSearchCandidate &a,
                                   const SpecialSearchCandidate &b)
{
   if (a.distance != b.distance)
      return a.distance < b.distance;
   if (a.id != b.id)
      return a.id < b.id;
   return a.activation_level < b.activation_level;
}

enum class SpecialCandidateInsertResult
{
   Inserted,
   BoundRejected
};

class SpecialCandidateQueue
{
public:
   void reset(size_t capacity, size_t top_k, bool use_heap = true);
   void initialize(std::vector<SpecialSearchCandidate> candidates);
   SpecialCandidateInsertResult insert(IdxType id, float distance,
                                       uint8_t activation_level);

   bool has_unexpanded();
   bool pop_closest_unexpanded(SpecialSearchCandidate &candidate);
   bool peek_two_unexpanded(float &first, float &second);
   float kth_distance();

   void mark_expanded_ids(const std::vector<IdxType> &sorted_ids);

   size_t size() const;
   size_t last_shifted_candidates() const { return last_shifted_candidates_; }
   std::vector<SpecialSearchCandidate> sorted_results() const;

private:
   struct Slot
   {
      SpecialSearchCandidate candidate;
      bool active = true;
      bool expanded = false;
   };

   struct OrderedSlot
   {
      SpecialSearchCandidate candidate;
      bool expanded = false;
   };

   struct ResultHeapCompare
   {
      const std::vector<Slot> *slots = nullptr;
      bool operator()(uint32_t a, uint32_t b) const;
   };

   struct ExpansionHeapCompare
   {
      const std::vector<Slot> *slots = nullptr;
      bool operator()(uint32_t a, uint32_t b) const;
   };

   void add_slot(const SpecialSearchCandidate &candidate);
   void prune_expansion_heap();

   size_t capacity_ = 0;
   size_t top_k_ = 0;
   bool use_heap_ = true;
   size_t ordered_cur_unexpanded_ = 0;
   size_t last_shifted_candidates_ = 0;
   std::vector<OrderedSlot> ordered_;
   std::vector<Slot> slots_;
   std::vector<uint32_t> result_heap_;
   std::vector<uint32_t> expansion_heap_;
   std::vector<float> kth_scratch_;
   // The graph state is (point, level): the same point reached in two layers
   // must remain independently expandable because the layers own different
   // edge sets. Final vector results are deduplicated after graph search.
   std::unordered_map<uint64_t, uint32_t> active_slot_by_state_;
};

} // namespace ANNS

#endif // ANNS_UNG_SPECIAL_CANDIDATE_QUEUE_H
