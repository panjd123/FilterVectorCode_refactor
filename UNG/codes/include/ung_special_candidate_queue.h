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
   // The same vector may represent several graph-layer states. Prefer its
   // coarsest state on an exact distance tie so promotion cannot lose to the
   // lower-layer copy merely because both share one distance.
   if (a.activation_level != b.activation_level)
      return a.activation_level > b.activation_level;
   return a.id < b.id;
}

// Expansion is deliberately layer-first: authorized coarse states provide
// navigation, while special_candidate_less remains distance-first for the
// retained result set and final top-k correctness.
inline bool special_candidate_expansion_less(const SpecialSearchCandidate &a,
                                             const SpecialSearchCandidate &b)
{
   if (a.activation_level != b.activation_level)
      return a.activation_level > b.activation_level;
   return special_candidate_less(a, b);
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
   void mark_expanded_states(
       const std::vector<SpecialSearchCandidate> &states);

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

   struct ResultHeapCompare
   {
      const std::vector<Slot> *slots = nullptr;
      bool operator()(uint32_t a, uint32_t b) const;
   };

   struct ExpansionHeapCompare
   {
      const std::vector<Slot> *slots = nullptr;
      bool layer_first = true;
      bool operator()(uint32_t a, uint32_t b) const;
   };

   void add_slot(const SpecialSearchCandidate &candidate, bool add_result_point);
   void prune_expansion_heap();
   bool point_is_active(IdxType id) const;

   size_t capacity_ = 0;
   size_t top_k_ = 0;
   bool use_heap_ = true;
   size_t last_shifted_candidates_ = 0;
   std::vector<Slot> slots_;
   // One token per distinct point. This is the only structure bounded by
   // capacity_, so multiple graph-layer states never consume extra Lsearch
   // result-beam slots.
   std::vector<uint32_t> result_heap_;
   // Independent (point, level) traversal states. A state is expandable only
   // while its point remains active in the unique-point result beam.
   std::vector<uint32_t> expansion_heap_;
   std::vector<float> kth_scratch_;
   std::unordered_map<IdxType, uint32_t> active_result_by_point_;
   // Graph states are deduplicated separately from result points because the
   // same vector may need to expand a different edge set in each layer.
   std::unordered_map<uint64_t, uint32_t> active_slot_by_state_;
};

// Select a bounded seed frontier without letting block iteration order decide
// which blocks survive. The closest seed from every non-empty block is the
// primary candidate. Primaries compete globally only when capacity is smaller
// than the number of blocks; otherwise every primary is retained and remaining
// capacity is filled by globally closest extra seeds.
std::vector<SpecialSearchCandidate> select_diverse_block_seed_candidates(
    const std::vector<std::vector<SpecialSearchCandidate>> &candidates_by_block,
    const std::vector<IdxType> &block_order,
    const std::vector<size_t> &keep_per_block,
    size_t capacity);

} // namespace ANNS

#endif // ANNS_UNG_SPECIAL_CANDIDATE_QUEUE_H
