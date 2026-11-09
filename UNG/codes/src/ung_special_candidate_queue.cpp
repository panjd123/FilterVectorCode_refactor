#include "include/ung_special_candidate_queue.h"

#include <algorithm>
#include <limits>

namespace ANNS
{

namespace
{
uint64_t candidate_state_key(IdxType id, uint8_t activation_level)
{
   return (static_cast<uint64_t>(activation_level) << 32) |
          static_cast<uint64_t>(id);
}
} // namespace

bool SpecialCandidateQueue::ResultHeapCompare::operator()(uint32_t a, uint32_t b) const
{
   return special_candidate_less((*slots)[a].candidate, (*slots)[b].candidate);
}

bool SpecialCandidateQueue::ExpansionHeapCompare::operator()(uint32_t a, uint32_t b) const
{
   const auto less = layer_first ? special_candidate_expansion_less
                                 : special_candidate_less;
   return less((*slots)[b].candidate, (*slots)[a].candidate);
}

void SpecialCandidateQueue::reset(size_t capacity, size_t top_k, bool use_heap)
{
   capacity_ = capacity;
   top_k_ = top_k;
   use_heap_ = use_heap;
   last_shifted_candidates_ = 0;
   slots_.clear();
   result_heap_.clear();
   expansion_heap_.clear();
   kth_scratch_.clear();
   active_result_by_point_.clear();
   active_slot_by_state_.clear();
   result_heap_.reserve(capacity_);
   kth_scratch_.reserve(capacity_);
}

void SpecialCandidateQueue::add_slot(const SpecialSearchCandidate &candidate,
                                     bool add_result_point)
{
   const uint32_t token = static_cast<uint32_t>(slots_.size());
   slots_.push_back(Slot{candidate, true, false});
   active_slot_by_state_[candidate_state_key(candidate.id,
                                              candidate.activation_level)] = token;

   if (add_result_point)
   {
      active_result_by_point_[candidate.id] = token;
      result_heap_.push_back(token);
      std::push_heap(result_heap_.begin(), result_heap_.end(),
                     ResultHeapCompare{&slots_});
   }

   expansion_heap_.push_back(token);
   std::push_heap(expansion_heap_.begin(), expansion_heap_.end(),
                  ExpansionHeapCompare{&slots_, use_heap_});
}

bool SpecialCandidateQueue::point_is_active(IdxType id) const
{
   return active_result_by_point_.find(id) != active_result_by_point_.end();
}

void SpecialCandidateQueue::initialize(std::vector<SpecialSearchCandidate> candidates)
{
   if (capacity_ == 0)
      return;
   std::unordered_map<uint64_t, size_t> unique_index;
   std::vector<SpecialSearchCandidate> unique_candidates;
   unique_candidates.reserve(candidates.size());
   for (const SpecialSearchCandidate &candidate : candidates)
   {
      const auto inserted = unique_index.emplace(
          candidate_state_key(candidate.id, candidate.activation_level),
          unique_candidates.size());
      if (inserted.second)
      {
         unique_candidates.push_back(candidate);
         continue;
      }
      SpecialSearchCandidate &existing = unique_candidates[inserted.first->second];
      existing.distance = std::min(existing.distance, candidate.distance);
   }
   candidates.swap(unique_candidates);
   std::sort(candidates.begin(), candidates.end(), special_candidate_less);
   std::unordered_map<IdxType, uint8_t> retained_points;
   retained_points.reserve(std::min(capacity_, candidates.size()));
   for (const SpecialSearchCandidate &candidate : candidates)
   {
      if (retained_points.find(candidate.id) != retained_points.end())
         continue;
      if (retained_points.size() >= capacity_)
         break;
      retained_points.emplace(candidate.id, 1);
   }
   slots_.reserve(std::max(slots_.capacity(), candidates.size()));
   for (const SpecialSearchCandidate &candidate : candidates)
   {
      if (retained_points.find(candidate.id) == retained_points.end())
         continue;
      add_slot(candidate, !point_is_active(candidate.id));
   }
}

SpecialCandidateInsertResult SpecialCandidateQueue::insert(IdxType id,
                                                            float distance,
                                                            uint8_t activation_level)
{
   const SpecialSearchCandidate candidate{id, distance, activation_level};
   last_shifted_candidates_ = 0;
   if (capacity_ == 0)
      return SpecialCandidateInsertResult::BoundRejected;

   const uint64_t state_key = candidate_state_key(id, activation_level);
   if (active_slot_by_state_.find(state_key) != active_slot_by_state_.end())
      return SpecialCandidateInsertResult::BoundRejected;

   if (point_is_active(id))
   {
      add_slot(candidate, false);
      return SpecialCandidateInsertResult::Inserted;
   }

   if (result_heap_.size() >= capacity_)
   {
      const uint32_t worst_token = result_heap_.front();
      if (special_candidate_less(slots_[worst_token].candidate, candidate))
         return SpecialCandidateInsertResult::BoundRejected;

      if (!use_heap_)
         for (uint32_t token : result_heap_)
            if (special_candidate_less(candidate, slots_[token].candidate))
               ++last_shifted_candidates_;

      std::pop_heap(result_heap_.begin(), result_heap_.end(), ResultHeapCompare{&slots_});
      const uint32_t evicted_token = result_heap_.back();
      const SpecialSearchCandidate &evicted = slots_[evicted_token].candidate;
      active_result_by_point_.erase(evicted.id);
      result_heap_.pop_back();
   }
   else
   {
      if (!use_heap_)
         for (uint32_t token : result_heap_)
            if (special_candidate_less(candidate, slots_[token].candidate))
               ++last_shifted_candidates_;
   }

   add_slot(candidate, true);
   return SpecialCandidateInsertResult::Inserted;
}

void SpecialCandidateQueue::prune_expansion_heap()
{
   const ExpansionHeapCompare compare{&slots_, use_heap_};
   while (!expansion_heap_.empty())
   {
      const Slot &slot = slots_[expansion_heap_.front()];
      if (slot.active && !slot.expanded && point_is_active(slot.candidate.id))
         break;
      std::pop_heap(expansion_heap_.begin(), expansion_heap_.end(), compare);
      expansion_heap_.pop_back();
   }
}

bool SpecialCandidateQueue::has_unexpanded()
{
   prune_expansion_heap();
   return !expansion_heap_.empty();
}

bool SpecialCandidateQueue::pop_closest_unexpanded(SpecialSearchCandidate &candidate)
{
   prune_expansion_heap();
   if (expansion_heap_.empty())
      return false;

   const ExpansionHeapCompare compare{&slots_, use_heap_};
   std::pop_heap(expansion_heap_.begin(), expansion_heap_.end(), compare);
   const uint32_t token = expansion_heap_.back();
   expansion_heap_.pop_back();
   slots_[token].expanded = true;
   candidate = slots_[token].candidate;
   return true;
}

bool SpecialCandidateQueue::peek_two_unexpanded(float &first, float &second)
{
   prune_expansion_heap();
   if (expansion_heap_.size() < 2)
      return false;

   const ExpansionHeapCompare compare{&slots_, use_heap_};
   std::pop_heap(expansion_heap_.begin(), expansion_heap_.end(), compare);
   const uint32_t first_token = expansion_heap_.back();
   expansion_heap_.pop_back();
   prune_expansion_heap();
   if (expansion_heap_.empty())
   {
      expansion_heap_.push_back(first_token);
      std::push_heap(expansion_heap_.begin(), expansion_heap_.end(), compare);
      return false;
   }

   first = slots_[first_token].candidate.distance;
   second = slots_[expansion_heap_.front()].candidate.distance;
   expansion_heap_.push_back(first_token);
   std::push_heap(expansion_heap_.begin(), expansion_heap_.end(), compare);
   return true;
}

float SpecialCandidateQueue::kth_distance()
{
   if (top_k_ == 0 || size() < top_k_)
      return std::numeric_limits<float>::infinity();

   kth_scratch_.clear();
   for (uint32_t token : result_heap_)
      kth_scratch_.push_back(slots_[token].candidate.distance);
   auto kth = kth_scratch_.begin() + static_cast<std::ptrdiff_t>(top_k_ - 1);
   std::nth_element(kth_scratch_.begin(), kth, kth_scratch_.end());
   return *kth;
}

void SpecialCandidateQueue::mark_expanded_ids(const std::vector<IdxType> &sorted_ids)
{
   if (sorted_ids.empty())
      return;
   for (Slot &slot : slots_)
   {
      if (std::binary_search(sorted_ids.begin(), sorted_ids.end(), slot.candidate.id))
         slot.expanded = true;
   }
}

void SpecialCandidateQueue::mark_expanded_states(
    const std::vector<SpecialSearchCandidate> &states)
{
   for (const SpecialSearchCandidate &state : states)
   {
      const auto found = active_slot_by_state_.find(
          candidate_state_key(state.id, state.activation_level));
      if (found != active_slot_by_state_.end())
         slots_[found->second].expanded = true;
   }
}

std::vector<SpecialSearchCandidate> SpecialCandidateQueue::sorted_results() const
{
   std::vector<SpecialSearchCandidate> result;
   result.reserve(result_heap_.size());
   for (uint32_t token : result_heap_)
      result.push_back(slots_[token].candidate);
   std::sort(result.begin(), result.end(), special_candidate_less);
   return result;
}

size_t SpecialCandidateQueue::size() const
{
   return result_heap_.size();
}

std::vector<SpecialSearchCandidate> select_diverse_block_seed_candidates(
    const std::vector<std::vector<SpecialSearchCandidate>> &candidates_by_block,
    const std::vector<IdxType> &block_order,
    const std::vector<size_t> &keep_per_block,
    size_t capacity)
{
   if (capacity == 0)
      return {};

   std::vector<SpecialSearchCandidate> primary_candidates;
   std::vector<SpecialSearchCandidate> extra_candidates;
   primary_candidates.reserve(block_order.size());

   for (IdxType block_id : block_order)
   {
      if (block_id >= candidates_by_block.size() ||
          block_id >= keep_per_block.size())
         continue;
      const size_t keep = keep_per_block[block_id];
      if (keep == 0 || candidates_by_block[block_id].empty())
         continue;

      std::vector<SpecialSearchCandidate> block_candidates =
          candidates_by_block[block_id];
      std::sort(block_candidates.begin(), block_candidates.end(),
                special_candidate_less);
      if (block_candidates.size() > keep)
         block_candidates.resize(keep);

      primary_candidates.push_back(block_candidates.front());
      extra_candidates.insert(extra_candidates.end(),
                              block_candidates.begin() + 1,
                              block_candidates.end());
   }

   std::sort(primary_candidates.begin(), primary_candidates.end(),
             special_candidate_less);
   if (primary_candidates.size() >= capacity)
   {
      primary_candidates.resize(capacity);
      return primary_candidates;
   }

   std::sort(extra_candidates.begin(), extra_candidates.end(),
             special_candidate_less);
   const size_t extras_to_keep =
       std::min(capacity - primary_candidates.size(), extra_candidates.size());
   primary_candidates.insert(primary_candidates.end(),
                             extra_candidates.begin(),
                             extra_candidates.begin() +
                                 static_cast<std::ptrdiff_t>(extras_to_keep));
   std::sort(primary_candidates.begin(), primary_candidates.end(),
             special_candidate_less);
   return primary_candidates;
}

} // namespace ANNS
