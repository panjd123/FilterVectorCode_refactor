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
   return special_candidate_less((*slots)[b].candidate, (*slots)[a].candidate);
}

void SpecialCandidateQueue::reset(size_t capacity, size_t top_k, bool use_heap)
{
   capacity_ = capacity;
   top_k_ = top_k;
   use_heap_ = use_heap;
   ordered_cur_unexpanded_ = 0;
   last_shifted_candidates_ = 0;
   ordered_.clear();
   slots_.clear();
   result_heap_.clear();
   expansion_heap_.clear();
   kth_scratch_.clear();
   active_slot_by_state_.clear();
   result_heap_.reserve(capacity_);
   kth_scratch_.reserve(capacity_);
   ordered_.reserve(capacity_ + 1);
}

void SpecialCandidateQueue::add_slot(const SpecialSearchCandidate &candidate)
{
   const uint32_t token = static_cast<uint32_t>(slots_.size());
   slots_.push_back(Slot{candidate, true, false});
   active_slot_by_state_[candidate_state_key(candidate.id,
                                              candidate.activation_level)] = token;

   result_heap_.push_back(token);
   std::push_heap(result_heap_.begin(), result_heap_.end(), ResultHeapCompare{&slots_});

   expansion_heap_.push_back(token);
   std::push_heap(expansion_heap_.begin(), expansion_heap_.end(), ExpansionHeapCompare{&slots_});
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
   if (candidates.size() > capacity_)
   {
      auto keep_end = candidates.begin() + static_cast<std::ptrdiff_t>(capacity_);
      std::nth_element(candidates.begin(), keep_end, candidates.end(), special_candidate_less);
      candidates.resize(capacity_);
   }

   if (!use_heap_)
   {
      std::sort(candidates.begin(), candidates.end(), special_candidate_less);
      ordered_.reserve(std::max(ordered_.capacity(), candidates.size() + 1));
      for (const SpecialSearchCandidate &candidate : candidates)
         ordered_.push_back(OrderedSlot{candidate, false});
      return;
   }

   slots_.reserve(std::max(slots_.capacity(), candidates.size()));
   for (const SpecialSearchCandidate &candidate : candidates)
      add_slot(candidate);
}

SpecialCandidateInsertResult SpecialCandidateQueue::insert(IdxType id,
                                                            float distance,
                                                            uint8_t activation_level)
{
   const SpecialSearchCandidate candidate{id, distance, activation_level};
   last_shifted_candidates_ = 0;
   if (capacity_ == 0)
      return SpecialCandidateInsertResult::BoundRejected;

   if (!use_heap_)
   {
      for (size_t index = 0; index < ordered_.size(); ++index)
      {
         OrderedSlot &slot = ordered_[index];
         if (slot.candidate.id != id ||
             slot.candidate.activation_level != activation_level)
            continue;
         return SpecialCandidateInsertResult::BoundRejected;
      }
      if (ordered_.size() >= capacity_ &&
          special_candidate_less(ordered_.back().candidate, candidate))
         return SpecialCandidateInsertResult::BoundRejected;

      size_t lo = 0;
      size_t hi = ordered_.size();
      while (lo < hi)
      {
         const size_t mid = (lo + hi) >> 1;
         if (special_candidate_less(candidate, ordered_[mid].candidate))
            hi = mid;
         else
            lo = mid + 1;
      }
      last_shifted_candidates_ = ordered_.size() - lo;
      ordered_.insert(ordered_.begin() + static_cast<std::ptrdiff_t>(lo),
                      OrderedSlot{candidate, false});
      if (ordered_.size() > capacity_)
         ordered_.pop_back();
      if (lo < ordered_cur_unexpanded_)
         ordered_cur_unexpanded_ = lo;
      return SpecialCandidateInsertResult::Inserted;
   }

   const uint64_t state_key = candidate_state_key(id, activation_level);
   if (active_slot_by_state_.find(state_key) != active_slot_by_state_.end())
      return SpecialCandidateInsertResult::BoundRejected;

   if (result_heap_.size() >= capacity_)
   {
      const uint32_t worst_token = result_heap_.front();
      if (special_candidate_less(slots_[worst_token].candidate, candidate))
         return SpecialCandidateInsertResult::BoundRejected;

      std::pop_heap(result_heap_.begin(), result_heap_.end(), ResultHeapCompare{&slots_});
      const uint32_t evicted_token = result_heap_.back();
      slots_[evicted_token].active = false;
      const SpecialSearchCandidate &evicted = slots_[evicted_token].candidate;
      const auto evicted_it = active_slot_by_state_.find(
          candidate_state_key(evicted.id, evicted.activation_level));
      if (evicted_it != active_slot_by_state_.end() &&
          evicted_it->second == evicted_token)
         active_slot_by_state_.erase(evicted_it);
      result_heap_.pop_back();
   }

   add_slot(candidate);
   return SpecialCandidateInsertResult::Inserted;
}

void SpecialCandidateQueue::prune_expansion_heap()
{
   const ExpansionHeapCompare compare{&slots_};
   while (!expansion_heap_.empty())
   {
      const Slot &slot = slots_[expansion_heap_.front()];
      if (slot.active && !slot.expanded)
         break;
      std::pop_heap(expansion_heap_.begin(), expansion_heap_.end(), compare);
      expansion_heap_.pop_back();
   }
}

bool SpecialCandidateQueue::has_unexpanded()
{
   if (!use_heap_)
   {
      while (ordered_cur_unexpanded_ < ordered_.size() &&
             ordered_[ordered_cur_unexpanded_].expanded)
         ++ordered_cur_unexpanded_;
      return ordered_cur_unexpanded_ < ordered_.size();
   }
   prune_expansion_heap();
   return !expansion_heap_.empty();
}

bool SpecialCandidateQueue::pop_closest_unexpanded(SpecialSearchCandidate &candidate)
{
   if (!use_heap_)
   {
      if (!has_unexpanded())
         return false;
      OrderedSlot &slot = ordered_[ordered_cur_unexpanded_];
      slot.expanded = true;
      candidate = slot.candidate;
      ++ordered_cur_unexpanded_;
      return true;
   }
   prune_expansion_heap();
   if (expansion_heap_.empty())
      return false;

   const ExpansionHeapCompare compare{&slots_};
   std::pop_heap(expansion_heap_.begin(), expansion_heap_.end(), compare);
   const uint32_t token = expansion_heap_.back();
   expansion_heap_.pop_back();
   slots_[token].expanded = true;
   candidate = slots_[token].candidate;
   return true;
}

bool SpecialCandidateQueue::peek_two_unexpanded(float &first, float &second)
{
   if (!use_heap_)
   {
      if (!has_unexpanded() || ordered_cur_unexpanded_ + 1 >= ordered_.size())
         return false;
      first = ordered_[ordered_cur_unexpanded_].candidate.distance;
      second = ordered_[ordered_cur_unexpanded_ + 1].candidate.distance;
      return true;
   }
   prune_expansion_heap();
   if (expansion_heap_.size() < 2)
      return false;

   const ExpansionHeapCompare compare{&slots_};
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

   if (!use_heap_)
      return ordered_[top_k_ - 1].candidate.distance;

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
   if (!use_heap_)
   {
      for (OrderedSlot &slot : ordered_)
      {
         if (std::binary_search(sorted_ids.begin(), sorted_ids.end(), slot.candidate.id))
            slot.expanded = true;
      }
      has_unexpanded();
      return;
   }
   for (uint32_t token : result_heap_)
   {
      Slot &slot = slots_[token];
      if (std::binary_search(sorted_ids.begin(), sorted_ids.end(), slot.candidate.id))
         slot.expanded = true;
   }
}

std::vector<SpecialSearchCandidate> SpecialCandidateQueue::sorted_results() const
{
   if (!use_heap_)
   {
      std::vector<SpecialSearchCandidate> result;
      result.reserve(ordered_.size());
      for (const OrderedSlot &slot : ordered_)
         result.push_back(slot.candidate);
      return result;
   }
   std::vector<SpecialSearchCandidate> result;
   result.reserve(result_heap_.size());
   for (uint32_t token : result_heap_)
      result.push_back(slots_[token].candidate);
   std::sort(result.begin(), result.end(), special_candidate_less);
   return result;
}

size_t SpecialCandidateQueue::size() const
{
   return use_heap_ ? result_heap_.size() : ordered_.size();
}

} // namespace ANNS
