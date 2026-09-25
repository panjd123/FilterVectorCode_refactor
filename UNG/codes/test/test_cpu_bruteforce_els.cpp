#include "ung_cpu_bruteforce_els.h"
#include "ung_entry_group.h"
#include "ung_query_route.h"

#include <cstdlib>
#include <iostream>
#include <string>
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

std::vector<uint64_t> one_word_bits(std::initializer_list<ANNS::IdxType> ids)
{
   std::vector<uint64_t> bits(1, 0);
   for (ANNS::IdxType id : ids)
      bits[0] |= uint64_t{1} << id;
   return bits;
}

bool bit_is_set(const std::vector<uint64_t> &bits, ANNS::IdxType gid)
{
   return (bits[gid >> 6] & (uint64_t{1} << (gid & 63))) != 0;
}
} // namespace

int main()
{
   std::vector<roaring::Roaring> descendants(8);
   descendants[2].add(4);
   descendants[2].add(5);
   descendants[3].add(6);

   const std::vector<uint64_t> candidate_bits = one_word_bits({2, 3, 4, 5, 6, 7});
   const std::vector<uint64_t> selected_frontier_bits = one_word_bits({2, 3});
   const std::vector<ANNS::IdxType> frontier_ids{2, 3};

   const std::vector<ANNS::IdxType> result =
       ANNS::cpu_bruteforce_els_select_with_roaring(candidate_bits,
                                                    selected_frontier_bits,
                                                    frontier_ids,
                                                    descendants,
                                                    8);

   const std::vector<ANNS::IdxType> expected{2, 3, 7};
   expect(result == expected, "selected frontier must be kept and covered descendants removed");

   const std::vector<std::vector<ANNS::LabelType>> row_fixture{
       {}, {1}, {1, 2}, {2, 3}, {4}};
   const auto requested_rows = ANNS::build_cpu_bruteforce_els_rows(row_fixture, 4, {1, 3});
   expect(requested_rows.label_group_bits.size() == 2, "only requested label rows must be built");
   expect(bit_is_set(requested_rows.label_group_bits.at(1), 1), "label 1 must contain group 1");
   expect(bit_is_set(requested_rows.label_group_bits.at(1), 2), "label 1 must contain group 2");
   expect(bit_is_set(requested_rows.label_group_bits.at(3), 3), "label 3 must contain group 3");
   expect(requested_rows.label_group_bits.count(2) == 0, "unrequested label 2 must not be allocated");

   const auto lazy_rows = ANNS::build_cpu_bruteforce_els_rows(row_fixture, 4, {2});
   expect(bit_is_set(lazy_rows.label_group_bits.at(2), 2), "lazy label 2 row must contain group 2");
   expect(bit_is_set(lazy_rows.label_group_bits.at(2), 3), "lazy label 2 row must contain group 3");

   const std::vector<std::vector<ANNS::LabelType>> group_labels{
       {},
       {1, 2},
       {1, 2, 3},
       {1, 2, 4},
       {1, 3},
       {1, 2, 3, 4},
   };
   const std::vector<ANNS::LabelType> query_labels{1, 2};
   const std::vector<ANNS::IdxType> exact_minimal =
       ANNS::cpu_bruteforce_els_select_scalar_exact(group_labels, query_labels, 5, 0);
   expect(exact_minimal == std::vector<ANNS::IdxType>{1},
          "scalar exact ELS must remove every candidate covered by a smaller candidate");

   const std::vector<std::vector<ANNS::LabelType>> cap_group_labels{
       {},
       {1},
       {1, 2},
       {1, 3},
       {1, 4},
       {1, 5},
   };
   const std::vector<ANNS::IdxType> capped_filtered =
       ANNS::cpu_bruteforce_els_select_scalar_exact(cap_group_labels, {1}, 5, 2);
   expect(capped_filtered == std::vector<ANNS::IdxType>({1, 5}),
          "scalar ELS must stop filtering once the remaining candidate count reaches the configured cap");

   const std::vector<std::vector<ANNS::LabelType>> same_size_group_labels{
       {},
       {1, 2},
       {1, 3},
       {1, 2, 4},
   };
   const std::vector<ANNS::IdxType> same_size_result =
       ANNS::cpu_bruteforce_els_select_scalar_exact(same_size_group_labels, {1}, 3, 0);
   expect(same_size_result == std::vector<ANNS::IdxType>({1, 2}),
          "scalar ELS must compare candidates only against strictly smaller label sets");

   const std::vector<std::vector<ANNS::LabelType>> source_bucket_group_labels{
       {},
       {1},
       {1, 2},
       {1, 2, 3},
       {1, 4},
       {1, 4, 5},
   };
   const std::vector<ANNS::IdxType> source_bucket_result =
       ANNS::cpu_bruteforce_els_select_scalar_exact(source_bucket_group_labels, {1}, 5, 3);
   expect(source_bucket_result == std::vector<ANNS::IdxType>({1, 3, 5}),
          "scalar ELS must stop after smaller source buckets reduce candidates to the cap");

   const ANNS::EntryGroupStrategy optimized =
       ANNS::parse_entry_group_strategy("cpu_bruteforce_els");
   expect(optimized == ANNS::EntryGroupStrategy::OptimizedLng,
          "legacy CPU ELS name must map to optimized_lng");
   expect(std::string(ANNS::entry_group_strategy_name(optimized)) == "optimized_lng",
          "entry strategy must use its canonical public name");

   const std::string key_a = ANNS::make_entry_group_label_cache_key(
       ANNS::EntryGroupStrategy::OptimizedLng, {4, 2, 4}, false, false);
   const std::string key_b = ANNS::make_entry_group_label_cache_key(
       ANNS::EntryGroupStrategy::OptimizedLng, {4, 4, 2}, false, false);
   expect(key_a == key_b, "entry-group cache key must canonicalize equivalent query labels");

   const std::string original_key = ANNS::make_entry_group_label_cache_key(
       ANNS::EntryGroupStrategy::Original, {4, 4, 2}, false, false);
   expect(key_a != original_key, "entry-group cache key must keep strategies separate");

   expect(!ANNS::should_stop_special_block_search(false, 10, 10, 20.0f, 21.0f, 10.0f, 100, 10),
          "special block early-stop must be disabled when the runtime flag is off");
   expect(ANNS::should_stop_special_block_search(true, 10, 10, 20.0f, 21.0f, 10.0f, 100, 10),
          "special block early-stop must stop when the next candidate is worse than the current result bound");
   expect(!ANNS::should_stop_special_block_search(true, 10, 10, 9.0f, 21.0f, 10.0f, 100, 10),
          "special block early-stop must continue while an unexpanded candidate can improve the result");
   expect(!ANNS::should_stop_special_block_search(true, 10, 10, 20.0f, 9.0f, 10.0f, 100, 10),
          "special block early-stop must require two unexpanded candidates past the result bound");

   unsetenv("UNG_SPECIAL_LIGHT_STATS");
   ANNS::SearchRuntimeConfig default_runtime = ANNS::make_search_runtime_config(
       1, 10, 1, "amazon", 10, false, false, false, false, false,
       0, 0, 0, 0, 0, 0, 0);
   expect(default_runtime.special_light_stats,
          "special block light stats must be enabled by default");
   setenv("UNG_SPECIAL_LIGHT_STATS", "1", 1);
   ANNS::SearchRuntimeConfig light_runtime = ANNS::make_search_runtime_config(
       1, 10, 1, "amazon", 10, false, false, false, false, false,
       0, 0, 0, 0, 0, 0, 0);
   expect(light_runtime.special_light_stats,
          "UNG_SPECIAL_LIGHT_STATS=1 must enable light stats mode");
   unsetenv("UNG_SPECIAL_LIGHT_STATS");

   std::cout << "cpu brute-force ELS roaring selection checks passed\n";
   return 0;
}
