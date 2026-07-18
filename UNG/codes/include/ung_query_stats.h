#ifndef UNG_QUERY_STATS_H
#define UNG_QUERY_STATS_H

#include <cstddef>

namespace ANNS
{
   struct QueryStats
   {
      float recall = 0.0f;
      double time_ms = 0.0;
      double search_time_ms = 0.0;
      double core_search_time_ms = 0.0;
      double descendants_merge_time_ms = 0.0;
      double coverage_merge_time_ms = 0.0;
      double get_min_super_sets_time_ms = 0.0;
      double idea1_flag_time_ms = 0.0;
      double idea2_flag_time_ms = 0.0;
      double idea1_selector_pred_time_ms = 0.0;
      double idea2_selector_pred_time_ms = 0.0;
      double bitmap_time_ms = 0.0;
      size_t num_distance_calcs = 0;
      int acorn_efs_used = 0;

      size_t num_nodes_visited = 0;
      size_t query_length = 0;
      long long trie_nodes_traversed = 0;

      bool is_idea1_used = false;
      int is_idea2_used = 0; // 0: UNG, 1: ACORN-gamma, 2: ACORN-gamma-improved

      size_t trie_total_nodes = 0;
      size_t trie_label_cardinality = 0;
      float trie_avg_path_length = 0.0f;
      float trie_avg_branching_factor = 0.0f;

      size_t candidate_set_size = 0;
      size_t successful_checks = 0;
      float shortcut_hit_ratio = 0.0f;
      long long redundant_upward_steps = 0;
      size_t recursive_calls = 0;
      size_t pruning_events = 0;
      float pruning_efficiency = 0.0f;

      size_t num_entry_points = 0;
      size_t num_lng_descendants = 0;
      size_t entry_group_matched_points = 0;
      float entry_group_total_coverage = 0.0f;

      bool special_search_enabled = false;
      bool special_free_use_regular = false;
      double special_cover_time_ms = 0.0;
      double special_entry_time_ms = 0.0;
      double special_special_edges_time_ms = 0.0;
      double special_regular_edges_time_ms = 0.0;
      double special_result_time_ms = 0.0;
      size_t special_query_matched_points = 0;
      size_t special_query_points = 0;
      size_t special_query_trivial_points = 0;
      size_t special_query_nontrivial_points = 0;
      size_t special_query_block_count = 0;
      size_t special_query_trivial_block_count = 0;
      size_t special_query_nontrivial_block_count = 0;
      float special_query_ratio = 0.0f;
      float special_query_nontrivial_ratio = 0.0f;

      size_t special_entry_free_points = 0;
      size_t special_entry_regular_points = 0;
      size_t special_regular_nodes_expanded = 0;
      size_t special_free_nodes_expanded = 0;
      size_t special_regular_candidates_inserted = 0;
      size_t special_free_candidates_inserted = 0;
      size_t special_free_upgrades = 0;
      size_t special_regular_edges_scanned = 0;
      size_t special_edges_scanned = 0;
      size_t special_intra_edges_scanned = 0;
      size_t special_inter_edges_scanned = 0;
      size_t special_heavy_edges_scanned = 0;
      size_t special_heavy_edges_accepted = 0;
      bool special_heavy_edges_enabled = false;
      size_t special_regular_edges_accepted = 0;
      size_t special_edges_accepted = 0;
      size_t special_regular_distance_calcs = 0;
      size_t special_free_distance_calcs = 0;
   };
}

#endif // UNG_QUERY_STATS_H
