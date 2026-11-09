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
      // Backend-independent breakdown used by query/summary CSV output.
      double entry_point_setup_time_ms = 0.0;
      double get_min_super_sets_time_ms = 0.0;
      size_t num_distance_calcs = 0;

      size_t num_nodes_visited = 0;
      size_t regular_edges_scanned = 0;
      size_t free_edges_scanned = 0;
      size_t query_length = 0;
      long long trie_nodes_traversed = 0;



      size_t candidate_set_size = 0;
      size_t successful_checks = 0;
      float shortcut_hit_ratio = 0.0f;
      long long redundant_upward_steps = 0;
      size_t recursive_calls = 0;
      size_t pruning_events = 0;
      float pruning_efficiency = 0.0f;

      size_t num_entry_points = 0;

      size_t special_trie_pivot_postings = 0;
      size_t special_trie_matching_pivots = 0;
      size_t special_trie_upward_nodes_visited = 0;
      size_t special_trie_downward_nodes_visited = 0;
      size_t special_trie_branches_pruned = 0;
      size_t special_trie_terminal_candidates = 0;
      size_t special_trie_terminal_descendants_pruned = 0;
      size_t special_trie_final_entries = 0;
      size_t special_trie_final_block_entries = 0;
      double special_trie_time_ms = 0.0;

      bool special_search_enabled = false;
      bool special_free_use_regular = false;
      double special_cover_time_ms = 0.0;
      double special_entry_time_ms = 0.0;
      double special_special_edges_time_ms = 0.0;
      double special_regular_edges_time_ms = 0.0;
      double special_result_time_ms = 0.0;

      size_t special_entry_free_points = 0;
      size_t special_entry_regular_points = 0;
      size_t special_group_entry_free_points = 0;
      size_t special_group_entry_regular_points = 0;
      size_t special_free_block_count = 0;
      size_t special_free_block_frontier_count = 0;
      // Multi-level diagnostics: level 1 is the middle (T1) overlay and
      // level 2 is the upper (T2) overlay.
      size_t special_query_middle_block_count = 0;
      size_t special_query_upper_block_count = 0;
      size_t special_query_upper_covered_points = 0;
      bool special_query_upper_enabled = true;
      size_t special_block_seed_points = 0;
      size_t special_block_seed_points_retained = 0;
      size_t special_block_seed_points_preexpanded = 0;
      bool special_block_router_enabled = false;
      bool special_block_router_lower_bound_enabled = false;
      size_t special_block_router_blocks_routed = 0;
      size_t special_block_router_children_scored = 0;
      size_t special_block_router_lower_bound_calcs = 0;
      size_t special_block_router_children_shortlisted = 0;
      size_t special_block_router_portal_children_scored = 0;
      size_t special_block_router_children_selected = 0;
      size_t special_block_router_portals_scored = 0;
      size_t special_block_router_portal_activations = 0;
      size_t special_block_router_deferred_portals_stored = 0;
      size_t special_block_router_deferred_portals_replayed = 0;
      size_t special_block_router_inter_edges_skipped = 0;
      double special_block_router_lower_bound_time_ms = 0.0;
      double special_block_router_portal_score_time_ms = 0.0;
      size_t special_preexpand_edges_scanned = 0;
      size_t special_preexpand_edges_accepted = 0;
      double special_preexpand_time_ms = 0.0;
      size_t special_entry_blocks = 0;
      size_t special_retained_entry_blocks = 0;
      size_t special_middle_blocks_searched = 0;
      size_t special_upper_blocks_searched = 0;
      size_t special_regular_nodes_expanded = 0;
      size_t special_free_nodes_expanded = 0;
      size_t special_middle_nodes_expanded = 0;
      size_t special_upper_nodes_expanded = 0;
      size_t special_free_node_cap_skipped = 0;
      size_t special_free_edges_cap_skipped = 0;
      size_t special_blocks_searched = 0;
      size_t special_regular_candidates_inserted = 0;
      size_t special_free_candidates_inserted = 0;
      size_t special_regular_edges_scanned = 0;
      size_t special_edges_scanned = 0;
      size_t special_intra_edges_scanned = 0;
      size_t special_inter_edges_scanned = 0;
      size_t special_middle_edges_scanned = 0;
      size_t special_upper_edges_scanned = 0;
      size_t special_inter_edges_coverage_rejected = 0;
      // Number of inter-block edges skipped by the optional per-free-node
      // inter-edge scan cap.  Intra edges have a separate existing cap.
      size_t special_free_inter_edges_cap_skipped = 0;
      size_t special_heavy_edges_scanned = 0;
      size_t special_heavy_edges_accepted = 0;
      bool special_heavy_edges_enabled = false;
      size_t special_regular_edges_accepted = 0;
      size_t special_edges_accepted = 0;
      size_t special_regular_distance_calcs = 0;
      size_t special_free_distance_calcs = 0;
      size_t special_queue_insert_attempts = 0;
      size_t special_queue_bound_rejections = 0;
      size_t special_queue_insertions = 0;
      size_t special_queue_shifted_candidates = 0;
      size_t special_cross_layer_states_inserted = 0;
      size_t special_distance_cache_hits = 0;
      bool favor_block_search_enabled = false;
      size_t favor_block_count = 0;
      size_t favor_block_points = 0;
      size_t favor_target_points = 0;
      size_t favor_td_candidates_inserted = 0;
      size_t favor_ntd_candidates_inserted = 0;
      size_t favor_td_results = 0;
      size_t favor_blocks_expanded = 0;
      size_t favor_block_edge_scans_skipped = 0;
      float favor_selectivity = 0.0f;
      float favor_exclusion_distance = 0.0f;
   };
}

#endif // UNG_QUERY_STATS_H
