#!/usr/bin/env python3
"""Focused tests for the query-free structural scale ladder."""

import unittest
from pathlib import Path

from analyze_auto_layer_policy import derive_mass_ladder, next_power_of_two, partition_row
from derive_static_hierarchy import derive_plan
from generate_auto_policy_cross_dataset_configs import (
    AUTOMATIC_ROLE,
    MANUAL_ROLE,
    ROUTING_POLICY,
    UNROUTED_ABLATION_ROLE,
    candidate_count_metadata,
    declared_candidate_plans,
    make_search_methods,
)
import gpu_isolation
from summarize_heldout_oracle import first_crossings, geometric_mean


class AutoLayerPolicyTest(unittest.TestCase):
    def setUp(self):
        # root -> two terminal children with masses 20 and 5.
        self.nodes = [
            (0, 0, 0, 2, 0, 0),
            (1, 0, 2, 0, 1, 0),
            (2, 0, 2, 0, 2, 0),
        ]
        self.children = [1, 2]
        self.points = [0, 20, 5]

    def test_next_power_of_two(self):
        self.assertEqual(next_power_of_two(6400), 8192)
        self.assertEqual(next_power_of_two(16), 16)

    def test_partition_uses_strict_threshold(self):
        row = partition_row(self.nodes, self.children, self.points, 25, 20)
        self.assertEqual(row["block_count"], 0)
        self.assertEqual(row["root_residual_points"], 25)

    def test_layer_count_stops_at_first_empty_partition(self):
        policy = derive_mass_ladder(
            self.nodes, self.children, self.points, 25,
            max_degree=4, build_width=2, cross_edges=2, max_levels=4,
        )
        self.assertEqual(policy["thresholds"], [8, 16])
        self.assertEqual(policy["layer_count"], 2)
        self.assertEqual(policy["decisions"][-1]["threshold"], 32)
        self.assertEqual(policy["decisions"][-1]["outcome"], "stop")

    def test_single_candidate_scale_falls_back_to_plain(self):
        policy = derive_mass_ladder(
            self.nodes, self.children, self.points, 25,
            max_degree=4, build_width=3, cross_edges=2, max_levels=4,
        )
        self.assertEqual(policy["candidate_thresholds"], [16])
        self.assertEqual(policy["candidate_layer_count"], 1)
        self.assertEqual(policy["thresholds"], [])
        self.assertEqual(policy["layer_count"], 0)
        self.assertEqual(
            policy["decisions"][-1]["outcome"],
            "discard_singleton_hierarchy",
        )


class DegreeRatioHeldoutProtocolTest(unittest.TestCase):
    def test_drh_outputs_are_dataset_specific_without_query_inputs(self):
        self.assertEqual(
            [(row["min_points"], row["topology"])
             for row in derive_plan(108077, 64, 4)],
            [(256, "lng"), (4096, "trie")],
        )
        self.assertEqual(
            [(row["min_points"], row["topology"])
             for row in derive_plan(288065, 64, 4)],
            [(512, "lng"), (8192, "trie")],
        )
        self.assertEqual(
            [(row["min_points"], row["topology"])
             for row in derive_plan(758935, 64, 4)],
            [(1024, "lng"), (16384, "trie")],
        )

    def test_manual_grid_contains_unique_depth_and_topology_controls(self):
        automatic = derive_plan(602453, 64, 4)
        cases = declared_candidate_plans(automatic, 602453, 16)
        self.assertEqual(len(cases), 36)
        self.assertEqual(
            {len(case["hierarchy_layers"]) for case in cases},
            {1, 2, 3},
        )
        self.assertEqual(
            sum(case["selection_role"] ==
                "predeclared_degree_ratio_hierarchy_v1" for case in cases),
            1,
        )
        signatures = {
            tuple((layer["min_points"], layer["topology"])
                  for layer in case["hierarchy_layers"])
            for case in cases
        }
        self.assertEqual(len(signatures), len(cases))
        self.assertEqual(
            candidate_count_metadata(cases),
            {
                "frozen_hierarchy_candidates": 36,
                "manual_alternatives": 35,
            },
        )

    def test_heldout_builds_declare_gpu_isolation_policy(self):
        # Full config generation needs dataset files; the policy itself must
        # remain query-independent and common to all held-out datasets.
        self.assertNotIn("query", gpu_isolation.DEFAULT_POLICY)
        self.assertEqual(gpu_isolation.DEFAULT_POLICY["device"], 0)

    def test_automatic_and_manual_searches_share_exact_authorization_gate(self):
        cases = [
            {
                "name": "auto",
                "hierarchy_layers": [{"min_points": 256, "topology": "lng"}],
                "selection_role": AUTOMATIC_ROLE,
            },
            {
                "name": "manual",
                "hierarchy_layers": [{"min_points": 512, "topology": "trie"}],
                "selection_role": MANUAL_ROLE,
            },
        ]
        methods = make_search_methods(Path("/index"), Path("/run"), cases)
        automatic = [m for m in methods if m.get("selection_role") == AUTOMATIC_ROLE]
        manual = [m for m in methods if m.get("selection_role") == MANUAL_ROLE]
        unrouted = [m for m in methods
                    if m.get("selection_role") == UNROUTED_ABLATION_ROLE]
        self.assertEqual(len(automatic), 1)
        self.assertEqual(len(manual), 1)
        self.assertEqual(len(unrouted), 1)
        for method in automatic + manual:
            self.assertEqual(method["routing_policy"], ROUTING_POLICY)
            self.assertEqual(
                method["env"]["UNG_SPECIAL_REQUIRE_UPPER_AUTHORIZATION"], "1")
        self.assertNotIn("routing_policy", unrouted[0])
        self.assertNotIn(
            "UNG_SPECIAL_REQUIRE_UPPER_AUTHORIZATION", unrouted[0]["env"])

    def test_first_crossing_uses_smallest_l_and_all_repeat_recall(self):
        rows = [
            {"method": "a", "workload": "w", "lsearch": 400,
             "recall_min": 0.89, "batch_ms_warm_median": 4.0},
            {"method": "a", "workload": "w", "lsearch": 500,
             "recall_min": 0.91, "batch_ms_warm_median": 5.0},
            {"method": "a", "workload": "w", "lsearch": 700,
             "recall_min": 0.93, "batch_ms_warm_median": 3.0},
        ]
        selected = first_crossings(rows, {"w": 0.90})
        self.assertEqual(selected[("a", "w")]["lsearch"], 500)

    def test_geometric_mean_is_not_arithmetic_mean(self):
        self.assertAlmostEqual(geometric_mean([1.0, 4.0]), 2.0)
        with self.assertRaises(ValueError):
            geometric_mean([1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
