#!/usr/bin/env python3
"""Focused tests for the query-free structural scale ladder."""

import unittest

from analyze_auto_layer_policy import derive_mass_ladder, next_power_of_two, partition_row


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


if __name__ == "__main__":
    unittest.main()
