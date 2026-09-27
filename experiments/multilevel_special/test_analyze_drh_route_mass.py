#!/usr/bin/env python3

from __future__ import annotations

import unittest
import csv
import json
import tempfile
from pathlib import Path

import analyze_drh_route_mass as target


class AnalyzeDrhRouteMassTest(unittest.TestCase):
    def test_uses_only_highest_authorized_layer(self) -> None:
        blocks = [
            {"level": 1, "point_count": 10, "root_labels": (1, 2)},
            {"level": 2, "point_count": 30, "root_labels": (1, 3)},
            {"level": 2, "point_count": 40, "root_labels": (1, 4)},
        ]
        masses = target.route_masses(blocks, [(1,), (2, 3)])
        self.assertEqual(
            masses,
            [
                {"level": 2, "block_count": 2, "direct_points": 70},
                {"level": 0, "block_count": 0, "direct_points": 0},
            ],
        )

    def test_summary_uses_nearest_rank_percentiles(self) -> None:
        masses = [
            {"level": 1, "block_count": 1, "direct_points": value}
            for value in [0, 10, 20, 30]
        ]
        summary = target.summarize(masses, 20)
        self.assertEqual(summary["mass_gate_enabled_count"], 2)
        self.assertEqual(summary["direct_mass_p50"], 10)
        self.assertEqual(summary["direct_mass_p95"], 30)

    def test_config_requires_unique_method(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({"methods": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected one method"):
                target.analyze_config(path, "missing", 16)

    def test_query_reader_accepts_comma_and_space_delimiters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "queries.txt"
            path.write_text("1,2,3\n4 5\n\n", encoding="utf-8")
            self.assertEqual(
                target.read_queries(path),
                [(1, 2, 3), (4, 5), ()],
            )

    def test_duplicate_labels_match_cpp_includes_semantics(self) -> None:
        self.assertTrue(target.sorted_includes((1, 1, 2), (1, 1)))
        self.assertFalse(target.sorted_includes((1, 2), (1, 1)))

    def test_coverage_profile_checks_labels_and_eligible_mass(self) -> None:
        masses = [
            {"level": 1, "block_count": 1, "direct_points": 7},
            {"level": 0, "block_count": 0, "direct_points": 0},
        ]
        queries = [(1,), ()]
        profile = [((1,), 7), ((), 10)]
        summary = target.summarize(masses, 5, queries, profile)
        self.assertEqual(summary["eligible_mass_invariant"], "checked_and_holds")
        self.assertEqual(summary["direct_to_eligible_fraction_max"], 1.0)
        with self.assertRaisesRegex(ValueError, "exceeds eligible mass"):
            target.summarize(masses, 5, queries, [((1,), 6), ((), 10)])

    def test_coverage_profile_reader_matches_query_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profiled.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("coverage_count", "labels"))
                writer.writerow((7, "2 1"))
                writer.writerow((10, ""))
            self.assertEqual(
                target.read_coverage_profile(path),
                [((1, 2), 7), ((), 10)],
            )


if __name__ == "__main__":
    unittest.main()
