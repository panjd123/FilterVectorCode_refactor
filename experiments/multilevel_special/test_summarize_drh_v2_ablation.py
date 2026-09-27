#!/usr/bin/env python3

from __future__ import annotations

import unittest

import summarize_drh_v2_ablation as target


class SummarizeDrhV2AblationTest(unittest.TestCase):
    def test_crossing_uses_minimum_measured_l_with_worst_warm_recall(self) -> None:
        rows = [
            {"method": "m", "workload": "w", "lsearch": "200",
             "recall": "0.92", "recall_min": "0.91"},
            {"method": "m", "workload": "w", "lsearch": "100",
             "recall": "0.91", "recall_min": "0.89"},
            {"method": "other", "workload": "w", "lsearch": "50",
             "recall": "1", "recall_min": "1"},
        ]
        status, row = target.operating_point(rows, "m", "w", 0.9)
        self.assertEqual(status, "crossing")
        self.assertEqual(row["lsearch"], "200")

    def test_no_crossing_retains_best_measured_point(self) -> None:
        rows = [
            {"method": "m", "workload": "w", "lsearch": "100",
             "recall": "0.89", "recall_min": "0.88"},
            {"method": "m", "workload": "w", "lsearch": "200",
             "recall": "0.88", "recall_min": "0.87"},
        ]
        status, row = target.operating_point(rows, "m", "w", 0.9)
        self.assertEqual(status, "no_crossing")
        self.assertEqual(row["lsearch"], "100")

    def test_nearest_rank_percentile(self) -> None:
        self.assertEqual(target.percentile([1.0, 2.0, 3.0, 4.0], 0.95), 4.0)
        self.assertEqual(target.percentile([], 0.95), "")


if __name__ == "__main__":
    unittest.main()
