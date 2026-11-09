#!/usr/bin/env python3

import csv
import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from result_completion import search_summary_complete, should_skip_completed_search


class SearchSummaryCompleteTest(unittest.TestCase):
    def write_summary(self, path: Path, rows: list[tuple[object, object, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Lsearch", "Average_Time_ms", "Average_Recall"])
            writer.writerows(rows)

    def test_complete_nonuniform_range_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "search_time_summary.csv"
            self.write_summary(summary, [(100, 2.0, 0.8), (200, 3.0, 0.9), (1000, 4.0, 1.0)])

            self.assertTrue(search_summary_complete(summary, [100, 200, 1000]))

    def test_curator_recall_column_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "search_time_summary.csv"
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Lsearch", "Recall", "Average_Time_ms"])
                writer.writerow([64, 0.8, 2.0])

            self.assertTrue(search_summary_complete(summary, [64]))

    def test_partial_range_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "search_time_summary.csv"
            self.write_summary(summary, [(100, 2.0, 0.8), (1000, 4.0, 1.0)])

            self.assertFalse(search_summary_complete(summary, [100, 200, 1000]))

    def test_empty_metrics_are_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "search_time_summary.csv"
            self.write_summary(summary, [(100, "", 0.8)])

            self.assertFalse(search_summary_complete(summary, [100]))

    def test_narrow_scope_can_force_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "search_time_summary.csv"
            self.write_summary(summary, [(100, 2.0, 0.8)])

            self.assertFalse(
                should_skip_completed_search(
                    summary,
                    [100],
                    {"skip_completed_searches": True},
                    {"skip_completed_searches": False},
                )
            )


if __name__ == "__main__":
    unittest.main()
