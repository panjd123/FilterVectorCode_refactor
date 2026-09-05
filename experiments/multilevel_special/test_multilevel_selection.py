import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import run_selection_sweep
import summarize_selection_sweep


class SelectionSweepTest(unittest.TestCase):
    def test_clean_method_env_removes_inherited_special_settings(self):
        env = run_selection_sweep.clean_method_env(
            {"PATH": "/bin", "UNG_SPECIAL_BLOCK_SEARCH": "1",
             "UNG_SPECIAL_OLD": "x", "UNG_DISABLE_ELS_REUSE": "1"},
            {"special_block_search": False, "env": {"UNG_SPECIAL_LIGHT_STATS": "1"}},
        )
        self.assertNotIn("UNG_SPECIAL_BLOCK_SEARCH", env)
        self.assertNotIn("UNG_SPECIAL_OLD", env)
        self.assertNotIn("UNG_DISABLE_ELS_REUSE", env)
        self.assertEqual(env["UNG_SPECIAL_LIGHT_STATS"], "1")

    def test_result_is_complete_requires_exact_l_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "search_time_summary.csv").write_text(
                "Lsearch,Average_Time_ms,Average_Recall\n100,1,0.8\n200,2,0.9\n"
            )
            self.assertTrue(run_selection_sweep.result_is_complete(root, [100, 200]))
            self.assertFalse(run_selection_sweep.result_is_complete(root, [100, 200, 300]))

    def test_equal_recall_uses_fastest_observed_feasible_point(self):
        rows = [
            {"workload": "w", "method": "single", "recall": 0.91, "batch_ms_warm": 10.0},
            {"workload": "w", "method": "multi", "recall": 0.90, "batch_ms_warm": 7.0},
            {"workload": "w", "method": "multi", "recall": 0.93, "batch_ms_warm": 8.0},
        ]
        result = summarize_selection_sweep.equal_recall_rows(rows, "single", [0.9])
        selected = {row["method"]: row for row in result}
        self.assertEqual(selected["multi"]["batch_ms_warm"], 7.0)
        self.assertAlmostEqual(selected["multi"]["speedup_vs_baseline"], 10.0 / 7.0)


if __name__ == "__main__":
    unittest.main()
