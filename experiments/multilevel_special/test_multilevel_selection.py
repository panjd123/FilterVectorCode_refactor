import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import run_selection_sweep
import run_build_sweep
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

    def test_run_lock_rejects_a_second_runner_for_same_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = run_selection_sweep.acquire_run_lock(root)
            try:
                with self.assertRaisesRegex(RuntimeError, "another selection sweep"):
                    run_selection_sweep.acquire_run_lock(root)
            finally:
                first.close()
            second = run_selection_sweep.acquire_run_lock(root)
            second.close()

    def test_build_env_is_explicit_and_drops_inherited_ung_settings(self):
        env = run_build_sweep.clean_build_env(
            {"PATH": "/bin", "UNG_SPECIAL_BLOCK_GPU_INTER": "1"},
            {"min_points": 1000, "env": {"UNG_SPECIAL_INTRA_ROUTE": "1"}},
            25000,
        )
        self.assertEqual(env["PATH"], "/bin")
        self.assertNotIn("UNG_SPECIAL_BLOCK_GPU_INTER", env)
        self.assertEqual(env["UNG_SPECIAL_BLOCK_MIN_POINTS"], "1000")
        self.assertEqual(env["UNG_SPECIAL_BLOCK_UPPER_MIN_POINTS"], "25000")

    def test_build_case_validation_checks_threshold_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            block = root / "block_index"
            results = root / "results"
            block.mkdir()
            results.mkdir()
            for name in run_build_sweep.REQUIRED_INDEX_FILES:
                (block / name).write_bytes(b"x")
            (block / "meta").write_text(
                "index_format=special_block_trie_multilevel_v1\n"
                "source_input=ung_index\n"
                "source_ung_fingerprint=abc\n"
                "num_points=10\nnum_groups=8\n"
                "special_block_min_points=1000\n"
                "special_block_upper_min_points=10000\n"
                "special_block_max_degree=64\n"
                "special_block_num_cross_edges=4\n"
                "special_block_upper_count=2\n"
            )
            (results / "build_time.csv").write_text("Metric,Value\ntotal_time,1\n")
            config = {"expected_source_fingerprint": "abc", "expected_num_points": 10,
                      "expected_num_groups": 8, "min_points": 1000,
                      "max_degree": 64, "num_cross_edges": 4}
            case = {"name": "t2_10000", "upper_min_points": 10000}
            self.assertEqual(run_build_sweep.validate_case(config, case, root)["special_block_upper_count"], "2")
            case["upper_min_points"] = 25000
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                run_build_sweep.validate_case(config, case, root)

    def test_source_validation_rejects_label_version_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = root / "index"
            index.mkdir()
            (index / "meta").write_text("num_points=2\n")
            (index / "labels.txt").write_text("1\n2\n")
            (index / "new_to_old_vec_ids").write_text("0\n1\n")
            base_bin = root / "base.bin"
            base_bin.write_bytes((2).to_bytes(4, "little") +
                                 (3).to_bytes(4, "little"))
            base_labels = root / "base_labels.txt"
            base_labels.write_text("1\n3\n")
            build_app = root / "builder"
            build_app.write_text("placeholder")
            config = {
                "build_app": str(build_app),
                "main_index": str(index),
                "base_bin_file": str(base_bin),
                "base_label_file": str(base_labels),
                "expected_num_points": 2,
            }
            with self.assertRaisesRegex(ValueError, "source index labels do not match"):
                run_build_sweep.validate_source(config)

    def test_source_validation_accepts_reordered_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = root / "index"
            index.mkdir()
            (index / "meta").write_text("num_points=3\n")
            (index / "labels.txt").write_text("3\n1\n2\n")
            (index / "new_to_old_vec_ids").write_text("2\n0\n1\n")
            base_bin = root / "base.bin"
            base_bin.write_bytes((3).to_bytes(4, "little") +
                                 (2).to_bytes(4, "little"))
            base_labels = root / "base_labels.txt"
            base_labels.write_text("1\n2\n3\n")
            build_app = root / "builder"
            build_app.write_text("placeholder")
            config = {
                "build_app": str(build_app),
                "main_index": str(index),
                "base_bin_file": str(base_bin),
                "base_label_file": str(base_labels),
                "expected_num_points": 3,
            }
            provenance = run_build_sweep.validate_source(config)
            self.assertEqual(provenance["label_alignment"],
                             "new_to_old_permutation")

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

    def test_baseline_l_targets_are_workload_specific(self):
        rows = [
            {"workload": "low", "method": "single", "lsearch": 500, "recall": 0.8},
            {"workload": "low", "method": "single", "lsearch": 1000, "recall": 0.9},
            {"workload": "high", "method": "single", "lsearch": 500, "recall": 0.4},
            {"workload": "high", "method": "single", "lsearch": 1000, "recall": 0.5},
        ]
        targets = summarize_selection_sweep.baseline_l_targets(rows, "single", [500, 1000])
        self.assertEqual(targets["low"], [0.8, 0.9])
        self.assertEqual(targets["high"], [0.4, 0.5])


if __name__ == "__main__":
    unittest.main()
