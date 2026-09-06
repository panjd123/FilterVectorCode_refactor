import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import run_selection_sweep
import run_build_sweep
import summarize_selection_sweep
import validate_selection_sweep


class SelectionSweepTest(unittest.TestCase):
    def test_validator_uses_method_specific_lsearch_grid(self):
        config = {"lsearch_values": [100]}
        self.assertEqual(
            validate_selection_sweep.expected_lsearch_values(
                config, {"name": "single", "lsearch_values": [100, 500]}),
            {100, 500},
        )
        self.assertEqual(
            validate_selection_sweep.expected_lsearch_values(
                config, {"name": "default"}),
            {100},
        )

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
            self.assertFalse(run_selection_sweep.result_is_complete(
                root, [100, 200], require_stage_breakdown=True))
            (root / "search_stage_details.csv").write_text(
                "Repeat,Lsearch,AverageQueryTotal_ms,AverageELS_ms,"
                "AverageEntryPointSetup_ms,AverageBlockAuthorization_ms,"
                "AverageGraphSearch_ms,AverageResidual_ms,ClosureError_ms\n"
                "0,100,1,0.1,0.2,0.1,0.5,0.1,0\n"
                "0,200,2,0.1,0.2,0.1,1.5,0.1,0\n"
            )
            self.assertTrue(run_selection_sweep.result_is_complete(
                root, [100, 200], require_stage_breakdown=True))

    def test_summarizer_reads_warm_stage_medians(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "method" / "workload"
            run.mkdir(parents=True)
            (run / "search_time_summary.csv").write_text(
                "Lsearch,Average_Time_ms,Average_Recall\n100,20,0.9\n")
            (run / "search_time_details.csv").write_text(
                "Repeat,Lsearch,Time_ms,Avg_Recall\n"
                "0,100,30,0.9\n1,100,20,0.9\n2,100,22,0.9\n")
            (run / "search_stage_details.csv").write_text(
                "Repeat,Lsearch,AverageQueryTotal_ms,AverageELS_ms,"
                "AverageEntryPointSetup_ms,AverageBlockAuthorization_ms,"
                "AverageGraphSearch_ms,AverageResidual_ms,ClosureError_ms\n"
                "0,100,3,0.3,0.3,0.3,1.8,0.3,0\n"
                "1,100,2,0.2,0.2,0.2,1.2,0.2,0\n"
                "2,100,2.2,0.4,0.2,0.2,1.2,0.2,0\n")
            rows = summarize_selection_sweep.read_rows({
                "output_root": str(root),
                "methods": [{"name": "method"}],
                "workloads": [{"name": "workload", "query_dir": "q",
                               "mean_selectivity": 0.5}],
            })
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["els_ms_warm_median"], 0.3)
            self.assertAlmostEqual(rows[0]["graph_ms_warm_median"], 1.2)
            self.assertAlmostEqual(rows[0]["closure_error_ms_max_abs"], 0.0)

    def test_search_binary_snapshot_is_content_addressed_and_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / "search"
            binary.write_bytes(b"version-one")
            snapshot, digest = run_selection_sweep.snapshot_search_app(binary, root / "out")
            self.assertTrue(snapshot.name.endswith(digest))
            self.assertEqual(snapshot.read_bytes(), b"version-one")
            self.assertEqual(snapshot.stat().st_mode & 0o222, 0)
            binary.write_bytes(b"version-two")
            next_snapshot, next_digest = run_selection_sweep.snapshot_search_app(
                binary, root / "out")
            self.assertNotEqual(next_digest, digest)
            self.assertNotEqual(next_snapshot, snapshot)

    def test_build_binary_snapshot_is_content_addressed_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "builder"
            binary.write_bytes(b"version-one")
            snapshot, digest = run_build_sweep.snapshot_build_app(binary, root / "out")
            self.assertEqual(run_build_sweep.sha256_file(snapshot), digest)
            self.assertEqual(snapshot.stat().st_mode & 0o222, 0)
            binary.write_bytes(b"version-two")
            second, second_digest = run_build_sweep.snapshot_build_app(binary, root / "out")
            self.assertNotEqual(digest, second_digest)
            self.assertNotEqual(snapshot, second)

    def test_query_provenance_rejects_wrong_block_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            main = root / "main"
            block = root / "block"
            data = root / "data"
            main.mkdir()
            block.mkdir()
            data.mkdir()
            (main / "meta").write_text("num_points=2\nnum_groups=2\n")
            (main / "labels.txt").write_text("1\n2\n")
            (data / "Amazon_base_labels.txt").write_text("1\n2\n")
            (block / "meta").write_text(
                "num_points=2\nnum_groups=2\nsource_ung_fingerprint=wrong\n"
            )
            config = {"main_index": str(main), "data_root": str(data),
                      "dataset": "Amazon", "expected_source_fingerprint": "right"}
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                run_selection_sweep.validate_provenance(
                    config, {"name": "multi", "block_index": str(block)})

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
            {},
            25000,
        )
        self.assertEqual(env["PATH"], "/bin")
        self.assertNotIn("UNG_SPECIAL_BLOCK_GPU_INTER", env)
        self.assertEqual(env["UNG_SPECIAL_BLOCK_MIN_POINTS"], "1000")
        self.assertEqual(env["UNG_SPECIAL_BLOCK_UPPER_MIN_POINTS"], "25000")

    def test_build_case_can_override_t1_without_changing_global_config(self):
        config = {"min_points": 1000, "env": {}}
        case = {"name": "single_2000", "min_points": 2000}
        env = run_build_sweep.clean_build_env({}, config, case, None)
        self.assertEqual(run_build_sweep.case_min_points(config, case), 2000)
        self.assertEqual(env["UNG_SPECIAL_BLOCK_MIN_POINTS"], "2000")
        self.assertNotIn("UNG_SPECIAL_BLOCK_UPPER_MIN_POINTS", env)

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
            self.assertEqual(
                provenance["build_binary_sha256"],
                run_build_sweep.sha256_file(build_app),
            )

    def test_equal_recall_uses_fastest_observed_feasible_point(self):
        rows = [
            {"workload": "w", "method": "single", "lsearch": 1000,
             "recall": 0.91, "batch_ms_warm": 10.0, "batch_ms_warm_median": 9.0},
            {"workload": "w", "method": "multi", "lsearch": 500,
             "recall": 0.90, "batch_ms_warm": 7.0, "batch_ms_warm_median": 6.0},
            {"workload": "w", "method": "multi", "lsearch": 700,
             "recall": 0.93, "batch_ms_warm": 8.0, "batch_ms_warm_median": 7.0},
        ]
        result = summarize_selection_sweep.equal_recall_rows(rows, "single", [0.9])
        selected = {row["method"]: row for row in result}
        self.assertEqual(selected["multi"]["batch_ms_warm"], 7.0)
        self.assertAlmostEqual(selected["multi"]["speedup_vs_baseline"], 10.0 / 7.0)
        self.assertAlmostEqual(selected["multi"]["speedup_vs_baseline_median"], 1.5)
        self.assertAlmostEqual(selected["multi"]["lsearch_reduction_vs_baseline"], 0.5)
        self.assertAlmostEqual(selected["multi"]["recall_margin"], 0.0)

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

    def test_conservative_equal_recall_uses_minimum_repeat_recall(self):
        rows = [
            {"workload": "w", "method": "single", "lsearch": 500,
             "recall": 0.91, "recall_min": 0.90, "batch_ms_warm": 10.0,
             "batch_ms_warm_median": 9.0},
            {"workload": "w", "method": "multi", "lsearch": 400,
             "recall": 0.92, "recall_min": 0.89, "batch_ms_warm": 6.0,
             "batch_ms_warm_median": 5.5},
            {"workload": "w", "method": "multi", "lsearch": 500,
             "recall": 0.93, "recall_min": 0.91, "batch_ms_warm": 8.0,
             "batch_ms_warm_median": 7.5},
        ]
        targets = summarize_selection_sweep.baseline_l_targets(
            rows, "single", [500], recall_field="recall_min")
        result = summarize_selection_sweep.equal_recall_rows_by_workload(
            rows, "single", targets, recall_field="recall_min")
        selected = {row["method"]: row for row in result}
        self.assertEqual(selected["multi"]["batch_ms_warm"], 8.0)
        self.assertAlmostEqual(selected["multi"]["speedup_vs_baseline"], 1.25)


if __name__ == "__main__":
    unittest.main()
