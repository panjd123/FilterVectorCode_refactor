import csv
import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path

import run_selection_sweep
import run_build_sweep
import generate_layer_tuning_config
import generate_layer_tuning_formal
import generate_layer_tuning_report
import select_layer_tuning
import summarize_selection_sweep
import validate_selection_sweep


class SelectionSweepTest(unittest.TestCase):
    def test_validator_parses_executed_command_options(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "command.txt"
            path.write_text("/tmp/search --num_threads 100 --Lsearch 10 20 --K 10\n")
            self.assertEqual(validate_selection_sweep.command_options(path), {
                "--num_threads": ["100"], "--Lsearch": ["10", "20"],
                "--K": ["10"],
            })

    def test_method_can_limit_formal_rerun_to_selected_workloads(self):
        method = {"enabled_workloads": ["a", "c"]}
        self.assertTrue(run_selection_sweep.method_enabled_for_workload(
            method, {"name": "a"}))
        self.assertFalse(run_selection_sweep.method_enabled_for_workload(
            method, {"name": "b"}))
        self.assertTrue(run_selection_sweep.method_enabled_for_workload(
            {}, {"name": "b"}))

    def test_layer_tuning_grid_covers_independent_legal_structures(self):
        config = generate_layer_tuning_config.make_config(Path("/repo"))
        methods = config["methods"]
        self.assertEqual(config["expected_num_queries"], 1000)
        self.assertEqual(sum(item["layer_count"] == 0 for item in methods), 1)
        self.assertEqual(
            {item["t1"] for item in methods if item["layer_count"] == 1},
            set(generate_layer_tuning_config.LAYER1_T1_VALUES),
        )
        two_level = [item for item in methods if item["layer_count"] == 2]
        self.assertEqual(len(two_level), 18)
        self.assertEqual(
            {item["t1"] for item in two_level},
            set(generate_layer_tuning_config.LAYER2_T1_VALUES),
        )
        self.assertTrue(all(item["t2"] > item["t1"] for item in two_level))
        self.assertTrue(all(
            set(item["lsearch_values_by_workload"]) ==
            {workload["name"] for workload in config["workloads"]}
            for item in methods
        ))
        self.assertTrue(all(
            item["env"].get("UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE") == "1"
            for item in methods if item["layer_count"] > 0
        ))
        self.assertNotIn(
            "UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE",
            next(item for item in methods if item["layer_count"] == 0)["env"],
        )

    def test_shared_tuning_requires_one_structure_to_cover_all_workloads(self):
        def row(method, layer, workload, latency, recall, t1=None):
            return {"method": method, "layer_count": layer, "workload": workload,
                    "batch_ms_warm_median": latency, "batch_ms_warm": latency,
                    "recall_min": recall, "lsearch": 100, "t1": t1, "t2": None}
        points = [
            row("plain", 0, "a", 10, .9), row("plain", 0, "b", 20, .9),
            row("t1_1", 1, "a", 5, .9, 1), row("t1_1", 1, "b", 30, .9, 1),
            row("t1_2", 1, "a", 7, .9, 2), row("t1_2", 1, "b", 10, .9, 2),
            row("oracle_only", 1, "a", 1, .9, 3),
        ]
        summaries, selected = select_layer_tuning.select_shared(
            points, {"a": .9, "b": .9})
        chosen = next(item for item in summaries if item["layer_count"] == 1)
        self.assertEqual(chosen["method"], "t1_2")
        self.assertEqual({row["method"] for row in selected
                          if row["layer_count"] == 1}, {"t1_2"})

    def test_boundary_audit_respects_two_level_legal_slices(self):
        points = []
        for t1, t2 in ((500, 4000), (500, 10000),
                       (2000, 4000), (2000, 10000)):
            points.append({"layer_count": 2, "method": f"m_{t1}_{t2}",
                           "t1": t1, "t2": t2})
        oracle = [{"layer_count": 2, "method": "m_2000_4000",
                   "workload": "w", "t1": 2000, "t2": 4000}]
        audit = select_layer_tuning.audit_structure_boundaries(points, oracle, [])
        self.assertEqual(
            {(row["axis"], row["direction"], row["measured_axis_values"])
             for row in audit},
            {("t1", "upper", "500;2000"),
             ("t2", "lower", "4000;10000")},
        )

    def test_boundary_audit_can_use_full_coarse_reference_grid(self):
        formal_points = [{"layer_count": 1, "method": "m2",
                          "t1": 2000, "t2": None}]
        oracle = [{"layer_count": 1, "method": "m2", "workload": "w",
                   "t1": 2000, "t2": None}]
        reference = [
            {"layer_count": 1, "method": "m1", "t1": 1000, "t2": None},
            {"layer_count": 1, "method": "m2", "t1": 2000, "t2": None},
            {"layer_count": 1, "method": "m4", "t1": 4000, "t2": None},
        ]
        self.assertEqual(
            select_layer_tuning.audit_structure_boundaries(
                formal_points, oracle, [], reference), [])

    def test_formal_grid_brackets_coarse_recall_crossing(self):
        rows = [
            {"lsearch": 100, "recall_min": .80},
            {"lsearch": 500, "recall_min": .89},
            {"lsearch": 1000, "recall_min": .91},
            {"lsearch": 2000, "recall_min": .93},
        ]
        self.assertEqual(
            generate_layer_tuning_formal.dense_crossing_grid(rows, .90),
            [500, 625, 750, 875, 1000, 2000],
        )

    def test_formal_grid_extends_when_crossing_is_at_l_boundary(self):
        rows = [
            {"lsearch": 100, "recall_min": .80},
            {"lsearch": 200, "recall_min": .90},
        ]
        self.assertEqual(
            generate_layer_tuning_formal.dense_crossing_grid(rows, .90),
            [100, 125, 150, 175, 200, 250],
        )

    def test_formal_oracle_only_structure_runs_only_selected_workload(self):
        def row(method, workload, latency, layer=1):
            return {"method": method, "layer_count": layer,
                    "workload": workload, "batch_ms_warm_median": latency,
                    "batch_ms_warm": latency, "recall_min": .91,
                    "lsearch": 100,
                    "t1": int(method[-1]) if layer else None, "t2": None}
        points = [
            row("plain", "a", 10, layer=0),
            row("plain", "b", 10, layer=0),
            row("m1", "a", 1), row("m1", "b", 100),
            row("m2", "a", 2), row("m2", "b", 2),
            row("m3", "a", 100), row("m3", "b", 1),
        ]
        selected = generate_layer_tuning_formal.selected_structure_workloads(
            points, {"a": .9, "b": .9}, shared_top_k=1, oracle_top_k=1)
        self.assertEqual(selected[(1, "m2")], {"a", "b"})
        self.assertEqual(selected[(1, "m1")], {"a"})
        self.assertEqual(selected[(1, "m3")], {"b"})

    def test_formal_config_uses_fresh_output_and_records_selection_policy(self):
        coarse = {
            "output_root": "/tmp/layer_tuning_query_coarse_amazon_x1",
            "recall_thresholds": {"w": .9},
            "workloads": [{"name": "w"}],
            "methods": [{"name": "plain", "layer_count": 0}],
        }
        points = [{"method": "plain", "layer_count": 0, "workload": "w",
                   "batch_ms_warm_median": 10, "batch_ms_warm": 10,
                   "recall_min": .91, "lsearch": 100, "t1": None, "t2": None}]
        formal = generate_layer_tuning_formal.make_formal_config(coarse, points)
        self.assertTrue(formal["output_root"].endswith(
            "layer_tuning_query_formal_fair_amazon_x1"))
        self.assertEqual(formal["num_repeats"], 7)
        self.assertEqual(formal["formal_selection"]["shared_top_k_per_layer"], 3)

    def test_layer_report_keeps_batch_and_per_query_stages_distinct(self):
        row = {
            "selection_scope": "shared_thresholds", "workload": "w",
            "mean_selectivity": "0.5", "target_recall": "0.9",
            "layer_count": "0", "method": "plain", "t1": "",
            "t2": "", "lsearch": "100", "recall": "0.91",
            "recall_min": "0.90", "batch_ms_warm_median": "20",
            "speedup_vs_layer0": "1", "query_total_ms_at_batch_median": "1",
            "stage_closure_ms_at_batch_median": "0",
            "els_ms_at_batch_median": "0.1", "entry_ms_at_batch_median": "0.2",
            "block_authorization_ms_at_batch_median": "0",
            "graph_ms_at_batch_median": "0.6", "residual_ms_at_batch_median": "0.1",
        }
        report = generate_layer_tuning_report.render_report(
            {"expected_num_queries": 1000},
            [{"layer_count": "0", "method": "plain", "t1": "", "t2": "",
              "geomean_speedup_vs_layer0": "1", "sum_batch_ms_warm_median": "20"}],
            [row], [row], [])
        self.assertIn("batch 的 warm-repeat 中位墙钟", report)
        self.assertIn("50000.0", report)
        self.assertIn("既有 UNG 主图", report)

    def test_layer_report_build_cost_is_incremental_and_selected(self):
        selected = [{"method": "layer0_plain"}, {"method": "layer1_t1_2000"}]
        manifest = {"runs": [{
            "name": "layer1_t1_2000", "min_points": 2000,
            "upper_min_points": None, "status": "complete", "returncode": 0,
            "elapsed_seconds": 42.0,
            "metadata": {
                "special_block_metadata_time(ms)": "1", "special_block_trie_build_time(ms)": "2",
                "special_edge_intra_build_time(ms)": "3", "special_edge_inter_build_time(ms)": "4",
                "special_trie_regular_edge_build_time(ms)": "5", "special_blocks_save_time(ms)": "6",
                "special_block_count": "7", "special_block_upper_count": "0",
                "special_edge_count": "8", "disk_bytes": str(1024 ** 3),
            },
        }, {"name": "unselected", "status": "complete", "returncode": 0}]}
        rows = generate_layer_tuning_report.selected_build_rows(selected, [], manifest)
        self.assertEqual([row["method"] for row in rows], ["layer1_t1_2000"])
        self.assertEqual(rows[0]["wall_s"], 42.0)

    def test_validator_uses_method_specific_lsearch_grid(self):
        config = {"lsearch_values": [100]}
        self.assertEqual(
            validate_selection_sweep.expected_lsearch_values(
                config, {"name": "single", "lsearch_values": [100, 500]},
                {"name": "w"}),
            {100, 500},
        )
        self.assertEqual(
            validate_selection_sweep.expected_lsearch_values(
                config, {"name": "default"}, {"name": "w"}),
            {100},
        )

    def test_workload_specific_lsearch_grid_is_most_specific(self):
        config = {"lsearch_values": [10]}
        method = {
            "name": "multi",
            "lsearch_values": [20],
            "lsearch_values_by_workload": {"high": [100, 200]},
        }
        self.assertEqual(
            run_selection_sweep.lsearch_values_for(
                config, method, {"name": "high", "lsearch_values": [30]}),
            [100, 200],
        )
        self.assertEqual(
            run_selection_sweep.lsearch_values_for(
                config, method, {"name": "low", "lsearch_values": [30]}),
            [30],
        )
        with self.assertRaisesRegex(ValueError, "invalid Lsearch grid"):
            run_selection_sweep.lsearch_values_for(
                config, {"name": "bad", "lsearch_values": []}, {"name": "w"})

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

    def test_result_reuse_requires_matching_execution_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / "search"
            binary.write_bytes(b"binary")
            (root / "search_time_summary.csv").write_text(
                "Lsearch,Average_Time_ms\n100,1\n")
            (root / "search_time_details.csv").write_text(
                "Repeat,Lsearch,Time_ms,Avg_Recall\n0,100,1,.9\n1,100,1,.9\n")
            command = [str(binary), "--Lsearch", "100"]
            (root / "command.txt").write_text(shlex.join(command) + "\n")
            environment = {"UNG_DISABLE_ELS_REUSE": "1"}
            (root / "environment.json").write_text(json.dumps(environment))
            digest = run_selection_sweep.sha256_file(binary)
            self.assertTrue(run_selection_sweep.result_is_complete(
                root, [100], expected_repeats=2, expected_command=command,
                expected_environment=environment, expected_binary_sha256=digest))
            self.assertFalse(run_selection_sweep.result_is_complete(
                root, [100], expected_repeats=3, expected_command=command,
                expected_environment=environment, expected_binary_sha256=digest))
            self.assertFalse(run_selection_sweep.result_is_complete(
                root, [100], expected_repeats=2, expected_command=command,
                expected_environment={"UNG_DISABLE_ELS_REUSE": "0"},
                expected_binary_sha256=digest))
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
            self.assertAlmostEqual(rows[0]["query_total_ms_at_batch_median"], 2.1)
            self.assertAlmostEqual(rows[0]["els_ms_at_batch_median"], 0.3)
            self.assertAlmostEqual(
                sum(rows[0][field] for field, _ in generate_layer_tuning_report.STAGES),
                rows[0]["query_total_ms_at_batch_median"])

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

    def test_query_provenance_rejects_wrong_block_thresholds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            main = root / "main"
            block = root / "block"
            data = root / "data"
            main.mkdir()
            block.mkdir()
            data.mkdir()
            (main / "meta").write_text("num_points=2\nnum_groups=3\n")
            (main / "labels.txt").write_text("1\n2\n")
            (data / "Amazon_base_labels.txt").write_text("1\n2\n")
            (block / "meta").write_text(
                "source_ung_fingerprint=abc\nnum_points=2\nnum_groups=3\n"
                "special_block_min_points=1000\nspecial_block_upper_min_points=0\n"
                "special_block_upper_count=0\n")
            config = {"main_index": str(main), "data_root": str(data),
                      "expected_source_fingerprint": "abc"}
            method = {"name": "single", "block_index": str(block),
                      "layer_count": 1, "t1": 2000}
            with self.assertRaisesRegex(ValueError, "T1 mismatch"):
                run_selection_sweep.validate_provenance(config, method)

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

    def test_manifest_update_replaces_dry_run_with_completed_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            common = {"method": "plain", "workload": "w"}
            run_selection_sweep.update_manifest(
                path, {**common, "status": "dry_run"})
            run_selection_sweep.update_manifest(
                path, {**common, "status": "complete", "reused_existing": True})
            runs = json.loads(path.read_text())["runs"]
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["status"], "complete")
            self.assertTrue(runs[0]["reused_existing"])

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
