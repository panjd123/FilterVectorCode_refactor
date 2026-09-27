import csv
import json
import os
import shlex
import struct
import tempfile
import unittest
from pathlib import Path

import experiment_cli
import run_selection_sweep
import run_build_sweep
import generate_layer_tuning_config
import generate_layer_tuning_formal
import generate_high_selectivity_tuning_config
import generate_layer_tuning_report
import select_layer_tuning
import summarize_selection_sweep
import validate_selection_sweep


class SelectionSweepTest(unittest.TestCase):
    def test_runner_accepts_empty_label_rows_for_full_containment_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            query = data / "full"
            gt = root / "gt" / "full"
            index = root / "index"
            query.mkdir(parents=True)
            gt.mkdir(parents=True)
            index.mkdir()
            (root / "search").write_text("")
            (data / "Amazon_base_labels.txt").write_text("1\n")
            (query / "Amazon_query.bin").write_bytes(
                struct.pack("<II", 2, 1) + struct.pack("<ff", 0.0, 1.0))
            (query / "Amazon_query_labels.txt").write_text("\n\n")
            (gt / "Amazon_gt_labels_containment.bin").write_bytes(b"\0" * 160)
            (index / "meta").write_text("")
            (index / "labels.txt").write_text("1\n")
            config = {
                "search_app": str(root / "search"), "data_root": str(data),
                "gt_root": str(root / "gt"), "main_index": str(index),
                "dataset": "Amazon", "K": 10, "expected_num_queries": 2,
            }
            workload = {"name": "full", "query_dir": "full"}
            self.assertEqual(run_selection_sweep.validate_case(
                config, {"name": "plain"}, workload), 2)

    def test_runner_uses_workload_scenario_override(self):
        config = {
            "data_root": "/data", "gt_root": "/gt",
            "main_index": "/index", "search_app": "/search",
            "dataset": "Amazon", "num_threads": 1, "K": 10,
            "num_repeats": 1, "num_entry_points": 16,
            "lsearch_values": [10], "scenario": "containment",
        }
        command = run_selection_sweep.build_command(
            config, {"name": "plain"},
            {"name": "full", "query_dir": "full", "scenario": "containment"},
            Path("/run"))
        self.assertEqual(command[command.index("--scenario") + 1], "containment")

    def test_high_selectivity_extensions_are_disjoint_from_coarse_points(self):
        repo = Path("/repo")
        coarse = generate_high_selectivity_tuning_config.make_config(repo, 3)
        plain = generate_high_selectivity_tuning_config.make_plain_extension_config(
            repo, 3)
        structure = (
            generate_high_selectivity_tuning_config.make_structure_extension_config(
                repo, 3))
        structure_boundary = (
            generate_high_selectivity_tuning_config.
            make_structure_boundary_extension_config(repo, 3))
        structure_endpoint = (
            generate_high_selectivity_tuning_config.
            make_structure_endpoint_extension_config(repo, 3))
        self.assertEqual([item["name"] for item in plain["methods"]],
                         ["layer0_plain"])
        coarse_plain = coarse["methods"][0]["lsearch_values_by_workload"]
        extension_plain = plain["methods"][0]["lsearch_values_by_workload"]
        for workload, values in extension_plain.items():
            self.assertTrue(set(values).isdisjoint(coarse_plain[workload]))
        self.assertEqual(len(structure["methods"]), 9)
        self.assertTrue(all(
            item.get("t2") is None or item["t2"] > item["t1"]
            for item in structure["methods"]))
        coarse_names = {item["name"] for item in coarse["methods"]}
        self.assertTrue(coarse_names.isdisjoint(
            item["name"] for item in structure["methods"]))
        existing_structure_names = coarse_names | {
            item["name"] for item in structure["methods"]}
        self.assertEqual(
            [item["name"] for item in structure_boundary["methods"]],
            ["layer1_t1_256000"],
        )
        self.assertTrue(existing_structure_names.isdisjoint(
            item["name"] for item in structure_boundary["methods"]))
        self.assertEqual(
            [item["name"] for item in structure_endpoint["methods"]],
            ["layer1_t1_700000"],
        )
        full = generate_high_selectivity_tuning_config.make_full_plain_extension_config(
            repo, 3)
        self.assertEqual(full["methods"][0]["enabled_workloads"], ["sel_100"])
        self.assertTrue(set(
            full["methods"][0]["lsearch_values_by_workload"]["sel_100"]
        ).isdisjoint(coarse_plain["sel_100"] + extension_plain["sel_100"]))
        formal_boundary = (generate_high_selectivity_tuning_config.
                           make_formal_boundary_extension_config(repo, 7))
        self.assertEqual(formal_boundary["num_repeats"], 7)
        self.assertEqual(
            {item["name"] for item in formal_boundary["methods"]},
            {"layer1_t1_128000", "layer2_t1_16000_t2_400000",
             "layer2_t1_8000_t2_400000",
             "layer2_t1_16000_t2_500000"},
        )
        self.assertEqual(
            next(item for item in formal_boundary["methods"]
                 if item["name"] == "layer2_t1_16000_t2_400000")
            ["lsearch_values_by_workload"]["sel_100"],
            [10, 15, 20],
        )
        self.assertIn(
            {"name": "layer2_t1_16000_t2_700000", "layer_count": 2,
             "t1": 16000, "t2": 700000},
            formal_boundary["boundary_reference_methods"],
        )

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
        original_pairs = {
            (t1, t2)
            for t1 in generate_layer_tuning_config.LAYER2_T1_VALUES
            for t2 in generate_layer_tuning_config.T2_VALUES
            if t2 > t1
        }
        expected_pairs = original_pairs | set(
            generate_layer_tuning_config.LAYER2_GUARD_PAIRS)
        self.assertEqual(len(two_level), len(expected_pairs))
        self.assertEqual(
            {(item["t1"], item["t2"]) for item in two_level},
            expected_pairs,
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
        self.assertEqual(
            len(config["boundary_reference_methods"]), len(methods))
        self.assertIn(
            {"name": "layer1_t1_700000", "layer_count": 1,
             "t1": 700000, "t2": None},
            config["boundary_reference_methods"],
        )
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

    def test_boundary_audit_accepts_config_style_reference_names(self):
        points = [{"layer_count": 1, "method": "m_2000",
                   "t1": 2000, "t2": None}]
        oracle = [{"layer_count": 1, "method": "m_2000",
                   "workload": "w", "t1": 2000, "t2": None}]
        references = [
            {"layer_count": 1, "name": "m_1000", "t1": 1000, "t2": None},
            {"layer_count": 1, "name": "m_2000", "t1": 2000, "t2": None},
            {"layer_count": 1, "name": "m_4000", "t1": 4000, "t2": None},
        ]
        self.assertEqual(
            select_layer_tuning.audit_structure_boundaries(
                points, oracle, [], references),
            [],
        )

    def test_boundary_audit_closes_strict_integer_legal_endpoint(self):
        oracle = [{"layer_count": 2, "method": "adjacent",
                   "workload": "w", "t1": 64000, "t2": 64001}]
        references = [
            {"layer_count": 2, "name": "lower_t1",
             "t1": 32000, "t2": 64001},
            {"layer_count": 2, "name": "adjacent",
             "t1": 64000, "t2": 64001},
            {"layer_count": 2, "name": "higher_t2",
             "t1": 64000, "t2": 80000},
        ]
        self.assertEqual(
            select_layer_tuning.audit_structure_boundaries([], oracle, [], references),
            [],
        )

    def test_boundary_audit_closes_threshold_above_dataset_cardinality(self):
        oracle = [{"layer_count": 1, "method": "endpoint",
                   "workload": "w", "t1": 700000, "t2": None}]
        references = [
            {"layer_count": 1, "name": "lower", "t1": 256000, "t2": None},
            {"layer_count": 1, "name": "endpoint", "t1": 700000, "t2": None},
        ]
        self.assertEqual(
            select_layer_tuning.audit_structure_boundaries(
                [], oracle, [], references, threshold_domain_max=602453),
            [],
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

    def test_lsearch_boundary_audit_requires_a_lower_measured_guard(self):
        points = [
            {"method": "m", "workload": "w", "lsearch": 100},
            {"method": "m", "workload": "w", "lsearch": 200},
        ]
        selected = [{"method": "m", "workload": "w",
                     "layer_count": 1, "t1": 10, "t2": None,
                     "lsearch": 100}]
        audit = select_layer_tuning.audit_lsearch_boundaries(
            points, selected, selected)
        self.assertEqual(len(audit), 2)
        self.assertTrue(all(row["direction"] == "lower" for row in audit))
        selected[0]["lsearch"] = 200
        self.assertEqual(
            select_layer_tuning.audit_lsearch_boundaries(
                points, selected, selected),
            [],
        )
        selected[0]["lsearch"] = 100
        self.assertEqual(
            select_layer_tuning.audit_lsearch_boundaries(
                points, selected, selected, minimum_lsearch=100),
            [],
        )

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

    def test_formal_shortlist_keeps_near_best_beyond_top_k(self):
        def row(method, workload, latency, layer=1):
            return {"method": method, "layer_count": layer,
                    "workload": workload, "batch_ms_warm_median": latency,
                    "batch_ms_warm": latency, "recall_min": .91,
                    "lsearch": 100,
                    "t1": int(method[-1]) if layer else None, "t2": None}
        points = [row("plain", "a", 100, layer=0)]
        points += [row("m1", "a", 10), row("m2", "a", 10.2),
                   row("m3", "a", 10.4), row("m4", "a", 10.6)]
        selected = generate_layer_tuning_formal.selected_structure_workloads(
            points, {"a": .9}, shared_top_k=1, oracle_top_k=1,
            near_best_ratio=1.05)
        self.assertIn((1, "m3"), selected)
        self.assertNotIn((1, "m4"), selected)

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

    def test_formal_config_can_merge_extension_method_sources(self):
        coarse = {
            "output_root": "/tmp/coarse", "recall_thresholds": {"w": .9},
            "workloads": [{"name": "w"}],
            "methods": [{"name": "plain", "layer_count": 0}],
        }
        extension = {"methods": [
            {"name": "m", "layer_count": 1, "t1": 10,
             "block_index": "/index/m"},
        ]}
        points = [
            {"method": "plain", "layer_count": 0, "workload": "w",
             "batch_ms_warm_median": 10, "batch_ms_warm": 10,
             "recall_min": .91, "lsearch": 100, "t1": None, "t2": None},
            {"method": "m", "layer_count": 1, "workload": "w",
             "batch_ms_warm_median": 5, "batch_ms_warm": 5,
             "recall_min": .91, "lsearch": 100, "t1": 10, "t2": None},
        ]
        formal = generate_layer_tuning_formal.make_formal_config(
            coarse, points, [coarse, extension], boundary_guards={},
            output_root_name="high_formal")
        self.assertEqual({item["name"] for item in formal["methods"]},
                         {"plain", "m"})
        self.assertTrue(formal["output_root"].endswith("high_formal"))
        self.assertEqual(formal["formal_selection"]["boundary_guards"], {})

    def test_formal_config_adds_predeclared_boundary_guards(self):
        coarse = {
            "output_root": "/tmp/layer_tuning_query_coarse_amazon_x1",
            "recall_thresholds": {"w": .9},
            "workloads": [{"name": "w"}],
            "methods": [
                {"name": "layer0_plain", "layer_count": 0},
                {"name": "layer2_t1_8000_t2_100000", "layer_count": 2,
                 "t1": 8000, "t2": 100000,
                 "block_index": "/builds/layer2_t1_8000_t2_100000/block_index"},
            ],
        }
        points = [
            {"method": "layer0_plain", "layer_count": 0, "workload": "w",
             "batch_ms_warm_median": 10, "batch_ms_warm": 10,
             "recall_min": .91, "lsearch": 100, "t1": None, "t2": None},
            {"method": "layer2_t1_8000_t2_100000", "layer_count": 2,
             "workload": "w", "batch_ms_warm_median": 5,
             "batch_ms_warm": 5, "recall_min": .91, "lsearch": 200,
             "t1": 8000, "t2": 100000},
        ]
        formal = generate_layer_tuning_formal.make_formal_config(coarse, points)
        guard = next(method for method in formal["methods"]
                     if method["name"] == "layer2_t1_8000_t2_200000")
        self.assertEqual(guard["enabled_workloads"], ["w"])
        self.assertEqual(guard["lsearch_values_by_workload"]["w"],
                         [100, 125, 150, 175, 200, 250])
        self.assertEqual(guard["block_index"],
                         "/builds/layer2_t1_8000_t2_200000/block_index")

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
        selected = [
            {"method": "layer0_plain", "layer_count": "0", "t1": "", "t2": ""},
            {"method": "layer1_t1_2000", "layer_count": "1",
             "t1": "2000", "t2": ""},
        ]
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

    def test_layer_report_accepts_validated_reused_build_without_wall_time(self):
        selected = [{"method": "layer2_t1_2000_t2_10000",
                     "layer_count": "2", "t1": "2000", "t2": "10000"}]
        manifest = {"runs": [{
            "name": "layer2_t1_2000_t2_10000", "min_points": 2000,
            "upper_min_points": 10000, "status": "complete",
            "reused_existing": True,
            "metadata": {
                "special_block_metadata_time(ms)": "1",
                "special_block_trie_build_time(ms)": "2",
                "special_edge_intra_build_time(ms)": "3",
                "special_edge_inter_build_time(ms)": "4",
                "special_trie_regular_edge_build_time(ms)": "5",
                "special_blocks_save_time(ms)": "6",
                "special_block_count": "7", "special_block_upper_count": "1",
                "special_edge_count": "8", "disk_bytes": str(1024 ** 3),
            },
        }]}
        rows = generate_layer_tuning_report.selected_build_rows(selected, [], manifest)
        self.assertIsNone(rows[0]["wall_s"])

    def test_layer_report_rejects_missing_returncode_for_nonreused_build(self):
        selected = [{"method": "layer1_t1_2000", "layer_count": "1",
                     "t1": "2000", "t2": ""}]
        manifest = {"runs": [{
            "name": "layer1_t1_2000", "min_points": 2000,
            "upper_min_points": None, "status": "complete",
            "reused_existing": False, "metadata": {},
        }]}
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            generate_layer_tuning_report.selected_build_rows(selected, [], manifest)

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
        with self.assertRaisesRegex(ValueError, "must be >= K=10"):
            run_selection_sweep.lsearch_values_for(
                {"K": 10, "lsearch_values": [9]},
                {"name": "invalid"}, {"name": "low"})
        with self.assertRaisesRegex(ValueError, "invalid Lsearch grid"):
            run_selection_sweep.lsearch_values_for(
                config, {"name": "bad", "lsearch_values": []}, {"name": "w"})

    def test_clean_method_env_removes_inherited_special_settings(self):
        env = run_selection_sweep.clean_method_env(
            {"PATH": "/bin", "UNG_SPECIAL_BLOCK_SEARCH": "1",
             "UNG_SPECIAL_OLD": "x", "UNG_DISABLE_ELS_REUSE": "1"},
            {},
            {"special_block_search": False, "env": {"UNG_SPECIAL_LIGHT_STATS": "1"}},
        )
        self.assertEqual(env["UNG_SPECIAL_BLOCK_SEARCH"], "0")
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
                "0,100,30,0.5\n1,100,20,0.91\n2,100,22,0.92\n")
            (run / "search_stage_details.csv").write_text(
                "Repeat,Lsearch,AverageQueryTotal_ms,AverageELS_ms,"
                "AverageEntryPointSetup_ms,AverageBlockAuthorization_ms,"
                "AverageGraphSearch_ms,AverageResidual_ms,ClosureError_ms\n"
                "0,100,3,0.3,0.3,0.3,1.8,0.3,0\n"
                "1,100,2,0.2,0.2,0.2,1.2,0.2,0\n"
                "2,100,2.2,0.4,0.2,0.2,1.2,0.2,0\n")
            (run / "search_work_details.csv").write_text(
                "Repeat,Lsearch,AverageNodesVisited,AverageTotalEdgesScanned,"
                "AverageTotalDistanceCalcs\n"
                "0,100,30,300,330\n"
                "1,100,20,200,220\n"
                "2,100,22,240,262\n")
            (run / "query_details_repeat3.csv").write_text(
                "Repeat,Lsearch,SpecialBlockSearchUsed\n"
                "0,100,1\n0,100,1\n0,100,1\n"
                "1,100,1\n1,100,0\n1,100,0\n"
                "2,100,1\n2,100,1\n2,100,0\n")
            rows = summarize_selection_sweep.read_rows({
                "output_root": str(root),
                "num_repeats": 3,
                "protocol": {"cold_repeats": 1, "measured_repeats": 2},
                "methods": [{"name": "method"}],
                "workloads": [{"name": "workload", "query_dir": "q",
                               "mean_selectivity": 0.5, "num_queries": 3}],
            })
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["qps_warm_median"], 1000.0 * 3 / 21)
            self.assertAlmostEqual(rows[0]["els_ms_warm_median"], 0.3)
            self.assertAlmostEqual(rows[0]["graph_ms_warm_median"], 1.2)
            self.assertAlmostEqual(rows[0]["closure_error_ms_max_abs"], 0.0)
            self.assertAlmostEqual(rows[0]["query_total_ms_at_batch_median"], 2.1)
            self.assertAlmostEqual(rows[0]["els_ms_at_batch_median"], 0.3)
            self.assertAlmostEqual(rows[0]["nodes_visited_warm_median"], 21)
            self.assertAlmostEqual(rows[0]["total_edges_scanned_warm_median"], 220)
            self.assertAlmostEqual(rows[0]["total_distance_calcs_warm_median"], 241)
            self.assertAlmostEqual(
                rows[0]["layered_path_activation_rate_warm_median"], 0.5)
            self.assertAlmostEqual(rows[0]["recall"], 0.915)
            self.assertAlmostEqual(rows[0]["recall_min"], 0.91)
            self.assertAlmostEqual(rows[0]["recall_max"], 0.92)
            self.assertAlmostEqual(
                sum(rows[0][field] for field, _ in generate_layer_tuning_report.STAGES),
                rows[0]["query_total_ms_at_batch_median"])
            report = root / "results.md"
            selected = summarize_selection_sweep.equal_recall_rows(
                rows, "method", [0.9])
            summarize_selection_sweep.write_markdown(
                report, selected, "method",
                summarize_selection_sweep.max_recall_rows(rows))
            text = report.read_text()
            self.assertIn("| QPS |", text)
            self.assertIn("142.857", text)
            self.assertIn("| none |", text)
            self.assertIn("所有 warm repeats", text)
            self.assertIn("warm min Recall", text)
            self.assertIn("0.910000", text)
            self.assertIn("达标点阶段耗时", text)
            self.assertIn("entry-point setup", text)
            self.assertIn("layered path", text)
            self.assertIn("50.0%", text)
            self.assertIn("达标点搜索工作量", text)
            self.assertIn("visited points", text)

    def test_summary_root_separates_performance_and_profile_passes(self):
        common = {"output_root": "/tmp/run", "pass_subdirs": True}
        self.assertEqual(
            summarize_selection_sweep.summary_root(
                {**common, "measurement_pass": "performance"}),
            Path("/tmp/run/summary/performance"),
        )
        self.assertEqual(
            summarize_selection_sweep.summary_root(
                {**common, "measurement_pass": "profile"}),
            Path("/tmp/run/summary/profile"),
        )
        self.assertEqual(
            experiment_cli.default_summary_path(
                {**common, "measurement_pass": "profile"}),
            Path("/tmp/run/summary/profile/results.csv"),
        )

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

    def test_search_binary_snapshot_rejects_campaign_hash_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / "search"
            binary.write_bytes(b"unexpected-version")
            with self.assertRaisesRegex(ValueError, "campaign-pinned SHA256"):
                run_selection_sweep.snapshot_search_app(
                    binary, root / "out", "0" * 64)

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
            (root / "build.log").write_text(
                "[special_edges] gpu_intra_enabled=0 gpu_intra_blocks=0 "
                "gpu_intra_points=0 gpu_intra_fallback_blocks=0\n"
                "[special_edges] gpu_inter_enabled=0 gpu_inter_used=0 "
                "gpu_inter_ms=0\n"
            )
            config = {"expected_source_fingerprint": "abc", "expected_num_points": 10,
                      "expected_num_groups": 8, "min_points": 1000,
                      "max_degree": 64, "num_cross_edges": 4}
            case = {"name": "t2_10000", "upper_min_points": 10000,
                    "benchmark_profile": "cpu"}
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

    def test_equal_recall_uses_smallest_observed_feasible_lsearch(self):
        rows = [
            {"workload": "w", "method": "single", "lsearch": 1000,
             "recall": 0.91, "batch_ms_warm": 10.0, "batch_ms_warm_median": 9.0},
            {"workload": "w", "method": "multi", "lsearch": 500,
             "recall": 0.90, "batch_ms_warm": 7.0, "batch_ms_warm_median": 6.0},
            {"workload": "w", "method": "multi", "lsearch": 700,
             "recall": 0.93, "batch_ms_warm": 3.0, "batch_ms_warm_median": 2.5},
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
