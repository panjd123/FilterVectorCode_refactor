import csv
import json
import tempfile
import unittest
from pathlib import Path

import experiment_core
import analyze_ung_plain_attribution
import generate_ung_plain_comparison


def base_config():
    return {
        "search_app": "/search", "main_index": "/index",
        "data_root": "/data", "gt_root": "/gt", "output_root": "/out",
        "K": 10, "num_threads": 4, "num_entry_points": 16,
        "num_repeats": 3, "lsearch_values": [10],
        "protocol": {"phase": "screen", "cold_repeats": 1,
                     "measured_repeats": 2, "recall_rule": "all_repeats"},
        "methods": [{"name": "plain", "layer_count": 0,
                     "special_block_search": False,
                     "entry_group_provider": "cpu_bruteforce_els"}],
        "workloads": [{"name": "sel_1", "query_dir": "q"}],
        "recall_thresholds": {"sel_1": 0.9},
    }


class ExperimentCoreTest(unittest.TestCase):
    def test_safe_ratio(self):
        self.assertEqual(analyze_ung_plain_attribution.safe_ratio(8, 2), 4)
        self.assertIsNone(analyze_ung_plain_attribution.safe_ratio(8, 0))

    def test_refinement_grid_brackets_crossing_without_retesting_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "details.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["Repeat", "Lsearch", "Avg_Recall"])
                writer.writeheader()
                for value, recall in ((100, .8), (200, .89), (400, .91)):
                    for repeat in range(3):
                        writer.writerow({"Repeat": repeat, "Lsearch": value,
                                         "Avg_Recall": recall})
            self.assertEqual(
                generate_ung_plain_comparison._refinement_grid(path, .9, 3),
                [240, 280, 320, 360, 400])

    def test_ung_plain_config_changes_only_entry_provider(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            config_dir = repo / "experiments/multilevel_special"
            config_dir.mkdir(parents=True)
            source = base_config()
            source.update({
                "expected_source_fingerprint": "source",
                "expected_base_labels_sha256": "base",
                "expected_main_index_labels_sha256": "index",
                "dataset": "Amazon", "expected_num_queries": 1000,
                "require_stage_breakdown": True,
                "methods": [{
                    "name": "layer0_plain", "layer_count": 0,
                    "special_block_search": False,
                    "entry_group_provider": "cpu_bruteforce_els",
                    "lsearch_values_by_workload": {"sel_1": [100]},
                }],
            })
            (config_dir / "config.auto_policy_formal_exact_level.json").write_text(
                __import__("json").dumps(source))
            config = generate_ung_plain_comparison.make_config(repo, "screen")
            experiment_core.validate_config(config)
            self.assertEqual(
                [item["entry_group_provider"] for item in config["methods"]],
                ["cpu_min_super_sets", "cpu_bruteforce_els"],
            )
            comparable = [{key: value for key, value in item.items()
                           if key not in {"name", "entry_group_provider"}}
                          for item in config["methods"]]
            self.assertEqual(comparable[0], comparable[1])

    def test_rejects_unpaired_sequential_runs_called_paired(self):
        config = base_config()
        config["protocol"]["paired_repeats"] = True
        with self.assertRaisesRegex(experiment_core.ExperimentConfigError, "execution_blocks"):
            experiment_core.validate_config(config)

    def test_method_matrix_separates_trie_lookup_from_lng_coverage(self):
        config = base_config()
        trie = dict(config["methods"][0], name="ung",
                    entry_group_provider="cpu_min_super_sets")
        bitset = config["methods"][0]
        self.assertIn("label trie", experiment_core.method_semantics(config, trie)["entry_structure"])
        semantics = experiment_core.method_semantics(config, bitset)
        self.assertIn("bitsets", semantics["entry_structure"])
        self.assertIn("LNG descendant", semantics["coverage_structure"])

    def test_crossing_requires_every_repeat_at_threshold(self):
        rows = [
            {"Lsearch": "10", "Avg_Recall": "0.91"},
            {"Lsearch": "10", "Avg_Recall": "0.89"},
            {"Lsearch": "20", "Avg_Recall": "0.92"},
            {"Lsearch": "20", "Avg_Recall": "0.93"},
        ]
        crossing, _ = experiment_core.choose_recall_crossing(rows, 0.9, 2)
        self.assertEqual(crossing, 20)

    def test_crossing_excludes_declared_cold_repeat(self):
        rows = [
            {"Repeat": "0", "Lsearch": "10", "Avg_Recall": "0.89"},
            {"Repeat": "1", "Lsearch": "10", "Avg_Recall": "0.91"},
            {"Repeat": "2", "Lsearch": "10", "Avg_Recall": "0.92"},
        ]
        crossing, _ = experiment_core.choose_recall_crossing(
            rows, 0.9, 3, cold_repeats=1)
        self.assertEqual(crossing, 10)

    def test_summary_excludes_declared_cold_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with (run / "search_time_details.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["Repeat", "Lsearch", "Time_ms", "Avg_Recall"])
                writer.writeheader()
                writer.writerows([
                    {"Repeat": 0, "Lsearch": 10, "Time_ms": 100, "Avg_Recall": .91},
                    {"Repeat": 1, "Lsearch": 10, "Time_ms": 10, "Avg_Recall": .91},
                    {"Repeat": 2, "Lsearch": 10, "Time_ms": 12, "Avg_Recall": .91},
                ])
            summary, samples = experiment_core.summarize_case(
                run, threshold=.9, protocol=experiment_core.protocol_for(base_config()),
                expected_repeats=3)
            self.assertEqual(samples, [10.0, 12.0])
            self.assertEqual(summary["warm_median_ms"], 11.0)

    def test_partial_summary_records_missing_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = base_config()
            config["output_root"] = temporary
            rows = experiment_core.summarize_experiment(
                config, "plain", allow_partial=True)
            self.assertEqual(rows[0]["status"], "unavailable")
            self.assertIn("search_time_details.csv", rows[0]["reason"])

    def test_baseline_speedup_interval_is_exactly_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = base_config()
            config["output_root"] = temporary
            run = Path(temporary) / "plain" / "sel_1"
            run.mkdir(parents=True)
            with (run / "search_time_details.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["Repeat", "Lsearch", "Time_ms", "Avg_Recall"])
                writer.writeheader()
                writer.writerows([
                    {"Repeat": 0, "Lsearch": 10, "Time_ms": 100, "Avg_Recall": .91},
                    {"Repeat": 1, "Lsearch": 10, "Time_ms": 10, "Avg_Recall": .91},
                    {"Repeat": 2, "Lsearch": 10, "Time_ms": 12, "Avg_Recall": .91},
                ])
            row = experiment_core.summarize_experiment(config, "plain")[0]
            self.assertEqual(row["speedup_vs_baseline"], 1.0)
            self.assertEqual(row["speedup_ci95_low"], 1.0)
            self.assertEqual(row["speedup_ci95_high"], 1.0)

    def test_partial_formal_config_uses_common_completed_workloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            config_dir = repo / "experiments/multilevel_special"
            config_dir.mkdir(parents=True)
            source = base_config()
            source.update({
                "expected_source_fingerprint": "source",
                "expected_base_labels_sha256": "base",
                "expected_main_index_labels_sha256": "index",
                "dataset": "Amazon", "expected_num_queries": 1000,
                "require_stage_breakdown": True,
                "methods": [{
                    "name": "layer0_plain", "layer_count": 0,
                    "special_block_search": False,
                    "entry_group_provider": "cpu_bruteforce_els",
                    "lsearch_values_by_workload": {"sel_1": [100]},
                }],
            })
            (config_dir / "config.auto_policy_formal_exact_level.json").write_text(
                json.dumps(source))
            for method in ("ung_original_entry", "plain_bitset_lng_entry"):
                result = repo / "runs/ung_plain_crossing_amazon_x1" / method / "sel_1"
                result.mkdir(parents=True)
                with (result / "search_time_details.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=[
                        "Repeat", "Lsearch", "Time_ms", "Avg_Recall"])
                    writer.writeheader()
                    for repeat in range(3):
                        writer.writerow({"Repeat": repeat, "Lsearch": 100,
                                         "Time_ms": 1, "Avg_Recall": .91})
            config = generate_ung_plain_comparison.make_config(
                repo, "formal", allow_partial=True)
            self.assertEqual([item["name"] for item in config["workloads"]], ["sel_1"])
            self.assertTrue(all(method["enabled_workloads"] == ["sel_1"]
                                for method in config["methods"]))


if __name__ == "__main__":
    unittest.main()
