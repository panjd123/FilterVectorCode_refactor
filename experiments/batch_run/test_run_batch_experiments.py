#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_batch_experiments as batch


class BatchExperimentRunnerTest(unittest.TestCase):
    def test_builds_expected_steps_and_filtered_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "batch.json"
            generated_dir = root / "generated"
            config_path.write_text(
                json.dumps(
                    {
                        "data_root": "/data",
                        "result_root": "/results",
                        "build_dir": "/build",
                        "generated_config_dir": str(generated_dir),
                        "datasets": [
                            {"dataset": "Reviews", "query_task": "reviews_task"},
                            {"dataset": "Amazon", "query_task": "amazon_task"},
                        ],
                        "methods": [
                            "UNG__hybrid",
                            "cpu_bruteforce_els_ung",
                            "cpu_bruteforce_els_special_blocks",
                            "favor",
                            "Navix",
                            "curator",
                        ],
                        "build": {"num_threads": 11, "max_degree": 48},
                        "search": {"K": 10, "num_threads": 22, "num_repeats": 2},
                        "method_search": {
                            "favor": {
                                "sweeps": [
                                    {"start": 100, "end": 1000, "step": 100},
                                    {"start": 2000, "end": 4000, "step": 1000},
                                ]
                            },
                            "Navix": {
                                "sweeps": [
                                    {"start": 100, "end": 1000, "step": 100},
                                    {"start": 2000, "end": 4000, "step": 1000},
                                ]
                            },
                            "search_comparison": {
                                "sweeps": [
                                    {"start": 100, "end": 1000, "step": 100},
                                    {"start": 2000, "end": 4000, "step": 1000},
                                ]
                            },
                        },
                        "overrides": {
                            "favor": {"auto_build_favor": False},
                            "search_comparison": {"search": {"K": 5}},
                        },
                    }
                ),
                encoding="utf-8",
            )

            cfg = batch.load_config(config_path)
            steps = batch.plan_steps(cfg, config_path)

            self.assertEqual(
                [step.name for step in steps],
                [
                    "UNG__hybrid",
                    "cpu_special_blocks_index",
                    "search_comparison_cpu_els",
                    "favor",
                    "Navix",
                    "curator",
                ],
            )
            self.assertEqual(len(steps), 6)

            batch.write_step_configs(steps)

            favor_cfg = json.loads(steps[3].config_path.read_text(encoding="utf-8"))
            self.assertEqual(favor_cfg["datasets"][0]["query_task"], "reviews_task")
            self.assertFalse(favor_cfg["auto_build_favor"])
            self.assertEqual(favor_cfg["build"]["num_threads"], 11)
            self.assertEqual(favor_cfg["build"]["max_degree"], 48)
            self.assertEqual(favor_cfg["search"]["num_threads"], 22)
            self.assertEqual(
                favor_cfg["ef_values"],
                [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000],
            )

            navix_cfg = json.loads(steps[4].config_path.read_text(encoding="utf-8"))
            self.assertEqual(navix_cfg["search"]["K"], 10)
            self.assertEqual(navix_cfg["search"]["num_repeats"], 2)
            self.assertEqual(
                navix_cfg["search"]["lsearch_values"],
                [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000],
            )

            ung_cfg = json.loads(steps[0].config_path.read_text(encoding="utf-8"))
            self.assertEqual(ung_cfg["datasets"], ["Reviews", "Amazon"])
            self.assertEqual(ung_cfg["index_name"], "UNG__hybrid")

            special_cfg = json.loads(steps[1].config_path.read_text(encoding="utf-8"))
            self.assertEqual(special_cfg["index_name"], "UNG_special_blocks_hybrid")

            search_cfg = json.loads(steps[2].config_path.read_text(encoding="utf-8"))
            self.assertEqual(search_cfg["search"]["K"], 5)
            self.assertEqual(
                search_cfg["search"]["lsearch_values"],
                [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000],
            )
            self.assertEqual(
                [method["name"] for method in search_cfg["methods"]],
                ["cpu_bruteforce_els_ung", "cpu_bruteforce_els_special_blocks"],
            )
            self.assertEqual(
                [method["index_name"] for method in search_cfg["methods"]],
                ["UNG__hybrid", "UNG_special_blocks_hybrid_bdeg128_bcross10"],
            )

            curator_cfg = json.loads(steps[5].config_path.read_text(encoding="utf-8"))
            self.assertEqual(curator_cfg["build"]["num_threads"], 11)
            for key in ["max_degree", "Lbuild", "alpha", "num_cross_edges", "M", "ef_construction"]:
                self.assertNotIn(key, curator_cfg["build"])

    def test_shell_index_steps_only_include_missing_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_root = root / "results"
            generated_dir = root / "generated"

            def write_index(dataset: str, index_name: str, *, special_blocks: bool = False) -> None:
                index_dir = result_root / dataset / "index" / index_name / "index_files"
                index_dir.mkdir(parents=True, exist_ok=True)
                for name in ["meta", "graph", "vecs.bin", "labels.txt"]:
                    (index_dir / name).write_text("ok", encoding="utf-8")
                if special_blocks:
                    for name in ["special_blocks.csv", "special_block_members.csv", "special_block_children.csv", "special_edges.csv"]:
                        (index_dir / name).write_text("ok", encoding="utf-8")

            write_index("Reviews", "UNG__hybrid")
            write_index("Reviews", "UNG_special_blocks_hybrid_bdeg128_bcross10", special_blocks=True)

            config_path = root / "batch.json"
            config_path.write_text(
                json.dumps(
                    {
                        "result_root": str(result_root),
                        "generated_config_dir": str(generated_dir),
                        "datasets": [
                            {"dataset": "Reviews", "query_task": "task"},
                            {"dataset": "Amazon", "query_task": "task"},
                        ],
                        "methods": ["UNG__hybrid", "cpu_bruteforce_els_special_blocks"],
                    }
                ),
                encoding="utf-8",
            )

            steps = batch.plan_steps(batch.load_config(config_path), config_path)
            batch.write_step_configs(steps)

            self.assertEqual([step.name for step in steps[:2]], ["UNG__hybrid", "cpu_special_blocks_index"])
            ung_cfg = json.loads(steps[0].config_path.read_text(encoding="utf-8"))
            special_cfg = json.loads(steps[1].config_path.read_text(encoding="utf-8"))
            self.assertEqual(ung_cfg["datasets"], ["Amazon"])
            self.assertEqual(special_cfg["datasets"], ["Amazon"])

    def test_shell_index_steps_are_omitted_when_all_indexes_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_root = root / "results"

            for index_name, special_blocks in [("UNG__hybrid", False), ("UNG_special_blocks_hybrid_bdeg128_bcross10", True)]:
                index_dir = result_root / "Reviews" / "index" / index_name / "index_files"
                index_dir.mkdir(parents=True, exist_ok=True)
                for name in ["meta", "graph", "vecs.bin", "labels.txt"]:
                    (index_dir / name).write_text("ok", encoding="utf-8")
                if special_blocks:
                    for name in ["special_blocks.csv", "special_block_members.csv", "special_block_children.csv", "special_edges.csv"]:
                        (index_dir / name).write_text("ok", encoding="utf-8")

            config_path = root / "batch.json"
            config_path.write_text(
                json.dumps(
                    {
                        "result_root": str(result_root),
                        "generated_config_dir": str(root / "generated"),
                        "datasets": [{"dataset": "Reviews", "query_task": "task"}],
                        "methods": ["UNG__hybrid", "cpu_bruteforce_els_special_blocks"],
                    }
                ),
                encoding="utf-8",
            )

            steps = batch.plan_steps(batch.load_config(config_path), config_path)
            batch.write_step_configs(steps)

            self.assertEqual([step.name for step in steps], ["search_comparison_cpu_els"])
            search_cfg = json.loads(steps[0].config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [method["index_name"] for method in search_cfg["methods"]],
                ["UNG_special_blocks_hybrid_bdeg128_bcross10"],
            )

    def test_search_indexes_follow_ung_and_special_build_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "batch.json"
            config_path.write_text(
                json.dumps(
                    {
                        "generated_config_dir": str(root / "generated"),
                        "datasets": [{"dataset": "Reviews", "query_task": "task"}],
                        "methods": ["UNG__hybrid", "cpu_bruteforce_els_ung", "cpu_bruteforce_els_special_blocks"],
                        "overrides": {
                            "UNG__hybrid": {"index_name": "UNG_custom"},
                            "cpu_special_blocks_index": {
                                "index_name": "UNG_special_custom",
                                "index_name_params": [
                                    {"source": "build", "key": "num_cross_edges", "label": "cross"}
                                ],
                            },
                        },
                        "build": {"num_cross_edges": 7},
                    }
                ),
                encoding="utf-8",
            )

            steps = batch.plan_steps(batch.load_config(config_path), config_path)
            batch.write_step_configs(steps)

            search_cfg = json.loads(steps[-1].config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [method["index_name"] for method in search_cfg["methods"]],
                ["UNG_custom", "UNG_special_custom_cross7"],
            )

    def test_can_disable_special_block_index_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "batch.json"
            config_path.write_text(
                json.dumps(
                    {
                        "generated_config_dir": str(root / "generated"),
                        "datasets": [{"dataset": "Reviews", "query_task": "task"}],
                        "methods": ["cpu_bruteforce_els_special_blocks"],
                        "ensure_special_blocks_index": False,
                    }
                ),
                encoding="utf-8",
            )

            steps = batch.plan_steps(batch.load_config(config_path), config_path)
            batch.write_step_configs(steps)

            self.assertEqual([step.name for step in steps], ["search_comparison_cpu_els"])
            search_cfg = json.loads(steps[0].config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [method["index_name"] for method in search_cfg["methods"]],
                ["UNG_special_blocks_hybrid_bdeg128_bcross10"],
            )


if __name__ == "__main__":
    unittest.main()
