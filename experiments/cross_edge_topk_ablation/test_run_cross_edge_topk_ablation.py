#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_cross_edge_topk_ablation as runner


class CrossEdgeTopkAblationRunnerTest(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "variants": [
                        {
                            "name": "ordinary_batch_matrix_sort",
                            "config": {
                                "run_name": "matrix",
                                "index_name": "idx_matrix",
                                "datasets": ["Reviews"],
                                "ung_env": {"UNG_GPU_TOPK_IMPL": "2"},
                            },
                        },
                        {
                            "name": "ordinary_batch_onchip_topk",
                            "config": {
                                "run_name": "onchip",
                                "index_name": "idx_onchip",
                                "datasets": ["Reviews"],
                                "ung_env": {"UNG_GPU_TOPK_IMPL": "3"},
                            },
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def test_materializes_selected_variants_and_plans_runner_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_config(root)
            generated_dir = root / "generated"

            steps = runner.plan_steps(
                config_path,
                generated_dir=generated_dir,
                only=["ordinary_batch_onchip_topk", "ordinary_batch_matrix_sort"],
            )

            self.assertEqual(
                [step.name for step in steps],
                ["ordinary_batch_onchip_topk", "ordinary_batch_matrix_sort"],
            )
            self.assertEqual(len(steps), 2)
            self.assertTrue((generated_dir / "ordinary_batch_onchip_topk.json").is_file())
            self.assertTrue((generated_dir / "ordinary_batch_matrix_sort.json").is_file())

            onchip_cfg = json.loads((generated_dir / "ordinary_batch_onchip_topk.json").read_text(encoding="utf-8"))
            self.assertEqual(onchip_cfg["ung_env"]["UNG_GPU_TOPK_IMPL"], "3")
            self.assertEqual(steps[0].command[-1], str(generated_dir / "ordinary_batch_onchip_topk.json"))

    def test_run_steps_stops_on_first_failure_by_default(self):
        steps = [
            runner.RunStep("first", ["false"]),
            runner.RunStep("second", ["true"]),
        ]
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return mock.Mock(returncode=1 if cmd == ["false"] else 0)

        with mock.patch.object(runner.subprocess, "run", side_effect=fake_run):
            rc = runner.run_steps(steps, continue_on_error=False)

        self.assertEqual(rc, 1)
        self.assertEqual(calls, [["false"]])


    def test_repeat_interleaves_and_suffixes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_config(root)
            generated_dir = root / "generated"

            steps = runner.plan_steps(config_path, generated_dir=generated_dir, repeat=2)

            self.assertEqual(
                [step.name for step in steps],
                [
                    "ordinary_batch_matrix_sort_r01",
                    "ordinary_batch_onchip_topk_r01",
                    "ordinary_batch_matrix_sort_r02",
                    "ordinary_batch_onchip_topk_r02",
                ],
            )
            matrix_r01 = json.loads((generated_dir / "ordinary_batch_matrix_sort_r01.json").read_text(encoding="utf-8"))
            self.assertEqual(matrix_r01["run_name"], "matrix_r01")
            self.assertEqual(matrix_r01["index_name"], "idx_matrix_r01")

    def test_taskset_and_stable_cpu_are_added_to_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_config(root)
            generated_dir = root / "generated"

            steps = runner.plan_steps(
                config_path,
                generated_dir=generated_dir,
                only=["ordinary_batch_matrix_sort"],
                cpu_list="0-15",
                stable_cpu=True,
            )

            self.assertEqual(steps[0].command[:3], ["taskset", "-c", "0-15"])
            self.assertEqual(steps[0].env["OMP_PROC_BIND"], "close")
            self.assertEqual(steps[0].env["OMP_DYNAMIC"], "FALSE")

    def test_default_config_keeps_ordinary_variants_aligned_with_ung_hybrid(self):
        config_path = Path(__file__).with_name("config.json")
        cfg = runner.load_config(config_path)
        variants = {variant["name"]: variant for variant in cfg["variants"]}

        self.assertIn("baseline_ung_hybrid_control", variants)
        self.assertEqual(
            variants["baseline_ung_hybrid_control"]["config"]["ung_env"],
            {
                "UNG_LNG_IMPL": "1",
                "UNG_DESCENDANTS_IMPL": "1",
            },
        )
        self.assertIn(
            "[UNG config] lng_impl=legacy_allocating",
            variants["baseline_ung_hybrid_control"]["expected_log_signals"],
        )
        self.assertIn(
            "[UNG config] descendants_impl=legacy_hash_bfs",
            variants["baseline_ung_hybrid_control"]["expected_log_signals"],
        )

        forbidden_non_cross_edge_env = {
            "UNG_GROUP_GRAPH_IMPL",
            "UNG_GET_MIN_SUPER_SETS_IMPL",
            "UNG_LNG_IMPL",
            "UNG_DESCENDANTS_IMPL",
            "UNG_COVERAGE_IMPL",
            "UNG_ADDITIONAL_EDGES_IMPL",
        }
        for name in ["ordinary_batch_matrix_sort", "ordinary_batch_onchip_topk"]:
            env = variants[name]["config"]["ung_env"]
            forbidden_present = forbidden_non_cross_edge_env & set(env)
            self.assertFalse(
                forbidden_present,
                f"{name} should only override cross-edge knobs, got {sorted(forbidden_present)}",
            )
            self.assertEqual(variants[name]["config"]["build"], cfg["shared"]["build"])


if __name__ == "__main__":
    unittest.main()
