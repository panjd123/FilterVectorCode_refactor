import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("seven_dataset_ablation", HERE / "run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class SevenDatasetExperimentTest(unittest.TestCase):
    def setUp(self):
        self.settings = runner.load_settings(HERE / "config.json")

    def test_materialized_configs_cover_requested_matrix_and_isolate_factors(self):
        configs = runner.load_configs(self.settings, self.settings["datasets"], self.settings["variants"])
        self.assertEqual(len(configs), 28)
        self.assertNotIn("Amazon", self.settings["datasets"])
        self.assertEqual([(v["gpu_intra"], v["load_balance"], v["optimized_topk"])
                          for v in self.settings["variants"]],
                         [(False, False, False), (True, False, False), (True, True, False), (True, True, True)])
        for (dataset, name), config in configs.items():
            with self.subTest(dataset=dataset, variant=name):
                self.assertEqual(config, runner.make_config(self.settings, dataset,
                    next(v for v in self.settings["variants"] if v["name"] == name)))
                self.assertEqual(config["ung_env"]["UNG_SPECIAL_INTER_RESERVE"], "1")
                self.assertEqual(config["ung_env"]["UNG_HIERARCHY_LAYERS"], "500:trie")

    def test_eight_dataset_dual_optimization_profile(self):
        settings = runner.load_settings(HERE / "config_gpu_balance_topk_eight.json")
        self.assertEqual(settings["datasets"], ["Amazon", *self.settings["datasets"]])
        self.assertEqual([v["name"] for v in settings["variants"]], ["gpu_balance_topk"])
        configs = runner.load_configs(settings, settings["datasets"], settings["variants"])
        self.assertEqual(len(configs), 8)
        for config in configs.values():
            self.assertEqual(config["ung_env"]["UNG_TAGORE_BATCH_STREAMS"], "4")
            self.assertEqual(config["ung_env"]["UNG_TAGORE_OPTIMIZED_TOPK"], "1")
            self.assertTrue(config["build_ung"])
            self.assertFalse(config["reuse_existing_index"])
            self.assertNotIn("UNG_DESCENDANTS_IMPL", config["ung_env"])

    def test_first_three_eight_dataset_profile_has_24_samples(self):
        settings = runner.load_settings(HERE / "config_first_three_eight.json")
        self.assertEqual(len(settings["datasets"]), 8)
        self.assertIn("Amazon", settings["datasets"])
        self.assertEqual([v["name"] for v in settings["variants"]], ["cpu", "gpu", "gpu_balance"])
        configs = runner.load_configs(settings, settings["datasets"], settings["variants"])
        self.assertEqual(len(configs), 24)
        self.assertTrue(all(c["build_ung"] and not c["reuse_existing_index"] for c in configs.values()))

    def test_plan_uses_lock_and_independent_base_per_sample(self):
        with tempfile.TemporaryDirectory() as temp, \
             mock.patch.object(runner, "input_info", return_value={"points": 1, "dim": 1}):
            root = Path(temp) / "plan"
            self.assertEqual(runner.main(["--dry-run", "--run-root", str(root)]), 0)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(len(manifest["steps"]), 28)
            sources = set()
            for step in manifest["steps"]:
                self.assertEqual(step["command"][:5], ["gpulock", "perf", "--wait-gpu-idle", "0", "--"])
                case = root / step["case"]
                config = json.loads((case / "config.json").read_text())
                sources.add(config["ung_index_path_prefix"])
                self.assertTrue(config["build_ung"])
                self.assertFalse(config["reuse_existing_index"])
            self.assertEqual(len(sources), 28)
            with self.assertRaises(FileExistsError):
                runner.main(["--dry-run", "--run-root", str(root)])

    def test_summary_keeps_datasets_separate_and_excludes_warmups(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = {"datasets": ["Genome", "Reviews"], "variants": [{"name": "gpu"}], "steps": []}
            rows = {}
            for dataset, values in [("Genome", [10, 20, 30]), ("Reviews", [100, 200, 300])]:
                for i, value in enumerate([99999, *values]):
                    step = {"dataset": dataset, "case": f"{dataset}/{i}", "status": "complete"}
                    manifest["steps"].append(step)
                    rows[step["case"]] = {"dataset": dataset, "variant": "gpu",
                        "phase": "warmup" if i == 0 else "measured", "gpu_intra": 1,
                        "load_balance": 0, "optimized_topk": 0,
                        **{key: value for key in (*runner.common.METRICS, *runner.common.PIPELINE_METRICS)}}
            with mock.patch.object(runner, "collect", side_effect=lambda step, *a: (rows[step["case"]], {})):
                runner.write_results(root, manifest)
            with (root / "comparison.csv").open() as stream:
                summary = list(runner.csv.DictReader(stream))
            self.assertEqual([r["samples"] for r in summary], ["3", "3"])
            self.assertEqual([float(r["end_to_end_wall_ms_median"]) for r in summary], [20, 200])


if __name__ == "__main__":
    unittest.main()
