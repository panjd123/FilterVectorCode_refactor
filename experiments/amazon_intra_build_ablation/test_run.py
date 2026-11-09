import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("amazon_intra_ablation", HERE / "run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class AblationTest(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads((HERE / "config.json").read_text())
        self.base = json.loads((HERE / self.settings["base_config"]).read_text())

    def test_variants_only_change_requested_factors(self):
        configs = [runner.effective_config(self.base, self.settings, variant)
                   for variant in self.settings["variants"]]
        factor_keys = {"UNG_SPECIAL_BLOCK_GPU_INTRA", "UNG_SPECIAL_BLOCK_GPU_INTRA_STRICT",
                       "UNG_SPECIAL_BLOCK_GPU_INTRA_REQUIRE_USED", "UNG_TAGORE_BATCH_STREAMS",
                       "UNG_TAGORE_OPTIMIZED_TOPK"}
        common = []
        for config, variant in zip(configs, self.settings["variants"]):
            env = config["ung_env"]
            self.assertEqual(env["UNG_TAGORE_OPTIMIZED_TOPK"], str(int(variant["optimized_topk"])))
            self.assertEqual(env["UNG_TAGORE_BATCH_STREAMS"], "4" if variant["load_balance"] else "1")
            self.assertEqual(env["UNG_FAST_GRNND_BATCH_EXACT_NX"], "0")
            self.assertTrue(config["build_ung"])
            self.assertFalse(config["reuse_existing_index"])
            common.append({key: value for key, value in env.items() if key not in factor_keys})
        self.assertTrue(all(item == common[0] for item in common))
        self.assertTrue(all(config["build"] == configs[0]["build"] for config in configs))

    def test_order_rotates_and_warmups_are_separate(self):
        variants = self.settings["variants"]
        steps = runner.plan(self.settings, variants, 3, 1)
        count = len(variants)
        self.assertEqual(len(steps), 4 * count)
        self.assertEqual([step[2]["name"] for step in steps[count:2 * count]], [v["name"] for v in variants])
        self.assertEqual(steps[2 * count][2]["name"], variants[1]["name"])

    def test_summary_excludes_warmups_and_uses_median(self):
        rows = []
        for phase, value in [("warmup", 1000), ("measured", 10), ("measured", 30), ("measured", 20)]:
            rows.append({"variant": "cpu", "phase": phase, "gpu_intra": 0,
                         "load_balance": 0, "optimized_topk": 0,
                         **{metric: value for metric in runner.METRICS}})
        summary = runner.summarize(rows)[0]
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["intra_ms_median"], 20)
        self.assertEqual(summary["intra_ms_max"], 30)

    def test_audit_rejects_fallback_wrong_mode_and_changed_partition(self):
        variant = self.settings["variants"][-1]
        config = runner.effective_config(self.base, self.settings, variant)
        meta = {key: "same" for key in runner.IDENTITY}
        meta.update(special_block_count="4", special_block_member_points="2000",
                    gpu_intra_requested_streams="4", gpu_intra_optimized_topk="1",
                    gpu_intra_fallback_blocks="0", gpu_intra_blocks="4",
                    cpu_intra_vamana_blocks="0", gpu_intra_points="2000")
        meta.update({field: "10" for field in runner.METRICS.values()})
        log = "[TagoreCuda][batch_balance]"
        identity = runner.validate(meta, config, variant, log)
        for key, value in [("gpu_intra_fallback_blocks", "1"),
                           ("gpu_intra_optimized_topk", "0"),
                           ("build_time(ms)", "nan"),
                           ("special_block_count", "5")]:
            invalid = {**meta, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate(invalid, config, variant, log, identity)
        with self.assertRaises(ValueError):
            runner.validate(meta, config, variant, "", identity)

    def test_dry_run_materializes_fresh_configs_and_lock_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "plan"
            self.assertEqual(runner.main(["--dry-run", "--repeat", "1", "--warmup", "0",
                                          "--run-root", str(root)]), 0)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(len(manifest["commands"]), len(self.settings["variants"]))
            self.assertTrue(all(cmd[:5] == ["gpulock", "perf", "--wait-gpu-idle", "0", "--"]
                                for cmd in manifest["commands"]))
            self.assertEqual(len(list(root.glob("measured_r01/*/config.json"))), len(self.settings["variants"]))
            with self.assertRaises(FileExistsError):
                runner.main(["--dry-run", "--run-root", str(root)])

    def test_executes_fresh_base_then_blocks_and_times_inside_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            config = runner.effective_config(self.base, self.settings, self.settings["variants"][-1])
            config_path = case / "config.json"
            config_path.write_text(json.dumps(config))
            with mock.patch.object(runner.subprocess, "run") as run, \
                 mock.patch.object(runner.time, "perf_counter", side_effect=[1.0, 4.0, 4.1, 9.0]), \
                 mock.patch.dict(runner.os.environ, {"UNG_SPECIAL_BLOCKS": "1"}):
                self.assertEqual(runner.execute_case(config_path), 0)
            self.assertEqual(len(run.call_args_list), 2)
            base_call, block_call = run.call_args_list
            self.assertTrue(base_call.args[0][0].endswith("build_UNG_index"))
            self.assertNotIn("UNG_SPECIAL_BLOCKS", base_call.kwargs["env"])
            self.assertTrue(block_call.args[0][0].endswith("build_special_block_index"))
            source_arg = block_call.args[0].index("--ung_index_path_prefix") + 1
            self.assertEqual(block_call.args[0][source_arg], str(case / "index_files"))
            timing = json.loads((case / "timing.json").read_text())
            self.assertEqual(timing["end_to_end_wall_ms"], 8000)
            self.assertEqual(timing["base_wall_ms"], 3000)
            self.assertAlmostEqual(timing["block_wall_ms"], 4900)

    def test_total_timer_includes_base_and_wall_scope_is_validated(self):
        timing = {"base_wall_ms": 3000, "block_wall_ms": 4900, "end_to_end_wall_ms": 8000}
        values = runner.pipeline_values({"index_time_add_rb(ms)": "2500"},
                                        {"build_time(ms)": "4700"}, timing)
        self.assertEqual(values["total_index_build_ms"], 7200)
        self.assertEqual(values["end_to_end_wall_ms"], 8000)
        with self.assertRaises(ValueError):
            runner.pipeline_values({"index_time_add_rb(ms)": "2500"},
                                    {"build_time(ms)": "4700"},
                                    {**timing, "end_to_end_wall_ms": 4900})

    def test_median_uses_paired_total_samples(self):
        rows = [{"variant": "cpu", "phase": "measured", "gpu_intra": 0,
                 "load_balance": 0, "optimized_topk": 0,
                 **{metric: 1 for metric in runner.METRICS},
                 "base_build_timer_ms": base, "total_index_build_ms": base + block}
                for base, block in [(1, 100), (2, 1), (100, 2)]]
        self.assertEqual(runner.summarize(rows)[0]["total_index_build_ms_median"], 101)

    def test_profile_counts_are_parsed_as_integers(self):
        log = "[special_edges][cpu_inter_profile] graph_pairs=379 graph_queries=2478374 graph_distance_computations=123456789\n"
        self.assertEqual(runner.profile_values(log)["graph_distance_computations"], 123456789)


if __name__ == "__main__":
    unittest.main()
