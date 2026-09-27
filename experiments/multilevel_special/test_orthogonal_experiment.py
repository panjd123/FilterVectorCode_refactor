import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import advance_authoritative_campaign
import derive_static_hierarchy
import experiment_core
import generate_authoritative_campaign
import generate_authoritative_build_configs
import generate_authoritative_build_quality
import gpu_isolation
import run_base_topology_build
import run_selection_sweep
import summarize_authoritative_build
import summarize_selection_sweep


def orthogonal_config():
    return {
        "method_schema": "orthogonal_v2",
        "search_app": "/search",
        "data_root": "/data",
        "gt_root": "/gt",
        "output_root": "/out",
        "K": 10,
        "num_threads": 4,
        "num_entry_points": 16,
        "num_repeats": 3,
        "lsearch_values": [10],
        "protocol": {
            "phase": "screen", "cold_repeats": 1,
            "measured_repeats": 2, "recall_rule": "all_repeats",
        },
        "methods": [{
            "name": "mixed_three",
            "main_index": "/index/lng",
            "base_topology": "lng",
            "entry_strategy": "trie",
            "hierarchy_layers": [
                {"min_points": 1024, "topology": "trie"},
                {"min_points": 16384, "topology": "lng"},
                {"min_points": 262144, "topology": "trie"},
            ],
            "special_block_search": True,
            "block_index": "/blocks/mixed_three",
        }],
        "workloads": [{"name": "sel_1", "query_dir": "q"}],
        "recall_thresholds": {"sel_1": 0.9},
    }


class OrthogonalExperimentTest(unittest.TestCase):
    def test_accepts_arbitrary_depth_and_encodes_plan(self):
        config = orthogonal_config()
        experiment_core.validate_config(config)
        method = config["methods"][0]
        self.assertEqual(
            experiment_core.encode_hierarchy_layers(method),
            "1024:trie,16384:lng,262144:trie",
        )
        semantics = experiment_core.method_semantics(config, method)
        self.assertEqual(semantics["layer_count"], 3)
        self.assertEqual(semantics["base_topology"], "lng")
        self.assertEqual(semantics["entry_strategy"], "trie")
        self.assertEqual(semantics["routing_policy"], "always_layered")

    def test_rejects_non_increasing_thresholds(self):
        config = orthogonal_config()
        config["methods"][0]["hierarchy_layers"][1]["min_points"] = 1024
        with self.assertRaisesRegex(
                experiment_core.ExperimentConfigError, "strictly increasing"):
            experiment_core.validate_config(config)

    def test_rejects_legacy_provider_in_new_schema(self):
        config = orthogonal_config()
        config["methods"][0]["entry_group_provider"] = "cpu_bruteforce_els"
        with self.assertRaisesRegex(
                experiment_core.ExperimentConfigError, "entry_strategy"):
            experiment_core.validate_config(config)

    def test_performance_and_profile_env_are_separate(self):
        config = orthogonal_config()
        method = config["methods"][0]
        performance = run_selection_sweep.clean_method_env({}, config, method)
        self.assertEqual(performance["UNG_SPECIAL_LIGHT_STATS"], "1")
        self.assertNotIn("UNG_SPECIAL_PROFILE_TIMING", performance)
        self.assertEqual(
            performance["UNG_HIERARCHY_LAYERS"],
            "1024:trie,16384:lng,262144:trie",
        )
        config["measurement_pass"] = "profile"
        profile = run_selection_sweep.clean_method_env({}, config, method)
        self.assertEqual(profile["UNG_SPECIAL_LIGHT_STATS"], "0")
        self.assertEqual(profile["UNG_SPECIAL_PROFILE_TIMING"], "1")

    def test_profile_pass_has_an_explicit_profile_protocol_phase(self):
        formal = orthogonal_config()
        formal["protocol"]["phase"] = "formal"
        profile = advance_authoritative_campaign.make_profile(formal)
        experiment_core.validate_config(profile)
        self.assertEqual(profile["measurement_pass"], "profile")
        self.assertEqual(profile["protocol"]["phase"], "profile")

    def test_rejects_profile_pass_and_protocol_phase_mismatch(self):
        config = orthogonal_config()
        config["measurement_pass"] = "profile"
        with self.assertRaisesRegex(
                experiment_core.ExperimentConfigError, "must be declared together"):
            experiment_core.validate_config(config)

    def test_summary_exposes_arbitrary_hierarchy_dimensions(self):
        config = orthogonal_config()
        method = config["methods"][0]
        fields = summarize_selection_sweep.method_summary_fields(config, method)
        self.assertEqual(fields["layer_count"], 3)
        self.assertEqual(fields["thresholds"], "1024,16384,262144")
        self.assertEqual(fields["t1"], 1024)
        self.assertEqual(fields["t2"], 16384)
        self.assertEqual(fields["base_topology"], "lng")
        self.assertEqual(fields["layer_topologies"], "trie,lng,trie")
        self.assertEqual(fields["entry_strategy"], "trie")
        self.assertEqual(fields["routing_policy"], "always_layered")

    def test_build_case_env_overrides_shared_env(self):
        config = orthogonal_config()
        config["env"] = {
            "UNG_SPECIAL_BLOCK_GPU_INTRA": "0",
            "UNG_SPECIAL_BLOCK_GPU_INTER": "0",
        }
        method = config["methods"][0]
        method["env"] = {"UNG_SPECIAL_BLOCK_GPU_INTER": "1"}
        env = __import__("run_build_sweep").clean_build_env(
            {"PATH": os.environ.get("PATH", "")}, config, method, None)
        self.assertEqual(env["UNG_SPECIAL_BLOCK_GPU_INTRA"], "0")
        self.assertEqual(env["UNG_SPECIAL_BLOCK_GPU_INTER"], "1")

    def test_base_build_case_env_overrides_shared_profile(self):
        config = {"env": {"UNG_BUILD_PROFILE": "current_cpu"}}
        case = {
            "base_topology": "lng",
            "env": {"UNG_BUILD_PROFILE": "paper_fused"},
        }
        env = run_base_topology_build.clean_env(config, case, "lng")
        self.assertEqual(env["UNG_BUILD_PROFILE"], "paper_fused")
        self.assertEqual(env["UNG_BASE_GROUP_TOPOLOGY"], "lng")
        self.assertEqual(env["UNG_SPECIAL_BLOCKS"], "0")

    def test_build_campaign_separates_timing_and_resource_passes(self):
        timing = generate_authoritative_build_configs.make_hierarchy_config(5, False)
        resource = generate_authoritative_build_configs.make_hierarchy_config(5, True)
        self.assertFalse(timing["resource_profile"])
        self.assertTrue(resource["resource_profile"])
        self.assertEqual(len(timing["cases"]), 60)
        self.assertEqual(len(resource["cases"]), 10)
        self.assertEqual(
            {case["structure"] for case in timing["cases"]},
            {"single_t1024_lng", "auto_drh_v1"},
        )
        auto = [case for case in timing["cases"]
                if case["structure"] == "auto_drh_v1"]
        self.assertTrue(all(case["hierarchy_layers"] == [
            {"min_points": 1024, "topology": "lng"},
            {"min_points": 16384, "topology": "trie"},
        ] for case in auto))
        self.assertEqual(timing["gpu_isolation"]["device"], 0)
        self.assertEqual(
            timing["gpu_isolation"]["idle_consecutive_samples"], 3)

    def test_gpu_idle_policy_is_fail_closed(self):
        policy = dict(gpu_isolation.DEFAULT_POLICY)
        idle = {
            "utilization_percent": 0,
            "used_memory_mib": 3,
            "compute_applications": [],
        }
        self.assertTrue(gpu_isolation.snapshot_is_idle(idle, policy))
        self.assertFalse(gpu_isolation.snapshot_is_idle(
            {**idle, "utilization_percent": 1}, policy))
        self.assertFalse(gpu_isolation.snapshot_is_idle(
            {**idle, "compute_applications": [{"pid": 123}]}, policy))

    def test_gpu_profiles_require_isolation_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = {"benchmark_profile": "full_gpu"}
            with self.assertRaisesRegex(ValueError, "isolation evidence"):
                gpu_isolation.validate_case_evidence(case, root)
            (root / "gpu_isolation.json").write_text(
                '{"mode": "idle_preflight_no_lock"}\n')
            gpu_isolation.validate_case_evidence(case, root)

    def test_gpu_perf_lock_requires_the_declared_device(self):
        with mock.patch.dict(os.environ, {
                "GPULOCK_LOCK_MODE": "perf",
                "GPULOCK_LOCKED_DEVICES": "1",
        }, clear=False):
            self.assertFalse(gpu_isolation.has_perf_lock(0))
            self.assertTrue(gpu_isolation.has_perf_lock(1))

    def test_end_to_end_build_claim_uses_speedup_lower_bound(self):
        rows = []
        for repeat in range(3):
            rows.extend([
                {"component": "base", "structure": "zero_layer",
                 "profile": "original_cpu", "timing_role": "measured",
                 "repeat": repeat, "wall_seconds": 10.0},
                {"component": "base", "structure": "zero_layer",
                 "profile": "accelerated_gpu", "timing_role": "measured",
                 "repeat": repeat, "wall_seconds": 3.0},
                {"component": "hierarchy", "structure": "auto_drh_v1",
                 "profile": "full_gpu", "timing_role": "measured",
                 "repeat": repeat, "wall_seconds": 2.0},
            ])
        summary = summarize_authoritative_build.summarize_end_to_end(rows)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["speedup_vs_original_cpu"], 2.0)
        self.assertTrue(summary[0]["no_slower_supported"])

    def test_build_summary_excludes_resource_probe_from_timing(self):
        rows = []
        for wall in (10.0, 12.0):
            rows.append({
                "component": "base", "structure": "zero_layer",
                "profile": "original_cpu", "resource_profile": False,
                "timing_role": "measured", "wall_seconds": wall,
                "internal_seconds": wall - 1.0, "index_bytes": 1024,
                "gpu_required": False, "gpu_isolation_mode": "not_required",
                "gpu_exclusive_lock": False, "gpu_idle_samples": 0,
            })
        rows.append({
            "component": "base", "structure": "zero_layer",
            "profile": "original_cpu", "resource_profile": True,
            "timing_role": "measured", "wall_seconds": 100.0,
            "internal_seconds": 99.0, "index_bytes": 1024,
            "gpu_required": False, "gpu_isolation_mode": "not_required",
            "gpu_exclusive_lock": False, "gpu_idle_samples": 0,
            "peak_rss_mib": 512.0, "peak_gpu_memory_mib": 0.0,
        })
        summary = summarize_authoritative_build.summarize(rows)
        self.assertEqual(summary[0]["measured_repeats"], 2)
        self.assertEqual(summary[0]["wall_median_seconds"], 11.0)
        self.assertEqual(summary[0]["peak_rss_mib"], 512.0)

    def test_end_to_end_ignores_duplicate_resource_repeat(self):
        rows = [
            {"component": "base", "structure": "zero_layer",
             "profile": "original_cpu", "resource_profile": False,
             "timing_role": "measured", "repeat": 0,
             "wall_seconds": 10.0},
            {"component": "base", "structure": "zero_layer",
             "profile": "accelerated_gpu", "resource_profile": False,
             "timing_role": "measured", "repeat": 0,
             "wall_seconds": 3.0},
            {"component": "hierarchy", "structure": "auto_drh_v1",
             "profile": "full_gpu", "resource_profile": False,
             "timing_role": "measured", "repeat": 0,
             "wall_seconds": 2.0},
            {"component": "base", "structure": "zero_layer",
             "profile": "original_cpu", "resource_profile": True,
             "timing_role": "measured", "repeat": 0,
             "wall_seconds": 100.0},
            {"component": "base", "structure": "zero_layer",
             "profile": "accelerated_gpu", "resource_profile": True,
             "timing_role": "measured", "repeat": 0,
             "wall_seconds": 100.0},
            {"component": "hierarchy", "structure": "auto_drh_v1",
             "profile": "full_gpu", "resource_profile": True,
             "timing_role": "measured", "repeat": 0,
             "wall_seconds": 100.0},
        ]
        summary = summarize_authoritative_build.summarize_end_to_end(rows)
        self.assertEqual(summary[0]["paired_repeats"], 1)
        self.assertEqual(summary[0]["speedup_vs_original_cpu"], 2.0)

    def test_build_summary_preserves_gpu_isolation_limit(self):
        rows = [{
            "component": "hierarchy",
            "structure": "auto_drh_v1",
            "profile": "full_gpu",
            "timing_role": "measured",
            "wall_seconds": 2.0,
            "internal_seconds": 1.8,
            "index_bytes": 1024,
            "gpu_required": True,
            "gpu_isolation_mode": "idle_preflight_no_lock",
            "gpu_exclusive_lock": False,
            "gpu_idle_samples": 3,
        }]
        summary = summarize_authoritative_build.summarize(rows)
        self.assertEqual(summary[0]["gpu_isolation_mode"],
                         "idle_preflight_no_lock")
        self.assertFalse(summary[0]["gpu_exclusive_lock"])
        self.assertEqual(summary[0]["gpu_idle_samples_min"], 3)

    def test_zero_layer_cannot_load_block_index(self):
        config = orthogonal_config()
        method = config["methods"][0]
        method["hierarchy_layers"] = []
        method["special_block_search"] = False
        with self.assertRaisesRegex(
                experiment_core.ExperimentConfigError, "zero-layer"):
            experiment_core.validate_config(config)

    def test_authoritative_screen_covers_entry_strategy_factor(self):
        config = generate_authoritative_campaign.make_screen(
            Path(__file__).resolve().parent /
            "config.authoritative_amazon_hierarchy_grid.json")
        methods = config["methods"]
        self.assertEqual(len(methods), 44)
        self.assertEqual(config["minimum_successful_child_seconds"], 43200)
        self.assertEqual(
            [method["name"] for method in methods[:3]],
            ["l0_lng_entry_optimized_lng", "l0_trie_entry_trie",
             "l2_t1024_16384_lt_entry_optimized_lng"],
        )
        zero_layer = [method for method in methods
                      if not method["hierarchy_layers"]]
        self.assertEqual(len(zero_layer), 6)
        self.assertEqual(
            {(method["base_topology"], method["entry_strategy"])
             for method in zero_layer},
            {(topology, strategy)
             for topology in ("lng", "trie")
             for strategy in ("original", "optimized_lng", "trie")},
        )
        layered = [method for method in methods if method["hierarchy_layers"]]
        self.assertEqual(len(layered), 38)
        factorial = [method for method in layered
                     if method.get("routing_policy") is None]
        routed = [method for method in layered
                  if method.get("routing_policy") ==
                  "require_upper_authorization"]
        self.assertEqual(len(factorial), 36)
        self.assertEqual(len(routed), 2)
        self.assertTrue(all(
            method["env"]["UNG_SPECIAL_REQUIRE_UPPER_AUTHORIZATION"] == "1"
            for method in routed))
        self.assertTrue(all(
            experiment_core.method_semantics(config, method)["routing_policy"] ==
            "require_upper_authorization"
            for method in routed))
        counts = {strategy: 0 for strategy in experiment_core.ENTRY_STRATEGIES}
        for method in factorial:
            counts[method["entry_strategy"]] += 1
        self.assertEqual(set(counts.values()), {12})

    def test_static_hierarchy_is_predeclared_for_amazon(self):
        layers = derive_static_hierarchy.derive_plan(602453, 64, 4)
        self.assertEqual(
            [(row["min_points"], row["topology"]) for row in layers],
            [(1024, "lng"), (16384, "trie")],
        )
        config = generate_authoritative_campaign.make_screen(
            Path(__file__).resolve().parent /
            "config.authoritative_amazon_hierarchy_grid.json")
        automatic = [method for method in config["methods"]
                     if method.get("selection_role") ==
                     "predeclared_degree_ratio_hierarchy_v1"]
        self.assertEqual(len(automatic), 3)
        for method in automatic:
            self.assertEqual(
                [(row["min_points"], row["topology"])
                 for row in method["hierarchy_layers"]],
                [(1024, "lng"), (16384, "trie")],
            )

    def test_screen_uses_separate_zero_and_layered_lsearch_grids(self):
        config = generate_authoritative_campaign.make_screen(
            Path(__file__).resolve().parent /
            "config.authoritative_amazon_hierarchy_grid.json")
        methods = {method["name"]: method for method in config["methods"]}
        zero = methods["l0_lng_entry_optimized_lng"]["lsearch_values_by_workload"]
        zero_trie = methods["l0_trie_entry_trie"]["lsearch_values_by_workload"]
        layered = methods[
            "l2_t1024_16384_lt_entry_optimized_lng"
        ]["lsearch_values_by_workload"]
        routed = methods[
            "l2_t1024_16384_lt_entry_optimized_lng_upper_routed"
        ]["lsearch_values_by_workload"]
        self.assertEqual(zero["sel_0p5"], layered["sel_0p5"])
        self.assertGreater(max(zero["sel_60"]), max(layered["sel_60"]))
        self.assertGreater(max(zero["sel_99"]), max(layered["sel_99"]))
        self.assertLess(min(layered["sel_80"]), min(zero["sel_80"]))
        self.assertLess(len(routed["sel_95"]), len(layered["sel_95"]))
        self.assertLess(min(zero_trie["sel_80"]), min(zero["sel_80"]))
        self.assertEqual(max(zero_trie["sel_99"]), 602453)
        self.assertTrue(all(
            values == sorted(set(values)) and values[-1] <= 602453
            for table in (zero, layered) for values in table.values()))

    def test_resume_retains_exact_elapsed_evidence(self):
        record = {}
        experiment_core.retain_elapsed_evidence(
            record,
            {"started_at": "start", "finished_at": "finish",
             "elapsed_seconds": 12.5,
             "elapsed_source": "monotonic_child_wall", "returncode": 0},
            Path("/nonexistent"),
            "command.txt",
            "search.log",
        )
        self.assertEqual(record["elapsed_seconds"], 12.5)
        self.assertEqual(record["elapsed_source"], "monotonic_child_wall")

    def test_resume_labels_recovered_elapsed_as_approximate(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            command = run_dir / "command.txt"
            log = run_dir / "search.log"
            command.write_text("command\n")
            log.write_text("complete\n")
            os.utime(command, (100.0, 100.0))
            os.utime(log, (112.25, 112.25))
            record = {}
            experiment_core.retain_elapsed_evidence(
                record, {"status": "complete"}, run_dir,
                "command.txt", "search.log")
            self.assertEqual(record["elapsed_seconds"], 12.25)
            self.assertEqual(
                record["elapsed_source"], "artifact_mtime_approximation")

    def test_hierarchy_gpu_backend_evidence_rejects_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "build.log").write_text(
                "[special_edges] gpu_intra_enabled=1 gpu_intra_blocks=2 "
                "gpu_intra_points=9000 gpu_intra_fallback_blocks=1\n"
                "[special_edges] gpu_inter_enabled=1 gpu_inter_used=1 "
                "gpu_inter_ms=12.5\n")
            with self.assertRaisesRegex(ValueError, "fallback"):
                __import__("run_build_sweep").validate_backend_evidence(
                    {"name": "bad", "benchmark_profile": "full_gpu"},
                    root, {})

    def test_hierarchy_gpu_backend_evidence_accepts_wmma(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "build.log").write_text(
                "[special_edges] gpu_intra_enabled=1 gpu_intra_blocks=2 "
                "gpu_intra_points=9000 gpu_intra_fallback_blocks=0\n"
                "[special_gpu_inter] mode=tf32_wmma\n"
                "[special_edges] gpu_inter_enabled=1 gpu_inter_used=1 "
                "gpu_inter_ms=12.5\n")
            meta = {}
            __import__("run_build_sweep").validate_backend_evidence(
                {"name": "good", "benchmark_profile": "full_gpu_wmma"},
                root, meta)
            self.assertEqual(meta["verified_gpu_inter_used"], "1")
            self.assertEqual(meta["verified_wmma_inter"], "1")

    def test_phase_output_root_is_campaign_relative(self):
        advance = __import__("advance_authoritative_campaign")
        self.assertEqual(
            advance.phase_output_root(
                {"output_root": "/runs/build_quality_screen"}, "crossing"),
            "/runs/build_quality_crossing")
        self.assertEqual(
            advance.phase_output_root(
                {"output_root": "/runs/amazon_screen"}, "formal"),
            "/runs/amazon_formal")

    def test_derived_phase_reuses_source_binary_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = {
                "output_root": str(root), "pass_subdirs": True,
                "measurement_pass": "performance",
            }
            digest = "a" * 64
            snapshot = root / ".binary_snapshots" / f"search_UNG_index.{digest}"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(b"binary")
            digest = advance_authoritative_campaign.sha256_file(snapshot)
            renamed = snapshot.with_name(f"search_UNG_index.{digest}")
            snapshot.rename(renamed)
            source["expected_search_binary_sha256"] = digest
            manifest = {
                "runs": [{"status": "complete",
                          "search_binary_sha256": digest}]
            }
            (root / "manifest_performance.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            result = {"selection_provenance": {}}
            pinned, observed = advance_authoritative_campaign.pin_source_binary(
                source, result)
            self.assertEqual(pinned, renamed.resolve())
            self.assertEqual(observed, digest)
            self.assertEqual(result["search_app"], str(renamed.resolve()))
            self.assertEqual(result["expected_search_binary_sha256"], digest)

    def test_build_quality_hierarchy_uses_final_gated_routing(self):
        with mock.patch.object(
                generate_authoritative_build_quality,
                "labels_hash", return_value="labels-sha256"):
            method = generate_authoritative_build_quality.hierarchy_method(
                "auto_drh_v1", "full_gpu")
        self.assertEqual(
            method["routing_policy"], "require_upper_authorization")
        self.assertTrue(method["special_block_search"])

    def test_build_quality_uses_query_profile_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "search"
            binary.write_bytes(b"profile-binary")
            digest = experiment_core.sha256_file(binary)
            with mock.patch.object(
                    generate_authoritative_build_quality,
                    "labels_hash", return_value="labels-sha256"):
                config = generate_authoritative_build_quality.make_config(
                    binary, digest)
            self.assertEqual(config["search_app"], str(binary))
            self.assertEqual(config["expected_search_binary_sha256"], digest)

    def test_summarizer_reads_declared_measurement_pass(self):
        self.assertEqual(
            summarize_selection_sweep.measurement_root({
                "output_root": "/runs/amazon_screen",
                "pass_subdirs": True,
                "measurement_pass": "performance",
            }),
            Path("/runs/amazon_screen/performance"),
        )
        self.assertEqual(
            summarize_selection_sweep.measurement_root({
                "output_root": "/runs/amazon_screen",
            }),
            Path("/runs/amazon_screen"),
        )

    def test_crossing_grid_marks_exhausted_budget_unavailable(self):
        advance = __import__("advance_authoritative_campaign")
        self.assertIsNone(advance.crossing_grid(
            {100: 0.5, 1000: 0.8}, threshold=0.9, k=10,
            max_lsearch=1000))
        self.assertEqual(
            advance.crossing_grid(
                {100: 0.5, 1000: 0.8}, threshold=0.9, k=10,
                max_lsearch=2000)[-1],
            1500)

    def test_formal_excludes_a_case_without_measured_crossing(self):
        advance = __import__("advance_authoritative_campaign")
        with tempfile.TemporaryDirectory() as temporary:
            config = orthogonal_config()
            config["output_root"] = temporary
            config["pass_subdirs"] = False
            config["minimum_successful_child_seconds"] = 43200
            run_dir = Path(temporary) / "mixed_three" / "sel_1"
            run_dir.mkdir(parents=True)
            (run_dir / "search_time_details.csv").write_text(
                "Repeat,Lsearch,Time_ms,Avg_Recall\n"
                "0,10,1.0,0.70\n"
                "1,10,1.0,0.80\n"
                "2,10,1.0,0.81\n"
            )
            formal = advance.make_formal(config)
            self.assertEqual(
                formal["campaign_minimum_successful_child_seconds"], 43200)
            self.assertEqual(formal["minimum_successful_child_seconds"], 0)
            self.assertEqual(formal["methods"][0]["enabled_workloads"], [])
            excluded = formal["selection_provenance"]["excluded_no_crossing"]
            self.assertEqual(len(excluded), 1)
            self.assertEqual(excluded[0]["max_measured_lsearch"], 10)
            self.assertEqual(excluded[0]["recall_at_max_lsearch"], 0.8)


if __name__ == "__main__":
    unittest.main()
