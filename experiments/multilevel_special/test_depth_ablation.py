import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_authoritative_campaign
import run_authoritative_build_campaign
import summarize_depth_ablation as depth
import continue_after_query


class DepthAblationTest(unittest.TestCase):
    def test_post_query_supervisor_uses_profile_manifest_and_serial_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "profile.json"
            config.write_text(json.dumps({
                "output_root": str(root / "formal"), "pass_subdirs": True,
                "measurement_pass": "profile",
            }))
            self.assertEqual(
                continue_after_query.profile_manifest_path(config),
                root / "formal/manifest_profile.json")
            command = continue_after_query.pipeline_command(
                root / "heldout.log", root / "build.log", True)
            self.assertIn("run_heldout_campaign.py", command)
            self.assertIn("--skip-generate", command)
            self.assertIn("run_authoritative_build_campaign.py", command)
            self.assertLess(
                command.index("run_heldout_campaign.py"),
                command.index("run_authoritative_build_campaign.py"))
            self.assertIn(" && ", command)

    def test_build_campaign_requires_completed_query_profile_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.json"
            with self.assertRaisesRegex(RuntimeError, "query profile"):
                run_authoritative_build_campaign.validate_query_gate(missing)
            profile = Path(temporary) / "profile.json"
            profile.write_text("{}")
            with mock.patch.object(run_authoritative_build_campaign, "run") as run:
                run_authoritative_build_campaign.validate_query_gate(profile)
            run.assert_called_once_with("validate_selection_sweep.py", str(profile))

    def test_campaign_summary_root_respects_measurement_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text(json.dumps({
                "output_root": "/runs/formal", "pass_subdirs": True,
                "measurement_pass": "performance",
            }))
            self.assertEqual(
                run_authoritative_campaign.pass_summary_root(config),
                Path("/runs/formal/summary/performance"))

    def test_campaign_finalization_rebuilds_aggregates_before_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            screen = root / "screen.json"
            formal = root / "formal.json"
            screen.write_text(json.dumps({
                "output_root": str(root / "screen"), "pass_subdirs": True,
                "measurement_pass": "performance",
            }))
            formal.write_text(json.dumps({
                "output_root": str(root / "formal"), "pass_subdirs": True,
                "measurement_pass": "performance",
            }))
            with mock.patch.object(run_authoritative_campaign, "run") as run:
                run_authoritative_campaign.finalize_outputs(screen, formal)
            commands = [call.args for call in run.call_args_list]
            self.assertEqual(commands[0][0], "summarize_selection_sweep.py")
            self.assertEqual(commands[0][1], str(screen))
            self.assertEqual(commands[1][0], "summarize_selection_sweep.py")
            self.assertEqual(commands[1][1], str(formal))
            self.assertEqual(commands[2][0], "plot_authoritative_recall_qps.py")
            self.assertEqual(commands[3][0], "summarize_depth_ablation.py")

    def test_crossing_oracles_use_minimum_l_then_fastest_method(self):
        rows = [
            {"method": "a", "workload": "w", "lsearch": 100,
             "recall_min": 0.89, "batch_ms_warm_median": 8.0,
             "qps_warm_median": 125.0},
            {"method": "a", "workload": "w", "lsearch": 200,
             "recall_min": 0.91, "batch_ms_warm_median": 10.0,
             "qps_warm_median": 100.0},
            {"method": "a", "workload": "w", "lsearch": 300,
             "recall_min": 0.94, "batch_ms_warm_median": 5.0,
             "qps_warm_median": 200.0},
            {"method": "b", "workload": "w", "lsearch": 150,
             "recall_min": 0.90, "batch_ms_warm_median": 7.0,
             "qps_warm_median": 140.0},
        ]
        crossings = depth.first_crossings(rows, {"w": 0.9})
        self.assertEqual(crossings[("a", "w")]["lsearch"], 200)
        self.assertEqual(
            depth.fastest_crossing(crossings, ["a", "b"], "w")["method"], "b")

    def test_global_method_requires_one_method_on_every_workload(self):
        crossings = {
            ("a", "x"): {"qps_warm_median": 10.0},
            ("a", "y"): {"qps_warm_median": 40.0},
            ("b", "x"): {"qps_warm_median": 30.0},
        }
        method, score = depth.global_method(crossings, ["a", "b"], ["x", "y"])
        self.assertEqual(method, "a")
        self.assertAlmostEqual(score, 20.0)

    def test_method_groups_keep_routing_separate(self):
        config = {"methods": [
            {"name": "plain", "hierarchy_layers": []},
            {"name": "one", "hierarchy_layers": [
                {"min_points": 1, "topology": "lng"}]},
            {"name": "drh", "hierarchy_layers": [
                {"min_points": 1, "topology": "lng"},
                {"min_points": 2, "topology": "trie"}],
             "selection_role": "predeclared_degree_ratio_hierarchy_v1",
             "entry_strategy": "optimized_lng"},
            {"name": "l2_t1024_16384_drh_upper", "hierarchy_layers": [
                {"min_points": 1, "topology": "lng"},
                {"min_points": 2, "topology": "trie"}],
             "selection_role": "automatic_upper_authorization_control",
             "entry_strategy": "optimized_lng", "routing_policy": "upper_authorized"},
        ]}
        groups = depth.method_groups(config, "plain")
        self.assertEqual(groups["best_one_layer"], ["one"])
        self.assertEqual(groups["best_two_layer"], ["drh"])
        self.assertEqual(groups["automatic_drh"], ["drh"])
        self.assertEqual(groups["automatic_routed"], ["l2_t1024_16384_drh_upper"])


if __name__ == "__main__":
    unittest.main()
