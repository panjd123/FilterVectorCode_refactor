#!/usr/bin/env python3

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_navix_baseline


class NaviXSearchValuesTest(unittest.TestCase):
    def test_lsearch_values_prefers_explicit_values_over_range(self):
        self.assertEqual(
            run_navix_baseline.lsearch_values(
                {
                    "lsearch_start": 100,
                    "lsearch_end": 4000,
                    "lsearch_step": 100,
                    "lsearch_values": [100, 200, 1000, 2000, 3000, 4000],
                }
            ),
            [100, 200, 1000, 2000, 3000, 4000],
        )


class NaviXBaselineRunnerTest(unittest.TestCase):
    def _write_config(self, root: Path, *, existing_index: bool = False) -> Path:
        data_root = root / "data"
        result_root = root / "results"
        build_dir = root / "build_ung_rel"
        dataset_dir = data_root / "Tiny"
        query_dir = dataset_dir / "query_a"
        dataset_dir.mkdir(parents=True)
        query_dir.mkdir()
        (dataset_dir / "Tiny_base.bin").write_bytes(struct.pack("<IIff", 1, 2, 0.0, 0.0))
        (dataset_dir / "Tiny_base_labels.txt").write_text("1\n", encoding="utf-8")
        (dataset_dir / "Tiny_base_label_info.txt").write_text("", encoding="utf-8")
        (dataset_dir / "Tiny_base_label_tree_roots.txt").write_text("", encoding="utf-8")
        (query_dir / "Tiny_query.bin").write_bytes(struct.pack("<IIff", 1, 2, 0.0, 0.0))
        (query_dir / "Tiny_query_labels.txt").write_text("1\n", encoding="utf-8")
        gt_file = result_root / "Tiny" / "GroundTruth" / "query_a" / "Tiny_gt_labels_containment.bin"
        gt_file.parent.mkdir(parents=True)
        gt_file.write_bytes(struct.pack("<If", 0, 0.0))
        if existing_index:
            index_dir = result_root / "Tiny" / "index" / "NaviX" / "index_files"
            index_dir.mkdir(parents=True)
            for name in ["meta", "graph", "vecs.bin", "labels.txt"]:
                (index_dir / name).write_text("index_name=NaviX\n", encoding="utf-8")
        cfg_path = root / "config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "data_root": str(data_root),
                    "result_root": str(result_root),
                    "build_dir": str(build_dir),
                    "source_dir": str(Path.cwd() / "UNG" / "codes"),
                    "build_jobs": 3,
                    "build": {"num_threads": 7, "max_degree": 32, "Lbuild": 96},
                    "search": {
                        "K": 1,
                        "num_repeats": 2,
                        "num_threads": 5,
                        "num_entry_points": 16,
                        "lsearch_start": 10,
                        "lsearch_end": 30,
                        "lsearch_step": 10,
                        "efs_start": 1,
                        "efs_step_slow": 1,
                        "efs_step_fast": 1,
                        "lsearch_threshold": 10,
                    },
                    "datasets": [{"dataset": "Tiny", "query_task": "query_a"}],
                }
            ),
            encoding="utf-8",
        )
        return cfg_path

    def test_main_compiles_builds_missing_index_and_searches_with_force_alg_5(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = self._write_config(root)
            calls = []

            def fake_run(cmd, output_path=None, env=None):
                calls.append([str(x) for x in cmd])
                if len(cmd) >= 2 and str(cmd[0]) == "cmake" and str(cmd[1]) == "--build":
                    (root / "build_ung_rel" / "apps").mkdir(parents=True, exist_ok=True)
                    (root / "build_ung_rel" / "tools").mkdir(parents=True, exist_ok=True)
                    for rel in ["apps/build_UNG_index", "apps/search_UNG_index", "tools/compute_groundtruth", "tools/fvecs_to_bin"]:
                        (root / "build_ung_rel" / rel).write_text("", encoding="utf-8")
                if len(cmd) >= 3 and str(cmd[0]).endswith("build_UNG_index"):
                    out_dir = Path(cmd[cmd.index("--index_path_prefix") + 1])
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for name in ["meta", "graph", "vecs.bin", "labels.txt"]:
                        (out_dir / name).write_text("index_name=NaviX\n", encoding="utf-8")
                if len(cmd) >= 3 and str(cmd[0]).endswith("search_UNG_index"):
                    result_dir = Path(cmd[cmd.index("--result_path_prefix") + 1])
                    result_dir.mkdir(parents=True, exist_ok=True)
                    (result_dir / "search_time_summary.csv").write_text(
                        "Lsearch,Average_Efs,Average_Time_ms,Average_Recall\n10,0,2,1\n",
                        encoding="utf-8",
                    )

            with mock.patch.object(run_navix_baseline, "run_command", side_effect=fake_run):
                with mock.patch.object(sys, "argv", ["run_navix_baseline.py", str(cfg_path)]):
                    self.assertEqual(run_navix_baseline.main(), 0)

            self.assertEqual(calls[0][:2], ["cmake", "-S"])
            self.assertEqual(calls[1][:2], ["cmake", "--build"])
            self.assertIn("build_UNG_index", calls[1])
            self.assertIn("search_UNG_index", calls[1])
            build_calls = [cmd for cmd in calls if cmd[0].endswith("build_UNG_index")]
            search_calls = [cmd for cmd in calls if cmd[0].endswith("search_UNG_index")]
            self.assertEqual(len(build_calls), 1)
            self.assertEqual(build_calls[0][build_calls[0].index("--index_type") + 1], "NaviX")
            self.assertEqual(build_calls[0][build_calls[0].index("--max_degree") + 1], "32")
            self.assertEqual(search_calls[0][search_calls[0].index("--force_use_alg") + 1], "5")
            self.assertEqual(search_calls[0][search_calls[0].index("--Lsearch") + 1: search_calls[0].index("--is_new_method")], ["10", "20", "30"])

    def test_existing_index_skips_build_but_still_searches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = self._write_config(root, existing_index=True)
            calls = []

            def fake_run(cmd, output_path=None, env=None):
                calls.append([str(x) for x in cmd])
                if len(cmd) >= 2 and str(cmd[0]) == "cmake" and str(cmd[1]) == "--build":
                    (root / "build_ung_rel" / "apps").mkdir(parents=True, exist_ok=True)
                    (root / "build_ung_rel" / "tools").mkdir(parents=True, exist_ok=True)
                    for rel in ["apps/build_UNG_index", "apps/search_UNG_index", "tools/compute_groundtruth", "tools/fvecs_to_bin"]:
                        (root / "build_ung_rel" / rel).write_text("", encoding="utf-8")
                if len(cmd) >= 3 and str(cmd[0]).endswith("search_UNG_index"):
                    result_dir = Path(cmd[cmd.index("--result_path_prefix") + 1])
                    result_dir.mkdir(parents=True, exist_ok=True)
                    (result_dir / "search_time_summary.csv").write_text(
                        "Lsearch,Average_Efs,Average_Time_ms,Average_Recall\n10,0,2,1\n",
                        encoding="utf-8",
                    )

            with mock.patch.object(run_navix_baseline, "run_command", side_effect=fake_run):
                with mock.patch.object(sys, "argv", ["run_navix_baseline.py", str(cfg_path)]):
                    self.assertEqual(run_navix_baseline.main(), 0)

            self.assertFalse(any(cmd[0].endswith("build_UNG_index") for cmd in calls))
            self.assertTrue(any(cmd[0].endswith("search_UNG_index") for cmd in calls))


if __name__ == "__main__":
    unittest.main()
