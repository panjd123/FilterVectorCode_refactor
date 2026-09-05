#!/usr/bin/env python3

import csv
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_acorn_baseline


class AcornSearchValuesTest(unittest.TestCase):
    def test_efs_values_prefers_explicit_values_over_range(self):
        self.assertEqual(
            run_acorn_baseline.efs_values(
                {
                    "efs_start": 100,
                    "efs_end": 400,
                    "efs_step": 100,
                    "efs_values": [100, 150, 300],
                }
            ),
            [100, 150, 300],
        )


class AcornIndexExistsTest(unittest.TestCase):
    def test_tiny_inverted_index_is_not_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            for name in ["acorn.index", "acorn1.index", "acorn.index.meta"]:
                (index_dir / name).write_text("index data\n", encoding="utf-8")
            (index_dir / "acorn.index.inverted_index").write_bytes((0).to_bytes(8, "little"))

            self.assertFalse(run_acorn_baseline.acorn_index_exists(index_dir))


class AcornDefaultConfigTest(unittest.TestCase):
    def test_default_config_runs_only_selected_recall_advantage_tasks(self):
        config_path = Path(__file__).with_name("config.json")
        cfg = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            cfg["datasets"],
            [
                {"dataset": "Reviews", "query_task": "query_selected_recall_advantage"},
                {"dataset": "Amazon", "query_task": "query_selected_recall_advantage"},
            ],
        )
        self.assertFalse(cfg.get("discover_datasets", False))



class AcornDatasetDiscoveryTest(unittest.TestCase):
    def _write_fvecs(self, path: Path) -> None:
        path.write_bytes(struct.pack("<Iff", 2, 0.0, 0.0))

    def test_discovers_nested_datasets_and_query_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "other-dataset" / "Nested"
            query_dir = data_dir / "query_A" / "sub_task"
            query_dir.mkdir(parents=True)
            self._write_fvecs(data_dir / "Nested_base.fvecs")
            (data_dir / "Nested_base_labels.txt").write_text("1,2\n", encoding="utf-8")
            self._write_fvecs(query_dir / "Nested_query.fvecs")
            (query_dir / "Nested_query_labels.txt").write_text("1\n", encoding="utf-8")

            discovered = run_acorn_baseline.discover_dataset_configs({"data_root": str(root / "data")})

            self.assertEqual(
                discovered,
                [
                    {
                        "dataset": "other-dataset/Nested",
                        "dataset_name": "Nested",
                        "query_task": "query_A/sub_task",
                    }
                ],
            )

    def test_nested_dataset_paths_use_leaf_name_for_files_and_safe_result_dir(self):
        cfg = {"data_root": "/data", "result_root": "/results"}
        search_cfg = {"efs_start": 10, "efs_step": 10, "efs_end": 30}
        paths = run_acorn_baseline.dataset_paths(
            cfg,
            {"dataset": "other-dataset/Nested", "dataset_name": "Nested", "query_task": "query_A/sub_task"},
            search_cfg,
        )

        self.assertEqual(paths["dataset"], "Nested")
        self.assertEqual(paths["base_fvecs"], Path("/data/other-dataset/Nested/Nested_base.fvecs"))
        self.assertEqual(paths["query_fvecs"], Path("/data/other-dataset/Nested/query_A/sub_task/Nested_query.fvecs"))
        self.assertEqual(
            paths["result_dir"],
            Path("/results/other-dataset__Nested/results/ACORN/query_A__sub_task_10_10_30/results"),
        )
        self.assertEqual(
            paths["gt_file"],
            Path("/results/other-dataset__Nested/GroundTruth/query_A/sub_task/Nested_gt_labels_containment.bin"),
        )


class AcornBaselineRunnerTest(unittest.TestCase):
    def _write_fvecs(self, path: Path) -> None:
        path.write_bytes(struct.pack("<Iff", 2, 0.0, 0.0))

    def _write_config(self, root: Path, *, existing_index: bool = False) -> Path:
        data_root = root / "data"
        result_root = root / "results"
        acorn_build_dir = root / "build_acorn_rel"
        ung_build_dir = root / "build_ung_rel"
        dataset_dir = data_root / "Tiny"
        query_dir = dataset_dir / "query_a"
        dataset_dir.mkdir(parents=True)
        query_dir.mkdir()
        self._write_fvecs(dataset_dir / "Tiny_base.fvecs")
        (dataset_dir / "Tiny_base.bin").write_bytes(struct.pack("<IIff", 1, 2, 0.0, 0.0))
        (dataset_dir / "Tiny_base_labels.txt").write_text("1\n", encoding="utf-8")
        self._write_fvecs(query_dir / "Tiny_query.fvecs")
        (query_dir / "Tiny_query.bin").write_bytes(struct.pack("<IIff", 1, 2, 0.0, 0.0))
        (query_dir / "Tiny_query_labels.txt").write_text("1\n", encoding="utf-8")
        gt_file = result_root / "Tiny" / "GroundTruth" / "query_a" / "Tiny_gt_labels_containment.bin"
        gt_file.parent.mkdir(parents=True)
        gt_file.write_bytes(struct.pack("<If", 0, 0.0))
        if existing_index:
            index_dir = result_root / "Tiny" / "index" / "ACORN" / "index_files"
            index_dir.mkdir(parents=True)
            for name in ["acorn.index", "acorn1.index", "acorn.index.meta", "acorn.index.inverted_index"]:
                (index_dir / name).write_text("index data\n", encoding="utf-8")
        cfg_path = root / "config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "data_root": str(data_root),
                    "result_root": str(result_root),
                    "source_dir": str(Path.cwd() / "ACORN"),
                    "build_dir": str(acorn_build_dir),
                    "ung_source_dir": str(Path.cwd() / "UNG" / "codes"),
                    "ung_build_dir": str(ung_build_dir),
                    "build_jobs": 3,
                    "build": {"M": 32, "M_beta": 64, "gamma": 80, "num_threads": 7},
                    "search": {
                        "K": 1,
                        "num_repeats": 2,
                        "num_threads": 5,
                        "efs_start": 10,
                        "efs_end": 30,
                        "efs_step": 10,
                        "if_bfs_filter": True,
                    },
                    "datasets": [{"dataset": "Tiny", "query_task": "query_a"}],
                }
            ),
            encoding="utf-8",
        )
        return cfg_path

    def test_acorn_configure_passes_configured_blas_lapack_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build_acorn_rel"
            lib = root / "openblas" / "lib" / "libopenblas.so"
            lib.parent.mkdir(parents=True)
            lib.write_text("", encoding="utf-8")
            calls = []

            def fake_run(cmd, output_path=None, env=None):
                calls.append([str(x) for x in cmd])
                if len(cmd) >= 2 and str(cmd[0]) == "cmake" and str(cmd[1]) == "--build":
                    (build_dir / "demos").mkdir(parents=True, exist_ok=True)
                    (build_dir / "demos" / "test_acorn").write_text("", encoding="utf-8")

            with mock.patch.object(run_acorn_baseline, "run_command", side_effect=fake_run):
                run_acorn_baseline.ensure_acorn_binary(
                    {
                        "source_dir": str(root / "ACORN"),
                        "build_dir": str(build_dir),
                        "blas_lapack_library": str(lib),
                    }
                )

            configure = calls[0]
            self.assertIn(f"-DBLAS_LIBRARIES={lib}", configure)
            self.assertIn(f"-DLAPACK_LIBRARIES={lib}", configure)
            self.assertIn("-DBLA_VENDOR=OpenBLAS", configure)
            self.assertIn("-DBUILD_TESTING=OFF", configure)
            self.assertNotIn("-DBUILD_TESTING=ON", configure)

    def test_main_compiles_builds_missing_index_and_searches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = self._write_config(root)
            calls = []

            def fake_run(cmd, output_path=None, env=None):
                calls.append([str(x) for x in cmd])
                if len(cmd) >= 2 and str(cmd[0]) == "cmake" and str(cmd[1]) == "--build":
                    if "--target" in cmd and "test_acorn" in [str(x) for x in cmd]:
                        (root / "build_acorn_rel" / "demos").mkdir(parents=True, exist_ok=True)
                        (root / "build_acorn_rel" / "demos" / "test_acorn").write_text("", encoding="utf-8")
                    else:
                        (root / "build_ung_rel" / "tools").mkdir(parents=True, exist_ok=True)
                        for rel in ["tools/compute_groundtruth", "tools/fvecs_to_bin"]:
                            (root / "build_ung_rel" / rel).write_text("", encoding="utf-8")
                if len(cmd) >= 2 and str(cmd[0]).endswith("test_acorn") and str(cmd[1]) == "build":
                    index_dir = Path(cmd[17]).parent
                    index_dir.mkdir(parents=True, exist_ok=True)
                    for name in ["acorn.index", "acorn1.index", "acorn.index.meta", "acorn.index.inverted_index"]:
                        (index_dir / name).write_text("index data\n", encoding="utf-8")
                if len(cmd) >= 2 and str(cmd[0]).endswith("test_acorn") and str(cmd[1]) == "search":
                    result_dir = Path(cmd[10])
                    result_dir.mkdir(parents=True, exist_ok=True)
                    (result_dir / "avg_Tiny_querya_M32_gamma80_threads5_repeat2_ifbfs1_efs10-30_10.csv").write_text(
                        "efs,acorn_Time_ms,acorn_QPS,acorn_Recall,acorn_n3_visited_avg,acorn_build_time_ms,acorn_index_size_MB,"
                        "acorn_1_Time_ms,acorn_1_QPS,acorn_1_Recall,acorn_1_n3_visited_avg,acorn_1_build_time_ms,acorn_1_index_size_MB\n"
                        "10,2,500,1,7,20,1,3,333,0.9,9,25,1\n",
                        encoding="utf-8",
                    )

            with mock.patch.object(run_acorn_baseline, "run_command", side_effect=fake_run):
                with mock.patch.object(sys, "argv", ["run_acorn_baseline.py", str(cfg_path)]):
                    self.assertEqual(run_acorn_baseline.main(), 0)

            self.assertEqual(calls[0][:2], ["cmake", "-S"])
            self.assertEqual(calls[1][:2], ["cmake", "--build"])
            self.assertNotIn("utils", calls[1])
            acorn_build_calls = [cmd for cmd in calls if cmd[0].endswith("test_acorn") and cmd[1] == "build"]
            acorn_search_calls = [cmd for cmd in calls if cmd[0].endswith("test_acorn") and cmd[1] == "search"]
            self.assertEqual(len(acorn_build_calls), 1)
            self.assertEqual(len(acorn_search_calls), 1)
            self.assertEqual(acorn_build_calls[0][2:7], ["1", "80", "Tiny", "32", "64"])
            self.assertEqual(acorn_search_calls[0][16], "10,20,30")
            summary = root / "results" / "Tiny" / "results" / "ACORN" / "query_a_10_10_30" / "results" / "search_time_summary.csv"
            with summary.open(newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["Lsearch"], "10")
            self.assertEqual(row["Average_Recall"], "1")

            acorn1_summary = root / "results" / "Tiny" / "results" / "ACORN-1" / "query_a_10_10_30" / "results" / "search_time_summary.csv"
            with acorn1_summary.open(newline="", encoding="utf-8") as f:
                acorn1_row = next(csv.DictReader(f))
            self.assertEqual(acorn1_row["Lsearch"], "10")
            self.assertEqual(acorn1_row["Average_Time_ms"], "3")
            self.assertEqual(acorn1_row["Average_Recall"], "0.9")

            acorn1_qps = acorn1_summary.with_name("search_time_summary_qps.csv")
            with acorn1_qps.open(newline="", encoding="utf-8") as f:
                acorn1_qps_row = next(csv.DictReader(f))
            self.assertEqual(acorn1_qps_row["QPS"], "333")

    def test_existing_index_skips_build_but_still_searches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = self._write_config(root, existing_index=True)
            calls = []

            def fake_run(cmd, output_path=None, env=None):
                calls.append([str(x) for x in cmd])
                if len(cmd) >= 2 and str(cmd[0]) == "cmake" and str(cmd[1]) == "--build":
                    (root / "build_acorn_rel" / "demos").mkdir(parents=True, exist_ok=True)
                    (root / "build_acorn_rel" / "demos" / "test_acorn").write_text("", encoding="utf-8")
                    (root / "build_ung_rel" / "tools").mkdir(parents=True, exist_ok=True)
                    for rel in ["tools/compute_groundtruth", "tools/fvecs_to_bin"]:
                        (root / "build_ung_rel" / rel).write_text("", encoding="utf-8")
                if len(cmd) >= 2 and str(cmd[0]).endswith("test_acorn") and str(cmd[1]) == "search":
                    result_dir = Path(cmd[10])
                    result_dir.mkdir(parents=True, exist_ok=True)
                    (result_dir / "avg_Tiny.csv").write_text(
                        "efs,acorn_Time_ms,acorn_QPS,acorn_Recall,acorn_n3_visited_avg,acorn_build_time_ms,acorn_index_size_MB,"
                        "acorn_1_Time_ms,acorn_1_QPS,acorn_1_Recall,acorn_1_n3_visited_avg,acorn_1_build_time_ms,acorn_1_index_size_MB\n"
                        "10,2,500,1,7,20,1,3,333,0.9,9,25,1\n",
                        encoding="utf-8",
                    )

            with mock.patch.object(run_acorn_baseline, "run_command", side_effect=fake_run):
                with mock.patch.object(sys, "argv", ["run_acorn_baseline.py", str(cfg_path)]):
                    self.assertEqual(run_acorn_baseline.main(), 0)

            self.assertFalse(any(cmd[0].endswith("test_acorn") and cmd[1] == "build" for cmd in calls))
            self.assertTrue(any(cmd[0].endswith("test_acorn") and cmd[1] == "search" for cmd in calls))


if __name__ == "__main__":
    unittest.main()
