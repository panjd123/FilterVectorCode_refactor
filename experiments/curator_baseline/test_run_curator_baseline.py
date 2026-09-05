#!/usr/bin/env python3

import csv
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_curator_baseline


class FakeBackend:
    built = []
    searched = []
    loaded = []
    saved = []

    def __init__(self):
        self.search_parameter = None
        self.search_budget = None

    @classmethod
    def from_config(cls, dim, cfg):
        backend = cls()
        backend.dim = dim
        backend.cfg = cfg
        return backend

    def build(self, base_vectors, base_ids):
        self.built.append((base_vectors.copy(), base_ids.copy()))

    def save(self, index_dir):
        self.saved.append(Path(index_dir))
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        (Path(index_dir) / "curator.index").write_text("fake", encoding="utf-8")

    def load(self, index_dir):
        self.loaded.append(Path(index_dir))

    def memory_usage_bytes(self):
        return 1234

    def set_search_budget(self, budget, parameter="search_ef"):
        self.search_budget = budget
        self.search_parameter = parameter

    def enable_stats_tracking(self, enable=True):
        self.stats_tracking = enable

    def search(self, query, k, qualified_ids):
        self.searched.append((self.search_parameter, self.search_budget, query.copy(), qualified_ids.copy()))
        if qualified_ids.size == 0:
            return np.full(k, -1, dtype=np.int64), {
                "qualified_labels_count": 0,
                "visited_points": 0,
                "visited_edges": 0,
                "distance_computations": 0,
            }
        out = np.full(k, -1, dtype=np.int64)
        out[0] = int(qualified_ids[0])
        return out, {
            "qualified_labels_count": int(qualified_ids.size),
            "visited_points": 3,
            "visited_edges": 4,
            "distance_computations": 5,
        }


class CuratorBaselineRunnerTest(unittest.TestCase):
    def setUp(self):
        FakeBackend.built.clear()
        FakeBackend.searched.clear()
        FakeBackend.loaded.clear()
        FakeBackend.saved.clear()

    def _write_config(self, root: Path) -> Path:
        data_root = root / "data"
        result_root = root / "results"
        dataset_dir = data_root / "Tiny"
        query_dir = dataset_dir / "query_a"
        dataset_dir.mkdir(parents=True)
        query_dir.mkdir()

        (dataset_dir / "Tiny_base.bin").write_bytes(
            struct.pack("<IIffffff", 3, 2, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0)
        )
        (dataset_dir / "Tiny_base_labels.txt").write_text("1,2\n2\n3\n", encoding="utf-8")
        (query_dir / "Tiny_query.bin").write_bytes(
            struct.pack("<IIffff", 2, 2, 0.0, 0.0, 2.0, 2.0)
        )
        (query_dir / "Tiny_query_labels.txt").write_text("1\n9\n", encoding="utf-8")
        gt_file = result_root / "Tiny" / "GroundTruth" / "query_a" / "Tiny_gt_labels_containment.bin"
        gt_file.parent.mkdir(parents=True)
        gt_file.write_bytes(struct.pack("<IfIf", 0, 0.0, 0xFFFFFFFF, 0.0))

        cfg_path = root / "config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "data_root": str(data_root),
                    "result_root": str(result_root),
                    "auto_compile": False,
                    "build": {"nlist": 4, "num_threads": 3},
                    "search": {
                        "K": 1,
                        "num_repeats": 1,
                        "num_threads": 2,
                        "search_parameter": "search_ef",
                        "ef_values": [2, 4],
                    },
                    "datasets": [{"dataset": "Tiny", "query_task": "query_a"}],
                }
            ),
            encoding="utf-8",
        )
        return cfg_path

    def test_main_runs_tiny_dataset_and_writes_project_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = self._write_config(root)

            qualified_calls = []
            original_qualified = run_curator_baseline.qualified_ids_for_containment_indexed

            def tracking_qualified(label_index, num_base, query_labels):
                qualified_calls.append(query_labels)
                return original_qualified(label_index, num_base, query_labels)

            with mock.patch.object(run_curator_baseline, "CuratorBackend", FakeBackend):
                with mock.patch.object(run_curator_baseline, "qualified_ids_for_containment_indexed", tracking_qualified):
                    with mock.patch.object(sys, "argv", ["run_curator_baseline.py", str(cfg_path)]):
                        self.assertEqual(run_curator_baseline.main(), 0)

            result_dir = (
                root
                / "results"
                / "Tiny"
                / "results"
                / "Curator"
                / "query_a_search_ef2_search_ef4"
                / "results"
            )
            summary = result_dir / "search_time_summary.csv"
            qps_summary = result_dir / "search_time_summary_qps.csv"
            result_file = result_dir / "curator_results.csv"
            self.assertTrue(summary.exists())
            self.assertTrue(qps_summary.exists())
            self.assertTrue(result_file.exists())
            build_time = root / "results" / "Tiny" / "index" / "Curator" / "build" / "build_time.csv"
            index_size = root / "results" / "Tiny" / "index" / "Curator" / "build" / "index_size.csv"
            meta = root / "results" / "Tiny" / "index" / "Curator" / "index_files" / "curator_meta.json"
            self.assertTrue(build_time.exists())
            self.assertTrue(index_size.exists())
            self.assertTrue(meta.exists())

            with summary.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            with result_file.open(newline="", encoding="utf-8") as f:
                result_rows = list(csv.DictReader(f))

            self.assertEqual(
                set(rows[0].keys()),
                {
                    "Lsearch",
                    "Recall",
                    "Average_Time_ms",
                    "Average_VisitedPoints",
                    "Average_VisitedEdges",
                    "Average_DistanceComputations",
                },
            )
            self.assertEqual([row["Lsearch"] for row in rows], ["2", "4"])
            self.assertEqual(sorted({row["search_ef"] for row in result_rows}), ["2", "4"])
            self.assertEqual(
                set(result_rows[0].keys()),
                {
                    "search_ef",
                    "QueryID",
                    "Recall",
                    "Search_Time_ms",
                    "VisitedPoints",
                    "VisitedEdges",
                    "DistanceComputations",
                    "ResultIDs",
                },
            )
            self.assertEqual(len(result_rows), 4)
            self.assertEqual(result_rows[0]["ResultIDs"], "0")
            self.assertAlmostEqual(float(result_rows[0]["Recall"]), 1.0)
            self.assertGreaterEqual(float(result_rows[0]["Search_Time_ms"]), 0.0)
            self.assertAlmostEqual(float(rows[0]["Recall"]), 0.5)
            self.assertAlmostEqual(float(rows[0]["Average_VisitedPoints"]), 1.5)
            self.assertAlmostEqual(float(rows[0]["Average_VisitedEdges"]), 2.0)
            self.assertAlmostEqual(float(rows[0]["Average_DistanceComputations"]), 2.5)
            self.assertEqual(FakeBackend.searched[0][0], "search_ef")
            self.assertEqual(FakeBackend.searched[0][1], 2)
            np.testing.assert_array_equal(FakeBackend.searched[0][3], np.array([0], dtype=np.uint32))
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "2")
            self.assertEqual(len(qualified_calls), 4)
            self.assertEqual(len(FakeBackend.searched), 4)
            self.assertEqual(FakeBackend.saved[0], root / "results" / "Tiny" / "index" / "Curator" / "index_files")

    def test_existing_index_metadata_loads_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = self._write_config(root)
            index_dir = root / "results" / "Tiny" / "index" / "Curator" / "index_files"
            index_dir.mkdir(parents=True)
            (index_dir / "curator.index").write_text("fake", encoding="utf-8")
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            build_cfg = run_curator_baseline.merged_build_cfg(cfg, cfg["datasets"][0])
            (index_dir / "curator_meta.json").write_text(
                json.dumps({
                    "index_name": "Curator",
                    "num_points": 3,
                    "dim": 2,
                    "build_ms": 123.456,
                    "memory_bytes": 999,
                    "disk_bytes": 888,
                    "build": build_cfg,
                    "persistent": True,
                }),
                encoding="utf-8",
            )

            with mock.patch.object(run_curator_baseline, "CuratorBackend", FakeBackend):
                with mock.patch.object(sys, "argv", ["run_curator_baseline.py", str(cfg_path)]):
                    self.assertEqual(run_curator_baseline.main(), 0)

            self.assertEqual(FakeBackend.built, [])
            self.assertEqual(FakeBackend.loaded[0], index_dir)
            build_time = root / "results" / "Tiny" / "index" / "Curator" / "build" / "build_time.csv"
            with build_time.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertAlmostEqual(float(rows[0]["Build Time (ms)"]), 123.456)
            self.assertEqual(rows[0]["Reused Existing Index"], "true")
            meta = json.loads((index_dir / "curator_meta.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(float(meta["build_ms"]), 123.456)


if __name__ == "__main__":
    unittest.main()
