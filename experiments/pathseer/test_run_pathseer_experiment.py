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

import run_pathseer_experiment


class PathSeerGroundTruthConversionTest(unittest.TestCase):
    def test_converts_ung_pairs_to_contiguous_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "groundtruth.bin"
            target = root / "pathseer_ids.bin"
            source.write_bytes(struct.pack("<IfIfIfIf", 4, 0.1, 7, 0.2, 1, 0.3, 0xFFFFFFFF, 0.4))

            run_pathseer_experiment.prepare_pathseer_groundtruth_file(
                source, target, topk=2, num_queries=2)

            self.assertEqual(target.read_bytes(), struct.pack("<iiii", 4, 7, 1, -1))


class PathSeerConfigTest(unittest.TestCase):
    def test_resolves_standard_project_layout_and_search_values(self):
        cfg = {
            "data_root": "/data",
            "result_root": "/results",
            "build": {"M": 64},
            "search": {
                "lsearch_start": 100,
                "lsearch_end": 300,
                "lsearch_step": 100,
            },
        }
        dataset_cfg = {"dataset": "Amazon", "query_task": "query_a"}
        method = {"name": "pathseer", "index_name": "PathSeer"}
        paths = run_pathseer_experiment.resolve_dataset_files(cfg, dataset_cfg, method)
        search_cfg = run_pathseer_experiment.merged_search_cfg(cfg, dataset_cfg, method)

        self.assertEqual(paths["base_fvecs"], Path("/data/Amazon/Amazon_base.fvecs"))
        self.assertEqual(paths["index_file"], Path("/results/Amazon/index/PathSeer/index_files/index.bin"))
        self.assertEqual(
            paths["pathseer_gt_file"],
            Path(
                "/results/Amazon/results/pathseer/query_a_100_100_300/results/"
                "Amazon_pathseer_gt_ids.bin"
            ),
        )
        self.assertEqual(search_cfg["M_exp_search"], 32)
        self.assertEqual(
            run_pathseer_experiment.search_values(cfg, dataset_cfg, method, search_cfg),
            [100, 200, 300],
        )


class PathSeerRunnerTest(unittest.TestCase):
    def test_build_and_search_use_adapter_and_ung_result_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            result_root = root / "results"
            adapter_build = root / "adapter_build"
            dataset_dir = data_root / "Tiny"
            query_dir = dataset_dir / "query_a"
            query_dir.mkdir(parents=True)
            adapter_build.mkdir()
            build_binary = adapter_build / "build_index"
            search_binary = adapter_build / "search_sweep"
            build_binary.write_text("", encoding="utf-8")
            search_binary.write_text("", encoding="utf-8")

            def write_fvecs(path: Path, rows: list[list[float]]) -> None:
                with path.open("wb") as stream:
                    for row in rows:
                        stream.write(struct.pack("<i", len(row)))
                        stream.write(struct.pack(f"<{len(row)}f", *row))

            write_fvecs(dataset_dir / "Tiny_base.fvecs", [[0.0, 0.0], [1.0, 1.0]])
            write_fvecs(query_dir / "Tiny_query.fvecs", [[0.0, 0.0]])
            (dataset_dir / "Tiny_base_labels.txt").write_text("1,2\n2\n", encoding="utf-8")
            (query_dir / "Tiny_query_labels.txt").write_text("1\n", encoding="utf-8")
            groundtruth = result_root / "Tiny" / "GroundTruth" / "query_a" / "Tiny_gt_labels_containment.bin"
            groundtruth.parent.mkdir(parents=True)
            groundtruth.write_bytes(struct.pack("<If", 0, 0.0))

            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "data_root": str(data_root),
                "result_root": str(result_root),
                "pathseer_build_binary": str(build_binary),
                "pathseer_search_binary": str(search_binary),
                "auto_build_pathseer": False,
                "build": {
                    "num_threads": 3,
                    "M": 8,
                    "M_exp": 16,
                    "ef_construction": 20,
                    "ef_construction_exp": 32,
                    "dist_fn": "L2",
                },
                "search": {
                    "K": 1,
                    "num_repeats": 1,
                    "num_threads": 2,
                    "M_exp_search": 4,
                    "lsearch_start": 10,
                    "lsearch_end": 20,
                    "lsearch_step": 10,
                },
                "methods": [{"name": "pathseer", "index_name": "PathSeer"}],
                "datasets": [{"dataset": "Tiny", "query_task": "query_a"}],
            }), encoding="utf-8")

            calls = []

            def fake_run(command, output_path=None, env=None):
                calls.append((command, output_path, env))
                if str(command[0]).endswith("build_index"):
                    index_file = Path(command[2])
                    index_file.parent.mkdir(parents=True, exist_ok=True)
                    index_file.write_bytes(b"index")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(
                        "index_memory_bytes=4096 index_memory_mib=0.004 "
                        "index_level0_bytes=2048 index_upper_level_bytes=0 "
                        "index_runtime_metadata_bytes=1024\n",
                        encoding="utf-8",
                    )
                else:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text("Lsearch=10, time=1ms, avg_recall=1\n", encoding="utf-8")

            with mock.patch.object(run_pathseer_experiment, "run_command", side_effect=fake_run):
                with mock.patch.object(sys, "argv", ["run_pathseer_experiment.py", str(config_path)]):
                    self.assertEqual(run_pathseer_experiment.main(), 0)

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0][0], build_binary)
            self.assertEqual(calls[0][0][3:9], ["3", "8", "16", "20", "32", "L2"])
            self.assertEqual(calls[1][0][0], search_binary)
            self.assertEqual(calls[1][0][-2:], ["10", "20"])
            self.assertIn("4", calls[1][0])
            self.assertTrue(
                (result_root / "Tiny" / "index" / "PathSeer" / "index_files" / "index.pathseer.json").exists()
            )
            with (result_root / "Tiny" / "index" / "PathSeer" / "others" / "index_build_time.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["index_memory_bytes"], "4096")
            self.assertEqual(
                calls[1][0][6],
                result_root / "Tiny" / "results" / "pathseer" / "query_a_10_10_20" / "results",
            )


if __name__ == "__main__":
    unittest.main()
