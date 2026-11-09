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

import run_rwalks_experiment


def write_fvecs(path: Path, rows: list[list[float]]) -> None:
    with path.open("wb") as stream:
        for row in rows:
            stream.write(struct.pack("<i", len(row)))
            stream.write(struct.pack(f"<{len(row)}f", *row))


class RWalksGroundTruthTest(unittest.TestCase):
    def test_converts_ung_pairs_to_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "groundtruth.bin"
            target = root / "ids.bin"
            source.write_bytes(struct.pack("<IfIfIfIf", 4, 0.1, 7, 0.2, 1, 0.3, 0xFFFFFFFF, 0.4))
            run_rwalks_experiment.prepare_rwalks_groundtruth_file(
                source, target, topk=2, num_queries=2)
            self.assertEqual(target.read_bytes(), struct.pack("<iiii", 4, 7, 1, -1))


class RWalksConfigTest(unittest.TestCase):
    def test_defaults_and_standard_layout(self):
        cfg = {"data_root": "/data", "result_root": "/results", "datasets": []}
        dataset = {"dataset": "Amazon", "query_task": "query_a"}
        method = {"name": "rwalks", "index_name": "RWalks"}
        paths = run_rwalks_experiment.resolve_dataset_files(cfg, dataset, method)
        build = run_rwalks_experiment.merged_build_cfg(cfg, dataset, method)
        search = run_rwalks_experiment.merged_search_cfg(cfg, dataset, method)
        self.assertEqual(paths["index_file"], Path("/results/Amazon/index/RWalks/index_files/index.bin"))
        self.assertEqual(paths["index_attr_file"], Path("/results/Amazon/index/RWalks/index_files/index_attr.bin"))
        self.assertEqual(
            paths["rwalks_gt_file"],
            Path(
                "/results/Amazon/results/rwalks/query_a_10_50_500/results/"
                "Amazon_rwalks_gt_ids.bin"
            ),
        )
        self.assertEqual((build["M"], build["ef_construction"], build["attr_depth"], build["attr_steps"]),
                         (16, 100, 3, 20))
        self.assertEqual((search["search_mode"], search["pruning_factor"]), (0, 0.0))

    def test_schema_is_union_of_configured_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data" / "Tiny"
            (data / "query_a").mkdir(parents=True)
            (data / "query_b").mkdir(parents=True)
            (data / "query_a" / "Tiny_query_labels.txt").write_text("1,3\n", encoding="utf-8")
            (data / "query_b" / "Tiny_query_labels.txt").write_text("2,3\n", encoding="utf-8")
            cfg = {
                "data_root": str(root / "data"), "result_root": str(root / "results"),
                "datasets": [
                    {"dataset": "Tiny", "query_task": "query_a"},
                    {"dataset": "Tiny", "query_task": "query_b"},
                ],
            }
            method = {"name": "rwalks", "index_name": "RWalks"}
            self.assertEqual(
                run_rwalks_experiment.configured_schema_labels(cfg, cfg["datasets"][0], method),
                [1, 2, 3],
            )


    def test_single_task_can_retain_original_index_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "Tiny"
            (data / "query_a").mkdir(parents=True)
            (data / "query_b").mkdir(parents=True)
            a = data / "query_a/Tiny_query_labels.txt"
            b = data / "query_b/Tiny_query_labels.txt"
            a.write_text("1,3\n")
            b.write_text("2,3\n")
            cfg = {"data_root": str(root), "result_root": str(root / "results"),
                   "datasets": [{"dataset": "Tiny", "query_task": "query_a"},
                                {"dataset": "Tiny", "query_task": "query_b"}]}
            method = {"name": "rwalks", "index_name": "RWalks"}
            original = run_rwalks_experiment.configured_schema_labels(cfg, cfg["datasets"][0], method)
            cfg["datasets"] = cfg["datasets"][:1]
            cfg["schema_query_label_files"] = [str(a), str(b)]
            retained = run_rwalks_experiment.configured_schema_labels(cfg, cfg["datasets"][0], method)
            self.assertEqual(retained, original)
            self.assertEqual(len(cfg["datasets"]), 1)


class RWalksRunnerTest(unittest.TestCase):
    def test_builds_two_file_index_then_uses_ung_result_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            result_root = root / "results"
            dataset_dir = data_root / "Tiny"
            query_dir = dataset_dir / "query_a"
            query_dir.mkdir(parents=True)
            write_fvecs(dataset_dir / "Tiny_base.fvecs", [[0.0, 0.0], [1.0, 1.0]])
            write_fvecs(query_dir / "Tiny_query.fvecs", [[0.0, 0.0]])
            (dataset_dir / "Tiny_base_labels.txt").write_text("1,2\n2\n", encoding="utf-8")
            (query_dir / "Tiny_query_labels.txt").write_text("1\n", encoding="utf-8")
            groundtruth = result_root / "Tiny" / "GroundTruth" / "query_a" / "Tiny_gt_labels_containment.bin"
            groundtruth.parent.mkdir(parents=True)
            groundtruth.write_bytes(struct.pack("<If", 0, 0.0))

            binary_dir = root / "bin"
            binary_dir.mkdir()
            build_binary = binary_dir / "build_index"
            search_binary = binary_dir / "search_sweep"
            build_binary.write_text("", encoding="utf-8")
            search_binary.write_text("", encoding="utf-8")
            config = {
                "data_root": str(data_root), "result_root": str(result_root),
                "rwalks_build_binary": str(build_binary),
                "rwalks_search_binary": str(search_binary),
                "auto_build_rwalks": False,
                "build": {"num_threads": 3, "M": 8, "ef_construction": 20,
                          "attr_depth": 2, "attr_steps": 4, "random_seed": 7},
                "search": {"K": 1, "num_repeats": 1, "num_threads": 2,
                           "lsearch_start": 10, "lsearch_end": 20, "lsearch_step": 10},
                "methods": [{"name": "rwalks", "index_name": "RWalks"}],
                "datasets": [{"dataset": "Tiny", "query_task": "query_a"}],
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            calls = []

            def fake_run(command, output_path=None, env=None):
                calls.append((command, output_path))
                if str(command[0]).endswith("build_index"):
                    index = Path(command[4])
                    attribute = index.with_name(index.stem + "_attr.bin")
                    index.write_bytes(b"index")
                    attribute.write_bytes(b"attributes")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(
                        "index_memory_bytes=4096 index_memory_mib=0.004 "
                        "index_level0_bytes=2048 index_upper_level_bytes=128 "
                        "index_runtime_metadata_bytes=896 index_attribute_bytes=1024\n",
                        encoding="utf-8",
                    )
                else:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text("Lsearch=10, time=1ms, avg_recall=1\n", encoding="utf-8")

            with mock.patch.object(run_rwalks_experiment, "run_command", side_effect=fake_run):
                with mock.patch.object(sys, "argv", ["run_rwalks_experiment.py", str(config_path)]):
                    self.assertEqual(run_rwalks_experiment.main(), 0)

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0][5:11], ["3", "8", "20", "2", "4", "7"])
            self.assertEqual(calls[1][0][-2:], ["10", "20"])
            index_root = result_root / "Tiny" / "index" / "RWalks"
            self.assertTrue((index_root / "index_files" / "index.bin").exists())
            self.assertTrue((index_root / "index_files" / "index_attr.bin").exists())
            self.assertTrue((index_root / "index_files" / "index.rwalks.json").exists())
            with (index_root / "others" / "index_build_time.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["index_memory_bytes"], "4096")
            self.assertEqual(
                calls[1][0][6],
                result_root / "Tiny" / "results" / "rwalks" / "query_a_10_10_20" / "results",
            )


if __name__ == "__main__":
    unittest.main()
