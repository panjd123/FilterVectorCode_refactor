#!/usr/bin/env python3

import json
import struct
import sys
import subprocess
import tempfile
import unittest
import csv
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_favor_experiment


class FavorLabelConversionTest(unittest.TestCase):
    def test_writes_containment_conditions_without_dense_attributes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_labels = root / "base_labels.txt"
            query_labels = root / "query_labels.txt"
            attribute_file = root / "attribute.txt"
            condition_file = root / "conditions.txt"
            base_labels.write_text("1,2\n2,3\n\n", encoding="utf-8")
            query_labels.write_text("2\n1,3\n\n", encoding="utf-8")

            run_favor_experiment.prepare_favor_condition_file(query_labels, condition_file)

            self.assertFalse(attribute_file.exists())
            self.assertEqual(
                condition_file.read_text(encoding="utf-8").splitlines(),
                [
                    "label_2 == 1",
                    "label_1 == 1 AND label_3 == 1",
                    "label_empty == 1",
                ],
            )


class FavorMetricParsingTest(unittest.TestCase):
    def test_parses_ung_style_search_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "favor_search.log"
            log_path.write_text(
                "=== Repeat 1/1 ===\n"
                "Start querying ...\n"
                "  efs=100, time=93553.9ms, avg_recall=0.38562\n",
                encoding="utf-8",
            )

            self.assertEqual(
                run_favor_experiment.parse_favor_metrics(log_path),
                {"recall": 0.38562, "average_ms": 93553.9, "qps": 0.0},
            )


class FavorGroundTruthConversionTest(unittest.TestCase):
    def test_converts_ung_pair_groundtruth_to_favor_id_groundtruth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ung_gt = root / "ung_gt.bin"
            favor_gt = root / "favor_gt_ids.bin"
            ung_gt.write_bytes(
                struct.pack(
                    "<IfIfIfIf",
                    42,
                    0.25,
                    7,
                    1.5,
                    99,
                    2.25,
                    123,
                    3.5,
                )
            )

            run_favor_experiment.prepare_favor_groundtruth_file(ung_gt, favor_gt, topk=2, num_queries=2)

            self.assertEqual(favor_gt.read_bytes(), struct.pack("<iiii", 42, 7, 99, 123))


class FavorFilterBatchTest(unittest.TestCase):
    def test_avx2_single_eq_condition_uses_configured_attribute_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "check_batch_avx2_test.cpp"
            binary = root / "check_batch_avx2_test"
            include_dir = Path(__file__).resolve().parents[2] / "FAVOR" / "include"
            source.write_text('#include <iostream>\n#include <set>\n#include <vector>\n#include "check.h"\n\nint main() {\n    std::vector<float> rows = {\n        1.0f, 0.0f,\n        0.0f, 1.0f,\n        1.0f, 1.0f,\n        0.0f, 0.0f,\n        1.0f, 0.0f,\n        0.0f, 1.0f,\n        1.0f, 1.0f,\n        0.0f, 0.0f,\n    };\n    std::vector<FilterConditionWithId> conditions;\n    conditions.emplace_back(1, "==", std::set<float>{1.0f});\n    OptimizedFilter filter(conditions);\n    std::vector<size_t> candidates;\n    filter.checkBatchAVX2(\n        reinterpret_cast<const char*>(rows.data()),\n        2 * sizeof(float),\n        0,\n        0,\n        8,\n        candidates);\n    for (size_t candidate : candidates) {\n        std::cout << candidate << " ";\n    }\n    return 0;\n}\n', encoding="utf-8")
            subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-O2",
                    "-mavx2",
                    "-I",
                    str(include_dir),
                    str(source),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            result = subprocess.run([str(binary)], check=True, text=True, capture_output=True)
            self.assertEqual(result.stdout.strip(), "1 2 5 6")


class FavorRunnerTest(unittest.TestCase):
    def test_build_and_search_use_favor_binaries_and_result_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            result_root = root / "results"
            favor_build = root / "favor_build"
            dataset_dir = data_root / "Tiny"
            query_dir = dataset_dir / "query_a"
            dataset_dir.mkdir(parents=True)
            query_dir.mkdir()
            (favor_build / "app").mkdir(parents=True)
            (favor_build / "app" / "build_index").write_text("", encoding="utf-8")
            (favor_build / "app" / "search").write_text("", encoding="utf-8")
            (favor_build / "app" / "search_sweep").write_text("", encoding="utf-8")
            (root / "favor_src").mkdir()
            (dataset_dir / "Tiny_base.fvecs").write_bytes(b"base")
            (query_dir / "Tiny_query.fvecs").write_bytes(b"query")
            (dataset_dir / "Tiny_base_labels.txt").write_text("1\n", encoding="utf-8")
            (query_dir / "Tiny_query_labels.txt").write_text("1\n", encoding="utf-8")
            gt_file = result_root / "Tiny" / "GroundTruth" / "query_a" / "Tiny_gt_labels_containment.bin"
            gt_file.parent.mkdir(parents=True)
            gt_file.write_bytes(struct.pack("<If", 0, 0.0))
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "data_root": str(data_root),
                        "result_root": str(result_root),
                        "favor_source_dir": str(root / "favor_src"),
                        "favor_build_dir": str(favor_build),
                        "index_name": "FAVOR",
                        "build": {"num_threads": 7, "max_degree": 48, "Lbuild": 160},
                        "search": {
                            "K": 1,
                            "num_repeats": 1,
                            "num_threads": 7,
                            "lsearch_start": 64,
                            "lsearch_end": 128,
                            "lsearch_step": 64
                        },
                        "methods": [{"name": "favor", "index_name": "FAVOR"}],
                        "datasets": [{"dataset": "Tiny", "query_task": "query_a"}],
                    }
                ),
                encoding="utf-8",
            )

            calls = []

            def fake_run(cmd, output_path=None, env=None):
                calls.append((cmd, output_path))
                if str(cmd[0]) == "cmake":
                    return
                if str(cmd[0]).endswith("build_index"):
                    Path(cmd[3]).parent.mkdir(parents=True, exist_ok=True)
                    Path(cmd[3]).write_bytes(b"index")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(
                        "index_memory_bytes=1024 index_memory_mib=0.001 "
                        "index_level0_bytes=768 index_upper_level_bytes=128 "
                        "index_runtime_metadata_bytes=128\n",
                        encoding="utf-8",
                    )
                else:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(
                        "COMMAND\n\n=== Repeat 1/1 ===\nStart querying ...\n  efs=64, time=2.5ms, avg_recall=0.5\n",
                        encoding="utf-8",
                    )

            with mock.patch.object(run_favor_experiment, "run_command", side_effect=fake_run):
                with mock.patch.object(sys, "argv", ["run_favor_experiment.py", str(config_path)]):
                    self.assertEqual(run_favor_experiment.main(), 0)

            self.assertEqual(calls[0][0][:2], ["cmake", "-S"])
            self.assertEqual(calls[1][0][:2], ["cmake", "--build"])
            self.assertIn("build_index", calls[1][0])
            self.assertIn("search", calls[1][0])
            self.assertIn("search_sweep", calls[1][0])
            self.assertEqual(str(calls[2][0][0]), str(favor_build / "app" / "build_index"))
            self.assertEqual(str(calls[3][0][0]), str(favor_build / "app" / "search_sweep"))
            self.assertEqual(calls[3][0][-2:], ["64", "128"])
            self.assertTrue(
                (result_root / "Tiny" / "index" / "FAVOR" / "index_files" / "index.bin").exists()
            )
            build_time_csv = result_root / "Tiny" / "index" / "FAVOR" / "others" / "index_build_time.csv"
            with build_time_csv.open(newline="", encoding="utf-8") as fp:
                build_time_rows = list(csv.DictReader(fp))
            self.assertEqual(len(build_time_rows), 1)
            self.assertEqual(build_time_rows[0]["dataset"], "Tiny")
            self.assertEqual(build_time_rows[0]["index_name"], "FAVOR")
            self.assertEqual(
                build_time_rows[0]["index_file"],
                str(result_root / "Tiny" / "index" / "FAVOR" / "index_files" / "index.bin"),
            )
            self.assertGreaterEqual(float(build_time_rows[0]["build_time_seconds"]), 0.0)
            self.assertGreaterEqual(float(build_time_rows[0]["build_time_ms"]), 0.0)
            self.assertEqual(build_time_rows[0]["index_memory_bytes"], "1024")
            self.assertEqual(build_time_rows[0]["index_level0_bytes"], "768")
            search_calls = [call for call in calls if str(call[0][0]).endswith("search_sweep")]
            self.assertEqual(len(search_calls), 1)
            self.assertIn("--num_threads", search_calls[0][0])
            self.assertEqual(
                search_calls[0][0][search_calls[0][0].index("--num_threads") + 1],
                "7",
            )
            self.assertEqual(
                search_calls[0][0][5],
                result_root / "Tiny" / "favor" / "query_a" / "Tiny_favor_gt_ids.bin",
            )


    def test_prefilter_thresholds_expand_favor_result_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            result_root = root / "results"
            favor_build = root / "favor_build"
            dataset_dir = data_root / "Tiny"
            query_dir = dataset_dir / "query_a"
            dataset_dir.mkdir(parents=True)
            query_dir.mkdir()
            (favor_build / "app").mkdir(parents=True)
            (favor_build / "app" / "build_index").write_text("", encoding="utf-8")
            (favor_build / "app" / "search").write_text("", encoding="utf-8")
            (favor_build / "app" / "search_sweep").write_text("", encoding="utf-8")
            (dataset_dir / "Tiny_base.fvecs").write_bytes(b"base")
            (query_dir / "Tiny_query.fvecs").write_bytes(b"query")
            (dataset_dir / "Tiny_base_labels.txt").write_text("1\n", encoding="utf-8")
            (query_dir / "Tiny_query_labels.txt").write_text("1\n", encoding="utf-8")
            gt_file = result_root / "Tiny" / "GroundTruth" / "query_a" / "Tiny_gt_labels_containment.bin"
            gt_file.parent.mkdir(parents=True)
            gt_file.write_bytes(struct.pack("<If", 0, 0.0))
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "data_root": str(data_root),
                        "result_root": str(result_root),
                        "favor_build_dir": str(favor_build),
                        "auto_build_favor": False,
                        "search": {
                            "K": 1,
                            "num_repeats": 1,
                            "num_threads": 7,
                            "lsearch_start": 64,
                            "lsearch_end": 64,
                            "lsearch_step": 64,
                        },
                        "prefilter_selectivity_thresholds": [0.05, 0.1],
                        "datasets": [{"dataset": "Tiny", "query_task": "query_a"}],
                    }
                ),
                encoding="utf-8",
            )

            calls = []

            def fake_run(cmd, output_path=None, env=None):
                calls.append((cmd, output_path))
                if str(cmd[0]).endswith("build_index"):
                    Path(cmd[3]).parent.mkdir(parents=True, exist_ok=True)
                    Path(cmd[3]).write_bytes(b"index")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(
                        "index_memory_bytes=1024 index_memory_mib=0.001 "
                        "index_level0_bytes=768 index_upper_level_bytes=128 "
                        "index_runtime_metadata_bytes=128\n",
                        encoding="utf-8",
                    )
                else:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(
                        "COMMAND\n\n=== Repeat 1/1 ===\nStart querying ...\n"
                        "  efs=64, time=2.5ms, avg_recall=0.5\n",
                        encoding="utf-8",
                    )

            with mock.patch.object(run_favor_experiment, "run_command", side_effect=fake_run):
                with mock.patch.object(sys, "argv", ["run_favor_experiment.py", str(config_path)]):
                    self.assertEqual(run_favor_experiment.main(), 0)

            search_calls = [call for call in calls if str(call[0][0]).endswith("search_sweep")]
            self.assertEqual(len(search_calls), 2)
            self.assertEqual(search_calls[0][0][8], result_root / "Tiny" / "results" / "favor_0.05" / "query_a_64_64_64" / "results")
            self.assertEqual(search_calls[1][0][8], result_root / "Tiny" / "results" / "favor_0.1" / "query_a_64_64_64" / "results")
            self.assertIn("--prefilter_selectivity_threshold", search_calls[0][0])
            self.assertEqual(
                search_calls[0][0][search_calls[0][0].index("--prefilter_selectivity_threshold") + 1],
                "0.05",
            )
            self.assertEqual(
                search_calls[1][0][search_calls[1][0].index("--prefilter_selectivity_threshold") + 1],
                "0.1",
            )


if __name__ == "__main__":
    unittest.main()
