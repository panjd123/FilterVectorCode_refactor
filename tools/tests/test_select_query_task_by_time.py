#!/usr/bin/env python3
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = TOOLS_DIR / "sel_query" / "select_query_task_by_time.py"


def load_module():
    spec = importlib.util.spec_from_file_location("select_query_task_by_time", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SelectQueryTaskByTimeTest(unittest.TestCase):
    def write_detail_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True)
        columns = ["repeat", "Lsearch", "efs", "QueryID", "Time_ms", "Recall"]
        lines = [",".join(columns)]
        for row in rows:
            lines.append(",".join(str(row.get(column, "")) for column in columns))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_query_task(self, path: Path) -> None:
        path.mkdir(parents=True)
        (path / "Toy_query_labels.txt").write_text("q0\nq1\nq2\nq3\n", encoding="utf-8")
        with (path / "Toy_query.bin").open("wb") as f:
            f.write(struct.pack("<II", 4, 2))
            for value in range(8):
                f.write(struct.pack("<f", float(value)))
        with (path / "Toy_query.fvecs").open("wb") as f:
            for row in range(4):
                f.write(struct.pack("<i", 2))
                f.write(struct.pack("<ff", float(row * 2), float(row * 2 + 1)))

    def test_selects_all_matching_queries_after_first_lsearch_and_materializes_task(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_root = tmp / "results"
            data_root = tmp / "data"
            source_task = "query_kind"
            task_result = f"{source_task}_1000_1000_3000"
            self.write_query_task(data_root / "Toy" / source_task)

            common_rows = [
                {"Lsearch": 1000, "QueryID": 0, "Time_ms": 10, "Recall": 0.9},
                {"Lsearch": 2000, "QueryID": 0, "Time_ms": 40, "Recall": 0.9},
                {"Lsearch": 2000, "QueryID": 1, "Time_ms": 10, "Recall": 0.9},
                {"Lsearch": 3000, "QueryID": 2, "Time_ms": 10, "Recall": 0.9},
            ]
            self.write_detail_csv(
                results_root / "UNG__hybrid" / task_result / "results" / "query_details_repeat1.csv",
                [
                    {"Lsearch": 1000, "QueryID": 0, "Time_ms": 200, "Recall": 0.9},
                    {"Lsearch": 2000, "QueryID": 0, "Time_ms": 200, "Recall": 0.9},
                    {"Lsearch": 2000, "QueryID": 1, "Time_ms": 20, "Recall": 0.9},
                    {"Lsearch": 3000, "QueryID": 2, "Time_ms": 10, "Recall": 0.9},
                ],
            )
            self.write_detail_csv(
                results_root / "gpu_bruteforce_els_ung" / task_result / "results" / "query_details_repeat1.csv",
                common_rows,
            )
            self.write_detail_csv(
                results_root / "gpu_bruteforce_els_special_blocks" / task_result / "results" / "query_details_repeat1.csv",
                [
                    {"Lsearch": 1000, "QueryID": 0, "Time_ms": 5, "Recall": 0.9},
                    {"Lsearch": 2000, "QueryID": 0, "Time_ms": 100, "Recall": 0.9},
                    {"Lsearch": 2000, "QueryID": 1, "Time_ms": 3, "Recall": 0.9},
                    {"Lsearch": 3000, "QueryID": 2, "Time_ms": 20, "Recall": 0.9},
                ],
            )

            config_path = tmp / "config.json"
            output_dir = data_root / "Toy" / "query_selected"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": "Toy",
                        "results_root": str(results_root),
                        "query_data_root": str(data_root),
                        "output_task": "query_selected",
                        "overwrite": True,
                        "selections": [
                            {
                                "name": "hybrid_slow",
                                "source_task": source_task,
                                "query_result_task": task_result,
                                "faster_method": "gpu_bruteforce_els_ung",
                                "slower_method": "UNG__hybrid",
                                "min_speedup": 3.0,
                            },
                            {
                                "name": "special_blocks_fast",
                                "source_task": source_task,
                                "query_result_task": task_result,
                                "faster_method": "gpu_bruteforce_els_special_blocks",
                                "slower_method": "gpu_bruteforce_els_ung",
                                "min_speedup": 3.0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = module.run_from_config(config_path)

            self.assertEqual(manifest["num_queries"], 2)
            self.assertEqual((output_dir / "Toy_query_labels.txt").read_text(encoding="utf-8"), "q0\nq1\n")
            with (output_dir / "Toy_query.bin").open("rb") as f:
                self.assertEqual(struct.unpack("<II", f.read(8)), (2, 2))
                self.assertEqual(struct.unpack("<ffff", f.read(16)), (0.0, 1.0, 2.0, 3.0))
            selected_rows = (output_dir / "selected_queries.csv").read_text(encoding="utf-8").splitlines()
            self.assertIn("rule_name,source_task,source_query_id,matched_lsearch", selected_rows[0])
            self.assertEqual(len(selected_rows), 3)


    def test_rule_can_limit_count_and_override_result_task_suffix_per_method(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_root = tmp / "results"
            source_task = "query_kind"
            faster_task_result = f"{source_task}_fast_suffix"
            slower_task_result = f"{source_task}_slow_suffix"
            rows = [
                {"Lsearch": 1000, "QueryID": 0, "Time_ms": 10, "Recall": 0.9},
                {"Lsearch": 2000, "QueryID": 0, "Time_ms": 10, "Recall": 0.9},
                {"Lsearch": 2000, "QueryID": 1, "Time_ms": 5, "Recall": 0.9},
                {"Lsearch": 2000, "QueryID": 2, "Time_ms": 20, "Recall": 0.9},
            ]
            self.write_detail_csv(
                results_root / "fast" / faster_task_result / "results" / "query_details_repeat1.csv",
                rows,
            )
            self.write_detail_csv(
                results_root / "slow" / slower_task_result / "results" / "query_details_repeat1.csv",
                [
                    {"Lsearch": 1000, "QueryID": 0, "Time_ms": 100, "Recall": 0.9},
                    {"Lsearch": 2000, "QueryID": 0, "Time_ms": 100, "Recall": 0.9},
                    {"Lsearch": 2000, "QueryID": 1, "Time_ms": 100, "Recall": 0.9},
                    {"Lsearch": 2000, "QueryID": 2, "Time_ms": 100, "Recall": 0.9},
                ],
            )

            matches = module.choose_rule_matches(
                {
                    "dataset": "Toy",
                    "results_root": str(results_root),
                    "query_data_root": str(tmp / "data"),
                    "output_task": "unused",
                    "result_task_suffix": "_default_suffix",
                    "selections": [
                        {
                            "name": "limited",
                            "source_task": source_task,
                            "faster_result_task_suffix": "_fast_suffix",
                            "slower_result_task_suffix": "_slow_suffix",
                            "faster_method": "fast",
                            "slower_method": "slow",
                            "min_speedup": 2.0,
                            "max_queries": 2,
                        }
                    ],
                }
            )

            self.assertEqual([match.source_query_id for match in matches], [1, 0])
            self.assertEqual([round(match.speedup, 1) for match in matches], [20.0, 10.0])
            self.assertTrue(str(matches[0].faster_csv).endswith("query_kind_fast_suffix/results/query_details_repeat1.csv"))
            self.assertTrue(str(matches[0].slower_csv).endswith("query_kind_slow_suffix/results/query_details_repeat1.csv"))

    def test_rule_can_align_different_lsearch_values_by_ordinal_position(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_root = tmp / "results"
            source_task = "query_kind"
            task_result = f"{source_task}_results"
            self.write_detail_csv(
                results_root / "fast" / task_result / "results" / "query_details_repeat1.csv",
                [
                    {"Lsearch": 100, "QueryID": 0, "Time_ms": 10, "Recall": 1.0},
                    {"Lsearch": 200, "QueryID": 0, "Time_ms": 10, "Recall": 1.0},
                    {"Lsearch": 300, "QueryID": 0, "Time_ms": 10, "Recall": 1.0},
                ],
            )
            self.write_detail_csv(
                results_root / "slow" / task_result / "results" / "query_details_repeat1.csv",
                [
                    {"Lsearch": 1000, "QueryID": 0, "Time_ms": 100, "Recall": 1.0},
                    {"Lsearch": 2000, "QueryID": 0, "Time_ms": 50, "Recall": 1.0},
                    {"Lsearch": 3000, "QueryID": 0, "Time_ms": 30, "Recall": 1.0},
                ],
            )

            matches = module.choose_rule_matches(
                {
                    "dataset": "Toy",
                    "results_root": str(results_root),
                    "query_data_root": str(tmp / "data"),
                    "output_task": "unused",
                    "selections": [
                        {
                            "name": "ordinal",
                            "source_task": source_task,
                            "query_result_task": task_result,
                            "faster_method": "fast",
                            "slower_method": "slow",
                            "lsearch_alignment": "ordinal",
                            "min_speedup": 2.0,
                        }
                    ],
                }
            )

            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].matched_lsearch, 200)
            self.assertEqual(matches[0].slower_lsearch, 2000)
            self.assertEqual(matches[0].speedup, 5.0)


if __name__ == "__main__":
    unittest.main()
