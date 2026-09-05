#!/usr/bin/env python3
import csv
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = TOOLS_DIR / "sel_query_baseline" / "select_query_by_recall_advantage.py"


def load_module():
    spec = importlib.util.spec_from_file_location("select_query_by_recall_advantage", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SelectQueryBaselineTest(unittest.TestCase):
    def write_detail_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True)
        columns = ["QueryID", "Lsearch", "Recall"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def write_csv(self, path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

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

    def test_selects_queries_by_first_recall_advantage_and_last_recall_floor(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_task_dir = tmp / "data" / "Toy" / "query_selected_time_advantage"
            output_dir = tmp / "data" / "Toy" / "query_selected_recall_advantage"
            results_root = tmp / "results"
            result_task = "query_selected_time_advantage_1000_1000_20000"
            special_csv = (
                results_root
                / "cpu_bruteforce_els_special_blocks"
                / result_task
                / "results"
                / "query_details_repeat1.csv"
            )
            favor_csv = results_root / "favor" / result_task / "results" / "query_details_repeat1.csv"

            self.write_query_task(input_task_dir)
            self.write_detail_csv(
                special_csv,
                [
                    {"QueryID": 0, "Lsearch": 1000, "Recall": 0.9},
                    {"QueryID": 0, "Lsearch": 2000, "Recall": 0.95},
                    {"QueryID": 1, "Lsearch": 1000, "Recall": 0.7},
                    {"QueryID": 1, "Lsearch": 2000, "Recall": 0.9},
                    {"QueryID": 2, "Lsearch": 1000, "Recall": 1.0},
                    {"QueryID": 2, "Lsearch": 3000, "Recall": 0.79},
                    {"QueryID": 3, "Lsearch": 1000, "Recall": 0.8},
                    {"QueryID": 3, "Lsearch": 3000, "Recall": 0.8},
                ],
            )
            self.write_detail_csv(
                favor_csv,
                [
                    {"QueryID": 0, "Lsearch": 1000, "Recall": 0.9},
                    {"QueryID": 1, "Lsearch": 1000, "Recall": 0.8},
                    {"QueryID": 2, "Lsearch": 1000, "Recall": 0.5},
                    {"QueryID": 3, "Lsearch": 1000, "Recall": 0.7},
                ],
            )

            config_path = tmp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": "Toy",
                        "input_task_dir": str(input_task_dir),
                        "results_root": str(results_root),
                        "query_result_task": result_task,
                        "output_task": "query_selected_recall_advantage",
                        "min_last_special_recall": 0.8,
                        "overwrite": True,
                    }
                ),
                encoding="utf-8",
            )

            manifest = module.run_from_config(config_path)

            self.assertEqual(manifest["num_queries"], 2)
            self.assertEqual(manifest["output_dir"], str(output_dir))
            self.assertEqual((output_dir / "Toy_query_labels.txt").read_text(encoding="utf-8"), "q0\nq3\n")
            with (output_dir / "Toy_query.bin").open("rb") as f:
                self.assertEqual(struct.unpack("<II", f.read(8)), (2, 2))
                self.assertEqual(struct.unpack("<ffff", f.read(16)), (0.0, 1.0, 6.0, 7.0))
            with (output_dir / "selected_queries.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([int(row["source_query_id"]) for row in rows], [0, 3])
            self.assertEqual(rows[0]["first_lsearch"], "1000")
            self.assertEqual(rows[0]["special_first_recall"], "0.9")
            self.assertEqual(rows[0]["favor_first_recall"], "0.9")
            self.assertEqual(rows[1]["special_last_recall"], "0.8")

    def test_selects_curator_time_advantage_and_repeats_fastest_to_target_count(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_task_dir = tmp / "data" / "Toy" / "query_selected_recall_advantage"
            output_dir = tmp / "data" / "Toy" / "query_selected_recall_curator_time_advantage"
            results_root = tmp / "results"
            result_task = "query_selected_recall_advantage_1000_1000_20000"
            special_csv = (
                results_root
                / "cpu_bruteforce_els_special_blocks"
                / result_task
                / "results"
                / "query_details_repeat1.csv"
            )
            curator_csv = results_root / "Curator" / result_task / "results" / "curator_results.csv"

            self.write_query_task(input_task_dir)
            self.write_csv(
                special_csv,
                ["QueryID", "Lsearch", "Recall", "Time_ms"],
                [
                    {"QueryID": 0, "Lsearch": 1000, "Recall": 0.97, "Time_ms": 2.0},
                    {"QueryID": 1, "Lsearch": 1000, "Recall": 0.96, "Time_ms": 1.0},
                    {"QueryID": 2, "Lsearch": 1000, "Recall": 0.94, "Time_ms": 0.5},
                    {"QueryID": 3, "Lsearch": 1000, "Recall": 0.99, "Time_ms": 10.0},
                ],
            )
            self.write_csv(
                curator_csv,
                ["search_ef", "QueryID", "Recall", "Search_Time_ms", "ResultIDs"],
                [
                    {"search_ef": 64, "QueryID": 0, "Recall": 0.96, "Search_Time_ms": 10.0, "ResultIDs": ""},
                    {"search_ef": 64, "QueryID": 1, "Recall": 0.96, "Search_Time_ms": 8.0, "ResultIDs": ""},
                    {"search_ef": 64, "QueryID": 2, "Recall": 0.99, "Search_Time_ms": 20.0, "ResultIDs": ""},
                    {"search_ef": 64, "QueryID": 3, "Recall": 0.99, "Search_Time_ms": 11.0, "ResultIDs": ""},
                ],
            )

            config_path = tmp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": "Toy",
                        "input_task_dir": str(input_task_dir),
                        "results_root": str(results_root),
                        "query_result_task": result_task,
                        "selection_mode": "curator_time_advantage",
                        "output_task": "query_selected_recall_curator_time_advantage",
                        "min_recall": 0.95,
                        "min_speedup": 2.0,
                        "target_num_queries": 5,
                        "overwrite": True,
                    }
                ),
                encoding="utf-8",
            )

            manifest = module.run_from_config(config_path)

            self.assertEqual(manifest["num_queries"], 5)
            self.assertEqual(manifest["unique_source_queries"], 2)
            self.assertEqual((output_dir / "Toy_query_labels.txt").read_text(encoding="utf-8"), "q1\nq0\nq1\nq0\nq1\n")
            with (output_dir / "Toy_query.bin").open("rb") as f:
                self.assertEqual(struct.unpack("<II", f.read(8)), (5, 2))
                self.assertEqual(
                    struct.unpack("<ffffffffff", f.read(40)),
                    (2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0),
                )
            with (output_dir / "selected_queries.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([int(row["source_query_id"]) for row in rows], [1, 0, 1, 0, 1])
            self.assertEqual(rows[0]["selection_mode"], "curator_time_advantage")
            self.assertEqual(rows[0]["speedup_vs_curator"], "8")
            self.assertEqual(rows[1]["speedup_vs_curator"], "5")



if __name__ == "__main__":
    unittest.main()
