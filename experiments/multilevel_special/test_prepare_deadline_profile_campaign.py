#!/usr/bin/env python3

import csv
import tempfile
import unittest
from pathlib import Path

import prepare_deadline_profile_campaign as profile


def write_details(path: Path, recalls: dict[int, tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("Repeat", "Lsearch", "efs", "Time_ms", "Avg_Recall"),
            lineterminator="\n")
        writer.writeheader()
        for lsearch, values in recalls.items():
            for repeat, recall in enumerate(values):
                writer.writerow({
                    "Repeat": repeat, "Lsearch": lsearch, "efs": 0,
                    "Time_ms": 1, "Avg_Recall": recall,
                })


class DeadlineProfilePreparationTest(unittest.TestCase):
    def test_conservative_crossing_uses_all_warm_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "details.csv"
            write_details(path, {
                40: (0.99, 0.91, 0.89),
                100: (0.80, 0.90, 0.92),
                500: (0.70, 0.95, 0.96),
            })
            self.assertEqual(profile.conservative_crossing(path, 1, 2, 0.9), 100)

    def test_crossing_rejects_incomplete_repeat_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "details.csv"
            write_details(path, {40: (0.91, 0.92, 0.93)})
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows[:-1])
            with self.assertRaisesRegex(ValueError, "incomplete repeat grid"):
                profile.conservative_crossing(path, 1, 2, 0.9)

    def test_profile_uses_best_measured_point_without_crossing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "details.csv"
            write_details(path, {
                40: (0.99, 0.80, 0.82),
                100: (0.99, 0.85, 0.84),
                500: (0.99, 0.85, 0.86),
            })
            self.assertEqual(
                profile.profile_operating_point(path, 1, 2, 0.9),
                ("no_crossing_best_measured", 500),
            )

    def test_profile_keeps_only_baseline_and_automatic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.json"
            binary = root / "search"
            binary.write_bytes(b"binary")
            methods = [
                {"name": "plain", "selection_role": "zero_layer_baseline"},
                {"name": "auto", "selection_role": "predeclared_degree_ratio_hierarchy_v1"},
                {"name": "manual", "selection_role": "predeclared_manual_oracle_grid"},
            ]
            source = {
                "dataset": "D", "output_root": str(root / "performance_root"),
                "measurement_pass": "performance", "pass_subdirs": True,
                "protocol": {"cold_repeats": 1, "measured_repeats": 2},
                "recall_thresholds": {"w": 0.9},
                "workloads": [{"name": "w"}], "methods": methods,
                "lsearch_values": [40, 100],
            }
            source_path.write_text("{}\n")
            for method in ("plain", "auto"):
                write_details(
                    root / "performance_root/performance" / method / "w" /
                    "search_time_details.csv",
                    {40: (0.8, 0.89, 0.91), 100: (0.9, 0.91, 0.92)},
                )
            result, cases = profile.make_profile_config(
                source, source_path, binary, profile.sha256_file(binary),
                "commit", root / "profile_root")
            self.assertEqual([m["name"] for m in result["methods"]], ["plain", "auto"])
            self.assertTrue(result["require_work_breakdown"])
            self.assertEqual(result["protocol"]["phase"], "profile")
            self.assertEqual(result["lsearch_values"], [100])
            self.assertEqual(len(cases), 2)

    def test_profile_can_select_explicit_methods_and_workloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.json"
            binary = root / "search"
            binary.write_bytes(b"binary")
            source = {
                "dataset": "D", "output_root": str(root / "performance_root"),
                "measurement_pass": "performance", "pass_subdirs": True,
                "protocol": {"cold_repeats": 1, "measured_repeats": 2},
                "recall_thresholds": {"narrow": 0.9, "broad": 0.9},
                "workloads": [{"name": "narrow"}, {"name": "broad"}],
                "methods": [
                    {"name": "plain", "selection_role": "zero_layer_baseline"},
                    {"name": "trie", "selection_role": "manual_grid"},
                    {"name": "unused", "selection_role": "manual_grid"},
                ],
                "lsearch_values": [100],
            }
            source_path.write_text("{}\n")
            for method in ("plain", "trie"):
                write_details(
                    root / "performance_root/performance" / method / "broad" /
                    "search_time_details.csv",
                    {100: (0.8, 0.91, 0.92)},
                )

            result, cases = profile.make_profile_config(
                source, source_path, binary, profile.sha256_file(binary),
                "commit", root / "profile_root",
                selected_method_names={"plain", "trie"},
                selected_workload_names={"broad"},
            )

            self.assertEqual([m["name"] for m in result["methods"]], ["plain", "trie"])
            self.assertEqual([w["name"] for w in result["workloads"]], ["broad"])
            self.assertEqual(result["recall_thresholds"], {"broad": 0.9})
            self.assertEqual(2, len(cases))


if __name__ == "__main__":
    unittest.main()
