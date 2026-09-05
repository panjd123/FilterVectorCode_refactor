#!/usr/bin/env python3
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = TOOLS_DIR / "datasets" / "analyze_ung_min_superset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_ung_min_superset", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AnalyzeUngMinSupersetTest(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        columns = ["repeat", "Lsearch", "efs", "QueryID", "Time_ms", "MinSupersetT_ms"]
        lines = [",".join(columns)]
        for row in rows:
            lines.append(",".join(str(row.get(column, "")) for column in columns))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_summarizes_gpu_smaller_and_large_share_by_lsearch(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gpu_path = tmp / "gpu.csv"
            ung_path = tmp / "ung.csv"
            self.write_csv(
                gpu_path,
                [
                    {"repeat": 0, "Lsearch": 1000, "efs": 0, "QueryID": 1, "Time_ms": 100, "MinSupersetT_ms": 20},
                    {"repeat": 0, "Lsearch": 1000, "efs": 0, "QueryID": 2, "Time_ms": 100, "MinSupersetT_ms": 10},
                    {"repeat": 0, "Lsearch": 2000, "efs": 0, "QueryID": 1, "Time_ms": 100, "MinSupersetT_ms": 15},
                    {"repeat": 0, "Lsearch": 2000, "efs": 0, "QueryID": 3, "Time_ms": 100, "MinSupersetT_ms": 50},
                ],
            )
            self.write_csv(
                ung_path,
                [
                    {"repeat": 0, "Lsearch": 1000, "efs": 0, "QueryID": 1, "Time_ms": 100, "MinSupersetT_ms": 60},
                    {"repeat": 0, "Lsearch": 1000, "efs": 0, "QueryID": 2, "Time_ms": 100, "MinSupersetT_ms": 11},
                    {"repeat": 0, "Lsearch": 2000, "efs": 0, "QueryID": 1, "Time_ms": 100, "MinSupersetT_ms": 40},
                    {"repeat": 0, "Lsearch": 2000, "efs": 0, "QueryID": 3, "Time_ms": 100, "MinSupersetT_ms": 30},
                ],
            )

            summary, details = module.analyze(gpu_path, ung_path, min_share=0.30, share_side="ung")

        by_lsearch = {row.lsearch: row for row in summary}
        self.assertEqual(by_lsearch[1000].matched_queries, 2)
        self.assertEqual(by_lsearch[1000].gpu_smaller_queries, 2)
        self.assertEqual(by_lsearch[1000].gpu_smaller_large_share_queries, 1)
        self.assertEqual(by_lsearch[2000].matched_queries, 2)
        self.assertEqual(by_lsearch[2000].gpu_smaller_queries, 1)
        self.assertEqual(by_lsearch[2000].gpu_smaller_large_share_queries, 1)
        self.assertEqual(len(details), 2)
        self.assertEqual({row.query_id for row in details}, {1})

    def test_materializes_selected_queries_from_source_task(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_dir = tmp / "source"
            output_dir = tmp / "selected"
            source_dir.mkdir()
            (source_dir / "Toy_query_labels.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
            with (source_dir / "Toy_query.bin").open("wb") as f:
                f.write(struct.pack("<II", 4, 2))
                for value in range(8):
                    f.write(struct.pack("<f", float(value)))
            with (source_dir / "Toy_query.fvecs").open("wb") as f:
                for row in range(4):
                    f.write(struct.pack("<I", 2))
                    f.write(struct.pack("<ff", float(row * 2), float(row * 2 + 1)))

            details = [
                module.DetailRow(0, 1000, 0, 2, 100, 60, 0.60, 100, 10, 0.10, 6.0),
                module.DetailRow(0, 1000, 0, 0, 100, 40, 0.40, 100, 20, 0.20, 2.0),
            ]

            module.materialize_query_task(
                details,
                output_dir,
                source_dir=source_dir,
                dataset="Toy",
                gpu_csv=tmp / "gpu.csv",
                ung_csv=tmp / "ung.csv",
                min_share=0.30,
                share_side="ung",
                overwrite=False,
            )

            self.assertEqual((output_dir / "Toy_query_labels.txt").read_text(encoding="utf-8"), "c\na\n")
            with (output_dir / "Toy_query.bin").open("rb") as f:
                self.assertEqual(struct.unpack("<II", f.read(8)), (2, 2))
                self.assertEqual(struct.unpack("<ffff", f.read(16)), (4.0, 5.0, 0.0, 1.0))
            selected_rows = (output_dir / "selected_queries.csv").read_text(encoding="utf-8").splitlines()
            self.assertIn("new_query_id,source_query_id", selected_rows[0])
            self.assertIn("0,2", selected_rows[1])
            manifest = json.loads((output_dir / "selection_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["num_queries"], 2)


if __name__ == "__main__":
    unittest.main()
