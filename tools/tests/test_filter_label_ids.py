#!/usr/bin/env python3
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = TOOLS_DIR / "filter_label_ids.py"


def load_filter_module():
    spec = importlib.util.spec_from_file_location("filter_label_ids", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FilterLabelIdsTest(unittest.TestCase):
    def test_filter_file_keeps_only_labels_below_threshold_and_preserves_rows(self):
        module = load_filter_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "labels.txt"
            output_path = Path(tmpdir) / "labels_lt1000.txt"
            input_path.write_text(
                "1,2,1000,1001,999\n"
                "1000,2000\n"
                "5, 6, 1200\n",
                encoding="utf-8",
            )

            stats = module.filter_label_file(input_path, output_path, threshold=1000)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"), "1,2,999\n1,2,999\n5,6\n"
            )
            self.assertEqual(stats.lines, 3)
            self.assertEqual(stats.labels_seen, 10)
            self.assertEqual(stats.labels_kept, 5)
            self.assertEqual(stats.labels_removed, 5)
            self.assertEqual(stats.empty_lines_filled, 1)

    def test_filter_file_fills_empty_rows_with_previous_output_labels(self):
        module = load_filter_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "labels.txt"
            output_path = Path(tmpdir) / "labels_lt1000.txt"
            input_path.write_text(
                "\n"
                "1,1000,2\n"
                "1000,2000\n"
                "\n"
                "3,4\n",
                encoding="utf-8",
            )

            stats = module.filter_label_file(input_path, output_path, threshold=1000)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"), "\n1,2\n1,2\n1,2\n3,4\n"
            )
            self.assertEqual(stats.lines, 5)
            self.assertEqual(stats.labels_seen, 7)
            self.assertEqual(stats.labels_kept, 4)
            self.assertEqual(stats.labels_removed, 3)
            self.assertEqual(stats.empty_lines_filled, 2)

    def test_filter_file_can_update_the_same_file_safely(self):
        module = load_filter_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            label_path = Path(tmpdir) / "labels.txt"
            label_path.write_text("1,1000,2\n2000,3\n", encoding="utf-8")

            stats = module.filter_label_file(label_path, label_path, threshold=1000)

            self.assertEqual(label_path.read_text(encoding="utf-8"), "1,2\n3\n")
            self.assertEqual(stats.lines, 2)
            self.assertEqual(stats.labels_seen, 5)
            self.assertEqual(stats.labels_kept, 3)
            self.assertEqual(stats.labels_removed, 2)
            self.assertEqual(stats.empty_lines_filled, 0)


if __name__ == "__main__":
    unittest.main()
