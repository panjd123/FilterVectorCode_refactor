#!/usr/bin/env python3
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent / "generate_hybrid_labels.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_hybrid_labels", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GenerateHybridLabelsTest(unittest.TestCase):
    def test_builds_reproducible_hybrid_labels_with_tail_cap_and_manifest(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            zipf = tmp / "zipf.txt"
            old = tmp / "old.txt"
            output = tmp / "hybrid.txt"
            manifest = tmp / "manifest.json"
            zipf.write_text("1,2\n1,3\n2,4\n", encoding="utf-8")
            old.write_text("1,2,10,20,30\n1,3,20,40\n2,4,30,50,60\n", encoding="utf-8")

            args = module.parse_args(
                [
                    "--zipf-base",
                    str(zipf),
                    "--old-base",
                    str(old),
                    "--output",
                    str(output),
                    "--manifest",
                    str(manifest),
                    "--tail-prob",
                    "1.0",
                    "--tail-min-freq",
                    "1",
                    "--tail-max-freq",
                    "2",
                    "--tail-max-per-point",
                    "2",
                    "--max-labels",
                    "4",
                    "--seed",
                    "7",
                ]
            )

            summary = module.generate(args)

            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            for line in lines:
                labels = [int(value) for value in line.split(",")]
                self.assertEqual(labels, sorted(set(labels)))
                self.assertLessEqual(len(labels), 4)
                self.assertGreaterEqual(max(labels), 10)
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(saved["num_points"], 3)
            self.assertEqual(saved["seed"], 7)
            self.assertEqual(saved["output_stats"]["max_labels_per_point"], 4)
            self.assertEqual(summary["output_stats"], saved["output_stats"])

    def test_row_mix_uses_whole_old_or_zipf_rows_and_can_skip_manifest(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            zipf = tmp / "zipf.txt"
            old = tmp / "old.txt"
            output = tmp / "hybrid.txt"
            zipf_rows = ["1,2", "1,3", "2,4", "3,5"]
            old_rows = ["10,20,30", "11,21,31", "12,22,32", "13,23,33"]
            zipf.write_text("\n".join(zipf_rows) + "\n", encoding="utf-8")
            old.write_text("\n".join(old_rows) + "\n", encoding="utf-8")

            args = module.parse_args(
                [
                    "--zipf-base",
                    str(zipf),
                    "--old-base",
                    str(old),
                    "--output",
                    str(output),
                    "--mode",
                    "row-mix",
                    "--old-row-ratio",
                    "0.5",
                    "--seed",
                    "3",
                    "--no-manifest",
                ]
            )

            summary = module.generate(args)

            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines, [old_rows[0], old_rows[1], zipf_rows[2], zipf_rows[3]])
            self.assertFalse(Path(str(output) + ".manifest.json").exists())
            self.assertEqual(summary["parameters"]["mode"], "row-mix")
            self.assertEqual(summary["parameters"]["old_row_placement"], "first")
            self.assertEqual(summary["mix_stats"]["old_rows"], 2)
            self.assertEqual(summary["mix_stats"]["zipf_rows"], 2)
            self.assertEqual(summary["mix_stats"]["old_range"], [0, 2])
            self.assertEqual(summary["mix_stats"]["zipf_range"], [2, 4])


    def test_can_fill_empty_rows_and_compact_label_ids(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            zipf = tmp / "zipf.txt"
            old = tmp / "old.txt"
            output = tmp / "hybrid.txt"
            zipf.write_text("\n".join(["7", "8", "1,2,3", "2,3"]) + "\n", encoding="utf-8")
            old.write_text("\n".join(["1,1000", "", "20,40", "50"]) + "\n", encoding="utf-8")

            args = module.parse_args(
                [
                    "--zipf-base",
                    str(zipf),
                    "--old-base",
                    str(old),
                    "--output",
                    str(output),
                    "--mode",
                    "row-mix",
                    "--old-row-ratio",
                    "0.5",
                    "--fill-empty-with-previous",
                    "--compact-label-ids",
                    "--no-manifest",
                ]
            )

            summary = module.generate(args)

            self.assertEqual(output.read_text(encoding="utf-8").splitlines(), ["1,2", "1,2", "3,4,5", "4,5"])
            self.assertEqual(summary["postprocess_stats"]["filled_empty_rows"], 1)
            self.assertEqual(summary["postprocess_stats"]["label_id_mapping_size"], 5)
            self.assertEqual(summary["postprocess_stats"]["label_id_min"], 1)
            self.assertEqual(summary["postprocess_stats"]["label_id_max"], 5)

    def test_rejects_mismatched_input_lengths(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            zipf = tmp / "zipf.txt"
            old = tmp / "old.txt"
            output = tmp / "hybrid.txt"
            zipf.write_text("1\n2\n", encoding="utf-8")
            old.write_text("1\n", encoding="utf-8")

            args = module.parse_args(
                [
                    "--zipf-base",
                    str(zipf),
                    "--old-base",
                    str(old),
                    "--output",
                    str(output),
                ]
            )

            with self.assertRaisesRegex(ValueError, "same number of rows"):
                module.generate(args)


if __name__ == "__main__":
    unittest.main()
