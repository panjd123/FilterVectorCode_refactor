#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_generate_base_labels as runner


class GenerateBaseLabelsRunnerTest(unittest.TestCase):
    def test_builds_generate_base_labels_command_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tool = root / "build" / "tools" / "generate_base_labels"
            tool.parent.mkdir(parents=True)
            tool.write_text("#!/bin/sh\n", encoding="utf-8")
            tool.chmod(0o755)
            config_path = root / "config.json"
            output_file = root / "labels" / "base_labels.txt"
            config_path.write_text(
                json.dumps(
                    {
                        "build_dir": str(root / "build"),
                        "output_file": str(output_file),
                        "num_points": 100,
                        "num_labels": 10,
                        "distribution_type": "zipf",
                        "expected_num_label": 3,
                        "max_num_label": 12,
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(runner.subprocess, "run") as run:
                run.return_value.returncode = 0
                runner.main([str(config_path)])

            run.assert_called_once_with(
                [
                    str(tool),
                    "--output_file",
                    str(output_file),
                    "--num_points",
                    "100",
                    "--num_labels",
                    "10",
                    "--distribution_type",
                    "zipf",
                    "--expected_num_label",
                    "3",
                    "--max_num_label",
                    "12",
                ],
                check=True,
            )
            self.assertTrue(output_file.parent.exists())

    def test_missing_tool_reports_build_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "build_dir": str(root / "missing_build"),
                        "output_file": str(root / "labels.txt"),
                        "num_points": 100,
                        "num_labels": 10,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "generate_base_labels"):
                runner.main([str(config_path)])

    def test_runs_each_dataset_entry_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tool = root / "build" / "tools" / "generate_base_labels"
            tool.parent.mkdir(parents=True)
            tool.write_text("#!/bin/sh\n", encoding="utf-8")
            tool.chmod(0o755)
            config_path = root / "config.json"
            first_output = root / "Genome" / "Genome_base_labels.txt"
            second_output = root / "Reviews" / "Reviews_base_labels.txt"
            config_path.write_text(
                json.dumps(
                    {
                        "build_dir": str(root / "build"),
                        "num_labels": 1000,
                        "distribution_type": "zipf",
                        "expected_num_label": 3,
                        "max_num_label": 12,
                        "datasets": [
                            {
                                "dataset": "Genome",
                                "output_file": str(first_output),
                                "num_points": 108077,
                            },
                            {
                                "dataset": "Reviews",
                                "output_file": str(second_output),
                                "num_points": 288065,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(runner.subprocess, "run") as run:
                run.return_value.returncode = 0
                runner.main([str(config_path)])

            self.assertEqual(run.call_count, 2)
            self.assertEqual(
                run.call_args_list[0].args[0],
                [
                    str(tool),
                    "--output_file",
                    str(first_output),
                    "--num_points",
                    "108077",
                    "--num_labels",
                    "1000",
                    "--distribution_type",
                    "zipf",
                    "--expected_num_label",
                    "3",
                    "--max_num_label",
                    "12",
                ],
            )
            self.assertEqual(
                run.call_args_list[1].args[0],
                [
                    str(tool),
                    "--output_file",
                    str(second_output),
                    "--num_points",
                    "288065",
                    "--num_labels",
                    "1000",
                    "--distribution_type",
                    "zipf",
                    "--expected_num_label",
                    "3",
                    "--max_num_label",
                    "12",
                ],
            )
            self.assertTrue(first_output.parent.exists())
            self.assertTrue(second_output.parent.exists())


if __name__ == "__main__":
    unittest.main()
