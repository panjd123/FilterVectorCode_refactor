import json
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


def write_executable(path: Path, contents: str):
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class RunCustomQueryTasksBuildDirTest(unittest.TestCase):
    def test_uses_build_dir_from_json_when_cli_arg_is_omitted(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            build_dir = tmp_path / "fake_build"
            tools_dir = build_dir / "tools"
            data_dir.mkdir()
            tools_dir.mkdir(parents=True)

            (data_dir / "Tiny_base_labels.txt").write_text("1\n")
            (data_dir / "Tiny_base.fvecs").write_bytes(b"")
            marker = tmp_path / "used_json_build_dir.marker"

            generate_script = "\n".join(
                [
                    "#!/usr/bin/env python3",
                    "import pathlib, sys",
                    "args = sys.argv",
                    f"pathlib.Path({str(marker)!r}).write_text('used')",
                    "pathlib.Path(args[args.index('--output_file') + 1]).write_text('1\\n')",
                    "pathlib.Path(args[args.index('--output_vectors_file') + 1]).write_bytes(b'fake')",
                    "",
                ]
            )
            convert_script = "\n".join(
                [
                    "#!/usr/bin/env python3",
                    "import pathlib, sys",
                    "args = sys.argv",
                    "pathlib.Path(args[args.index('--output_file') + 1]).write_bytes(b'bin')",
                    "",
                ]
            )
            write_executable(tools_dir / "generate_mixed_queries", generate_script)
            write_executable(tools_dir / "fvecs_to_bin", convert_script)

            config_path = tmp_path / "query_tasks.json"
            config_path.write_text(
                json.dumps(
                    {
                        "build_dir": str(build_dir),
                        "query_tasks": [
                            {
                                "enabled": True,
                                "task_name": "json_build",
                                "mode": "generate",
                                "dataset": "Tiny",
                                "data_dir": str(data_dir),
                                "overwrite": True,
                                "generation_params": {"num_points": 1},
                            }
                        ],
                    }
                )
            )

            subprocess.run(
                ["python3", str(repo_root / "scripts" / "run_custom_query_tasks.py"), str(config_path)],
                cwd=repo_root,
                check=True,
            )

            self.assertEqual("used", marker.read_text())
            self.assertTrue((data_dir / "query_json_build" / "Tiny_query.bin").is_file())

    def test_passes_fixed_selectivity_range_parameters(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            tools_dir = tmp_path / "fake_build" / "tools"
            data_dir.mkdir()
            tools_dir.mkdir(parents=True)
            (data_dir / "Tiny_base_labels.txt").write_text("1,2\n")
            (data_dir / "Tiny_base.fvecs").write_bytes(b"")
            captured_args = tmp_path / "generate_args.json"

            generate_script = "\n".join(
                [
                    "#!/usr/bin/env python3",
                    "import json, pathlib, sys",
                    "args = sys.argv",
                    f"pathlib.Path({str(captured_args)!r}).write_text(json.dumps(args[1:]))",
                    "pathlib.Path(args[args.index('--output_file') + 1]).write_text('1,2\\n')",
                    "pathlib.Path(args[args.index('--output_vectors_file') + 1]).write_bytes(b'fake')",
                    "",
                ]
            )
            convert_script = "\n".join(
                [
                    "#!/usr/bin/env python3",
                    "import pathlib, sys",
                    "args = sys.argv",
                    "pathlib.Path(args[args.index('--output_file') + 1]).write_bytes(b'bin')",
                    "",
                ]
            )
            write_executable(tools_dir / "generate_mixed_queries", generate_script)
            write_executable(tools_dir / "fvecs_to_bin", convert_script)

            config_path = tmp_path / "query_tasks.json"
            config_path.write_text(
                json.dumps(
                    {
                        "build_dir": str(tmp_path / "fake_build"),
                        "query_tasks": [
                            {
                                "enabled": True,
                                "task_name": "fixed_range",
                                "mode": "variable_sub_base",
                                "dataset": "Tiny",
                                "data_dir": str(data_dir),
                                "overwrite": True,
                                "sub_base_params": {
                                    "num_points": 10,
                                    "min_selectivity": 0.1,
                                    "max_selectivity": 0.4,
                                    "selectivity_candidate_pool_size": 123,
                                    "random_seed": 456,
                                    "selection_target_selectivity": 0.15,
                                    "preferred_selectivity_min": 0.1,
                                    "preferred_selectivity_max": 0.2,
                                    "duplicate_selectivity_min": 0.1,
                                    "duplicate_selectivity_max": 0.15,
                                    "include_selectivity_extremes": True,
                                },
                            }
                        ],
                    }
                )
            )

            subprocess.run(
                ["python3", str(repo_root / "scripts" / "run_custom_query_tasks.py"), str(config_path)],
                cwd=repo_root,
                check=True,
            )
            args = json.loads(captured_args.read_text())
            self.assertEqual("0.1", args[args.index("--min-selectivity") + 1])
            self.assertEqual("0.4", args[args.index("--max-selectivity") + 1])
            self.assertEqual(
                "123", args[args.index("--selectivity-candidate-pool-size") + 1]
            )
            self.assertEqual("456", args[args.index("--random-seed") + 1])
            self.assertEqual(
                "0.15", args[args.index("--selection-target-selectivity") + 1]
            )
            self.assertEqual(
                "0.1", args[args.index("--preferred-selectivity-min") + 1]
            )
            self.assertEqual(
                "0.2", args[args.index("--preferred-selectivity-max") + 1]
            )
            self.assertEqual(
                "0.1", args[args.index("--duplicate-selectivity-min") + 1]
            )
            self.assertEqual(
                "0.15", args[args.index("--duplicate-selectivity-max") + 1]
            )
            self.assertEqual(
                "true", args[args.index("--include-selectivity-extremes") + 1]
            )
            self.assertNotIn("--target-per-query-selectivity", args)
            self.assertNotIn("--target-average-selectivity", args)

    def test_rejects_half_configured_selectivity_range(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            tools_dir = tmp_path / "fake_build" / "tools"
            data_dir.mkdir()
            tools_dir.mkdir(parents=True)
            (data_dir / "Tiny_base_labels.txt").write_text("1,2\n")
            (data_dir / "Tiny_base.fvecs").write_bytes(b"")
            write_executable(tools_dir / "generate_mixed_queries", "#!/bin/sh\nexit 0\n")
            write_executable(tools_dir / "fvecs_to_bin", "#!/bin/sh\nexit 0\n")

            config_path = tmp_path / "query_tasks.json"
            config_path.write_text(
                json.dumps(
                    {
                        "build_dir": str(tmp_path / "fake_build"),
                        "query_tasks": [
                            {
                                "enabled": True,
                                "task_name": "bad_range",
                                "mode": "variable_sub_base",
                                "dataset": "Tiny",
                                "data_dir": str(data_dir),
                                "overwrite": True,
                                "sub_base_params": {"min_selectivity": 0.1},
                            }
                        ],
                    }
                )
            )

            result = subprocess.run(
                ["python3", str(repo_root / "scripts" / "run_custom_query_tasks.py"), str(config_path)],
                cwd=repo_root,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(0, result.returncode)
            self.assertIn("must set both min_selectivity and max_selectivity", result.stderr)


if __name__ == "__main__":
    unittest.main()
