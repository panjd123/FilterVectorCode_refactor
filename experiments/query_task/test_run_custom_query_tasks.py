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

    def test_passes_batch_average_selectivity_parameters(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            tools_dir = tmp_path / "fake_build" / "tools"
            data_dir.mkdir()
            tools_dir.mkdir(parents=True)
            (data_dir / "Tiny_base_labels.txt").write_text("1\n")
            (data_dir / "Tiny_base.fvecs").write_bytes(b"")
            captured_args = tmp_path / "generate_args.json"

            generate_script = "\n".join(
                [
                    "#!/usr/bin/env python3",
                    "import json, pathlib, sys",
                    "args = sys.argv",
                    f"pathlib.Path({str(captured_args)!r}).write_text(json.dumps(args[1:]))",
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
                        "build_dir": str(tmp_path / "fake_build"),
                        "query_tasks": [
                            {
                                "enabled": True,
                                "task_name": "average_selectivity",
                                "mode": "variable_sub_base",
                                "dataset": "Tiny",
                                "data_dir": str(data_dir),
                                "overwrite": True,
                                "sub_base_params": {
                                    "num_points": 10,
                                    "target_average_selectivity": 0.5,
                                    "average_selectivity_tolerance": 0.002,
                                    "average_candidate_pool_size": 123,
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
            self.assertEqual("0.5", args[args.index("--target-average-selectivity") + 1])
            self.assertEqual("0.002", args[args.index("--average-selectivity-tolerance") + 1])
            self.assertEqual("123", args[args.index("--average-candidate-pool-size") + 1])


if __name__ == "__main__":
    unittest.main()
