#!/usr/bin/env python3

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "run_cpu_special_blocks_experiment.sh"


class CpuSpecialBlocksRunnerTest(unittest.TestCase):
    def test_keeps_dataset_others_but_does_not_write_run_logs_or_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            result_root = root / "results"
            build_dir = root / "build"
            config_dir = root / "config"
            fake_bin = root / "bin"
            dataset_dir = data_root / "Tiny"
            dataset_dir.mkdir(parents=True)
            build_app = build_dir / "apps" / "build_UNG_index"
            build_app.parent.mkdir(parents=True)

            (dataset_dir / "Tiny_base.bin").write_bytes(b"tiny")
            (dataset_dir / "Tiny_base_labels.txt").write_text("0\n", encoding="utf-8")
            build_app.write_text(
                "#!/usr/bin/env bash\n"
                "set -e\n"
                "while [ \"$#\" -gt 0 ]; do\n"
                "  case \"$1\" in\n"
                "    --index_path_prefix) index_dir=\"$2\"; shift 2 ;;\n"
                "    *) shift ;;\n"
                "  esac\n"
                "done\n"
                "mkdir -p \"$index_dir\"\n"
                "printf 'num_points=1\\n' > \"${index_dir}/meta\"\n",
                encoding="utf-8",
            )
            build_app.chmod(0o755)

            fake_bin.mkdir()
            cmake_log = root / "cmake.log"
            fake_cmake = fake_bin / "cmake"
            fake_cmake.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\n' \"$*\" >> \"${FAKE_CMAKE_LOG}\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_cmake.chmod(0o755)

            config_dir.mkdir()
            config_path = config_dir / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_name": "cpu_special_blocks_test",
                        "output_layout": "index_by_dataset",
                        "index_name": "UNG_test",
                        "data_root": str(data_root),
                        "result_root": str(result_root),
                        "build_dir": str(build_dir),
                        "datasets": ["Tiny"],
                        "build": {},
                        "ung_env": {},
                    }
                ),
                encoding="utf-8",
            )

            env = {
                **__import__("os").environ,
                "PATH": f"{fake_bin}:{__import__('os').environ['PATH']}",
                "FAKE_CMAKE_LOG": str(cmake_log),
            }
            subprocess.run([str(RUNNER), str(config_path)], check=True, cwd=REPO_ROOT, env=env)

            cmake_calls = cmake_log.read_text(encoding="utf-8")
            self.assertIn("-S", cmake_calls)
            self.assertIn("--build", cmake_calls)
            self.assertIn("--target build_UNG_index", cmake_calls)

            out_dir = result_root / "Tiny" / "index" / "UNG_test"
            self.assertTrue((out_dir / "others" / "build.log").exists())
            self.assertTrue((out_dir / "others" / "time.txt").exists())
            self.assertFalse(list(config_dir.glob("*.log")))
            self.assertFalse(list(result_root.rglob("summary.csv")))


if __name__ == "__main__":
    unittest.main()
