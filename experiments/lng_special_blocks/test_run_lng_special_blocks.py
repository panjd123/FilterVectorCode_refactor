#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_lng_special_blocks.py")


def load_runner():
    spec = importlib.util.spec_from_file_location("run_lng_special_blocks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LngSpecialBlocksRunnerTest(unittest.TestCase):
    def test_rebuilds_target_when_source_is_newer_than_existing_executable(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            exe = build_dir / "apps" / "build_UNG_index"
            exe.parent.mkdir(parents=True)
            exe.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            exe.chmod(0o755)

            source = root / "src" / "changed.cpp"
            source.parent.mkdir(parents=True)
            source.write_text("// newer than exe\n", encoding="utf-8")
            old_time = 100
            new_time = 200
            exe.touch()
            source.touch()
            runner.os.utime(exe, (old_time, old_time))
            runner.os.utime(source, (new_time, new_time))

            calls = []
            original_run_command = runner.run_command
            original_sources = runner.target_source_paths
            try:
                runner.run_command = lambda cmd, **_kwargs: calls.append([str(x) for x in cmd])
                runner.target_source_paths = lambda _target: [source]

                result = runner.ensure_target(build_dir, "build_UNG_index", build_policy="if_stale")
            finally:
                runner.run_command = original_run_command
                runner.target_source_paths = original_sources

            self.assertEqual(result, exe)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][:2], ["cmake", "-S"])
            self.assertEqual(calls[1][:3], ["cmake", "--build", str(build_dir)])

    def test_does_not_rebuild_when_existing_executable_is_fresh(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            exe = build_dir / "apps" / "search_UNG_index"
            exe.parent.mkdir(parents=True)
            exe.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            exe.chmod(0o755)

            source = root / "src" / "old.cpp"
            source.parent.mkdir(parents=True)
            source.write_text("// older than exe\n", encoding="utf-8")
            exe.touch()
            source.touch()
            runner.os.utime(source, (100, 100))
            runner.os.utime(exe, (200, 200))

            calls = []
            original_run_command = runner.run_command
            original_sources = runner.target_source_paths
            try:
                runner.run_command = lambda cmd, **_kwargs: calls.append([str(x) for x in cmd])
                runner.target_source_paths = lambda _target: [source]

                result = runner.ensure_target(build_dir, "search_UNG_index", build_policy="if_stale")
            finally:
                runner.run_command = original_run_command
                runner.target_source_paths = original_sources

            self.assertEqual(result, exe)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
