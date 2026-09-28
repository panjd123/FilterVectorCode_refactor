#!/usr/bin/env python3
"""Tests for the isolated authorization-profile rerun."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_authoritative_instrumented_profile as profile_runner


class InstrumentedProfileTest(unittest.TestCase):
    def test_profile_config_is_isolated_and_keeps_formal_operating_points(self) -> None:
        formal = {
            "output_root": "/tmp/campaign/amazon_formal",
            "search_app": "/tmp/old/search_UNG_index",
            "num_repeats": 16,
            "measurement_pass": "performance",
            "methods": [{
                "name": "baseline", "enabled_workloads": ["sel_1"],
                "lsearch_values_by_workload": {"sel_1": [1200]},
            }],
            "workloads": [{"name": "sel_1"}],
            "protocol": {
                "phase": "formal", "cold_repeats": 1,
                "measured_repeats": 15, "recall_rule": "all_repeats",
            },
        }
        profile = profile_runner.make_profile_config(
            formal, Path("/tmp/new/search_UNG_index"), "abc", "def")
        self.assertEqual(
            profile["output_root"],
            "/tmp/campaign/amazon_profile_instrumented")
        self.assertEqual(profile["search_app"], "/tmp/new/search_UNG_index")
        self.assertEqual(profile["num_repeats"], 4)
        self.assertEqual(profile["protocol"]["phase"], "profile")
        self.assertEqual(profile["protocol"]["measured_repeats"], 3)
        self.assertEqual(
            profile["methods"][0]["lsearch_values_by_workload"]["sel_1"],
            [1200])
        self.assertEqual(formal["search_app"], "/tmp/old/search_UNG_index")

    def test_active_process_filter_excludes_probe_and_self(self) -> None:
        completed = mock.Mock(
            stdout=(
                "1 pgrep -af search_UNG_index\n"
                "2 python run_authoritative_instrumented_profile.py\n"
                "3 /tmp/search_UNG_index --K 10\n"))
        with mock.patch("subprocess.run", return_value=completed):
            self.assertEqual(
                profile_runner.active_experiment_processes(),
                ["3 /tmp/search_UNG_index --K 10"])

    def test_manifest_binary_check_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "output_root": str(root), "measurement_pass": "profile",
                "pass_subdirs": True,
                "methods": [{"name": "m"}],
                "workloads": [{"name": "w"}],
            }
            manifest = root / "manifest_profile.json"
            manifest.write_text(json.dumps({"runs": [{
                "method": "m", "workload": "w", "status": "complete",
                "returncode": 0, "search_binary_sha256": "old",
            }]}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "binary mismatch"):
                profile_runner.validate_profile_binary(config, "new")

    def test_atomic_config_write_uses_lf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile_runner.write_json_atomic(path, {"value": 1})
            self.assertEqual(
                path.read_bytes(), b'{\n  "value": 1\n}\n')

    def test_profile_build_inherits_reference_dependency_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "CMakeCache.txt"
            keys = (
                "GRAPHDB_ROOT", "TAGORE_ROOT", "BOOST_ROOT", "OPENBLAS_ROOT",
                "ZLIB_ROOT", "CROARING_ROOT", "ONNXRUNTIME_DIR",
            )
            cache.write_text("".join(
                f"{key}:PATH=/reference/{key.lower()}\n" for key in keys),
                encoding="utf-8")
            self.assertEqual(
                profile_runner.inherited_cmake_paths(cache),
                [f"-D{key}=/reference/{key.lower()}" for key in keys],
            )


if __name__ == "__main__":
    unittest.main()
