#!/usr/bin/env python3
"""Tests for unattended authoritative-study finalization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import continue_after_build as finalizer


class ContinueAfterBuildTest(unittest.TestCase):
    def test_required_artifacts_cover_query_heldout_and_build(self) -> None:
        paths = finalizer.required_artifacts(Path("/run"), Path("/results"))
        rendered = "\n".join(str(path) for path in paths)
        self.assertIn("amazon_formal", rendered)
        self.assertIn("heldout_oracle_by_workload.csv", rendered)
        self.assertIn("build_end_to_end.csv", rendered)

    def test_paper_command_uses_instrumented_profile_and_all_policies(self) -> None:
        command = finalizer.paper_generation_command(
            Path("/run"), Path("/results"))
        rendered = " ".join(command)
        self.assertIn("config.authoritative_amazon_profile_instrumented.json", rendered)
        self.assertIn("amazon_profile_instrumented", rendered)
        self.assertEqual(command.count("--heldout-formal-config"), 3)
        self.assertEqual(command.count("--heldout-policy"), 3)
        self.assertEqual(command.count("--build-config"), 4)

    def test_active_process_filter_excludes_probe_and_self(self) -> None:
        completed = mock.Mock(stdout=(
            "1 pgrep -af search_UNG_index\n"
            "2 python continue_after_build.py\n"
            "3 /tmp/search_UNG_index --K 10\n"))
        with mock.patch("subprocess.run", return_value=completed):
            self.assertEqual(
                finalizer.active_experiment_processes(),
                ["3 /tmp/search_UNG_index --K 10"])

    def test_manifest_is_atomic_and_hashable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(b"evidence")
            manifest = root / "manifest.json"
            finalizer.write_manifest(manifest, {"sha256": finalizer.sha256(source)})
            self.assertIn(finalizer.sha256(source), manifest.read_text())


if __name__ == "__main__":
    unittest.main()
