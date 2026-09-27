#!/usr/bin/env python3
"""Tests for unattended authoritative-study finalization."""

from __future__ import annotations

import json
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
        self.assertIn("config.authoritative_amazon_screen_emptyfix.json", rendered)
        self.assertIn("amazon_profile_instrumented", rendered)
        self.assertIn("amazon_screen/summary/performance/figures", rendered)
        self.assertIn("MULTILEVEL_SPECIAL_BLOCK_AUTHORITATIVE_RESULTS_CN.md", rendered)
        self.assertIn("--validator", command)
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

    def make_provenance_fixture(
        self, root: Path,
    ) -> tuple[list[str], list[dict[str, str]], Path]:
        config = root / "formal.json"
        result = root / "formal.csv"
        policy = root / "policy.json"
        validator = root / "validator.py"
        manifest = root / "manifest.json"
        figures = root / "figures"
        config.write_text(json.dumps({"output_root": str(root)}), encoding="utf-8")
        result.write_text("status\ncomplete\n", encoding="utf-8")
        policy.write_text(json.dumps({"dataset": "Genome"}), encoding="utf-8")
        validator.write_text("# validation logic\n", encoding="utf-8")
        manifest.write_text('{"runs": []}\n', encoding="utf-8")
        figures.mkdir()
        for name in ("plot_manifest.json",
                     *(f"{family}.{suffix}"
                       for family in finalizer.FIGURE_FAMILIES
                       for suffix in ("pdf", "png"))):
            (figures / name).write_text(f"{name}\n", encoding="utf-8")
        command = [
            "python", "generator.py",
            "--amazon-formal-config", str(config),
            "--amazon-formal", str(result),
            "--heldout-policy", str(policy),
            "--validator", str(validator),
            "--amazon-figures-dir", str(figures),
        ]
        snapshot = finalizer.paper_input_snapshot(command)
        generated = root / "generated_results.tex"
        generated.write_text("".join(
            f"% {row['kind']} {row['label']} {row['sha256']}\n"
            for row in snapshot), encoding="utf-8")
        return command, snapshot, generated

    def test_generated_provenance_covers_every_current_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, snapshot, generated = self.make_provenance_fixture(Path(directory))
            finalizer.validate_generated_provenance(generated, snapshot)

    def test_generated_provenance_rejects_stale_input_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, snapshot, generated = self.make_provenance_fixture(root)
            (root / "formal.csv").write_text(
                "status\nchanged\n", encoding="utf-8")
            changed = finalizer.paper_input_snapshot([
                "python", "generator.py",
                "--amazon-formal-config", str(root / "formal.json"),
                "--amazon-formal", str(root / "formal.csv"),
                "--heldout-policy", str(root / "policy.json"),
                "--validator", str(root / "validator.py"),
                "--amazon-figures-dir", str(root / "figures"),
            ])
            with self.assertRaisesRegex(RuntimeError, "stale"):
                finalizer.validate_generated_provenance(generated, changed)

    def test_input_snapshot_rejects_change_during_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, snapshot, _ = self.make_provenance_fixture(root)
            (root / "formal.csv").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                finalizer.verify_input_snapshot(snapshot)

    def test_tectonic_logs_are_required_clean_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.log").write_text(
                "Output written on main.xdv.\n", encoding="utf-8")
            (root / "main.blg").write_text(
                "Database file #1: references.bib\n", encoding="utf-8")
            records = finalizer.validate_tectonic_logs(root)
            self.assertEqual(len(records), 2)
            self.assertTrue(all(len(row["sha256"]) == 64 for row in records))

            (root / "main.log").write_text(
                "LaTeX Warning: Citation `missing' on page 1 undefined.\n",
                encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "publication checks"):
                finalizer.validate_tectonic_logs(root)


if __name__ == "__main__":
    unittest.main()
