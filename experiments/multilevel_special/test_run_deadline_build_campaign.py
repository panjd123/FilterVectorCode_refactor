#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

import run_deadline_build_campaign as deadline


class DeadlineBuildRunnerTest(unittest.TestCase):
    def test_summary_command_uses_all_four_generated_configs(self) -> None:
        campaign = {
            "summarizer": "/repo/summarize.py",
            "summary_output_dir": "/run/summary",
            "stages": [
                {"phase": name, "config": f"/repo/{name}.json"}
                for name in deadline.BUILD_PHASES
            ],
        }
        command = deadline.summary_command(campaign)
        self.assertEqual(command[1], "/repo/summarize.py")
        for phase in deadline.BUILD_PHASES:
            self.assertIn(f"/repo/{phase}.json", command)
        self.assertEqual(command[-1], "/run/summary")

    def test_query_gate_requires_finished_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.json"
            path.write_text(json.dumps({"runs": []}))
            with self.assertRaisesRegex(RuntimeError, "still active"):
                deadline.require_finished_query_campaign(path)
            path.write_text(json.dumps({"finished_at_utc": "done"}))
            deadline.require_finished_query_campaign(path)

    def test_run_bounded_reports_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            status, returncode, elapsed = deadline.run_bounded(
                [sys.executable, "-c", "import time; time.sleep(2)"],
                log, 0.05)
            self.assertEqual(status, "timeout")
            self.assertEqual(returncode, 124)
            self.assertLess(elapsed, 2)
            self.assertIn("DEADLINE TIMEOUT", log.read_text())

    def test_update_record_replaces_prior_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = {"runs": [{
                "phase": "base_timing", "case": "x",
                "status": "timeout", "attempt": 1,
            }]}
            deadline.update_record(path, state, {
                "phase": "base_timing", "case": "x", "status": "complete",
            })
            self.assertEqual(len(state["runs"]), 1)
            self.assertEqual(state["runs"][0]["attempt"], 2)
            self.assertEqual(state["runs"][0]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
