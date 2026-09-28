#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import run_deadline_profile_campaign as profile


class DeadlineProfileRunnerTest(unittest.TestCase):
    def test_manifest_finished_requires_explicit_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            self.assertFalse(profile.manifest_finished(path))
            path.write_text(json.dumps({"runs": []}))
            self.assertFalse(profile.manifest_finished(path))
            path.write_text(json.dumps({"finished_at_utc": "done"}))
            self.assertTrue(profile.manifest_finished(path))
            path.write_text(json.dumps({
                "finished_at_utc": "done", "status": "complete_with_failures",
            }))
            self.assertFalse(profile.manifest_finished(path))
            path.write_text(json.dumps({
                "finished_at_utc": "done", "status": "complete",
            }))
            self.assertTrue(profile.manifest_finished(path))

    def test_append_record_replaces_same_stage_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = {"runs": [{
                "stage": "profile_query", "dataset": "D",
                "method": "m", "workload": "w", "status": "timeout",
            }]}
            profile.append_record(path, state, {
                "stage": "profile_query", "dataset": "D",
                "method": "m", "workload": "w", "status": "complete",
            })
            self.assertEqual(len(state["runs"]), 1)
            self.assertEqual(state["runs"][0]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
