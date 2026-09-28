#!/usr/bin/env python3

import sys
import types
import unittest

# The record helpers under test do not depend on the remote-only bounded runner.
sys.modules.setdefault("run_deadline_build_campaign", types.ModuleType(
    "run_deadline_build_campaign"))

import run_remaining_evidence_campaign as campaign


class RemainingEvidenceCampaignTest(unittest.TestCase):
    def test_replace_record_discards_stale_failure_and_increments_attempt(self):
        state = {"runs": [{
            "stage": "query", "method": "m", "workload": "w",
            "status": "failed", "attempt": 1,
        }]}

        campaign.replace_record(state, {
            "stage": "query", "method": "m", "workload": "w",
            "status": "complete",
        })

        self.assertEqual(1, len(state["runs"]))
        self.assertEqual("complete", state["runs"][0]["status"])
        self.assertEqual(2, state["runs"][0]["attempt"])
        self.assertTrue(campaign.is_complete(state, "query", "m", "w"))


if __name__ == "__main__":
    unittest.main()
