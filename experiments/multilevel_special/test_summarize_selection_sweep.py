import unittest

import summarize_selection_sweep as summary


class SummarizeSelectionSweepTest(unittest.TestCase):
    def test_require_baseline_accepts_measured_method(self):
        summary.require_baseline(
            [{"method": "plain"}, {"method": "layered"}], "plain")

    def test_require_baseline_rejects_missing_method(self):
        with self.assertRaisesRegex(
                ValueError, "baseline method 'stale' has no measured rows"):
            summary.require_baseline([{"method": "plain"}], "stale")

    def test_require_baseline_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "available methods: \\(none\\)"):
            summary.require_baseline([], "plain")


if __name__ == "__main__":
    unittest.main()
