#!/usr/bin/env python3

import unittest

import summarize_deadline_evidence as summary


def point(lsearch: int, recall_min: float, recall_mean: float) -> dict:
    return {
        "lsearch": lsearch,
        "recall_min": recall_min,
        "recall": recall_mean,
    }


class DeadlineSummaryTest(unittest.TestCase):
    def test_operating_point_uses_first_conservative_crossing(self) -> None:
        rows = [
            point(100, 0.89, 0.91),
            point(500, 0.90, 0.92),
            point(2500, 0.95, 0.96),
        ]

        status, selected = summary.operating_point(rows, 0.90)

        self.assertEqual(status, "crossing")
        self.assertEqual(selected["lsearch"], 500)

    def test_operating_point_reports_best_measured_no_crossing(self) -> None:
        rows = [
            point(100, 0.70, 0.71),
            point(500, 0.89, 0.90),
            point(2500, 0.88, 0.95),
        ]

        status, selected = summary.operating_point(rows, 0.90)

        self.assertEqual(status, "no_crossing")
        self.assertEqual(selected["lsearch"], 500)

    def test_operating_point_keeps_missing_explicit(self) -> None:
        status, selected = summary.operating_point([], 0.90)

        self.assertEqual(status, "missing")
        self.assertIsNone(selected)

    def test_role_validation_rejects_non_gated_manual(self) -> None:
        methods = [
            {"name": "baseline", "selection_role": summary.BASELINE_ROLE},
            {"name": "auto", "selection_role": summary.AUTOMATIC_ROLE,
             "routing_policy": summary.ROUTING_POLICY},
        ]
        methods.extend({
            "name": f"manual_{index}", "selection_role": summary.MANUAL_ROLE,
            "routing_policy": (
                "always_layered" if index == 4 else summary.ROUTING_POLICY),
        } for index in range(summary.EXPECTED_MANUAL_ALTERNATIVES))

        with self.assertRaisesRegex(ValueError, "lacks exact routing gate"):
            summary.method_roles({"methods": methods})


if __name__ == "__main__":
    unittest.main()
