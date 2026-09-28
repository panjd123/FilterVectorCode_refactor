import tempfile
import unittest
from pathlib import Path

import plot_authoritative_recall_qps as plotting


class AuthoritativePlotTest(unittest.TestCase):
    def test_default_families_are_complete_paper_subset(self):
        self.assertEqual(
            plotting.PAPER_FAMILY_NAMES,
            (
                "principal_zero",
                "one_layer_topology",
                "representative_depth",
                "upper_authorization",
            ),
        )
        self.assertTrue(set(plotting.PAPER_FAMILY_NAMES) <= set(plotting.FAMILIES))

    def test_representative_depth_uses_complete_measured_family(self):
        self.assertEqual(
            [name for name, _ in plotting.FAMILIES["representative_depth"]],
            [
                "l0_lng_entry_optimized_lng",
                "l1_t1024_lng_entry_optimized_lng",
                "l2_t1024_16384_lt_entry_optimized_lng",
            ])
        self.assertEqual(
            [label for _, label in plotting.FAMILIES["representative_depth"]],
            [
                "0L-LNG",
                "1L-LNG: T1=1,024",
                "2L-LT: T1=1,024, T2=16,384",
            ])

    def test_one_layer_topology_fixes_threshold_and_entry(self):
        self.assertEqual(
            [name for name, _ in plotting.FAMILIES["one_layer_topology"]],
            [
                "l1_t1024_lng_entry_optimized_lng",
                "l1_t1024_trie_entry_optimized_lng",
            ])

    def test_threshold_family_fixes_topology_and_entry_strategy(self):
        self.assertEqual(
            [name for name, _ in plotting.FAMILIES["threshold_depth"]],
            [
                "l1_t1024_lng_entry_optimized_lng",
                "l1_t8192_lng_entry_optimized_lng",
                "l2_t1024_16384_lt_entry_optimized_lng",
                "l2_t8192_131072_lt_entry_optimized_lng",
            ])

    def test_curve_follows_lsearch_when_measured_recall_is_nonmonotone(self):
        points = [
            {"lsearch": "300", "recall": "0.92"},
            {"lsearch": "100", "recall": "0.88"},
            {"lsearch": "200", "recall": "0.87"},
        ]
        ordered = plotting.sweep_order(points)
        self.assertEqual([row["lsearch"] for row in ordered],
                         ["100", "200", "300"])
        self.assertEqual([row["recall"] for row in ordered],
                         ["0.88", "0.87", "0.92"])

    def test_partial_plot_records_missing_method_workload(self):
        rows = [{
            "workload": "w", "method": "measured", "recall": "0.91",
            "lsearch": "100", "qps_warm_median": "123.0",
        }]
        workloads = [{"name": "w", "mean_selectivity": 0.1}]
        methods = [("measured", "Measured"), ("missing", "Missing")]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "curve"
            missing = plotting.plot_family(
                rows, workloads, methods, {"w": 0.9}, output,
                allow_partial=True)
            self.assertEqual(missing, [("missing", "w")])
            self.assertTrue(output.with_suffix(".pdf").is_file())
            self.assertTrue(output.with_suffix(".png").is_file())
            with self.assertRaisesRegex(RuntimeError, "incomplete measured family"):
                plotting.plot_family(
                    rows, workloads, methods, {"w": 0.9}, output,
                    allow_partial=False)


if __name__ == "__main__":
    unittest.main()
