import tempfile
import unittest
from pathlib import Path

import plot_authoritative_recall_qps as plotting


class AuthoritativePlotTest(unittest.TestCase):
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
