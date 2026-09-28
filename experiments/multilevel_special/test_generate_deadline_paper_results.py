import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("generate_deadline_paper_results.py")
SPEC = importlib.util.spec_from_file_location("deadline_results", MODULE_PATH)
assert SPEC and SPEC.loader
deadline_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deadline_results)


class DeadlinePaperResultsTest(unittest.TestCase):
    def test_drh_v2_table_keeps_each_workload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "genome/search/summary/performance"
            output.mkdir(parents=True)
            path = output / "equal_recall_conservative.csv"
            fields = [
                "summary_path", "workload", "method", "mean_selectivity",
                "qps_warm_median",
            ]
            methods = [
                "l0_lng_entry_optimized_lng",
                "l2_t256_4096_lt_entry_optimized_lng_upper_routed",
                "l2_t256_4096_lt_entry_optimized_lng_upper_routed_next_scale_mass",
            ]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for workload, selectivity in (("narrow", "0.01"), ("broad", "0.05")):
                    for index, method in enumerate(methods):
                        writer.writerow({
                            "summary_path": f"/runs/genome/{workload}/{method}",
                            "workload": workload,
                            "method": method,
                            "mean_selectivity": selectivity,
                            "qps_warm_median": str(100 + index),
                        })

            rendered, rows = deadline_results.drh_v2_table(root)

            self.assertEqual(2, len(rows))
            self.assertIn("1.000\\%", rendered)
            self.assertIn("5.000\\%", rendered)

    def test_partial_figure_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            points = root / "all_points.csv"
            points.write_text("method,workload\n", encoding="utf-8")
            (root / "plot_manifest.json").write_text(
                '{"allow_partial": true}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "partial plot manifest"):
                deadline_results.validate_figures(root, points)


if __name__ == "__main__":
    unittest.main()
