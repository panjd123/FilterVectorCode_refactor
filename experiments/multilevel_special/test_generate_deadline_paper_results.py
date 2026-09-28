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
    def test_one_layer_topology_table_keeps_nc_and_max_recall(self) -> None:
        methods = (
            "l1_t1024_lng_entry_optimized_lng",
            "l1_t1024_trie_entry_optimized_lng",
        )
        points = []
        crossings = []
        for workload_index, workload in enumerate(deadline_results.WORKLOAD_ORDER):
            for method_index, method in enumerate(methods):
                row = {
                    "workload": workload,
                    "method": method,
                    "mean_selectivity": str((workload_index + 1) / 100),
                    "lsearch": "100",
                    "recall_min": str(0.80 + method_index / 100),
                    "qps_warm_median": str(100 + method_index),
                }
                points.append(row)
                if workload_index >= 5:
                    crossing = dict(row)
                    crossing["recall_min"] = "0.91"
                    crossings.append(crossing)

        rendered, rows = deadline_results.one_layer_topology_table(
            crossings, points)

        self.assertEqual(9, len(rows))
        self.assertIsNone(rows[0]["trie_over_lng"])
        self.assertAlmostEqual(1.01, rows[-1]["trie_over_lng"])
        self.assertIn("NC & NC & -- & 0.8000 & 0.8100", rendered)

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
