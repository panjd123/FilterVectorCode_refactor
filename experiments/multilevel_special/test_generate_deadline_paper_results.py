import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("generate_deadline_paper_results.py")
SPEC = importlib.util.spec_from_file_location("deadline_results", MODULE_PATH)
assert SPEC and SPEC.loader
deadline_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deadline_results)


class DeadlinePaperResultsTest(unittest.TestCase):
    def test_two_layer_topology_table_fails_on_missing_method(self) -> None:
        methods = (
            "l2_t1024_16384_ll_entry_optimized_lng",
            "l2_t1024_16384_lt_entry_optimized_lng",
            "l2_t1024_16384_tl_entry_optimized_lng",
            "l2_t1024_16384_tt_entry_optimized_lng",
        )
        points = []
        for workload in deadline_results.WORKLOAD_ORDER:
            for index, method in enumerate(methods):
                points.append({
                    "workload": workload, "method": method,
                    "mean_selectivity": "0.1", "lsearch": "100",
                    "recall_min": str(0.90 + index / 100),
                    "qps_warm_median": str(100 + index),
                })
        rendered, markdown = deadline_results.two_layer_topology_tables(
            points, points)
        self.assertIn("LL QPS & LT QPS & TL QPS & TT QPS", rendered)
        self.assertIn("## 两层 topology 公平消融", markdown)
        with self.assertRaisesRegex(ValueError, "missing two-layer topology points"):
            deadline_results.two_layer_topology_tables(points, points[:-1])

    def test_construction_summary_requires_repeats_and_renders_resources(self) -> None:
        rows = []
        end_to_end = []
        for profile in sorted(deadline_results.REQUIRED_HIERARCHY_BUILD_PROFILES):
            rows.append({
                "component": "hierarchy", "profile": profile,
                "measured_repeats": "2", "wall_median_seconds": "10",
                "wall_cv": "0.1", "speedup_vs_component_cpu": "4",
                "peak_rss_mib": "100",
                "peak_gpu_memory_mib": "" if profile == "cpu" else "200",
            })
            end_to_end.append({
                "hierarchy_profile": profile, "stage_repeats": "2",
                "original_cpu_base_median_seconds": "200",
                "composed_base_plus_hierarchy_seconds": "60",
                "speedup_vs_original_cpu": "3.333",
            })

        rendered, markdown = deadline_results.construction_summary_tables(
            rows, end_to_end)

        self.assertIn("full\\_gpu & 2 & 10.00", rendered)
        self.assertIn("| full_gpu | 2 | 10.00", markdown)
        rows[0]["measured_repeats"] = "1"
        with self.assertRaisesRegex(ValueError, "two measured repeats"):
            deadline_results.construction_summary_tables(rows, end_to_end)

    def test_construction_summary_requires_resource_and_complete_profile_sets(self) -> None:
        rows = []
        end_to_end = []
        for profile in sorted(deadline_results.REQUIRED_HIERARCHY_BUILD_PROFILES):
            rows.append({
                "component": "hierarchy", "profile": profile,
                "measured_repeats": "2", "wall_median_seconds": "10",
                "wall_cv": "0.1", "speedup_vs_component_cpu": "4",
                "peak_rss_mib": "100", "peak_gpu_memory_mib": "200",
            })
            end_to_end.append({
                "hierarchy_profile": profile, "stage_repeats": "2",
                "original_cpu_base_median_seconds": "200",
                "composed_base_plus_hierarchy_seconds": "60",
                "speedup_vs_original_cpu": "3.333",
            })
        rows[0]["peak_rss_mib"] = ""
        with self.assertRaisesRegex(ValueError, "missing peak RSS"):
            deadline_results.construction_summary_tables(rows, end_to_end)
        rows[0]["peak_rss_mib"] = "100"
        with self.assertRaisesRegex(ValueError, "five hierarchy profiles"):
            deadline_results.construction_summary_tables(rows[:-1], end_to_end)
        with self.assertRaisesRegex(ValueError, "five hierarchy profiles"):
            deadline_results.construction_summary_tables(rows, end_to_end[:-1])

    def test_complete_build_manifest_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps({
                "status": "complete_with_failures",
                "runs": [{"phase": "hierarchy_resource", "case": "cpu",
                          "status": "failed"}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "build campaign is not complete"):
                deadline_results.require_complete_build_manifest(path)
            path.write_text(json.dumps({
                "status": "complete",
                "runs": [{"phase": "hierarchy_resource", "case": "cpu",
                          "status": "complete"}],
            }), encoding="utf-8")
            deadline_results.require_complete_build_manifest(path)

    def test_amazon_profile_table_requires_edges_and_preserves_status(self) -> None:
        methods = (
            "l0_lng_entry_optimized_lng",
            "l0_trie_entry_trie",
            "l1_t1024_lng_entry_optimized_lng",
            "l1_t1024_trie_entry_optimized_lng",
            "l2_t1024_16384_lt_entry_optimized_lng_upper_routed",
        )
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            manifest.write_text(__import__("json").dumps({
                "cases": [
                    {"workload": "sel_10", "method": method,
                     "performance_status": "crossing" if index else "no_crossing_best_measured"}
                    for index, method in enumerate(methods)
                ],
            }))
            rows = []
            for method in methods:
                rows.append({
                    "workload": "sel_10", "method": method,
                    "mean_selectivity": "0.1", "els_ms_warm_median": "1",
                    "entry_ms_warm_median": "2",
                    "block_authorization_ms_warm_median": "3",
                    "graph_ms_warm_median": "4",
                    "nodes_visited_warm_median": "5",
                    "total_edges_scanned_warm_median": "6",
                    "total_distance_calcs_warm_median": "7",
                })

            rendered, markdown = deadline_results.amazon_profile_tables(rows, manifest)

            self.assertIn("10.000\\% & max & 0L-LNG", rendered)
            self.assertIn("| 10.000% | max | 0L-LNG |", markdown)
            rows[0]["total_edges_scanned_warm_median"] = "0"
            with self.assertRaisesRegex(ValueError, "edge counters are disabled"):
                deadline_results.amazon_profile_tables(rows, manifest)

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
