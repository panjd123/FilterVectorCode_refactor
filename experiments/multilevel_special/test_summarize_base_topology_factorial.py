import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("summarize_base_topology_factorial.py")
SPEC = importlib.util.spec_from_file_location("topology_factorial", MODULE_PATH)
assert SPEC and SPEC.loader
topology_factorial = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(topology_factorial)


class TopologyFactorialSummaryTest(unittest.TestCase):
    def test_method_map_covers_complete_binary_topology_factorial(self) -> None:
        expected = {
            "L", "T", "LL", "LT", "TL", "TT",
            "LLL", "LLT", "LTL", "LTT", "TLL", "TLT", "TTL", "TTT",
        }
        self.assertEqual(expected, set(topology_factorial.METHOD_TO_CODE.values()))

    def test_display_code_keeps_base_separate_from_overlays(self) -> None:
        self.assertEqual("0L[L]", topology_factorial.display_code("L"))
        self.assertEqual("1L[T|L]", topology_factorial.display_code("TL"))
        self.assertEqual("2L[L|LT]", topology_factorial.display_code("LLT"))

    def test_result_cell_exposes_no_crossing_max_recall(self) -> None:
        self.assertEqual(
            "NC (0.873)",
            topology_factorial.result_cell(
                {"warm_median_qps": "", "max_recall": "0.8728"}
            ),
        )
        self.assertEqual(
            "123.46",
            topology_factorial.result_cell(
                {"warm_median_qps": "123.456", "max_recall": "0.95"}
            ),
        )

    def test_normalize_crossings_accepts_legacy_qps_and_synthesizes_nc(self) -> None:
        keyed = topology_factorial.normalize_crossings(
            [{
                "workload": "sel_1",
                "method": "l0_lng_entry_optimized_lng",
                "mean_selectivity": "0.01",
                "qps_warm_median": "125.5",
                "recall": "0.91",
                "lsearch": "1000",
            }],
            [{
                "workload": "sel_1",
                "method": "l0_trie_entry_trie",
                "mean_selectivity": "0.01",
                "recall_min": "0.87",
            }],
        )
        self.assertEqual("complete", keyed[("sel_1", "L")]["status"])
        self.assertEqual(125.5, keyed[("sel_1", "L")]["warm_median_qps"])
        self.assertEqual("unavailable", keyed[("sel_1", "T")]["status"])

    def test_manifest_binary_hashes_uses_only_completed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"runs": [
                {"status": "complete", "search_binary_sha256": "abc"},
                {"status": "dry_run", "search_binary_sha256": "stale"},
            ]}))
            self.assertEqual({"abc"}, topology_factorial.manifest_binary_hashes(path))

    def test_manifest_build_hashes_reads_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"runs": [{
                "status": "complete",
                "source_provenance": {"build_binary_sha256": "builder"},
            }]}))
            self.assertEqual({"builder"}, topology_factorial.manifest_build_hashes(path))

    def test_fixed_table_keeps_nc_and_uses_fixed_comparison_denominators(self) -> None:
        rows = [
            {"workload": workload, "topology_code": code,
             "status": "unavailable" if code == "T" else "complete",
             "mean_selectivity": .01, "warm_median_qps": value}
            for workload in topology_factorial.WORKLOAD_ORDER
            for code, value in (("L", 10.), ("T", ""), ("TL", 20.), ("TLT", 40.))
        ]
        latex = "\n".join(topology_factorial.fixed_configuration_latex(rows))
        self.assertEqual(9, latex.count("10.00 & NC & 20.00 & 40.00 & 4.000 & 2.000"))

    def test_fixed_summary_does_not_reward_missing_crossings(self) -> None:
        rows = []
        workloads = list(topology_factorial.WORKLOAD_ORDER)
        for index, workload in enumerate(workloads):
            rows.append({
                "workload": workload,
                "topology_code": "L",
                "status": "complete",
                "warm_median_qps": 100.0,
            })
            rows.append({
                "workload": workload,
                "topology_code": "T",
                "status": "complete" if index < len(workloads) - 1 else "unavailable",
                "warm_median_qps": 200.0 if index < len(workloads) - 1 else "",
            })
        summary = {
            row["topology_code"]: row
            for row in topology_factorial.summarize_fixed_configurations(rows, ["L", "T"])
        }
        self.assertEqual(9, summary["L"]["recall_crossings"])
        self.assertEqual(8, summary["T"]["recall_crossings"])
        self.assertEqual("", summary["T"]["full_grid_rank"])
        self.assertEqual("", summary["T"]["full_grid_geomean_fraction_of_oracle"])
        self.assertAlmostEqual(0.5 ** (8.0 / 9.0), summary["L"]["full_grid_geomean_fraction_of_oracle"])


if __name__ == "__main__":
    unittest.main()
