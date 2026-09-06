#!/usr/bin/env python3
"""Consistency checks for generated paper tables."""

import csv
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_paper_results as generator
import audit_current_source_regression as current_source_audit
import summarize_rebuild_regression as rebuild_audit
import summarize_selection_sweep as selection_summary


class PaperResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        generator.main()
        path = SCRIPT_DIR / "results_summary" / "paper_results.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            cls.rows = list(csv.DictReader(stream))

    def test_complete_matrix(self) -> None:
        counts = Counter((row["comparison"], row["workload"]) for row in self.rows)
        for workload in generator.WORKLOADS:
            self.assertEqual(counts[("internal", workload)], 4)
            self.assertEqual(counts[("external", workload)], 6)

    def test_status_matches_measured_recall(self) -> None:
        for row in self.rows:
            recall = float(row["recall"])
            threshold = float(row["recall_threshold"])
            if row["status"] == "pass":
                self.assertGreaterEqual(recall, threshold, row)
            else:
                self.assertEqual(row["status"], "quality_limit")
                self.assertLess(recall, threshold, row)

    def test_sources_are_repository_relative(self) -> None:
        for row in self.rows:
            self.assertTrue(row["source"].startswith("results_summary/source/"), row)
            self.assertFalse(Path(row["source"]).is_absolute(), row)

    def test_expected_quality_limits(self) -> None:
        actual = {
            (row["workload"], row["method"])
            for row in self.rows if row["status"] == "quality_limit"
        }
        self.assertEqual(actual, {
            ("sel_10", "Single-level"),
            ("sel_0p5", "NaviX"), ("sel_1", "NaviX"),
            ("sel_0p5", "ACORN"), ("sel_1", "ACORN"),
            ("sel_10", "ACORN"),
        })

    def test_key_claim_values(self) -> None:
        by_key = {(r["comparison"], r["workload"], r["method"]): r for r in self.rows}
        self.assertAlmostEqual(float(by_key[("internal", "sel_50", "Tuned Multi-level")]["batch_median_ms"]), 383.260)
        self.assertAlmostEqual(float(by_key[("internal", "sel_75", "Tuned Multi-level")]["speedup_vs_plain"]), 57.376549)
        self.assertAlmostEqual(float(by_key[("internal", "sel_1", "Original Multi-level")]["batch_median_ms"]), 141.684)
        self.assertEqual(
            by_key[("internal", "sel_25", "Original Multi-level")]["source"],
            "results_summary/source/internal_paired_formal_sel25.csv",
        )

    def test_all_formal_internal_manifests_use_one_binary(self) -> None:
        for path in generator.FORMAL_INTERNAL_MANIFESTS:
            payload = json.loads(path.read_text(encoding="utf-8"))
            hashes = {run["search_binary_sha256"] for run in payload["runs"]}
            self.assertEqual(hashes, {generator.EXPECTED_SEARCH_BINARY_SHA256}, path)

    def test_upper_level_ablation_is_preserved_in_measured_pool(self) -> None:
        path = SCRIPT_DIR / "results_summary" / "internal_canonical_measured_points.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        ablations = [row for row in rows if row["method"] == "Upper-level ablation"]
        self.assertEqual({row["workload"] for row in ablations}, {"sel_25", "sel_50", "sel_75"})
        self.assertEqual(len(ablations), 3)

    def test_manifest_validation_rejects_mixed_binary(self) -> None:
        payload = {"runs": [{"status": "complete", "returncode": 0, "search_binary_sha256": "wrong"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "mixed search binaries"):
                generator.validate_binary_manifest(path)

    def test_every_internal_aggregate_exactly_matches_manifest_grid(self) -> None:
        repeat_counts = generator.validate_all_internal_sources()
        self.assertTrue(repeat_counts)
        self.assertTrue(all(repeats == 6 for repeats in repeat_counts.values()))

    def test_manifest_grid_validation_rejects_missing_budget(self) -> None:
        source_csv = generator.HIGH_PAIRED["sel_25"]
        source_manifest = generator.HIGH_PAIRED_MANIFEST["sel_25"]
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "points.csv"
            rows = source_csv.read_text(encoding="utf-8").splitlines()
            csv_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "aggregate grid mismatch"):
                generator.validate_internal_source(csv_path, source_manifest)

    def test_build_results_are_source_data_and_manifested(self) -> None:
        self.assertTrue(generator.BUILD_SOURCE.is_file())
        source_rows = generator.read(generator.BUILD_SOURCE)
        output_rows = generator.read(SCRIPT_DIR / "results_summary" / "build_results.csv")
        self.assertEqual(output_rows, source_rows)
        manifest_rows = generator.read(SCRIPT_DIR / "results_summary" / "source_manifest.csv")
        self.assertIn(
            "results_summary/source/build_results_source.csv",
            {row["path"] for row in manifest_rows},
        )

    def test_csv_writers_use_repository_lf_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.csv"
            rows = [{"name": "example", "value": 1}]
            generator.write_csv(output, ["name", "value"], rows)
            self.assertNotIn(b"\r\n", output.read_bytes())
            selection_summary.write_csv(output, rows)
            self.assertNotIn(b"\r\n", output.read_bytes())

    def test_current_source_regression_is_complete_and_recall_stable(self) -> None:
        rows = generator.read(
            SCRIPT_DIR / "results_summary" / "current_source_regression.csv"
        )
        self.assertEqual(len(rows), 24)
        self.assertEqual(
            Counter(row["workload"] for row in rows),
            Counter({workload: 4 for workload in generator.WORKLOADS}),
        )
        self.assertEqual(
            {row["fresh_binary_sha256"] for row in rows},
            {"88d7dba189478cd11402a8433076d220c7ad68ca9f9a6366118f78430752f37f"},
        )
        special = [row for row in rows if row["method"] != "Plain UNG"]
        self.assertTrue(all(float(row["recall_delta"]) == 0.0 for row in special))
        low = [row for row in rows if row["workload"] == "sel_0p5"]
        self.assertTrue(all(int(row["measured_repeats"]) == 20 for row in low))
        self.assertEqual(set(current_source_audit.FRESH_CONFIGS), {"0p5", "1", "10", "25", "50", "75"})

    def test_rebuild_regression_preserves_structure_and_robust_quality(self) -> None:
        builds = generator.read(
            SCRIPT_DIR / "results_summary/current_source_rebuilds.csv"
        )
        self.assertEqual(len(builds), 3)
        self.assertEqual(len({row["builder_sha256"] for row in builds}), 1)
        self.assertEqual(
            {row["builder_sha256_source"] for row in builds},
            {"historical_audit_record"},
        )
        for field in ("special_blocks_sha256", "special_trie_sha256", "regular_edges_sha256"):
            self.assertEqual(len({row[field] for row in builds}), 1, field)
        self.assertEqual({row["inter_edge_count"] for row in builds}, {"14965020"})
        self.assertGreater(len({row["special_edges_sha256"] for row in builds}), 1)

        main = generator.read(
            SCRIPT_DIR / "results_summary/current_source_rebuild_query_regression.csv"
        )
        self.assertEqual(len(main), 12)
        self.assertEqual(sum(row["passes_recall_threshold"] == "0" for row in main), 1)
        failed = [row for row in main if row["passes_recall_threshold"] == "0"]
        self.assertEqual((failed[0]["rebuild"], failed[0]["workload"], failed[0]["lsearch"]),
                         ("2", "sel_50", "500"))

        sweep = generator.read(
            SCRIPT_DIR / "results_summary/current_source_rebuild_sel50_l_sweep.csv"
        )
        self.assertEqual(len(sweep), 3 * len(rebuild_audit.L_VALUES))
        robust = [row for row in sweep if row["robust_selected"] == "1"]
        self.assertEqual(len(robust), 3)
        self.assertEqual({row["lsearch"] for row in robust}, {"550"})
        self.assertTrue(all(float(row["recall"]) >= 0.85 for row in robust))

        backends = generator.read(
            SCRIPT_DIR / "results_summary/current_source_build_backend.csv"
        )
        self.assertEqual({row["large_intra_backend"] for row in backends},
                         {"GPU FastGrnnd", "CPU Vamana"})
        gpu = next(row for row in backends if row["large_intra_backend"] == "GPU FastGrnnd")
        self.assertEqual(gpu["build_samples"], "3")
        self.assertGreater(float(gpu["total_speedup_vs_cpu_vamana"]), 10.0)


if __name__ == "__main__":
    unittest.main()
