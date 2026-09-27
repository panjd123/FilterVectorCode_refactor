#!/usr/bin/env python3
"""Tests for the fail-closed authoritative LaTeX result generator."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import generate_authoritative_paper_results as generator


WORKLOADS = [
    (f"sel_{index}", value)
    for index, value in enumerate(generator.EXPECTED_AMAZON_SELECTIVITIES)
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def method(name: str, layers: int, topology: str, entry: str,
           role: str = "") -> dict[str, object]:
    result = {
        "name": name, "hierarchy_layers": [
            {"min_points": 1024 * (16 ** index), "topology": "lng"}
            for index in range(layers)
        ],
        "base_topology": topology, "entry_strategy": entry,
        "selection_role": role,
    }
    if role == generator.ROUTED_DRH_ROLE:
        result["routing_policy"] = "require_upper_authorization"
    return result


class AuthoritativePaperResultsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        methods = [
            method(generator.BASELINE_METHOD, 0, "lng", "optimized_lng"),
            method("l0_lng_entry_original", 0, "lng", "original"),
            method("l0_lng_entry_trie", 0, "lng", "trie"),
            method("l0_trie_entry_optimized_lng", 0, "trie", "optimized_lng"),
            method("l0_trie_entry_original", 0, "trie", "original"),
            method("l0_trie_entry_trie", 0, "trie", "trie"),
            method("gated_drh", 2, "lng", "optimized_lng",
                   generator.ROUTED_DRH_ROLE),
        ]
        self.formal_config = {
            "dataset": "Amazon", "methods": methods,
            "workloads": [
                {"name": name, "mean_selectivity": selectivity}
                for name, selectivity in WORKLOADS
            ],
            "recall_thresholds": {name: 0.9 for name, _ in WORKLOADS},
        }
        self.profile_config = self.formal_config
        self.heldout_configs = [
            {
                "dataset": dataset,
                "workloads": [{"name": f"{dataset.lower()}_query"}],
            }
            for dataset in sorted(generator.EXPECTED_HELDOUT_DATASETS)
        ]
        self.paths = self.make_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_fixture(self) -> generator.ResultPaths:
        formal = []
        profile = []
        for method_row in self.formal_config["methods"]:
            for index, (workload, selectivity) in enumerate(WORKLOADS):
                base = {
                    "workload": workload, "mean_selectivity": selectivity,
                    "method": method_row["name"],
                    "layer_count": len(method_row["hierarchy_layers"]),
                    "thresholds": ",".join(
                        str(layer["min_points"])
                        for layer in method_row["hierarchy_layers"]) or "none",
                    "base_topology": method_row["base_topology"],
                    "layer_topologies": ",".join(
                        str(layer["topology"])
                        for layer in method_row["hierarchy_layers"]) or "none",
                    "entry_strategy": method_row["entry_strategy"],
                    "routing_policy": "require_upper_authorization"
                    if method_row["name"] == "gated_drh" else "always_layered",
                    "lsearch": 100 + index, "recall_min": 0.91,
                    "qps_warm_median": 1000.0 + index, "target_recall": 0.9,
                }
                formal.append(base)
                profile.append({
                    **base, "query_total_ms_warm_median": 1.0,
                    "els_ms_warm_median": 0.1,
                    "entry_ms_warm_median": 0.1,
                    "block_authorization_ms_warm_median": 0.1,
                    "graph_ms_warm_median": 0.6,
                    "residual_ms_warm_median": 0.1,
                    "stage_closure_ms_at_batch_median": 0.0,
                    "layered_path_activation_rate_warm_median":
                        0.5 if method_row["name"] == "gated_drh" else 0.0,
                    "nodes_visited_warm_median": 100,
                    "regular_edges_scanned_warm_median": 120,
                    "special_intra_edges_scanned_warm_median": 50,
                    "special_inter_edges_scanned_warm_median": 30,
                    "total_edges_scanned_warm_median": 200,
                    "entry_point_distance_calcs_warm_median": 40,
                    "graph_search_distance_calcs_warm_median": 260,
                    "total_distance_calcs_warm_median": 300,
                })
        depth = []
        for index, (workload, selectivity) in enumerate(WORKLOADS):
            for category in generator.DEPTH_CATEGORIES:
                depth.append({
                    "status": "complete", "dataset": "Amazon",
                    "workload": workload, "mean_selectivity": selectivity,
                    "target_recall": 0.9, "category": category,
                    "method": category, "hierarchy": "none",
                    "entry_strategy": "optimized_lng", "lsearch": 100 + index,
                    "recall_min": 0.91, "qps": 1000 + index,
                    "speedup_vs_plain": 1.0, "speedup_ci95_low": 0.95,
                    "speedup_ci95_high": 1.05,
                })
        depth_global = [{
            "dataset": "Amazon", "category": category, "status": "complete",
            "method": category, "hierarchy": "none",
            "entry_strategy": "optimized_lng", "routing_policy": "always_layered",
            "geomean_qps": 1000, "speedup_vs_plain": 1.0,
            "workload_count": 9,
        } for category in generator.DEPTH_CATEGORIES]
        heldout = []
        heldout_global = []
        for config in self.heldout_configs:
            dataset = config["dataset"]
            workload = config["workloads"][0]["name"]
            heldout.append({
                "status": "complete", "dataset": dataset, "workload": workload,
                "mean_selectivity": 0.1, "target_recall": 0.9,
                "baseline_recall_min": 0.91, "automatic_hierarchy": "1024:lng",
                "automatic_recall_min": 0.91, "oracle_hierarchy": "2048:trie",
                "oracle_recall_min": 0.91, "automatic_speedup_vs_baseline": 2.0,
                "automatic_qps_fraction_of_oracle": 0.9,
                "automatic_speedup_ci95_low": 1.8,
                "automatic_speedup_ci95_high": 2.2,
                "automatic_oracle_fraction_ci95_low": 0.8,
                "automatic_oracle_fraction_ci95_high": 1.0,
            })
            heldout_global.append({
                "dataset": dataset, "status": "complete", "workload_count": 1,
                "automatic_speedup_vs_baseline": 2.0,
                "automatic_qps_fraction_of_global_oracle": 0.9,
            })
        build_summary = [
            {
                "component": "base", "structure": "zero_layer",
                "profile": "original_cpu", "measured_repeats": 5,
                "wall_median_seconds": 10, "wall_p95_seconds": 11,
                "index_median_mib": 100, "peak_rss_mib": 512,
                "peak_gpu_memory_mib": "",
                "gpu_required": "False", "gpu_exclusive_lock": "",
                "gpu_idle_samples_min": "",
            },
            {
                "component": "hierarchy", "structure": "auto_drh_v1",
                "profile": "full_gpu", "measured_repeats": 5,
                "wall_median_seconds": 2, "wall_p95_seconds": 2.2,
                "index_median_mib": 20, "peak_rss_mib": 256,
                "peak_gpu_memory_mib": 1024,
                "gpu_required": "True", "gpu_exclusive_lock": "False",
                "gpu_idle_samples_min": 3,
            },
        ]
        build_e2e = [{
            "structure": "auto_drh_v1", "hierarchy_profile": "full_gpu",
            "paired_repeats": 5, "original_cpu_base_median_seconds": 10,
            "accelerated_base_plus_hierarchy_median_seconds": 8,
            "speedup_vs_original_cpu": 1.25, "speedup_ci95_low": 1.1,
            "speedup_ci95_high": 1.4,
            "overhead_vs_accelerated_base_median": 1.2,
            "no_slower_supported": "True",
        }]
        files = {
            "amazon_formal": formal, "amazon_depth": depth,
            "amazon_depth_global": depth_global, "amazon_profile": profile,
            "heldout_workload": heldout, "heldout_global": heldout_global,
            "build_summary": build_summary, "build_end_to_end": build_e2e,
        }
        paths = {}
        for name, rows in files.items():
            path = self.root / f"{name}.csv"
            write_csv(path, rows)
            paths[name] = path
        return generator.ResultPaths(**paths)

    def generate(self) -> str:
        return generator.generate_document(
            self.paths, self.formal_config, self.profile_config,
            self.heldout_configs)

    def test_complete_fixture_generates_all_sections(self) -> None:
        output = self.generate()
        self.assertNotIn(r"\pending", output)
        for heading in (
            "Query Performance", "Automatic Versus Manual Hierarchies",
            "Mechanism Breakdown", "Construction",
        ):
            self.assertIn(heading, output)
        self.assertIn("99.001\\%", output)
        self.assertIn("source-sha256", output)
        self.assertIn("Special intra", output)
        self.assertIn("Entry dist.", output)
        self.assertIn("DRH/plain [95\\% CI]", output)
        self.assertIn("Host/GPU MiB", output)

    def test_missing_depth_cell_fails(self) -> None:
        with self.paths.amazon_depth.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        write_csv(self.paths.amazon_depth, rows[:-1])
        with self.assertRaisesRegex(ValueError, "depth matrix mismatch"):
            self.generate()

    def test_profile_must_use_formal_lsearch(self) -> None:
        with self.paths.amazon_profile.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["lsearch"] = "9999"
        write_csv(self.paths.amazon_profile, rows)
        with self.assertRaisesRegex(ValueError, "profile L differs"):
            self.generate()

    def test_profile_requires_activation_rate(self) -> None:
        with self.paths.amazon_profile.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["layered_path_activation_rate_warm_median"] = ""
        write_csv(self.paths.amazon_profile, rows)
        with self.assertRaisesRegex(ValueError, "missing numeric field"):
            self.generate()

    def test_build_requires_five_repeats(self) -> None:
        with self.paths.build_summary.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["measured_repeats"] = "4"
        write_csv(self.paths.build_summary, rows)
        with self.assertRaisesRegex(ValueError, "fewer than five"):
            self.generate()

    def test_atomic_output_is_unchanged_after_validation_failure(self) -> None:
        output = self.root / "generated.tex"
        output.write_text("sentinel\n", encoding="utf-8")
        with self.paths.heldout_global.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        write_csv(self.paths.heldout_global, rows[:-1])
        with self.assertRaisesRegex(ValueError, "global rows"):
            document = self.generate()
            generator.atomic_write(output, document)
        self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")

    def test_missing_validator_fails_closed(self) -> None:
        with self.assertRaises(FileNotFoundError):
            generator.validated_query_config(
                self.root / "missing-config.json",
                self.root / "missing-validator.py", "formal", 15)


if __name__ == "__main__":
    unittest.main()
