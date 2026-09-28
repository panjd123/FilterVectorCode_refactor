#!/usr/bin/env python3
"""Tests for the fail-closed authoritative LaTeX result generator."""

from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import generate_authoritative_paper_results as generator


SCRIPT_DIR = Path(__file__).resolve().parent
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
    def test_recall_qps_figures_include_direct_depth_and_topology_ablations(self):
        self.assertEqual(
            tuple(name for name, _ in generator.RECALL_QPS_FIGURES),
            (
                "principal_zero",
                "representative_depth",
                "upper_authorization",
            ),
        )

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
        for name, thresholds, topologies, entry in (
            ("l1_t1024_lng_entry_optimized_lng", (1024,), ("lng",),
             "optimized_lng"),
            ("l1_t8192_lng_entry_optimized_lng", (8192,), ("lng",),
             "optimized_lng"),
            ("l2_t1024_16384_ll_entry_optimized_lng", (1024, 16384),
             ("lng", "lng"), "optimized_lng"),
            ("l2_t1024_16384_lt_entry_optimized_lng", (1024, 16384),
             ("lng", "trie"), "optimized_lng"),
            ("l2_t1024_16384_tl_entry_optimized_lng", (1024, 16384),
             ("trie", "lng"), "optimized_lng"),
            ("l2_t1024_16384_tt_entry_optimized_lng", (1024, 16384),
             ("trie", "trie"), "optimized_lng"),
            ("l2_t8192_131072_lt_entry_optimized_lng", (8192, 131072),
             ("lng", "trie"), "optimized_lng"),
            ("l2_t1024_16384_lt_entry_original", (1024, 16384),
             ("lng", "trie"), "original"),
            ("l2_t1024_16384_lt_entry_trie", (1024, 16384),
             ("lng", "trie"), "trie"),
        ):
            row = method(name, len(topologies), "lng", entry)
            for layer, threshold, topology in zip(
                    row["hierarchy_layers"], thresholds, topologies):
                layer["min_points"] = threshold
                layer["topology"] = topology
            methods.append(row)
        self.formal_config = {
            "dataset": "Amazon", "methods": methods,
            "num_threads": 100,
            "workloads": [
                {"name": name, "mean_selectivity": selectivity}
                for name, selectivity in WORKLOADS
            ],
            "recall_thresholds": {name: 0.9 for name, _ in WORKLOADS},
            "selection_provenance": {
                "shared_max_lsearch_by_workload": dict(zip(
                    (name for name, _ in WORKLOADS),
                    (2000, 2200, 5000, 64000, 150000, 320000,
                     602453, 602453, 602453))),
                "upstream_excluded_no_crossing": [],
                "excluded_no_crossing": [],
            },
        }
        self.screen_config = copy.deepcopy(self.formal_config)
        for method_row in self.screen_config["methods"]:
            method_row["lsearch_values_by_workload"] = {
                name: [100 + index]
                for index, (name, _) in enumerate(WORKLOADS)
            }
        self.profile_config = self.formal_config
        self.heldout_configs = []
        self.heldout_policies = {}
        self.heldout_policy_paths = []
        num_points = {
            "Genome": 108077, "Reviews": 288065, "VariousImg": 758935,
        }
        for dataset in sorted(generator.EXPECTED_HELDOUT_DATASETS):
            workload = {
                "name": f"{dataset.lower()}_query",
                "query_dir": f"{dataset.lower()}_query",
                "mean_selectivity": 0.1,
                "num_queries": 3000,
            }
            layers = generator.derive_plan(num_points[dataset], 64, 4)
            config_layers = [
                {"min_points": layer["min_points"],
                 "topology": layer["topology"]}
                for layer in layers
            ]
            automatic_name = f"{dataset.lower()}_automatic"
            unrouted_name = f"{dataset.lower()}_automatic_unrouted"
            methods = [{
                "name": generator.BASELINE_METHOD,
                "selection_role": generator.HELDOUT_BASELINE_ROLE,
                "base_topology": "lng", "entry_strategy": "optimized_lng",
                "hierarchy_layers": [], "special_block_search": False,
            }, {
                "name": automatic_name,
                "selection_role": generator.HELDOUT_AUTOMATIC_ROLE,
                "base_topology": "lng", "entry_strategy": "optimized_lng",
                "hierarchy_layers": config_layers,
                "special_block_search": True,
                "routing_policy": generator.HELDOUT_ROUTING_POLICY,
            }, {
                "name": unrouted_name,
                "selection_role": generator.HELDOUT_UNROUTED_ROLE,
                "base_topology": "lng", "entry_strategy": "optimized_lng",
                "hierarchy_layers": config_layers,
                "special_block_search": True,
            }]
            for index in range(35):
                depth = 1 + index % 3
                threshold = 100 + index
                manual_layers = [
                    {
                        "min_points": threshold * (16 ** level),
                        "topology": "lng" if level == 0 else "trie",
                    }
                    for level in range(depth)
                ]
                methods.append({
                    "name": f"{dataset.lower()}_manual_{index}",
                    "selection_role": generator.HELDOUT_MANUAL_ROLE,
                    "base_topology": "lng",
                    "entry_strategy": "optimized_lng",
                    "hierarchy_layers": manual_layers,
                    "special_block_search": True,
                    "routing_policy": generator.HELDOUT_ROUTING_POLICY,
                })
            config = {
                "dataset": dataset,
                "num_threads": 100,
                "expected_num_points": num_points[dataset],
                "methods": methods,
                "workloads": [workload],
            }
            policy = {
                "schema_version": 2,
                "dataset": dataset,
                "policy": "gated_degree_ratio_hierarchy_v1",
                "query_calibrated": False,
                "inputs": {
                    "num_points": num_points[dataset], "dimension": 128,
                    "max_degree": 64, "num_cross_edges": 4,
                    "scale_ratio": 16,
                },
                "automatic_hierarchy_layers": layers,
                "automatic_method": automatic_name,
                "automatic_routing_policy":
                    generator.HELDOUT_ROUTING_POLICY,
                "unrouted_ablation_method": unrouted_name,
                "manual_grid_frozen_before_search": True,
                "manual_grid_uses_automatic_routing_policy": True,
                "frozen_hierarchy_candidates": 36,
                "manual_alternatives": 35,
                "manual_depths": [1, 2, 3],
                "workloads": [workload],
            }
            policy_path = self.root / f"{dataset.lower()}_policy.json"
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            self.heldout_configs.append(config)
            self.heldout_policies[dataset] = policy
            self.heldout_policy_paths.append(policy_path)
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
                    "num_entries_warm_median": 16,
                    "entry_group_matched_points_warm_median": 1024,
                })
        depth = []
        for index, (workload, selectivity) in enumerate(WORKLOADS):
            for category in generator.DEPTH_CATEGORIES:
                depth.append({
                    "status": "complete", "dataset": "Amazon",
                    "workload": workload, "mean_selectivity": selectivity,
                    "target_recall": 0.9, "category": category,
                    "method": ("gated_drh" if category == "automatic_routed"
                               else category),
                    "hierarchy": "none",
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
        build_summary = []
        for profile_name in generator.BASE_BUILD_PROFILES:
            gpu = profile_name in {
                "naive_gpu", "paper_fused", "accelerated_gpu"}
            build_summary.append({
                "component": "base", "structure": "zero_layer",
                "profile": profile_name, "measured_repeats": 5,
                "wall_median_seconds": 10, "wall_p95_seconds": 11,
                "index_median_mib": 100, "peak_rss_mib": 512,
                "peak_gpu_memory_mib": 1024 if gpu else "",
                "gpu_required": str(gpu), "gpu_exclusive_lock": "False",
                "gpu_idle_samples_min": 3 if gpu else "",
            })
        build_e2e = []
        for structure in generator.HIERARCHY_BUILD_STRUCTURES:
            for profile_name in generator.HIERARCHY_BUILD_PROFILES:
                gpu = profile_name != "cpu"
                build_summary.append({
                    "component": "hierarchy", "structure": structure,
                    "profile": profile_name, "measured_repeats": 5,
                    "wall_median_seconds": 2, "wall_p95_seconds": 2.2,
                    "index_median_mib": 20, "peak_rss_mib": 256,
                    "peak_gpu_memory_mib": 1024 if gpu else "",
                    "gpu_required": str(gpu),
                    "gpu_exclusive_lock": "False",
                    "gpu_idle_samples_min": 3 if gpu else "",
                })
                build_e2e.append({
                    "structure": structure,
                    "hierarchy_profile": profile_name,
                    "stage_repeats": 5,
                    "composition_method":
                        "sum_of_stage_medians_independent_bootstrap",
                    "original_cpu_base_median_seconds": 10,
                    "composed_base_plus_hierarchy_seconds": 8,
                    "speedup_vs_original_cpu": 1.25,
                    "speedup_ci95_low": 1.1,
                    "speedup_ci95_high": 1.4,
                    "overhead_vs_accelerated_base_median": 1.2,
                    "no_slower_supported": "True",
                })
        files = {
            "amazon_formal": formal, "amazon_depth": depth,
            "amazon_depth_global": depth_global, "amazon_profile": profile,
            "heldout_workload": heldout, "heldout_global": heldout_global,
            "build_summary": build_summary, "build_end_to_end": build_e2e,
            "build_quality_formal": formal,
        }
        paths = {}
        for name, rows in files.items():
            path = self.root / f"{name}.csv"
            write_csv(path, rows)
            paths[name] = path
        self.screen_points = self.root / "screen_points.csv"
        write_csv(self.screen_points, formal)
        return generator.ResultPaths(**paths)

    def generate(self) -> str:
        return generator.generate_document(
            self.paths, self.screen_config, self.screen_points,
            self.formal_config, self.profile_config,
            self.heldout_configs, list(self.heldout_policies.values()),
            self.formal_config)

    def test_complete_fixture_generates_all_sections(self) -> None:
        output = self.generate()
        self.assertNotIn(r"\pending", output)
        for heading in (
            "Query Performance", "Automatic Versus Manual Hierarchies",
            "Mechanism Breakdown", "Construction",
        ):
            self.assertIn(heading, output)
        self.assertIn(r"\newcommand{\authoritativeAbstractResult}", output)
        self.assertIn(r"\newcommand{\authoritativeConclusionResult}", output)
        self.assertIn(r"\newcommand{\authoritativeDatasetTable}", output)
        self.assertIn(
            r"\newcommand{\authoritativeSharedSearchBudgetTable}", output)
        self.assertIn(r"\newcommand{\authoritativeResults}", output)
        self.assertIn(r"\newcommand{\authoritativeScreenAppendix}", output)
        self.assertIn("Complete Amazon Screen Matrix", output)
        self.assertIn("M01", output)
        self.assertIn("minimum-warm-Recall", output)
        self.assertIn("9/9 selectivity workloads", output)
        self.assertIn("99.001\\%", output)
        self.assertIn("source-sha256", output)
        self.assertEqual(
            output.count("% source-sha256"),
            len(self.paths.__dataclass_fields__))
        for field in self.paths.__dataclass_fields__:
            self.assertIn(f"% source-sha256 {field} ", output)
        self.assertIn("Special intra", output)
        self.assertIn("Entry dist.", output)
        self.assertIn("Principal zero-layer comparison", output)
        self.assertIn("Trie/LNG", output)
        self.assertIn("Observed Query Regimes", output)
        self.assertIn("diagnostic profile ratios", output)
        self.assertIn("per-query mean", output)
        self.assertIn("rather than sums of rounded table cells", output)
        self.assertIn("DRH/plain [95\\% CI]", output)
        self.assertIn("Host/GPU MiB", output)
        self.assertIn("Downstream query quality", output)
        self.assertIn("Two-layer topology ablation", output)
        self.assertIn("Entry-strategy ablation", output)
        self.assertIn("Threshold sensitivity", output)
        self.assertIn(r"\label{tab:amazon-two-layer-topology}", output)
        self.assertIn(r"\label{tab:amazon-drh-entry}", output)
        self.assertIn(r"\label{tab:amazon-threshold-depth}", output)
        self.assertIn("Held-out datasets and query-independent DRH plans", output)
        self.assertIn("Genome", output)
        self.assertIn("108,077", output)
        self.assertIn("Shared maximum measured", output)
        self.assertIn("602,453", output)
        self.assertIn(r"\label{tab:shared-lsearch-budgets}", output)

    def test_indirect_provenance_hashes_validator_and_manifests(self) -> None:
        validator = self.root / "validator.py"
        query_manifest = self.root / "query" / "manifest.json"
        build_manifest = self.root / "build" / "manifest.json"
        validator.write_text("# validator\n", encoding="utf-8")
        query_manifest.parent.mkdir()
        build_manifest.parent.mkdir()
        query_manifest.write_text('{"runs": []}\n', encoding="utf-8")
        build_manifest.write_text('{"runs": []}\n', encoding="utf-8")
        query_config_path = self.root / "query.json"
        build_config_path = self.root / "build.json"
        lines = generator.render_indirect_provenance(
            validator, [query_config_path],
            [{"output_root": str(query_manifest.parent)}],
            [build_config_path],
            [{"output_root": str(build_manifest.parent)}])
        rendered = "\n".join(lines)
        self.assertIn(
            f"validator-sha256 validator.py {generator.source_digest(validator)}",
            rendered)
        self.assertIn(
            f"manifest-sha256 query.json {generator.source_digest(query_manifest)}",
            rendered)
        self.assertIn(
            f"manifest-sha256 build.json {generator.source_digest(build_manifest)}",
            rendered)

    def test_recall_qps_figures_require_complete_bound_artifacts(self) -> None:
        figure_dir = self.root / "figures"
        figure_dir.mkdir()
        screen_config = self.root / "screen.json"
        screen_points = self.root / "all_points.csv"
        screen_config.write_text('{"phase": "screen"}\n', encoding="utf-8")
        screen_points.write_text("recall,qps\n0.9,100\n", encoding="utf-8")
        families = {}
        for family, _ in generator.RECALL_QPS_FIGURES:
            families[family] = {"missing_method_workloads": []}
            for suffix in ("pdf", "png"):
                (figure_dir / f"{family}.{suffix}").write_bytes(
                    f"{family}.{suffix}".encode())
        manifest = {
            "allow_partial": False,
            "config_sha256": generator.source_digest(screen_config),
            "all_points_sha256": generator.source_digest(screen_points),
            "families": families,
        }
        (figure_dir / "plot_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        pdfs, inputs = generator.validate_recall_qps_figures(
            figure_dir, screen_config, screen_points)
        self.assertEqual(set(pdfs), set(families))
        self.assertEqual(len(inputs), 1 + 2 * len(families))
        manifest["allow_partial"] = True
        (figure_dir / "plot_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "partial"):
            generator.validate_recall_qps_figures(
                figure_dir, screen_config, screen_points)

    def test_markdown_report_contains_all_authoritative_sections(self) -> None:
        self.generate()
        figures = {
            family: self.root / "figures" / f"{family}.pdf"
            for family, _ in generator.RECALL_QPS_FIGURES
        }
        report = generator.generate_markdown_report(
            self.paths, self.screen_config, self.screen_points,
            self.formal_config, self.profile_config, self.heldout_configs,
            list(self.heldout_policies.values()), self.formal_config, figures,
            self.root / "report.md")
        for heading in (
            "方法与评估口径", "Amazon 层数消融", "零层 Trie 与 LNG",
            "无校准 DRH 与冻结人工 Oracle", "查询阶段与工作量 Breakdown",
            "GPU 辅助构建", "Recall-QPS 曲线", "解释边界",
            "零层 topology 与入口策略完整消融",
            "两层逐层 topology 消融", "固定 DRH 的入口策略消融",
            "层数与阈值尺度消融", "全部 screen 方法的",
        ):
            self.assertIn(heading, report)
        self.assertIn("Entry distances", report)
        self.assertIn("Special inter", report)
        self.assertIn("Total edges", report)
        self.assertIn("Matched points", report)
        self.assertIn("构建产物的下游查询质量", report)
        self.assertIn("Base graph 的五个冻结 profile", report)
        self.assertIn("Hierarchy sidecar 的 intra 与 inter backend 独立控制", report)
        self.assertIn("grouped fused distance/top-k", report)
        self.assertIn("mode=tf32_wmma", report)
        self.assertIn("预声明 quality cohort", report)
        self.assertIn("同一个配置覆盖全部九档选择率的汇总", report)
        self.assertIn("LNG + original", report)
        self.assertIn("Trie + optimized LNG", report)
        self.assertIn("GPU evidence", report)
        self.assertIn("idle-3", report)
        self.assertIn("逐 Lsearch 原始点", report)
        self.assertIn("100 个查询工作线程", report)
        self.assertIn("query-level parallel batch throughput", report)
        self.assertIn("共同最大实测", report)
        self.assertIn("共同 Recall 搜索预算", report)
        self.assertIn("602,453", report)
        self.assertIn("L/Q/R", report)
        self.assertIn("先在单个 warm repeat 内取每查询均值", report)
        self.assertIn("不由表中四舍五入后的分项相加", report)
        self.assertNotIn("pending", report.lower())

    def test_screen_matrix_reports_crossing_or_best_measured_recall(self) -> None:
        rows = generator.read_csv(
            self.screen_points, generator.SCREEN_POINT_FIELDS)
        key = (generator.BASELINE_METHOD, WORKLOADS[0][0])
        point = next(row for row in rows
                     if (row["method"], row["workload"]) == key)
        point["recall_min"] = "0.89"
        summary = generator.summarize_screen_points(
            rows, self.screen_config,
            [name for name, _ in WORKLOADS])
        self.assertEqual(summary[key]["status"], "no_crossing")
        self.assertEqual(summary[key]["lsearch"], 100)

        crossing = copy.deepcopy(point)
        crossing["lsearch"] = "200"
        crossing["recall_min"] = "0.92"
        rows.append(crossing)
        earlier_crossing = copy.deepcopy(point)
        earlier_crossing["lsearch"] = "150"
        earlier_crossing["recall_min"] = "0.90"
        rows.append(earlier_crossing)
        method_config = next(
            row for row in self.screen_config["methods"]
            if row["name"] == generator.BASELINE_METHOD)
        method_config["lsearch_values_by_workload"][WORKLOADS[0][0]] = [
            100, 150, 200]
        summary = generator.summarize_screen_points(
            rows, self.screen_config,
            [name for name, _ in WORKLOADS])
        self.assertEqual(summary[key]["status"], "complete")
        self.assertEqual(summary[key]["lsearch"], 150)

    def test_screen_matrix_rejects_duplicate_lsearch(self) -> None:
        rows = generator.read_csv(
            self.screen_points, generator.SCREEN_POINT_FIELDS)
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate L"):
            generator.summarize_screen_points(
                rows, self.screen_config,
                [name for name, _ in WORKLOADS])

    def test_markdown_report_rejects_invalid_build_evidence(self) -> None:
        with self.paths.build_summary.open(
                newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        gpu = next(row for row in rows if row["gpu_required"] == "True")
        gpu["gpu_idle_samples_min"] = "2"
        write_csv(self.paths.build_summary, rows)
        figures = {
            family: self.root / "figures" / f"{family}.pdf"
            for family, _ in generator.RECALL_QPS_FIGURES
        }
        with self.assertRaisesRegex(ValueError, "idle preflight"):
            generator.generate_markdown_report(
                self.paths, self.screen_config, self.screen_points,
                self.formal_config, self.profile_config, self.heldout_configs,
                list(self.heldout_policies.values()), self.formal_config,
                figures, self.root / "report.md")

    def test_query_protocol_rejects_mixed_thread_counts(self) -> None:
        configs = [copy.deepcopy(self.formal_config) for _ in range(2)]
        configs[1]["num_threads"] = 99
        with self.assertRaisesRegex(ValueError, "inconsistent num_threads"):
            generator.validate_query_thread_count(configs)

    def test_formal_provenance_binds_nc_to_shared_budget(self) -> None:
        config = copy.deepcopy(self.formal_config)
        budgets = {name: 1000 for name, _ in WORKLOADS}
        config["selection_provenance"] = {
            "shared_max_lsearch_by_workload": budgets,
            "upstream_excluded_no_crossing": [],
            "excluded_no_crossing": [],
        }
        generator.validate_amazon_formal_selection_provenance(config)

        method_row = config["methods"][0]
        workload = WORKLOADS[0][0]
        method_row["enabled_workloads"] = [
            name for name, _ in WORKLOADS if name != workload]
        config["selection_provenance"]["excluded_no_crossing"] = [{
            "method": method_row["name"], "workload": workload,
            "max_measured_lsearch": 999,
            "target_recall": 0.9,
        }]
        with self.assertRaisesRegex(ValueError, "differs from shared budget"):
            generator.validate_amazon_formal_selection_provenance(config)

    def test_document_and_report_validate_formal_provenance(self) -> None:
        del self.formal_config["selection_provenance"]
        with self.assertRaisesRegex(ValueError, "lacks selection provenance"):
            self.generate()
        figures = {
            family: self.root / "figures" / f"{family}.pdf"
            for family, _ in generator.RECALL_QPS_FIGURES
        }
        with self.assertRaisesRegex(ValueError, "lacks selection provenance"):
            generator.generate_markdown_report(
                self.paths, self.screen_config, self.screen_points,
                self.formal_config, self.profile_config, self.heldout_configs,
                list(self.heldout_policies.values()), self.formal_config,
                figures, self.root / "report.md")

    def test_build_claim_must_follow_confidence_interval(self) -> None:
        with self.paths.build_end_to_end.open(
                newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["no_slower_supported"] = "False"
        write_csv(self.paths.build_end_to_end, rows)
        with self.assertRaisesRegex(ValueError, "disagrees with CI"):
            self.generate()

    def test_principal_zero_layer_table_marks_missing_trie_crossing(self) -> None:
        formal_rows = generator.read_csv(self.paths.amazon_formal,
                                         generator.QUERY_FIELDS)
        formal = generator.validate_formal_rows(
            formal_rows, self.formal_config, "Amazon formal")
        missing = (generator.PRINCIPAL_TRIE_METHOD, WORKLOADS[-1][0])
        del formal[missing]
        table = "\n".join(generator.render_principal_zero_layer_by_workload(
            formal, [WORKLOADS[-1][0]]))
        self.assertIn("NC & NC", table)

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

    def test_profile_rejects_negative_breakdown_metric(self) -> None:
        with self.paths.amazon_profile.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["graph_ms_warm_median"] = "-0.1"
        write_csv(self.paths.amazon_profile, rows)
        with self.assertRaisesRegex(ValueError, "negative profile metric"):
            self.generate()

    def test_profile_rejects_edge_total_below_component(self) -> None:
        with self.paths.amazon_profile.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["total_edges_scanned_warm_median"] = "100"
        write_csv(self.paths.amazon_profile, rows)
        with self.assertRaisesRegex(ValueError, "total edge count"):
            self.generate()

    def test_profile_rejects_distance_total_below_component(self) -> None:
        with self.paths.amazon_profile.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["total_distance_calcs_warm_median"] = "200"
        write_csv(self.paths.amazon_profile, rows)
        with self.assertRaisesRegex(ValueError, "total distance count"):
            self.generate()

    def test_build_requires_five_repeats(self) -> None:
        with self.paths.build_summary.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["measured_repeats"] = "4"
        write_csv(self.paths.build_summary, rows)
        with self.assertRaisesRegex(ValueError, "fewer than five"):
            self.generate()

    def test_build_config_binds_command_snapshot_to_manifest_hash(self) -> None:
        output_root = self.root / "build-output"
        case_root = output_root / "case0"
        case_root.mkdir(parents=True)
        executable = self.root / "build_UNG_index.snapshot"
        executable.write_bytes(b"actual build binary")
        (case_root / "command.txt").write_text(
            f"{executable} --flag value\n", encoding="utf-8")
        config = {
            "build_app": str(self.root / "build_UNG_index"),
            "output_root": str(output_root),
            "resource_profile": False,
            "cases": [{"name": "case0"}],
        }
        config_path = self.root / "build-config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        (output_root / "manifest.json").write_text(json.dumps({"runs": [{
            "name": "case0", "status": "complete", "returncode": 0,
            "elapsed_seconds": 1.0, "case_root": str(case_root),
            "provenance": {"build_binary_sha256": "0" * 64},
        }]}), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "command binary differs"):
            generator.validate_build_config(config_path)

    def test_build_config_set_rejects_timing_resource_binary_drift(self) -> None:
        configs = []
        for builder, component in (
                ("build_UNG_index", "base"),
                ("build_special_block_index", "hierarchy")):
            for resource in (False, True):
                configs.append({
                    "build_app": str(self.root / builder),
                    "resource_profile": resource,
                    "_validated_build_binary_sha256":
                        (("a" if component == "base" and not resource else "b")
                         * 64),
                })
        with self.assertRaisesRegex(ValueError, "build binary mismatch"):
            generator.validate_build_config_set(configs)

    def test_build_summary_requires_complete_frozen_matrix(self) -> None:
        with self.paths.build_summary.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        write_csv(self.paths.build_summary, rows[:-1])
        with self.assertRaisesRegex(ValueError, "build summary matrix mismatch"):
            self.generate()

    def test_composed_build_requires_complete_frozen_matrix(self) -> None:
        with self.paths.build_end_to_end.open(
                newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        write_csv(self.paths.build_end_to_end, rows[:-1])
        with self.assertRaisesRegex(ValueError, "end-to-end build matrix mismatch"):
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

    def test_query_config_rejects_manifest_binary_drift(self) -> None:
        output = self.root / "query-output"
        output.mkdir()
        config = {
            "output_root": str(output),
            "expected_search_binary_sha256": "expected",
            "protocol": {
                "phase": "formal", "cold_repeats": 1,
                "measured_repeats": 15, "recall_rule": "all_repeats",
            },
            "methods": [{"name": "method"}],
            "workloads": [{"name": "workload"}],
        }
        config_path = self.root / "query-config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        (output / "manifest.json").write_text(json.dumps({
            "runs": [{
                "method": "method", "workload": "workload",
                "search_binary_sha256": "observed",
            }],
        }), encoding="utf-8")
        validator = self.root / "validator.py"
        validator.write_text("raise SystemExit(0)\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from pinned binary"):
            generator.validated_query_config(
                config_path, validator, "formal", 15)

    def test_build_quality_must_use_performance_binary(self) -> None:
        generator.validate_performance_binary_hashes(
            "same", {"same"}, "same")
        with self.assertRaisesRegex(
                ValueError, "build-quality formal runs must use one immutable"):
            generator.validate_performance_binary_hashes(
                "performance", {"performance"}, "different")

    def test_build_quality_identity_parses_generated_method_names(self) -> None:
        expected = {
            "quality_base_original_cpu": ("base", "original_cpu"),
            "quality_base_accelerated_gpu": ("base", "accelerated_gpu"),
            "quality_single_t1024_lng_cpu": ("single_t1024_lng", "cpu"),
            "quality_single_t1024_lng_full_gpu":
                ("single_t1024_lng", "full_gpu"),
            "quality_single_t1024_lng_full_gpu_wmma":
                ("single_t1024_lng", "full_gpu_wmma"),
            "quality_auto_drh_v1_cpu": ("auto_drh_v1", "cpu"),
            "quality_auto_drh_v1_full_gpu": ("auto_drh_v1", "full_gpu"),
            "quality_auto_drh_v1_full_gpu_wmma":
                ("auto_drh_v1", "full_gpu_wmma"),
        }
        self.assertEqual(
            {name: generator.build_quality_identity(name) for name in expected},
            expected)

    def test_heldout_policies_bind_to_frozen_formal_configs(self) -> None:
        validated = generator.validate_heldout_policy_set(
            self.heldout_policy_paths, self.heldout_configs)
        self.assertEqual(
            [row[0] for row in validated],
            sorted(generator.EXPECTED_HELDOUT_DATASETS))
        self.assertTrue(all(len(generator.source_digest(row[1])) == 64
                            for row in validated))

    def test_heldout_policy_rejects_query_calibration(self) -> None:
        config = self.heldout_configs[0]
        policy = copy.deepcopy(self.heldout_policies[config["dataset"]])
        policy["query_calibrated"] = True
        path = self.root / "calibrated-policy.json"
        path.write_text(json.dumps(policy), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must not be query calibrated"):
            generator.validate_heldout_policy_protocol(path, config)

    def test_heldout_policy_rederives_automatic_layers(self) -> None:
        config = self.heldout_configs[0]
        policy = copy.deepcopy(self.heldout_policies[config["dataset"]])
        policy["automatic_hierarchy_layers"][0]["min_points"] *= 2
        path = self.root / "changed-layers-policy.json"
        path.write_text(json.dumps(policy), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match DRH derivation"):
            generator.validate_heldout_policy_protocol(path, config)

    def test_heldout_policy_rejects_wrong_frozen_candidate_count(self) -> None:
        config = self.heldout_configs[0]
        policy = copy.deepcopy(self.heldout_policies[config["dataset"]])
        policy["frozen_hierarchy_candidates"] = 35
        path = self.root / "wrong-frozen-candidate-count-policy.json"
        path.write_text(json.dumps(policy), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "36 frozen hierarchy candidates"):
            generator.validate_heldout_policy_protocol(path, config)

    def test_heldout_policy_rejects_wrong_manual_alternative_count(self) -> None:
        config = self.heldout_configs[0]
        policy = copy.deepcopy(self.heldout_policies[config["dataset"]])
        policy["manual_alternatives"] = 36
        path = self.root / "wrong-manual-alternative-count-policy.json"
        path.write_text(json.dumps(policy), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly 35 manual alternatives"):
            generator.validate_heldout_policy_protocol(path, config)

    def test_heldout_policy_rejects_ambiguous_legacy_count(self) -> None:
        config = self.heldout_configs[0]
        policy = copy.deepcopy(self.heldout_policies[config["dataset"]])
        policy["manual_hierarchy_cases"] = 36
        path = self.root / "legacy-candidate-count-policy.json"
        path.write_text(json.dumps(policy), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ambiguous legacy"):
            generator.validate_heldout_policy_protocol(path, config)

    def test_heldout_policy_requires_one_gate_for_manual_grid(self) -> None:
        config = copy.deepcopy(self.heldout_configs[0])
        manual = next(
            row for row in config["methods"]
            if row["selection_role"] == generator.HELDOUT_MANUAL_ROLE)
        manual["routing_policy"] = "always_layered"
        dataset = config["dataset"]
        path = next(path for path in self.heldout_policy_paths
                    if dataset.lower() in path.name)
        with self.assertRaisesRegex(ValueError, "do not share one gate"):
            generator.validate_heldout_policy_protocol(path, config)

    def test_heldout_policy_binds_workload_query_count(self) -> None:
        config = copy.deepcopy(self.heldout_configs[0])
        config["workloads"][0]["num_queries"] -= 1
        dataset = config["dataset"]
        path = next(path for path in self.heldout_policy_paths
                    if dataset.lower() in path.name)
        with self.assertRaisesRegex(ValueError, "query count mismatch"):
            generator.validate_heldout_policy_protocol(path, config)

    def test_paper_shell_uses_one_generated_macro_contract(self) -> None:
        paper = SCRIPT_DIR.parent.parent / "docs/papers/multilevel_ung"
        if not paper.is_dir():
            paper = SCRIPT_DIR
        main = (paper / "main.tex").read_text(encoding="utf-8")
        placeholder = (paper / "generated_results.tex").read_text(
            encoding="utf-8")
        self.assertLess(
            main.index(r"\input{generated_results}"),
            main.index(r"\begin{document}"))
        self.assertEqual(main.count(r"\input{generated_results}"), 1)
        self.assertEqual(main.count(r"\authoritativeAbstractResult"), 1)
        self.assertEqual(main.count(r"\authoritativeConclusionResult"), 1)
        self.assertEqual(main.count(r"\authoritativeDatasetTable"), 1)
        self.assertEqual(main.count(r"\authoritativeQueryExecution"), 1)
        self.assertEqual(
            main.count(r"\authoritativeSharedSearchBudgetTable"), 1)
        self.assertEqual(main.count(r"\authoritativeResults"), 1)
        self.assertEqual(main.count(r"\authoritativeRecallQPSFigures"), 1)
        self.assertEqual(main.count(r"\authoritativeScreenAppendix"), 1)
        self.assertNotIn(r"\pending{", main)
        for name in (
                "authoritativeAbstractResult",
                "authoritativeConclusionResult",
                "authoritativeDatasetTable",
                "authoritativeQueryExecution",
                "authoritativeSharedSearchBudgetTable",
                "authoritativeResults",
                "authoritativeRecallQPSFigures",
                "authoritativeScreenAppendix"):
            self.assertIn(r"\newcommand{\%s}" % name, placeholder)


if __name__ == "__main__":
    unittest.main()
