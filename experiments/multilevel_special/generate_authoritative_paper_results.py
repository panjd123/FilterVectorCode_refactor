#!/usr/bin/env python3
"""Generate the authoritative nine-selectivity LaTeX result section.

This generator is deliberately separate from ``generate_paper_results.py``,
which belongs to the historical six-workload study.  It fails before touching
the output unless every query validator passes, every build manifest is
complete, and the aggregate CSVs agree with the validated configurations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from derive_static_hierarchy import derive_plan


HERE = Path(__file__).resolve().parent
EXPECTED_AMAZON_SELECTIVITIES = (
    0.00499249, 0.00903055, 0.05038117, 0.09906615, 0.30027242,
    0.60047491, 0.80023699, 0.95019781, 0.99000600,
)
EXPECTED_HELDOUT_DATASETS = {"Genome", "Reviews", "VariousImg"}
HELDOUT_AUTOMATIC_ROLE = "predeclared_degree_ratio_hierarchy_v1"
HELDOUT_MANUAL_ROLE = "predeclared_manual_oracle_grid"
HELDOUT_UNROUTED_ROLE = "degree_ratio_hierarchy_v1_unrouted_ablation"
HELDOUT_BASELINE_ROLE = "zero_layer_baseline"
HELDOUT_ROUTING_POLICY = "require_upper_authorization"
EXPECTED_FROZEN_HIERARCHY_CASES = 36
EXPECTED_MANUAL_ALTERNATIVES = EXPECTED_FROZEN_HIERARCHY_CASES - 1
RECALL_QPS_FIGURES = (
    ("principal_zero", "Principal zero-layer systems"),
    ("zero_topology_fixed_entry", "Zero-layer topology with fixed entry discovery"),
    ("depth_fixed_lng", "Hierarchy depth with fixed LNG topology and entry discovery"),
    ("two_layer_topology", "Per-layer topology within the two-layer hierarchy"),
    ("threshold_depth", "Threshold sensitivity within one- and two-layer hierarchies"),
    ("entry_strategy_on_drh", "Entry discovery on the fixed DRH hierarchy"),
    ("upper_authorization", "Exact upper-authorization routing"),
)
DEPTH_CATEGORIES = (
    "plain", "best_one_layer", "best_two_layer", "automatic_drh",
    "automatic_routed",
)
BASELINE_METHOD = "l0_lng_entry_optimized_lng"
PRINCIPAL_TRIE_METHOD = "l0_trie_entry_trie"
ROUTED_DRH_ROLE = "predeclared_degree_ratio_hierarchy_v1_upper_authorization_control"
ZERO_LAYER_ABLATION = (
    ("l0_lng_entry_original", "LNG + original"),
    ("l0_lng_entry_optimized_lng", "LNG + optimized LNG"),
    ("l0_lng_entry_trie", "LNG + Trie"),
    ("l0_trie_entry_original", "Trie + original"),
    ("l0_trie_entry_optimized_lng", "Trie + optimized LNG"),
    ("l0_trie_entry_trie", "Trie + Trie"),
)
TWO_LAYER_TOPOLOGY_ABLATION = (
    ("l2_t1024_16384_ll_entry_optimized_lng", "LNG/LNG"),
    ("l2_t1024_16384_lt_entry_optimized_lng", "LNG/Trie"),
    ("l2_t1024_16384_tl_entry_optimized_lng", "Trie/LNG"),
    ("l2_t1024_16384_tt_entry_optimized_lng", "Trie/Trie"),
)
THRESHOLD_DEPTH_ABLATION = (
    ("l1_t1024_lng_entry_optimized_lng", "1L: 1,024"),
    ("l1_t8192_lng_entry_optimized_lng", "1L: 8,192"),
    ("l2_t1024_16384_lt_entry_optimized_lng", "2L: 1,024/16,384"),
    ("l2_t8192_131072_lt_entry_optimized_lng", "2L: 8,192/131,072"),
)
DRH_ENTRY_ABLATION = (
    ("l2_t1024_16384_lt_entry_original", "Original"),
    ("l2_t1024_16384_lt_entry_optimized_lng", "Optimized LNG"),
    ("l2_t1024_16384_lt_entry_trie", "Trie"),
)
QUERY_FIELDS = {
    "workload", "mean_selectivity", "method", "layer_count", "thresholds",
    "base_topology", "layer_topologies", "entry_strategy", "routing_policy",
    "lsearch", "recall_min", "qps_warm_median", "target_recall",
}
PROFILE_FIELDS = QUERY_FIELDS | {
    "query_total_ms_warm_median", "els_ms_warm_median",
    "entry_ms_warm_median", "block_authorization_ms_warm_median",
    "graph_ms_warm_median", "residual_ms_warm_median",
    "stage_closure_ms_at_batch_median",
    "layered_path_activation_rate_warm_median",
    "nodes_visited_warm_median", "regular_edges_scanned_warm_median",
    "special_intra_edges_scanned_warm_median",
    "special_inter_edges_scanned_warm_median",
    "total_edges_scanned_warm_median",
    "entry_point_distance_calcs_warm_median",
    "graph_search_distance_calcs_warm_median",
    "total_distance_calcs_warm_median",
    "num_entries_warm_median", "entry_group_matched_points_warm_median",
}


@dataclass(frozen=True)
class ResultPaths:
    amazon_formal: Path
    amazon_depth: Path
    amazon_depth_global: Path
    amazon_profile: Path
    heldout_workload: Path
    heldout_global: Path
    build_summary: Path
    build_end_to_end: Path
    build_quality_formal: Path


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty aggregate")
    return rows


def number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value in (None, ""):
        raise ValueError(f"missing numeric field {field}: {row}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field}: {value}")
    return result


def integer(row: dict[str, str], field: str) -> int:
    value = number(row, field)
    if value != int(value):
        raise ValueError(f"non-integral {field}: {value}")
    return int(value)


def geometric_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.mean(math.log(value) for value in values))


def method_enabled(method: dict[str, Any], workload: str) -> bool:
    enabled = method.get("enabled_workloads")
    return enabled is None or workload in enabled


def expected_pairs(config: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(method["name"]), str(workload["name"]))
        for method in config["methods"]
        for workload in config["workloads"]
        if method_enabled(method, str(workload["name"]))
    }


def manifest_path(config: dict[str, Any]) -> Path:
    root = Path(config["output_root"])
    pass_name = str(config.get("measurement_pass", "performance"))
    return root / (f"manifest_{pass_name}.json"
                   if config.get("pass_subdirs", False) else "manifest.json")


def validated_query_config(
    path: Path, validator: Path, phase: str, measured_repeats: int,
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not validator.is_file():
        raise FileNotFoundError(validator)
    result = subprocess.run(
        [sys.executable, str(validator), str(path)],
        cwd=validator.parent, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"query validator failed for {path}:\n{result.stdout.rstrip()}")
    config = json.loads(path.read_text(encoding="utf-8"))
    protocol = config.get("protocol", {})
    expected = {
        "phase": phase, "cold_repeats": 1,
        "measured_repeats": measured_repeats, "recall_rule": "all_repeats",
    }
    mismatches = {
        key: (protocol.get(key), value) for key, value in expected.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{path}: protocol mismatch {mismatches}")
    manifest = json.loads(manifest_path(config).read_text(encoding="utf-8"))
    current = {
        (str(row.get("method")), str(row.get("workload"))): row
        for row in manifest.get("runs", [])
        if (str(row.get("method")), str(row.get("workload")))
        in expected_pairs(config)
    }
    hashes = {row.get("search_binary_sha256") for row in current.values()}
    if len(hashes) != 1 or None in hashes:
        raise ValueError(f"{path}: query manifest has inconsistent binary hashes")
    binary_hash = str(next(iter(hashes)))
    expected_hash = config.get("expected_search_binary_sha256")
    if expected_hash is not None and binary_hash != expected_hash:
        raise ValueError(f"{path}: query manifest differs from pinned binary")
    return config, binary_hash


def validate_build_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    manifest = Path(config["output_root"]) / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    latest = {str(row.get("name")): row for row in payload.get("runs", [])}
    problems = []
    for case in config.get("cases", []):
        row = latest.get(str(case["name"]))
        if row is None:
            problems.append(f"{case['name']}: missing")
        elif row.get("status") != "complete" or int(row.get("returncode", 0)) != 0:
            problems.append(
                f"{case['name']}: status={row.get('status')} "
                f"returncode={row.get('returncode')}")
        elif float(row.get("elapsed_seconds", 0)) <= 0:
            problems.append(f"{case['name']}: missing positive elapsed time")
    if not config.get("cases"):
        problems.append("configuration has no cases")
    if problems:
        raise RuntimeError(f"incomplete build config {path}:\n" + "\n".join(problems))
    return config


def validate_build_config_set(configs: list[dict[str, Any]]) -> None:
    observed = set()
    for config in configs:
        builder = Path(config.get("build_app", "")).name
        if builder == "build_UNG_index":
            component = "base"
        elif builder == "build_special_block_index":
            component = "hierarchy"
        else:
            raise ValueError(f"unexpected authoritative builder: {builder}")
        observed.add((component, bool(config.get("resource_profile", False))))
    expected = {
        ("base", False), ("base", True),
        ("hierarchy", False), ("hierarchy", True),
    }
    if observed != expected or len(configs) != len(expected):
        raise ValueError(
            f"build config set mismatch: expected {sorted(expected)}, "
            f"got {sorted(observed)}")


def validate_performance_binary_hashes(
    formal_hash: str, heldout_hashes: set[str], build_quality_hash: str,
    screen_hash: str | None = None,
) -> None:
    """Require every result-bearing performance pass to use one binary."""
    hashes = heldout_hashes | {formal_hash, build_quality_hash}
    if screen_hash is not None:
        hashes.add(screen_hash)
    if len(hashes) != 1:
        raise ValueError(
            "Amazon formal, held-out formal, and build-quality formal runs "
            "must use one immutable query binary")


def validate_config_dataset(
    config: dict[str, Any], expected_dataset: str,
) -> None:
    if str(config.get("dataset")) != expected_dataset:
        raise ValueError(
            f"expected dataset {expected_dataset}, got {config.get('dataset')}")


def hierarchy_signature(method: dict[str, Any]) -> tuple[tuple[int, str], ...]:
    layers = method.get("hierarchy_layers")
    if not isinstance(layers, list):
        raise ValueError(f"invalid hierarchy_layers for {method.get('name')}")
    signature = []
    previous = 0
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError(f"invalid hierarchy layer for {method.get('name')}")
        threshold = layer.get("min_points")
        topology = layer.get("topology")
        if type(threshold) is not int or threshold <= previous:
            raise ValueError(
                f"non-increasing hierarchy thresholds for {method.get('name')}")
        if topology not in {"lng", "trie"}:
            raise ValueError(f"invalid hierarchy topology for {method.get('name')}")
        signature.append((threshold, str(topology)))
        previous = threshold
    return tuple(signature)


def validate_heldout_policy_protocol(
    path: Path, config: dict[str, Any],
) -> dict[str, Any]:
    """Re-derive DRH and bind a frozen held-out policy to its formal config."""
    if not path.is_file():
        raise FileNotFoundError(path)
    policy = json.loads(path.read_text(encoding="utf-8"))
    dataset = str(config.get("dataset"))
    if policy.get("schema_version") != 2:
        raise ValueError(f"{path}: expected policy schema_version 2")
    if policy.get("dataset") != dataset:
        raise ValueError(
            f"{path}: policy/config dataset mismatch: "
            f"{policy.get('dataset')} != {dataset}")
    if policy.get("policy") != "gated_degree_ratio_hierarchy_v1":
        raise ValueError(f"{path}: unexpected automatic policy")
    if policy.get("query_calibrated") is not False:
        raise ValueError(f"{path}: held-out policy must not be query calibrated")
    if policy.get("manual_grid_frozen_before_search") is not True:
        raise ValueError(f"{path}: manual grid was not frozen before search")
    if policy.get("manual_grid_uses_automatic_routing_policy") is not True:
        raise ValueError(f"{path}: manual grid does not use the automatic gate")
    if policy.get("automatic_routing_policy") != HELDOUT_ROUTING_POLICY:
        raise ValueError(f"{path}: unexpected automatic routing policy")

    inputs = policy.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"{path}: missing policy inputs")
    required_inputs = (
        "num_points", "dimension", "max_degree", "num_cross_edges",
        "scale_ratio",
    )
    for field in required_inputs:
        if type(inputs.get(field)) is not int or inputs[field] <= 0:
            raise ValueError(f"{path}: invalid positive integer input {field}")
    if inputs["num_points"] != config.get("expected_num_points"):
        raise ValueError(f"{path}: policy N does not match formal config")
    expected_ratio = max(
        2, round(inputs["max_degree"] / inputs["num_cross_edges"]))
    if inputs["scale_ratio"] != expected_ratio:
        raise ValueError(f"{path}: inconsistent degree ratio")
    derived_layers = derive_plan(
        inputs["num_points"], inputs["max_degree"],
        inputs["num_cross_edges"])
    if policy.get("automatic_hierarchy_layers") != derived_layers:
        raise ValueError(f"{path}: automatic hierarchy does not match DRH derivation")
    automatic_signature = tuple(
        (int(layer["min_points"]), str(layer["topology"]))
        for layer in derived_layers)

    methods = config.get("methods")
    if not isinstance(methods, list):
        raise ValueError(f"{path}: formal config has no method list")
    names = [str(method.get("name")) for method in methods]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate formal method names")
    roles = Counter(str(method.get("selection_role")) for method in methods)
    expected_roles = {
        HELDOUT_AUTOMATIC_ROLE: 1,
        HELDOUT_MANUAL_ROLE: EXPECTED_MANUAL_ALTERNATIVES,
        HELDOUT_UNROUTED_ROLE: 1,
        HELDOUT_BASELINE_ROLE: 1,
    }
    if roles != expected_roles:
        raise ValueError(
            f"{path}: held-out method roles mismatch: "
            f"expected {expected_roles}, got {dict(roles)}")
    frozen_candidates = policy.get("frozen_hierarchy_candidates")
    if (type(frozen_candidates) is not int
            or frozen_candidates != EXPECTED_FROZEN_HIERARCHY_CASES):
        raise ValueError(
            f"{path}: expected 36 frozen hierarchy candidates "
            "(DRH plus 35 manual alternatives)")
    manual_alternatives = policy.get("manual_alternatives")
    if (type(manual_alternatives) is not int
            or manual_alternatives != EXPECTED_MANUAL_ALTERNATIVES):
        raise ValueError(f"{path}: expected exactly 35 manual alternatives")
    if frozen_candidates != manual_alternatives + roles[HELDOUT_AUTOMATIC_ROLE]:
        raise ValueError(f"{path}: frozen candidate counts do not close")
    if "manual_hierarchy_cases" in policy:
        raise ValueError(f"{path}: ambiguous legacy manual_hierarchy_cases field")

    by_role = {
        role: [method for method in methods
               if method.get("selection_role") == role]
        for role in expected_roles
    }
    baseline = by_role[HELDOUT_BASELINE_ROLE][0]
    if (
        baseline.get("name") != BASELINE_METHOD
        or baseline.get("base_topology") != "lng"
        or baseline.get("entry_strategy") != "optimized_lng"
        or hierarchy_signature(baseline)
        or baseline.get("special_block_search") is not False
    ):
        raise ValueError(f"{path}: invalid held-out zero-layer baseline")

    automatic = by_role[HELDOUT_AUTOMATIC_ROLE][0]
    unrouted = by_role[HELDOUT_UNROUTED_ROLE][0]
    if automatic.get("name") != policy.get("automatic_method"):
        raise ValueError(f"{path}: automatic method name mismatch")
    if unrouted.get("name") != policy.get("unrouted_ablation_method"):
        raise ValueError(f"{path}: unrouted ablation name mismatch")
    for method in (automatic, unrouted):
        if hierarchy_signature(method) != automatic_signature:
            raise ValueError(f"{path}: automatic hierarchy/config mismatch")
        if (method.get("base_topology") != "lng"
                or method.get("entry_strategy") != "optimized_lng"
                or method.get("special_block_search") is not True):
            raise ValueError(f"{path}: invalid automatic method capability")
    if automatic.get("routing_policy") != HELDOUT_ROUTING_POLICY:
        raise ValueError(f"{path}: automatic method does not use the policy gate")
    if unrouted.get("routing_policy", "always_layered") != "always_layered":
        raise ValueError(f"{path}: ungated ablation is unexpectedly routed")

    candidates = by_role[HELDOUT_MANUAL_ROLE] + [automatic]
    signatures = []
    for method in candidates:
        signature = hierarchy_signature(method)
        if not signature:
            raise ValueError(f"{path}: empty hierarchy in frozen candidate grid")
        if signature[-1][0] >= inputs["num_points"]:
            raise ValueError(f"{path}: hierarchy threshold exceeds dataset size")
        if (method.get("base_topology") != "lng"
                or method.get("entry_strategy") != "optimized_lng"
                or method.get("special_block_search") is not True
                or method.get("routing_policy") != HELDOUT_ROUTING_POLICY):
            raise ValueError(f"{path}: frozen candidates do not share one gate")
        signatures.append(signature)
    if len(set(signatures)) != EXPECTED_FROZEN_HIERARCHY_CASES:
        raise ValueError(f"{path}: frozen hierarchy candidates are not unique")
    observed_depths = sorted({len(signature) for signature in signatures})
    if policy.get("manual_depths") != observed_depths:
        raise ValueError(f"{path}: frozen hierarchy depths mismatch")

    policy_workloads = policy.get("workloads")
    config_workloads = config.get("workloads")
    if not isinstance(policy_workloads, list) or not isinstance(config_workloads, list):
        raise ValueError(f"{path}: invalid workload metadata")
    policy_by_name = {str(row.get("name")): row for row in policy_workloads}
    config_by_name = {str(row.get("name")): row for row in config_workloads}
    if (len(policy_by_name) != len(policy_workloads)
            or len(config_by_name) != len(config_workloads)
            or set(policy_by_name) != set(config_by_name)):
        raise ValueError(f"{path}: policy/config workload names mismatch")
    for workload, policy_row in policy_by_name.items():
        config_row = config_by_name[workload]
        if policy_row.get("num_queries") != config_row.get("num_queries"):
            raise ValueError(f"{path}: query count mismatch for {workload}")
        if abs(float(policy_row.get("mean_selectivity")) -
               float(config_row.get("mean_selectivity"))) > 1e-15:
            raise ValueError(f"{path}: selectivity mismatch for {workload}")
    return policy


def validate_heldout_policy_set(
    paths: list[Path], configs: list[dict[str, Any]],
) -> list[tuple[str, Path, dict[str, Any]]]:
    configs_by_dataset = {str(config.get("dataset")): config for config in configs}
    if set(configs_by_dataset) != EXPECTED_HELDOUT_DATASETS:
        raise ValueError("held-out formal config dataset set is incomplete")
    validated = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        dataset = str(payload.get("dataset"))
        if dataset in seen:
            raise ValueError(f"duplicate held-out policy for {dataset}")
        if dataset not in configs_by_dataset:
            raise ValueError(f"unexpected held-out policy dataset {dataset}")
        policy = validate_heldout_policy_protocol(
            resolved, configs_by_dataset[dataset])
        validated.append((dataset, resolved, policy))
        seen.add(dataset)
    if seen != EXPECTED_HELDOUT_DATASETS:
        raise ValueError(f"held-out policy dataset set mismatch: {sorted(seen)}")
    return sorted(validated, key=lambda row: row[0])


def validate_amazon_workloads(config: dict[str, Any]) -> list[str]:
    validate_config_dataset(config, "Amazon")
    workloads = sorted(
        config["workloads"], key=lambda row: float(row["mean_selectivity"]))
    if len(workloads) != len(EXPECTED_AMAZON_SELECTIVITIES):
        raise ValueError(f"Amazon must contain nine workloads, got {len(workloads)}")
    observed = [float(row["mean_selectivity"]) for row in workloads]
    for actual, expected in zip(observed, EXPECTED_AMAZON_SELECTIVITIES):
        if abs(actual - expected) > 5e-7:
            raise ValueError(
                f"unexpected Amazon selectivity sequence: {observed}")
    if any(abs(float(config["recall_thresholds"][row["name"]]) - 0.90) > 1e-12
           for row in workloads):
        raise ValueError("Amazon Recall thresholds must all be 0.90")
    return [str(row["name"]) for row in workloads]


def unique_rows(
    rows: list[dict[str, str]], keys: tuple[str, ...], source: str,
) -> dict[tuple[str, ...], dict[str, str]]:
    result: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[field] for field in keys)
        if key in result:
            raise ValueError(f"{source}: duplicate key {key}")
        result[key] = row
    return result


def validate_formal_rows(
    rows: list[dict[str, str]], config: dict[str, Any], source: str,
) -> dict[tuple[str, str], dict[str, str]]:
    indexed = unique_rows(rows, ("method", "workload"), source)
    expected = expected_pairs(config)
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(
            f"{source}: method/workload mismatch; missing={missing[:5]} "
            f"extra={extra[:5]}")
    targets = config["recall_thresholds"]
    methods = {str(row["name"]): row for row in config["methods"]}
    workloads = {str(row["name"]): row for row in config["workloads"]}
    for (method, workload), row in indexed.items():
        target = float(targets[workload])
        if abs(number(row, "target_recall") - target) > 1e-12:
            raise ValueError(f"{source}: target mismatch for {method}/{workload}")
        if number(row, "recall_min") < target:
            raise ValueError(f"{source}: non-crossing formal row {method}/{workload}")
        if integer(row, "lsearch") <= 0 or number(row, "qps_warm_median") <= 0:
            raise ValueError(f"{source}: invalid operating point {method}/{workload}")
        method_config = methods[method]
        layers = method_config.get("hierarchy_layers", [])
        expected_metadata = {
            "layer_count": str(len(layers)),
            "thresholds": ",".join(
                str(layer["min_points"]) for layer in layers) or "none",
            "base_topology": str(method_config["base_topology"]),
            "layer_topologies": ",".join(
                str(layer["topology"]) for layer in layers) or "none",
            "entry_strategy": str(method_config["entry_strategy"]),
            "routing_policy": str(method_config.get(
                "routing_policy", "always_layered")),
        }
        for field, expected_value in expected_metadata.items():
            if row[field] != expected_value:
                raise ValueError(
                    f"{source}: {field} mismatch for {method}/{workload}: "
                    f"{row[field]} != {expected_value}")
        if abs(number(row, "mean_selectivity") -
               float(workloads[workload]["mean_selectivity"])) > 1e-12:
            raise ValueError(f"{source}: selectivity mismatch for {method}/{workload}")
    return indexed


def validate_depth_rows(
    rows: list[dict[str, str]], workloads: list[str],
) -> dict[tuple[str, str], dict[str, str]]:
    indexed = unique_rows(rows, ("category", "workload"), "Amazon depth")
    expected = {(category, workload) for category in DEPTH_CATEGORIES
                for workload in workloads}
    if set(indexed) != expected:
        raise ValueError(
            "Amazon depth matrix mismatch; missing="
            f"{sorted(expected - set(indexed))[:5]} extra="
            f"{sorted(set(indexed) - expected)[:5]}")
    for (category, workload), row in indexed.items():
        status = row["status"]
        if status not in {"complete", "no_crossing"}:
            raise ValueError(f"invalid depth status {status}: {category}/{workload}")
        if category == "plain" and status != "complete":
            raise ValueError(f"plain baseline has no crossing: {workload}")
        if status == "complete":
            if number(row, "recall_min") < number(row, "target_recall"):
                raise ValueError(f"non-conservative depth crossing: {category}/{workload}")
            if number(row, "qps") <= 0 or number(row, "speedup_vs_plain") <= 0:
                raise ValueError(f"invalid depth performance: {category}/{workload}")
            if number(row, "speedup_ci95_low") <= 0 or number(
                    row, "speedup_ci95_high") <= 0:
                raise ValueError(f"invalid depth confidence interval: {category}/{workload}")
    return indexed


def validate_depth_global(rows: list[dict[str, str]]) -> None:
    indexed = unique_rows(rows, ("category",), "Amazon depth global")
    if set(key[0] for key in indexed) != set(DEPTH_CATEGORIES):
        raise ValueError("Amazon depth global categories are incomplete")
    if indexed[("plain",)]["status"] != "complete":
        raise ValueError("global plain baseline is incomplete")
    for (category,), row in indexed.items():
        if integer(row, "workload_count") != 9:
            raise ValueError(f"global depth workload count mismatch: {category}")
        if row["status"] == "complete":
            if number(row, "geomean_qps") <= 0 or number(
                    row, "speedup_vs_plain") <= 0:
                raise ValueError(f"invalid global depth result: {category}")


def validate_profile_rows(
    rows: list[dict[str, str]], config: dict[str, Any],
    formal: dict[tuple[str, str], dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    indexed = validate_formal_rows(rows, config, "Amazon profile")
    if set(indexed) != set(formal):
        raise ValueError("profile and formal method/workload sets differ")
    for key, row in indexed.items():
        if integer(row, "lsearch") != integer(formal[key], "lsearch"):
            raise ValueError(f"profile L differs from formal L: {key}")
        for field in PROFILE_FIELDS - QUERY_FIELDS:
            number(row, field)
        activation = number(row, "layered_path_activation_rate_warm_median")
        if not 0.0 <= activation <= 1.0:
            raise ValueError(f"invalid layered activation rate: {key}")
        if abs(number(row, "stage_closure_ms_at_batch_median")) > 1e-6:
            raise ValueError(f"stage closure exceeds tolerance: {key}")
    return indexed


def validate_heldout_rows(
    workload_rows: list[dict[str, str]], global_rows: list[dict[str, str]],
    configs: list[dict[str, Any]],
) -> None:
    datasets = {str(config["dataset"]) for config in configs}
    if datasets != EXPECTED_HELDOUT_DATASETS:
        raise ValueError(f"held-out dataset set mismatch: {sorted(datasets)}")
    expected = {
        (str(config["dataset"]), str(workload["name"]))
        for config in configs for workload in config["workloads"]
    }
    indexed = unique_rows(
        workload_rows, ("dataset", "workload"), "held-out workload")
    if set(indexed) != expected:
        raise ValueError("held-out workload matrix does not match validated configs")
    for key, row in indexed.items():
        if row["status"] not in {"complete", "automatic_no_crossing"}:
            raise ValueError(f"invalid held-out status: {key}")
        if number(row, "baseline_recall_min") < number(row, "target_recall"):
            raise ValueError(f"held-out baseline misses Recall: {key}")
        if number(row, "oracle_recall_min") < number(row, "target_recall"):
            raise ValueError(f"held-out oracle misses Recall: {key}")
        if row["status"] == "complete":
            if number(row, "automatic_recall_min") < number(row, "target_recall"):
                raise ValueError(f"held-out automatic method misses Recall: {key}")
            number(row, "automatic_speedup_vs_baseline")
            number(row, "automatic_qps_fraction_of_oracle")
            number(row, "automatic_speedup_ci95_low")
            number(row, "automatic_speedup_ci95_high")
            number(row, "automatic_oracle_fraction_ci95_low")
            number(row, "automatic_oracle_fraction_ci95_high")
    globals_by_dataset = unique_rows(global_rows, ("dataset",), "held-out global")
    if {key[0] for key in globals_by_dataset} != datasets:
        raise ValueError("held-out global rows do not match validated configs")
    expected_counts = {
        str(config["dataset"]): len(config["workloads"]) for config in configs
    }
    for (dataset,), row in globals_by_dataset.items():
        if integer(row, "workload_count") != expected_counts[dataset]:
            raise ValueError(f"held-out global workload count mismatch: {dataset}")


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean: {value}")


def validate_build_rows(
    summary: list[dict[str, str]], end_to_end: list[dict[str, str]],
) -> None:
    seen = unique_rows(summary, ("component", "structure", "profile"), "build summary")
    if not {"base", "hierarchy"}.issubset({key[0] for key in seen}):
        raise ValueError("build summary must contain base and hierarchy components")
    for key, row in seen.items():
        if integer(row, "measured_repeats") < 5:
            raise ValueError(f"build result has fewer than five repeats: {key}")
        for field in ("wall_median_seconds", "wall_p95_seconds", "index_median_mib"):
            if number(row, field) <= 0:
                raise ValueError(f"non-positive build metric {field}: {key}")
        if number(row, "peak_rss_mib") <= 0:
            raise ValueError(f"missing host-memory measurement: {key}")
        if parse_bool(row["gpu_required"]):
            if number(row, "peak_gpu_memory_mib") <= 0:
                raise ValueError(f"missing GPU-memory measurement: {key}")
            locked = row["gpu_exclusive_lock"] not in (None, "") and parse_bool(
                row["gpu_exclusive_lock"])
            idle_samples = integer(row, "gpu_idle_samples_min")
            if not locked and idle_samples < 3:
                raise ValueError(f"GPU build lacks lock or idle preflight: {key}")
    if not end_to_end:
        raise ValueError("end-to-end build summary is empty")
    unique_rows(end_to_end, ("structure", "hierarchy_profile"), "end-to-end build")
    for row in end_to_end:
        if integer(row, "stage_repeats") < 5:
            raise ValueError(
                "composed build result has fewer than five stage repeats")
        if row["composition_method"] != \
                "sum_of_stage_medians_independent_bootstrap":
            raise ValueError("invalid full-build composition method")
        for field in (
            "original_cpu_base_median_seconds",
            "composed_base_plus_hierarchy_seconds",
            "speedup_vs_original_cpu", "speedup_ci95_low", "speedup_ci95_high",
            "overhead_vs_accelerated_base_median",
        ):
            if number(row, field) <= 0:
                raise ValueError(f"non-positive end-to-end build metric {field}")
        lower = number(row, "speedup_ci95_low")
        upper = number(row, "speedup_ci95_high")
        if lower > upper:
            raise ValueError("full-build confidence interval is reversed")
        if parse_bool(row["no_slower_supported"]) != (lower >= 1.0):
            raise ValueError("full-build no-slower decision disagrees with CI")


def tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt(value: Any, digits: int = 2) -> str:
    if value in (None, ""):
        return "NC"
    return f"{float(value):.{digits}f}"


def compact_method(row: dict[str, str]) -> str:
    category = row.get("category", "")
    labels = {
        "plain": "0-layer", "best_one_layer": "best 1-layer",
        "best_two_layer": "best 2-layer", "automatic_drh": "ungated DRH",
        "automatic_routed": "gated DRH",
    }
    return labels.get(category, row.get("method", "method"))


def render_depth_table(
    depth: dict[tuple[str, str], dict[str, str]], workloads: list[str],
) -> list[str]:
    lines = [
        r"\subsection{Query Performance}",
        "All values below use the smallest measured $L_{search}$ for which every "
        "warm repeat reaches Recall@10 $\\ge 0.90$; NC denotes no measured "
        "crossing in the declared grid.",
        r"\begin{table*}[t]", r"\centering", r"\scriptsize",
        r"\caption{Amazon QPS at the conservative Recall crossing. Parentheses give speedup and its bootstrap 95\% confidence interval relative to zero-layer LNG.}",
        r"\label{tab:amazon-depth}",
        r"\begin{tabular}{r@{\quad}rrrr}", r"\toprule",
        r"Selectivity & 0-layer LNG & Best 1-layer & Best 2-layer & Gated DRH \\",
        r"\midrule",
    ]
    for workload in workloads:
        rowset = {category: depth[(category, workload)]
                  for category in DEPTH_CATEGORIES}
        selectivity = number(rowset["plain"], "mean_selectivity")
        values = []
        for category in ("plain", "best_one_layer", "best_two_layer", "automatic_routed"):
            row = rowset[category]
            if row["status"] != "complete":
                values.append("NC")
            elif category == "plain":
                values.append(fmt(row["qps"], 1))
            else:
                values.append(
                    f"{fmt(row['qps'], 1)} ({fmt(row['speedup_vs_plain'], 2)}$\\times$; "
                    f"[{fmt(row['speedup_ci95_low'], 2)}, "
                    f"{fmt(row['speedup_ci95_high'], 2)}])")
        lines.append(f"{100.0 * selectivity:.3f}\\% & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return lines


def render_depth_global(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{One unchanged Amazon configuration across all nine selectivities.}",
        r"\label{tab:amazon-depth-global}",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Category & Geomean QPS & vs. 0-layer \\", r"\midrule",
    ]
    order = {name: index for index, name in enumerate(DEPTH_CATEGORIES)}
    for row in sorted(rows, key=lambda item: order[item["category"]]):
        if row["status"] == "complete":
            qps = fmt(row["geomean_qps"], 1)
            speedup = fmt(row["speedup_vs_plain"], 2) + r"$\times$"
        else:
            qps = speedup = "NC"
        lines.append(
            f"{tex_escape(compact_method(row))} & {qps} & {speedup} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return lines


def render_zero_layer_table(
    formal: dict[tuple[str, str], dict[str, str]],
    config: dict[str, Any], workloads: list[str],
) -> list[str]:
    methods = [method for method in config["methods"]
               if len(method.get("hierarchy_layers", [])) == 0]
    baseline_qps = [number(formal[(BASELINE_METHOD, workload)], "qps_warm_median")
                    for workload in workloads
                    if (BASELINE_METHOD, workload) in formal]
    if len(baseline_qps) != len(workloads):
        raise ValueError("principal zero-layer baseline is incomplete")
    baseline_geomean = geometric_mean(baseline_qps)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Zero-layer topology and entry-strategy ablation on Amazon. Global QPS is reported only when all nine workloads cross.}",
        r"\label{tab:amazon-zero-layer}",
        r"\begin{tabular}{llrr}", r"\toprule",
        r"Topology & Entry & Crossings & Geomean QPS \\", r"\midrule",
    ]
    for method in sorted(methods, key=lambda row: (
            str(row["base_topology"]), str(row["entry_strategy"]))):
        name = str(method["name"])
        qps = [number(formal[(name, workload)], "qps_warm_median")
               for workload in workloads if (name, workload) in formal]
        global_text = "NC"
        if len(qps) == len(workloads):
            value = geometric_mean(qps)
            global_text = f"{value:.1f} ({value / baseline_geomean:.2f}$\\times$)"
        lines.append(
            f"{tex_escape(method['base_topology'].upper())} & "
            f"{tex_escape(method['entry_strategy'])} & {len(qps)}/9 & "
            f"{global_text} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return lines


def render_principal_zero_layer_by_workload(
    formal: dict[tuple[str, str], dict[str, str]], workloads: list[str],
) -> list[str]:
    """Render the direct zero-layer Trie-versus-LNG comparison."""
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Principal zero-layer comparison at the conservative Recall crossing. Each cell reports $L_{search}$/QPS; NC means no measured crossing.}",
        r"\label{tab:amazon-zero-layer-by-selectivity}",
        r"\begin{tabular}{rrrr}", r"\toprule",
        r"Selectivity & LNG & Trie & Trie/LNG \\", r"\midrule",
    ]
    for workload in workloads:
        baseline = formal.get((BASELINE_METHOD, workload))
        if baseline is None:
            raise ValueError(f"principal LNG baseline is missing: {workload}")
        trie = formal.get((PRINCIPAL_TRIE_METHOD, workload))
        selectivity = 100.0 * number(baseline, "mean_selectivity")
        lng_qps = number(baseline, "qps_warm_median")
        lng_cell = f"{integer(baseline, 'lsearch'):,}/{lng_qps:.1f}"
        if trie is None:
            trie_cell = ratio = "NC"
        else:
            trie_qps = number(trie, "qps_warm_median")
            trie_cell = f"{integer(trie, 'lsearch'):,}/{trie_qps:.1f}"
            ratio = f"{trie_qps / lng_qps:.2f}$\\times$"
        lines.append(
            f"{selectivity:.3f}\\% & {lng_cell} & {trie_cell} & {ratio} "
            + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return lines


def render_equal_recall_ablation(
    formal: dict[tuple[str, str], dict[str, str]],
    config: dict[str, Any], workloads: list[str],
    methods: tuple[tuple[str, str], ...], caption: str, label: str,
) -> list[str]:
    """Render every declared method at its conservative Recall crossing."""
    declared = {str(method["name"]) for method in config["methods"]}
    missing = [name for name, _ in methods if name not in declared]
    if missing:
        raise ValueError(f"{label}: methods are absent from config: {missing}")
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{" + "r" * (len(methods) + 1) + "}",
        r"\toprule",
        "Selectivity & " + " & ".join(
            tex_escape(display) for _, display in methods) + r" \\",
        r"\midrule",
    ]
    for workload in workloads:
        baseline = formal.get((BASELINE_METHOD, workload))
        if baseline is None:
            raise ValueError(f"principal LNG baseline is missing: {workload}")
        cells = []
        for name, _ in methods:
            row = formal.get((name, workload))
            cells.append(
                "NC" if row is None else
                f"{integer(row, 'lsearch'):,}/{number(row, 'qps_warm_median'):.1f}")
        lines.append(
            f"{100.0 * number(baseline, 'mean_selectivity'):.3f}\\% & "
            + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return lines


def render_two_layer_topology_table(
    formal: dict[tuple[str, str], dict[str, str]],
    config: dict[str, Any], workloads: list[str],
) -> list[str]:
    return render_equal_recall_ablation(
        formal, config, workloads, TWO_LAYER_TOPOLOGY_ABLATION,
        "Two-layer topology ablation at $T_1=1{,}024$ and "
        "$T_2=16{,}384$ with optimized LNG entry discovery. Each cell "
        "reports $L_{search}$/QPS.",
        "tab:amazon-two-layer-topology")


def render_drh_entry_table(
    formal: dict[tuple[str, str], dict[str, str]],
    config: dict[str, Any], workloads: list[str],
) -> list[str]:
    return render_equal_recall_ablation(
        formal, config, workloads, DRH_ENTRY_ABLATION,
        "Entry-strategy ablation on the fixed ungated DRH hierarchy "
        "($1{,}024$: LNG, $16{,}384$: Trie). Each cell reports "
        "$L_{search}$/QPS.",
        "tab:amazon-drh-entry")


def render_threshold_depth_table(
    formal: dict[tuple[str, str], dict[str, str]],
    config: dict[str, Any], workloads: list[str],
) -> list[str]:
    return render_equal_recall_ablation(
        formal, config, workloads, THRESHOLD_DEPTH_ABLATION,
        "Threshold sensitivity for one-layer LNG and two-layer LNG/Trie "
        "structures with optimized LNG entry discovery. Each cell reports "
        "$L_{search}$/QPS.",
        "tab:amazon-threshold-depth")


def render_query_interpretation(
    formal: dict[tuple[str, str], dict[str, str]],
    depth: dict[tuple[str, str], dict[str, str]],
    profile: dict[tuple[str, str], dict[str, str]],
    workloads: list[str],
) -> list[str]:
    """Turn validated operating points into cautious, data-backed prose."""
    trie_ratios = []
    trie_faster = []
    trie_missing = []
    for workload in workloads:
        baseline = formal[(BASELINE_METHOD, workload)]
        trie = formal.get((PRINCIPAL_TRIE_METHOD, workload))
        if trie is None:
            trie_missing.append(workload)
            continue
        ratio = number(trie, "qps_warm_median") / number(
            baseline, "qps_warm_median")
        trie_ratios.append(ratio)
        if ratio > 1.0:
            trie_faster.append(workload)

    def percentage_list(names: list[str]) -> str:
        return ", ".join(
            f"{100.0 * number(formal[(BASELINE_METHOD, name)], 'mean_selectivity'):.3f}\\%"
            for name in names)

    zero_text = (
        f"Among the {len(trie_ratios)} workloads where both principal zero-layer "
        f"methods reach the Recall target, Trie is faster on {len(trie_faster)} "
        f"of {len(trie_ratios)}, "
        f"and its QPS ratio to LNG ranges from {min(trie_ratios):.2f}$\\times$ "
        f"to {max(trie_ratios):.2f}$\\times$."
        if trie_ratios else
        "The principal zero-layer Trie method has no measured Recall crossing."
    )
    if trie_missing:
        zero_text += (
            " It has no measured crossing at " + percentage_list(trie_missing)
            + ".")

    routed = [depth[("automatic_routed", workload)] for workload in workloads]
    routed_complete = [row for row in routed if row["status"] == "complete"]
    low_mid = [row for row in routed_complete
               if number(row, "mean_selectivity") < 0.60]
    high = [row for row in routed_complete
            if number(row, "mean_selectivity") >= 0.60]

    def speed_range(rows: list[dict[str, str]]) -> str:
        values = [number(row, "speedup_vs_plain") for row in rows]
        return f"{min(values):.2f}--{max(values):.2f}$\\times$"

    regime_parts = []
    if low_mid:
        regime_parts.append(
            f"below 60\\% selectivity it spans {speed_range(low_mid)}")
    if high:
        regime_parts.append(
            f"at 60\\% and above it spans {speed_range(high)}")
    regime_text = (
        f"Gated DRH reaches the target on {len(routed_complete)}/{len(workloads)} "
        "Amazon workloads; " + ", while ".join(regime_parts) + "."
        if regime_parts else
        "Gated DRH has no measured Recall crossing in the declared grid."
    )

    activated = []
    for workload in workloads:
        baseline = profile.get((BASELINE_METHOD, workload))
        routed_method = depth[("automatic_routed", workload)]["method"]
        candidate = profile.get((routed_method, workload))
        if (baseline is None or candidate is None
                or number(candidate, "layered_path_activation_rate_warm_median") <= 0):
            continue
        if (number(baseline, "graph_search_distance_calcs_warm_median") <= 0
                or number(baseline, "graph_ms_warm_median") <= 0):
            raise ValueError(
                f"non-positive baseline graph work for profiled {workload}")
        activated.append((baseline, candidate))
    mechanism_text = (
        "No profiled workload activates an upper layer, so the profile does not "
        "support a layered-path mechanism claim."
    )
    if activated:
        distance_ratios = [
            number(candidate, "graph_search_distance_calcs_warm_median") /
            number(baseline, "graph_search_distance_calcs_warm_median")
            for baseline, candidate in activated
        ]
        graph_time_ratios = [
            number(candidate, "graph_ms_warm_median") /
            number(baseline, "graph_ms_warm_median")
            for baseline, candidate in activated
        ]
        authorization = [
            number(candidate, "block_authorization_ms_warm_median")
            for _, candidate in activated
        ]
        mechanism_text = (
            f"Across {len(activated)} workloads with observed layered-path "
            f"activation, gated DRH uses {min(distance_ratios):.2f}--"
            f"{max(distance_ratios):.2f}$\\times$ the baseline graph-search "
            f"distance calculations and {min(graph_time_ratios):.2f}--"
            f"{max(graph_time_ratios):.2f}$\\times$ its graph-stage time. "
            f"Measured authorization costs {min(authorization):.3f}--"
            f"{max(authorization):.3f} ms/query. These are diagnostic profile "
            "ratios at the formal operating points, not primary QPS measurements."
        )

    return [
        r"\subsection{Observed Query Regimes}",
        r"\paragraph{Zero-layer topology.} " + zero_text,
        r"\paragraph{Hierarchy.} " + regime_text,
        r"\paragraph{Mechanism.} " + mechanism_text,
        "",
    ]


def render_heldout_dataset_table(policies: list[dict[str, Any]]) -> list[str]:
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize",
        r"\caption{Held-out datasets and query-independent DRH plans. Selectivity is measured from the fixed workload before timing; it is reported for characterization and is not an input to DRH.}",
        r"\label{tab:heldout-datasets}",
        r"\begin{tabular}{lrrlrrl}", r"\toprule",
        r"Dataset & $N$ & $d$ & Workload & Queries & Mean selectivity & DRH plan \\",
        r"\midrule",
    ]
    for policy in sorted(policies, key=lambda row: str(row["dataset"])):
        inputs = policy["inputs"]
        plan = ", ".join(
            f"{layer['min_points']}:{str(layer['topology']).upper()}"
            for layer in policy["automatic_hierarchy_layers"])
        for workload in sorted(
                policy["workloads"], key=lambda row: float(row["mean_selectivity"])):
            lines.append(
                f"{tex_escape(policy['dataset'])} & "
                f"{int(inputs['num_points']):,} & {int(inputs['dimension'])} & "
                f"{tex_escape(workload['name'])} & "
                f"{int(workload['num_queries']):,} & "
                f"{100.0 * float(workload['mean_selectivity']):.3f}\\% & "
                f"{tex_escape(plan)} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return lines


def render_heldout(
    workload_rows: list[dict[str, str]], global_rows: list[dict[str, str]],
) -> list[str]:
    lines = [
        r"\subsection{Automatic Versus Manual Hierarchies}",
        "The oracle candidate set (DRH plus 35 manually enumerated alternatives) "
        "is frozen before held-out queries are read, and every candidate uses "
        "the same exact upper-authorization gate.",
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Calibration-free gated DRH versus the frozen per-workload configuration oracle on held-out datasets.}",
        r"\label{tab:heldout-oracle}",
        r"\begin{tabular}{llrrll}", r"\toprule",
        r"Dataset & Selectivity & DRH/plain [95\% CI] & DRH/oracle [95\% CI] & DRH hierarchy & Oracle hierarchy \\",
        r"\midrule",
    ]
    for row in sorted(workload_rows, key=lambda item: (
            item["dataset"], number(item, "mean_selectivity"))):
        speedup = (
            fmt(row["automatic_speedup_vs_baseline"], 2) + r"$\times$ ["
            + fmt(row["automatic_speedup_ci95_low"], 2) + ", "
            + fmt(row["automatic_speedup_ci95_high"], 2) + "]"
            if row["status"] == "complete" else "NC")
        oracle = (
            fmt(row["automatic_qps_fraction_of_oracle"], 2) + " ["
            + fmt(row["automatic_oracle_fraction_ci95_low"], 2) + ", "
            + fmt(row["automatic_oracle_fraction_ci95_high"], 2) + "]"
            if row["status"] == "complete" else "NC")
        lines.append(
            f"{tex_escape(row['dataset'])} & {100.0 * number(row, 'mean_selectivity'):.3f}\\% & "
            f"{speedup} & {oracle} & {tex_escape(row['automatic_hierarchy'])} & "
            f"{tex_escape(row['oracle_hierarchy'])} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    lines.extend([
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{One unchanged DRH plan versus one unchanged manual plan per held-out dataset.}",
        r"\label{tab:heldout-global}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Dataset & Workloads & DRH/plain & DRH/oracle \\", r"\midrule",
    ])
    for row in sorted(global_rows, key=lambda item: item["dataset"]):
        auto = row["status"] == "complete"
        speedup = (fmt(row["automatic_speedup_vs_baseline"], 2) + r"$\times$"
                   if auto else "NC")
        oracle = (fmt(row["automatic_qps_fraction_of_global_oracle"], 2)
                  if auto else "NC")
        lines.append(
            f"{tex_escape(row['dataset'])} & {integer(row, 'workload_count')} & "
            f"{speedup} & {oracle} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return lines


def render_profile(
    profile: dict[tuple[str, str], dict[str, str]],
    config: dict[str, Any], workloads: list[str],
) -> list[str]:
    role_by_name = {str(method["name"]): str(method.get("selection_role", ""))
                    for method in config["methods"]}
    drh = [name for name, role in role_by_name.items() if role == ROUTED_DRH_ROLE]
    if len(drh) != 1:
        raise ValueError(f"expected one gated DRH method, got {drh}")
    selected_names = (BASELINE_METHOD, drh[0])
    rows = [profile[(name, workload)] for workload in workloads
            for name in selected_names if (name, workload) in profile]
    lines = [
        r"\subsection{Mechanism Breakdown}",
        "Profile-pass times are milliseconds per query and are not used for the "
        "primary QPS result. Activation is the measured fraction of queries that "
        "actually execute a layered path.",
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Stage timing at the formal operating point.}",
        r"\label{tab:query-stages}",
        r"\begin{tabular}{rlrrrrrrr}", r"\toprule",
        r"Sel. & Method & Active & Total & Entry-group & Entry setup & Authorization & Graph & Residual \\",
        r"\midrule",
    ]
    for row in rows:
        label = "0-layer" if row["method"] == BASELINE_METHOD else "gated DRH"
        lines.append(
            f"{100.0 * number(row, 'mean_selectivity'):.3f}\\% & {label} & "
            f"{100.0 * number(row, 'layered_path_activation_rate_warm_median'):.1f}\\% & "
            f"{fmt(row['query_total_ms_warm_median'], 3)} & "
            f"{fmt(row['els_ms_warm_median'], 3)} & "
            f"{fmt(row['entry_ms_warm_median'], 3)} & "
            f"{fmt(row['block_authorization_ms_warm_median'], 3)} & "
            f"{fmt(row['graph_ms_warm_median'], 3)} & "
            f"{fmt(row['residual_ms_warm_median'], 3)} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    lines.extend([
        r"\begin{table*}[t]", r"\centering", r"\scriptsize",
        r"\caption{Search work at the same profiled operating points.}",
        r"\label{tab:query-work}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{rlrrrrrrrrrr}", r"\toprule",
        r"Sel. & Method & Entries & Matched pts. & Visited & Base edges & Special intra & Special inter & Total edges & Entry dist. & Graph dist. & Total dist. \\",
        r"\midrule",
    ])
    for row in rows:
        label = "0-layer" if row["method"] == BASELINE_METHOD else "gated DRH"
        lines.append(
            f"{100.0 * number(row, 'mean_selectivity'):.3f}\\% & {label} & "
            f"{fmt(row['num_entries_warm_median'], 0)} & "
            f"{fmt(row['entry_group_matched_points_warm_median'], 0)} & "
            f"{fmt(row['nodes_visited_warm_median'], 0)} & "
            f"{fmt(row['regular_edges_scanned_warm_median'], 0)} & "
            f"{fmt(row['special_intra_edges_scanned_warm_median'], 0)} & "
            f"{fmt(row['special_inter_edges_scanned_warm_median'], 0)} & "
            f"{fmt(row['total_edges_scanned_warm_median'], 0)} & "
            f"{fmt(row['entry_point_distance_calcs_warm_median'], 0)} & "
            f"{fmt(row['graph_search_distance_calcs_warm_median'], 0)} & "
            f"{fmt(row['total_distance_calcs_warm_median'], 0)} " + r"\\")
    lines.extend([
        r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""])
    return lines


def render_build(
    summary: list[dict[str, str]], end_to_end: list[dict[str, str]],
) -> list[str]:
    lines = [
        r"\subsection{Construction}",
        "Construction time is process wall time through validated files on disk. "
        "GPU entries are end-to-end measurements, not isolated kernel timings.",
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Component construction time, footprint, and measured resources.}",
        r"\label{tab:build-components}",
        r"\begin{tabular}{lllrrrrrl}", r"\toprule",
        r"Component & Structure & Profile & Repeats & Median s & P95 s & Index MiB & Host/GPU MiB & GPU evidence \\",
        r"\midrule",
    ]
    for row in sorted(summary, key=lambda item: (
            item["component"], item["structure"], item["profile"])):
        gpu = fmt(row["peak_gpu_memory_mib"], 1) \
            if row["peak_gpu_memory_mib"] not in (None, "") else "--"
        host = fmt(row["peak_rss_mib"], 1) \
            if row["peak_rss_mib"] not in (None, "") else "--"
        if parse_bool(row["gpu_required"]):
            evidence = (
                "lock" if parse_bool(row["gpu_exclusive_lock"]) else
                f"idle-{integer(row, 'gpu_idle_samples_min')}")
        else:
            evidence = "--"
        lines.append(
            f"{tex_escape(row['component'])} & {tex_escape(row['structure'])} & "
            f"{tex_escape(row['profile'])} & {integer(row, 'measured_repeats')} & "
            f"{fmt(row['wall_median_seconds'], 2)} & {fmt(row['wall_p95_seconds'], 2)} & "
            f"{fmt(row['index_median_mib'], 1)} & {host}/{gpu} & "
            f"{tex_escape(evidence)} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    lines.extend([
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Full-index construction cost composed from separately measured base and hierarchy stages. Confidence intervals use independent resampling across stages.}",
        r"\label{tab:build-e2e}",
        r"\begin{tabular}{llrrrrl}", r"\toprule",
        r"Structure & Hierarchy profile & Stage repeats & Composed total s & Speedup & 95\% CI & No-slower support \\",
        r"\midrule",
    ])
    for row in sorted(end_to_end, key=lambda item: (
            item["structure"], item["hierarchy_profile"])):
        lines.append(
            f"{tex_escape(row['structure'])} & {tex_escape(row['hierarchy_profile'])} & "
            f"{integer(row, 'stage_repeats')} & "
            f"{fmt(row['composed_base_plus_hierarchy_seconds'], 2)} & "
            f"{fmt(row['speedup_vs_original_cpu'], 2)}$\\times$ & "
            f"[{fmt(row['speedup_ci95_low'], 2)}, {fmt(row['speedup_ci95_high'], 2)}] & "
            f"{'yes' if parse_bool(row['no_slower_supported']) else 'no'} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return lines


def build_quality_identity(name: str) -> tuple[str, str]:
    if name.startswith("quality_base_"):
        return "base", name.removeprefix("quality_base_")
    if name.startswith("quality_"):
        remainder = name.removeprefix("quality_")
        profile = next(
            (candidate for candidate in ("full_gpu_wmma", "full_gpu", "cpu")
             if remainder.endswith("_" + candidate)), "unknown")
        return remainder.removesuffix("_" + profile), profile
    return name, "unknown"


def render_build_quality(
    quality: dict[tuple[str, str], dict[str, str]],
    config: dict[str, Any], workloads: list[str],
) -> list[str]:
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Downstream query quality of independently constructed CPU/GPU indexes. QPS is diagnostic; construction comparisons use wall time.}",
        r"\label{tab:build-quality}",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Index method & Build profile & Recall crossings & Min Recall & Geomean QPS \\",
        r"\midrule",
    ]
    for method in config["methods"]:
        name = str(method["name"])
        rows = [quality[(name, workload)] for workload in workloads
                if (name, workload) in quality]
        crossings = len(rows)
        minimum = min((number(row, "recall_min") for row in rows), default=None)
        qps = ([number(row, "qps_warm_median") for row in rows]
               if crossings == len(workloads) else [])
        structure, profile = build_quality_identity(name)
        lines.append(
            f"{tex_escape(structure)} & {tex_escape(profile)} & "
            f"{crossings}/{len(workloads)} & "
            f"{minimum:.4f}" if minimum is not None else
            f"{tex_escape(structure)} & {tex_escape(profile)} & "
            f"{crossings}/{len(workloads)} & NC")
        lines[-1] += (
            f" & {geometric_mean(qps):.1f} " + r"\\"
            if qps else " & NC " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return lines


def render_claims(
    depth: dict[tuple[str, str], dict[str, str]],
    workloads: list[str], heldout_rows: list[dict[str, str]],
    end_to_end: list[dict[str, str]],
) -> tuple[str, str]:
    routed = [depth[("automatic_routed", workload)] for workload in workloads]
    complete = [row for row in routed if row["status"] == "complete"]
    high = [row for row in complete if number(row, "mean_selectivity") >= 0.60]
    if not high:
        raise ValueError("gated DRH has no high-selectivity Amazon crossing")
    high_speedups = [number(row, "speedup_vs_plain") for row in high]
    heldout_complete = [
        row for row in heldout_rows if row["status"] == "complete"]
    heldout_fraction = (
        min(number(row, "automatic_qps_fraction_of_oracle")
            for row in heldout_complete)
        if heldout_complete else None)
    best_build = max(
        end_to_end, key=lambda row: number(row, "speedup_vs_original_cpu"))
    build_speedup = number(best_build, "speedup_vs_original_cpu")
    build_lo = number(best_build, "speedup_ci95_low")
    build_hi = number(best_build, "speedup_ci95_high")
    oracle_clause = (
        f"and retains at least {100.0 * heldout_fraction:.1f}\\% of the "
        "per-workload manual-oracle QPS on those crossings"
        if heldout_fraction is not None else
        "while no held-out automatic crossing is observed"
    )
    abstract = (
        f"On Amazon, gated DRH reaches the conservative Recall target on "
        f"{len(complete)}/{len(workloads)} selectivity workloads and provides "
        f"{min(high_speedups):.2f}--{max(high_speedups):.2f}$\\times$ speedup "
        "over zero-layer LNG at the measured selectivities of at least 60\\%. "
        f"Across the held-out study it crosses "
        f"{len(heldout_complete)}/{len(heldout_rows)} workloads {oracle_clause}. "
        f"The best stage-composed accelerated-base-plus-hierarchy cost is "
        f"{build_speedup:.2f}$\\times$ the original CPU base build "
        f"(95\\% CI [{build_lo:.2f}, {build_hi:.2f}])."
    )
    conclusion = (
        f"Across the completed evidence, gated DRH crosses "
        f"{len(complete)}/{len(workloads)} Amazon workloads and "
        f"{len(heldout_complete)}/{len(heldout_rows)} held-out workloads; at "
        f"Amazon selectivities of at least 60\\% its speedup is "
        f"{min(high_speedups):.2f}--{max(high_speedups):.2f}$\\times$. "
        f"The strongest stage-composed full-index construction configuration reaches "
        f"{build_speedup:.2f}$\\times$ speedup with an independent-bootstrap "
        "95\\% interval "
        f"of [{build_lo:.2f}, {build_hi:.2f}]."
    )
    return abstract, conclusion


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_recall_qps_figures(
    figure_dir: Path, screen_config: Path, screen_points: Path,
) -> tuple[dict[str, Path], list[Path]]:
    """Validate complete measured curve artifacts and their input binding."""
    manifest_path = figure_dir / "plot_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("allow_partial") is not False:
        raise ValueError("Recall-QPS plot manifest permits partial results")
    if manifest.get("config_sha256") != source_digest(screen_config):
        raise ValueError("Recall-QPS plots do not match the screen config")
    if manifest.get("all_points_sha256") != source_digest(screen_points):
        raise ValueError("Recall-QPS plots do not match measured screen points")
    families = manifest.get("families")
    expected = {name for name, _ in RECALL_QPS_FIGURES}
    if not isinstance(families, dict) or set(families) != expected:
        raise ValueError("Recall-QPS plot family set is incomplete")
    pdfs: dict[str, Path] = {}
    inputs = [manifest_path]
    for family, _ in RECALL_QPS_FIGURES:
        record = families[family]
        if record.get("missing_method_workloads") != []:
            raise ValueError(f"Recall-QPS plot is partial: {family}")
        for suffix in ("pdf", "png"):
            path = figure_dir / f"{family}.{suffix}"
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(path)
            inputs.append(path)
            if suffix == "pdf":
                pdfs[family] = path.resolve()
    return pdfs, inputs


def render_recall_qps_figures(figures: dict[str, Path]) -> str:
    lines = [r"\newcommand{\authoritativeRecallQPSFigures}{%"]
    for family, caption in RECALL_QPS_FIGURES:
        path = figures[family]
        lines.extend([
            r"\begin{figure*}[t]",
            r"\centering",
            rf"\includegraphics[width=0.98\textwidth]{{\detokenize{{{path}}}}}",
            rf"\caption{{Measured Recall@10--QPS curves for {caption}. "
            r"Every marker is an executed screen point, lines follow increasing "
            r"$L_{search}$ without monotonic smoothing, and the dashed line is "
            r"the predeclared Recall target.}",
            rf"\label{{fig:{family.replace('_', '-')}}}",
            r"\end{figure*}",
        ])
    lines.append("}")
    return "\n".join(lines)


def generate_markdown_report(
    paths: ResultPaths,
    amazon_formal_config: dict[str, Any],
    amazon_profile_config: dict[str, Any],
    heldout_policies: list[dict[str, Any]],
    build_quality_config: dict[str, Any],
    figures: dict[str, Path],
    report_path: Path,
) -> str:
    """Render the presentation report from the same already-validated inputs."""
    workloads = validate_amazon_workloads(amazon_formal_config)
    formal_rows = read_csv(paths.amazon_formal, QUERY_FIELDS)
    formal = validate_formal_rows(
        formal_rows, amazon_formal_config, "Amazon formal")
    depth_rows = read_csv(paths.amazon_depth, {
        "status", "dataset", "workload", "mean_selectivity", "target_recall",
        "category", "method", "hierarchy", "entry_strategy", "lsearch",
        "recall_min", "qps", "speedup_vs_plain", "speedup_ci95_low",
        "speedup_ci95_high",
    })
    depth = validate_depth_rows(depth_rows, workloads)
    depth_global = read_csv(paths.amazon_depth_global, {
        "dataset", "category", "status", "method", "hierarchy",
        "entry_strategy", "routing_policy", "geomean_qps",
        "speedup_vs_plain", "workload_count",
    })
    validate_depth_global(depth_global)
    profile_rows = read_csv(paths.amazon_profile, PROFILE_FIELDS)
    profile = validate_profile_rows(
        profile_rows, amazon_profile_config, formal)
    heldout_rows = read_csv(paths.heldout_workload, {
        "status", "dataset", "workload", "mean_selectivity", "target_recall",
        "baseline_recall_min", "automatic_hierarchy", "automatic_recall_min",
        "oracle_hierarchy", "oracle_recall_min",
        "automatic_speedup_vs_baseline", "automatic_qps_fraction_of_oracle",
        "automatic_speedup_ci95_low", "automatic_speedup_ci95_high",
        "automatic_oracle_fraction_ci95_low",
        "automatic_oracle_fraction_ci95_high",
    })
    heldout_global = read_csv(paths.heldout_global, {
        "dataset", "status", "workload_count", "automatic_speedup_vs_baseline",
        "automatic_qps_fraction_of_global_oracle",
    })
    heldout_configs = [
        {"dataset": policy["dataset"], "workloads": policy["workloads"]}
        for policy in heldout_policies
    ]
    validate_heldout_rows(heldout_rows, heldout_global, heldout_configs)
    build_summary = read_csv(paths.build_summary, {
        "component", "structure", "profile", "measured_repeats",
        "wall_median_seconds", "wall_p95_seconds", "index_median_mib",
        "peak_rss_mib", "peak_gpu_memory_mib", "gpu_required",
        "gpu_exclusive_lock", "gpu_idle_samples_min",
    })
    build_e2e = read_csv(paths.build_end_to_end, {
        "structure", "hierarchy_profile", "stage_repeats", "composition_method",
        "original_cpu_base_median_seconds",
        "composed_base_plus_hierarchy_seconds",
        "speedup_vs_original_cpu", "speedup_ci95_low", "speedup_ci95_high",
        "overhead_vs_accelerated_base_median", "no_slower_supported",
    })
    validate_build_rows(build_summary, build_e2e)
    quality_workloads = validate_amazon_workloads(build_quality_config)
    if quality_workloads != workloads:
        raise ValueError("Amazon formal and build-quality workloads differ")
    quality_rows = read_csv(paths.build_quality_formal, QUERY_FIELDS)
    quality = validate_formal_rows(
        quality_rows, build_quality_config, "Build quality formal")

    lines = [
        "# ML-UNG 权威实验报告",
        "",
        "> 本文由最终 fail-closed 流水线从已验证实验文件自动生成。所有 QPS "
        "比较均使用相同数据、查询、精确 ground truth、K、线程数、查询二进制、"
        "graph backend 与 Recall 协议。",
        "",
        "## 1. 方法与评估口径",
        "",
        "ML-UNG 将层数与阈值、每层 LNG/Trie topology、三种 entry-group "
        "strategy（original、optimized_lng、trie）以及 routing 明确解耦。候选只"
        "扫描其 activation level 拥有的边，不进行 edge fallthrough、隐式晋级或"
        "跨层混扫。DRH 仅使用 N、层内最大度 R 和跨 block 度 C 自动决定层数、"
        "阈值与逐层 topology，不读取 query distribution、延迟或 Recall。",
        "",
        "Recall crossing 定义为所有 warm repeats 均达到 Recall@10 >= 0.90 的"
        "最小实测 Lsearch；不插值、不外推。screen 使用 2 个 warm repeats，formal "
        "使用 15 个 warm repeats，profile 使用独立 instrumentation binary 且不进入"
        "主 QPS。",
        "",
        "## 2. 数据集与自动层次",
        "",
        "| 数据集 | N | 维度 | workload | queries | 平均选择率 | DRH |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for policy in sorted(heldout_policies, key=lambda row: str(row["dataset"])):
        inputs = policy["inputs"]
        plan = ", ".join(
            f"{layer['min_points']}:{str(layer['topology']).upper()}"
            for layer in policy["automatic_hierarchy_layers"])
        for workload in sorted(
                policy["workloads"], key=lambda row: float(row["mean_selectivity"])):
            lines.append(
                f"| {policy['dataset']} | {int(inputs['num_points']):,} | "
                f"{int(inputs['dimension'])} | {workload['name']} | "
                f"{int(workload['num_queries']):,} | "
                f"{100.0 * float(workload['mean_selectivity']):.3f}% | {plan} |")

    lines.extend([
        "", "Amazon 是开发集，使用 602,453 个 768D vectors 和九档选择率；"
        "Genome、Reviews、VariousImg 仅用于冻结后的迁移验证。",
        "", "## 3. Amazon 层数消融：等 Recall QPS", "",
        "| 选择率 | 0 层 QPS | 最优 1 层 QPS / vs 0 | 最优 2 层 QPS / vs 0 | "
        "gated DRH QPS / vs 0 |",
        "|---:|---:|---:|---:|---:|",
    ])
    for workload in workloads:
        rowset = {category: depth[(category, workload)] for category in (
            "plain", "best_one_layer", "best_two_layer", "automatic_routed")}
        cells = []
        for category in ("plain", "best_one_layer", "best_two_layer",
                         "automatic_routed"):
            row = rowset[category]
            if row["status"] != "complete":
                cells.append("NC")
            elif category == "plain":
                cells.append(fmt(row["qps"], 1))
            else:
                cells.append(
                    f"{fmt(row['qps'], 1)} / {fmt(row['speedup_vs_plain'], 2)}x "
                    f"[{fmt(row['speedup_ci95_low'], 2)}, "
                    f"{fmt(row['speedup_ci95_high'], 2)}]")
        selectivity = 100.0 * number(rowset["plain"], "mean_selectivity")
        lines.append(f"| {selectivity:.3f}% | " + " | ".join(cells) + " |")

    lines.extend([
        "", "同一个配置覆盖全部九档选择率的汇总：", "",
        "| 类别 | 方法 | 层次 | 入口策略 | Geomean QPS | vs 0 层 |",
        "|---|---|---|---|---:|---:|",
    ])
    order = {name: index for index, name in enumerate(DEPTH_CATEGORIES)}
    for row in sorted(depth_global, key=lambda item: order[item["category"]]):
        complete = row["status"] == "complete"
        lines.append(
            f"| {compact_method(row)} | {row['method']} | {row['hierarchy']} | "
            f"{row['entry_strategy']} | "
            f"{fmt(row['geomean_qps'], 1) if complete else 'NC'} | "
            f"{fmt(row['speedup_vs_plain'], 2) + 'x' if complete else 'NC'} |")

    lines.extend([
        "", "## 4. 零层 Trie 与 LNG", "",
        "主比较同时展示系统默认组合；随后给出固定 entry strategy 的完整曲线，"
        "用于把 topology 效应与入口算法效应分开。NC 表示实测范围内未达到 Recall。",
        "", "| 选择率 | LNG L/QPS | Trie L/QPS | Trie/LNG |",
        "|---:|---:|---:|---:|",
    ])
    for workload in workloads:
        lng = formal[(BASELINE_METHOD, workload)]
        trie = formal.get((PRINCIPAL_TRIE_METHOD, workload))
        if trie is None:
            trie_cell = ratio = "NC"
        else:
            trie_qps = number(trie, "qps_warm_median")
            trie_cell = f"{integer(trie, 'lsearch'):,}/{trie_qps:.1f}"
            ratio = f"{trie_qps / number(lng, 'qps_warm_median'):.2f}x"
        lines.append(
            f"| {100.0 * number(lng, 'mean_selectivity'):.3f}% | "
            f"{integer(lng, 'lsearch'):,}/{number(lng, 'qps_warm_median'):.1f} | "
            f"{trie_cell} | {ratio} |")

    def append_markdown_ablation(
        heading: str, methods: tuple[tuple[str, str], ...], note: str,
    ) -> None:
        declared = {str(method["name"])
                    for method in amazon_formal_config["methods"]}
        missing = [name for name, _ in methods if name not in declared]
        if missing:
            raise ValueError(f"{heading}: methods are absent from config: {missing}")
        lines.extend([
            "", f"### {heading}", "", note, "",
            "| 选择率 | " + " | ".join(label for _, label in methods) + " |",
            "|---:|" + "---:|" * len(methods),
        ])
        for workload in workloads:
            baseline = formal[(BASELINE_METHOD, workload)]
            cells = []
            for name, _ in methods:
                row = formal.get((name, workload))
                cells.append(
                    "NC" if row is None else
                    f"{integer(row, 'lsearch'):,}/"
                    f"{number(row, 'qps_warm_median'):.1f}")
            lines.append(
                f"| {100.0 * number(baseline, 'mean_selectivity'):.3f}% | "
                + " | ".join(cells) + " |")

    append_markdown_ablation(
        "零层 topology 与入口策略完整消融", ZERO_LAYER_ABLATION,
        "六种正交组合在保守 Recall crossing 下的结果；每格为 Lsearch/QPS，"
        "NC 表示声明的实测网格内未 crossing。")

    append_markdown_ablation(
        "两层逐层 topology 消融", TWO_LAYER_TOPOLOGY_ABLATION,
        "固定 T1=1,024、T2=16,384 和 optimized LNG entry；每格为 "
        "Lsearch/QPS。")
    append_markdown_ablation(
        "层数与阈值尺度消融", THRESHOLD_DEPTH_ABLATION,
        "固定 optimized LNG entry；一层使用 LNG，二层使用 LNG/Trie；"
        "每格为 Lsearch/QPS。")
    append_markdown_ablation(
        "固定 DRH 的入口策略消融", DRH_ENTRY_ABLATION,
        "固定 1,024:LNG、16,384:Trie 的 ungated hierarchy；每格为 "
        "Lsearch/QPS。")

    lines.extend([
        "", "## 5. 无校准 DRH 与冻结人工 Oracle", "",
        "Oracle candidate set 由 DRH 和 35 个预先冻结的手工替代方案组成；所有候选"
        "使用同一个 exact upper-authorization gate，因此差距只反映深度、阈值和"
        "逐层 topology。",
        "", "| 数据集 | workload | 选择率 | DRH/plain [95% CI] | "
        "DRH/oracle [95% CI] | DRH | Oracle |",
        "|---|---|---:|---:|---:|---|---|",
    ])
    for row in sorted(heldout_rows, key=lambda item: (
            item["dataset"], number(item, "mean_selectivity"))):
        if row["status"] == "complete":
            speed = (
                f"{fmt(row['automatic_speedup_vs_baseline'], 2)}x "
                f"[{fmt(row['automatic_speedup_ci95_low'], 2)}, "
                f"{fmt(row['automatic_speedup_ci95_high'], 2)}]")
            oracle = (
                f"{fmt(row['automatic_qps_fraction_of_oracle'], 2)} "
                f"[{fmt(row['automatic_oracle_fraction_ci95_low'], 2)}, "
                f"{fmt(row['automatic_oracle_fraction_ci95_high'], 2)}]")
        else:
            speed = oracle = "NC"
        lines.append(
            f"| {row['dataset']} | {row['workload']} | "
            f"{100.0 * number(row, 'mean_selectivity'):.3f}% | {speed} | "
            f"{oracle} | {row['automatic_hierarchy']} | {row['oracle_hierarchy']} |")
    lines.extend([
        "", "跨 workload 的冻结配置结果：", "",
        "| 数据集 | workloads | DRH/plain | DRH/global oracle |",
        "|---|---:|---:|---:|",
    ])
    for row in sorted(heldout_global, key=lambda item: item["dataset"]):
        complete = row["status"] == "complete"
        lines.append(
            f"| {row['dataset']} | {integer(row, 'workload_count')} | "
            f"{fmt(row['automatic_speedup_vs_baseline'], 2) + 'x' if complete else 'NC'} | "
            f"{fmt(row['automatic_qps_fraction_of_global_oracle'], 2) if complete else 'NC'} |")

    role_by_name = {
        str(method["name"]): str(method.get("selection_role", ""))
        for method in amazon_profile_config["methods"]
    }
    drh_names = [name for name, role in role_by_name.items()
                 if role == ROUTED_DRH_ROLE]
    if len(drh_names) != 1:
        raise ValueError("profile report requires exactly one gated DRH method")
    selected_names = (BASELINE_METHOD, drh_names[0])
    lines.extend([
        "", "## 6. 查询阶段与工作量 Breakdown", "",
        "时间单位为 ms/query；profile 只用于机制解释，不进入主 QPS。",
        "", "| 选择率 | 方法 | 激活率 | 总时间 | 入口组 | 入口点 | 授权 | 图搜索 | "
        "Residual |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for workload in workloads:
        for name in selected_names:
            row = profile.get((name, workload))
            if row is None:
                continue
            label = "0-layer" if name == BASELINE_METHOD else "gated DRH"
            lines.append(
                f"| {100.0 * number(row, 'mean_selectivity'):.3f}% | {label} | "
                f"{100.0 * number(row, 'layered_path_activation_rate_warm_median'):.1f}% | "
                f"{fmt(row['query_total_ms_warm_median'], 3)} | "
                f"{fmt(row['els_ms_warm_median'], 3)} | "
                f"{fmt(row['entry_ms_warm_median'], 3)} | "
                f"{fmt(row['block_authorization_ms_warm_median'], 3)} | "
                f"{fmt(row['graph_ms_warm_median'], 3)} | "
                f"{fmt(row['residual_ms_warm_median'], 3)} |")
    lines.extend([
        "", "| 选择率 | 方法 | Entries | Matched points | 访问点 | Base edges | "
        "Special intra | Special inter | Total edges | Entry distances | "
        "Graph distances | Total distances |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for workload in workloads:
        for name in selected_names:
            row = profile.get((name, workload))
            if row is None:
                continue
            label = "0-layer" if name == BASELINE_METHOD else "gated DRH"
            lines.append(
                f"| {100.0 * number(row, 'mean_selectivity'):.3f}% | {label} | "
                f"{fmt(row['num_entries_warm_median'], 0)} | "
                f"{fmt(row['entry_group_matched_points_warm_median'], 0)} | "
                f"{fmt(row['nodes_visited_warm_median'], 0)} | "
                f"{fmt(row['regular_edges_scanned_warm_median'], 0)} | "
                f"{fmt(row['special_intra_edges_scanned_warm_median'], 0)} | "
                f"{fmt(row['special_inter_edges_scanned_warm_median'], 0)} | "
                f"{fmt(row['total_edges_scanned_warm_median'], 0)} | "
                f"{fmt(row['entry_point_distance_calcs_warm_median'], 0)} | "
                f"{fmt(row['graph_search_distance_calcs_warm_median'], 0)} | "
                f"{fmt(row['total_distance_calcs_warm_median'], 0)} |")

    lines.extend([
        "", "## 7. GPU 辅助构建", "",
        "构建由 base graph 和独立 hierarchy sidecar 两部分组成。base graph 依次"
        "构建 exact-label group 内图、group-to-group LNG、descendant/coverage "
        "metadata 和 vector-level cross-group edges；sidecar 再按层独立构建"
        " block topology、intra-block graph 与 inter-block edges。两者最终都"
        "物化为查询端直接加载的 host adjacency，不能表述为完全 device-resident "
        "construction。",
        "",
        "Base graph 的五个冻结 profile：",
        "",
        "| profile | group 内图 | metadata | cross-group edges |",
        "|---|---|---|---|",
        "| original_cpu | CPU Vamana | 原始 sort/LNG、hash-BFS descendants、"
        "topological coverage | 原始 CPU builder |",
        "| current_cpu | CPU Vamana | 优化后的 bucket/LNG/descendants/coverage | "
        "CPU Vamana builder |",
        "| naive_gpu | CPU Vamana | 优化 metadata | batched SGEMM + 独立 top-k |",
        "| paper_fused | CPU Vamana | 优化 metadata | grouped fused distance/top-k "
        "+ ID-only writeback |",
        "| accelerated_gpu | FastGrnnd CUDA | 优化 metadata | fused GPU cross edges "
        "+ CPU exact-scan additional edges |",
        "",
        "Fused cross-edge backend 在容量允许时常驻 base vectors 与 norms；否则按"
        "声明的显存预算流式处理 vector/query chunks。它消除逐 group-pair 调用和"
        "不需要的 distance writeback，但保留并计入 host adjacency materialization。",
        "",
        "Hierarchy sidecar 的 intra 与 inter backend 独立控制：",
        "",
        "| profile | intra-block route | inter-block route |",
        "|---|---|---|",
        "| cpu | small exact/complete；medium sampled Vamana；large CPU Vamana | CPU |",
        "| hybrid_gpu_intra | small/medium CPU；large FastGrnnd-style CUDA | CPU |",
        "| hybrid_gpu_intra_inter | 同上 | fused CUDA top-k |",
        "| full_gpu | bounded completion 后走 CUDA | fused CUDA-core top-k |",
        "| full_gpu_wmma | bounded completion 后走 CUDA | fused TF32-WMMA top-k |",
        "",
        "GPU case 必须在日志中证明请求的 intra/inter 工作确实发生，GPU intra "
        "fallback 必须为 0；full_gpu_wmma 还必须观测到 `mode=tf32_wmma`。每层"
        "序列化后都会 reload 并做结构验证。",
        "",
        "构建时间是进程端到端 wall time，包含产生并验证磁盘文件；resource pass "
        "与 timing pass 分离，CUDA kernel timing 不替代端到端时间。",
        "", "| 组件 | 结构 | profile | repeats | median s | p95 s | index MiB | "
        "peak RSS MiB | peak GPU MiB | GPU evidence |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in sorted(build_summary, key=lambda item: (
            item["component"], item["structure"], item["profile"])):
        if parse_bool(row["gpu_required"]):
            evidence = (
                "lock" if parse_bool(row["gpu_exclusive_lock"]) else
                f"idle-{integer(row, 'gpu_idle_samples_min')}")
        else:
            evidence = "--"
        lines.append(
            f"| {row['component']} | {row['structure']} | {row['profile']} | "
            f"{integer(row, 'measured_repeats')} | "
            f"{fmt(row['wall_median_seconds'], 2)} | "
            f"{fmt(row['wall_p95_seconds'], 2)} | "
            f"{fmt(row['index_median_mib'], 1)} | "
            f"{fmt(row['peak_rss_mib'], 1) if row['peak_rss_mib'] else '--'} | "
            f"{fmt(row['peak_gpu_memory_mib'], 1) if row['peak_gpu_memory_mib'] else '--'} | "
            f"{evidence} |")
    lines.extend([
        "", "分阶段实测后组合的完整构建成本：", "",
        "点估计为 base 与 hierarchy 各自 wall-time 中位数之和；95% CI 对三个"
        "阶段独立重采样，不把不同 phase 中相同编号的 repeat 当作配对样本。", "",
        "| 结构 | hierarchy profile | stage repeats | 原始 CPU s | "
        "组合 base + hierarchy s | speedup [95% CI] | hierarchy overhead |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in build_e2e:
        lines.append(
            f"| {row['structure']} | {row['hierarchy_profile']} | "
            f"{integer(row, 'stage_repeats')} | "
            f"{fmt(row['original_cpu_base_median_seconds'], 2)} | "
            f"{fmt(row['composed_base_plus_hierarchy_seconds'], 2)} | "
            f"{fmt(row['speedup_vs_original_cpu'], 2)}x "
            f"[{fmt(row['speedup_ci95_low'], 2)}, {fmt(row['speedup_ci95_high'], 2)}] | "
            f"{fmt(row['overhead_vs_accelerated_base_median'], 2)}x |")

    lines.extend([
        "", "### 构建产物的下游查询质量", "",
        "预声明 quality cohort 中的每个 CPU/GPU 构建产物均重新加载并执行相同的"
        " formal Recall 协议；"
        "QPS 仅用于发现构建质量回归，不替代构建 wall time。",
        "", "| 索引结构 | 构建 profile | Recall crossings | 最低 Recall | "
        "Geomean QPS |",
        "|---|---|---:|---:|---:|",
    ])
    for method in build_quality_config["methods"]:
        name = str(method["name"])
        rows = [quality[(name, workload)] for workload in workloads
                if (name, workload) in quality]
        structure, build_profile = build_quality_identity(name)
        minimum = min((number(row, "recall_min") for row in rows), default=None)
        qps = ([number(row, "qps_warm_median") for row in rows]
               if len(rows) == len(workloads) else [])
        lines.append(
            f"| {structure} | {build_profile} | {len(rows)}/{len(workloads)} | "
            f"{minimum:.4f}" if minimum is not None else
            f"| {structure} | {build_profile} | {len(rows)}/{len(workloads)} | NC")
        lines[-1] += (
            f" | {geometric_mean(qps):.1f} |" if qps else " | NC |")

    lines.extend([
        "", "## 8. Recall-QPS 曲线", "",
        "每个 marker 都是 screen 阶段实际执行点，连线按 Lsearch 递增顺序且不做"
        "单调平滑；虚线为预声明 Recall 门槛。",
        "",
    ])
    for family, caption in RECALL_QPS_FIGURES:
        png = figures[family].with_suffix(".png")
        relative = os.path.relpath(png, report_path.resolve().parent)
        lines.extend([f"### {caption}", "", f"![{caption}]({relative})", ""])
    lines.extend([
        "## 9. 解释边界", "",
        "- 更多层不保证单调加速：合法 upper seeds 仍共享有界队列，可能增加距离计算"
        "或挤占低层候选。",
        "- gate 未授权时，图搜索路径与对应零层方法一致，但 exact authorization "
        "本身仍有可测成本。",
        "- Trie topology 是 LNG 的稀疏替代，不保证保留全部 subset reachability；"
        "未达到 Recall 的点按 NC 报告。",
        "- DRH 是 query-independent structural heuristic，而非运行时最优性证明；"
        "其有效性由 held-out oracle regret 衡量。",
        "- 当前 GPU pipeline 仍保留 host adjacency materialization boundary，"
        "不能表述为完全 device-resident construction。",
        "",
    ])
    return "\n".join(lines)


def render_indirect_provenance(
    validator: Path,
    query_config_paths: list[Path],
    query_configs: list[dict[str, Any]],
    build_config_paths: list[Path],
    build_configs: list[dict[str, Any]],
) -> list[str]:
    """Bind validation code and every consulted run manifest to the output."""
    if len(query_config_paths) != len(query_configs):
        raise ValueError("query config provenance length mismatch")
    if len(build_config_paths) != len(build_configs):
        raise ValueError("build config provenance length mismatch")
    lines = [
        f"% validator-sha256 {validator.name} {source_digest(validator)}",
    ]
    for path, config in zip(query_config_paths, query_configs):
        lines.append(
            f"% manifest-sha256 {path.name} "
            f"{source_digest(manifest_path(config))}")
    for path, config in zip(build_config_paths, build_configs):
        manifest = Path(config["output_root"]) / "manifest.json"
        lines.append(
            f"% manifest-sha256 {path.name} {source_digest(manifest)}")
    return lines


def generate_document(
    paths: ResultPaths, amazon_formal_config: dict[str, Any],
    amazon_profile_config: dict[str, Any], heldout_configs: list[dict[str, Any]],
    heldout_policies: list[dict[str, Any]],
    build_quality_config: dict[str, Any],
) -> str:
    workloads = validate_amazon_workloads(amazon_formal_config)
    profile_workloads = validate_amazon_workloads(amazon_profile_config)
    if profile_workloads != workloads:
        raise ValueError("Amazon formal and profile workloads differ")

    formal_rows = read_csv(paths.amazon_formal, QUERY_FIELDS)
    formal = validate_formal_rows(formal_rows, amazon_formal_config, "Amazon formal")
    depth_rows = read_csv(paths.amazon_depth, {
        "status", "dataset", "workload", "mean_selectivity", "target_recall",
        "category", "method", "hierarchy", "entry_strategy", "lsearch",
        "recall_min", "qps", "speedup_vs_plain", "speedup_ci95_low",
        "speedup_ci95_high",
    })
    depth = validate_depth_rows(depth_rows, workloads)
    depth_global = read_csv(paths.amazon_depth_global, {
        "dataset", "category", "status", "method", "hierarchy",
        "entry_strategy", "routing_policy", "geomean_qps",
        "speedup_vs_plain", "workload_count",
    })
    validate_depth_global(depth_global)
    profile_rows = read_csv(paths.amazon_profile, PROFILE_FIELDS)
    profile = validate_profile_rows(
        profile_rows, amazon_profile_config, formal)
    heldout_rows = read_csv(paths.heldout_workload, {
        "status", "dataset", "workload", "mean_selectivity", "target_recall",
        "baseline_recall_min", "automatic_hierarchy", "automatic_recall_min",
        "oracle_hierarchy", "oracle_recall_min",
        "automatic_speedup_vs_baseline", "automatic_qps_fraction_of_oracle",
        "automatic_speedup_ci95_low", "automatic_speedup_ci95_high",
        "automatic_oracle_fraction_ci95_low",
        "automatic_oracle_fraction_ci95_high",
    })
    heldout_global = read_csv(paths.heldout_global, {
        "dataset", "status", "workload_count", "automatic_speedup_vs_baseline",
        "automatic_qps_fraction_of_global_oracle",
    })
    validate_heldout_rows(heldout_rows, heldout_global, heldout_configs)
    build_summary = read_csv(paths.build_summary, {
        "component", "structure", "profile", "measured_repeats",
        "wall_median_seconds", "wall_p95_seconds", "index_median_mib",
        "peak_rss_mib", "peak_gpu_memory_mib", "gpu_required",
        "gpu_exclusive_lock", "gpu_idle_samples_min",
    })
    build_end_to_end = read_csv(paths.build_end_to_end, {
        "structure", "hierarchy_profile", "stage_repeats", "composition_method",
        "original_cpu_base_median_seconds",
        "composed_base_plus_hierarchy_seconds",
        "speedup_vs_original_cpu", "speedup_ci95_low", "speedup_ci95_high",
        "overhead_vs_accelerated_base_median", "no_slower_supported",
    })
    validate_build_rows(build_summary, build_end_to_end)
    quality_workloads = validate_amazon_workloads(build_quality_config)
    if quality_workloads != workloads:
        raise ValueError("Amazon formal and build-quality workloads differ")
    build_quality_rows = read_csv(paths.build_quality_formal, QUERY_FIELDS)
    build_quality = validate_formal_rows(
        build_quality_rows, build_quality_config, "Build quality formal")

    body = []
    body.extend(render_depth_table(depth, workloads))
    body.extend(render_depth_global(depth_global))
    body.extend(render_principal_zero_layer_by_workload(formal, workloads))
    body.extend(render_zero_layer_table(formal, amazon_formal_config, workloads))
    body.extend(render_two_layer_topology_table(
        formal, amazon_formal_config, workloads))
    body.extend(render_threshold_depth_table(
        formal, amazon_formal_config, workloads))
    body.extend(render_drh_entry_table(
        formal, amazon_formal_config, workloads))
    body.extend(render_query_interpretation(formal, depth, profile, workloads))
    body.extend(render_heldout(heldout_rows, heldout_global))
    body.extend(render_profile(profile, amazon_profile_config, workloads))
    body.extend(render_build(build_summary, build_end_to_end))
    body.extend(render_build_quality(
        build_quality, build_quality_config, workloads))
    abstract, conclusion = render_claims(
        depth, workloads, heldout_rows, build_end_to_end)

    lines = [
        "% Generated by generate_authoritative_paper_results.py.",
        "% All inputs passed the authoritative fail-closed checks.",
    ]
    for field in paths.__dataclass_fields__:
        path = getattr(paths, field)
        lines.append(f"% source-sha256 {field} {source_digest(path)}")
    lines.extend([
        "",
        r"\newcommand{\authoritativeAbstractResult}{%",
        abstract,
        "}",
        r"\newcommand{\authoritativeConclusionResult}{%",
        conclusion,
        "}",
        r"\newcommand{\authoritativeDatasetTable}{%",
        *render_heldout_dataset_table(heldout_policies),
        "}",
        r"\newcommand{\authoritativeResults}{%",
        *body,
        "}",
    ])
    document = "\n".join(lines).rstrip() + "\n"
    if "\\pending" in document:
        raise AssertionError("authoritative output contains a pending marker")
    return document


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amazon-screen-config", type=Path, required=True)
    parser.add_argument("--amazon-formal-config", type=Path, required=True)
    parser.add_argument("--amazon-profile-config", type=Path, required=True)
    parser.add_argument("--heldout-formal-config", type=Path, action="append",
                        required=True)
    parser.add_argument("--heldout-policy", type=Path, action="append",
                        required=True)
    parser.add_argument("--build-quality-formal-config", type=Path, required=True)
    parser.add_argument("--build-config", type=Path, action="append", required=True)
    parser.add_argument("--amazon-formal", type=Path, required=True)
    parser.add_argument("--amazon-screen-points", type=Path, required=True)
    parser.add_argument("--amazon-figures-dir", type=Path, required=True)
    parser.add_argument("--amazon-depth", type=Path, required=True)
    parser.add_argument("--amazon-depth-global", type=Path, required=True)
    parser.add_argument("--amazon-profile", type=Path, required=True)
    parser.add_argument("--heldout-workload", type=Path, required=True)
    parser.add_argument("--heldout-global", type=Path, required=True)
    parser.add_argument("--build-summary", type=Path, required=True)
    parser.add_argument("--build-end-to-end", type=Path, required=True)
    parser.add_argument("--build-quality-formal", type=Path, required=True)
    parser.add_argument("--validator", type=Path,
                        default=HERE / "validate_selection_sweep.py")
    parser.add_argument(
        "--output", type=Path,
        default=HERE.parent.parent / "docs/papers/multilevel_ung/generated_results.tex")
    parser.add_argument(
        "--report-output", type=Path,
        default=HERE.parent.parent /
        "docs/reports/MULTILEVEL_SPECIAL_BLOCK_AUTHORITATIVE_RESULTS_CN.md")
    args = parser.parse_args(argv)
    if len(args.heldout_formal_config) != 3:
        parser.error("exactly three --heldout-formal-config values are required")
    if len(args.heldout_policy) != 3:
        parser.error("exactly three --heldout-policy values are required")
    if len(args.build_config) != 4:
        parser.error("exactly four --build-config values are required")

    amazon_screen, screen_hash = validated_query_config(
        args.amazon_screen_config.resolve(), args.validator.resolve(), "screen", 2)
    amazon_formal, formal_hash = validated_query_config(
        args.amazon_formal_config.resolve(), args.validator.resolve(), "formal", 15)
    amazon_profile, profile_hash = validated_query_config(
        args.amazon_profile_config.resolve(), args.validator.resolve(), "profile", 3)
    heldout_configs = []
    heldout_hashes = set()
    for path in args.heldout_formal_config:
        config, binary_hash = validated_query_config(
            path.resolve(), args.validator.resolve(), "formal", 15)
        heldout_configs.append(config)
        heldout_hashes.add(binary_hash)
    heldout_policies = validate_heldout_policy_set(
        args.heldout_policy, heldout_configs)
    build_quality_config, build_quality_hash = validated_query_config(
        args.build_quality_formal_config.resolve(), args.validator.resolve(),
        "formal", 15)
    validate_performance_binary_hashes(
        formal_hash, heldout_hashes, build_quality_hash, screen_hash)
    if validate_amazon_workloads(amazon_screen) != validate_amazon_workloads(
            amazon_formal):
        raise ValueError("Amazon screen and formal workloads differ")
    # A dedicated instrumented profile binary is permitted, but its provenance
    # remains explicit through the validated manifest and this diagnostic.
    if profile_hash != formal_hash:
        print(
            "profile uses a distinct instrumented binary: "
            f"formal={formal_hash} profile={profile_hash}", file=sys.stderr)
    build_configs = [validate_build_config(path.resolve())
                     for path in args.build_config]
    validate_build_config_set(build_configs)

    paths = ResultPaths(
        amazon_formal=args.amazon_formal.resolve(),
        amazon_depth=args.amazon_depth.resolve(),
        amazon_depth_global=args.amazon_depth_global.resolve(),
        amazon_profile=args.amazon_profile.resolve(),
        heldout_workload=args.heldout_workload.resolve(),
        heldout_global=args.heldout_global.resolve(),
        build_summary=args.build_summary.resolve(),
        build_end_to_end=args.build_end_to_end.resolve(),
        build_quality_formal=args.build_quality_formal.resolve(),
    )
    document = generate_document(
        paths, amazon_formal, amazon_profile, heldout_configs,
        [policy for _, _, policy in heldout_policies], build_quality_config)
    figures, figure_inputs = validate_recall_qps_figures(
        args.amazon_figures_dir.resolve(), args.amazon_screen_config.resolve(),
        args.amazon_screen_points.resolve())
    document = document.rstrip() + "\n" + render_recall_qps_figures(figures) + "\n"
    report = generate_markdown_report(
        paths, amazon_formal, amazon_profile,
        [policy for _, _, policy in heldout_policies], build_quality_config,
        figures,
        args.report_output.resolve())
    provenance = [
        f"% query-binary-sha256 formal-heldout-build-quality {formal_hash}",
        f"% query-binary-sha256 profile {profile_hash}",
    ]
    for dataset, path, _ in heldout_policies:
        provenance.append(
            f"% heldout-policy-sha256 {dataset} {source_digest(path)}")
    for path in (
            args.amazon_screen_config, args.amazon_formal_config,
            args.amazon_profile_config,
            *args.heldout_formal_config, args.build_quality_formal_config,
            *args.build_config):
        resolved = path.resolve()
        provenance.append(
            f"% config-sha256 {resolved.name} {source_digest(resolved)}")
    provenance.append(
        f"% source-sha256 amazon_screen_points "
        f"{source_digest(args.amazon_screen_points.resolve())}")
    for path in figure_inputs:
        provenance.append(
            f"% figure-sha256 {path.name} {source_digest(path)}")
    query_config_paths = [
        args.amazon_screen_config.resolve(),
        args.amazon_formal_config.resolve(),
        args.amazon_profile_config.resolve(),
        *(path.resolve() for path in args.heldout_formal_config),
        args.build_quality_formal_config.resolve(),
    ]
    provenance.extend(render_indirect_provenance(
        args.validator.resolve(), query_config_paths,
        [amazon_screen, amazon_formal, amazon_profile, *heldout_configs,
         build_quality_config],
        [path.resolve() for path in args.build_config], build_configs))
    document = "\n".join(provenance) + "\n" + document
    report = (
        "<!--\n" + "\n".join(line.removeprefix("% ") for line in provenance)
        + "\n-->\n\n" + report)
    atomic_write(args.output.resolve(), document)
    atomic_write(args.report_output.resolve(), report)
    print(args.output.resolve())
    print(args.report_output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
