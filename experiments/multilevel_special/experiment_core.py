#!/usr/bin/env python3
"""Shared experiment schema, statistics, and audit helpers.

This module is intentionally independent of a particular paper experiment.
Runners own process execution; this module owns the meanings of a method, a
measurement protocol, and a Recall-matched result.  Keeping those meanings in
one place prevents per-report scripts from silently changing the evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


LEGACY_ENTRY_PROVIDERS = {
    "cpu_min_super_sets",
    "cpu_bruteforce_els",
    "cpu_bruteforce_els_scalar",
    "gpu_cover_frontier",
    "special_block_trie",
}
ENTRY_STRATEGIES = {"original", "optimized_lng", "trie"}
GROUP_TOPOLOGIES = {"lng", "trie"}
MEASUREMENT_PASSES = {"performance", "profile"}
PROTOCOL_PHASES = {
    "smoke", "screen", "crossing", "formal", "profile", "critical",
}


class ExperimentConfigError(ValueError):
    """Raised when an experiment declaration is ambiguous or unsafe."""


@dataclass(frozen=True)
class Protocol:
    phase: str
    cold_repeats: int
    measured_repeats: int
    recall_rule: str
    bootstrap_samples: int
    paired_repeats: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    validate_config(config)
    return config


def protocol_for(config: dict[str, Any]) -> Protocol:
    raw = config.get("protocol", {})
    repeats = int(config.get("num_repeats", 0))
    cold = int(raw.get("cold_repeats", 1 if repeats > 1 else 0))
    measured = int(raw.get("measured_repeats", repeats - cold))
    return Protocol(
        phase=str(raw.get("phase", "formal" if repeats >= 7 else "screen")),
        cold_repeats=cold,
        measured_repeats=measured,
        recall_rule=str(raw.get("recall_rule", "all_repeats")),
        bootstrap_samples=int(raw.get("bootstrap_samples", 10000)),
        paired_repeats=bool(raw.get("paired_repeats", False)),
    )


def uses_orthogonal_method_schema(config: dict[str, Any]) -> bool:
    return config.get("method_schema") == "orthogonal_v2"


def method_main_index(config: dict[str, Any], method: dict[str, Any]) -> str:
    value = method.get("main_index", config.get("main_index"))
    if not value:
        raise ExperimentConfigError(
            f"{method.get('name', '<unnamed>')}: main_index is required")
    return str(value)


def hierarchy_layers(method: dict[str, Any]) -> list[dict[str, Any]]:
    """Return canonical arbitrary-depth layer declarations.

    Legacy configs are accepted for reproducing old evidence, but every new
    experiment must use the orthogonal_v2 schema and explicit layer topology.
    """
    declared = method.get("hierarchy_layers")
    if declared is not None:
        return [
            {"min_points": int(layer["min_points"]),
             "topology": str(layer["topology"])}
            for layer in declared
        ]
    count = int(method.get("layer_count", 0))
    layers: list[dict[str, Any]] = []
    if count >= 1:
        layers.append({"min_points": int(method["t1"]), "topology": "trie"})
    if count >= 2:
        layers.append({"min_points": int(method["t2"]), "topology": "trie"})
    return layers


def encode_hierarchy_layers(method: dict[str, Any]) -> str:
    return ",".join(
        f"{layer['min_points']}:{layer['topology']}"
        for layer in hierarchy_layers(method)
    )


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "search_app", "data_root", "gt_root", "output_root",
        "K", "num_threads", "num_entry_points", "num_repeats",
        "lsearch_values", "methods", "workloads",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ExperimentConfigError(f"missing config fields: {', '.join(missing)}")
    protocol = protocol_for(config)
    repeats = int(config["num_repeats"])
    if protocol.phase not in PROTOCOL_PHASES:
        raise ExperimentConfigError(f"unknown protocol phase: {protocol.phase}")
    if protocol.cold_repeats < 0 or protocol.measured_repeats <= 0:
        raise ExperimentConfigError("protocol requires non-negative cold and positive measured repeats")
    if protocol.cold_repeats + protocol.measured_repeats != repeats:
        raise ExperimentConfigError(
            "protocol cold_repeats + measured_repeats must equal num_repeats")
    if protocol.recall_rule not in {"all_repeats", "mean"}:
        raise ExperimentConfigError("recall_rule must be all_repeats or mean")
    if protocol.paired_repeats and not config.get("execution_blocks"):
        raise ExperimentConfigError(
            "paired_repeats requires declared execution_blocks; sequential method runs are not paired")
    measurement_pass = str(config.get("measurement_pass", "performance"))
    if measurement_pass not in MEASUREMENT_PASSES:
        raise ExperimentConfigError(
            f"measurement_pass must be one of {sorted(MEASUREMENT_PASSES)}")
    if (measurement_pass == "profile") != (protocol.phase == "profile"):
        raise ExperimentConfigError(
            "measurement_pass=profile and protocol.phase=profile must be declared together")

    method_names: set[str] = set()
    for method in config["methods"]:
        name = str(method.get("name", ""))
        if not name or name in method_names:
            raise ExperimentConfigError(f"missing or duplicate method name: {name!r}")
        method_names.add(name)
        method_main_index(config, method)
        layers = hierarchy_layers(method)
        special = bool(method.get("special_block_search", bool(layers)))
        if len(layers) > 254:
            raise ExperimentConfigError(f"{name}: at most 254 materialized layers are supported")
        previous = 0
        for layer in layers:
            threshold = int(layer["min_points"])
            topology = str(layer["topology"])
            if threshold <= previous:
                raise ExperimentConfigError(
                    f"{name}: hierarchy thresholds must be positive and strictly increasing")
            if topology not in GROUP_TOPOLOGIES:
                raise ExperimentConfigError(f"{name}: unknown layer topology {topology}")
            previous = threshold
        if bool(layers) != special:
            raise ExperimentConfigError(
                f"{name}: special_block_search must equal whether hierarchy_layers is non-empty")
        if layers and not method.get("block_index"):
            raise ExperimentConfigError(f"{name}: layered search requires block_index")
        if not layers and method.get("block_index"):
            raise ExperimentConfigError(f"{name}: zero-layer search cannot load block_index")
        if uses_orthogonal_method_schema(config):
            topology = str(method.get("base_topology", ""))
            if topology not in GROUP_TOPOLOGIES:
                raise ExperimentConfigError(f"{name}: base_topology must be lng or trie")
            strategy = str(method.get("entry_strategy", ""))
            if strategy not in ENTRY_STRATEGIES:
                raise ExperimentConfigError(
                    f"{name}: entry_strategy must be original, optimized_lng, or trie")
            if "entry_group_provider" in method:
                raise ExperimentConfigError(
                    f"{name}: orthogonal_v2 uses entry_strategy, not entry_group_provider")
        else:
            provider = str(method.get("entry_group_provider", "cpu_bruteforce_els"))
            if provider not in LEGACY_ENTRY_PROVIDERS:
                raise ExperimentConfigError(f"{name}: unknown entry_group_provider {provider}")
            if provider == "special_block_trie" and not special:
                raise ExperimentConfigError(f"{name}: special_block_trie requires Special Block search")

    workload_names = [str(item.get("name", "")) for item in config["workloads"]]
    if not all(workload_names) or len(set(workload_names)) != len(workload_names):
        raise ExperimentConfigError("workload names must be present and unique")
    thresholds = config.get("recall_thresholds", {})
    unknown_thresholds = sorted(set(thresholds) - set(workload_names))
    if unknown_thresholds:
        raise ExperimentConfigError(f"Recall thresholds reference unknown workloads: {unknown_thresholds}")


def iter_cases(config: dict[str, Any]) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for method in config["methods"]:
        enabled = method.get("enabled_workloads")
        for workload in config["workloads"]:
            if enabled is None or workload["name"] in enabled:
                yield method, workload


def method_semantics(config: dict[str, Any], method: dict[str, Any]) -> dict[str, Any]:
    """Return orthogonal method dimensions instead of relying on a nickname."""
    if uses_orthogonal_method_schema(config):
        layers = hierarchy_layers(method)
        return {
            "method": method["name"],
            "main_index": method_main_index(config, method),
            "base_topology": method["base_topology"],
            "entry_strategy": method["entry_strategy"],
            "hierarchy_plan": encode_hierarchy_layers(method) or "none",
            "layer_count": len(layers),
            "layer_topologies": ",".join(layer["topology"] for layer in layers) or "none",
            "special_search": bool(layers),
            "routing_policy": method.get("routing_policy", "always_layered"),
            "measurement_pass": config.get("measurement_pass", "performance"),
        }
    provider = str(method.get("entry_group_provider", "cpu_bruteforce_els"))
    if provider == "cpu_min_super_sets":
        entry_structure = "label trie; exact minimal supersets"
        coverage_structure = "none in provider"
    elif provider == "cpu_bruteforce_els":
        entry_structure = "inverted label bitsets; bounded frontier"
        coverage_structure = "LNG descendant sets (Roaring by default)"
    elif provider == "cpu_bruteforce_els_scalar":
        entry_structure = "scalar group-label scan"
        coverage_structure = "pairwise set-containment elimination"
    elif provider == "special_block_trie":
        entry_structure = "Special Block trie"
        coverage_structure = "terminal/block frontier"
    else:
        entry_structure = "GPU label-bitset frontier"
        coverage_structure = "LNG descendants"
    layers = len(hierarchy_layers(method))
    return {
        "method": method["name"],
        "main_index": config["main_index"],
        "main_graph": method.get("main_graph", "shared UNG vector graph"),
        "entry_group_provider": provider,
        "entry_structure": entry_structure,
        "coverage_structure": coverage_structure,
        "special_overlay": bool(method.get("special_block_search", False)),
        "layer_count": layers,
        "block_partition": method.get("block_partition", "none" if layers == 0 else "index metadata"),
        "t1": method.get("t1"),
        "t2": method.get("t2"),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def bootstrap_median_ratio(
    numerator: Sequence[float], denominator: Sequence[float], *, seed: int,
    samples: int = 10000, paired: bool = False,
) -> tuple[float, float]:
    if not numerator or not denominator:
        raise ValueError("bootstrap samples cannot be empty")
    if paired and len(numerator) != len(denominator):
        raise ValueError("paired bootstrap requires equal sample counts")
    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(samples):
        if paired:
            indices = [rng.randrange(len(numerator)) for _ in numerator]
            a = [numerator[i] for i in indices]
            b = [denominator[i] for i in indices]
        else:
            a = [numerator[rng.randrange(len(numerator))] for _ in numerator]
            b = [denominator[rng.randrange(len(denominator))] for _ in denominator]
        ratios.append(statistics.median(a) / statistics.median(b))
    return percentile(ratios, 0.025), percentile(ratios, 0.975)


def _rows_by_lsearch(rows: Iterable[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["Lsearch"]), []).append(row)
    return grouped


def choose_recall_crossing(
    rows: Sequence[dict[str, str]], threshold: float, expected_repeats: int,
    recall_rule: str = "all_repeats", cold_repeats: int = 0,
) -> tuple[int, list[dict[str, str]]]:
    if cold_repeats < 0 or cold_repeats >= expected_repeats:
        raise ValueError("cold_repeats must leave at least one measured repeat")
    for lsearch, candidates in sorted(_rows_by_lsearch(rows).items()):
        if len(candidates) != expected_repeats:
            raise ValueError(
                f"L={lsearch}: expected {expected_repeats} repeats, found {len(candidates)}")
        candidates.sort(key=lambda row: int(row.get("Repeat", 0)))
        measured = [row for row in candidates
                    if int(row.get("Repeat", 0)) >= cold_repeats]
        if len(measured) != expected_repeats - cold_repeats:
            raise ValueError(
                f"L={lsearch}: expected {expected_repeats - cold_repeats} measured "
                f"repeats, found {len(measured)}")
        recalls = [float(row["Avg_Recall"]) for row in measured]
        feasible = min(recalls) >= threshold if recall_rule == "all_repeats" else statistics.mean(recalls) >= threshold
        if feasible:
            return lsearch, candidates
    raise ValueError(f"no measured Lsearch reaches Recall {threshold}")


def summarize_case(
    run_dir: Path, *, threshold: float, protocol: Protocol, expected_repeats: int,
) -> tuple[dict[str, Any], list[float]]:
    details = read_csv(run_dir / "search_time_details.csv")
    lsearch, selected = choose_recall_crossing(
        details, threshold, expected_repeats, protocol.recall_rule,
        protocol.cold_repeats)
    selected.sort(key=lambda row: int(row["Repeat"]))
    warm_rows = [row for row in selected if int(row["Repeat"]) >= protocol.cold_repeats]
    if len(warm_rows) != protocol.measured_repeats:
        raise ValueError(
            f"{run_dir}: expected {protocol.measured_repeats} measured repeats, found {len(warm_rows)}")
    warm = [float(row["Time_ms"]) for row in warm_rows]
    recalls = [float(row["Avg_Recall"]) for row in warm_rows]
    mean = statistics.mean(warm)
    result: dict[str, Any] = {
        "lsearch": lsearch,
        "recall_min": min(recalls),
        "recall_mean": statistics.mean(recalls),
        "warm_median_ms": statistics.median(warm),
        "warm_mean_ms": mean,
        "warm_cv": statistics.stdev(warm) / mean if len(warm) > 1 else 0.0,
        "warm_p95_ms": percentile(warm, 0.95),
    }

    stage_path = run_dir / "search_stage_details.csv"
    if stage_path.is_file():
        stage = [row for row in read_csv(stage_path)
                 if int(row["Lsearch"]) == lsearch
                 and int(row["Repeat"]) >= protocol.cold_repeats]
        if len(stage) != protocol.measured_repeats:
            raise ValueError(f"{run_dir}: stage rows do not match measured repeats")
        stage_columns = {
            "els_median_ms_per_query": "AverageELS_ms",
            "entry_setup_median_ms_per_query": "AverageEntryPointSetup_ms",
            "authorization_median_ms_per_query": "AverageBlockAuthorization_ms",
            "graph_median_ms_per_query": "AverageGraphSearch_ms",
            "residual_median_ms_per_query": "AverageResidual_ms",
        }
        for output, source in stage_columns.items():
            result[output] = statistics.median(float(row[source]) for row in stage)
        result["max_abs_closure_error_ms"] = max(
            abs(float(row["ClosureError_ms"])) for row in stage)

    query_paths = sorted(run_dir.glob("query_details_repeat*.csv"))
    if query_paths:
        query_rows = [row for row in read_csv(query_paths[-1])
                      if int(row["Lsearch"]) == lsearch]
        if query_rows:
            query_columns = {
                "mean_num_entries": "NumEntries",
                "mean_dist_calcs": "DistCalcs",
                "mean_nodes_visited": "NumNodeVisited",
                "mean_entry_group_matches": "EntryGroupMatchedPoints",
            }
            for output, source in query_columns.items():
                if source in query_rows[0]:
                    result[output] = statistics.mean(float(row[source]) for row in query_rows)
    return result, warm


def summarize_experiment(
    config: dict[str, Any], baseline: str, *, bootstrap_samples: int | None = None,
    allow_partial: bool = False,
) -> list[dict[str, Any]]:
    protocol = protocol_for(config)
    thresholds = config.get("recall_thresholds", {})
    if not thresholds:
        raise ExperimentConfigError("summarization requires recall_thresholds")
    rows: list[dict[str, Any]] = []
    samples: dict[tuple[str, str], list[float]] = {}
    output_root = Path(config["output_root"])
    if config.get("pass_subdirs", False):
        output_root /= str(config.get("measurement_pass", "performance"))
    for method, workload in iter_cases(config):
        workload_name = workload["name"]
        if workload_name not in thresholds:
            raise ExperimentConfigError(f"missing Recall threshold for {workload_name}")
        run_dir = output_root / method["name"] / workload_name
        try:
            summary, warm = summarize_case(
                run_dir, threshold=float(thresholds[workload_name]),
                protocol=protocol, expected_repeats=int(config["num_repeats"]),
            )
        except (FileNotFoundError, ValueError) as error:
            if not allow_partial:
                raise
            rows.append({
                "method": method["name"], "workload": workload_name,
                "mean_selectivity": workload.get("mean_selectivity"),
                "target_recall": float(thresholds[workload_name]),
                "status": "unavailable", "reason": str(error),
            })
            continue
        summary.update({
            "method": method["name"],
            "workload": workload_name,
            "mean_selectivity": workload.get("mean_selectivity"),
            "target_recall": float(thresholds[workload_name]),
            "layer_count": len(hierarchy_layers(method)),
            "base_topology": method.get("base_topology"),
            "entry_strategy": method.get("entry_strategy",
                                          method.get("entry_group_provider", "cpu_bruteforce_els")),
            "hierarchy_plan": encode_hierarchy_layers(method) or "none",
            "measurement_pass": config.get("measurement_pass", "performance"),
            "status": "complete",
        })
        num_queries = int(workload.get("num_queries", config.get("expected_num_queries", 0)))
        if num_queries > 0:
            summary["warm_median_qps"] = 1000.0 * num_queries / summary["warm_median_ms"]
        rows.append(summary)
        samples[(method["name"], workload_name)] = warm

    by_key = {(row["method"], row["workload"]): row for row in rows}
    count = bootstrap_samples or protocol.bootstrap_samples
    for row in rows:
        if row.get("status") != "complete":
            continue
        workload = row["workload"]
        base_key = (baseline, workload)
        if base_key not in by_key:
            continue
        base = by_key[base_key]
        row["speedup_vs_baseline"] = base["warm_median_ms"] / row["warm_median_ms"]
        if row["method"] == baseline:
            row["speedup_ci95_low"] = 1.0
            row["speedup_ci95_high"] = 1.0
            continue
        lo, hi = bootstrap_median_ratio(
            samples[base_key], samples[(row["method"], workload)],
            seed=int(hashlib.sha256(f"{baseline}/{row['method']}/{workload}".encode()).hexdigest()[:8], 16),
            samples=count, paired=protocol.paired_repeats,
        )
        row["speedup_ci95_low"] = lo
        row["speedup_ci95_high"] = hi
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def existing_manifest_record(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    """Return the prior record matching all identity fields, if present."""
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    return next((row for row in state.get("runs", [])
                 if all(row.get(key) == value for key, value in identity.items())), None)


def require_matching_build_binary_for_reuse(
    existing: dict[str, Any] | None, expected_sha256: str,
) -> None:
    """Reject artifact reuse unless its original builder identity is explicit."""
    if existing is None:
        raise RuntimeError(
            "refusing to reuse build artifact without a prior manifest record")
    provenance = existing.get("provenance")
    if not isinstance(provenance, dict):
        provenance = existing.get("source_provenance")
    observed = (provenance.get("build_binary_sha256")
                if isinstance(provenance, dict) else None)
    if observed != expected_sha256:
        raise RuntimeError(
            "refusing to reuse build artifact from a different or unknown "
            f"binary: expected {expected_sha256}, observed {observed}")


def recovered_artifact_elapsed_seconds(run_dir: Path, command_name: str,
                                       log_name: str) -> float | None:
    """Approximate duration for legacy artifacts that predate elapsed ledgers."""
    command = run_dir / command_name
    log = run_dir / log_name
    if not command.is_file() or not log.is_file():
        return None
    return max(0.0, log.stat().st_mtime - command.stat().st_mtime)


def retain_elapsed_evidence(record: dict[str, Any], existing: dict[str, Any] | None,
                            run_dir: Path, command_name: str,
                            log_name: str) -> None:
    """Keep measured child wall time across resume without overstating legacy data."""
    if existing and existing.get("elapsed_seconds") is not None:
        for key in ("started_at", "finished_at", "elapsed_seconds",
                    "elapsed_source", "returncode"):
            if key in existing:
                record[key] = existing[key]
        record.setdefault("elapsed_source", "monotonic_child_wall")
        return
    recovered = recovered_artifact_elapsed_seconds(
        run_dir, command_name, log_name)
    if recovered is not None:
        record["elapsed_seconds"] = recovered
        record["elapsed_source"] = "artifact_mtime_approximation"
