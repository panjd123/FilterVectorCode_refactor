#!/usr/bin/env python3
"""Prepare a bounded, paper-first evidence campaign from frozen configs.

The full 35-alternative held-out oracle remains the preferred final study.
This deadline campaign deliberately keeps DRH plus five predeclared manual
alternatives that vary depth, threshold scale, and per-layer topology.  It
never reads query outcomes while choosing the subset.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RUN_ROOT = REPO / "runs/deadline_evidence_20260927"
DATASETS = ("genome", "reviews", "variousimg")
LSEARCH_VALUES = (40, 100, 500, 2500, 10000, 40000)
BASELINE_ROLE = "zero_layer_baseline"
AUTOMATIC_ROLE = "predeclared_degree_ratio_hierarchy_v1"
MANUAL_ROLE = "predeclared_manual_oracle_grid"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def layers(method: dict[str, Any]) -> tuple[tuple[int, str], ...]:
    return tuple(
        (int(layer["min_points"]), str(layer["topology"]))
        for layer in method.get("hierarchy_layers", []))


def structure_name(method: dict[str, Any]) -> str:
    name = str(method["name"])
    marker = "_entry_"
    if marker not in name:
        raise ValueError(f"method name lacks {marker}: {name}")
    return name.split(marker, 1)[0]


def one(rows: list[dict[str, Any]], description: str) -> dict[str, Any]:
    if len(rows) != 1:
        raise ValueError(f"expected one {description}, got {[r['name'] for r in rows]}")
    return rows[0]


def select_methods(config: dict[str, Any]) -> list[dict[str, Any]]:
    methods = config["methods"]
    baseline = one(
        [row for row in methods if row.get("selection_role") == BASELINE_ROLE],
        "zero-layer baseline")
    automatic = one(
        [row for row in methods if row.get("selection_role") == AUTOMATIC_ROLE],
        "automatic DRH")
    automatic_layers = layers(automatic)
    if len(automatic_layers) != 2:
        raise ValueError(f"deadline protocol expects two-layer DRH: {automatic_layers}")
    t1, t2 = automatic_layers[0][0], automatic_layers[1][0]
    manual = [row for row in methods if row.get("selection_role") == MANUAL_ROLE]

    one_layer = one([
        row for row in manual if layers(row) == ((t1, "lng"),)
    ], "one-layer T1 LNG alternative")
    same_ll = one([
        row for row in manual if layers(row) == ((t1, "lng"), (t2, "lng"))
    ], "same-threshold LNG/LNG alternative")
    same_tt = one([
        row for row in manual if layers(row) == ((t1, "trie"), (t2, "trie"))
    ], "same-threshold Trie/Trie alternative")

    lower = [row for row in manual if len(layers(row)) == 2
             and tuple(topology for _, topology in layers(row)) == ("lng", "trie")
             and layers(row)[0][0] < t1 and layers(row)[1][0] < t2]
    higher = [row for row in manual if len(layers(row)) == 2
              and tuple(topology for _, topology in layers(row)) == ("lng", "trie")
              and layers(row)[0][0] > t1 and layers(row)[1][0] > t2]
    lower_scale = max(lower, key=lambda row: layers(row)[0][0] * layers(row)[1][0])
    higher_scale = min(higher, key=lambda row: layers(row)[0][0] * layers(row)[1][0])
    selected = [baseline, automatic, one_layer, same_ll, same_tt,
                lower_scale, higher_scale]
    names = [str(row["name"]) for row in selected]
    if len(set(names)) != len(names):
        raise ValueError(f"deadline method subset contains duplicates: {names}")
    return selected


def prepare_dataset(dataset: str) -> dict[str, Any]:
    source_search = HERE / f"config.authoritative_heldout_{dataset}_screen.json"
    source_build = HERE / f"config.authoritative_heldout_{dataset}_build.json"
    search = json.loads(source_search.read_text(encoding="utf-8"))
    build = json.loads(source_build.read_text(encoding="utf-8"))
    selected = select_methods(search)
    hierarchy_root = RUN_ROOT / "heldout" / dataset / "hierarchy"
    search_root = RUN_ROOT / "heldout" / dataset / "search"

    selected_structures = {structure_name(row) for row in selected
                           if row.get("hierarchy_layers")}
    build["cases"] = [
        copy.deepcopy(row) for row in build["cases"]
        if str(row["name"]) in selected_structures]
    if {str(row["name"]) for row in build["cases"]} != selected_structures:
        raise ValueError(f"{dataset}: selected build cases are incomplete")
    build["output_root"] = str(hierarchy_root)
    for case in build["cases"]:
        # These sidecars exist only to evaluate held-out query behavior.  GPU
        # construction performance is measured by the separate Amazon build
        # study, so the backend declaration must match the CPU-disabled env
        # inherited from the frozen held-out grid.
        case["benchmark_profile"] = "cpu"
    build["purpose"] = (
        "Deadline-bounded held-out structural oracle: query-independent DRH "
        "plus five frozen manual alternatives.")

    deadline_methods = []
    for source in selected:
        method = copy.deepcopy(source)
        method["lsearch_values"] = list(LSEARCH_VALUES)
        if method.get("hierarchy_layers"):
            method["block_index"] = str(
                hierarchy_root / structure_name(method) / "block_index")
        deadline_methods.append(method)
    search["methods"] = deadline_methods
    search["output_root"] = str(search_root)
    search["lsearch_values"] = list(LSEARCH_VALUES)
    search["max_lsearch"] = max(LSEARCH_VALUES)
    search["num_repeats"] = 3
    search["protocol"] = {
        **search["protocol"], "phase": "screen", "cold_repeats": 1,
        "measured_repeats": 2, "recall_rule": "all_repeats",
        "paired_repeats": False,
    }
    search["purpose"] = (
        "Deadline-bounded held-out evidence. Candidate subset is selected "
        "without query outcomes; every case has one cold and two warm repeats.")

    build_path = HERE / f"config.deadline_heldout_{dataset}_build.json"
    search_path = HERE / f"config.deadline_heldout_{dataset}_search.json"
    atomic_json(build_path, build)
    atomic_json(search_path, search)
    return {
        "dataset": search["dataset"],
        "build_config": str(build_path),
        "search_config": str(search_path),
        "workloads": [row["name"] for row in search["workloads"]],
        "baseline_method": selected[0]["name"],
        "automatic_method": selected[1]["name"],
        "manual_methods": [row["name"] for row in selected[2:]],
        "hierarchy_cases": [row["name"] for row in build["cases"]],
    }


def main() -> int:
    datasets = [prepare_dataset(dataset) for dataset in DATASETS]
    manifest = {
        "schema_version": 1,
        "deadline": "2026-09-28T08:00:00+08:00",
        "query_calibrated": False,
        "selection_rule": (
            "DRH plus five predeclared structural alternatives: one-layer "
            "T1 LNG, same-threshold LL and TT, and nearest lower/higher LT scales."),
        "query_protocol": {
            "lsearch_values": list(LSEARCH_VALUES),
            "cold_repeats": 1,
            "warm_repeats": 2,
            "recall_crossing": "minimum measured L with every warm repeat >= target",
        },
        "datasets": datasets,
    }
    output = HERE / "config.deadline_evidence_manifest.json"
    atomic_json(output, manifest)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
