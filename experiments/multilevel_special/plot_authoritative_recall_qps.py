#!/usr/bin/env python3
"""Plot the declared Amazon Recall/QPS ablations from measured CSV points."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FAMILIES = {
    "principal_zero": [
        ("l0_lng_entry_optimized_lng", "LNG-0 + optimized LNG entry"),
        ("l0_trie_entry_trie", "Trie-0 + Trie entry"),
    ],
    "zero_topology_fixed_entry": [
        ("l0_lng_entry_optimized_lng", "LNG-0"),
        ("l0_trie_entry_optimized_lng", "Trie-0"),
    ],
    "depth_fixed_lng": [
        ("l0_lng_entry_optimized_lng", "0 layer"),
        ("l1_t1024_lng_entry_optimized_lng", "1 layer: 1024/LNG"),
        ("l2_t1024_16384_ll_entry_optimized_lng", "2 layers: LNG/LNG"),
        ("l2_t1024_16384_lt_entry_optimized_lng", "2 layers: LNG/Trie (DRH)"),
    ],
    "two_layer_topology": [
        ("l2_t1024_16384_ll_entry_optimized_lng", "LNG/LNG"),
        ("l2_t1024_16384_lt_entry_optimized_lng", "LNG/Trie"),
        ("l2_t1024_16384_tl_entry_optimized_lng", "Trie/LNG"),
        ("l2_t1024_16384_tt_entry_optimized_lng", "Trie/Trie"),
    ],
    "entry_strategy_on_drh": [
        ("l2_t1024_16384_lt_entry_original", "original LNG entry"),
        ("l2_t1024_16384_lt_entry_optimized_lng", "optimized LNG entry"),
        ("l2_t1024_16384_lt_entry_trie", "Trie entry"),
    ],
    "upper_authorization": [
        ("l2_t1024_16384_lt_entry_optimized_lng", "always layered"),
        ("l2_t1024_16384_lt_entry_optimized_lng_upper_routed", "upper authorized"),
    ],
}

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
MARKERS = ["o", "s", "^", "D", "v"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def workload_order(config: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(config["workloads"], key=lambda item: float(item["mean_selectivity"]))


def plot_family(
    rows: list[dict[str, str]], workloads: list[dict[str, Any]],
    methods: list[tuple[str, str]], target_by_workload: dict[str, float],
    output: Path, allow_partial: bool,
) -> list[tuple[str, str]]:
    columns = 3
    rows_count = math.ceil(len(workloads) / columns)
    figure, axes = plt.subplots(
        rows_count, columns, figsize=(10.2, 2.85 * rows_count), squeeze=False)
    missing: list[tuple[str, str]] = []
    legend_handles = []
    legend_labels = []
    for index, workload in enumerate(workloads):
        axis = axes[index // columns][index % columns]
        workload_name = str(workload["name"])
        for method_index, (method, label) in enumerate(methods):
            points = [
                item for item in rows
                if item["workload"] == workload_name and item["method"] == method
            ]
            if not points:
                missing.append((method, workload_name))
                continue
            points.sort(key=lambda item: (float(item["recall"]), int(item["lsearch"])))
            handle, = axis.plot(
                [float(item["recall"]) for item in points],
                [float(item["qps_warm_median"]) for item in points],
                color=COLORS[method_index % len(COLORS)],
                marker=MARKERS[method_index % len(MARKERS)],
                markersize=3.5, linewidth=1.25, label=label,
            )
            if index == 0:
                legend_handles.append(handle)
                legend_labels.append(label)
        axis.axvline(target_by_workload[workload_name], color="#666666",
                     linestyle="--", linewidth=0.8)
        axis.set_yscale("log")
        axis.grid(True, which="both", linewidth=0.35, alpha=0.45)
        axis.set_title(
            f"{100.0 * float(workload['mean_selectivity']):.3g}% selectivity",
            fontsize=9)
        axis.set_xlabel("Recall@10")
        axis.set_ylabel("QPS")
    for index in range(len(workloads), rows_count * columns):
        axes[index // columns][index % columns].axis("off")
    if missing and not allow_partial:
        plt.close(figure)
        preview = ", ".join(f"{method}/{workload}" for method, workload in missing[:8])
        raise RuntimeError(f"incomplete measured family {output.stem}: {preview}")
    if legend_handles:
        figure.legend(legend_handles, legend_labels, loc="upper center",
                      ncol=min(len(legend_handles), 4), frameon=False,
                      bbox_to_anchor=(0.5, 1.01))
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("all_points", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--family", action="append", choices=sorted(FAMILIES))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    points = read_csv(args.all_points)
    workloads = workload_order(config)
    requested = args.family or list(FAMILIES)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "all_points": str(args.all_points.resolve()),
        "all_points_sha256": sha256(args.all_points),
        "allow_partial": args.allow_partial,
        "families": {},
    }
    for family in requested:
        methods = FAMILIES[family]
        missing = plot_family(
            points, workloads, methods, config["recall_thresholds"],
            args.output_dir / family, args.allow_partial)
        manifest["families"][family] = {
            "methods": [method for method, _ in methods],
            "missing_method_workloads": [list(item) for item in missing],
        }
    (args.output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
