#!/usr/bin/env python3
"""Plot the measured medians and min/max ranges from comparison.csv."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    with args.summary.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    order = {name: i for i, name in enumerate(("cpu", "gpu_baseline", "load_balance_only", "topk_only", "load_balance_topk"))}
    rows.sort(key=lambda row: order[row["variant"]])
    names = {"cpu": "CPU", "gpu_baseline": "GPU: neither", "load_balance_only": "GPU: balance",
             "topk_only": "GPU: topK", "load_balance_topk": "GPU: both"}
    end_to_end = "end_to_end_wall_ms_median" in rows[0]
    total_metric = "end_to_end_wall_ms" if end_to_end else "total_index_build_ms"
    total_title = "End-to-end: base Trie + block index" if end_to_end else "Base + block build timers (historical)"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    colors = {"cpu": "#64748b", "gpu_baseline": "#a855f7", "load_balance_only": "#f59e0b",
              "topk_only": "#3b82f6", "load_balance_topk": "#10b981"}
    for ax, metric, title in zip(axes, ("intra_ms", total_metric),
                                  ("Intra-block graph construction", total_title)):
        medians = [float(row[metric + "_median"]) / 1000 for row in rows]
        lower = [median - float(row[metric + "_min"]) / 1000 for median, row in zip(medians, rows)]
        upper = [float(row[metric + "_max"]) / 1000 - median for median, row in zip(medians, rows)]
        bars = ax.bar(range(len(rows)), medians, color=[colors[row["variant"]] for row in rows],
                      yerr=[lower, upper], capsize=4, width=0.65)
        ax.set_xticks(range(len(rows)), [names[row["variant"]] for row in rows], rotation=12)
        ax.set_ylabel("Time (seconds)")
        ax.set_title(title)
        ax.set_ylim(0, max(median + error for median, error in zip(medians, upper)) * 1.2)
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    counts = sorted({int(row["samples"]) for row in rows})
    fig.suptitle(f"Amazon: block threshold 500, degree 32 — samples per method: {counts}")
    for suffix in (".png", ".pdf"):
        output = args.summary.with_suffix(suffix)
        fig.savefig(output, dpi=180)
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
