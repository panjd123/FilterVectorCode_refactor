#!/usr/bin/env python3
"""Summarize PTNC speedups using the exact inputs of the all-datasets plot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import plot_recall_qps_all_datasets as plot
from preprocess_search_results import processed_output_is_valid


THRESHOLDS = (0.9, 0.95)
SELECTIVITIES = {
    "query_minlen1_sel40to100pct": ("high", "40–100%"),
    "query_minlen1_sel10to40pct": ("medium", "10–40%"),
    "query_minlen1_sel0to10pct": ("low", "0–10%"),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_best_baselines(selected: list[dict]) -> tuple[list[dict], list[dict]]:
    """Compare PTNC with the fastest qualifying baseline in each task."""
    details, summary = [], []
    for threshold in THRESHOLDS:
        for task, (selectivity, selection_range) in SELECTIVITIES.items():
            group = []
            for dataset in plot.DATASET_ORDER:
                candidates = [row for row in selected if (
                    row["dataset"] == dataset and row["query_task"] == task
                    and row["recall_threshold"] == threshold
                )]
                ptnc = next(row for row in candidates if row["algorithm"] == "PTNC")
                baselines = [row for row in candidates if row["algorithm"] != "PTNC" and row["status"] == "ok"]
                best = max(baselines, key=lambda row: row["best_qualifying_qps"], default=None)
                valid = ptnc["status"] == "ok" and best is not None
                row = {
                    "recall_threshold": threshold, "dataset": dataset,
                    "query_task": task, "selectivity": selectivity,
                    "nominal_selectivity_range": selection_range,
                    "speedup": ptnc["best_qualifying_qps"] / best["best_qualifying_qps"] if valid else "NA",
                    "status": "ok" if valid else (
                        f"PTNC:{ptnc['status']}" if ptnc["status"] != "ok" else "no_qualifying_baseline"
                    ),
                    "best_baseline": best["algorithm"] if best else "NA",
                    "ptnc_qps": ptnc["best_qualifying_qps"],
                    "best_baseline_qps": best["best_qualifying_qps"] if best else "NA",
                    "ptnc_selected_recall": ptnc["selected_recall"],
                    "best_baseline_selected_recall": best["selected_recall"] if best else "NA",
                    "qualified_baseline_count": len(baselines),
                    "qualified_baselines": ";".join(item["algorithm"] for item in baselines),
                    "excluded_baselines": ";".join(item["algorithm"] for item in candidates if item["algorithm"] != "PTNC" and item["status"] != "ok"),
                    "tied_best_baselines": ";".join(item["algorithm"] for item in baselines if item["best_qualifying_qps"] == best["best_qualifying_qps"]) if best else "",
                }
                group.append(row)
                details.append(row)
            valid_rows = [row for row in group if row["status"] == "ok"]
            ratios = [row["speedup"] for row in valid_rows]
            low, high = (min(ratios), max(ratios)) if ratios else ("NA", "NA")
            summary.append({
                "recall_threshold": threshold, "selectivity": selectivity,
                "nominal_selectivity_range": selection_range,
                "valid_dataset_count": len(ratios),
                "mean_speedup": statistics.fmean(ratios) if ratios else "NA",
                "max_speedup": high, "min_speedup": low,
                "max_datasets": ";".join(row["dataset"] for row in valid_rows if row["speedup"] == high),
                "min_datasets": ";".join(row["dataset"] for row in valid_rows if row["speedup"] == low),
                "ptnc_faster_count": sum(value > 1 for value in ratios),
                "na_datasets": ";".join(row["dataset"] for row in group if row["status"] != "ok"),
            })
    return details, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=plot.RESULT_ROOT / "ptnc_baseline_speedups",
    )
    args = parser.parse_args()
    # Read existing drawing inputs without changing experiment or drawing CSVs.
    for dataset in plot.DATASETS:
        for algorithm in plot.ALGORITHM_RESULTS:
            source = plot.resolve_source(dataset, algorithm)
            if source is not None and not processed_output_is_valid(plot.processed_path(source)):
                raise ValueError(f"Run the plotting preprocessor first: {source}")
    panels, missing = plot.load_all_series()
    selected = []
    lookups = {}
    provenance = []
    for dataset, panel in zip(plot.DATASETS, panels):
        name, task = str(dataset["dataset"]), str(dataset["query_task"])
        curves = {series.label: series for series in panel}
        for algorithm in plot.ALGORITHM_RESULTS:
            label = str(algorithm["label"])
            source = plot.resolve_source(dataset, algorithm)
            drawing = plot.processed_path(source) if source is not None else None
            series = curves.get(label)
            points = series.points if series else ()
            maximum = max((point.recall for point in points), default=None)
            if drawing is not None:
                provenance.append({
                    "dataset": name, "query_task": task, "algorithm": label,
                    "source_csv": str(source), "drawing_csv": str(drawing),
                    "drawing_sha256": hashlib.sha256(drawing.read_bytes()).hexdigest(),
                })
            for threshold in THRESHOLDS:
                qualifying = [point for point in points if point.recall >= threshold]
                best = max(qualifying, key=lambda point: (point.qps, point.recall), default=None)
                row = {
                    "dataset": name, "query_task": task,
                    "selectivity": SELECTIVITIES[task][0],
                    "recall_threshold": threshold, "algorithm": label,
                    "status": "ok" if best else ("recall_below_threshold" if points else "missing_curve"),
                    "max_recall": maximum if maximum is not None else "NA",
                    "selected_recall": best.recall if best else "NA",
                    "best_qualifying_qps": best.qps if best else "NA",
                    "source_csv": str(source) if source is not None else "NA",
                    "drawing_csv": str(drawing) if drawing is not None else "NA",
                }
                selected.append(row)
                lookups[name, task, threshold, label] = row

    baselines = [label for label in plot.LEGEND_ORDER if label != "PTNC"]
    details, summary = [], []
    for threshold in THRESHOLDS:
        for task, (selectivity, selection_range) in SELECTIVITIES.items():
            for baseline in baselines:
                group = []
                for name in plot.DATASET_ORDER:
                    ptnc = lookups[name, task, threshold, "PTNC"]
                    other = lookups[name, task, threshold, baseline]
                    valid = ptnc["status"] == other["status"] == "ok"
                    row = {
                        "recall_threshold": threshold, "selectivity": selectivity,
                        "nominal_selectivity_range": selection_range,
                        "dataset": name, "baseline": baseline,
                        "speedup": ptnc["best_qualifying_qps"] / other["best_qualifying_qps"] if valid else "NA",
                        "status": "ok" if valid else (
                            f"PTNC:{ptnc['status']}" if ptnc["status"] != "ok" else f"baseline:{other['status']}"
                        ),
                        "ptnc_qps": ptnc["best_qualifying_qps"],
                        "baseline_qps": other["best_qualifying_qps"],
                        "ptnc_selected_recall": ptnc["selected_recall"],
                        "baseline_selected_recall": other["selected_recall"],
                        "baseline_max_recall": other["max_recall"],
                    }
                    group.append(row)
                    details.append(row)
                eligible = [row for row in group if row["status"] == "ok"]
                ratios = [row["speedup"] for row in eligible]
                low = min(ratios) if ratios else "NA"
                high = max(ratios) if ratios else "NA"
                summary.append({
                    "recall_threshold": threshold, "selectivity": selectivity,
                    "nominal_selectivity_range": selection_range, "baseline": baseline,
                    "valid_dataset_count": len(ratios),
                    "mean_speedup": statistics.fmean(ratios) if ratios else "NA",
                    "max_speedup": high, "min_speedup": low,
                    "max_datasets": ";".join(row["dataset"] for row in eligible if row["speedup"] == high),
                    "min_datasets": ";".join(row["dataset"] for row in eligible if row["speedup"] == low),
                    "na_datasets": ";".join(row["dataset"] for row in group if row["status"] != "ok"),
                })

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "details.csv", details)
    write_csv(output / "selected_points.csv", selected)
    best_details, best_summary = summarize_best_baselines(selected)
    write_csv(output / "best_baseline_details.csv", best_details)
    write_csv(output / "best_baseline_summary.csv", best_summary)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plot_script": str(Path(plot.__file__).resolve()),
        "method": "For each algorithm, use max QPS among plotted points with recall >= threshold; no interpolation.",
        "speedup": "PTNC best qualifying QPS / baseline best qualifying QPS",
        "best_baseline_speedup": "PTNC best qualifying QPS / max best qualifying QPS over all non-PTNC algorithms in the same dataset/task; exclude baselines below the recall threshold.",
        "mean": "Unweighted arithmetic mean of per-dataset speedups; exclude NA.",
        "qps": "num_queries * 1000 / Average_Time_ms, as in plot.read_points",
        "inputs": "Existing search_time_summary_draw.csv files; plot preprocessing independently sorts only the time column after dropping the first row.",
        "plot_sha256": hashlib.sha256(Path(plot.__file__).read_bytes()).hexdigest(),
        "missing_curves": missing, "provenance": provenance,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary)} summary rows, {len(details)} comparisons, {len(selected)} selected-point rows to {output}")
    for threshold in THRESHOLDS:
        print(f"\nRecall >= {threshold}")
        print("selectivity | baseline | n | mean | max | min | NA")
        for row in summary:
            if row["recall_threshold"] == threshold:
                values = [f"{row[key]:.2f}" if isinstance(row[key], float) else str(row[key])
                          for key in ("selectivity", "baseline", "valid_dataset_count", "mean_speedup", "max_speedup", "min_speedup", "na_datasets")]
                print(" | ".join(values))
    print("\nPTNC versus the fastest qualifying baseline:")
    for row in best_details:
        speedup = f"{row['speedup']:.6f}" if row["status"] == "ok" else "NA"
        print(f"{row['recall_threshold']} | {row['selectivity']} | {row['dataset']} | {row['best_baseline']} | {speedup}")
    print("\nBest-baseline summary:")
    for row in best_summary:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
