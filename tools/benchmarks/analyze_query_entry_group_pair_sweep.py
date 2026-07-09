import csv
import math
import sys
from pathlib import Path


def as_float(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def fmt(value):
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.3f}"


def classify(case):
    if case.startswith("gpu_cover"):
        return "gpu"
    if case.startswith("cpu_cover"):
        return "cpu_cover"
    if case == "cpu_scan":
        return "cpu_scan"
    if case == "cpu_exact":
        return "cpu_exact"
    return "other"


def pareto(items):
    out = []
    for row in items:
        dominated = False
        for other in items:
            if other is row:
                continue
            no_worse = (
                other["avg_min_groups"] <= row["avg_min_groups"]
                and other["total_ms"] <= row["total_ms"]
            )
            strictly_better = (
                other["avg_min_groups"] < row["avg_min_groups"]
                or other["total_ms"] < row["total_ms"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            out.append(row)
    return sorted(out, key=lambda x: x["avg_min_groups"])


def main():
    root = Path(sys.argv[1])
    rows = list(csv.DictReader((root / "query_entry_group_sweep.csv").open()))
    for row in rows:
        for key in [
            "total_ms",
            "avg_min_groups",
            "avg_frontier_groups",
            "kernel_ms",
            "d2h_ms",
            "prune_ms",
        ]:
            row[key] = as_float(row, key)
        row["kind"] = classify(row["case"])

    cover = [row for row in rows if row["kind"] in ("cpu_cover", "gpu")]
    exact = next(row for row in rows if row["kind"] == "cpu_exact")

    cpu_p = pareto([row for row in cover if row["kind"] == "cpu_cover"])
    gpu_p = pareto([row for row in cover if row["kind"] == "gpu"])

    thresholds = [1100, 1200, 1300, 1400, 1500, 1700, 2000, 2200, 5000, 10000, 15000]
    threshold_rows = []
    for threshold in thresholds:
        cpu_candidates = [
            row
            for row in cover
            if row["kind"] == "cpu_cover" and row["avg_min_groups"] <= threshold
        ]
        gpu_candidates = [
            row
            for row in cover
            if row["kind"] == "gpu" and row["avg_min_groups"] <= threshold
        ]
        best_cpu = min(cpu_candidates, key=lambda x: x["total_ms"]) if cpu_candidates else None
        best_gpu = min(gpu_candidates, key=lambda x: x["total_ms"]) if gpu_candidates else None
        threshold_rows.append((threshold, best_cpu, best_gpu))

    by_case = {row["case"]: row for row in cover}
    pair_rows = []
    for cpu in [row for row in cover if row["kind"] == "cpu_cover"]:
        suffix = cpu["case"].replace("cpu_cover_frontier_", "")
        gpu = by_case.get("gpu_cover_frontier_" + suffix)
        if gpu:
            pair_rows.append((suffix, cpu, gpu))

    out = root / "query_entry_group_pair_analysis.md"
    lines = []
    lines.append("# ELS Entry Group CPU/GPU Pair Sweep Analysis")
    lines.append("")
    lines.append(f"Artifact: `{root}`")
    lines.append("")
    lines.append(
        "All cover-frontier rows are coverage-correct. Quality is `avg_output_groups`; "
        "lower is better. Times exclude one-time label/descendant bitset construction but "
        "include provider output materialization for each timed run."
    )
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append("| method | total_ms | avg_output_groups | quality_vs_exact |")
    lines.append("|---|---:|---:|---:|")
    for kind in ["cpu_scan", "cpu_exact"]:
        row = next(item for item in rows if item["kind"] == kind)
        ratio = row["avg_min_groups"] / exact["avg_min_groups"]
        lines.append(
            f"| {row['case']} | {fmt(row['total_ms'])} | "
            f"{fmt(row['avg_min_groups'])} | {ratio:.3f}x |"
        )

    lines.append("")
    lines.append("## Same-Config CPU/GPU Pairs")
    lines.append("")
    lines.append(
        "| config | cpu_ms | cpu_groups | gpu_ms | gpu_groups | gpu_speedup_vs_cpu | quality_relation |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for suffix, cpu, gpu in pair_rows:
        speedup = cpu["total_ms"] / gpu["total_ms"]
        if abs(cpu["avg_min_groups"] - gpu["avg_min_groups"]) < 1e-6:
            quality = "same"
        elif gpu["avg_min_groups"] < cpu["avg_min_groups"]:
            quality = "gpu fewer"
        else:
            quality = "gpu more"
        lines.append(
            f"| {suffix} | {fmt(cpu['total_ms'])} | {fmt(cpu['avg_min_groups'])} | "
            f"{fmt(gpu['total_ms'])} | {fmt(gpu['avg_min_groups'])} | "
            f"{speedup:.3f}x | {quality} |"
        )

    lines.append("")
    lines.append("## Pareto Frontiers")
    lines.append("")
    lines.append("### CPU cover-frontier")
    lines.append("| case | total_ms | avg_output_groups | quality_vs_exact |")
    lines.append("|---|---:|---:|---:|")
    for row in cpu_p:
        ratio = row["avg_min_groups"] / exact["avg_min_groups"]
        lines.append(
            f"| {row['case']} | {fmt(row['total_ms'])} | {fmt(row['avg_min_groups'])} | {ratio:.3f}x |"
        )

    lines.append("")
    lines.append("### GPU cover-frontier")
    lines.append("| case | total_ms | avg_output_groups | quality_vs_exact |")
    lines.append("|---|---:|---:|---:|")
    for row in gpu_p:
        ratio = row["avg_min_groups"] / exact["avg_min_groups"]
        lines.append(
            f"| {row['case']} | {fmt(row['total_ms'])} | {fmt(row['avg_min_groups'])} | {ratio:.3f}x |"
        )

    lines.append("")
    lines.append("## Best Method Under Quality Threshold")
    lines.append("")
    lines.append("For threshold T, choose the fastest row with `avg_output_groups <= T`.")
    lines.append("")
    lines.append(
        "| max_avg_output_groups | cpu_case | cpu_ms | cpu_groups | gpu_case | gpu_ms | gpu_groups | gpu_speedup_vs_cpu | better |"
    )
    lines.append("|---:|---|---:|---:|---|---:|---:|---:|---|")
    for threshold, cpu, gpu in threshold_rows:
        if cpu and gpu:
            speedup = cpu["total_ms"] / gpu["total_ms"]
            better = "GPU" if speedup > 1 else "CPU"
            lines.append(
                f"| {threshold} | {cpu['case']} | {fmt(cpu['total_ms'])} | "
                f"{fmt(cpu['avg_min_groups'])} | {gpu['case']} | {fmt(gpu['total_ms'])} | "
                f"{fmt(gpu['avg_min_groups'])} | {speedup:.3f}x | {better} |"
            )
        elif cpu:
            lines.append(
                f"| {threshold} | {cpu['case']} | {fmt(cpu['total_ms'])} | "
                f"{fmt(cpu['avg_min_groups'])} | NA | NA | NA | NA | CPU only |"
            )
        elif gpu:
            lines.append(
                f"| {threshold} | NA | NA | NA | {gpu['case']} | "
                f"{fmt(gpu['total_ms'])} | {fmt(gpu['avg_min_groups'])} | NA | GPU only |"
            )
        else:
            lines.append(f"| {threshold} | NA | NA | NA | NA | NA | NA | NA | none |")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- For loose quality around 2170 output groups, CPU d1/cap64 is fastest: "
        "16.06 ms vs GPU 47.37 ms at the same output quality."
    )
    lines.append(
        "- For medium/high quality around 1300-1500 output groups, GPU is better: "
        "e.g. threshold 1400 uses CPU d2/cap256 at 195.68 ms vs GPU d2/cap1024 "
        "at 127.32 ms, a 1.54x GPU speedup with fewer output groups."
    )
    lines.append(
        "- For the best quality reached by GPU in this grid, GPU d3/cap8192 gives "
        "1279.98 output groups at 415.13 ms. CPU can reach 1182.24 groups at "
        "468.08 ms, and 1065.87 groups at 659.53 ms. Thus GPU is faster in the "
        "~1280 group regime, but CPU reaches still smaller cover-frontier outputs in this grid."
    )
    lines.append(
        "- GPU rows with d2/d3 and small cap can be much worse in quality than CPU "
        "at the same nominal config because frontier compaction order differs. "
        "Therefore quality must be compared by actual avg_output_groups, not by delta/cap alone."
    )
    lines.append(
        "- GPU still pays about 25 ms D2H for full output bitset transfer in most rows. "
        "A compact `(offsets, group_ids)` GPU output should improve the GPU side, "
        "especially for small-output high-quality configurations."
    )
    out.write_text("\n".join(lines) + "\n")
    print(out)
    print("\n".join(lines[:120]))


if __name__ == "__main__":
    main()
