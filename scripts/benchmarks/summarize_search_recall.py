#!/usr/bin/env python3
import argparse
import csv
import statistics
from pathlib import Path


def percentile(values, p):
    if not values:
        return ""
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def read_build_time(run_dir):
    candidates = [
        run_dir / "results" / "build_time.csv",
        run_dir.parent / "results" / "build_time.csv",
    ]
    out = {}
    for path in candidates:
        if not path.exists():
            continue
        with path.open() as f:
            next(f, None)
            for line in f:
                line = line.strip()
                if not line or "," not in line:
                    continue
                k, v = line.split(",", 1)
                out[k] = v
        break
    return out


def summarize_run(name, run_dir):
    result_dir = run_dir / "results"
    summary_path = result_dir / "search_time_summary.csv"
    detail_paths = sorted(result_dir.glob("query_details_repeat*.csv"))
    detail_path = detail_paths[-1] if detail_paths else None
    if not summary_path.exists():
        raise FileNotFoundError(f"missing {summary_path}")

    details_by_l = {}
    if detail_path and detail_path.exists():
        with detail_path.open() as f:
            for row in csv.DictReader(f):
                lsearch = row["Lsearch"]
                details_by_l.setdefault(lsearch, []).append(row)

    build = read_build_time(run_dir)
    rows = []
    with summary_path.open() as f:
        for row in csv.DictReader(f):
            lsearch = row["Lsearch"]
            details = details_by_l.get(lsearch, [])
            times = [float(r["Time_ms"]) for r in details if r.get("Time_ms")]
            recalls = [float(r["Recall"]) for r in details if r.get("Recall")]
            nodes = [float(r["NumNodeVisited"]) for r in details if r.get("NumNodeVisited")]
            calcs = [float(r["DistCalcs"]) for r in details if r.get("DistCalcs")]
            rows.append(
                {
                    "variant": name,
                    "run_dir": str(run_dir),
                    "index_ms": build.get("index_time", ""),
                    "group_ms": build.get("build_graph_time", ""),
                    "cross_ms": build.get("build_cross_edges_time", ""),
                    "lsearch": lsearch,
                    "avg_efs": row.get("Average_Efs", ""),
                    "batch_time_ms": row.get("Average_Time_ms", ""),
                    "avg_recall": row.get("Average_Recall", ""),
                    "query_p50_ms": percentile(times, 50),
                    "query_p95_ms": percentile(times, 95),
                    "query_p99_ms": percentile(times, 99),
                    "mean_query_recall": statistics.mean(recalls) if recalls else "",
                    "mean_visited_nodes": statistics.mean(nodes) if nodes else "",
                    "mean_dist_calcs": statistics.mean(calcs) if calcs else "",
                    "query_details": str(detail_path) if detail_path else "",
                    "search_summary": str(summary_path),
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "runs",
        nargs="+",
        help="Run directories or name=run_dir pairs. Each run_dir should contain results/search_time_summary.csv.",
    )
    ap.add_argument("--output", "-o", default="", help="Optional CSV output path")
    args = ap.parse_args()

    rows = []
    for item in args.runs:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).name
        rows.extend(summarize_run(name, Path(path)))

    fields = [
        "variant",
        "index_ms",
        "group_ms",
        "cross_ms",
        "lsearch",
        "avg_efs",
        "batch_time_ms",
        "avg_recall",
        "query_p50_ms",
        "query_p95_ms",
        "query_p99_ms",
        "mean_query_recall",
        "mean_visited_nodes",
        "mean_dist_calcs",
        "run_dir",
        "search_summary",
        "query_details",
    ]
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        f = out.open("w", newline="")
        close = True
    else:
        import sys

        f = sys.stdout
        close = False
    try:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if close:
            f.close()


if __name__ == "__main__":
    main()
