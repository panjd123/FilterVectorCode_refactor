#!/usr/bin/env python3
import argparse
import csv
import os
import statistics


DETAIL_COLS = [
    "Time_ms",
    "search_time_ms",
    "core_search_time_ms",
    "Recall",
    "DistCalcs",
    "NumNodeVisited",
]


def read_detail_rows(root, query):
    path = os.path.join(root, query, "results", "query_details_repeat1.csv")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_summary_rows(root, query):
    path = os.path.join(root, query, "results", "search_time_summary.csv")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean(values):
    xs = list(values)
    return statistics.mean(xs) if xs else 0.0


def summarize_details(rows, min_lsearch):
    selected = [r for r in rows if int(r["Lsearch"]) >= min_lsearch]
    out = {"n": len(selected)}
    for col in DETAIL_COLS:
        out[col] = mean(float(r[col]) for r in selected)
    out["search_noncore_ms"] = out["search_time_ms"] - out["core_search_time_ms"]
    out["total_nonsearch_ms"] = out["Time_ms"] - out["search_time_ms"]
    return out


def summarize_batch(rows, min_lsearch):
    selected = [r for r in rows if int(r["Lsearch"]) >= min_lsearch]
    return {
        "n": len(selected),
        "Average_Time_ms": mean(float(r["Average_Time_ms"]) for r in selected),
        "Average_Recall": mean(float(r["Average_Recall"]) for r in selected),
    }


def print_metric(name, ung_value, special_value, unit="ms"):
    ratio = special_value / ung_value if ung_value else 0.0
    speedup = ung_value / special_value if special_value else 0.0
    suffix = f" {unit}" if unit else ""
    print(
        f"{name:24s} ung={ung_value:10.4f}{suffix} "
        f"special={special_value:10.4f}{suffix} "
        f"special/ung={ratio:7.4f} speedup={speedup:7.4f}x"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ung-dir", required=True)
    parser.add_argument("--special-dir", required=True)
    parser.add_argument("--query", default="query_minlen1_cov100k")
    parser.add_argument("--min-lsearch", type=int, default=2000)
    args = parser.parse_args()

    ung_details = summarize_details(read_detail_rows(args.ung_dir, args.query), args.min_lsearch)
    special_details = summarize_details(read_detail_rows(args.special_dir, args.query), args.min_lsearch)
    ung_batch = summarize_batch(read_summary_rows(args.ung_dir, args.query), args.min_lsearch)
    special_batch = summarize_batch(read_summary_rows(args.special_dir, args.query), args.min_lsearch)

    print(f"query={args.query} min_lsearch={args.min_lsearch}")
    print(f"detail_rows ung={ung_details['n']} special={special_details['n']}")
    for col in ["Time_ms", "search_time_ms", "core_search_time_ms",
                "search_noncore_ms", "total_nonsearch_ms"]:
        print_metric(col, ung_details[col], special_details[col])
    print_metric("Recall", ung_details["Recall"], special_details["Recall"], unit="")
    print_metric("DistCalcs", ung_details["DistCalcs"], special_details["DistCalcs"], unit="")
    print_metric("NumNodeVisited", ung_details["NumNodeVisited"], special_details["NumNodeVisited"], unit="")
    print(f"batch_rows ung={ung_batch['n']} special={special_batch['n']}")
    print_metric("Batch Average_Time_ms", ung_batch["Average_Time_ms"], special_batch["Average_Time_ms"])
    print_metric("Batch Average_Recall", ung_batch["Average_Recall"], special_batch["Average_Recall"], unit="")


if __name__ == "__main__":
    main()
