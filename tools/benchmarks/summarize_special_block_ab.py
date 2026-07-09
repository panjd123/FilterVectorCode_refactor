#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any, Iterable


def read_kv_csv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def read_summary_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def mean(values: Iterable[float]) -> float:
    xs = list(values)
    return statistics.mean(xs) if xs else 0.0


def pctl(values: Iterable[float], pct: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def bucket_for(value: float, cuts: list[float]) -> str:
    lo = 0.0
    for cut in cuts:
        if value < cut:
            return f"[{lo:.2f},{cut:.2f})"
        lo = cut
    return f"[{lo:.2f},1.00]"


def detail_paths(result_dir: Path) -> list[Path]:
    return sorted(result_dir.glob("query_details_repeat*.csv"))


def summarize_query_details(
    result_dir: Path, cuts: list[float], lsearch: str | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    for path in detail_paths(result_dir):
        with path.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    if lsearch not in (None, ""):
        rows = [r for r in rows if r.get("Lsearch") == lsearch]

    if not rows:
        return {}, [], []

    overall = {
        "query_rows": len(rows),
        "mean_query_ms": mean(to_float(r.get("Time_ms")) for r in rows),
        "p50_query_ms": pctl((to_float(r.get("Time_ms")) for r in rows), 50),
        "p95_query_ms": pctl((to_float(r.get("Time_ms")) for r in rows), 95),
        "mean_recall": mean(to_float(r.get("Recall")) for r in rows),
        "mean_dist_calcs": mean(to_float(r.get("DistCalcs")) for r in rows),
        "mean_visited": mean(to_float(r.get("NumNodeVisited")) for r in rows),
        "weighted_special_ratio": weighted_ratio(rows, "SpecialQueryPoints", "SpecialQueryMatchedPoints"),
        "weighted_nontrivial_ratio": weighted_ratio(rows, "SpecialQueryNontrivialPoints", "SpecialQueryMatchedPoints"),
        "mean_special_edges_scanned": mean(to_float(r.get("SpecialEdgesScanned")) for r in rows),
        "mean_regular_edges_scanned": mean(to_float(r.get("SpecialRegularEdgesScanned")) for r in rows),
        "mean_free_dist_calcs": mean(to_float(r.get("SpecialFreeDistCalcs")) for r in rows),
        "mean_regular_dist_calcs": mean(to_float(r.get("SpecialRegularDistCalcs")) for r in rows),
    }

    by_special = summarize_buckets(rows, "SpecialQueryRatio", cuts, "special")
    by_nontrivial = summarize_buckets(rows, "SpecialQueryNontrivialRatio", cuts, "nontrivial")
    return overall, by_special, by_nontrivial


def weighted_ratio(rows: list[dict[str, str]], num_key: str, den_key: str) -> float:
    num = sum(to_float(r.get(num_key)) for r in rows)
    den = sum(to_float(r.get(den_key)) for r in rows)
    return num / den if den else 0.0


def summarize_buckets(rows: list[dict[str, str]], ratio_key: str, cuts: list[float], prefix: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        b = bucket_for(to_float(row.get(ratio_key)), cuts)
        buckets.setdefault(b, []).append(row)
    out = []
    for bucket in sorted(buckets):
        xs = buckets[bucket]
        out.append(
            {
                "bucket_type": prefix,
                "bucket": bucket,
                "query_rows": len(xs),
                "mean_query_ms": mean(to_float(r.get("Time_ms")) for r in xs),
                "p95_query_ms": pctl((to_float(r.get("Time_ms")) for r in xs), 95),
                "mean_recall": mean(to_float(r.get("Recall")) for r in xs),
                "mean_dist_calcs": mean(to_float(r.get("DistCalcs")) for r in xs),
                "mean_visited": mean(to_float(r.get("NumNodeVisited")) for r in xs),
                "weighted_special_ratio": weighted_ratio(xs, "SpecialQueryPoints", "SpecialQueryMatchedPoints"),
                "weighted_nontrivial_ratio": weighted_ratio(xs, "SpecialQueryNontrivialPoints", "SpecialQueryMatchedPoints"),
                "mean_special_edges_scanned": mean(to_float(r.get("SpecialEdgesScanned")) for r in xs),
                "mean_regular_edges_scanned": mean(to_float(r.get("SpecialRegularEdgesScanned")) for r in xs),
            }
        )
    return out


def collect_run(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = read_summary_csv(root / "summary.csv")
    run_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    build_rows: list[dict[str, Any]] = []
    cuts = [0.25, 0.50, 0.75, 0.90]
    for srow in summary_rows:
        variant = srow.get("variant", "")
        result_dir = root / variant / "results"
        build = read_kv_csv(result_dir / "build_time.csv")
        lsearch = srow.get("lsearch", "")
        query_overall, by_special, by_nontrivial = summarize_query_details(result_dir, cuts, lsearch)
        row: dict[str, Any] = dict(srow)
        row.update({f"build_{k}": v for k, v in build.items()})
        row.update(query_overall)
        row["run_root"] = str(root)
        run_rows.append(row)

        brow = {"run_root": str(root), "variant": variant}
        brow.update(build)
        build_rows.append(brow)

        for b in by_special + by_nontrivial:
            b["run_root"] = str(root)
            b["variant"] = variant
            b["lsearch"] = lsearch
            bucket_rows.append(b)
    return run_rows, bucket_rows, build_rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(run_rows: list[dict[str, Any]], bucket_rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as f:
        f.write("# Special Block Build/Query A/B Summary\n\n")
        f.write("This summary is diagnostic unless the run root is an explicit x1/original or x1-restored full-quality filtered-search run.\n\n")
        f.write("## Runs\n\n")
        f.write("| run_root | variant | Lsearch | T | skip_trivial | index_ms | group_ms | cross_ms | special_meta_ms | special_intra_ms | special_inter_ms | avg_query_ms | p95_query_ms | recall | dist_calcs | visited | weighted_special | weighted_nontrivial |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in run_rows:
            f.write(
                f"| `{r.get('run_root','')}` | {r.get('variant','')} | "
                f"{r.get('lsearch','')} | {r.get('build_special_block_min_points','')} | {r.get('build_special_block_skip_trivial','')} | "
                f"{fmt(r.get('index_ms'))} | {fmt(r.get('group_ms'))} | {fmt(r.get('cross_ms'))} | "
                f"{fmt(r.get('build_special_block_metadata_time'))} | {fmt(r.get('build_special_edge_intra_build_time'))} | {fmt(r.get('build_special_edge_inter_build_time'))} | "
                f"{fmt(r.get('mean_query_ms'))} | {fmt(r.get('p95_query_ms'))} | {fmt(r.get('mean_recall'))} | "
                f"{fmt(r.get('mean_dist_calcs'))} | {fmt(r.get('mean_visited'))} | "
                f"{fmt(r.get('weighted_special_ratio'))} | {fmt(r.get('weighted_nontrivial_ratio'))} |\n"
            )
        f.write("\n## Coverage Buckets\n\n")
        f.write("| variant | Lsearch | bucket_type | bucket | rows | mean_ms | p95_ms | recall | dist_calcs | visited | weighted_special | weighted_nontrivial | special_edges | regular_edges |\n")
        f.write("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in bucket_rows:
            f.write(
                f"| {r.get('variant','')} | {r.get('lsearch','')} | {r.get('bucket_type','')} | {r.get('bucket','')} | "
                f"{r.get('query_rows','')} | {fmt(r.get('mean_query_ms'))} | {fmt(r.get('p95_query_ms'))} | "
                f"{fmt(r.get('mean_recall'))} | {fmt(r.get('mean_dist_calcs'))} | {fmt(r.get('mean_visited'))} | "
                f"{fmt(r.get('weighted_special_ratio'))} | {fmt(r.get('weighted_nontrivial_ratio'))} | "
                f"{fmt(r.get('mean_special_edges_scanned'))} | {fmt(r.get('mean_regular_edges_scanned'))} |\n"
            )


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return "" if value is None else str(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_roots", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    all_runs: list[dict[str, Any]] = []
    all_buckets: list[dict[str, Any]] = []
    all_builds: list[dict[str, Any]] = []
    for root in args.run_roots:
        runs, buckets, builds = collect_run(root)
        all_runs.extend(runs)
        all_buckets.extend(buckets)
        all_builds.extend(builds)

    write_csv(all_runs, args.out_dir / "special_block_run_summary.csv")
    write_csv(all_buckets, args.out_dir / "special_block_bucket_summary.csv")
    write_csv(all_builds, args.out_dir / "special_block_build_summary.csv")
    write_markdown(all_runs, all_buckets, args.out_dir / "special_block_ab_summary.md")
    print(args.out_dir / "special_block_ab_summary.md")


if __name__ == "__main__":
    main()
