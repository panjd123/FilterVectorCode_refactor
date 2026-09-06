#!/usr/bin/env python3
"""Summarize independent rebuild determinism and robust sel_50 search points."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


BUILD_ROOTS = (
    "fresh_current_source_build_t1_2000_t2_25000_20260906",
    "fresh_current_source_build_t1_2000_t2_25000_repeat2_20260906",
    "fresh_current_source_build_t1_2000_t2_25000_repeat3_20260906",
)
QUERY_LABELS = ("rebuilt_bundle", "rebuilt_bundle_repeat2", "rebuilt_bundle_repeat3")
L_VALUES = (500, 550, 600, 650, 700)
QUALITY_THRESHOLD = 0.85
BUILD_FIELDS = (
    "rebuild", "builder_sha256", "builder_sha256_source",
    "build_time_ms", "frozen_build_time_ms",
    "fresh_over_frozen", "special_blocks_sha256", "special_trie_sha256",
    "regular_edges_sha256", "special_edges_sha256", "special_edge_count",
    "intra_edge_count", "inter_edge_count", "loaded_memory_allocated_bytes",
)
QUERY_FIELDS = (
    "rebuild", "lsearch", "recall", "median_ms", "min_ms", "max_ms",
    "cv", "measured_repeats", "passes_recall_0p85", "robust_selected",
    "search_binary_sha256", "raw_run_dir",
)
MAIN_QUERY_FIELDS = (
    "rebuild", "workload", "lsearch", "recall_threshold",
    "frozen_recall", "fresh_recall", "recall_delta",
    "frozen_median_ms", "fresh_median_ms", "fresh_over_frozen",
    "cv", "measured_repeats", "passes_recall_threshold",
    "search_binary_sha256", "raw_run_dir",
)
BACKEND_FIELDS = (
    "large_intra_backend", "build_samples", "total_builder_ms",
    "intra_edge_build_ms", "inter_edge_build_ms",
    "total_speedup_vs_cpu_vamana", "raw_build_root",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_meta(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarize(
    runs: Path, frozen_build_ms: float, paper_results: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    build_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    main_query_rows: list[dict[str, object]] = []
    search_hashes: set[str] = set()
    frozen_tuned = {
        row["workload"]: row for row in read_rows(paper_results)
        if row["comparison"] == "internal" and row["method"] == "Tuned Multi-level"
    }
    for index, (build_name, query_label) in enumerate(zip(BUILD_ROOTS, QUERY_LABELS), 1):
        build_root = runs / build_name
        case = build_root / "t1_2000_t2_25000"
        manifest = json.loads((build_root / "manifest.json").read_text())
        record = manifest["runs"][0]
        if record.get("status") != "complete" or record.get("reused_existing"):
            raise RuntimeError(f"rebuild {index} is not a fresh complete build")
        meta = parse_meta(case / "block_index/meta")
        if (meta.get("special_block_count"), meta.get("special_block_upper_count")) != ("111", "8"):
            raise RuntimeError(f"unexpected block topology in rebuild {index}")
        builder_hash = record.get("source_provenance", {}).get("build_binary_sha256")
        builder_hash_source = "manifest_snapshot"
        if not builder_hash:
            # The three historical fresh runs predate builder snapshotting; bind
            # them to the hash recorded alongside this audited experiment.
            builder_hash = "6ff471a86b94387a52a5ae4c0cf6a228fdf57ff52ba5fc7d7e77467aced50692"
            builder_hash_source = "historical_audit_record"
        build_ms = float(meta["build_time(ms)"])
        build_rows.append({
            "rebuild": index, "builder_sha256": builder_hash,
            "builder_sha256_source": builder_hash_source,
            "build_time_ms": f"{build_ms:.6f}",
            "frozen_build_time_ms": f"{frozen_build_ms:.6f}",
            "fresh_over_frozen": f"{build_ms / frozen_build_ms:.6f}",
            "special_blocks_sha256": sha256(case / "block_index/special_blocks.bin"),
            "special_trie_sha256": sha256(case / "block_index/special_block_trie.bin"),
            "regular_edges_sha256": sha256(case / "block_index/special_trie_regular_edges.bin"),
            "special_edges_sha256": sha256(case / "block_index/special_edges.bin"),
            "special_edge_count": meta["special_edge_count"],
            "intra_edge_count": meta["special_edge_intra_count"],
            "inter_edge_count": meta["special_edge_inter_count"],
            "loaded_memory_allocated_bytes": meta["loaded_memory_allocated_bytes"],
        })
        query_root = runs / f"fresh_current_source_{query_label}_sel50_robust_l_20260906"
        query_manifest = json.loads((query_root / "manifest.json").read_text())
        query_record = query_manifest["runs"][0]
        search_hashes.add(query_record["search_binary_sha256"])
        details_path = query_root / "multi_t1_2000_t2_25k/sel_50/search_time_details.csv"
        with details_path.open(newline="", encoding="utf-8") as stream:
            details = list(csv.DictReader(stream))
        per_l: dict[int, dict[str, object]] = {}
        for budget in L_VALUES:
            selected = [row for row in details if int(row["Lsearch"]) == budget]
            warm = [float(row["Time_ms"]) for row in selected if int(row["Repeat"]) > 0]
            recall = statistics.fmean(float(row["Avg_Recall"]) for row in selected)
            mean = statistics.fmean(warm)
            per_l[budget] = {
                "rebuild": index, "lsearch": budget, "recall": f"{recall:.6f}",
                "median_ms": f"{statistics.median(warm):.6f}",
                "min_ms": f"{min(warm):.6f}", "max_ms": f"{max(warm):.6f}",
                "cv": f"{statistics.stdev(warm) / mean:.6f}",
                "measured_repeats": len(warm),
                "passes_recall_0p85": int(recall >= QUALITY_THRESHOLD),
                "robust_selected": 0,
                "search_binary_sha256": query_record["search_binary_sha256"],
                "raw_run_dir": str(details_path.parent),
            }
        query_rows.extend(per_l.values())
        if index <= 2:
            for suffix in ("0p5_repeat21", "1", "10", "25", "50", "75"):
                workload = "sel_0p5" if suffix.startswith("0p5") else f"sel_{suffix}"
                main_root = runs / f"fresh_current_source_{query_label}_sel{suffix}_20260906"
                main_manifest = json.loads((main_root / "manifest.json").read_text())
                main_record = main_manifest["runs"][0]
                search_hashes.add(main_record["search_binary_sha256"])
                detail_path = main_root / "multi_t1_2000_t2_25k" / workload / "search_time_details.csv"
                details = read_rows(detail_path)
                budgets = {int(row["Lsearch"]) for row in details}
                if len(budgets) != 1:
                    raise RuntimeError(f"expected one operating point in {detail_path}")
                warm = [float(row["Time_ms"]) for row in details if int(row["Repeat"]) > 0]
                recall = statistics.fmean(float(row["Avg_Recall"]) for row in details)
                mean = statistics.fmean(warm)
                frozen = frozen_tuned[workload]
                frozen_recall = float(frozen["recall"])
                frozen_ms = float(frozen["batch_median_ms"])
                threshold = float(frozen["recall_threshold"])
                median = statistics.median(warm)
                main_query_rows.append({
                    "rebuild": index, "workload": workload,
                    "lsearch": next(iter(budgets)),
                    "recall_threshold": f"{threshold:.6f}",
                    "frozen_recall": f"{frozen_recall:.6f}",
                    "fresh_recall": f"{recall:.6f}",
                    "recall_delta": f"{recall - frozen_recall:.6f}",
                    "frozen_median_ms": f"{frozen_ms:.6f}",
                    "fresh_median_ms": f"{median:.6f}",
                    "fresh_over_frozen": f"{median / frozen_ms:.6f}",
                    "cv": f"{statistics.stdev(warm) / mean:.6f}",
                    "measured_repeats": len(warm),
                    "passes_recall_threshold": int(recall >= threshold),
                    "search_binary_sha256": main_record["search_binary_sha256"],
                    "raw_run_dir": str(detail_path.parent),
                })
    if len({row["builder_sha256"] for row in build_rows}) != 1:
        raise RuntimeError("mixed builder binaries")
    if len(search_hashes) != 1:
        raise RuntimeError("mixed search binaries")
    robust_l = min(
        budget for budget in L_VALUES
        if all(float(row["recall"]) >= QUALITY_THRESHOLD
               for row in query_rows if int(row["lsearch"]) == budget)
    )
    for row in query_rows:
        row["robust_selected"] = int(int(row["lsearch"]) == robust_l)
    return build_rows, query_rows, main_query_rows


def summarize_backends(runs: Path, gpu_builds: list[dict[str, object]]) -> list[dict[str, object]]:
    gpu_total = statistics.median(float(row["build_time_ms"]) for row in gpu_builds)
    gpu_intra = statistics.median(
        float(parse_meta(
            runs / BUILD_ROOTS[index] / "t1_2000_t2_25000/block_index/meta"
        )["special_edge_intra_build_time(ms)"])
        for index in range(len(BUILD_ROOTS))
    )
    gpu_inter = statistics.median(
        float(parse_meta(
            runs / BUILD_ROOTS[index] / "t1_2000_t2_25000/block_index/meta"
        )["special_edge_inter_build_time(ms)"])
        for index in range(len(BUILD_ROOTS))
    )
    cpu_root = runs / "fresh_current_source_build_t1_2000_t2_25000_cpu_large_20260906"
    cpu_meta = parse_meta(cpu_root / "t1_2000_t2_25000/block_index/meta")
    cpu_total = float(cpu_meta["build_time(ms)"])
    return [
        {
            "large_intra_backend": "GPU FastGrnnd", "build_samples": 3,
            "total_builder_ms": f"{gpu_total:.6f}",
            "intra_edge_build_ms": f"{gpu_intra:.6f}",
            "inter_edge_build_ms": f"{gpu_inter:.6f}",
            "total_speedup_vs_cpu_vamana": f"{cpu_total / gpu_total:.6f}",
            "raw_build_root": ";".join(str(runs / root) for root in BUILD_ROOTS),
        },
        {
            "large_intra_backend": "CPU Vamana", "build_samples": 1,
            "total_builder_ms": f"{cpu_total:.6f}",
            "intra_edge_build_ms": cpu_meta["special_edge_intra_build_time(ms)"],
            "inter_edge_build_ms": cpu_meta["special_edge_inter_build_time(ms)"],
            "total_speedup_vs_cpu_vamana": "1.000000",
            "raw_build_root": str(cpu_root),
        },
    ]


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=here.parents[1] / "runs")
    parser.add_argument("--output-dir", type=Path, default=here / "results_summary")
    parser.add_argument("--frozen-build-ms", type=float, default=58760.0)
    parser.add_argument("--paper-results", type=Path,
                        default=here / "results_summary/paper_results.csv")
    args = parser.parse_args()
    builds, queries, main_queries = summarize(
        args.runs, args.frozen_build_ms, args.paper_results
    )
    write_csv(args.output_dir / "current_source_rebuilds.csv", BUILD_FIELDS, builds)
    write_csv(args.output_dir / "current_source_rebuild_sel50_l_sweep.csv", QUERY_FIELDS, queries)
    write_csv(args.output_dir / "current_source_rebuild_query_regression.csv",
              MAIN_QUERY_FIELDS, main_queries)
    write_csv(args.output_dir / "current_source_build_backend.csv", BACKEND_FIELDS,
              summarize_backends(args.runs, builds))
    build_times = [float(row["build_time_ms"]) for row in builds]
    robust = [row for row in queries if row["robust_selected"]]
    print(
        f"wrote {len(builds)} rebuilds, {len(main_queries)} main query points, "
        f"and {len(queries)} sel_50 L-sweep points; "
        f"build median={statistics.median(build_times):.3f} ms; "
        f"robust L={robust[0]['lsearch']}; Recall range="
        f"[{min(float(r['recall']) for r in robust):.4f}, "
        f"{max(float(r['recall']) for r in robust):.4f}]"
    )


if __name__ == "__main__":
    main()
