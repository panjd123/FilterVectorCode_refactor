#!/usr/bin/env python3
"""Select query tasks from UNG result metrics and materialize a new query set.

The script scans per-query timing CSV files produced by multiple UNG variants,
keeps query/Lsearch points that satisfy metric thresholds, chooses one
representative Lsearch per source query, and optionally writes a merged query
task directory with bin/fvecs/labels copied from the original query tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import struct
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_METHODS = (
    "UNG",
    "gpu_bruteforce_els_ung",
    "gpu_bruteforce_els_special_blocks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select queries where special_blocks has a visible core-search "
            "advantage over the original UNG, then build a new query task."
        )
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", default="Amazon")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-task", action="append", default=None)
    parser.add_argument("--methods", nargs=3, default=list(DEFAULT_METHODS))
    parser.add_argument("--selection-csv", type=Path, default=None)
    parser.add_argument(
        "--selection-mode",
        choices=("threshold", "composite_special_els", "time_gap_topk"),
        default="threshold",
        help=(
            "threshold applies one set of metric thresholds. "
            "composite_special_els builds a mixed set: special-core advantage, "
            "ELS-long/improved, then special recall>=0.8 fill. "
            "time_gap_topk selects queries where special_blocks Time_ms is "
            "much smaller than gpu_bruteforce_els_ung Time_ms."
        ),
    )
    parser.add_argument("--min-recall", type=float, default=0.9)
    parser.add_argument("--min-ung-core-ms", type=float, default=10.0)
    parser.add_argument("--min-ung-core-share", type=float, default=0.30)
    parser.add_argument("--min-core-speedup", type=float, default=2.0)
    parser.add_argument("--min-total-speedup", type=float, default=1.5)
    parser.add_argument(
        "--min-special-core-share",
        type=float,
        default=0.0,
        help="Optional lower bound for special_blocks core/total time share.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Keep only the top N selected queries after ranking; 0 keeps all.",
    )
    parser.add_argument("--target-special-core", type=int, default=300)
    parser.add_argument("--target-els", type=int, default=500)
    parser.add_argument("--target-total", type=int, default=1000)
    parser.add_argument(
        "--special-core-min-recall",
        type=float,
        default=0.0,
        help="Recall lower bound for the first composite special-core bucket.",
    )
    parser.add_argument("--els-min-recall", type=float, default=0.8)
    parser.add_argument("--els-min-ung-ms", type=float, default=10.0)
    parser.add_argument("--els-min-speedup", type=float, default=10.0)
    parser.add_argument(
        "--els-min-special-total-speedup",
        type=float,
        default=0.0,
        help=(
            "Optional lower bound for UNG_Time/special_Time in the ELS bucket; "
            "use 0.95 to allow special to be at most about 5% slower than UNG."
        ),
    )
    parser.add_argument("--fill-min-recall", type=float, default=0.8)
    parser.add_argument("--fill-min-ung-core-ms", type=float, default=0.0)
    parser.add_argument("--fill-min-ung-core-share", type=float, default=0.0)
    parser.add_argument("--fill-min-core-speedup", type=float, default=2.0)
    parser.add_argument("--fill-min-total-speedup", type=float, default=1.0)
    parser.add_argument(
        "--special-core-compare",
        choices=("ung", "gpu"),
        default="ung",
        help="Compare special_blocks core speed against original UNG or gpu_bruteforce_els_ung in composite mode.",
    )
    parser.add_argument(
        "--els-core-compare",
        choices=("special_vs_ung", "gpu_vs_ung"),
        default="special_vs_ung",
        help="Core non-regression check used for the ELS bucket.",
    )
    parser.add_argument(
        "--els-min-core-speedup",
        type=float,
        default=0.0,
        help="Optional lower bound for the ELS bucket core speedup selected by --els-core-compare; use 0.95 to allow ~5% slower.",
    )
    parser.add_argument("--time-gap-min-special-max-recall", type=float, default=0.6)
    parser.add_argument("--time-gap-min-speedup", type=float, default=1.0)
    parser.add_argument("--time-gap-min-delta-ms", type=float, default=0.0)
    parser.add_argument(
        "--time-gap-rank-by",
        choices=("delta", "speedup", "score"),
        default="delta",
        help="Ranking for time_gap_topk after one best Lsearch is chosen per query.",
    )
    parser.add_argument(
        "--no-write-query-files",
        action="store_true",
        help="Only write selection CSV/manifest, not query bin/fvecs/labels.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def to_float(row: dict[str, str], name: str) -> float:
    try:
        return float(row.get(name, "") or 0)
    except ValueError:
        return 0.0


def to_int(row: dict[str, str], name: str) -> int:
    try:
        return int(float(row.get(name, "") or 0))
    except ValueError:
        return 0


def discover_source_tasks(results_root: Path, method: str) -> list[str]:
    method_root = results_root / method
    paths = sorted(method_root.glob("*/results/query_details_repeat1.csv"))
    return [p.parent.parent.name for p in paths]


def load_result_rows(results_root: Path, methods: list[str], source_tasks: list[str]):
    rows: dict[str, dict[tuple[str, int, int], dict[str, str]]] = {}
    for method in methods:
        rows[method] = {}
        for task in source_tasks:
            path = results_root / method / task / "results" / "query_details_repeat1.csv"
            if not path.exists():
                raise FileNotFoundError(f"missing result CSV: {path}")
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (task, to_int(row, "QueryID"), to_int(row, "Lsearch"))
                    rows[method][key] = row
    return rows


def compute_candidate(
    key: tuple[str, int, int],
    method_rows: dict[str, dict[tuple[str, int, int], dict[str, str]]],
    methods: list[str],
) -> dict[str, float | int | str] | None:
    ung = method_rows[methods[0]][key]
    gpu = method_rows[methods[1]][key]
    special = method_rows[methods[2]][key]

    ung_time = to_float(ung, "Time_ms")
    gpu_time = to_float(gpu, "Time_ms")
    special_time = to_float(special, "Time_ms")
    ung_core = to_float(ung, "core_search_time_ms")
    gpu_core = to_float(gpu, "core_search_time_ms")
    special_core = to_float(special, "core_search_time_ms")
    if ung_time <= 0 or gpu_time <= 0 or special_time <= 0 or special_core <= 0:
        return None

    ung_recall = to_float(ung, "Recall")
    gpu_recall = to_float(gpu, "Recall")
    special_recall = to_float(special, "Recall")
    min_recall = min(ung_recall, gpu_recall, special_recall)

    task, query_id, lsearch = key
    candidate: dict[str, float | int | str] = {
        "source_query_task": task,
        "query_id": query_id,
        "Lsearch": lsearch,
        "UNG_Time_ms": ung_time,
        "UNG_MinSupersetT_ms": to_float(ung, "MinSupersetT_ms"),
        "UNG_core_search_time_ms": ung_core,
        "UNG_core_share": ung_core / ung_time,
        "UNG_Recall": ung_recall,
        "gpu_els_UNG_Time_ms": gpu_time,
        "gpu_els_UNG_MinSupersetT_ms": to_float(gpu, "MinSupersetT_ms"),
        "gpu_els_UNG_core_search_time_ms": gpu_core,
        "gpu_els_UNG_core_share": gpu_core / gpu_time,
        "gpu_els_UNG_Recall": gpu_recall,
        "special_blocks_Time_ms": special_time,
        "special_blocks_MinSupersetT_ms": to_float(special, "MinSupersetT_ms"),
        "special_blocks_core_search_time_ms": special_core,
        "special_blocks_core_share": special_core / special_time,
        "special_blocks_Recall": special_recall,
        "special_vs_UNG_core_speedup": ung_core / special_core,
        "special_vs_gpu_core_speedup": gpu_core / special_core,
        "special_vs_UNG_total_speedup": ung_time / special_time,
        "special_vs_gpu_total_speedup": gpu_time / special_time,
        "gpu_vs_UNG_core_speedup": ung_core / gpu_core if gpu_core > 0 else 0.0,
        "gpu_vs_UNG_total_speedup": ung_time / gpu_time,
        "gpu_vs_UNG_MinSuperset_speedup": (
            to_float(ung, "MinSupersetT_ms") / to_float(gpu, "MinSupersetT_ms")
            if to_float(gpu, "MinSupersetT_ms") > 0
            else 0.0
        ),
        "special_vs_UNG_MinSuperset_speedup": (
            to_float(ung, "MinSupersetT_ms") / to_float(special, "MinSupersetT_ms")
            if to_float(special, "MinSupersetT_ms") > 0
            else 0.0
        ),
        "min_recall": min_recall,
        "avg_recall": (ung_recall + gpu_recall + special_recall) / 3.0,
        "special_minus_UNG_recall": special_recall - ung_recall,
        "SpecialQueryRatio": to_float(special, "SpecialQueryRatio"),
        "SpecialQueryPoints": to_float(special, "SpecialQueryPoints"),
        "SpecialQueryBlockCount": to_float(special, "SpecialQueryBlockCount"),
    }
    candidate["display_score"] = (
        100.0 * float(candidate["min_recall"])
        + 10.0 * min(float(candidate["special_vs_UNG_total_speedup"]), 10.0)
        + 5.0 * float(candidate["UNG_core_share"])
        + min(float(candidate["special_vs_UNG_core_speedup"]), 10.0)
    )
    return candidate


def passes_thresholds(row: dict[str, float | int | str], args: argparse.Namespace) -> bool:
    return (
        float(row["min_recall"]) >= args.min_recall
        and float(row["UNG_core_search_time_ms"]) >= args.min_ung_core_ms
        and float(row["UNG_core_share"]) >= args.min_ung_core_share
        and float(row["special_blocks_core_share"]) >= args.min_special_core_share
        and float(row["special_vs_UNG_core_speedup"]) >= args.min_core_speedup
        and float(row["special_vs_UNG_total_speedup"]) >= args.min_total_speedup
    )


def ranking_key(row: dict[str, float | int | str]):
    return (
        float(row["min_recall"]),
        float(row["special_vs_UNG_total_speedup"]),
        float(row["UNG_core_share"]),
        float(row["special_vs_UNG_core_speedup"]),
        float(row["UNG_core_search_time_ms"]),
    )


def query_key(row: dict[str, float | int | str]) -> tuple[str, int]:
    return (str(row["source_query_task"]), int(row["query_id"]))


def common_candidates(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    methods = list(args.methods)
    source_tasks = args.source_task or discover_source_tasks(args.results_root, methods[0])
    method_rows = load_result_rows(args.results_root, methods, source_tasks)
    common_keys = set(method_rows[methods[0]])
    for method in methods[1:]:
        common_keys &= set(method_rows[method])

    candidates = []
    for key in common_keys:
        row = compute_candidate(key, method_rows, methods)
        if row is not None:
            candidates.append(row)
    return candidates


def choose_unique(
    rows: Iterable[dict[str, float | int | str]],
    count: int,
    used: set[tuple[str, int]],
    key_fn,
    category: str,
) -> list[dict[str, float | int | str]]:
    chosen = []
    for row in sorted(rows, key=key_fn, reverse=True):
        key = query_key(row)
        if key in used:
            continue
        out = dict(row)
        out["selected_category"] = category
        chosen.append(out)
        used.add(key)
        if count > 0 and len(chosen) >= count:
            break
    return chosen


def time_gap_rank_key(row: dict[str, float | int | str], rank_by: str):
    if rank_by == "speedup":
        return (
            float(row["special_vs_gpu_total_speedup"]),
            float(row["gpu_minus_special_Time_ms"]),
            float(row["special_max_recall"]),
        )
    if rank_by == "score":
        return (
            float(row["gpu_minus_special_Time_ms"]) * max(float(row["special_max_recall"]), 0.0),
            float(row["special_vs_gpu_total_speedup"]),
            float(row["gpu_minus_special_Time_ms"]),
        )
    return (
        float(row["gpu_minus_special_Time_ms"]),
        float(row["special_vs_gpu_total_speedup"]),
        float(row["special_max_recall"]),
    )


def select_time_gap_topk(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    candidates = common_candidates(args)
    special_max_recall_by_query: dict[tuple[str, int], float] = defaultdict(float)
    for row in candidates:
        key = query_key(row)
        special_max_recall_by_query[key] = max(
            special_max_recall_by_query[key], float(row["special_blocks_Recall"])
        )

    best_by_query: dict[tuple[str, int], dict[str, float | int | str]] = {}
    for row in candidates:
        key = query_key(row)
        special_max_recall = special_max_recall_by_query[key]
        gpu_time = float(row["gpu_els_UNG_Time_ms"])
        special_time = float(row["special_blocks_Time_ms"])
        if special_time <= 0:
            continue
        delta = gpu_time - special_time
        speedup = gpu_time / special_time
        if special_max_recall < args.time_gap_min_special_max_recall:
            continue
        if speedup < args.time_gap_min_speedup:
            continue
        if delta < args.time_gap_min_delta_ms:
            continue
        out = dict(row)
        out["selected_category"] = "SPECIAL_TIME_GAP_TOPK"
        out["special_max_recall"] = special_max_recall
        out["gpu_minus_special_Time_ms"] = delta
        out["time_gap_rank_score"] = time_gap_rank_key(out, args.time_gap_rank_by)[0]
        if key not in best_by_query or time_gap_rank_key(out, args.time_gap_rank_by) > time_gap_rank_key(
            best_by_query[key], args.time_gap_rank_by
        ):
            best_by_query[key] = out

    selected = sorted(
        best_by_query.values(),
        key=lambda row: time_gap_rank_key(row, args.time_gap_rank_by),
        reverse=True,
    )
    limit = args.max_queries or args.target_total
    if limit > 0:
        selected = selected[:limit]
    return selected


def select_composite_special_els(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    candidates = common_candidates(args)
    used: set[tuple[str, int]] = set()
    selected: list[dict[str, float | int | str]] = []

    special_core_speed_col = (
        "special_vs_gpu_core_speedup" if args.special_core_compare == "gpu" else "special_vs_UNG_core_speedup"
    )
    special_total_speed_col = (
        "special_vs_gpu_total_speedup" if args.special_core_compare == "gpu" else "special_vs_UNG_total_speedup"
    )
    special_core_share_col = (
        "gpu_els_UNG_core_share" if args.special_core_compare == "gpu" else "UNG_core_share"
    )
    special_core_ms_col = (
        "gpu_els_UNG_core_search_time_ms" if args.special_core_compare == "gpu" else "UNG_core_search_time_ms"
    )

    special_core = [
        row
        for row in candidates
        if float(row["min_recall"]) >= args.special_core_min_recall
        and float(row[special_core_ms_col]) >= args.min_ung_core_ms
        and float(row[special_core_share_col]) >= args.min_ung_core_share
        and float(row["special_blocks_core_share"]) >= args.min_special_core_share
        and float(row[special_core_speed_col]) >= args.min_core_speedup
        and float(row[special_total_speed_col]) >= args.min_total_speedup
    ]
    selected.extend(
        choose_unique(
            special_core,
            args.target_special_core,
            used,
            lambda row: (
                float(row["min_recall"]),
                float(row[special_total_speed_col]),
                float(row[special_core_share_col]),
                float(row[special_core_speed_col]),
                float(row[special_core_ms_col]),
            ),
            "SPECIAL_CORE_SHARE_ADVANTAGE",
        )
    )

    els_long = [
        row
        for row in candidates
        if float(row["min_recall"]) >= args.els_min_recall
        and float(row["UNG_MinSupersetT_ms"]) >= args.els_min_ung_ms
        and max(
            float(row["gpu_vs_UNG_MinSuperset_speedup"]),
            float(row["special_vs_UNG_MinSuperset_speedup"]),
        )
        >= args.els_min_speedup
        and float(row["special_vs_UNG_total_speedup"]) >= args.els_min_special_total_speedup
        and (
            args.els_min_core_speedup <= 0
            or float(
                row[
                    "gpu_vs_UNG_core_speedup"
                    if args.els_core_compare == "gpu_vs_ung"
                    else "special_vs_UNG_core_speedup"
                ]
            )
            >= args.els_min_core_speedup
        )
    ]
    selected.extend(
        choose_unique(
            els_long,
            args.target_els,
            used,
            lambda row: (
                float(row["UNG_MinSupersetT_ms"]),
                max(
                    float(row["gpu_vs_UNG_MinSuperset_speedup"]),
                    float(row["special_vs_UNG_MinSuperset_speedup"]),
                ),
                float(row["min_recall"]),
                float(row["special_vs_UNG_total_speedup"]),
            ),
            "ELS_LONG_IMPROVED_RECALL80",
        )
    )

    remaining = args.target_total - len(selected)
    if remaining > 0:
        fill = [
            row
            for row in candidates
            if float(row["min_recall"]) >= args.fill_min_recall
            and float(row["UNG_core_search_time_ms"]) >= args.fill_min_ung_core_ms
            and float(row["UNG_core_share"]) >= args.fill_min_ung_core_share
            and float(row["special_vs_UNG_core_speedup"]) >= args.fill_min_core_speedup
            and float(row["special_vs_UNG_total_speedup"]) >= args.fill_min_total_speedup
        ]
        selected.extend(
            choose_unique(
                fill,
                remaining,
                used,
                lambda row: (
                    float(row["min_recall"]),
                    float(row["special_vs_UNG_total_speedup"]),
                    float(row["special_vs_UNG_core_speedup"]),
                    float(row["UNG_core_share"]),
                    float(row["UNG_core_search_time_ms"]),
                ),
                "SPECIAL_CORE_RECALL80_FILL",
            )
        )

    remaining = args.target_total - len(selected)
    if remaining > 0:
        selected.extend(
            choose_unique(
                els_long,
                remaining,
                used,
                lambda row: (
                    float(row["UNG_MinSupersetT_ms"]),
                    max(
                        float(row["gpu_vs_UNG_MinSuperset_speedup"]),
                        float(row["special_vs_UNG_MinSuperset_speedup"]),
                    ),
                    float(row["min_recall"]),
                    float(row["special_vs_UNG_total_speedup"]),
                ),
                "ELS_LONG_EXTRA_RECALL80",
            )
        )

    if args.max_queries > 0:
        selected = selected[: args.max_queries]
    elif args.target_total > 0:
        selected = selected[: args.target_total]
    return selected


def select_from_results(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    if args.selection_mode == "composite_special_els":
        return select_composite_special_els(args)
    if args.selection_mode == "time_gap_topk":
        return select_time_gap_topk(args)

    methods = list(args.methods)
    source_tasks = args.source_task or discover_source_tasks(args.results_root, methods[0])
    method_rows = load_result_rows(args.results_root, methods, source_tasks)
    common_keys = set(method_rows[methods[0]])
    for method in methods[1:]:
        common_keys &= set(method_rows[method])

    best_by_query: dict[tuple[str, int], dict[str, float | int | str]] = {}
    for key in common_keys:
        row = compute_candidate(key, method_rows, methods)
        if row is None or not passes_thresholds(row, args):
            continue
        query_key = (str(row["source_query_task"]), int(row["query_id"]))
        if query_key not in best_by_query or ranking_key(row) > ranking_key(best_by_query[query_key]):
            best_by_query[query_key] = row

    selected = sorted(best_by_query.values(), key=ranking_key, reverse=True)
    if args.max_queries > 0:
        selected = selected[: args.max_queries]
    return selected


def select_from_csv(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    if args.selection_csv is None:
        raise ValueError("--selection-csv is required")
    with args.selection_csv.open(newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    rows = sorted(rows, key=lambda r: (
        float(r.get("min_recall", 0) or 0),
        float(r.get("special_vs_UNG_total_speedup", 0) or 0),
        float(r.get("UNG_core_share", 0) or 0),
        float(r.get("special_vs_UNG_core_speedup", 0) or 0),
    ), reverse=True)
    if args.max_queries > 0:
        rows = rows[: args.max_queries]
    return rows


def read_labels(path: Path) -> list[str]:
    with path.open() as f:
        return [line.rstrip("\n") for line in f]


def read_bin_rows(path: Path, row_indices: Iterable[int]):
    wanted = set(row_indices)
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"bad bin header: {path}")
        n, dim = struct.unpack("<II", header)
        record_bytes = dim * 4
        rows = {}
        for idx in sorted(wanted):
            if idx < 0 or idx >= n:
                raise IndexError(f"row {idx} out of range for {path} with n={n}")
            f.seek(8 + idx * record_bytes)
            payload = f.read(record_bytes)
            if len(payload) != record_bytes:
                raise ValueError(f"short read in {path} at row {idx}")
            rows[idx] = payload
    return dim, rows


def read_fvecs_rows(path: Path, row_indices: Iterable[int]):
    wanted = set(row_indices)
    rows = {}
    dim = None
    with path.open("rb") as f:
        row_no = 0
        while wanted - rows.keys():
            header = f.read(4)
            if not header:
                break
            if len(header) != 4:
                raise ValueError(f"truncated fvecs header in {path}")
            (cur_dim,) = struct.unpack("<I", header)
            payload = f.read(cur_dim * 4)
            if len(payload) != cur_dim * 4:
                raise ValueError(f"truncated fvecs payload in {path}")
            if dim is None:
                dim = cur_dim
            elif cur_dim != dim:
                raise ValueError(f"inconsistent fvecs dim in {path}")
            if row_no in wanted:
                rows[row_no] = payload
            row_no += 1
    missing = wanted - rows.keys()
    if missing:
        raise ValueError(f"{path} missing rows: {sorted(missing)[:10]}")
    if dim is None:
        raise ValueError(f"empty fvecs file: {path}")
    return dim, rows


def output_fieldnames(rows: list[dict[str, float | int | str]]) -> list[str]:
    preferred = [
        "new_query_id",
        "selected_category",
        "source_query_task",
        "query_id",
        "Lsearch",
        "display_score",
        "time_gap_rank_score",
        "gpu_minus_special_Time_ms",
        "special_max_recall",
        "UNG_Time_ms",
        "UNG_MinSupersetT_ms",
        "UNG_core_search_time_ms",
        "UNG_core_share",
        "UNG_Recall",
        "gpu_els_UNG_Time_ms",
        "gpu_els_UNG_MinSupersetT_ms",
        "gpu_els_UNG_core_search_time_ms",
        "gpu_els_UNG_core_share",
        "gpu_els_UNG_Recall",
        "special_blocks_Time_ms",
        "special_blocks_MinSupersetT_ms",
        "special_blocks_core_search_time_ms",
        "special_blocks_core_share",
        "special_blocks_Recall",
        "special_vs_UNG_core_speedup",
        "special_vs_gpu_core_speedup",
        "special_vs_UNG_total_speedup",
        "special_vs_gpu_total_speedup",
        "gpu_vs_UNG_core_speedup",
        "gpu_vs_UNG_total_speedup",
        "gpu_vs_UNG_MinSuperset_speedup",
        "special_vs_UNG_MinSuperset_speedup",
        "min_recall",
        "avg_recall",
        "special_minus_UNG_recall",
        "SpecialQueryRatio",
        "SpecialQueryPoints",
        "SpecialQueryBlockCount",
    ]
    seen = set()
    fields = []
    for name in preferred:
        if any(name in row for row in rows):
            fields.append(name)
            seen.add(name)
    for row in rows:
        for name in row:
            if name not in seen:
                fields.append(name)
                seen.add(name)
    return fields


def materialize_query_task(
    selected: list[dict[str, float | int | str]],
    args: argparse.Namespace,
) -> None:
    dataset = args.dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_out = args.output_dir / f"{dataset}_query_labels.txt"
    bin_out = args.output_dir / f"{dataset}_query.bin"
    fvecs_out = args.output_dir / f"{dataset}_query.fvecs"
    selected_out = args.output_dir / "selected_queries.csv"
    manifest_out = args.output_dir / "selection_manifest.json"
    sources_out = args.output_dir / "selected_sources.csv"

    outputs = [selected_out, manifest_out, sources_out]
    if not args.no_write_query_files:
        outputs.extend([labels_out, bin_out, fvecs_out])
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("output exists; use --overwrite: " + ", ".join(existing))

    by_task: dict[str, set[int]] = defaultdict(set)
    for row in selected:
        by_task[str(row["source_query_task"])].add(int(row["query_id"]))

    labels_by_task = {}
    bin_rows_by_task = {}
    fvecs_rows_by_task = {}
    output_dim = None
    if not args.no_write_query_files:
        for task, ids in by_task.items():
            task_dir = args.data_root / task
            labels_path = task_dir / f"{dataset}_query_labels.txt"
            bin_path = task_dir / f"{dataset}_query.bin"
            fvecs_path = task_dir / f"{dataset}_query.fvecs"
            if not labels_path.exists():
                raise FileNotFoundError(labels_path)
            if not bin_path.exists():
                raise FileNotFoundError(bin_path)
            if not fvecs_path.exists():
                raise FileNotFoundError(fvecs_path)
            labels = read_labels(labels_path)
            for idx in ids:
                if idx < 0 or idx >= len(labels):
                    raise IndexError(f"label row {idx} out of range for {labels_path}")
            labels_by_task[task] = labels
            bin_dim, bin_rows = read_bin_rows(bin_path, ids)
            fvecs_dim, fvecs_rows = read_fvecs_rows(fvecs_path, ids)
            if bin_dim != fvecs_dim:
                raise ValueError(f"dimension mismatch for {task}: bin={bin_dim}, fvecs={fvecs_dim}")
            if output_dim is None:
                output_dim = bin_dim
            elif output_dim != bin_dim:
                raise ValueError(f"dimension mismatch across tasks: {output_dim} != {bin_dim}")
            bin_rows_by_task[task] = bin_rows
            fvecs_rows_by_task[task] = fvecs_rows

    selected_with_ids = []
    for new_query_id, row in enumerate(selected):
        out_row = dict(row)
        out_row["new_query_id"] = new_query_id
        selected_with_ids.append(out_row)

    fields = output_fieldnames(selected_with_ids)
    tmp_selected = selected_out.with_suffix(selected_out.suffix + ".tmp")
    with tmp_selected.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected_with_ids)
    shutil.move(str(tmp_selected), str(selected_out))

    tmp_sources = sources_out.with_suffix(sources_out.suffix + ".tmp")
    with tmp_sources.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "new_query_id",
                "source_query_task",
                "source_query_id",
                "source_query_bin",
                "source_query_fvecs",
                "source_query_labels",
            ],
        )
        writer.writeheader()
        for row in selected_with_ids:
            task = str(row["source_query_task"])
            task_dir = args.data_root / task
            writer.writerow(
                {
                    "new_query_id": row["new_query_id"],
                    "source_query_task": task,
                    "source_query_id": row["query_id"],
                    "source_query_bin": str(task_dir / f"{dataset}_query.bin"),
                    "source_query_fvecs": str(task_dir / f"{dataset}_query.fvecs"),
                    "source_query_labels": str(task_dir / f"{dataset}_query_labels.txt"),
                }
            )
    shutil.move(str(tmp_sources), str(sources_out))

    if not args.no_write_query_files:
        if output_dim is None:
            raise ValueError("no selected rows to materialize")
        tmp_labels = labels_out.with_suffix(labels_out.suffix + ".tmp")
        tmp_bin = bin_out.with_suffix(bin_out.suffix + ".tmp")
        tmp_fvecs = fvecs_out.with_suffix(fvecs_out.suffix + ".tmp")
        with tmp_labels.open("w") as labels_f, tmp_bin.open("wb") as bin_f, tmp_fvecs.open("wb") as fvecs_f:
            bin_f.write(struct.pack("<II", len(selected_with_ids), output_dim))
            for row in selected_with_ids:
                task = str(row["source_query_task"])
                source_id = int(row["query_id"])
                labels_f.write(labels_by_task[task][source_id] + "\n")
                payload = bin_rows_by_task[task][source_id]
                bin_f.write(payload)
                fvecs_f.write(struct.pack("<I", output_dim))
                fvecs_f.write(fvecs_rows_by_task[task][source_id])
        shutil.move(str(tmp_labels), str(labels_out))
        shutil.move(str(tmp_bin), str(bin_out))
        shutil.move(str(tmp_fvecs), str(fvecs_out))

    manifest = {
        "dataset": dataset,
        "results_root": str(args.results_root),
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "selection_csv": str(args.selection_csv) if args.selection_csv else None,
        "selection_mode": args.selection_mode,
        "methods": list(args.methods),
        "thresholds": {
            "min_recall": args.min_recall,
            "min_ung_core_ms": args.min_ung_core_ms,
            "min_ung_core_share": args.min_ung_core_share,
            "min_core_speedup": args.min_core_speedup,
            "min_total_speedup": args.min_total_speedup,
            "min_special_core_share": args.min_special_core_share,
            "max_queries": args.max_queries,
            "target_special_core": args.target_special_core,
            "target_els": args.target_els,
            "target_total": args.target_total,
            "special_core_min_recall": args.special_core_min_recall,
            "els_min_recall": args.els_min_recall,
            "els_min_ung_ms": args.els_min_ung_ms,
            "els_min_speedup": args.els_min_speedup,
            "els_min_special_total_speedup": args.els_min_special_total_speedup,
            "fill_min_recall": args.fill_min_recall,
            "fill_min_ung_core_ms": args.fill_min_ung_core_ms,
            "fill_min_ung_core_share": args.fill_min_ung_core_share,
            "fill_min_core_speedup": args.fill_min_core_speedup,
            "fill_min_total_speedup": args.fill_min_total_speedup,
            "special_core_compare": args.special_core_compare,
            "els_core_compare": args.els_core_compare,
            "els_min_core_speedup": args.els_min_core_speedup,
            "time_gap_min_special_max_recall": args.time_gap_min_special_max_recall,
            "time_gap_min_speedup": args.time_gap_min_speedup,
            "time_gap_min_delta_ms": args.time_gap_min_delta_ms,
            "time_gap_rank_by": args.time_gap_rank_by,
        },
        "num_queries": len(selected_with_ids),
        "category_counts": {
            category: sum(1 for row in selected_with_ids if str(row.get("selected_category", "")) == category)
            for category in sorted({str(row.get("selected_category", "")) for row in selected_with_ids})
        },
        "source_counts": {
            task: sum(1 for row in selected_with_ids if str(row["source_query_task"]) == task)
            for task in sorted(by_task)
        },
    }
    tmp_manifest = manifest_out.with_suffix(manifest_out.suffix + ".tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    shutil.move(str(tmp_manifest), str(manifest_out))


def main() -> None:
    args = parse_args()
    selected = select_from_csv(args) if args.selection_csv else select_from_results(args)
    materialize_query_task(selected, args)
    print(f"selected {len(selected)} queries")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
