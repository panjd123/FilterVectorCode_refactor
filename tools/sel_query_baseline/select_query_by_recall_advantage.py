#!/usr/bin/env python3
"""Build a query task where special-block recall is at least the baseline recall."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path(__file__).with_name("select_query_config.json")


@dataclass(frozen=True)
class SelectedQuery:
    source_query_id: int
    first_lsearch: int
    special_first_recall: float
    baseline_first_recall: float
    special_last_lsearch: int
    special_last_recall: float
    selection_mode: str = "recall_advantage"
    special_time_ms: float | None = None
    curator_time_ms: float | None = None
    speedup_vs_curator: float | None = None
    curator_lsearch: int | None = None
    rank: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select queries whose first-Lsearch special recall is no worse than the baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"JSON config path. Defaults to {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the selection summary; do not write query task files.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    required = ("dataset", "input_task_dir", "results_root", "query_result_task")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"missing config keys: {', '.join(missing)}")
    return config


def to_int(value: str) -> int:
    return int(float(value))


def to_float(value: str) -> float:
    return float(value)


def detail_csv_path(results_root: Path, method: str, query_result_task: str) -> Path:
    return results_root / method / query_result_task / "results" / "query_details_repeat1.csv"


def read_recall_rows(path: Path) -> dict[int, dict[int, float]]:
    if not path.exists():
        raise FileNotFoundError(f"missing result CSV: {path}")
    rows: dict[int, dict[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"QueryID", "Lsearch", "Recall"}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            query_id = to_int(row["QueryID"])
            lsearch = to_int(row["Lsearch"])
            rows.setdefault(query_id, {})[lsearch] = to_float(row["Recall"])
    return rows


def result_csv_path(results_root: Path, method: str, query_result_task: str, filename: str) -> Path:
    return results_root / method / query_result_task / "results" / filename


def read_query_metric_rows(
    path: Path,
    lsearch_column: str,
    time_column: str,
) -> dict[tuple[int, int], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing result CSV: {path}")
    rows: dict[tuple[int, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"QueryID", lsearch_column, "Recall", time_column}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            key = (to_int(row["QueryID"]), to_int(row[lsearch_column]))
            rows[key] = row
    return rows


def repeat_to_count(selected: list[SelectedQuery], target_count: int) -> list[SelectedQuery]:
    if not selected or target_count <= 0:
        return selected
    if len(selected) >= target_count:
        return selected[:target_count]
    repeated: list[SelectedQuery] = []
    while len(repeated) < target_count:
        for query in selected:
            repeated.append(query)
            if len(repeated) == target_count:
                break
    return repeated


def fastest_recall_rows_by_query(
    rows: dict[tuple[int, int], dict[str, str]],
    time_column: str,
    min_recall: float,
) -> dict[int, tuple[int, dict[str, str]]]:
    best: dict[int, tuple[int, dict[str, str]]] = {}
    for (query_id, search_budget), row in rows.items():
        if to_float(row["Recall"]) < min_recall:
            continue
        current = best.get(query_id)
        if current is None or to_float(row[time_column]) < to_float(current[1][time_column]):
            best[query_id] = (search_budget, row)
    return best


def select_curator_time_advantage_queries(config: dict[str, Any]) -> tuple[list[SelectedQuery], Path, Path]:
    results_root = Path(config["results_root"])
    query_result_task = str(config["query_result_task"])
    special_method = str(config.get("special_method", "cpu_bruteforce_els_special_blocks"))
    curator_method = str(config.get("curator_method", "Curator"))
    curator_query_result_task = str(config.get("curator_query_result_task", query_result_task))
    special_result_file = str(config.get("special_result_file", "query_details_repeat1.csv"))
    curator_result_file = str(config.get("curator_result_file", "curator_results.csv"))
    special_lsearch_column = str(config.get("special_lsearch_column", "Lsearch"))
    curator_lsearch_column = str(config.get("curator_lsearch_column", "search_ef"))
    special_time_column = str(config.get("special_time_column", "Time_ms"))
    curator_time_column = str(config.get("curator_time_column", "Search_Time_ms"))
    min_recall = float(config.get("min_recall", 0.95))
    min_speedup = float(config.get("min_speedup", 2.0))
    min_delta_ms = float(config.get("min_delta_ms", 0.0))
    target_num_queries = int(config.get("target_num_queries", 1000))

    special_csv = result_csv_path(results_root, special_method, query_result_task, special_result_file)
    curator_csv = result_csv_path(results_root, curator_method, curator_query_result_task, curator_result_file)
    special_rows = read_query_metric_rows(special_csv, special_lsearch_column, special_time_column)
    curator_rows = read_query_metric_rows(curator_csv, curator_lsearch_column, curator_time_column)

    special_best = fastest_recall_rows_by_query(special_rows, special_time_column, min_recall)
    curator_best = fastest_recall_rows_by_query(curator_rows, curator_time_column, min_recall)

    selected: list[SelectedQuery] = []
    for query_id in sorted(set(special_best) & set(curator_best)):
        special_lsearch, special = special_best[query_id]
        curator_lsearch, curator = curator_best[query_id]
        special_recall = to_float(special["Recall"])
        special_time = to_float(special[special_time_column])
        curator_time = to_float(curator[curator_time_column])
        if special_time <= 0:
            continue
        speedup = curator_time / special_time
        if speedup < min_speedup or (curator_time - special_time) < min_delta_ms:
            continue
        selected.append(
            SelectedQuery(
                source_query_id=query_id,
                first_lsearch=special_lsearch,
                special_first_recall=special_recall,
                baseline_first_recall=to_float(curator["Recall"]),
                special_last_lsearch=curator_lsearch,
                special_last_recall=special_recall,
                selection_mode="curator_time_advantage",
                special_time_ms=special_time,
                curator_time_ms=curator_time,
                speedup_vs_curator=speedup,
                curator_lsearch=curator_lsearch,
            )
        )

    ranked = sorted(
        selected,
        key=lambda query: (-(query.speedup_vs_curator or 0.0), query.special_time_ms or 0.0, query.source_query_id),
    )
    ranked = [
        SelectedQuery(
            source_query_id=query.source_query_id,
            first_lsearch=query.first_lsearch,
            special_first_recall=query.special_first_recall,
            baseline_first_recall=query.baseline_first_recall,
            special_last_lsearch=query.special_last_lsearch,
            special_last_recall=query.special_last_recall,
            selection_mode=query.selection_mode,
            special_time_ms=query.special_time_ms,
            curator_time_ms=query.curator_time_ms,
            speedup_vs_curator=query.speedup_vs_curator,
            curator_lsearch=query.curator_lsearch,
            rank=rank,
        )
        for rank, query in enumerate(ranked, start=1)
    ]
    return repeat_to_count(ranked, target_num_queries), special_csv, curator_csv


def select_queries(config: dict[str, Any]) -> tuple[list[SelectedQuery], Path, Path]:
    results_root = Path(config["results_root"])
    query_result_task = str(config["query_result_task"])
    special_method = str(config.get("special_method", "cpu_bruteforce_els_special_blocks"))
    baseline_method = str(config.get("baseline_method", "favor"))
    min_last_special_recall = float(config.get("min_last_special_recall", 0.8))

    special_csv = detail_csv_path(results_root, special_method, query_result_task)
    baseline_csv = detail_csv_path(results_root, baseline_method, query_result_task)
    special_rows = read_recall_rows(special_csv)
    baseline_rows = read_recall_rows(baseline_csv)

    selected: list[SelectedQuery] = []
    for query_id in sorted(set(special_rows) & set(baseline_rows)):
        common_lsearches = sorted(set(special_rows[query_id]) & set(baseline_rows[query_id]))
        if not common_lsearches:
            continue
        first_lsearch = common_lsearches[0]
        special_first_recall = special_rows[query_id][first_lsearch]
        baseline_first_recall = baseline_rows[query_id][first_lsearch]
        if special_first_recall < baseline_first_recall:
            continue

        special_last_lsearch = max(special_rows[query_id])
        special_last_recall = special_rows[query_id][special_last_lsearch]
        if special_last_recall < min_last_special_recall:
            continue

        selected.append(
            SelectedQuery(
                source_query_id=query_id,
                first_lsearch=first_lsearch,
                special_first_recall=special_first_recall,
                baseline_first_recall=baseline_first_recall,
                special_last_lsearch=special_last_lsearch,
                special_last_recall=special_last_recall,
            )
        )

    return selected, special_csv, baseline_csv


def read_labels(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def read_selected_bin_rows(path: Path, row_indices: set[int]) -> tuple[int, dict[int, bytes]]:
    rows: dict[int, bytes] = {}
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"bad bin header: {path}")
        num_rows, dim = struct.unpack("<II", header)
        record_bytes = dim * 4
        missing = [idx for idx in row_indices if idx < 0 or idx >= num_rows]
        if missing:
            raise ValueError(f"{path} does not contain query ids: {missing[:10]}")
        for idx in sorted(row_indices):
            f.seek(8 + idx * record_bytes)
            payload = f.read(record_bytes)
            if len(payload) != record_bytes:
                raise ValueError(f"short read in {path} at row {idx}")
            rows[idx] = payload
    return dim, rows


def read_selected_fvec_rows(path: Path, row_indices: set[int]) -> tuple[int, dict[int, bytes]]:
    rows: dict[int, bytes] = {}
    dim = None
    wanted = set(row_indices)
    with path.open("rb") as f:
        row_no = 0
        while wanted:
            header = f.read(4)
            if not header:
                break
            if len(header) != 4:
                raise ValueError(f"truncated fvecs dimension header in {path}")
            (cur_dim,) = struct.unpack("<i", header)
            payload = f.read(cur_dim * 4)
            if len(payload) != cur_dim * 4:
                raise ValueError(f"truncated fvecs payload in {path}")
            if dim is None:
                dim = cur_dim
            elif cur_dim != dim:
                raise ValueError(f"inconsistent fvecs dimension in {path}: {cur_dim} != {dim}")
            if row_no in wanted:
                rows[row_no] = payload
                wanted.remove(row_no)
            row_no += 1
    if wanted:
        raise ValueError(f"{path} does not contain query ids: {sorted(wanted)[:10]}")
    return int(dim or 0), rows


def output_paths(output_dir: Path, dataset: str, overwrite: bool) -> dict[str, Path]:
    paths = {
        "labels": output_dir / f"{dataset}_query_labels.txt",
        "fvecs": output_dir / f"{dataset}_query.fvecs",
        "bin": output_dir / f"{dataset}_query.bin",
        "selected": output_dir / "selected_queries.csv",
        "manifest": output_dir / "selection_manifest.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("output already exists; set overwrite=true or use another output_task: " + ", ".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)
    return paths


def format_recall(value: float) -> str:
    return f"{value:g}"


def materialize_query_task(
    config: dict[str, Any],
    selected: list[SelectedQuery],
    special_csv: Path,
    baseline_csv: Path,
) -> dict[str, Any]:
    dataset = str(config["dataset"])
    input_task_dir = Path(config["input_task_dir"])
    output_dir = Path(config.get("output_dir") or (input_task_dir.parent / str(config.get("output_task", "query_selected_recall_advantage"))))
    overwrite = bool(config.get("overwrite", False))
    paths = output_paths(output_dir, dataset, overwrite)

    source_query_ids = {query.source_query_id for query in selected}
    labels = read_labels(input_task_dir / f"{dataset}_query_labels.txt")
    missing_labels = [idx for idx in source_query_ids if idx < 0 or idx >= len(labels)]
    if missing_labels:
        raise ValueError(f"{input_task_dir} labels do not contain query ids: {missing_labels[:10]}")

    bin_dim, bin_rows = read_selected_bin_rows(input_task_dir / f"{dataset}_query.bin", source_query_ids)
    fvec_dim, fvec_rows = read_selected_fvec_rows(input_task_dir / f"{dataset}_query.fvecs", source_query_ids)
    if bin_dim != fvec_dim:
        raise ValueError(f"{input_task_dir} bin/fvecs dimensions differ: {bin_dim} != {fvec_dim}")

    with paths["labels"].open("w", encoding="utf-8") as f:
        for query in selected:
            f.write(labels[query.source_query_id] + "\n")

    with paths["bin"].open("wb") as f:
        f.write(struct.pack("<II", len(selected), bin_dim))
        for query in selected:
            f.write(bin_rows[query.source_query_id])

    with paths["fvecs"].open("wb") as f:
        for query in selected:
            f.write(struct.pack("<i", fvec_dim))
            f.write(fvec_rows[query.source_query_id])

    with paths["selected"].open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "new_query_id",
            "source_query_id",
            "first_lsearch",
            "special_first_recall",
            "baseline_first_recall",
            "favor_first_recall",
            "special_last_lsearch",
            "special_last_recall",
            "selection_mode",
            "special_time_ms",
            "curator_time_ms",
            "speedup_vs_curator",
            "curator_lsearch",
            "rank",
            "input_task_dir",
            "special_csv",
            "baseline_csv",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for new_query_id, query in enumerate(selected):
            writer.writerow(
                {
                    "new_query_id": new_query_id,
                    "source_query_id": query.source_query_id,
                    "first_lsearch": query.first_lsearch,
                    "special_first_recall": format_recall(query.special_first_recall),
                    "baseline_first_recall": format_recall(query.baseline_first_recall),
                    "favor_first_recall": format_recall(query.baseline_first_recall),
                    "special_last_lsearch": query.special_last_lsearch,
                    "special_last_recall": format_recall(query.special_last_recall),
                    "selection_mode": query.selection_mode,
                    "special_time_ms": "" if query.special_time_ms is None else format_recall(query.special_time_ms),
                    "curator_time_ms": "" if query.curator_time_ms is None else format_recall(query.curator_time_ms),
                    "speedup_vs_curator": "" if query.speedup_vs_curator is None else format_recall(query.speedup_vs_curator),
                    "curator_lsearch": "" if query.curator_lsearch is None else query.curator_lsearch,
                    "rank": "" if query.rank is None else query.rank,
                    "input_task_dir": str(input_task_dir),
                    "special_csv": str(special_csv),
                    "baseline_csv": str(baseline_csv),
                }
            )

    manifest = {
        "dataset": dataset,
        "input_task_dir": str(input_task_dir),
        "results_root": str(config["results_root"]),
        "query_result_task": str(config["query_result_task"]),
        "special_method": str(config.get("special_method", "cpu_bruteforce_els_special_blocks")),
        "baseline_method": str(config.get("baseline_method", "favor")),
        "selection_mode": str(config.get("selection_mode", "recall_advantage")),
        "curator_method": str(config.get("curator_method", "Curator")),
        "min_last_special_recall": float(config.get("min_last_special_recall", 0.8)),
        "min_recall": float(config.get("min_recall", 0.95)),
        "min_speedup": float(config.get("min_speedup", 2.0)),
        "target_num_queries": int(config.get("target_num_queries", 0) or 0),
        "output_dir": str(output_dir),
        "num_queries": len(selected),
        "unique_source_queries": len({query.source_query_id for query in selected}),
        "files": {name: str(path) for name, path in paths.items()},
    }
    with paths["manifest"].open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def run_from_config(config_path: Path, dry_run: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    selection_mode = str(config.get("selection_mode", "recall_advantage"))
    if selection_mode == "curator_time_advantage":
        selected, special_csv, baseline_csv = select_curator_time_advantage_queries(config)
    elif selection_mode == "recall_advantage":
        selected, special_csv, baseline_csv = select_queries(config)
    else:
        raise ValueError(f"unsupported selection_mode: {selection_mode}")
    if not selected:
        raise RuntimeError("no query matched the configured selection rules")
    if dry_run:
        return {
            "dataset": config["dataset"],
            "input_task_dir": config["input_task_dir"],
            "num_queries": len(selected),
            "special_csv": str(special_csv),
            "baseline_csv": str(baseline_csv),
        }
    return materialize_query_task(config, selected, special_csv, baseline_csv)


def main() -> None:
    args = parse_args()
    manifest = run_from_config(args.config, dry_run=args.dry_run)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
