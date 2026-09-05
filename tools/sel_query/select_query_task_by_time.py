#!/usr/bin/env python3
"""Select query rows where one search method is much faster than another.

The script is config-driven so dataset paths, methods, query tasks, and speedup
thresholds can be changed without editing code.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path(__file__).with_name("select_query_config.json")


@dataclass(frozen=True)
class Match:
    rule_name: str
    source_task: str
    source_query_id: int
    matched_lsearch: int
    slower_lsearch: int
    faster_method: str
    slower_method: str
    faster_time_ms: float
    slower_time_ms: float
    speedup: float
    faster_csv: Path
    slower_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a new query task from per-query timing advantages."
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
    required = ("dataset", "results_root", "query_data_root", "output_task", "selections")
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


def read_detail_rows(path: Path, time_column: str) -> dict[tuple[int, int], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing result CSV: {path}")
    rows: dict[tuple[int, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"QueryID", "Lsearch", time_column}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            key = (to_int(row["QueryID"]), to_int(row["Lsearch"]))
            rows[key] = row
    return rows


def result_task_name(
    rule: dict[str, Any], result_task_suffix: str, method_role: str | None = None
) -> str:
    if rule.get("query_result_task"):
        return str(rule["query_result_task"])
    suffix_key = f"{method_role}_result_task_suffix" if method_role else ""
    suffix = str(rule.get(suffix_key, rule.get("result_task_suffix", result_task_suffix)))
    return f"{rule['source_task']}{suffix}"


def aligned_lsearch_pairs(
    faster_rows: dict[tuple[int, int], dict[str, str]],
    slower_rows: dict[tuple[int, int], dict[str, str]],
    alignment: str,
) -> list[tuple[int, int]]:
    faster_lsearches = sorted({lsearch for _, lsearch in faster_rows})
    slower_lsearches = sorted({lsearch for _, lsearch in slower_rows})
    if alignment == "exact":
        return [(lsearch, lsearch) for lsearch in sorted(set(faster_lsearches) & set(slower_lsearches))]
    if alignment == "ordinal":
        return list(zip(faster_lsearches, slower_lsearches))
    raise ValueError(f"unsupported lsearch_alignment: {alignment}; expected exact or ordinal")


def choose_rule_matches(config: dict[str, Any]) -> list[Match]:
    results_root = Path(config["results_root"])
    result_task_suffix = str(config.get("result_task_suffix", ""))
    default_skip_first = bool(config.get("skip_first_lsearch", True))
    default_time_column = str(config.get("time_column", "Time_ms"))
    default_min_recall = float(config.get("min_recall", 0.0))
    matches: list[Match] = []

    for rule in config["selections"]:
        rule_name = str(rule["name"])
        source_task = str(rule["source_task"])
        faster_method = str(rule["faster_method"])
        slower_method = str(rule["slower_method"])
        faster_query_result_task = result_task_name(rule, result_task_suffix, "faster")
        slower_query_result_task = result_task_name(rule, result_task_suffix, "slower")
        # Methods may use different names for the same elapsed-time metric.
        # Keep `time_column` as the common-column shorthand, while allowing
        # each side of a comparison to override it independently.
        time_column = str(rule.get("time_column", default_time_column))
        faster_time_column = str(rule.get("faster_time_column", time_column))
        slower_time_column = str(rule.get("slower_time_column", time_column))
        lsearch_alignment = str(rule.get("lsearch_alignment", "exact"))
        min_speedup = float(rule.get("min_speedup", config.get("min_speedup", 2.0)))
        min_delta_ms = float(rule.get("min_delta_ms", config.get("min_delta_ms", 0.0)))
        min_recall = float(rule.get("min_recall", default_min_recall))
        max_queries = int(rule.get("max_queries", config.get("max_queries", 0)) or 0)
        skip_first_lsearch = bool(rule.get("skip_first_lsearch", default_skip_first))

        faster_csv = detail_csv_path(results_root, faster_method, faster_query_result_task)
        slower_csv = detail_csv_path(results_root, slower_method, slower_query_result_task)
        faster_rows = read_detail_rows(faster_csv, faster_time_column)
        slower_rows = read_detail_rows(slower_csv, slower_time_column)
        lsearch_pairs = aligned_lsearch_pairs(faster_rows, slower_rows, lsearch_alignment)
        if not lsearch_pairs:
            raise RuntimeError(f"no aligned Lsearch values for rule {rule_name}")
        if skip_first_lsearch:
            lsearch_pairs = lsearch_pairs[1:]

        best_by_query: dict[int, Match] = {}
        for faster_lsearch, slower_lsearch in lsearch_pairs:
            faster_query_ids = {
                query_id for query_id, lsearch in faster_rows if lsearch == faster_lsearch
            }
            slower_query_ids = {
                query_id for query_id, lsearch in slower_rows if lsearch == slower_lsearch
            }
            for query_id in sorted(faster_query_ids & slower_query_ids):
                faster = faster_rows[(query_id, faster_lsearch)]
                slower = slower_rows[(query_id, slower_lsearch)]
                faster_time = to_float(faster[faster_time_column])
                slower_time = to_float(slower[slower_time_column])
                if faster_time <= 0:
                    continue
                speedup = slower_time / faster_time
                if speedup < min_speedup or (slower_time - faster_time) < min_delta_ms:
                    continue
                if min_recall:
                    recalls = []
                    for row in (faster, slower):
                        if "Recall" in row and row["Recall"] != "":
                            recalls.append(to_float(row["Recall"]))
                    if recalls and min(recalls) < min_recall:
                        continue

                current = best_by_query.get(query_id)
                if current is not None and current.speedup >= speedup:
                    continue
                best_by_query[query_id] = Match(
                    rule_name=rule_name,
                    source_task=source_task,
                    source_query_id=query_id,
                    matched_lsearch=faster_lsearch,
                    slower_lsearch=slower_lsearch,
                    faster_method=faster_method,
                    slower_method=slower_method,
                    faster_time_ms=faster_time,
                    slower_time_ms=slower_time,
                    speedup=speedup,
                    faster_csv=faster_csv,
                    slower_csv=slower_csv,
                )

        rule_matches = sorted(best_by_query.values(), key=lambda item: (-item.speedup, item.source_query_id))
        if max_queries > 0:
            rule_matches = rule_matches[:max_queries]
        matches.extend(rule_matches)

    return matches


def dedupe_matches(matches: list[Match]) -> list[Match]:
    selected: dict[tuple[str, int], Match] = {}
    for match in matches:
        key = (match.source_task, match.source_query_id)
        current = selected.get(key)
        if current is None or match.speedup > current.speedup:
            selected[key] = match
    return [selected[key] for key in sorted(selected)]


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


def ensure_output_paths(output_dir: Path, dataset: str, overwrite: bool) -> dict[str, Path]:
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


def materialize_query_task(config: dict[str, Any], matches: list[Match]) -> dict[str, Any]:
    dataset = str(config["dataset"])
    data_root = Path(config["query_data_root"])
    output_task = str(config["output_task"])
    output_dir = Path(config.get("output_dir") or (data_root / dataset / output_task))
    overwrite = bool(config.get("overwrite", False))
    paths = ensure_output_paths(output_dir, dataset, overwrite)

    needed_by_task: dict[str, set[int]] = defaultdict(set)
    for match in matches:
        needed_by_task[match.source_task].add(match.source_query_id)

    labels_by_task: dict[str, list[str]] = {}
    bin_rows_by_task: dict[str, dict[int, bytes]] = {}
    fvec_rows_by_task: dict[str, dict[int, bytes]] = {}
    output_dim = None
    for source_task, query_ids in needed_by_task.items():
        source_dir = data_root / dataset / source_task
        labels = read_labels(source_dir / f"{dataset}_query_labels.txt")
        missing_labels = [idx for idx in query_ids if idx < 0 or idx >= len(labels)]
        if missing_labels:
            raise ValueError(f"{source_dir} labels do not contain query ids: {missing_labels[:10]}")
        labels_by_task[source_task] = labels

        bin_dim, bin_rows = read_selected_bin_rows(source_dir / f"{dataset}_query.bin", query_ids)
        fvec_dim, fvec_rows = read_selected_fvec_rows(source_dir / f"{dataset}_query.fvecs", query_ids)
        if bin_dim != fvec_dim:
            raise ValueError(f"{source_dir} bin/fvecs dimensions differ: {bin_dim} != {fvec_dim}")
        if output_dim is None:
            output_dim = bin_dim
        elif output_dim != bin_dim:
            raise ValueError(f"cannot merge query tasks with different dimensions: {output_dim} != {bin_dim}")
        bin_rows_by_task[source_task] = bin_rows
        fvec_rows_by_task[source_task] = fvec_rows

    with paths["labels"].open("w", encoding="utf-8") as f:
        for match in matches:
            f.write(labels_by_task[match.source_task][match.source_query_id] + "\n")

    with paths["bin"].open("wb") as f:
        f.write(struct.pack("<II", len(matches), int(output_dim or 0)))
        for match in matches:
            f.write(bin_rows_by_task[match.source_task][match.source_query_id])

    with paths["fvecs"].open("wb") as f:
        for match in matches:
            f.write(struct.pack("<i", int(output_dim or 0)))
            f.write(fvec_rows_by_task[match.source_task][match.source_query_id])

    with paths["selected"].open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "new_query_id",
            "rule_name",
            "source_task",
            "source_query_id",
            "matched_lsearch",
            "slower_lsearch",
            "faster_method",
            "slower_method",
            "faster_time_ms",
            "slower_time_ms",
            "speedup",
            "faster_csv",
            "slower_csv",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for new_query_id, match in enumerate(matches):
            row = match.__dict__.copy()
            row["new_query_id"] = new_query_id
            row["faster_csv"] = str(match.faster_csv)
            row["slower_csv"] = str(match.slower_csv)
            writer.writerow({name: row[name] for name in fieldnames})

    manifest = {
        "dataset": dataset,
        "results_root": str(config["results_root"]),
        "query_data_root": str(data_root),
        "output_dir": str(output_dir),
        "num_queries": len(matches),
        "source_tasks": sorted(needed_by_task),
        "files": {name: str(path) for name, path in paths.items()},
    }
    with paths["manifest"].open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def run_from_config(config_path: Path, dry_run: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    matches = dedupe_matches(choose_rule_matches(config))
    if not matches:
        raise RuntimeError("no query matched the configured timing rules")
    if dry_run:
        return {
            "dataset": config["dataset"],
            "num_queries": len(matches),
            "source_tasks": sorted({match.source_task for match in matches}),
        }
    return materialize_query_task(config, matches)


def main() -> None:
    args = parse_args()
    manifest = run_from_config(args.config, dry_run=args.dry_run)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
