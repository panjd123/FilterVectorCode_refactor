#!/usr/bin/env python3
"""Remove query rows whose result timing exceeds a configured threshold."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path


def read_query_ids(path: Path, time_column: str, threshold_ms: float) -> tuple[list[int], list[int]]:
    """Return kept and removed QueryIDs, removing a query if any row exceeds threshold."""
    by_query: dict[int, bool] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"QueryID", time_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            query_id = int(float(row["QueryID"]))
            elapsed_ms = float(row[time_column])
            by_query[query_id] = by_query.get(query_id, False) or elapsed_ms > threshold_ms
    removed = sorted(query_id for query_id, is_slow in by_query.items() if is_slow)
    kept = sorted(query_id for query_id, is_slow in by_query.items() if not is_slow)
    return kept, removed


def read_fvec_row(path: Path, query_id: int) -> bytes:
    with path.open("rb") as f:
        for row_id in range(query_id + 1):
            header = f.read(4)
            if len(header) != 4:
                raise ValueError(f"truncated fvecs dimension header in {path}")
            (dim,) = struct.unpack("<i", header)
            payload = f.read(dim * 4)
            if len(payload) != dim * 4:
                raise ValueError(f"truncated fvecs payload in {path}")
        return struct.pack("<i", dim) + payload


def read_bin_row(path: Path, query_id: int) -> tuple[int, bytes]:
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"bad bin header: {path}")
        num_rows, dim = struct.unpack("<II", header)
        if query_id < 0 or query_id >= num_rows:
            raise ValueError(f"query id {query_id} outside {path}")
        f.seek(8 + query_id * dim * 4)
        payload = f.read(dim * 4)
        if len(payload) != dim * 4:
            raise ValueError(f"short bin row in {path} at query {query_id}")
    return dim, payload


def filter_groundtruth(source_path: Path, output_path: Path, kept_ids: list[int], source_count: int) -> None:
    data = source_path.read_bytes()
    if source_count <= 0 or len(data) % source_count != 0:
        raise ValueError(f"cannot infer fixed groundtruth row size from {source_path}")
    row_bytes = len(data) // source_count
    with output_path.open("wb") as f:
        for query_id in kept_ids:
            start = query_id * row_bytes
            f.write(data[start : start + row_bytes])


def materialize(
    source_dir: Path,
    output_dir: Path,
    dataset: str,
    kept_ids: list[int],
    result_csv: Path,
    removed_ids: list[int],
    threshold_ms: float,
    overwrite: bool,
    groundtruth_path: Path | None,
    groundtruth_output: Path | None,
) -> dict[str, object]:
    paths = {
        "labels": output_dir / f"{dataset}_query_labels.txt",
        "fvecs": output_dir / f"{dataset}_query.fvecs",
        "bin": output_dir / f"{dataset}_query.bin",
        "profile": output_dir / f"profiled_minlen2_cov1k.csv",
        "removed": output_dir / "removed_query_ids.csv",
        "manifest": output_dir / "filter_manifest.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("output exists; use another output directory or --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = (source_dir / f"{dataset}_query_labels.txt").read_text(encoding="utf-8").splitlines()
    if len(labels) <= max(kept_ids + removed_ids, default=-1):
        raise ValueError("labels do not contain all result QueryIDs")
    profile_src = source_dir / "profiled_minlen2_cov1k.csv"
    with profile_src.open(newline="", encoding="utf-8") as f:
        profile_rows = list(csv.DictReader(f))
        profile_fields = list(f and (profile_rows[0].keys() if profile_rows else []))
    if len(profile_rows) != len(labels):
        raise ValueError(f"profile row count {len(profile_rows)} != labels {len(labels)}")

    bin_src = source_dir / f"{dataset}_query.bin"
    fvec_src = source_dir / f"{dataset}_query.fvecs"
    dims = {read_bin_row(bin_src, query_id)[0] for query_id in kept_ids[:1]}
    dim = next(iter(dims), read_bin_row(bin_src, 0)[0])
    with paths["labels"].open("w", encoding="utf-8") as f_labels, paths["bin"].open("wb") as f_bin, paths["fvecs"].open("wb") as f_fvec:
        f_bin.write(struct.pack("<II", len(kept_ids), dim))
        for query_id in kept_ids:
            f_labels.write(labels[query_id] + "\n")
            row_dim, bin_payload = read_bin_row(bin_src, query_id)
            if row_dim != dim:
                raise ValueError("inconsistent bin dimension")
            f_bin.write(bin_payload)
            f_fvec.write(read_fvec_row(fvec_src, query_id))

    with paths["profile"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=profile_fields)
        writer.writeheader()
        for query_id in kept_ids:
            row = dict(profile_rows[query_id])
            if "QueryID" in row:
                row["QueryID"] = str(len([x for x in kept_ids if x < query_id]))
            writer.writerow(row)

    with paths["removed"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_query_id"])
        writer.writerows([[query_id] for query_id in removed_ids])

    manifest = {
        "dataset": dataset,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "result_csv": str(result_csv),
        "time_column": "Time_ms",
        "threshold_ms": threshold_ms,
        "source_queries": len(labels),
        "kept_queries": len(kept_ids),
        "removed_queries": len(removed_ids),
        "files": {key: str(value) for key, value in paths.items()},
    }

    if groundtruth_path is not None:
        if groundtruth_output is None:
            raise ValueError("groundtruth_output is required when groundtruth_path is set")
        groundtruth_output.parent.mkdir(parents=True, exist_ok=True)
        filter_groundtruth(groundtruth_path, groundtruth_output, kept_ids, len(labels))
        manifest["groundtruth_output"] = str(groundtruth_output)
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--result-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--threshold-ms", type=float, default=5000.0)
    parser.add_argument("--groundtruth", type=Path)
    parser.add_argument("--groundtruth-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    kept, removed = read_query_ids(args.result_csv, "Time_ms", args.threshold_ms)
    manifest = materialize(
        args.source_dir,
        args.output_dir,
        args.dataset,
        kept,
        args.result_csv,
        removed,
        args.threshold_ms,
        args.overwrite,
        args.groundtruth,
        args.groundtruth_output,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
