#!/usr/bin/env python3
"""Build a deduplicated query task from profiled coverage CSV files."""

import argparse
import csv
import json
import struct
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recursively scan profiled_*.csv files, select rows whose "
            "coverage_count is in a target range, deduplicate by query labels, "
            "and write a new query task directory."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Dataset/query root to scan, for example FilterVectorData/Amazon.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset name. Defaults to root directory name.")
    parser.add_argument("--min-coverage", type=int, default=2000)
    parser.add_argument("--max-coverage", type=int, default=15000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output query task directory. Defaults to root/query_coverage_MIN_MAX.",
    )
    parser.add_argument(
        "--profile-glob",
        default="**/profiled_*.csv",
        help="Glob, relative to root, for profile CSV files.",
    )
    parser.add_argument(
        "--vector-mode",
        choices=("same-dir", "nearest-parent", "none"),
        default="same-dir",
        help=(
            "How to find source query fvecs for selected rows. same-dir is safest; "
            "nearest-parent can be useful for nested profile directories if row order matches."
        ),
    )
    parser.add_argument(
        "--max-output",
        type=int,
        default=0,
        help="Optional cap on selected unique queries; 0 means no cap.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    return parser.parse_args()


def normalize_labels(label_text: str) -> tuple[int, ...]:
    labels = [int(x) for x in label_text.split()]
    return tuple(sorted(labels))


def labels_to_text(labels: tuple[int, ...]) -> str:
    return " ".join(str(x) for x in labels)


def find_vector_file(profile_path: Path, root: Path, dataset: str, mode: str) -> Path | None:
    if mode == "none":
        return None

    names = (f"{dataset}_query.fvecs", "query.fvecs")
    candidates = [profile_path.parent]
    if mode == "nearest-parent":
        candidates.extend(profile_path.parents)

    root = root.resolve()
    for directory in candidates:
        try:
            directory.resolve().relative_to(root)
        except ValueError:
            continue
        for name in names:
            path = directory / name
            if path.exists():
                return path
        if directory.resolve() == root:
            break
    return None


def read_matching_rows(args, dataset: str, output_dir: Path):
    selected = []
    seen_labels = set()
    stats = {
        "profile_files_scanned": 0,
        "rows_scanned": 0,
        "rows_in_range": 0,
        "duplicate_rows_skipped": 0,
        "bad_rows_skipped": 0,
    }

    output_resolved = output_dir.resolve()
    profile_paths = sorted(args.root.glob(args.profile_glob))
    for profile_path in profile_paths:
        try:
            profile_path.resolve().relative_to(output_resolved)
            continue
        except ValueError:
            pass

        stats["profile_files_scanned"] += 1
        vector_file = find_vector_file(profile_path, args.root, dataset, args.vector_mode)
        with profile_path.open(newline="") as f:
            reader = csv.DictReader(f)
            if "coverage_count" not in (reader.fieldnames or []) or "labels" not in (reader.fieldnames or []):
                continue

            for row_index, row in enumerate(reader):
                stats["rows_scanned"] += 1
                try:
                    coverage = int(row["coverage_count"])
                    labels = normalize_labels(row["labels"])
                except (KeyError, TypeError, ValueError):
                    stats["bad_rows_skipped"] += 1
                    continue

                if coverage < args.min_coverage or coverage > args.max_coverage:
                    continue
                stats["rows_in_range"] += 1

                if labels in seen_labels:
                    stats["duplicate_rows_skipped"] += 1
                    continue
                seen_labels.add(labels)

                selected.append(
                    {
                        "coverage_count": coverage,
                        "labels": labels,
                        "source_profile": profile_path,
                        "source_row": row_index,
                        "source_vector_file": vector_file,
                    }
                )
                if args.max_output and len(selected) >= args.max_output:
                    return selected, stats

    return selected, stats


def read_selected_fvec_rows(vector_file: Path, row_indices: set[int]):
    rows = {}
    dim = None
    row_no = 0
    with vector_file.open("rb") as f:
        while True:
            header = f.read(4)
            if not header:
                break
            if len(header) != 4:
                raise ValueError(f"truncated fvecs dimension header in {vector_file}")
            (cur_dim,) = struct.unpack("<i", header)
            if cur_dim <= 0:
                raise ValueError(f"invalid fvecs dimension {cur_dim} in {vector_file}")
            payload = f.read(cur_dim * 4)
            if len(payload) != cur_dim * 4:
                raise ValueError(f"truncated fvecs payload in {vector_file}")
            if dim is None:
                dim = cur_dim
            elif cur_dim != dim:
                raise ValueError(f"inconsistent fvecs dimension in {vector_file}: {cur_dim} != {dim}")
            if row_no in row_indices:
                rows[row_no] = payload
                if len(rows) == len(row_indices):
                    break
            row_no += 1
    missing = sorted(row_indices - rows.keys())
    if missing:
        raise ValueError(f"{vector_file} does not contain selected row indices: {missing[:10]}")
    return dim, rows


def write_outputs(selected, stats, args, dataset: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / f"{dataset}_query_labels.txt"
    profile_path = output_dir / f"profiled_coverage_{args.min_coverage}_{args.max_coverage}.csv"
    sources_path = output_dir / "selected_sources.csv"
    manifest_path = output_dir / "selection_manifest.json"
    fvecs_path = output_dir / f"{dataset}_query.fvecs"
    bin_path = output_dir / f"{dataset}_query.bin"

    outputs = [labels_path, profile_path, sources_path, manifest_path]
    if any(path.exists() for path in outputs) and not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        raise FileExistsError("output already exists; use --overwrite: " + ", ".join(existing))

    with labels_path.open("w") as f:
        for item in selected:
            f.write(labels_to_text(item["labels"]) + "\n")

    with profile_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["coverage_count", "labels"])
        writer.writeheader()
        for item in selected:
            writer.writerow(
                {
                    "coverage_count": item["coverage_count"],
                    "labels": labels_to_text(item["labels"]),
                }
            )

    with sources_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "new_query_id",
                "coverage_count",
                "labels",
                "source_profile",
                "source_row",
                "source_vector_file",
            ],
        )
        writer.writeheader()
        for new_query_id, item in enumerate(selected):
            writer.writerow(
                {
                    "new_query_id": new_query_id,
                    "coverage_count": item["coverage_count"],
                    "labels": labels_to_text(item["labels"]),
                    "source_profile": str(item["source_profile"]),
                    "source_row": item["source_row"],
                    "source_vector_file": str(item["source_vector_file"] or ""),
                }
            )

    vector_items = [item for item in selected if item["source_vector_file"] is not None]
    skipped_vector_rows = len(selected) - len(vector_items)
    wrote_fvecs = False
    if vector_items:
        vector_outputs = [fvecs_path, bin_path]
        existing_vector_outputs = [str(path) for path in vector_outputs if path.exists()]
        if existing_vector_outputs and not args.overwrite:
            raise FileExistsError("output already exists; use --overwrite: " + ", ".join(existing_vector_outputs))

        needed_by_file = defaultdict(set)
        for item in vector_items:
            needed_by_file[item["source_vector_file"]].add(item["source_row"])

        rows_by_file = {}
        output_dim = None
        for vector_file, row_indices in needed_by_file.items():
            dim, rows = read_selected_fvec_rows(vector_file, row_indices)
            if output_dim is None:
                output_dim = dim
            elif dim != output_dim:
                raise ValueError(f"cannot merge fvecs with different dimensions: {dim} != {output_dim}")
            rows_by_file[vector_file] = rows

        with fvecs_path.open("wb") as f:
            for item in selected:
                vector_file = item["source_vector_file"]
                if vector_file is None:
                    continue
                f.write(struct.pack("<i", output_dim))
                f.write(rows_by_file[vector_file][item["source_row"]])
        with bin_path.open("wb") as f:
            f.write(struct.pack("<II", len(vector_items), output_dim))
            for item in selected:
                vector_file = item["source_vector_file"]
                if vector_file is None:
                    continue
                f.write(rows_by_file[vector_file][item["source_row"]])
        wrote_fvecs = True

    manifest = {
        "dataset": dataset,
        "root": str(args.root),
        "output_dir": str(output_dir),
        "min_coverage": args.min_coverage,
        "max_coverage": args.max_coverage,
        "vector_mode": args.vector_mode,
        "num_unique_queries": len(selected),
        "wrote_fvecs": wrote_fvecs,
        "skipped_vector_rows": skipped_vector_rows,
        "stats": stats,
        "files": {
            "query_labels": str(labels_path),
            "profiled": str(profile_path),
            "sources": str(sources_path),
            "query_fvecs": str(fvecs_path) if wrote_fvecs else "",
            "query_bin": str(bin_path) if wrote_fvecs else "",
        },
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def main():
    args = parse_args()
    args.root = args.root.resolve()
    dataset = args.dataset or args.root.name
    output_dir = args.output_dir or (args.root / f"query_coverage_{args.min_coverage}_{args.max_coverage}")
    output_dir = output_dir.resolve()

    selected, stats = read_matching_rows(args, dataset, output_dir)
    if not selected:
        raise RuntimeError(
            f"no unique queries found with coverage_count in [{args.min_coverage}, {args.max_coverage}]"
        )

    manifest = write_outputs(selected, stats, args, dataset, output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
