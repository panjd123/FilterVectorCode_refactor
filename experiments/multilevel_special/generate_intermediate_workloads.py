#!/usr/bin/env python3
"""Build deterministic Amazon workloads at 5%, 30%, 60%, and 99%.

The first three workloads start from the nearest lower existing Amazon suite
and deterministically replace the fewest possible predicates by label 1. This
keeps the original query vectors and most of the original predicate diversity,
while moving the batch mean to the target without ANN latency or recall.

No real non-empty Amazon predicate exceeds label 1's 96.7017% coverage. The
99% workload is therefore an explicit batch-level mixture of label 1 and the
empty (unfiltered) predicate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
from pathlib import Path


NAMES = {
    "sel_5": "query_nested_avgsel5pct",
    "sel_30": "query_nested_avgsel30pct",
    "sel_60": "query_nested_avgsel60pct",
    "sel_99": "query_label1_empty_avgsel99pct",
}
TARGETS = (
    ("sel_5", 0.05, "query_minlen5_avgsel1pct"),
    ("sel_30", 0.30, "query_minlen2_avgsel25pct"),
    ("sel_60", 0.60, "query_minlen1_avgsel50pct"),
)
NUM_QUERIES = 1000
SEED = 20260917
ALGORITHM = "nested-label1-substitution-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_labels(text: str) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in text.replace(",", " ").split()}))


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        if sha256(target) != sha256(source):
            raise RuntimeError(f"existing vector file differs: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def read_profile(query_dir: Path) -> list[tuple[tuple[int, ...], int]]:
    label_path = query_dir / "Amazon_query_labels.txt"
    profiles = sorted(query_dir.glob("profiled_*.csv"))
    if not label_path.is_file() or len(profiles) != 1:
        raise FileNotFoundError(f"expected labels and one profile in {query_dir}")
    labels = [canonical_labels(line) for line in label_path.read_text().splitlines()]
    with profiles[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(labels) != len(rows):
        raise ValueError(f"label/profile row mismatch in {query_dir}")
    result = []
    for index, (label_set, row) in enumerate(zip(labels, rows)):
        profile_labels = canonical_labels(row["labels"])
        if profile_labels != label_set:
            raise ValueError(f"profile label mismatch in {query_dir}, row {index}")
        result.append((label_set, int(row["coverage_count"])))
    return result


def nested_substitute(source_rows: list[tuple[tuple[int, ...], int]],
                      label1_coverage: int, target_selectivity: float,
                      num_points: int, seed: int,
                      minimum_coverage: int) -> tuple[list[tuple[int, tuple[int, ...]]], list[int]]:
    """Replace a deterministic prefix by label 1 and choose the closest mean."""
    rows = [(coverage, labels) for labels, coverage in source_rows]
    mandatory = [index for index, (coverage, labels) in enumerate(rows)
                 if labels != (1,) and coverage < minimum_coverage]
    optional = [index for index, (coverage, labels) in enumerate(rows)
                if labels != (1,) and coverage >= minimum_coverage]
    random.Random(seed).shuffle(optional)
    order = mandatory + optional
    target_total = target_selectivity * num_points * len(rows)
    running_total = sum(coverage for coverage, _ in rows)
    best = (abs(running_total - target_total), 0)
    for used, index in enumerate(order, 1):
        running_total += label1_coverage - rows[index][0]
        candidate = (abs(running_total - target_total), used)
        if candidate < best:
            best = candidate
        if running_total > target_total and used > best[1]:
            break
    changed = order[:best[1]]
    for index in changed:
        rows[index] = (label1_coverage, (1,))
    return rows, changed


def write_workload(data_root: Path, name: str,
                   rows: list[tuple[int, tuple[int, ...]]],
                   num_points: int, vector_source: str, provenance: dict) -> dict:
    target = data_root / NAMES[name]
    target.mkdir(parents=True, exist_ok=True)
    source = data_root / vector_source
    for suffix in ("bin", "fvecs"):
        link_or_copy(source / f"Amazon_query.{suffix}", target / f"Amazon_query.{suffix}")
    labels_path = target / "Amazon_query_labels.txt"
    labels_path.write_text(
        "\n".join(",".join(map(str, labels)) for _, labels in rows) + "\n")
    profile_path = target / f"profiled_{NAMES[name].removeprefix('query_')}.csv"
    with profile_path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("coverage_count", "labels"))
        writer.writerows((coverage, " ".join(map(str, labels)))
                         for coverage, labels in rows)
    coverages = [coverage for coverage, _ in rows]
    manifest = {
        "schema_version": 1,
        "name": name,
        "query_dir": NAMES[name],
        "generation_algorithm": (ALGORITHM if name != "sel_99"
                                 else "label1-empty-even-mixture-v1"),
        "target_mean_selectivity": provenance["target"],
        "num_queries": len(rows),
        "num_base_points": num_points,
        "mean_selectivity": sum(coverages) / len(coverages) / num_points,
        "min_selectivity": min(coverages) / num_points,
        "max_selectivity": max(coverages) / num_points,
        "unique_predicates": len({labels for _, labels in rows}),
        "empty_predicates": sum(not labels for _, labels in rows),
        "vector_source_query_dir": vector_source,
        "predicate_source_query_dirs": provenance["predicate_source_query_dirs"],
        "selection_detail": provenance,
        "query_bin_sha256": sha256(target / "Amazon_query.bin"),
        "query_labels_sha256": sha256(labels_path),
        "profile_sha256": sha256(profile_path),
    }
    (target / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def compute_gt(tool: Path, data_root: Path, gt_root: Path, query_dir: str,
               threads: int, k: int, overwrite: bool) -> Path:
    output = gt_root / query_dir / "Amazon_gt_labels_containment.bin"
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = NUM_QUERIES * k * 8
    if output.exists() and not overwrite:
        if output.stat().st_size != expected_bytes:
            raise ValueError(f"bad existing GT size: {output}")
        return output
    command = [str(tool), "--data_type", "float", "--dist_fn", "L2",
               "--scenario", "containment", "--K", str(k),
               "--num_threads", str(threads),
               "--base_bin_file", str(data_root / "Amazon_base.bin"),
               "--base_label_file", str(data_root / "Amazon_base_labels.txt"),
               "--query_bin_file", str(data_root / query_dir / "Amazon_query.bin"),
               "--query_label_file", str(data_root / query_dir / "Amazon_query_labels.txt"),
               "--gt_file", str(output)]
    subprocess.run(command, check=True)
    if output.stat().st_size != expected_bytes:
        raise ValueError(f"computed GT has unexpected size: {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path("/home/graphdb/FilterVectorData/Amazon"))
    parser.add_argument("--gt-root", type=Path,
                        default=Path("/home/graphdb/FilterVectorResult/Amazon/GroundTruth"))
    parser.add_argument("--compute-groundtruth", type=Path)
    parser.add_argument("--threads", type=int, default=72)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--overwrite-groundtruth", action="store_true")
    args = parser.parse_args()

    num_points = sum(1 for _ in (args.data_root / "Amazon_base_labels.txt").open())
    label1_rows = read_profile(args.data_root / "query_minlen1_label1_sel967pct")
    label1_coverage = label1_rows[0][1]
    if any(labels != (1,) or coverage != label1_coverage
           for labels, coverage in label1_rows):
        raise ValueError("label-1 anchor is not constant")

    manifests = {}
    for offset, (name, target, source_dir) in enumerate(TARGETS):
        source_rows = read_profile(args.data_root / source_dir)
        rows, changed = nested_substitute(
            source_rows, label1_coverage, target, num_points, SEED + offset,
            args.k)
        if any(0 < coverage < args.k for coverage, _ in rows):
            raise ValueError(f"{name} contains a non-empty predicate with fewer than K answers")
        manifests[name] = write_workload(
            args.data_root, name, rows, num_points, source_dir,
            {"target": target, "seed": SEED + offset,
             "changed_query_indices": changed,
             "predicate_source_query_dirs": [source_dir, "query_minlen1_label1_sel967pct"]})
    empty_count = round(NUM_QUERIES * (0.99 * num_points - label1_coverage) /
                        (num_points - label1_coverage))
    rows_99 = []
    previous = 0
    for index in range(NUM_QUERIES):
        current = ((index + 1) * empty_count) // NUM_QUERIES
        if current > previous:
            rows_99.append((num_points, ()))
        else:
            rows_99.append((label1_coverage, (1,)))
        previous = current
    manifests["sel_99"] = write_workload(
        args.data_root, "sel_99", rows_99, num_points,
        "query_minlen1_label1_sel967pct",
        {"target": 0.99, "label1_coverage": label1_coverage,
         "label1_count": NUM_QUERIES - empty_count, "empty_count": empty_count,
         "predicate_source_query_dirs": ["query_minlen1_label1_sel967pct",
                                           "query_empty_full100pct"]})

    if args.compute_groundtruth:
        for name, manifest in manifests.items():
            gt = compute_gt(args.compute_groundtruth, args.data_root, args.gt_root,
                            NAMES[name], args.threads, args.k,
                            args.overwrite_groundtruth)
            manifest["groundtruth_sha256"] = sha256(gt)
            manifest["groundtruth_bytes"] = gt.stat().st_size
            (args.data_root / NAMES[name] / "generation_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n")
    for name, manifest in manifests.items():
        print(name, json.dumps({key: manifest[key] for key in (
            "mean_selectivity", "min_selectivity", "max_selectivity",
            "unique_predicates", "empty_predicates")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
