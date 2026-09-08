#!/usr/bin/env python3
"""Build deterministic nested Amazon-x1 workloads above 75% selectivity.

The existing 75% query vectors are retained verbatim. For 80--95%, a
deterministic prefix of non-label-1 queries is relaxed to label 1, the widest
real Amazon label (96.70% coverage). The 96.7% anchor assigns label 1 to every
query. The 100% control uses empty containment predicates.
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
from collections import Counter
from pathlib import Path


TARGETS = (("sel_80", 0.80), ("sel_85", 0.85),
           ("sel_90", 0.90), ("sel_95", 0.95))
SEED = 20260908
SOURCE_QUERY_DIR = "query_minlen1_avgsel75pct"
OUTPUT_DIRS = {
    "sel_80": "query_minlen1_nested_avgsel80pct",
    "sel_85": "query_minlen1_nested_avgsel85pct",
    "sel_90": "query_minlen1_nested_avgsel90pct",
    "sel_95": "query_minlen1_nested_avgsel95pct",
    "sel_967": "query_minlen1_label1_sel967pct",
    "sel_100": "query_empty_full100pct",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        if sha256(target) != sha256(source):
            raise RuntimeError(f"existing query vectors differ: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def label_counts(path: Path) -> tuple[int, Counter[int]]:
    counts: Counter[int] = Counter()
    rows = 0
    with path.open() as stream:
        for line in stream:
            labels = [int(value) for value in line.strip().split(",") if value]
            if not labels:
                raise ValueError(f"empty base-label row {rows}")
            counts.update(labels)
            rows += 1
    return rows, counts


def closest_prefix(source: list[int], order: list[int], counts: Counter[int],
                   num_points: int, target: float) -> tuple[list[int], list[int], float]:
    labels = list(source)
    total = sum(counts[label] for label in labels)
    best = (abs(total / len(labels) / num_points - target), 0, total)
    for used, index in enumerate(order, 1):
        total += counts[1] - counts[labels[index]]
        candidate = (abs(total / len(labels) / num_points - target), used, total)
        if candidate[0] <= best[0]:
            best = candidate
        if total / len(labels) / num_points > target and used > best[1]:
            break
    changed = order[:best[1]]
    for index in changed:
        labels[index] = 1
    return labels, changed, best[2] / len(labels) / num_points


def write_workload(data_root: Path, source_dir: Path, name: str, labels: list[str],
                   coverages: list[int], changed: list[int], num_points: int) -> dict:
    target = data_root / OUTPUT_DIRS[name]
    target.mkdir(parents=True, exist_ok=True)
    for suffix in ("bin", "fvecs"):
        link_or_copy(source_dir / f"Amazon_query.{suffix}",
                     target / f"Amazon_query.{suffix}")
    label_path = target / "Amazon_query_labels.txt"
    label_path.write_text("\n".join(labels) + "\n")
    profile_path = target / f"profiled_{OUTPUT_DIRS[name].removeprefix('query_')}.csv"
    with profile_path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("coverage_count", "labels"))
        writer.writerows(zip(coverages, labels))
    manifest = {
        "schema_version": 1, "name": name, "query_dir": OUTPUT_DIRS[name],
        "source_query_dir": SOURCE_QUERY_DIR, "seed": SEED,
        "num_queries": len(labels), "num_base_points": num_points,
        "mean_selectivity": sum(coverages) / len(coverages) / num_points,
        "min_selectivity": min(coverages) / num_points,
        "max_selectivity": max(coverages) / num_points,
        "changed_query_indices": changed,
        "query_bin_sha256": sha256(target / "Amazon_query.bin"),
        "query_labels_sha256": sha256(label_path),
        "profile_sha256": sha256(profile_path),
    }
    (target / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def compute_gt(tool: Path, data_root: Path, query_dir: Path, output: Path,
               threads: int, k: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size == 1000 * k * 8:
        return
    command = [str(tool), "--data_type", "float", "--dist_fn", "L2",
               "--scenario", "containment", "--K", str(k),
               "--num_threads", str(threads),
               "--base_bin_file", str(data_root / "Amazon_base.bin"),
               "--base_label_file", str(data_root / "Amazon_base_labels.txt"),
               "--query_bin_file", str(query_dir / "Amazon_query.bin"),
               "--query_label_file", str(query_dir / "Amazon_query_labels.txt"),
               "--gt_file", str(output)]
    subprocess.run(command, check=True)


def compose_gt(source: Path, label1: Path, output: Path, changed: set[int], k: int) -> None:
    row_bytes = k * 8
    source_bytes, label1_bytes = source.read_bytes(), label1.read_bytes()
    expected = 1000 * row_bytes
    if len(source_bytes) != expected or len(label1_bytes) != expected:
        raise ValueError("anchor GT size does not match 1000 queries")
    merged = bytearray(source_bytes)
    for index in changed:
        start = index * row_bytes
        merged[start:start + row_bytes] = label1_bytes[start:start + row_bytes]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(merged)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path("/home/graphdb/FilterVectorData/Amazon"))
    parser.add_argument("--gt-root", type=Path,
                        default=Path("/home/graphdb/FilterVectorResult/Amazon/GroundTruth"))
    parser.add_argument("--compute-groundtruth", type=Path)
    parser.add_argument("--threads", type=int, default=72)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    source_dir = args.data_root / SOURCE_QUERY_DIR
    source_labels = [int(line.strip()) for line in
                     (source_dir / "Amazon_query_labels.txt").open()]
    if len(source_labels) != 1000 or any(label <= 0 for label in source_labels):
        raise ValueError("expected 1000 non-empty single-label source queries")
    num_points, counts = label_counts(args.data_root / "Amazon_base_labels.txt")
    if 1 not in counts or any(label not in counts for label in source_labels):
        raise ValueError("query label is absent from the Amazon base labels")

    order = [index for index, label in enumerate(source_labels) if label != 1]
    random.Random(SEED).shuffle(order)
    manifests, changed_by_name = {}, {}
    for name, target_selectivity in TARGETS:
        labels, changed, actual = closest_prefix(
            source_labels, order, counts, num_points, target_selectivity)
        manifests[name] = write_workload(
            args.data_root, source_dir, name, [str(label) for label in labels],
            [counts[label] for label in labels], changed, num_points)
        changed_by_name[name] = set(changed)
        print(f"{name}: {actual:.10f}, changed={len(changed)}")

    manifests["sel_967"] = write_workload(
        args.data_root, source_dir, "sel_967", ["1"] * len(source_labels),
        [counts[1]] * len(source_labels), order, num_points)
    manifests["sel_100"] = write_workload(
        args.data_root, source_dir, "sel_100", [""] * len(source_labels),
        [num_points] * len(source_labels), list(range(len(source_labels))), num_points)

    if args.compute_groundtruth:
        source_gt = args.gt_root / SOURCE_QUERY_DIR / "Amazon_gt_labels_containment.bin"
        label1_gt = args.gt_root / OUTPUT_DIRS["sel_967"] / "Amazon_gt_labels_containment.bin"
        full_gt = args.gt_root / OUTPUT_DIRS["sel_100"] / "Amazon_gt_labels_containment.bin"
        compute_gt(args.compute_groundtruth, args.data_root,
                   args.data_root / OUTPUT_DIRS["sel_967"], label1_gt, args.threads, args.k)
        compute_gt(args.compute_groundtruth, args.data_root,
                   args.data_root / OUTPUT_DIRS["sel_100"], full_gt, args.threads, args.k)
        for name, _ in TARGETS:
            compose_gt(source_gt, label1_gt,
                       args.gt_root / OUTPUT_DIRS[name] / "Amazon_gt_labels_containment.bin",
                       changed_by_name[name], args.k)
        for name in manifests:
            gt = args.gt_root / OUTPUT_DIRS[name] / "Amazon_gt_labels_containment.bin"
            manifests[name]["groundtruth_sha256"] = sha256(gt)
            (args.data_root / OUTPUT_DIRS[name] / "generation_manifest.json").write_text(
                json.dumps(manifests[name], indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
