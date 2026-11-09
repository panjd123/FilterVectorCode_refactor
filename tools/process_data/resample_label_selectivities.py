#!/usr/bin/env python3
"""Rebuild label files while independently adjusting selected labels.

Every uncontrolled label is copied from the source without modification.  A
controlled label can either be redrawn independently for every row or only
added to missing rows.  Targets are sampled per dataset from the requested
interval, and all random choices are deterministic for a fixed seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path


def stable_seed(base_seed: int, dataset: str, purpose: str) -> int:
    payload = f"{base_seed}:{dataset}:{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def parse_row(line: str, path: Path, line_number: int) -> list[int]:
    stripped = line.strip()
    if not stripped:
        raise ValueError(f"{path}:{line_number}: empty label row")
    try:
        labels = [int(value) for value in stripped.split(",")]
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: invalid label row") from exc
    if labels[0] < 0 or any(b <= a for a, b in zip(labels, labels[1:])):
        raise ValueError(f"{path}:{line_number}: labels must be sorted and unique")
    return labels


def inspect(path: Path, controlled: set[int]) -> tuple[int, int, Counter[int]]:
    rows = 0
    occurrences = 0
    counts: Counter[int] = Counter()
    with path.open("r", encoding="utf-8", buffering=1024 * 1024) as source:
        for rows, line in enumerate(source, start=1):
            labels = parse_row(line, path, rows)
            occurrences += len(labels)
            counts.update(
                label for label in labels if label in controlled or label in (1, 2)
            )
    if rows == 0:
        raise ValueError(f"empty label file: {path}")
    return rows, occurrences, counts


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as destination:
            temporary = destination.name
            json.dump(value, destination, indent=2, sort_keys=True)
            destination.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def default_output(source: Path) -> Path:
    suffix = "_base_labels_old.txt"
    if not source.name.endswith(suffix):
        raise ValueError(f"cannot derive output from {source}; expected *{suffix}")
    return source.with_name(source.name.removesuffix("_old.txt") + ".txt")


def rebuild(
    source: Path,
    *,
    labels: tuple[int, ...],
    target_min: float,
    target_max: float,
    base_seed: int,
    output: Path | None,
    backup_tag: str,
    progress_every: int,
    add_only: bool,
) -> dict[str, object]:
    source = source.expanduser().resolve()
    destination = (output or default_output(source)).expanduser().resolve()
    if source == destination:
        raise ValueError("source and output must differ")
    if not source.is_file():
        raise FileNotFoundError(source)

    controlled = set(labels)
    rows, old_occurrences, old_counts = inspect(source, controlled)
    dataset = source.parent.name
    target_rng = random.Random(stable_seed(base_seed, dataset, "targets"))
    targets = {label: target_rng.uniform(target_min, target_max) for label in labels}
    add_probabilities = {
        label: (
            0.0
            if old_counts[label] / rows >= targets[label]
            else (targets[label] - old_counts[label] / rows)
            / (1.0 - old_counts[label] / rows)
        )
        for label in labels
    }
    row_rng = random.Random(stable_seed(base_seed, dataset, "memberships"))
    final_counts = Counter()
    new_occurrences = 0

    backup = destination.with_name(
        f"{destination.stem}_{backup_tag}{destination.suffix}"
    )
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")

    temporary: str | None = None
    try:
        with source.open("r", encoding="utf-8", buffering=1024 * 1024) as input_file:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", buffering=1024 * 1024,
                dir=destination.parent, prefix=f".{destination.name}.resample.",
                suffix=".tmp", delete=False,
            ) as output_file:
                temporary = output_file.name
                for row_number, line in enumerate(input_file, start=1):
                    original = parse_row(line, source, row_number)
                    if add_only:
                        rebuilt = list(original)
                        present = set(original)
                        for label in labels:
                            if label in present:
                                final_counts[label] += 1
                            elif row_rng.random() < add_probabilities[label]:
                                rebuilt.append(label)
                                final_counts[label] += 1
                    else:
                        rebuilt = [label for label in original if label not in controlled]
                        for label in labels:
                            if row_rng.random() < targets[label]:
                                rebuilt.append(label)
                                final_counts[label] += 1
                    rebuilt.sort()
                    new_occurrences += len(rebuilt)
                    output_file.write(",".join(map(str, rebuilt)))
                    output_file.write("\n")
                    if progress_every and row_number % progress_every == 0:
                        print(f"{dataset}: {row_number:,}/{rows:,}", flush=True)

        if destination.exists():
            os.link(destination, backup)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)

    metadata: dict[str, object] = {
        "dataset": dataset,
        "source_file": str(source),
        "output_file": str(destination),
        "replaced_file_backup": str(backup) if backup.exists() else None,
        "base_seed": base_seed,
        "mode": "add_only" if add_only else "resample",
        "rows": rows,
        "old_average_labels_per_point": old_occurrences / rows,
        "new_average_labels_per_point": new_occurrences / rows,
        "preserved_label_1_count": old_counts[1],
        "preserved_label_1_selectivity": old_counts[1] / rows,
        "preserved_label_2_count": old_counts[2],
        "preserved_label_2_selectivity": old_counts[2] / rows,
        "controls": [
            {
                "label": label,
                "original_count": old_counts[label],
                "original_selectivity": old_counts[label] / rows,
                "target_selectivity": targets[label],
                "add_probability": add_probabilities[label] if add_only else None,
                "final_count": final_counts[label],
                "final_selectivity": final_counts[label] / rows,
            }
            for label in labels
        ],
    }
    label_tag = "_".join(map(str, labels))
    metadata_path = source.parent / (
        f"{dataset}_labels{label_tag}_target{100 * target_min:g}to"
        f"{100 * target_max:g}pct.json"
    )
    atomic_json(metadata_path, metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+", type=int, default=(2, 3, 4))
    parser.add_argument("--target-min", type=float, default=0.45)
    parser.add_argument("--target-max", type=float, default=0.50)
    parser.add_argument("--base-seed", type=int, default=20261009)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backup-tag", default="before_label_adjustment")
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument(
        "--add-only",
        action="store_true",
        help="Preserve existing memberships and only add labels to missing rows",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = tuple(args.labels)
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("--labels must be unique and non-empty")
    if 1 in labels:
        raise ValueError("label 1 is preserved and cannot be resampled")
    if not 0.0 < args.target_min < args.target_max < 1.0:
        raise ValueError("targets must satisfy 0 < min < max < 1")
    if args.output is not None and len(args.files) != 1:
        raise ValueError("--output is only valid for one input")

    for source in args.files:
        metadata = rebuild(
            source,
            labels=labels,
            target_min=args.target_min,
            target_max=args.target_max,
            base_seed=args.base_seed,
            output=args.output,
            backup_tag=args.backup_tag,
            progress_every=args.progress_every,
            add_only=args.add_only,
        )
        print(
            f"DATASET dataset={metadata['dataset']} rows={metadata['rows']} "
            f"label1={metadata['preserved_label_1_selectivity']:.6%} "
            f"label2={metadata['preserved_label_2_selectivity']:.6%} "
            f"old_avg={metadata['old_average_labels_per_point']:.6f} "
            f"new_avg={metadata['new_average_labels_per_point']:.6f}"
        )
        for control in metadata["controls"]:
            print(
                f"CONTROL label={control['label']} "
                f"original={control['original_selectivity']:.6%} "
                f"target={control['target_selectivity']:.6%} "
                f"final={control['final_selectivity']:.6%}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
