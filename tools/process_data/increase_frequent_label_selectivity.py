#!/usr/bin/env python3
"""Independently increase several frequent labels with random additions.

This generalizes the Laion adjustment: existing memberships are preserved;
among labels below a configurable selectivity ceiling, the most frequent ones
are selected and independently added to missing rows with descending random
probabilities (70%, 60%, ..., 20% by default).

With ``--in-place``, the source is atomically replaced after generation and its
original inode is retained through a ``*_base_labels_old.txt`` hard-link backup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path


DEFAULT_PROBABILITIES = (0.70, 0.60, 0.50, 0.40, 0.30, 0.20)


def dataset_seed(base_seed: int, path: Path) -> int:
    payload = f"{base_seed}:{path.parent.name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def parse_row(line: str, path: Path, line_number: int) -> list[int]:
    stripped = line.strip()
    if not stripped:
        raise ValueError(f"{path}:{line_number}: empty label row")
    try:
        labels = [int(value) for value in stripped.split(",")]
    except ValueError as exc:
        raise ValueError(
            f"{path}:{line_number}: invalid comma-separated labels"
        ) from exc
    if labels[0] < 0:
        raise ValueError(f"{path}:{line_number}: labels must be non-negative")
    if any(right <= left for left, right in zip(labels, labels[1:])):
        raise ValueError(f"{path}:{line_number}: labels are not unique and sorted")
    return labels


def inspect_input(path: Path) -> tuple[int, int, Counter[int], int]:
    rows = 0
    occurrences = 0
    max_label = -1
    frequencies: Counter[int] = Counter()
    with path.open("r", encoding="utf-8", buffering=1024 * 1024) as source:
        for rows, line in enumerate(source, start=1):
            labels = parse_row(line, path, rows)
            frequencies.update(labels)
            occurrences += len(labels)
            max_label = max(max_label, labels[-1])
    if rows == 0:
        raise ValueError(f"empty label file: {path}")
    return rows, occurrences, frequencies, max_label


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_random_controls{input_path.suffix}")


def default_backup_path(input_path: Path, backup_tag: str) -> Path:
    return input_path.with_name(
        f"{input_path.stem}_{backup_tag}{input_path.suffix}"
    )


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary_name = destination.name
            json.dump(value, destination, indent=2, sort_keys=True)
            destination.write("\n")
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def adjust_file(
    input_path: Path,
    *,
    probabilities: tuple[float, ...],
    ceiling: float,
    target_final: float | None,
    top_k: int,
    explicit_labels: tuple[int, ...] | None,
    explicit_targets: tuple[tuple[int, float], ...] | None,
    base_seed: int,
    output_path: Path | None,
    backup_path: Path | None,
    in_place: bool,
    overwrite: bool,
    progress_every: int,
    backup_tag: str,
) -> dict[str, object]:
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input label file does not exist: {input_path}")
    rows, old_occurrences, frequencies, max_label = inspect_input(input_path)

    if explicit_targets is not None:
        controlled_labels = tuple(label for label, _ in explicit_targets)
        final_targets = tuple(target for _, target in explicit_targets)
        missing = [label for label in controlled_labels if label not in frequencies]
        if missing:
            raise ValueError(f"{input_path}: labels not found: {missing}")
        probabilities = tuple(
            0.0
            if frequencies[label] / rows >= target
            else (target - frequencies[label] / rows)
            / (1.0 - frequencies[label] / rows)
            for label, target in explicit_targets
        )
    elif target_final is not None:
        ranked = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
        controlled_labels = (
            explicit_labels
            if explicit_labels is not None
            else tuple(label for label, _ in ranked[:top_k])
        )
        missing = [label for label in controlled_labels if label not in frequencies]
        if missing:
            raise ValueError(f"{input_path}: labels not found: {missing}")
        probabilities = tuple(
            0.0
            if frequencies[label] / rows >= target_final
            else (target_final - frequencies[label] / rows)
            / (1.0 - frequencies[label] / rows)
            for label in controlled_labels
        )
        final_targets = tuple(target_final for _ in controlled_labels)
    elif explicit_labels is None:
        eligible = [
            (label, count)
            for label, count in frequencies.items()
            if count / rows < ceiling
        ]
        eligible.sort(key=lambda item: (-item[1], item[0]))
        if len(eligible) < len(probabilities):
            raise ValueError(
                f"{input_path}: only {len(eligible)} labels are below ceiling "
                f"{ceiling:.2%}, need {len(probabilities)}"
            )
        controlled_labels = tuple(label for label, _ in eligible[: len(probabilities)])
        final_targets = tuple(None for _ in controlled_labels)
    else:
        controlled_labels = explicit_labels
        missing = [label for label in controlled_labels if label not in frequencies]
        if missing:
            raise ValueError(f"{input_path}: labels not found: {missing}")
        final_targets = tuple(None for _ in controlled_labels)
    if len(controlled_labels) != len(probabilities):
        raise ValueError("the number of labels must equal the number of probabilities")

    output_path = (
        input_path if in_place else (output_path or default_output_path(input_path))
    ).expanduser().resolve()
    backup_path = (
        (backup_path or default_backup_path(input_path, backup_tag))
        .expanduser()
        .resolve()
        if in_place
        else None
    )
    if not in_place and output_path == input_path:
        raise ValueError("output equals input; pass --in-place explicitly")
    if not in_place and output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; pass --overwrite to replace it"
        )
    if in_place and backup_path is not None and backup_path.exists():
        raise FileExistsError(f"backup already exists: {backup_path}")

    seed = dataset_seed(base_seed, input_path)
    rng = random.Random(seed)
    additions = {label: 0 for label in controlled_labels}
    final_counts = {label: frequencies[label] for label in controlled_labels}
    temporary_name: str | None = None
    started = time.monotonic()
    try:
        with input_path.open("r", encoding="utf-8", buffering=1024 * 1024) as source:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                buffering=1024 * 1024,
                dir=input_path.parent,
                prefix=f".{input_path.name}.random.",
                suffix=".tmp",
                delete=False,
            ) as destination:
                temporary_name = destination.name
                for row_number, line in enumerate(source, start=1):
                    labels = parse_row(line, input_path, row_number)
                    present = set(labels)
                    for label, probability in zip(controlled_labels, probabilities):
                        selected = rng.random() < probability
                        if label not in present and selected:
                            labels.append(label)
                            present.add(label)
                            additions[label] += 1
                            final_counts[label] += 1
                    labels.sort()
                    destination.write(",".join(map(str, labels)))
                    destination.write("\n")
                    if progress_every > 0 and row_number % progress_every == 0:
                        elapsed = time.monotonic() - started
                        print(
                            f"{input_path.parent.name}: processed "
                            f"{row_number:,}/{rows:,} rows in {elapsed:.1f}s",
                            file=sys.stderr,
                            flush=True,
                        )

        if in_place:
            assert backup_path is not None
            os.link(input_path, backup_path)
            os.replace(temporary_name, input_path)
        else:
            os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    controls = []
    for label, probability, final_target in zip(
        controlled_labels, probabilities, final_targets
    ):
        controls.append(
            {
                "label": label,
                "target_final_selectivity": final_target,
                "old_count": frequencies[label],
                "old_selectivity": frequencies[label] / rows,
                "add_probability": probability,
                "added_count": additions[label],
                "final_count": final_counts[label],
                "final_selectivity": final_counts[label] / rows,
            }
        )
    metadata: dict[str, object] = {
        "dataset": input_path.parent.name,
        "label_file": str(input_path if in_place else output_path),
        "backup_file": str(backup_path) if backup_path is not None else None,
        "base_seed": base_seed,
        "derived_seed": seed,
        "rows": rows,
        "max_label": max_label,
        "selection_ceiling": ceiling,
        "target_final_selectivity": target_final,
        "target_final_selectivities": (
            {str(label): target for label, target in explicit_targets}
            if explicit_targets is not None
            else None
        ),
        "old_average_labels_per_point": old_occurrences / rows,
        "new_average_labels_per_point": (
            old_occurrences + sum(additions.values())
        )
        / rows,
        "controls": controls,
    }
    if explicit_targets is not None:
        metadata_suffix = "explicit_label_targets"
    elif target_final is not None:
        metadata_suffix = f"top{len(controlled_labels)}_target{100 * target_final:g}pct"
    else:
        metadata_suffix = "independent_control_labels"
    metadata_path = input_path.with_name(
        f"{input_path.parent.name}_{metadata_suffix}.json"
    )
    write_json_atomic(metadata_path, metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--probabilities", nargs="+", type=float, default=DEFAULT_PROBABILITIES
    )
    parser.add_argument(
        "--ceiling",
        type=float,
        default=0.20,
        help="Only auto-select labels whose current selectivity is below this value",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=int,
        help="Explicit labels for a single input file; disables automatic selection",
    )
    parser.add_argument(
        "--label-targets",
        nargs="+",
        metavar="LABEL=TARGET",
        help=(
            "Explicit per-label final selectivities, for example "
            "1=0.8 2=0.7; missing memberships are sampled independently"
        ),
    )
    parser.add_argument(
        "--target-final",
        type=float,
        help=(
            "Select the current top-K labels and independently add missing "
            "members until each label reaches this expected final selectivity"
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20261009)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--backup-tag", default="old")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    explicit_targets: tuple[tuple[int, float], ...] | None = None
    if args.label_targets is not None:
        parsed_targets: list[tuple[int, float]] = []
        for value in args.label_targets:
            try:
                label_text, target_text = value.split("=", 1)
                label = int(label_text)
                target = float(target_text)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"invalid --label-targets value {value!r}; expected LABEL=TARGET"
                ) from exc
            if label < 0:
                raise ValueError("label targets must use non-negative labels")
            if not 0.0 < target <= 1.0:
                raise ValueError("label targets must be in (0, 1]")
            parsed_targets.append((label, target))
        labels = [label for label, _ in parsed_targets]
        if len(labels) != len(set(labels)):
            raise ValueError("--label-targets contains duplicate labels")
        explicit_targets = tuple(parsed_targets)
    if explicit_targets is not None and (
        args.labels is not None or args.target_final is not None
    ):
        raise ValueError(
            "--label-targets cannot be combined with --labels or --target-final"
        )
    probabilities = tuple(args.probabilities)
    if not probabilities or any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("probabilities must be non-empty and in [0, 1]")
    if args.target_final is None and explicit_targets is None and any(
        right >= left for left, right in zip(probabilities, probabilities[1:])
    ):
        raise ValueError("probabilities must be strictly decreasing")
    if args.target_final is not None and not 0.0 < args.target_final <= 1.0:
        raise ValueError("--target-final must be in (0, 1]")
    if args.top_k < 2:
        raise ValueError("--top-k must be at least 2")
    if not 0.0 < args.ceiling <= 1.0:
        raise ValueError("--ceiling must be in (0, 1]")
    if len(args.files) > 1 and (
        args.output is not None
        or args.backup is not None
        or args.labels is not None
        or explicit_targets is not None
    ):
        raise ValueError(
            "--output, --backup, and --labels may only be used with one input file"
        )

    explicit_labels = tuple(args.labels) if args.labels is not None else None
    for path in args.files:
        metadata = adjust_file(
            path,
            probabilities=probabilities,
            ceiling=args.ceiling,
            target_final=args.target_final,
            top_k=args.top_k,
            explicit_labels=explicit_labels,
            explicit_targets=explicit_targets,
            base_seed=args.base_seed,
            output_path=args.output,
            backup_path=args.backup,
            in_place=args.in_place,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
            backup_tag=args.backup_tag,
        )
        print(
            f"DATASET dataset={metadata['dataset']} rows={metadata['rows']} "
            f"old_avg={metadata['old_average_labels_per_point']:.6f} "
            f"new_avg={metadata['new_average_labels_per_point']:.6f}"
        )
        for control in metadata["controls"]:
            print(
                f"CONTROL label={control['label']} "
                f"old={control['old_selectivity']:.6%} "
                f"add_probability={control['add_probability']:.2%} "
                f"added={control['added_count']} "
                f"final={control['final_selectivity']:.6%}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
