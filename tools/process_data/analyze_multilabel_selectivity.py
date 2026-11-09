#!/usr/bin/env python3
"""Analyze selectivity coverage of frequent multi-label query subsets.

For a query label set S, a base vector matches when S is a subset of the
vector's label set.  The tool selects the globally most frequent labels,
counts their joint membership masks, and reports the closest subset with at
least two labels for each requested target selectivity.
"""

from __future__ import annotations

import argparse
import heapq
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TARGETS = (0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70)


@dataclass(frozen=True)
class Candidate:
    labels: tuple[int, ...]
    count: int
    selectivity: float


def parse_labels(line: str, path: Path, line_number: int) -> list[int]:
    stripped = line.strip()
    if not stripped:
        raise ValueError(f"{path}:{line_number}: empty label row")
    try:
        labels = [int(value) for value in stripped.split(",")]
    except ValueError as exc:
        raise ValueError(
            f"{path}:{line_number}: invalid comma-separated label row"
        ) from exc
    if any(right <= left for left, right in zip(labels, labels[1:])):
        raise ValueError(f"{path}:{line_number}: labels are not unique and sorted")
    return labels


def analyze(
    path: Path,
    *,
    top_k: int,
    targets: tuple[float, ...],
    min_query_labels: int,
) -> tuple[int, list[tuple[int, int]], list[Candidate]]:
    frequencies: Counter[int] = Counter()
    rows = 0
    with path.open("r", encoding="utf-8", buffering=1024 * 1024) as source:
        for rows, line in enumerate(source, start=1):
            frequencies.update(parse_labels(line, path, rows))
    if rows == 0:
        raise ValueError(f"empty label file: {path}")

    frequent = heapq.nlargest(
        min(top_k, len(frequencies)), frequencies.items(), key=lambda item: item[1]
    )
    frequent_labels = tuple(label for label, _ in frequent)
    label_to_bit = {label: bit for bit, label in enumerate(frequent_labels)}
    mask_counts = [0] * (1 << len(frequent_labels))

    with path.open("r", encoding="utf-8", buffering=1024 * 1024) as source:
        for line_number, line in enumerate(source, start=1):
            mask = 0
            for label in parse_labels(line, path, line_number):
                bit = label_to_bit.get(label)
                if bit is not None:
                    mask |= 1 << bit
            mask_counts[mask] += 1

    candidates: list[Candidate] = []
    for subset in range(1, 1 << len(frequent_labels)):
        if subset.bit_count() < min_query_labels:
            continue
        count = sum(
            mask_count
            for mask, mask_count in enumerate(mask_counts)
            if mask & subset == subset
        )
        labels = tuple(
            frequent_labels[bit]
            for bit in range(len(frequent_labels))
            if subset & (1 << bit)
        )
        candidates.append(Candidate(labels, count, count / rows))

    nearest = [
        min(candidates, key=lambda candidate: abs(candidate.selectivity - target))
        for target in targets
    ]
    return rows, frequent, nearest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-query-labels", type=int, default=2)
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=DEFAULT_TARGETS,
        help="Target selectivities expressed as fractions (default: 0.20 ... 0.70)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.top_k <= 20:
        raise ValueError("--top-k must be between 1 and 20")
    if args.min_query_labels < 1:
        raise ValueError("--min-query-labels must be positive")
    targets = tuple(args.targets)
    if any(not 0.0 <= target <= 1.0 for target in targets):
        raise ValueError("all targets must be in [0, 1]")

    for path in args.files:
        path = path.expanduser().resolve()
        rows, frequent, nearest = analyze(
            path,
            top_k=args.top_k,
            targets=targets,
            min_query_labels=args.min_query_labels,
        )
        print(f"DATASET file={path} rows={rows} top_k={len(frequent)}")
        print(
            "TOP "
            + " ".join(
                f"{label}:{count / rows:.6%}" for label, count in frequent
            )
        )
        for target, candidate in zip(targets, nearest):
            labels = ",".join(map(str, sorted(candidate.labels)))
            error = abs(candidate.selectivity - target)
            print(
                f"NEAREST target={target:.2%} labels={labels} "
                f"length={len(candidate.labels)} count={candidate.count} "
                f"actual={candidate.selectivity:.6%} error_pp={100 * error:.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
