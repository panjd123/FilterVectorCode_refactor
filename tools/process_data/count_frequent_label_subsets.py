#!/usr/bin/env python3
"""Count distinct non-empty label query subsets above support thresholds."""

from __future__ import annotations

import argparse
import itertools
from collections import Counter
from pathlib import Path

import numpy as np


def parse_row(line: str) -> tuple[int, ...]:
    return tuple(map(int, line.strip().split(",")))


def label_frequencies(path: Path) -> tuple[int, Counter[int]]:
    rows = 0
    frequencies: Counter[int] = Counter()
    with path.open("r", encoding="utf-8", buffering=1024 * 1024) as source:
        for rows, line in enumerate(source, start=1):
            frequencies.update(parse_row(line))
    return rows, frequencies


def exact_counts(
    path: Path,
    rows: int,
    labels: tuple[int, ...],
    thresholds: tuple[float, ...],
) -> dict[float, Counter[int]]:
    label_to_bit = {label: bit for bit, label in enumerate(labels)}
    mask_counts = np.zeros(1 << len(labels), dtype=np.int64)
    with path.open("r", encoding="utf-8", buffering=1024 * 1024) as source:
        for line in source:
            mask = 0
            for label in parse_row(line):
                bit = label_to_bit.get(label)
                if bit is not None:
                    mask |= 1 << bit
            mask_counts[mask] += 1

    supports = mask_counts.copy()
    for bit in range(len(labels)):
        step = 1 << bit
        blocks = supports.reshape(-1, step * 2)
        blocks[:, :step] += blocks[:, step:]

    results = {threshold: Counter() for threshold in thresholds}
    minimum_counts = {
        threshold: int(np.ceil(threshold * rows)) for threshold in thresholds
    }
    for subset in range(1, 1 << len(labels)):
        length = subset.bit_count()
        support = int(supports[subset])
        for threshold in thresholds:
            if support >= minimum_counts[threshold]:
                results[threshold][length] += 1
    return results


def bounded_counts(
    path: Path,
    rows: int,
    labels: tuple[int, ...],
    thresholds: tuple[float, ...],
    max_length: int,
) -> dict[float, Counter[int]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    byte_count = (rows + 7) // 8
    bitsets = [bytearray(byte_count) for _ in labels]
    with path.open("r", encoding="utf-8", buffering=1024 * 1024) as source:
        for row, line in enumerate(source):
            byte_index = row >> 3
            bit = 1 << (row & 7)
            for label in parse_row(line):
                index = label_to_index.get(label)
                if index is not None:
                    bitsets[index][byte_index] |= bit
    integer_sets = [int.from_bytes(bits, "little") for bits in bitsets]

    results = {threshold: Counter() for threshold in thresholds}
    minimum_counts = {
        threshold: int(np.ceil(threshold * rows)) for threshold in thresholds
    }
    for length in range(1, max_length + 1):
        for indices in itertools.combinations(range(len(labels)), length):
            intersection = integer_sets[indices[0]]
            for index in indices[1:]:
                intersection &= integer_sets[index]
            support = intersection.bit_count()
            for threshold in thresholds:
                if support >= minimum_counts[threshold]:
                    results[threshold][length] += 1
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--thresholds", nargs="+", type=float, default=(0.10, 0.20))
    parser.add_argument("--exact-label-limit", type=int, default=20)
    parser.add_argument("--bounded-max-length", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = tuple(sorted(set(args.thresholds)))
    if any(not 0.0 < threshold <= 1.0 for threshold in thresholds):
        raise ValueError("thresholds must be in (0, 1]")
    minimum_threshold = min(thresholds)

    for source_path in args.files:
        path = source_path.expanduser().resolve()
        rows, frequencies = label_frequencies(path)
        labels = tuple(
            label
            for label, count in sorted(
                frequencies.items(), key=lambda item: (-item[1], item[0])
            )
            if count / rows >= minimum_threshold
        )
        exact = len(labels) <= args.exact_label_limit
        if exact:
            results = exact_counts(path, rows, labels, thresholds)
        else:
            results = bounded_counts(
                path,
                rows,
                labels,
                thresholds,
                args.bounded_max_length,
            )
        print(
            f"DATASET dataset={path.parent.name} rows={rows} "
            f"candidate_labels={len(labels)} mode={'exact' if exact else 'lower_bound'}"
        )
        for threshold in thresholds:
            by_length = results[threshold]
            total = sum(by_length.values())
            details = ",".join(
                f"len{length}={count}" for length, count in sorted(by_length.items())
            )
            print(
                f"FREQUENT support_ge={threshold:.2%} label_sets={total} "
                f"{details}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
