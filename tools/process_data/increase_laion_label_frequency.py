#!/usr/bin/env python3
"""Randomly increase selected label frequencies in a base-label text file.

Existing label memberships are always preserved.  For every row and every
configured label that is not already present, the label is added independently
with the configured probability.  A fixed seed makes the generated file
reproducible while keeping the assignment unrelated to the input row order.

The default configuration raises Laion labels 2..7 with probabilities 70% down
to 20% and writes a new file; the source file is never overwritten by default.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import time
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = WORKSPACE_ROOT / "FilterVectorData/Laion/Laion_base_labels.txt"
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT / "FilterVectorData/Laion/Laion_base_labels_midselect.txt"
)
DEFAULT_PROBABILITIES = {
    2: 0.70,
    3: 0.60,
    4: 0.50,
    5: 0.40,
    6: 0.30,
    7: 0.20,
}


def parse_probability(value: str) -> tuple[int, float]:
    """Parse a ``LABEL=PROBABILITY`` command-line value."""
    try:
        raw_label, raw_probability = value.split("=", 1)
        label = int(raw_label)
        probability = float(raw_probability)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected LABEL=PROBABILITY, got {value!r}"
        ) from exc
    if label < 0:
        raise argparse.ArgumentTypeError("label must be non-negative")
    if not 0.0 <= probability <= 1.0:
        raise argparse.ArgumentTypeError("probability must be in [0, 1]")
    return label, probability


def augment_labels(
    input_path: Path,
    output_path: Path,
    probabilities: dict[int, float],
    *,
    seed: int,
    overwrite: bool,
    progress_every: int,
) -> tuple[int, dict[int, int], dict[int, int]]:
    """Create the augmented label file and return rows/additions/final counts."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input label file does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("input and output must differ; write a new file first")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; pass --overwrite to replace it"
        )
    if not probabilities:
        raise ValueError("at least one label probability is required")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    controlled_labels = tuple(sorted(probabilities))
    rng = random.Random(seed)
    additions = {label: 0 for label in controlled_labels}
    final_counts = {label: 0 for label in controlled_labels}
    row_count = 0
    temporary_name: str | None = None
    started = time.monotonic()

    try:
        with input_path.open("r", encoding="utf-8", buffering=1024 * 1024) as source:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                buffering=1024 * 1024,
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as destination:
                temporary_name = destination.name
                for row_count, line in enumerate(source, start=1):
                    stripped = line.strip()
                    if not stripped:
                        raise ValueError(f"{input_path}:{row_count}: empty label row")
                    try:
                        labels = [int(value) for value in stripped.split(",")]
                    except ValueError as exc:
                        raise ValueError(
                            f"{input_path}:{row_count}: invalid comma-separated labels"
                        ) from exc

                    present = set(labels)
                    if len(present) != len(labels):
                        raise ValueError(
                            f"{input_path}:{row_count}: duplicate labels in input row"
                        )

                    # Draw once for every controlled label, even when already
                    # present, so the random stream is stable for every row.
                    for label in controlled_labels:
                        selected = rng.random() < probabilities[label]
                        if label not in present and selected:
                            labels.append(label)
                            present.add(label)
                            additions[label] += 1
                        if label in present:
                            final_counts[label] += 1

                    labels.sort()
                    destination.write(",".join(map(str, labels)))
                    destination.write("\n")

                    if progress_every > 0 and row_count % progress_every == 0:
                        elapsed = time.monotonic() - started
                        print(
                            f"processed {row_count:,} rows in {elapsed:.1f}s",
                            file=sys.stderr,
                            flush=True,
                        )

        os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    return row_count, additions, final_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--probability",
        action="append",
        type=parse_probability,
        metavar="LABEL=PROBABILITY",
        help=(
            "Label and random addition probability; repeat for multiple labels. "
            "Defaults to 2=.70, 3=.60, ..., 7=.20."
        ),
    )
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    probabilities = (
        dict(args.probability) if args.probability else DEFAULT_PROBABILITIES
    )
    rows, additions, final_counts = augment_labels(
        args.input,
        args.output,
        probabilities,
        seed=args.seed,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )

    print(f"output={args.output.expanduser().resolve()}")
    print(f"seed={args.seed}")
    print(f"rows={rows}")
    for label in sorted(probabilities):
        selectivity = final_counts[label] / rows if rows else 0.0
        print(
            f"label={label} add_probability={probabilities[label]:.2%} "
            f"added={additions[label]} final_count={final_counts[label]} "
            f"selectivity={selectivity:.6%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
