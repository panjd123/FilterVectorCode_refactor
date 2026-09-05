#!/usr/bin/env python3
"""Fill empty base-label rows with the previous label set."""

from __future__ import annotations

import argparse
from pathlib import Path


def fill_empty_rows(text: str) -> tuple[str, list[int]]:
    """Return text with blank rows filled and one-based changed line numbers."""
    lines = text.splitlines(keepends=True)
    changed: list[int] = []
    previous: str | None = None

    for line_number, line in enumerate(lines, start=1):
        content = line.rstrip("\r\n")
        if content.strip():
            previous = content
            continue

        if previous is None:
            raise ValueError(
                f"line {line_number} is empty and has no previous label set"
            )

        newline = line[len(content) :]
        lines[line_number - 1] = previous + newline
        changed.append(line_number)

    return "".join(lines), changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill empty base-label rows with the immediately previous row."
    )
    parser.add_argument("input", type=Path, help="input base-label file")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--in-place",
        action="store_true",
        help="replace the input file after a successful transformation",
    )
    destination.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write the transformed file to this path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report empty rows; do not write a file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check and (args.in_place or args.output is not None):
        raise SystemExit("--check cannot be combined with --in-place or --output")
    if not args.check and not args.in_place and args.output is None:
        raise SystemExit("one of --check, --in-place, or --output is required")

    input_path: Path = args.input
    text = input_path.read_text(encoding="utf-8")
    transformed, changed = fill_empty_rows(text)

    if args.check:
        if changed:
            print(f"{input_path}: empty rows: {', '.join(map(str, changed))}")
        else:
            print(f"{input_path}: no empty rows")
        return 0

    output_path = input_path if args.in_place else args.output
    assert output_path is not None
    output_path.write_text(transformed, encoding="utf-8", newline="")
    print(f"filled {len(changed)} row(s): {', '.join(map(str, changed)) or 'none'}")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
