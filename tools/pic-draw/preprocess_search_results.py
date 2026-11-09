#!/usr/bin/env python3
"""Prepare search summaries for Recall-QPS plotting.

Processing rules:
1. Remove the first data row (the CSV header is retained).
2. Sort the remaining ``Average_Time_ms`` values in ascending order.
3. Put only those sorted values back into the time column.  Every other cell
   remains in its original row and is written unchanged.
4. Save the result to a different CSV file; never overwrite the source file.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


# =============================================================================
# User-adjustable defaults
# =============================================================================

TIME_COLUMN = "Average_Time_ms"
DROP_FIRST_ROWS = 1
DEFAULT_OUTPUT_NAME = "search_time_summary_draw.csv"
# 预处理文件只保留绘图所需的四列，顺序也以此处为准。
OUTPUT_COLUMNS = (
    "Lsearch",
    "Average_Efs",
    "Average_Time_ms",
    "Average_Recall",
)


def processed_output_is_valid(
    output_path: Path, *, time_column: str = TIME_COLUMN
) -> bool:
    """Return True when an existing output is usable without rebuilding."""
    if not output_path.is_file():
        return False
    try:
        with output_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(OUTPUT_COLUMNS):
                return False
            rows = list(reader)
        if not rows:
            return False
        times = [float(row[time_column]) for row in rows]
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return all(
        math.isfinite(value) and value > 0 for value in times
    ) and all(left <= right for left, right in zip(times, times[1:]))



def preprocess_file(
    input_path: Path,
    output_path: Path,
    *,
    time_column: str = TIME_COLUMN,
    drop_first_rows: int = DROP_FIRST_ROWS,
    force: bool = False,
) -> int:
    """Preprocess one summary CSV and return the number of written rows."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if input_path == output_path:
        raise ValueError("Output must be a new file; refusing to overwrite the source CSV")
    if drop_first_rows < 0:
        raise ValueError("drop_first_rows must be non-negative")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    # 已有文件满足四列表头、非空且时间升序时直接复用；force=True 可强制重建。
    if not force and processed_output_is_valid(output_path, time_column=time_column):
        return 0

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV has no header: {input_path}")
        missing_columns = [name for name in OUTPUT_COLUMNS if name not in fieldnames]
        if missing_columns:
            names = ", ".join(missing_columns)
            raise ValueError(f"{input_path}: missing output column(s): {names}")
        if time_column not in fieldnames:
            raise ValueError(f"{input_path}: missing time column {time_column!r}")
        rows = list(reader)

    if len(rows) <= drop_first_rows:
        raise ValueError(
            f"{input_path}: contains {len(rows)} data row(s), cannot drop "
            f"{drop_first_rows} and keep plotting data"
        )

    # Copy the retained dictionaries so the caller's source representation is
    # never mutated.  Row order and all non-time cells remain unchanged.
    output_rows = [dict(row) for row in rows[drop_first_rows:]]
    sorted_times: list[tuple[float, str]] = []
    for source_line, row in enumerate(output_rows, start=drop_first_rows + 2):
        raw_value = row.get(time_column, "")
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{input_path}:{source_line}: invalid {time_column}: {raw_value!r}"
            ) from exc
        if not math.isfinite(numeric_value) or numeric_value <= 0:
            raise ValueError(
                f"{input_path}:{source_line}: {time_column} must be finite and positive"
            )
        # Retain the source string formatting while sorting by numeric value.
        sorted_times.append((numeric_value, str(raw_value)))

    sorted_times.sort(key=lambda item: item[0])
    for row, (_, raw_time) in zip(output_rows, sorted_times):
        row[time_column] = raw_time

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {name: row[name] for name in OUTPUT_COLUMNS} for row in output_rows
        )

    return len(output_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop initial rows and independently sort one CSV time column."
    )
    parser.add_argument("input", type=Path, help="Source search_time_summary.csv")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "New output CSV path (default: search_time_summary_draw.csv next to input)"
        ),
    )
    parser.add_argument("--time-column", default=TIME_COLUMN)
    parser.add_argument("--drop-first-rows", type=int, default=DROP_FIRST_ROWS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing valid processed CSV instead of reusing it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output or args.input.with_name(DEFAULT_OUTPUT_NAME)
    row_count = preprocess_file(
        args.input,
        output_path,
        time_column=args.time_column,
        drop_first_rows=args.drop_first_rows,
        force=args.force,
    )
    if row_count == 0:
        print(f"Reused {output_path.expanduser().resolve()} (already preprocessed)")
    else:
        print(f"Wrote {output_path.expanduser().resolve()} ({row_count} row(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
