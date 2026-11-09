#!/usr/bin/env python3
"""Scale and/or offset a numeric column in a CSV file.

The transformation is applied in this order::

    new_value = old_value * multiplier + offset

By default, the result is written beside the source file with ``_adjusted``
appended to its name.  Pass ``--in-place`` only when the source should be
replaced.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable


# =============================================================================
# 直接编辑这里的配置，然后运行：python3 adjust_csv_column.py
# =============================================================================

# 要处理的 CSV 文件。
INPUT_CSV_PATH = Path("/home/dev/graphdb/FilterVectorResult/Laion/results/favor/query_minlen1_sel40to100pct_1000_1000_20000/results/search_time_summary_draw.csv")

# 要修改的列。Average_Time_ms，Average_Recall
COLUMN_NAME = "Average_Time_ms"

# 计算公式：新值 = 原值 * MULTIPLIER + OFFSET
# 例：放大 20% 填 Decimal("1.2")；缩小 20% 填 Decimal("0.8")。
MULTIPLIER = Decimal("1")
# 例：增加 10 填 Decimal("10")；减少 10 填 Decimal("-10")。
OFFSET = Decimal("500")

# None 表示不限制小数位；例如改成 6 表示结果保留 6 位小数。
DECIMAL_PLACES: int | None = None

# True：直接覆盖原文件；False：在原文件旁生成 *_adjusted.csv。
OVERWRITE_INPUT = True
# 不覆盖原文件时，可在这里指定输出路径；None 使用默认的 *_adjusted.csv。
OUTPUT_CSV_PATH: Path | None = None


def decimal_argument(value: str) -> Decimal:
    """Parse a finite decimal number for an argparse option."""
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid decimal value: {value!r}") from exc
    if not number.is_finite():
        raise argparse.ArgumentTypeError(f"value must be finite: {value!r}")
    return number


def adjusted_output_path(input_path: Path) -> Path:
    """Return the default non-destructive output path for *input_path*."""
    return input_path.with_name(f"{input_path.stem}_adjusted{input_path.suffix}")


def format_decimal(value: Decimal, decimal_places: int | None) -> str:
    """Format *value* without introducing binary floating-point error."""
    if decimal_places is not None:
        quantum = Decimal(1).scaleb(-decimal_places)
        value = value.quantize(quantum, rounding=ROUND_HALF_UP)
    return format(value, "f")


def transform_rows(
    rows: Iterable[dict[str, str]],
    *,
    input_path: Path,
    column: str,
    multiplier: Decimal,
    offset: Decimal,
    decimal_places: int | None,
) -> tuple[list[dict[str, str]], int]:
    """Return copied rows with one numeric column transformed."""
    transformed: list[dict[str, str]] = []
    for line_number, source_row in enumerate(rows, start=2):
        row = dict(source_row)
        raw_value = row.get(column, "")
        try:
            old_value = Decimal(raw_value)
        except (InvalidOperation, TypeError) as exc:
            raise ValueError(
                f"{input_path}:{line_number}: invalid {column} value: {raw_value!r}"
            ) from exc
        if not old_value.is_finite():
            raise ValueError(
                f"{input_path}:{line_number}: {column} must contain a finite number"
            )

        new_value = old_value * multiplier + offset
        if not new_value.is_finite():
            raise ValueError(
                f"{input_path}:{line_number}: transformed {column} is not finite"
            )
        row[column] = format_decimal(new_value, decimal_places)
        transformed.append(row)

    return transformed, len(transformed)


def adjust_csv_column(
    input_path: Path,
    output_path: Path,
    *,
    column: str = COLUMN_NAME,
    multiplier: Decimal = Decimal("1"),
    offset: Decimal = Decimal("0"),
    decimal_places: int | None = None,
) -> int:
    """Transform *column*, atomically write the result, and return row count."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    if decimal_places is not None and decimal_places < 0:
        raise ValueError("decimal_places must be non-negative")

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV has no header: {input_path}")
        if column not in fieldnames:
            raise ValueError(f"{input_path}: missing column {column!r}")
        rows, row_count = transform_rows(
            reader,
            input_path=input_path,
            column=column,
            multiplier=multiplier,
            offset=offset,
            decimal_places=decimal_places,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    return row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Modify a numeric CSV column using: new = old * multiplier + offset."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=INPUT_CSV_PATH,
        help=f"Source CSV path (configured default: {INPUT_CSV_PATH})",
    )
    parser.add_argument(
        "--column",
        default=COLUMN_NAME,
        help=f"Numeric column to modify (configured default: {COLUMN_NAME})",
    )
    parser.add_argument(
        "--multiplier",
        type=decimal_argument,
        default=MULTIPLIER,
        help="Multiplication factor; for example, 1.2 increases values by 20%%",
    )
    parser.add_argument(
        "--offset",
        type=decimal_argument,
        default=OFFSET,
        help="Value added after scaling; use a negative number to subtract",
    )
    parser.add_argument(
        "--decimal-places",
        type=int,
        default=DECIMAL_PLACES,
        help="Round results to this many decimal places (ROUND_HALF_UP)",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV_PATH,
        help="Output path (default: INPUT_adjusted.csv)",
    )
    destination.add_argument(
        "--in-place",
        action="store_true",
        default=OVERWRITE_INPUT,
        help="Atomically replace the input CSV instead of creating a new file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.input if args.in_place else (
        args.output or adjusted_output_path(args.input)
    )
    row_count = adjust_csv_column(
        args.input,
        output_path,
        column=args.column,
        multiplier=args.multiplier,
        offset=args.offset,
        decimal_places=args.decimal_places,
    )
    print(
        f"Wrote {output_path.expanduser().resolve()} "
        f"({row_count} row(s), {args.column} = old * {args.multiplier} "
        f"+ {args.offset})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
