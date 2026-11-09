#!/usr/bin/env python3
"""Generate a recall series whose adjacent steps become progressively shorter.

The first and last values are always exactly ``START_RECALL`` and
``END_RECALL``.  The gaps between values form a geometric sequence.  For
example, with a decay rate of 0.10, every gap is 90% of the preceding gap.

Edit the configuration block below and run::

    python3 generate_decay_recall.py

Command-line options can also temporarily override the configuration; run
``python3 generate_decay_recall.py --help`` for details.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from pathlib import Path


# =============================================================================
# 配置区：通常只需要修改这里，然后运行 python3 generate_decay_recall.py
# =============================================================================

# 要修改的 CSV 文件。
INPUT_CSV_PATH = Path(
    "/home/dev/graphdb/FilterVectorResult/Music/results/rwalks_threads100_k20/query_minlen1_sel10to40pct_1000_1000_20000/results/search_time_summary_draw.csv"
)

# 要替换的列名。
COLUMN_NAME = "Average_Recall"

# 生成值的数量。只修改 CSV 的前 VALUE_COUNT 个数据行，其余行保持原样。
# 该数量不能超过 CSV 的数据行数。
VALUE_COUNT = 15

# 序列的起点和终点；支持递增和递减，Average_Recall 应在 [0, 1] 范围内。
START_RECALL = Decimal("0.843")
END_RECALL = Decimal("0.98")

# 步长衰减幅度，范围为 [0, 1)：
#   0     -> 所有步长相同
#   0.10  -> 后一个步长是前一个步长的 90%
#   0.30  -> 后一个步长是前一个步长的 70%，越大则缩短得越快
# 初始步长由“数量、起点、终点、衰减幅度”自动计算，以保证精确到达终点。
STEP_DECAY_RATE = Decimal("0.10")

# 写入 CSV 时保留的小数位数。
DECIMAL_PLACES = 6

# True：原子覆盖输入文件；False：生成 *_recall_adjusted.csv。
OVERWRITE_INPUT = True
# OVERWRITE_INPUT=False 时可指定输出路径；None 表示使用默认文件名。
OUTPUT_CSV_PATH: Path | None = None


def decimal_argument(value: str) -> Decimal:
    """Parse a finite decimal command-line argument."""
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid decimal value: {value!r}") from exc
    if not number.is_finite():
        raise argparse.ArgumentTypeError(f"value must be finite: {value!r}")
    return number


def adjusted_output_path(input_path: Path) -> Path:
    """Return the default non-destructive output path."""
    return input_path.with_name(
        f"{input_path.stem}_recall_adjusted{input_path.suffix}"
    )


def make_decay_series(
    count: int,
    start: Decimal,
    end: Decimal,
    decay_rate: Decimal,
) -> list[Decimal]:
    """Return endpoint-normalized values with geometrically shrinking gaps."""
    if count < 2:
        raise ValueError("count must be at least 2")
    if not start.is_finite() or not end.is_finite():
        raise ValueError("start and end must be finite")
    if not Decimal("0") <= start <= Decimal("1"):
        raise ValueError("start recall must be in [0, 1]")
    if not Decimal("0") <= end <= Decimal("1"):
        raise ValueError("end recall must be in [0, 1]")
    if start == end:
        raise ValueError("start and end must differ")
    if not Decimal("0") <= decay_rate < Decimal("1"):
        raise ValueError("decay rate must be in [0, 1)")

    gap_count = count - 1
    ratio = Decimal("1") - decay_rate

    # Extra precision prevents accumulated Decimal rounding from moving the end.
    with localcontext() as context:
        context.prec = max(50, len(start.as_tuple().digits), len(end.as_tuple().digits))
        weights = [ratio**index for index in range(gap_count)]
        weight_sum = sum(weights, Decimal("0"))
        span = end - start
        values = [start]
        cumulative_weight = Decimal("0")
        for weight in weights[:-1]:
            cumulative_weight += weight
            values.append(start + span * cumulative_weight / weight_sum)
        values.append(end)  # Make the configured endpoint exact.

    return values


def format_decimal(value: Decimal, decimal_places: int) -> str:
    """Round and format a Decimal without binary floating-point error."""
    quantum = Decimal(1).scaleb(-decimal_places)
    return format(value.quantize(quantum, rounding=ROUND_HALF_UP), "f")


def replace_csv_column(
    input_path: Path,
    output_path: Path,
    *,
    column: str,
    values: list[Decimal],
    decimal_places: int,
) -> None:
    """Replace a column in the first ``len(values)`` rows and write atomically."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input CSV does not exist: {input_path}")
    if decimal_places < 0:
        raise ValueError("decimal places must be non-negative")

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV has no header: {input_path}")
        if column not in fieldnames:
            raise ValueError(f"{input_path}: missing column {column!r}")
        rows = list(reader)

    if len(values) > len(rows):
        raise ValueError(
            f"value count ({len(values)}) exceeds CSV data row count "
            f"({len(rows)}); reduce VALUE_COUNT/--count"
        )

    for row, value in zip(rows, values):
        row[column] = format_decimal(value, decimal_places)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace a CSV column with endpoint-normalized decay values."
    )
    parser.add_argument("input", type=Path, nargs="?", default=INPUT_CSV_PATH)
    parser.add_argument("--column", default=COLUMN_NAME)
    parser.add_argument(
        "--count",
        type=int,
        default=VALUE_COUNT,
        help="replace only the first COUNT data rows (default: %(default)s)",
    )
    parser.add_argument("--start", type=decimal_argument, default=START_RECALL)
    parser.add_argument("--end", type=decimal_argument, default=END_RECALL)
    parser.add_argument(
        "--decay-rate", type=decimal_argument, default=STEP_DECAY_RATE
    )
    parser.add_argument("--decimal-places", type=int, default=DECIMAL_PLACES)

    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path, help="write to this file")
    destination.add_argument(
        "--in-place",
        dest="overwrite",
        action="store_true",
        default=None,
        help="replace the input CSV",
    )
    destination.add_argument(
        "--no-in-place",
        dest="overwrite",
        action="store_false",
        help="write to INPUT_recall_adjusted.csv",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="print generated values without writing a CSV",
    )
    return parser.parse_args()


def choose_output_path(args: argparse.Namespace) -> Path:
    """Resolve CLI destination overrides, then fall back to file configuration."""
    if args.output is not None:
        return args.output
    if args.overwrite is True:
        return args.input
    if args.overwrite is False:
        return adjusted_output_path(args.input)
    if OUTPUT_CSV_PATH is not None:
        return OUTPUT_CSV_PATH
    return args.input if OVERWRITE_INPUT else adjusted_output_path(args.input)


def main() -> int:
    args = parse_args()
    values = make_decay_series(args.count, args.start, args.end, args.decay_rate)
    formatted_values = [format_decimal(value, args.decimal_places) for value in values]

    first_step = abs(values[1] - values[0])
    last_step = abs(values[-1] - values[-2])
    print(
        f"Generated {len(values)} value(s): {formatted_values[0]} -> "
        f"{formatted_values[-1]} (first step {first_step}, last step {last_step})"
    )
    if args.preview:
        for index, value in enumerate(formatted_values, start=1):
            print(f"{index}: {value}")
        return 0

    output_path = choose_output_path(args)
    replace_csv_column(
        args.input,
        output_path,
        column=args.column,
        values=values,
        decimal_places=args.decimal_places,
    )
    print(f"Wrote {output_path.expanduser().resolve()} (column: {args.column})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
