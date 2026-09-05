#!/usr/bin/env python3
"""Compare UNG MinSupersetT_ms against gpu_bruteforce_els_ung per query."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_GPU_CSV = Path(
    "/home/dev/graphdb/FilterVectorResult/Amazon/results/gpu_bruteforce_els_ung/"
    "zipf/query_minlen1_cov20k_1000_1000_20000/results/query_details_repeat1.csv"
)
DEFAULT_UNG_CSV = Path(
    "/home/dev/graphdb/FilterVectorResult/Amazon/results/UNG/"
    "zipf/query_minlen1_cov20k_1000_1000_20000/results/query_details_repeat1.csv"
)
DEFAULT_SOURCE_QUERY_DIR = Path(
    "/home/dev/graphdb/FilterVectorData/Amazon/zipf_query/query_minlen1_cov20k"
)

KEY_COLUMNS = ("repeat", "Lsearch", "efs", "QueryID")
REQUIRED_COLUMNS = (*KEY_COLUMNS, "Time_ms", "MinSupersetT_ms")


@dataclass(frozen=True)
class DetailRow:
    repeat: int
    lsearch: int
    efs: int
    query_id: int
    ung_time_ms: float
    ung_min_superset_ms: float
    ung_min_superset_share: float
    gpu_time_ms: float
    gpu_min_superset_ms: float
    gpu_min_superset_share: float
    min_superset_speedup: float


@dataclass(frozen=True)
class SummaryRow:
    lsearch: int
    matched_queries: int
    gpu_smaller_queries: int
    gpu_smaller_large_share_queries: int
    gpu_smaller_pct: float
    gpu_smaller_large_share_pct: float
    ung_share_mean: float
    ung_share_median: float
    gpu_share_mean: float
    gpu_share_median: float
    min_superset_speedup_mean: float
    min_superset_speedup_median: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each Lsearch, count queries where gpu_bruteforce_els_ung has "
            "smaller MinSupersetT_ms than UNG, and where that stage is a large "
            "share of Time_ms."
        )
    )
    parser.add_argument("--gpu-csv", type=Path, default=DEFAULT_GPU_CSV)
    parser.add_argument("--ung-csv", type=Path, default=DEFAULT_UNG_CSV)
    parser.add_argument(
        "--min-share",
        type=float,
        default=0.30,
        help="Large-share threshold for MinSupersetT_ms / Time_ms. Default: 0.30.",
    )
    parser.add_argument(
        "--share-side",
        choices=("ung", "gpu", "both", "either"),
        default="ung",
        help=(
            "Which algorithm's MinSupersetT_ms/Time_ms share must meet --min-share. "
            "Default: ung."
        ),
    )
    parser.add_argument(
        "--details-out",
        type=Path,
        default=None,
        help="Optional CSV path for query rows satisfying both conditions.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional CSV path for per-Lsearch summary rows.",
    )
    parser.add_argument("--dataset", default="Amazon", help="Dataset filename prefix. Default: Amazon.")
    parser.add_argument(
        "--source-query-dir",
        type=Path,
        default=DEFAULT_SOURCE_QUERY_DIR,
        help="Directory containing <dataset>_query.bin/.fvecs/_labels.txt for source QueryID rows.",
    )
    parser.add_argument(
        "--materialize-lsearch",
        type=int,
        default=0,
        help="If set, build a new query task from matching rows for this Lsearch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output query task directory used with --materialize-lsearch.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def to_float(row: dict[str, str], column: str) -> float:
    try:
        return float(row.get(column, "") or 0.0)
    except ValueError:
        return 0.0


def to_int(row: dict[str, str], column: str) -> int:
    try:
        return int(float(row.get(column, "") or 0))
    except ValueError:
        return 0


def load_rows(path: Path) -> dict[tuple[int, int, int, int], dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} missing required columns: {', '.join(missing)}")

        rows = {}
        for row in reader:
            key = tuple(to_int(row, column) for column in KEY_COLUMNS)
            rows[key] = row
        return rows


def min_superset_share(row: dict[str, str]) -> float:
    time_ms = to_float(row, "Time_ms")
    if time_ms <= 0:
        return 0.0
    return to_float(row, "MinSupersetT_ms") / time_ms


def share_is_large(ung_share: float, gpu_share: float, min_share: float, share_side: str) -> bool:
    if share_side == "ung":
        return ung_share >= min_share
    if share_side == "gpu":
        return gpu_share >= min_share
    if share_side == "both":
        return ung_share >= min_share and gpu_share >= min_share
    if share_side == "either":
        return ung_share >= min_share or gpu_share >= min_share
    raise ValueError(f"unknown share side: {share_side}")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def analyze(
    gpu_csv: Path,
    ung_csv: Path,
    min_share: float = 0.30,
    share_side: str = "ung",
) -> tuple[list[SummaryRow], list[DetailRow]]:
    gpu_rows = load_rows(gpu_csv)
    ung_rows = load_rows(ung_csv)
    common_keys = sorted(set(gpu_rows) & set(ung_rows), key=lambda key: (key[1], key[0], key[2], key[3]))

    matched_by_lsearch: dict[int, int] = defaultdict(int)
    smaller_by_lsearch: dict[int, list[DetailRow]] = defaultdict(list)
    large_share_details: list[DetailRow] = []

    for key in common_keys:
        repeat, lsearch, efs, query_id = key
        gpu = gpu_rows[key]
        ung = ung_rows[key]
        matched_by_lsearch[lsearch] += 1

        gpu_min = to_float(gpu, "MinSupersetT_ms")
        ung_min = to_float(ung, "MinSupersetT_ms")
        if not gpu_min < ung_min:
            continue

        ung_share = min_superset_share(ung)
        gpu_share = min_superset_share(gpu)
        detail = DetailRow(
            repeat=repeat,
            lsearch=lsearch,
            efs=efs,
            query_id=query_id,
            ung_time_ms=to_float(ung, "Time_ms"),
            ung_min_superset_ms=ung_min,
            ung_min_superset_share=ung_share,
            gpu_time_ms=to_float(gpu, "Time_ms"),
            gpu_min_superset_ms=gpu_min,
            gpu_min_superset_share=gpu_share,
            min_superset_speedup=ung_min / gpu_min if gpu_min > 0 else 0.0,
        )
        smaller_by_lsearch[lsearch].append(detail)
        if share_is_large(ung_share, gpu_share, min_share, share_side):
            large_share_details.append(detail)

    large_by_lsearch: dict[int, list[DetailRow]] = defaultdict(list)
    for detail in large_share_details:
        large_by_lsearch[detail.lsearch].append(detail)

    summary = []
    for lsearch in sorted(matched_by_lsearch):
        matched = matched_by_lsearch[lsearch]
        smaller = smaller_by_lsearch[lsearch]
        large = large_by_lsearch[lsearch]
        summary.append(
            SummaryRow(
                lsearch=lsearch,
                matched_queries=matched,
                gpu_smaller_queries=len(smaller),
                gpu_smaller_large_share_queries=len(large),
                gpu_smaller_pct=len(smaller) / matched if matched else 0.0,
                gpu_smaller_large_share_pct=len(large) / matched if matched else 0.0,
                ung_share_mean=mean(row.ung_min_superset_share for row in large),
                ung_share_median=median(row.ung_min_superset_share for row in large),
                gpu_share_mean=mean(row.gpu_min_superset_share for row in large),
                gpu_share_median=median(row.gpu_min_superset_share for row in large),
                min_superset_speedup_mean=mean(row.min_superset_speedup for row in large),
                min_superset_speedup_median=median(row.min_superset_speedup for row in large),
            )
        )
    return summary, large_share_details


def summary_fieldnames() -> list[str]:
    return [field.name for field in SummaryRow.__dataclass_fields__.values()]


def detail_fieldnames() -> list[str]:
    return [field.name for field in DetailRow.__dataclass_fields__.values()]


def write_summary(path: Path, summary: list[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames())
        writer.writeheader()
        for row in summary:
            writer.writerow(row.__dict__)


def read_labels(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def read_bin_rows(path: Path, row_indices: list[int]) -> tuple[int, dict[int, bytes]]:
    wanted = sorted(set(row_indices))
    rows: dict[int, bytes] = {}
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"bad bin header: {path}")
        n, dim = struct.unpack("<II", header)
        record_bytes = dim * 4
        for idx in wanted:
            if idx < 0 or idx >= n:
                raise IndexError(f"row {idx} out of range for {path} with n={n}")
            f.seek(8 + idx * record_bytes)
            payload = f.read(record_bytes)
            if len(payload) != record_bytes:
                raise ValueError(f"short read in {path} at row {idx}")
            rows[idx] = payload
    return dim, rows


def read_fvecs_rows(path: Path, row_indices: list[int]) -> tuple[int, dict[int, bytes]]:
    wanted = set(row_indices)
    rows: dict[int, bytes] = {}
    dim = None
    with path.open("rb") as f:
        row_no = 0
        while wanted - rows.keys():
            header = f.read(4)
            if not header:
                break
            if len(header) != 4:
                raise ValueError(f"truncated fvecs header in {path}")
            (cur_dim,) = struct.unpack("<I", header)
            payload = f.read(cur_dim * 4)
            if len(payload) != cur_dim * 4:
                raise ValueError(f"truncated fvecs payload in {path}")
            if dim is None:
                dim = cur_dim
            elif cur_dim != dim:
                raise ValueError(f"inconsistent fvecs dim in {path}")
            if row_no in wanted:
                rows[row_no] = payload
            row_no += 1
    missing = wanted - rows.keys()
    if missing:
        raise ValueError(f"{path} missing rows: {sorted(missing)[:10]}")
    if dim is None:
        raise ValueError(f"empty fvecs file: {path}")
    return dim, rows


def default_output_dir(source_dir: Path, lsearch: int, min_share: float) -> Path:
    share_tag = int(round(min_share * 100))
    return source_dir.parent / f"{source_dir.name}_gpu_els_min_superset_lsearch{lsearch}_share{share_tag}"


def materialize_query_task(
    details: list[DetailRow],
    output_dir: Path,
    *,
    source_dir: Path,
    dataset: str,
    gpu_csv: Path,
    ung_csv: Path,
    min_share: float,
    share_side: str,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_in = source_dir / f"{dataset}_query_labels.txt"
    bin_in = source_dir / f"{dataset}_query.bin"
    fvecs_in = source_dir / f"{dataset}_query.fvecs"
    labels_out = output_dir / f"{dataset}_query_labels.txt"
    bin_out = output_dir / f"{dataset}_query.bin"
    fvecs_out = output_dir / f"{dataset}_query.fvecs"
    selected_out = output_dir / "selected_queries.csv"
    sources_out = output_dir / "selected_sources.csv"
    manifest_out = output_dir / "selection_manifest.json"

    outputs = [labels_out, bin_out, fvecs_out, selected_out, sources_out, manifest_out]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("output exists; use --overwrite: " + ", ".join(existing))
    if not details:
        raise ValueError("no selected rows to materialize")
    for required in [labels_in, bin_in, fvecs_in]:
        if not required.exists():
            raise FileNotFoundError(required)

    source_ids = [row.query_id for row in details]
    labels = read_labels(labels_in)
    for idx in source_ids:
        if idx < 0 or idx >= len(labels):
            raise IndexError(f"label row {idx} out of range for {labels_in}")
    bin_dim, bin_rows = read_bin_rows(bin_in, source_ids)
    fvecs_dim, fvecs_rows = read_fvecs_rows(fvecs_in, source_ids)
    if bin_dim != fvecs_dim:
        raise ValueError(f"dimension mismatch: bin={bin_dim}, fvecs={fvecs_dim}")

    selected_fields = ["new_query_id", "source_query_id", *detail_fieldnames()]
    with selected_out.with_suffix(".csv.tmp").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected_fields)
        writer.writeheader()
        for new_query_id, row in enumerate(details):
            out = {"new_query_id": new_query_id, "source_query_id": row.query_id, **row.__dict__}
            writer.writerow(out)
    shutil.move(str(selected_out.with_suffix(".csv.tmp")), str(selected_out))

    with sources_out.with_suffix(".csv.tmp").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "new_query_id",
                "source_query_id",
                "source_query_bin",
                "source_query_fvecs",
                "source_query_labels",
            ],
        )
        writer.writeheader()
        for new_query_id, row in enumerate(details):
            writer.writerow(
                {
                    "new_query_id": new_query_id,
                    "source_query_id": row.query_id,
                    "source_query_bin": str(bin_in),
                    "source_query_fvecs": str(fvecs_in),
                    "source_query_labels": str(labels_in),
                }
            )
    shutil.move(str(sources_out.with_suffix(".csv.tmp")), str(sources_out))

    with labels_out.with_suffix(".txt.tmp").open("w", encoding="utf-8") as labels_f, bin_out.with_suffix(
        ".bin.tmp"
    ).open("wb") as bin_f, fvecs_out.with_suffix(".fvecs.tmp").open("wb") as fvecs_f:
        bin_f.write(struct.pack("<II", len(details), bin_dim))
        for row in details:
            source_id = row.query_id
            labels_f.write(labels[source_id] + "\n")
            bin_f.write(bin_rows[source_id])
            fvecs_f.write(struct.pack("<I", fvecs_dim))
            fvecs_f.write(fvecs_rows[source_id])
    shutil.move(str(labels_out.with_suffix(".txt.tmp")), str(labels_out))
    shutil.move(str(bin_out.with_suffix(".bin.tmp")), str(bin_out))
    shutil.move(str(fvecs_out.with_suffix(".fvecs.tmp")), str(fvecs_out))

    manifest = {
        "dataset": dataset,
        "num_queries": len(details),
        "source_query_dir": str(source_dir),
        "output_dir": str(output_dir),
        "gpu_csv": str(gpu_csv),
        "ung_csv": str(ung_csv),
        "selection": {
            "lsearch": details[0].lsearch if details else None,
            "min_share": min_share,
            "share_side": share_side,
            "predicate": "gpu.MinSupersetT_ms < ung.MinSupersetT_ms and selected share >= min_share",
        },
        "output_files": {
            "query_bin": str(bin_out),
            "query_fvecs": str(fvecs_out),
            "query_labels": str(labels_out),
            "selected_queries": str(selected_out),
            "selected_sources": str(sources_out),
        },
    }
    manifest_out.with_suffix(".json.tmp").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.move(str(manifest_out.with_suffix(".json.tmp")), str(manifest_out))


def write_details(path: Path, details: list[DetailRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = detail_fieldnames()
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in details:
            writer.writerow(row.__dict__)


def print_summary(summary: list[SummaryRow], details: list[DetailRow], min_share: float, share_side: str) -> None:
    print(f"large-share threshold: MinSupersetT_ms / Time_ms >= {min_share:.2%} on {share_side}")
    print(f"total matched large-share gpu-smaller queries: {len(details)}")
    print()
    header = (
        "Lsearch matched gpu_smaller gpu_smaller_large_share "
        "gpu_smaller_large_share_pct ung_share_mean ung_share_median "
        "gpu_share_mean gpu_share_median speedup_mean speedup_median"
    )
    print(header)
    for row in summary:
        print(
            f"{row.lsearch} "
            f"{row.matched_queries} "
            f"{row.gpu_smaller_queries} "
            f"{row.gpu_smaller_large_share_queries} "
            f"{row.gpu_smaller_large_share_pct:.2%} "
            f"{row.ung_share_mean:.4f} "
            f"{row.ung_share_median:.4f} "
            f"{row.gpu_share_mean:.4f} "
            f"{row.gpu_share_median:.4f} "
            f"{row.min_superset_speedup_mean:.2f} "
            f"{row.min_superset_speedup_median:.2f}"
        )


def main() -> int:
    args = parse_args()
    summary, details = analyze(args.gpu_csv, args.ung_csv, args.min_share, args.share_side)
    print_summary(summary, details, args.min_share, args.share_side)
    if args.details_out:
        write_details(args.details_out, details)
        print(f"\nwrote details: {args.details_out}")
    if args.summary_out:
        write_summary(args.summary_out, summary)
        print(f"wrote summary: {args.summary_out}")
    if args.materialize_lsearch:
        selected = [row for row in details if row.lsearch == args.materialize_lsearch]
        output_dir = args.output_dir or default_output_dir(args.source_query_dir, args.materialize_lsearch, args.min_share)
        materialize_query_task(
            selected,
            output_dir,
            source_dir=args.source_query_dir,
            dataset=args.dataset,
            gpu_csv=args.gpu_csv,
            ung_csv=args.ung_csv,
            min_share=args.min_share,
            share_side=args.share_side,
            overwrite=args.overwrite,
        )
        print(f"wrote query task: {output_dir}")
        print(f"materialized queries: {len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
