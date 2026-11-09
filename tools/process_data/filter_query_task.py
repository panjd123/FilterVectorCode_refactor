#!/usr/bin/env python3
"""Create a filtered, non-overwriting copy of a UNG query task."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--destination-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exclude-ids", required=True)
    parser.add_argument("--reason", required=True)
    return parser.parse_args()


def parse_excluded(raw: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values:
        raise ValueError("--exclude-ids must contain at least one QueryID")
    if values[0] < 0:
        raise ValueError("QueryIDs must be non-negative")
    return values


def read_bin(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f"invalid query bin header: {path}")
    count, dimension = struct.unpack_from("<II", data)
    expected = 8 + count * dimension * 4
    if len(data) != expected:
        raise ValueError(f"invalid query bin size: expected {expected}, got {len(data)}")
    return count, dimension, data[8:]


def filter_fixed_records(payload: bytes, record_size: int, keep_ids: list[int]) -> bytes:
    return b"".join(
        payload[index * record_size : (index + 1) * record_size]
        for index in keep_ids
    )


def filter_fvecs(path: Path, count: int, dimension: int, keep_ids: list[int]) -> bytes:
    data = path.read_bytes()
    record_size = 4 + dimension * 4
    if len(data) != count * record_size:
        raise ValueError(
            f"invalid fvecs size: expected {count * record_size}, got {len(data)}"
        )
    for index in range(count):
        actual_dimension = struct.unpack_from("<I", data, index * record_size)[0]
        if actual_dimension != dimension:
            raise ValueError(
                f"fvecs dimension mismatch at QueryID {index}: {actual_dimension} != {dimension}"
            )
    return filter_fixed_records(data, record_size, keep_ids)


def filter_lines(path: Path, count: int, keep_ids: list[int]) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != count:
        raise ValueError(f"expected {count} lines in {path}, got {len(lines)}")
    return [lines[index] for index in keep_ids]


def write_profile(source: Path, destination: Path, count: int, keep_ids: list[int]) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    if len(rows) != count:
        raise ValueError(f"expected {count} profile rows in {source}, got {len(rows)}")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows[index] for index in keep_ids)


def main() -> int:
    args = parse_args()
    source = args.source_dir.resolve()
    destination = args.destination_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing destination: {destination}")

    query_bin = source / f"{args.dataset}_query.bin"
    query_fvecs = source / f"{args.dataset}_query.fvecs"
    query_labels = source / f"{args.dataset}_query_labels.txt"
    count, dimension, bin_payload = read_bin(query_bin)
    excluded = parse_excluded(args.exclude_ids)
    invalid = [query_id for query_id in excluded if query_id >= count]
    if invalid:
        raise ValueError(f"excluded QueryIDs outside [0, {count}): {invalid}")
    excluded_set = set(excluded)
    keep_ids = [query_id for query_id in range(count) if query_id not in excluded_set]
    labels = filter_lines(query_labels, count, keep_ids)
    fvecs_payload = filter_fvecs(query_fvecs, count, dimension, keep_ids)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        filtered_bin = filter_fixed_records(bin_payload, dimension * 4, keep_ids)
        (temporary / query_bin.name).write_bytes(
            struct.pack("<II", len(keep_ids), dimension) + filtered_bin
        )
        (temporary / query_fvecs.name).write_bytes(fvecs_payload)
        (temporary / query_labels.name).write_text(
            "".join(f"{line}\n" for line in labels), encoding="utf-8"
        )

        profiles = sorted(source.glob("profiled_*.csv"))
        if len(profiles) > 1:
            raise ValueError(f"expected at most one profile CSV in {source}: {profiles}")
        if profiles:
            write_profile(profiles[0], temporary / profiles[0].name, count, keep_ids)

        with (temporary / "query_id_map.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["QueryID", "OriginalQueryID"])
            writer.writerows(enumerate(keep_ids))

        manifest = {
            "dataset": args.dataset,
            "source_query_task": source.name,
            "filtered_query_task": destination.name,
            "source_query_count": count,
            "filtered_query_count": len(keep_ids),
            "dimension": dimension,
            "excluded_original_query_ids": excluded,
            "reason": args.reason,
        }
        (temporary / "filter_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        f"created {destination}: {len(keep_ids)} of {count} queries retained; "
        f"excluded {len(excluded)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
