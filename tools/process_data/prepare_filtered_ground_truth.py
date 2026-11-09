#!/usr/bin/env python3
"""Create filtered ground-truth files by exact fixed-record selection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"])
    result_root = Path(config["result_root"])

    for item in config["datasets"]:
        dataset = item["dataset"]
        filtered_task = item["query_task"]
        source_task, separator, _suffix = filtered_task.partition("_exclude_trie_bad_")
        if not separator:
            raise ValueError(f"not a filtered query task: {filtered_task}")

        filtered_data_dir = data_root / dataset / filtered_task
        manifest = json.loads(
            (filtered_data_dir / "filter_manifest.json").read_text(encoding="utf-8")
        )
        source_count = int(manifest["source_query_count"])
        filtered_count = int(manifest["filtered_query_count"])
        with (filtered_data_dir / "query_id_map.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            mapping = [int(row["OriginalQueryID"]) for row in csv.DictReader(handle)]
        if len(mapping) != filtered_count:
            raise ValueError(f"mapping count mismatch: {filtered_data_dir}")

        source_gt = (
            result_root
            / dataset
            / "GroundTruth"
            / source_task
            / f"{dataset}_gt_labels_containment.bin"
        )
        destination_dir = result_root / dataset / "GroundTruth" / filtered_task
        destination_gt = destination_dir / source_gt.name
        if destination_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite existing ground truth: {destination_dir}"
            )

        payload = source_gt.read_bytes()
        if source_count <= 0 or len(payload) % source_count:
            raise ValueError(
                f"source GT size {len(payload)} is not divisible by {source_count}: "
                f"{source_gt}"
            )
        record_size = len(payload) // source_count
        if record_size != 80:
            raise ValueError(
                f"expected K=10 GT record size 80, got {record_size}: {source_gt}"
            )
        filtered_payload = b"".join(
            payload[query_id * record_size : (query_id + 1) * record_size]
            for query_id in mapping
        )
        if len(filtered_payload) != filtered_count * record_size:
            raise AssertionError("filtered GT byte count mismatch")

        destination_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination_dir.name}.", dir=destination_dir.parent)
        )
        try:
            (temporary / source_gt.name).write_bytes(filtered_payload)
            os.replace(temporary, destination_dir)
        except BaseException:
            if temporary.exists():
                for child in temporary.iterdir():
                    child.unlink()
                temporary.rmdir()
            raise
        print(
            f"created {destination_gt}: records={filtered_count}, "
            f"record_size={record_size}, bytes={len(filtered_payload)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
