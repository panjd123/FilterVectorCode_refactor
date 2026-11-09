#!/usr/bin/env python3
"""Summarize the eight datasets and nine algorithms used by the comparison plots.

Disk size is the sum of file lengths in index_files and (for PTNC)
block_index_files, including vectors, metadata and retained duplicate files.
ACORN-1 artifacts are excluded because the plots use the ACORN variant.
This script only reads existing results; it does not build or load indexes.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


DATASETS = ("Genome", "Reviews", "Amazon", "VImg", "Music", "BReviews", "Tiktok", "Laion")
ALGORITHMS = {
    "Milvus": "Milvus_HNSW_default",
    "ACORN": "ACORN",
    "NaviX": "NaviX",
    "RWalks": "RWalks_default",
    "PathSeer": "PathSeer_M64_Mexp128",
    "Curator": "Curator",
    "UNG": "UNG",
    "FAVOR": "FAVOR",
    "PTNC": "Trie_block_500_M_32_cross_block_4",
}
GIB = 1024 ** 3


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def metadata(path, separator="="):
    return dict(line.split(separator, 1) for line in path.read_text().splitlines()
                if separator in line)


def write_csv(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(root):
    records, inventory = [], []
    for dataset in DATASETS:
        for algorithm, directory in ALGORITHMS.items():
            base = root / dataset / "index" / directory
            if algorithm == "FAVOR":
                with (base / "index_files/index.bin").open("rb") as handle:
                    if handle.read(8) != b"FAVROA01":
                        raise ValueError(f"FAVOR must use CRoaring; rebuild obsolete index: {base}")
            folders = [base / "index_files"]
            if algorithm == "PTNC":
                folders.append(base / "block_index_files")
            disk_bytes = 0
            for folder in folders:
                files = sorted(path for path in folder.rglob("*") if path.is_file())
                if not files:
                    raise ValueError(f"No index files found: {folder}")
                for path in files:
                    length = path.stat().st_size
                    included = not (algorithm == "ACORN" and path.name.startswith("acorn1.index"))
                    if included:
                        disk_bytes += length
                    inventory.append(dict(dataset=dataset, algorithm=algorithm,
                                          path=str(path), bytes=length,
                                          is_symlink=path.is_symlink(), included=included))
            row = dict(dataset=dataset, algorithm=algorithm, index_directory=str(base),
                       disk_bytes=disk_bytes, disk_gib=disk_bytes / GIB,
                       build_time_seconds=0.0, base_build_seconds="", block_build_seconds="",
                       pipeline_wall_seconds="", reported_memory_bytes="",
                       reported_memory_gib="", structure_memory_mib="",
                       structure_with_roaring_memory_mib="", block_loaded_memory_bytes="",
                       build_record_count=1, build_time_source="", memory_source="", notes="")
            if algorithm in ("Milvus", "RWalks", "PathSeer", "FAVOR"):
                source = base / "others/index_build_time.csv"
                builds = csv_rows(source)
                latest = builds[-1]
                assert latest["dataset"] == dataset
                row.update(build_time_seconds=float(latest["build_time_seconds"]),
                           reported_memory_bytes=int(latest["index_memory_bytes"]),
                           build_record_count=len(builds), memory_source=str(source))
                if len(builds) > 1:
                    row["notes"] = "Last appended build record selected; earlier builds retained in source CSV. "
                if algorithm == "FAVOR":
                    log = (base / "others/favor_build.log").read_text()
                    if "label roaring schema" not in log:
                        raise ValueError(f"FAVOR build log does not match its CRoaring index: {base}")
                    row["notes"] += "Current FAVOR attribute schema: CRoaring (FAVROA01)."
            elif algorithm == "ACORN":
                source = base / "index_files/acorn.index.meta"
                info = metadata(source, ":")
                assert int(info["total_size_bytes"]) == (base / "index_files/acorn.index").stat().st_size
                row["build_time_seconds"] = float(info["build_time_s"])
                row["notes"] = "Build time from persisted ACORN metadata; serialized bytes are not runtime memory."
            elif algorithm == "Curator":
                source = base / "index_files/curator_meta.json"
                info = json.loads(source.read_text())
                row.update(build_time_seconds=info["build_ms"] / 1000,
                           reported_memory_bytes=info["memory_bytes"], memory_source=str(source))
                row["notes"] = "Original persisted build_ms used even when later searches reused the index."
            elif algorithm == "NaviX":
                source = base / "index_files/meta"
                row["build_time_seconds"] = float(metadata(source)["index_time(ms)"]) / 1000
            else:
                source = base / "index_files/meta"
                info = metadata(source)
                row["build_time_seconds"] = float(info["index_time_add_rb(ms)"]) / 1000
                row["structure_memory_mib"] = float(info["index_size(MB)"])
                row["structure_with_roaring_memory_mib"] = float(info["_index_size_add_rb(MB)"])
                row["notes"] = "Base structure memory counters exclude vector storage; not full runtime memory."
                if algorithm == "PTNC":
                    block_source = base / "block_index_files/meta"
                    block_info = metadata(block_source)
                    row.update(base_build_seconds=row["build_time_seconds"],
                               block_build_seconds=float(block_info["build_time(ms)"]) / 1000,
                               block_loaded_memory_bytes=int(block_info["loaded_memory_allocated_bytes"]))
                    row["build_time_seconds"] += row["block_build_seconds"]
                    timing = metadata(base / "others/time.txt")
                    row["pipeline_wall_seconds"] = float(timing["pipeline_wall_seconds"])
                    row["notes"] += " PTNC time = base (including Roaring) + independent block build; disk includes both folders."
                    source = f"{source};{block_source}"
                row["memory_source"] = str(source)
            row["build_time_source"] = str(source)
            if row["reported_memory_bytes"] != "":
                row["reported_memory_gib"] = row["reported_memory_bytes"] / GIB
            assert row["disk_bytes"] > 0 and row["build_time_seconds"] > 0
            records.append(row)
    assert len(records) == len(DATASETS) * len(ALGORITHMS) == 72
    return records, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parents[3]
    parser.add_argument("--result-root", type=Path, default=workspace / "FilterVectorResult")
    parser.add_argument("--output-dir", type=Path,
                        default=workspace / "FilterVectorResult/index_build_summary")
    args = parser.parse_args()
    records, inventory = collect(args.result_root.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "index_build_details.csv", records)
    write_csv(output / "index_file_inventory.csv", inventory)
    for name, metric in (("index_size_gib.csv", "disk_gib"),
                         ("build_time_seconds.csv", "build_time_seconds"),
                         ("reported_memory_gib.csv", "reported_memory_gib")):
        matrix = [dict(dataset=dataset, **{
            r["algorithm"]: (round(r[metric], 6) if r[metric] != "" else "")
            for r in records if r["dataset"] == dataset}) for dataset in DATASETS]
        write_csv(output / name, matrix)
    (output / "methodology.json").write_text(json.dumps(dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        result_root=str(args.result_root.resolve()), datasets=DATASETS, algorithms=ALGORITHMS,
        record_count=len(records),
        size_definition="Sum of stat().st_size for files in index_files and PTNC block_index_files, excluding acorn1.index* artifacts (different variant). Includes vectors, labels, metadata and retained duplicate vector representations. Excludes logs/results and filesystem block overhead. Symlink targets counted once per path; not deduplicated.",
        units={"GiB": "2^30 bytes", "MiB": "2^20 bytes", "build_time": "seconds"},
        time_definition="Existing algorithm build timers; not standardized end-to-end wall time. Prefer persisted metadata precision. PTNC sums base index_time_add_rb(ms) and independent block build_time(ms). ACORN uses persisted build_time_s. No averaging across rebuilds.",
        memory_definition="Only directly reported full-index estimates populate reported_memory_* (40/72 records). ACORN/NaviX have no full runtime counter. UNG/PTNC base counters exclude vectors; PTNC block counter describes the sidecar only, so these remain separate columns. These counters are not process RSS.",
        caveats=["ACORN log overrides requested build threads to 32, while several other algorithms use 60; existing time comparisons have differing thread counts.",
                 "FAVOR attribute format is identified from each current build log; check per-row notes. The requested configuration is CRoaring for all eight datasets.",
                 "UNG/PTNC directories retain vecs.bin and reordered_vecs.fvecs; disk totals include both."]
    ), ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(records)} dataset/algorithm records and {len(inventory)} file records to {output}")
    for dataset in DATASETS:
        print(dataset + ": " + ", ".join(
            f"{r['algorithm']}={r['disk_gib']:.3f} GiB/{r['build_time_seconds']:.2f} s"
            for r in records if r["dataset"] == dataset))


if __name__ == "__main__":
    main()
