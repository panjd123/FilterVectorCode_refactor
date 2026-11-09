#!/usr/bin/env python3
"""Reuse CRoaring FAVOR indexes and rebuild obsolete formats for eight datasets.

Uses the existing experiment runner's builder, timing and CSV bookkeeping.
Does not run queries. Obsolete indexes are replaced at their existing paths.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from summarize_index_build import DATASETS


def main():
    workspace = Path(__file__).resolve().parents[3]
    project = workspace / "FilterVectorCode_multilevel_special"
    sys.path.insert(0, str(project / "experiments/favor"))
    import run_favor_experiment as favor

    cfg = dict(data_root=str(workspace / "FilterVectorData"),
               result_root=str(workspace / "FilterVectorResult"),
               index_name="FAVOR", build=dict(num_threads=60, M=64, ef_construction=200))
    build_bin = project / "FAVOR/build/app/build_index"
    output = workspace / "FilterVectorResult/index_build_summary/favor_roaring_migration.json"
    records = json.loads(output.read_text())["datasets"] if output.exists() else []
    for dataset in DATASETS:
        paths = favor.resolve_dataset_files(cfg, dict(dataset=dataset, query_task="query_minlen1_sel0to10pct"))
        index = paths["index_file"]
        with index.open("rb") as handle:
            before = handle.read(8).decode("ascii", errors="replace")
        rebuild = not favor.is_current_favor_index(index) or not paths["label_schema_file"].exists()
        if rebuild:
            print(f"Rebuilding {dataset}/FAVOR from {before} to FAVROA01", flush=True)
        favor.build_index(build_bin, cfg, paths)
        assert favor.is_current_favor_index(index)
        assert paths["label_schema_file"].is_file()
        if not any(row["dataset"] == dataset for row in records):
            records.append(dict(dataset=dataset, previous_format=before, current_format="FAVROA01",
                                rebuilt=rebuild, index_file=str(index), bytes=index.stat().st_size,
                                num_threads=60, M=64, ef_construction=200,
                                completed_at_utc=datetime.now(timezone.utc).isoformat()))
        output.write_text(json.dumps(dict(datasets=records), indent=2) + "\n")
    print("Verified all eight FAVOR indexes use CRoaring (FAVROA01).", flush=True)


if __name__ == "__main__":
    main()
