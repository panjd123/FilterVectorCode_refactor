#!/usr/bin/env python3
"""Generate the nine-method campaign by changing only query-task entries.

Usage: python3 experiments/query_label_length/generate_configs.py
Existing generated files in this directory are regenerated from current templates.
Each distinct source-task parameter profile is applied to BOTH length groups.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
PROJECT = BUNDLE.parents[1]
DATASETS = ("Amazon", "VImg", "Music")
SOURCE_TASKS = ("query_minlen1_sel0to10pct", "query_minlen1_sel10to40pct")
TASKS = ("query_len1to3_sel0to40pct", "query_len4to6_sel0to40pct")
# Keys are the command-line method names; values identify existing entry points.
METHODS = {
    "trie_block": ("search_comparison", "run_search_comparison.py"),
    "ung": ("search_comparison", "run_search_comparison.py"),
    "acorn": ("acorn_baseline", "run_acorn_baseline.py"),
    "favor": ("favor", "run_favor_experiment.py"),
    "milvus_hnsw": ("milvus_hnsw", "run_milvus_hnsw_experiment.py"),
    "navix": ("navix_baseline", "run_navix_baseline.py"),
    "pathseer": ("pathseer", "run_pathseer_experiment.py"),
    "rwalks": ("rwalks", "run_rwalks_experiment.py"),
    "curator": ("curator_baseline", "run_curator_baseline_with_env.sh"),
}


def source_path(dataset: str, method: str) -> Path:
    directory = METHODS[method][0]
    if method in ("trie_block", "ung"):
        label = dataset if dataset == "VImg" else dataset.lower()
        suffix = "_ung" if method == "ung" else ""
        filename = f"config_{label}{suffix}.json"
    else:
        filename = f"{dataset}_config.json"
    return PROJECT / "experiments" / directory / filename


def query_profiles(template: dict, dataset: str) -> list[dict]:
    """Preserve every distinct 0-10%/10-40% row apart from its task name."""
    profiles = []
    found = set()
    for row in template["datasets"]:
        if row["dataset"] != dataset or row["query_task"] not in SOURCE_TASKS:
            continue
        found.add(row["query_task"])
        settings = {k: copy.deepcopy(v) for k, v in row.items() if k != "query_task"}
        match = next((p for p in profiles if p["settings"] == settings), None)
        if match is None:
            profiles.append(dict(settings=settings, source_tasks=[row["query_task"]]))
        else:
            match["source_tasks"].append(row["query_task"])
    if found != set(SOURCE_TASKS):
        raise ValueError(f"Missing source tasks for {dataset}: {found}")
    return profiles


def transform(template: dict, dataset: str) -> tuple[dict, list[dict]]:
    profiles = query_profiles(template, dataset)
    generated = copy.deepcopy(template)
    generated["datasets"] = []
    for profile in profiles:
        for task in TASKS:
            generated["datasets"].append(dict(copy.deepcopy(profile["settings"]), query_task=task))
    return generated, profiles


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    manifest = dict(datasets=list(DATASETS), methods=list(METHODS), query_tasks=list(TASKS),
                    parameter_policy="Only datasets/query_task entries change; both length groups use every distinct source parameter profile.",
                    jobs=[])
    for dataset in DATASETS:
        for method, (directory, entry) in METHODS.items():
            source = source_path(dataset, method)
            template = json.loads(source.read_text())
            generated, profiles = transform(template, dataset)
            destination = BUNDLE / "configs" / dataset / f"{method}.json"
            write_json(destination, generated)
            manifest["jobs"].append(dict(
                dataset=dataset, method=method,
                config=str(destination.relative_to(BUNDLE)),
                source_config=str(source.relative_to(PROJECT)),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                runner=str(Path("experiments") / directory / entry),
                parameter_profiles=profiles, workload_count=len(generated["datasets"])))
    write_json(BUNDLE / "manifest.json", manifest)
    print(f"Generated {len(manifest['jobs'])} configurations; "
          f"{sum(j['workload_count'] for j in manifest['jobs'])} task/parameter combinations.")


if __name__ == "__main__":
    main()
