#!/usr/bin/env python3
"""Generate two independent single-task experiments from existing baseline configs.

threads10: K=10, 10 search threads. k20: K=20, 100 search threads.
Build parameters and task-specific search sweeps remain unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
PROJECT = BUNDLE.parents[1]
DATASETS = ("Amazon", "VImg", "Music")
TASK = "query_minlen1_sel10to40pct"
GROUPS = {"threads10": dict(K=10, num_threads=10), "k20": dict(K=20, num_threads=100)}
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


def source_path(dataset, method):
    directory = METHODS[method][0]
    if method in ("trie_block", "ung"):
        label = dataset if dataset == "VImg" else dataset.lower()
        filename = f"config_{label}{'_ung' if method == 'ung' else ''}.json"
    else:
        filename = f"{dataset}_config.json"
    return PROJECT / "experiments" / directory / filename


def transform(template, dataset, method, group):
    cfg = copy.deepcopy(template)
    rows = [row for row in cfg["datasets"] if row["dataset"] == dataset and row["query_task"] == TASK]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one source task: {dataset}/{method}")
    cfg["datasets"] = rows
    parameters = GROUPS[group]
    cfg["search"].update(parameters)
    # Narrower scopes must not override the selected experiment parameters.
    for scope in [rows[0], *cfg.get("methods", [])]:
        for key, value in parameters.items():
            if key in scope.get("search", {}):
                scope["search"][key] = value
    suffix = f"_threads{parameters['num_threads']}_k{parameters['K']}"
    if cfg.get("methods"):
        for entry in cfg["methods"]:
            entry["name"] += suffix
    else:
        cfg["method_name"] += suffix
    gt_task = TASK if parameters["K"] == 10 else f"{TASK}_k20"
    rows[0]["groundtruth_file"] = str(Path(cfg["result_root"]) / dataset / "GroundTruth" / gt_task / f"{dataset}_gt_labels_containment.bin")
    if method == "rwalks":
        # Keep the original union of task labels so only search parameters change.
        cfg["schema_query_label_files"] = [
            str(Path(cfg["data_root"]) / dataset / row["query_task"] / f"{dataset}_query_labels.txt")
            for row in template["datasets"] if row["dataset"] == dataset]
    return cfg


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    manifest = dict(query_task=TASK, groups=GROUPS, datasets=list(DATASETS), methods=list(METHODS), jobs=[])
    for group in GROUPS:
        for dataset in DATASETS:
            for method, (directory, entry) in METHODS.items():
                source = source_path(dataset, method)
                cfg = transform(json.loads(source.read_text()), dataset, method, group)
                target = BUNDLE / "configs" / group / dataset / f"{method}.json"
                write_json(target, cfg)
                manifest["jobs"].append(dict(group=group, dataset=dataset, method=method,
                    config=str(target.relative_to(BUNDLE)), source_config=str(source.relative_to(PROJECT)),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    runner=str(Path("experiments") / directory / entry)))
    write_json(BUNDLE / "manifest.json", manifest)
    print(f"Generated {len(manifest['jobs'])} configurations: 27 per experiment group.")


if __name__ == "__main__":
    main()
