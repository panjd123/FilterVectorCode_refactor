#!/usr/bin/env python3

import csv
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def read_num_vectors(bin_path: Path) -> int:
    with bin_path.open("rb") as f:
        header = f.read(8)
    if len(header) != 8:
        raise RuntimeError(f"invalid bin header: {bin_path}")
    num, _dim = struct.unpack("<II", header)
    return int(num)


def run_command(cmd, output_path: Path | None = None, env=None) -> None:
    command_line = "COMMAND: " + " ".join(str(x) for x in cmd)
    log(command_line)

    out = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"command output: {output_path}")
        out = output_path.open("w")
        out.write(command_line + "\n\n")
        out.flush()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if out is not None:
                out.write(line)
                out.flush()
            print(line, end="", flush=True)
        proc.stdout.close()
        returncode = proc.wait()
    finally:
        if out is not None:
            out.close()

    if returncode != 0:
        detail = f", see {output_path}" if output_path is not None else ""
        raise RuntimeError(f"command failed ({returncode}){detail}")


def ensure_query_bin(build_dir: Path, data_dir: Path, dataset: str, query_task: str) -> Path:
    query_dir = data_dir / query_task
    query_bin = query_dir / f"{dataset}_query.bin"
    if query_bin.exists():
        return query_bin
    query_fvecs = query_dir / f"{dataset}_query.fvecs"
    if not query_fvecs.exists():
        raise FileNotFoundError(f"missing query bin/fvecs: {query_bin} / {query_fvecs}")
    tool = build_dir / "tools" / "fvecs_to_bin"
    if not tool.exists():
        raise FileNotFoundError(f"missing fvecs_to_bin: {tool}")
    run_command(
        [
            str(tool),
            "--data_type",
            "float",
            "--input_file",
            str(query_fvecs),
            "--output_file",
            str(query_bin),
        ],
    )
    return query_bin


def ensure_gt(cfg, dataset_cfg, query_bin: Path, gt_dir: Path) -> Path:
    dataset = dataset_cfg["dataset"]
    query_task = dataset_cfg["query_task"]
    data_root = Path(cfg["data_root"])
    build_dir = Path(cfg["build_dir"])
    search_cfg = dict(cfg["search"])
    search_cfg.update(dataset_cfg.get("search", {}))
    data_dir = data_root / dataset
    gt_file = gt_dir / f"{dataset}_gt_labels_containment.bin"
    if gt_file.exists():
        log(f"[{dataset}] GT exists: {gt_file}")
        return gt_file

    gt_dir.mkdir(parents=True, exist_ok=True)
    compute_gt = build_dir / "tools" / "compute_groundtruth"
    if not compute_gt.exists():
        raise FileNotFoundError(f"missing compute_groundtruth: {compute_gt}")

    log(f"[{dataset}] GT missing, computing: {gt_file}")
    run_command(
        [
            str(compute_gt),
            "--data_type",
            "float",
            "--dist_fn",
            "L2",
            "--scenario",
            "containment",
            "--K",
            str(search_cfg["K"]),
            "--num_threads",
            str(search_cfg["num_threads"]),
            "--base_bin_file",
            str(data_dir / f"{dataset}_base.bin"),
            "--base_label_file",
            str(data_dir / f"{dataset}_base_labels.txt"),
            "--query_bin_file",
            str(query_bin),
            "--query_label_file",
            str(data_dir / query_task / f"{dataset}_query_labels.txt"),
            "--gt_file",
            str(gt_file),
        ],
    )
    return gt_file


def bool_arg(value: bool) -> str:
    return "true" if value else "false"


def result_query_dir_name(query_task: str, search_cfg: dict) -> str:
    return (
        f"{query_task}_{search_cfg['lsearch_start']}_"
        f"{search_cfg['lsearch_step']}_{search_cfg['lsearch_end']}"
    )


def write_qps_summary(summary_csv: Path, output_csv: Path, num_queries: int) -> None:
    with summary_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "QPS" not in fieldnames:
        fieldnames.append("QPS")
    for row in rows:
        avg_ms = float(row.get("Average_Time_ms", "0") or 0)
        row["QPS"] = f"{(num_queries * 1000.0 / avg_ms) if avg_ms > 0 else 0.0:.6f}"
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_search(cfg, dataset_cfg, method, query_bin: Path, gt_file: Path, num_queries: int) -> None:
    dataset = dataset_cfg["dataset"]
    query_task = dataset_cfg["query_task"]
    data_root = Path(cfg["data_root"])
    result_root = Path(cfg["result_root"])
    build_dir = Path(cfg["build_dir"])
    search_cfg = dict(cfg["search"])
    search_cfg.update(dataset_cfg.get("search", {}))
    data_dir = data_root / dataset
    query_dir = data_dir / query_task
    index_dir = result_root / dataset / "index" / method["index_name"] / "index_files"
    result_query_task = result_query_dir_name(query_task, search_cfg)
    result_dir = result_root / dataset / "results" / method["name"] / result_query_task
    raw_results_dir = result_dir / "results"
    other_dir = result_dir / "others"
    raw_results_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)

    if not (index_dir / "meta").exists():
        raise FileNotFoundError(f"missing index meta: {index_dir / 'meta'}")

    lsearch_values = list(
        range(
            int(search_cfg["lsearch_start"]),
            int(search_cfg["lsearch_end"]) + 1,
            int(search_cfg["lsearch_step"]),
        )
    )

    query_group_file = query_dir / f"{dataset}_query_source_groups.txt"
    if not query_group_file.exists():
        query_group_file = result_dir / "missing_query_source_groups.txt"

    env = os.environ.copy()
    if method.get("special_block_search", False):
        env["UNG_SPECIAL_BLOCK_SEARCH"] = "1"
    else:
        env.pop("UNG_SPECIAL_BLOCK_SEARCH", None)
        env.pop("UNG_SPECIAL_SEARCH_MODE", None)

    for key, value in method.get("env", {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)

    cmd = [
        str(build_dir / "apps" / "search_UNG_index"),
        "--data_type",
        "float",
        "--dataset",
        dataset,
        "--dist_fn",
        "L2",
        "--num_threads",
        str(search_cfg["num_threads"]),
        "--K",
        str(search_cfg["K"]),
        "--num_repeats",
        str(search_cfg["num_repeats"]),
        "--is_new_method",
        "true",
        "--force_use_alg",
        "1",
        "--is_idea2_available",
        "false",
        "--is_new_trie_method",
        bool_arg(method.get("is_new_trie_method", False)),
        "--is_rec_more_start",
        bool_arg(method.get("is_rec_more_start", False)),
        "--base_bin_file",
        str(data_dir / f"{dataset}_base.bin"),
        "--query_bin_file",
        str(query_bin),
        "--query_label_file",
        str(query_dir / f"{dataset}_query_labels.txt"),
        "--query_group_id_file",
        str(query_group_file),
        "--gt_file",
        str(gt_file),
        "--index_path_prefix",
        str(index_dir) + "/",
        "--result_path_prefix",
        str(raw_results_dir) + "/",
        "--selector_model_prefix",
        str(cfg.get("selector_model_prefix", result_root / "SelectModels")),
        "--scenario",
        "containment",
        "--num_entry_points",
        str(search_cfg["num_entry_points"]),
        "--Lsearch",
        *[str(v) for v in lsearch_values],
        "--lsearch_start",
        str(search_cfg["lsearch_start"]),
        "--lsearch_step",
        str(search_cfg["lsearch_step"]),
        "--efs_start",
        str(search_cfg["efs_start"]),
        "--efs_step_slow",
        str(search_cfg["efs_step_slow"]),
        "--efs_step_fast",
        str(search_cfg["efs_step_fast"]),
        "--lsearch_threshold",
        str(search_cfg["lsearch_threshold"]),
        "--entry_group_provider",
        method["entry_group_provider"],
        "--graph_search_backend",
        "neighbor_list",
        "--skip_query_features",
        "true",
        "--skip_bitmap_comparison",
        "true",
    ]

    log(f"[{dataset}][{method['name']}] search start")
    run_command(cmd, other_dir / f"{dataset}_search_output.txt", env=env)

    summary_csv = raw_results_dir / "search_time_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"missing search summary: {summary_csv}")
    write_qps_summary(summary_csv, raw_results_dir / "search_time_summary_qps.csv", num_queries)
    log(f"[{dataset}][{method['name']}] search done")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_search_comparison.py <config.json>", file=sys.stderr)
        return 2
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())

    log(f"config: {cfg_path}")

    build_dir = Path(cfg["build_dir"])
    search_bin = build_dir / "apps" / "search_UNG_index"
    if not search_bin.exists():
        raise FileNotFoundError(f"missing search_UNG_index: {search_bin}")

    for dataset_cfg in cfg["datasets"]:
        dataset = dataset_cfg["dataset"]
        query_task = dataset_cfg["query_task"]
        data_dir = Path(cfg["data_root"]) / dataset
        result_root = Path(cfg["result_root"]) / dataset
        gt_dir = result_root / "GroundTruth" / query_task
        query_bin = ensure_query_bin(build_dir, data_dir, dataset, query_task)
        num_queries = read_num_vectors(query_bin)
        log(f"[{dataset}] query_task={query_task} num_queries={num_queries}")
        gt_file = ensure_gt(cfg, dataset_cfg, query_bin, gt_dir)
        for method in cfg["methods"]:
            run_search(cfg, dataset_cfg, method, query_bin, gt_file, num_queries)
    log("all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
