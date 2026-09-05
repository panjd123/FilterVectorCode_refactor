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


def run_command(cmd, output_path: Path | None = None, env=None) -> None:
    command_line = "COMMAND: " + " ".join(str(x) for x in cmd)
    log(command_line)
    out = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = output_path.open("w", encoding="utf-8")
        out.write(command_line + "\n\n")
        out.flush()
    try:
        proc = subprocess.Popen(
            [str(x) for x in cmd],
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


def read_num_vectors(bin_path: Path) -> int:
    with bin_path.open("rb") as f:
        header = f.read(8)
    if len(header) != 8:
        raise RuntimeError(f"invalid bin header: {bin_path}")
    num, _dim = struct.unpack("<II", header)
    return int(num)


def bool_arg(value: bool) -> str:
    return "true" if value else "false"


def merged_build_cfg(cfg: dict, dataset_cfg: dict) -> dict:
    build_cfg = {
        "num_threads": 32,
        "max_degree": 64,
        "Lbuild": 100,
        "alpha": 1.2,
        "index_type": "NaviX",
    }
    build_cfg.update(cfg.get("build", {}))
    build_cfg.update(dataset_cfg.get("build", {}))
    return build_cfg


def merged_search_cfg(cfg: dict, dataset_cfg: dict) -> dict:
    search_cfg = {
        "K": 10,
        "num_repeats": 1,
        "num_threads": cfg.get("build", {}).get("num_threads", 32),
        "num_entry_points": 16,
        "lsearch_start": 100,
        "lsearch_end": 1000,
        "lsearch_step": 100,
        "efs_start": 1,
        "efs_step_slow": 1,
        "efs_step_fast": 1,
        "lsearch_threshold": 100,
    }
    search_cfg.update(cfg.get("search", {}))
    search_cfg.update(dataset_cfg.get("search", {}))
    return search_cfg


def lsearch_values(search_cfg: dict) -> list[int]:
    if search_cfg.get("lsearch_values"):
        return [int(value) for value in search_cfg["lsearch_values"]]
    return list(
        range(
            int(search_cfg["lsearch_start"]),
            int(search_cfg["lsearch_end"]) + 1,
            int(search_cfg["lsearch_step"]),
        )
    )


def result_query_dir_name(query_task: str, search_cfg: dict) -> str:
    return (
        f"{query_task}_{search_cfg['lsearch_start']}_"
        f"{search_cfg['lsearch_step']}_{search_cfg['lsearch_end']}"
    )


def dataset_paths(cfg: dict, dataset_cfg: dict, search_cfg: dict) -> dict[str, Path | str]:
    dataset = dataset_cfg["dataset"]
    query_task = dataset_cfg["query_task"]
    data_dir = Path(cfg["data_root"]) / dataset
    query_dir = data_dir / query_task
    result_dataset_dir = Path(cfg["result_root"]) / dataset
    index_name = dataset_cfg.get("index_name", cfg.get("index_name", "NaviX"))
    result_method = dataset_cfg.get("method_name", cfg.get("method_name", "NaviX"))
    result_task = result_query_dir_name(query_task, search_cfg)
    return {
        "dataset": dataset,
        "query_task": query_task,
        "data_dir": data_dir,
        "query_dir": query_dir,
        "base_bin": Path(dataset_cfg.get("base_bin_file", data_dir / f"{dataset}_base.bin")),
        "base_fvecs": Path(dataset_cfg.get("base_fvecs", data_dir / f"{dataset}_base.fvecs")),
        "base_labels": Path(dataset_cfg.get("base_label_file", data_dir / f"{dataset}_base_labels.txt")),
        "base_label_info": Path(dataset_cfg.get("base_label_info_file", data_dir / f"{dataset}_base_label_info.txt")),
        "base_label_tree_roots": Path(dataset_cfg.get("base_label_tree_roots", data_dir / f"{dataset}_base_label_tree_roots.txt")),
        "query_bin": Path(dataset_cfg.get("query_bin_file", query_dir / f"{dataset}_query.bin")),
        "query_fvecs": Path(dataset_cfg.get("query_fvecs", query_dir / f"{dataset}_query.fvecs")),
        "query_labels": Path(dataset_cfg.get("query_label_file", query_dir / f"{dataset}_query_labels.txt")),
        "gt_file": Path(dataset_cfg.get("groundtruth_file", result_dataset_dir / "GroundTruth" / query_task / f"{dataset}_gt_labels_containment.bin")),
        "index_dir": Path(dataset_cfg.get("index_path_prefix", result_dataset_dir / "index" / index_name / "index_files")),
        "result_dir": Path(dataset_cfg.get("result_path_prefix", result_dataset_dir / "results" / result_method / result_task / "results")),
        "other_dir": Path(dataset_cfg.get("other_path_prefix", result_dataset_dir / "results" / result_method / result_task / "others")),
    }


def ensure_binaries(cfg: dict) -> tuple[Path, Path, Path, Path]:
    source_dir = Path(cfg.get("source_dir", Path(__file__).resolve().parents[2] / "UNG" / "codes"))
    build_dir = Path(cfg.get("build_dir", Path(__file__).resolve().parents[2] / "build_ung_rel"))
    auto_compile = bool(cfg.get("auto_compile", True))
    jobs = str(cfg.get("build_jobs", os.environ.get("BUILD_JOBS", 16)))

    if auto_compile:
        build_dir.mkdir(parents=True, exist_ok=True)
        run_command(["cmake", "-S", source_dir, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release"])
        run_command(
            [
                "cmake",
                "--build",
                build_dir,
                f"-j{jobs}",
                "--target",
                "build_UNG_index",
                "search_UNG_index",
                "compute_groundtruth",
                "fvecs_to_bin",
            ]
        )

    build_index_bin = build_dir / "apps" / "build_UNG_index"
    search_bin = build_dir / "apps" / "search_UNG_index"
    compute_gt_bin = build_dir / "tools" / "compute_groundtruth"
    fvecs_to_bin = build_dir / "tools" / "fvecs_to_bin"
    for binary in [build_index_bin, search_bin, compute_gt_bin, fvecs_to_bin]:
        if not binary.exists():
            raise FileNotFoundError(f"missing required binary: {binary}")
    return build_index_bin, search_bin, compute_gt_bin, fvecs_to_bin


def ensure_query_bin(fvecs_to_bin: Path, paths: dict[str, Path | str]) -> Path:
    query_bin = paths["query_bin"]
    query_fvecs = paths["query_fvecs"]
    assert isinstance(query_bin, Path)
    assert isinstance(query_fvecs, Path)
    if query_bin.exists():
        return query_bin
    if not query_fvecs.exists():
        raise FileNotFoundError(f"missing query bin/fvecs: {query_bin} / {query_fvecs}")
    run_command(
        [
            fvecs_to_bin,
            "--data_type",
            "float",
            "--input_file",
            query_fvecs,
            "--output_file",
            query_bin,
        ]
    )
    return query_bin


def ensure_gt(compute_gt_bin: Path, cfg: dict, paths: dict[str, Path | str], query_bin: Path, search_cfg: dict) -> Path:
    gt_file = paths["gt_file"]
    assert isinstance(gt_file, Path)
    if gt_file.exists():
        log(f"[{paths['dataset']}] GT exists: {gt_file}")
        return gt_file
    base_bin = paths["base_bin"]
    base_labels = paths["base_labels"]
    query_labels = paths["query_labels"]
    assert isinstance(base_bin, Path)
    assert isinstance(base_labels, Path)
    assert isinstance(query_labels, Path)
    gt_file.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            compute_gt_bin,
            "--data_type",
            cfg.get("data_type", "float"),
            "--dist_fn",
            cfg.get("dist_fn", "L2"),
            "--scenario",
            cfg.get("scenario", "containment"),
            "--K",
            str(search_cfg["K"]),
            "--num_threads",
            str(search_cfg["num_threads"]),
            "--base_bin_file",
            base_bin,
            "--base_label_file",
            base_labels,
            "--query_bin_file",
            query_bin,
            "--query_label_file",
            query_labels,
            "--gt_file",
            gt_file,
        ],
        paths["other_dir"] / f"{paths['dataset']}_compute_gt.log",
    )
    return gt_file


def navix_index_exists(index_dir: Path) -> bool:
    return all((index_dir / name).exists() for name in ["meta", "graph", "vecs.bin", "labels.txt"])


def ensure_index(build_index_bin: Path, cfg: dict, paths: dict[str, Path | str], build_cfg: dict) -> Path:
    index_dir = paths["index_dir"]
    assert isinstance(index_dir, Path)
    if navix_index_exists(index_dir):
        log(f"[{paths['dataset']}] NaviX index exists: {index_dir}")
        return index_dir
    for key in ["base_bin", "base_labels"]:
        path = paths[key]
        assert isinstance(path, Path)
        if not path.exists():
            raise FileNotFoundError(f"missing {key}: {path}")
    index_dir.mkdir(parents=True, exist_ok=True)
    result_build_dir = index_dir.parent / "build"
    result_build_dir.mkdir(parents=True, exist_ok=True)
    base_label_info = paths["base_label_info"]
    base_label_tree_roots = paths["base_label_tree_roots"]
    assert isinstance(base_label_info, Path)
    assert isinstance(base_label_tree_roots, Path)
    run_command(
        [
            build_index_bin,
            "--dataset",
            paths["dataset"],
            "--data_type",
            cfg.get("data_type", "float"),
            "--dist_fn",
            cfg.get("dist_fn", "L2"),
            "--base_bin_file",
            paths["base_bin"],
            "--base_label_file",
            paths["base_labels"],
            "--base_label_info_file",
            base_label_info,
            "--base_label_tree_roots",
            base_label_tree_roots,
            "--num_threads",
            str(build_cfg["num_threads"]),
            "--index_path_prefix",
            str(index_dir) + "/",
            "--result_path_prefix",
            str(result_build_dir) + "/",
            "--index_type",
            build_cfg.get("index_type", "NaviX"),
            "--max_degree",
            str(build_cfg["max_degree"]),
            "--Lbuild",
            str(build_cfg["Lbuild"]),
            "--alpha",
            str(build_cfg.get("alpha", 1.2)),
        ],
        paths["other_dir"] / f"{paths['dataset']}_navix_build.log",
    )
    if not navix_index_exists(index_dir):
        missing = [name for name in ["meta", "graph", "vecs.bin", "labels.txt"] if not (index_dir / name).exists()]
        raise FileNotFoundError(f"NaviX build finished but required files are missing in {index_dir}: {missing}")
    return index_dir


def write_qps_summary(summary_csv: Path, output_csv: Path, num_queries: int) -> None:
    with summary_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "QPS" not in fieldnames:
        fieldnames.append("QPS")
    for row in rows:
        avg_ms = float(row.get("Average_Time_ms", "0") or 0)
        row["QPS"] = f"{(num_queries * 1000.0 / avg_ms) if avg_ms > 0 else 0.0:.6f}"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_search(search_bin: Path, cfg: dict, paths: dict[str, Path | str], query_bin: Path, gt_file: Path, search_cfg: dict, num_queries: int) -> None:
    result_dir = paths["result_dir"]
    other_dir = paths["other_dir"]
    assert isinstance(result_dir, Path)
    assert isinstance(other_dir, Path)
    result_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)
    values = lsearch_values(search_cfg)
    cmd = [
        search_bin,
        "--dataset",
        paths["dataset"],
        "--data_type",
        cfg.get("data_type", "float"),
        "--dist_fn",
        cfg.get("dist_fn", "L2"),
        "--base_bin_file",
        paths["base_bin"],
        "--query_bin_file",
        query_bin,
        "--query_label_file",
        paths["query_labels"],
        "--gt_file",
        gt_file,
        "--K",
        str(search_cfg["K"]),
        "--num_threads",
        str(search_cfg["num_threads"]),
        "--result_path_prefix",
        str(result_dir) + "/",
        "--index_path_prefix",
        str(paths["index_dir"]) + "/",
        "--Lsearch",
        *[str(v) for v in values],
        "--is_new_method",
        "true",
        "--is_idea2_available",
        "false",
        "--is_new_trie_method",
        "false",
        "--is_rec_more_start",
        "false",
        "--force_use_alg",
        "5",
        "--num_repeats",
        str(search_cfg["num_repeats"]),
        "--scenario",
        cfg.get("scenario", "containment"),
        "--num_entry_points",
        str(search_cfg["num_entry_points"]),
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
        "cpu_min_super_sets",
        "--graph_search_backend",
        "neighbor_list",
        "--skip_query_features",
        "true",
        "--skip_bitmap_comparison",
        "true",
    ]
    run_command(cmd, other_dir / f"{paths['dataset']}_navix_search.log")
    summary_csv = result_dir / "search_time_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"missing search summary: {summary_csv}")
    write_qps_summary(summary_csv, result_dir / "search_time_summary_qps.csv", num_queries)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_navix_baseline.py <config.json>", file=sys.stderr)
        return 2
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    log(f"config: {cfg_path}")
    build_index_bin, search_bin, compute_gt_bin, fvecs_to_bin = ensure_binaries(cfg)

    for dataset_cfg in cfg.get("datasets", []):
        search_cfg = merged_search_cfg(cfg, dataset_cfg)
        build_cfg = merged_build_cfg(cfg, dataset_cfg)
        paths = dataset_paths(cfg, dataset_cfg, search_cfg)
        query_bin = ensure_query_bin(fvecs_to_bin, paths)
        num_queries = read_num_vectors(query_bin)
        gt_file = ensure_gt(compute_gt_bin, cfg, paths, query_bin, search_cfg)
        ensure_index(build_index_bin, cfg, paths, build_cfg)
        run_search(search_bin, cfg, paths, query_bin, gt_file, search_cfg, num_queries)
        log(f"[{paths['dataset']}] NaviX baseline done")
    log("all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
