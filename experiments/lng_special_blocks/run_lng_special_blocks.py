#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".cuh", ".cmake", ".txt"}


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(cmd: list[str], *, env: dict[str, str] | None = None, output_path: Path | None = None) -> None:
    line = "COMMAND: " + " ".join(str(x) for x in cmd)
    log(line)
    out = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = output_path.open("w", encoding="utf-8")
        out.write(line + "\n\n")
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
        for line_out in proc.stdout:
            print(line_out, end="", flush=True)
            if out is not None:
                out.write(line_out)
                out.flush()
        proc.stdout.close()
        rc = proc.wait()
    finally:
        if out is not None:
            out.close()
    if rc != 0:
        detail = f", see {output_path}" if output_path else ""
        raise RuntimeError(f"command failed ({rc}){detail}")


def target_source_paths(target: str) -> list[Path]:
    src_root = REPO_ROOT / "UNG/codes"
    if not src_root.exists():
        return []
    paths = [src_root / "CMakeLists.txt"]
    paths.extend(
        path
        for path in src_root.rglob("*")
        if path.is_file()
        and (path.suffix in SOURCE_SUFFIXES or path.name == "CMakeLists.txt")
        and "__pycache__" not in path.parts
    )
    return paths


def target_is_stale(exe: Path, target: str) -> bool:
    if not exe.exists():
        return True
    exe_mtime = exe.stat().st_mtime
    return any(path.stat().st_mtime > exe_mtime for path in target_source_paths(target) if path.exists())


def should_build_target(exe: Path, target: str, build_policy: str) -> bool:
    if build_policy == "always":
        return True
    if build_policy == "if_missing":
        return not (exe.exists() and os.access(exe, os.X_OK))
    if build_policy == "if_stale":
        return not (exe.exists() and os.access(exe, os.X_OK)) or target_is_stale(exe, target)
    raise ValueError(f"invalid build policy: {build_policy} (expected always/if_missing/if_stale)")


def ensure_target(
    build_dir: Path,
    target: str,
    *,
    build_policy: str = "if_stale",
    jobs_env: str = "BUILD_JOBS",
) -> Path:
    rel = {
        "build_UNG_index": Path("apps/build_UNG_index"),
        "search_UNG_index": Path("apps/search_UNG_index"),
        "compute_groundtruth": Path("tools/compute_groundtruth"),
        "fvecs_to_bin": Path("tools/fvecs_to_bin"),
    }[target]
    exe = build_dir / rel
    if not should_build_target(exe, target, build_policy):
        return exe
    reason = "missing" if not exe.exists() else build_policy
    log(f"{reason} {target}: {exe}; building it")
    run_command(["cmake", "-S", str(REPO_ROOT / "UNG/codes"), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"])
    run_command(["cmake", "--build", str(build_dir), f"-j{os.environ.get(jobs_env, '16')}", "--target", target])
    if not exe.exists():
        raise FileNotFoundError(f"target build did not create executable: {exe}")
    return exe



def build_jobs_env(cfg: dict) -> str:
    return str(cfg.get("build_tools", {}).get("jobs_env", "BUILD_JOBS"))


def target_build_policy(cfg: dict, target: str) -> str:
    build_tools = cfg.get("build_tools", {})
    default_policy = str(build_tools.get("policy", "if_stale"))
    per_target = build_tools.get("targets", {})
    return str(per_target.get(target, default_policy))


def read_num_vectors(bin_path: Path) -> int:
    with bin_path.open("rb") as f:
        header = f.read(8)
    if len(header) != 8:
        raise RuntimeError(f"invalid bin header: {bin_path}")
    n, _dim = struct.unpack("<II", header)
    return int(n)


def bool_arg(value: bool) -> str:
    return "true" if value else "false"


def lsearch_values(search_cfg: dict) -> list[int]:
    if search_cfg.get("lsearch_values"):
        return [int(v) for v in search_cfg["lsearch_values"]]
    return list(range(int(search_cfg["lsearch_start"]), int(search_cfg["lsearch_end"]) + 1, int(search_cfg["lsearch_step"])))


def result_query_dir_name(query_task: str, search_cfg: dict) -> str:
    return f"{query_task}_{search_cfg['lsearch_start']}_{search_cfg['lsearch_step']}_{search_cfg['lsearch_end']}"


def dataset_name(dataset_cfg) -> str:
    return dataset_cfg if isinstance(dataset_cfg, str) else dataset_cfg["dataset"]


def normalize_dataset_cfg(dataset_cfg) -> dict:
    if isinstance(dataset_cfg, str):
        return {"dataset": dataset_cfg}
    return dict(dataset_cfg)


def dataset_paths(cfg: dict, dataset_cfg: dict) -> dict[str, Path]:
    dataset = dataset_cfg["dataset"]
    data_dir = Path(cfg["data_root"]) / dataset
    placeholder = data_dir / ".missing_optional_build_input_placeholder"
    base_label_info = Path(dataset_cfg.get("base_label_info_file", data_dir / f"{dataset}_base_labels_info.log"))
    if not base_label_info.exists():
        fallback = data_dir / "log"
        base_label_info = fallback if fallback.exists() else placeholder
    tree_roots = Path(dataset_cfg.get("base_label_tree_roots", data_dir / "tree_roots.txt"))
    if not tree_roots.exists():
        tree_roots = placeholder
    return {
        "data_dir": data_dir,
        "placeholder": placeholder,
        "base_bin": Path(dataset_cfg.get("base_bin_file", data_dir / f"{dataset}_base.bin")),
        "base_labels": Path(dataset_cfg.get("base_label_file", data_dir / f"{dataset}_base_labels.txt")),
        "base_label_info": base_label_info,
        "tree_roots": tree_roots,
    }


def ensure_optional_placeholders(paths: dict[str, Path]) -> None:
    placeholder = paths["placeholder"]
    if paths["base_label_info"] == placeholder or paths["tree_roots"] == placeholder:
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        placeholder.touch()


def ensure_query_bin(cfg: dict, dataset_cfg: dict, build_dir: Path) -> Path:
    dataset = dataset_cfg["dataset"]
    query_task = dataset_cfg["query_task"]
    data_dir = Path(cfg["data_root"]) / dataset
    query_dir = data_dir / query_task
    query_bin = Path(dataset_cfg.get("query_bin_file", query_dir / f"{dataset}_query.bin"))
    if query_bin.exists():
        return query_bin
    if not cfg.get("steps", {}).get("convert_query_fvecs_if_missing", True):
        raise FileNotFoundError(f"missing query bin: {query_bin}")
    query_fvecs = query_dir / f"{dataset}_query.fvecs"
    if not query_fvecs.exists():
        raise FileNotFoundError(f"missing query bin/fvecs: {query_bin} / {query_fvecs}")
    tool = ensure_target(
        build_dir,
        "fvecs_to_bin",
        build_policy=target_build_policy(cfg, "fvecs_to_bin"),
        jobs_env=build_jobs_env(cfg),
    )
    run_command([str(tool), "--data_type", "float", "--input_file", str(query_fvecs), "--output_file", str(query_bin)])
    return query_bin


def ensure_groundtruth(cfg: dict, dataset_cfg: dict, build_dir: Path, query_bin: Path) -> Path:
    dataset = dataset_cfg["dataset"]
    query_task = dataset_cfg["query_task"]
    data_dir = Path(cfg["data_root"]) / dataset
    result_root = Path(cfg["result_root"]) / dataset
    gt_file = Path(dataset_cfg.get("gt_file", result_root / "GroundTruth" / query_task / f"{dataset}_gt_labels_containment.bin"))
    if gt_file.exists():
        return gt_file
    if not cfg.get("steps", {}).get("compute_groundtruth_if_missing", True):
        raise FileNotFoundError(f"missing groundtruth: {gt_file}")
    gt_file.parent.mkdir(parents=True, exist_ok=True)
    search_cfg = {**cfg["search"], **dataset_cfg.get("search", {})}
    compute_gt = ensure_target(
        build_dir,
        "compute_groundtruth",
        build_policy=target_build_policy(cfg, "compute_groundtruth"),
        jobs_env=build_jobs_env(cfg),
    )
    run_command([
        str(compute_gt),
        "--data_type", cfg["build"].get("data_type", "float"),
        "--dist_fn", cfg["build"].get("dist_fn", "L2"),
        "--scenario", "containment",
        "--K", str(search_cfg["K"]),
        "--num_threads", str(search_cfg["num_threads"]),
        "--base_bin_file", str(data_dir / f"{dataset}_base.bin"),
        "--base_label_file", str(data_dir / f"{dataset}_base_labels.txt"),
        "--query_bin_file", str(query_bin),
        "--query_label_file", str(data_dir / query_task / f"{dataset}_query_labels.txt"),
        "--gt_file", str(gt_file),
    ])
    return gt_file


def method_env(cfg: dict, method: dict, *, include_search_env: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in cfg.get("common_ung_env", {}).items():
        env[key] = str(value)
    for key, value in method.get("ung_env", {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    if include_search_env:
        for key, value in method.get("search_env", {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = str(value)
    return env


def build_index(cfg: dict, dataset_cfg: dict, method: dict, build_bin: Path) -> None:
    dataset = dataset_cfg["dataset"]
    paths = dataset_paths(cfg, dataset_cfg)
    ensure_optional_placeholders(paths)
    if not paths["base_bin"].exists() or not paths["base_labels"].exists():
        raise FileNotFoundError(f"missing base files for {dataset}: {paths['base_bin']} / {paths['base_labels']}")
    result_root = Path(cfg["result_root"])
    index_dir = result_root / dataset / "index" / method["index_name"]
    index_files = index_dir / "index_files"
    results_dir = index_dir / "results"
    others_dir = index_dir / "others"
    index_files.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    others_dir.mkdir(parents=True, exist_ok=True)
    build_cfg = cfg["build"]
    cmd = [
        str(build_bin),
        "--dataset", dataset,
        "--data_type", str(build_cfg.get("data_type", "float")),
        "--dist_fn", str(build_cfg.get("dist_fn", "L2")),
        "--base_bin_file", str(paths["base_bin"]),
        "--base_label_file", str(paths["base_labels"]),
        "--base_label_info_file", str(paths["base_label_info"]),
        "--base_label_tree_roots", str(paths["tree_roots"]),
        "--num_threads", str(build_cfg["num_threads"]),
        "--index_path_prefix", str(index_files) + "/",
        "--result_path_prefix", str(results_dir) + "/",
        "--scenario", str(build_cfg.get("scenario", "general")),
        "--index_type", str(build_cfg.get("index_type", "Vamana")),
        "--num_cross_edges", str(build_cfg["num_cross_edges"]),
        "--max_degree", str(build_cfg["max_degree"]),
        "--Lbuild", str(build_cfg["Lbuild"]),
        "--alpha", str(build_cfg["alpha"]),
    ]
    log(f"[{dataset}][{method['name']}] build index -> {index_files}")
    run_command(cmd, env=method_env(cfg, method), output_path=others_dir / "build.log")


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


def run_search(cfg: dict, dataset_cfg: dict, method: dict, search_bin: Path, query_bin: Path, gt_file: Path, num_queries: int) -> None:
    dataset = dataset_cfg["dataset"]
    query_task = dataset_cfg["query_task"]
    data_dir = Path(cfg["data_root"]) / dataset
    query_dir = data_dir / query_task
    search_cfg = {**cfg["search"], **dataset_cfg.get("search", {})}
    result_root = Path(cfg["result_root"])
    index_files = result_root / dataset / "index" / method["index_name"] / "index_files"
    if not (index_files / "meta").exists():
        raise FileNotFoundError(f"missing index meta: {index_files / 'meta'}")
    result_task = result_query_dir_name(query_task, search_cfg)
    result_dir = result_root / dataset / "results" / method["name"] / result_task
    raw_results = result_dir / "results"
    others_dir = result_dir / "others"
    raw_results.mkdir(parents=True, exist_ok=True)
    others_dir.mkdir(parents=True, exist_ok=True)
    query_group_file = query_dir / f"{dataset}_query_source_groups.txt"
    if not query_group_file.exists():
        query_group_file = others_dir / "missing_query_source_groups.txt"
        query_group_file.touch()
    values = lsearch_values(search_cfg)
    cmd = [
        str(search_bin),
        "--data_type", str(cfg["build"].get("data_type", "float")),
        "--dataset", dataset,
        "--dist_fn", str(cfg["build"].get("dist_fn", "L2")),
        "--num_threads", str(search_cfg["num_threads"]),
        "--K", str(search_cfg["K"]),
        "--num_repeats", str(search_cfg["num_repeats"]),
        "--is_new_method", "true",
        "--force_use_alg", str(search_cfg.get("force_use_alg", 1)),
        "--is_idea2_available", "false",
        "--is_new_trie_method", bool_arg(method.get("is_new_trie_method", False)),
        "--is_rec_more_start", bool_arg(method.get("is_rec_more_start", False)),
        "--base_bin_file", str(data_dir / f"{dataset}_base.bin"),
        "--query_bin_file", str(query_bin),
        "--query_label_file", str(query_dir / f"{dataset}_query_labels.txt"),
        "--query_group_id_file", str(query_group_file),
        "--gt_file", str(gt_file),
        "--index_path_prefix", str(index_files) + "/",
        "--result_path_prefix", str(raw_results) + "/",
        "--selector_model_prefix", str(cfg.get("selector_model_prefix", result_root / "SelectModels")),
        "--scenario", "containment",
        "--num_entry_points", str(search_cfg["num_entry_points"]),
        "--Lsearch", *[str(v) for v in values],
        "--lsearch_start", str(search_cfg["lsearch_start"]),
        "--lsearch_step", str(search_cfg["lsearch_step"]),
        "--efs_start", str(search_cfg["efs_start"]),
        "--efs_step_slow", str(search_cfg["efs_step_slow"]),
        "--efs_step_fast", str(search_cfg["efs_step_fast"]),
        "--lsearch_threshold", str(search_cfg["lsearch_threshold"]),
        "--entry_group_provider", str(search_cfg.get("entry_group_provider", "cpu_bruteforce_els")),
        "--graph_search_backend", str(search_cfg.get("graph_search_backend", "neighbor_list")),
        "--skip_query_features", "true",
        "--skip_bitmap_comparison", "true",
    ]
    if "scalar_els_cap" in search_cfg:
        cmd.extend(["--scalar_els_cap", str(search_cfg["scalar_els_cap"])])
    log(f"[{dataset}][{method['name']}] search -> {raw_results}")
    run_command(cmd, env=method_env(cfg, method, include_search_env=True), output_path=others_dir / "search.log")
    summary = raw_results / "search_time_summary.csv"
    if summary.exists():
        write_qps_summary(summary, raw_results / "search_time_summary_qps.csv", num_queries)


def selected_methods(cfg: dict, only: list[str] | None) -> list[dict]:
    methods = cfg.get("methods", [])
    if not only:
        return methods
    by_name = {m["name"]: m for m in methods}
    missing = [name for name in only if name not in by_name]
    if missing:
        raise ValueError(f"unknown method(s): {', '.join(missing)}")
    return [by_name[name] for name in only]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and search trie/LNG special-block UNG indexes.")
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--only", action="append", default=[], help="only run this method name; can repeat")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate config and print selected methods only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config(args.config)
    methods = selected_methods(cfg, args.only or None)
    log(f"config: {args.config}")
    log("methods: " + ", ".join(m["name"] for m in methods))
    if args.dry_run:
        return 0
    build_dir = Path(cfg["build_dir"])
    build_bin = ensure_target(
        build_dir,
        "build_UNG_index",
        build_policy=target_build_policy(cfg, "build_UNG_index"),
        jobs_env=build_jobs_env(cfg),
    )
    search_bin = ensure_target(
        build_dir,
        "search_UNG_index",
        build_policy=target_build_policy(cfg, "search_UNG_index"),
        jobs_env=build_jobs_env(cfg),
    )
    do_build = cfg.get("steps", {}).get("build_indexes", True) and not args.search_only
    do_search = cfg.get("steps", {}).get("run_search", True) and not args.build_only
    for raw_dataset_cfg in cfg["datasets"]:
        dataset_cfg = normalize_dataset_cfg(raw_dataset_cfg)
        if "query_task" not in dataset_cfg and do_search:
            raise ValueError(f"dataset {dataset_cfg['dataset']} needs query_task for search")
        query_bin = None
        gt_file = None
        num_queries = 0
        if do_search:
            query_bin = ensure_query_bin(cfg, dataset_cfg, build_dir)
            num_queries = read_num_vectors(query_bin)
            gt_file = ensure_groundtruth(cfg, dataset_cfg, build_dir, query_bin)
        for method in methods:
            if do_build:
                build_index(cfg, dataset_cfg, method, build_bin)
            if do_search:
                assert query_bin is not None and gt_file is not None
                run_search(cfg, dataset_cfg, method, search_bin, query_bin, gt_file, num_queries)
    log("all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
