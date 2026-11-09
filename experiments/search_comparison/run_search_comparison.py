#!/usr/bin/env python3

import csv
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from result_completion import describe_search_values, should_skip_completed_search


NATIVE_UNG_INDEX_FILES = (
    "meta",
    "vecs.bin",
    "labels.txt",
    "group_id_to_label_set",
    "group_id_to_range",
    "group_entry_points",
    "group_id_to_vec_ids.dat",
    "new_to_old_vec_ids",
    "trie",
    "graph",
    "global_graph",
    "global_vamana_entry_point",
    "lng_out_neighbors.dat",
)


NATIVE_UNG_ENV_DEFAULTS = {
    "UNG_BUILD_PROFILE": "custom",
    "UNG_GROUP_GRAPH_IMPL": "0",
    "UNG_GET_MIN_SUPER_SETS_IMPL": "0",
    "UNG_LNG_IMPL": "1",
    "UNG_CROSS_EDGE_IMPL": "0",
    "UNG_ADDITIONAL_EDGES_IMPL": "0",
    "UNG_BASE_GROUP_TOPOLOGY": "lng",
}


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
    search_cfg = effective_search_cfg(cfg, dataset_cfg)
    data_dir = data_root / dataset
    gt_file = Path(dataset_cfg.get("groundtruth_file", gt_dir / f"{dataset}_gt_labels_containment.bin"))
    if gt_file.exists():
        expected_bytes = read_num_vectors(query_bin) * int(search_cfg["K"]) * 8
        if gt_file.stat().st_size != expected_bytes:
            raise ValueError(f"GroundTruth size does not match K={search_cfg['K']}: {gt_file}")
        log(f"[{dataset}] GT exists: {gt_file}")
        return gt_file

    gt_file.parent.mkdir(parents=True, exist_ok=True)
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


def read_kv_file(path: Path) -> dict[str, str]:
    values = {}
    for raw_line in path.read_text().splitlines():
        key, separator, value = raw_line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values


def validate_native_ung_meta(meta_path: Path) -> None:
    meta = read_kv_file(meta_path)
    base_topology = meta.get("base_group_topology", "lng")
    if base_topology != "lng":
        raise RuntimeError(
            f"native UNG index must use LNG base topology, got "
            f"base_group_topology={base_topology}: {meta_path}"
        )
    if meta.get("hierarchy_layers", ""):
        raise RuntimeError(
            f"native UNG index must not contain hierarchy layers: {meta_path}"
        )
    if meta.get("lng_constructed", "1") != "1":
        raise RuntimeError(f"native UNG index did not construct LNG: {meta_path}")


def ensure_native_ung_index(cfg: dict, dataset_cfg: dict, method: dict) -> None:
    auto_build = method.get("auto_build_ung", cfg.get("auto_build_ung", False))
    if not auto_build:
        return
    if (
        method.get("special_block_search", False)
        or method.get("entry_group_provider") == "special_block_trie"
    ):
        raise ValueError(
            f"{method['name']}: auto_build_ung only supports native UNG methods"
        )

    dataset = dataset_cfg["dataset"]
    data_dir = Path(cfg["data_root"]) / dataset
    index_root = Path(cfg["result_root"]) / dataset / "index" / method["index_name"]
    index_dir = index_root / "index_files"
    missing = [
        name for name in NATIVE_UNG_INDEX_FILES if not (index_dir / name).exists()
    ]
    if not missing:
        validate_native_ung_meta(index_dir / "meta")
        log(f"[{dataset}] native UNG index exists: {index_dir}")
        return

    build_dir = Path(cfg["build_dir"])
    build_bin = build_dir / "apps" / "build_UNG_index"
    if not build_bin.exists():
        raise FileNotFoundError(f"missing build_UNG_index: {build_bin}")

    base_bin = data_dir / f"{dataset}_base.bin"
    base_labels = data_dir / f"{dataset}_base_labels.txt"
    for input_path in (base_bin, base_labels):
        if not input_path.exists():
            raise FileNotFoundError(f"missing native UNG build input: {input_path}")

    results_dir = index_root / "results"
    others_dir = index_root / "others"
    index_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    others_dir.mkdir(parents=True, exist_ok=True)
    placeholder = others_dir / "placeholder.txt"
    if not placeholder.exists():
        placeholder.write_text("")
    label_info = data_dir / f"{dataset}_base_labels_info.log"
    if not label_info.exists():
        label_info = data_dir / "log"
    if not label_info.exists():
        label_info = placeholder
    tree_roots = data_dir / "tree_roots.txt"
    if not tree_roots.exists():
        tree_roots = placeholder

    build_cfg = dict(cfg.get("build", {}))
    build_env_cfg = dict(NATIVE_UNG_ENV_DEFAULTS)
    build_env_cfg.update(cfg.get("ung_env", {}))
    if build_env_cfg.get("UNG_BASE_GROUP_TOPOLOGY") != "lng":
        raise ValueError("auto-built native UNG requires UNG_BASE_GROUP_TOPOLOGY=lng")
    if build_env_cfg.get("UNG_HIERARCHY_LAYERS"):
        raise ValueError("auto-built native UNG must not configure UNG_HIERARCHY_LAYERS")
    special_keys = sorted(
        key
        for key, value in build_env_cfg.items()
        if key.startswith("UNG_SPECIAL_")
        and value not in (None, "", "0", 0, False)
    )
    if special_keys:
        raise ValueError(
            "auto-built native UNG must not configure special-block options: "
            + ", ".join(special_keys)
        )

    env = os.environ.copy()
    for key in list(env):
        if key.startswith("UNG_SPECIAL_") or key == "UNG_HIERARCHY_LAYERS":
            env.pop(key, None)
    for key, value in build_env_cfg.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)

    log(
        f"[{dataset}] native UNG index incomplete ({', '.join(missing)}); "
        f"building LNG-based index: {index_dir}"
    )
    run_command(
        [
            str(build_bin),
            "--data_type",
            str(build_cfg.get("data_type", "float")),
            "--dist_fn",
            str(build_cfg.get("dist_fn", "L2")),
            "--num_threads",
            str(build_cfg.get("num_threads", 32)),
            "--max_degree",
            str(build_cfg.get("max_degree", 32)),
            "--Lbuild",
            str(build_cfg.get("Lbuild", 100)),
            "--alpha",
            str(build_cfg.get("alpha", 1.2)),
            "--num_cross_edges",
            str(build_cfg.get("num_cross_edges", 4)),
            "--base_bin_file",
            str(base_bin),
            "--base_label_file",
            str(base_labels),
            "--base_label_info_file",
            str(label_info),
            "--base_label_tree_roots",
            str(tree_roots),
            "--index_path_prefix",
            str(index_dir) + "/",
            "--result_path_prefix",
            str(results_dir) + "/",
            "--scenario",
            str(build_cfg.get("scenario", "general")),
            "--dataset",
            dataset,
        ],
        others_dir / "ung_build.log",
        env=env,
    )

    missing_after_build = [
        name for name in NATIVE_UNG_INDEX_FILES if not (index_dir / name).exists()
    ]
    if missing_after_build:
        raise RuntimeError(
            f"native UNG build completed but index is incomplete: "
            f"{', '.join(missing_after_build)}"
        )
    validate_native_ung_meta(index_dir / "meta")
    log(f"[{dataset}] native UNG build done: {index_dir}")


def bool_arg(value: bool) -> str:
    return "true" if value else "false"


def effective_search_cfg(cfg: dict, dataset_cfg: dict, method: dict | None = None) -> dict:
    """Merge search settings from global, dataset, and optional method scopes."""
    search_cfg = dict(cfg["search"])
    search_cfg.update(dataset_cfg.get("search", {}))
    if method is not None:
        search_cfg.update(method.get("search", {}))
    return search_cfg


def result_query_dir_name(query_task: str, search_cfg: dict) -> str:
    if all(
        key in search_cfg
        for key in (
            "lsearch_prefix_start",
            "lsearch_prefix_end",
            "lsearch_prefix_step",
        )
    ):
        values = lsearch_values(search_cfg)
        if values and values[0] < int(search_cfg["lsearch_start"]):
            return (
                f"{query_task}_{values[0]}_"
                f"{search_cfg['lsearch_prefix_step']}_{values[-1]}"
            )
    return (
        f"{query_task}_{search_cfg['lsearch_start']}_"
        f"{search_cfg['lsearch_step']}_{search_cfg['lsearch_end']}"
    )


def lsearch_values(search_cfg: dict) -> list[int]:
    if search_cfg.get("lsearch_values"):
        values = [int(value) for value in search_cfg["lsearch_values"]]
    else:
        values = list(
            range(
                int(search_cfg["lsearch_start"]),
                int(search_cfg["lsearch_end"]) + 1,
                int(search_cfg["lsearch_step"]),
            )
        )

    prefix_keys = (
        "lsearch_prefix_start",
        "lsearch_prefix_end",
        "lsearch_prefix_step",
    )
    configured_prefix_keys = [key for key in prefix_keys if key in search_cfg]
    if not configured_prefix_keys:
        return values
    if len(configured_prefix_keys) != len(prefix_keys):
        missing = ", ".join(key for key in prefix_keys if key not in search_cfg)
        raise ValueError(f"incomplete Lsearch prefix configuration; missing: {missing}")

    prefix_start = int(search_cfg["lsearch_prefix_start"])
    prefix_end = int(search_cfg["lsearch_prefix_end"])
    prefix_step = int(search_cfg["lsearch_prefix_step"])
    if prefix_step <= 0:
        raise ValueError("lsearch_prefix_step must be positive")
    if prefix_start > prefix_end:
        raise ValueError("lsearch_prefix_start must not exceed lsearch_prefix_end")
    if not values:
        raise ValueError("at least one Lsearch value is required")

    # A prefix is only the part strictly before the method's main sweep. This
    # lets a method-level 10..100 override opt out naturally even when the
    # dataset's default search configuration has a 100..1000 prefix.
    first_main_value = min(values)
    prefix_values = [
        value
        for value in range(prefix_start, prefix_end + 1, prefix_step)
        if value < first_main_value
    ]
    return prefix_values + values



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
    search_cfg = effective_search_cfg(cfg, dataset_cfg, method)
    data_dir = data_root / dataset
    query_dir = data_dir / query_task
    index_dir = result_root / dataset / "index" / method["index_name"] / "index_files"
    if method.get("block_index_name"):
        block_index_dir = (
            result_root
            / dataset
            / "index"
            / method["block_index_name"]
            / "block_index_files"
        )
    else:
        block_index_dir = index_dir.parent / "block_index_files"
    result_query_task = result_query_dir_name(query_task, search_cfg)
    result_dir = result_root / dataset / "results" / method["name"] / result_query_task
    raw_results_dir = result_dir / "results"
    other_dir = result_dir / "others"
    raw_results_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)

    values = lsearch_values(search_cfg)
    summary_csv = raw_results_dir / "search_time_summary.csv"
    if should_skip_completed_search(summary_csv, values, cfg, dataset_cfg, method):
        log(
            f"[{dataset}][{method['name']}] existing results cover "
            f"{describe_search_values(values)}; skipping search: {summary_csv}"
        )
        return

    if not (index_dir / "meta").exists():
        raise FileNotFoundError(f"missing index meta: {index_dir / 'meta'}")

    query_group_file = query_dir / f"{dataset}_query_source_groups.txt"
    if not query_group_file.exists():
        query_group_file = result_dir / "missing_query_source_groups.txt"

    env = os.environ.copy()
    if method.get("special_block_search", False):
        env["UNG_SPECIAL_BLOCK_SEARCH"] = "1"
    else:
        for key in list(env):
            if key.startswith("UNG_SPECIAL_"):
                env.pop(key, None)

    for key, value in method.get("env", {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)

    reuse_els = method.get("reuse_els", True)
    if not isinstance(reuse_els, bool):
        raise TypeError(f"{method['name']}.reuse_els must be a boolean")
    if reuse_els:
        env.pop("UNG_DISABLE_ELS_REUSE", None)
    else:
        env["UNG_DISABLE_ELS_REUSE"] = "1"
        env["UNG_DISABLE_CPU_ELS_WARMUP"] = "1"
        env["UNG_DISABLE_SPECIAL_BLOCK_TRIE_WARMUP"] = "1"

    if reuse_els and method.get("entry_group_provider") == "special_block_trie":
        warmup_special_block_trie = method.get("warmup_special_block_trie", True)
        env["UNG_DISABLE_SPECIAL_BLOCK_TRIE_WARMUP"] = (
            "0" if warmup_special_block_trie else "1"
        )

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
        str(method.get("force_use_alg", 1)),
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
        *[str(v) for v in values],
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
    uses_block_index = (
        method.get("special_block_search", False)
        or method.get("entry_group_provider") == "special_block_trie"
    )
    if uses_block_index and (block_index_dir / "meta").exists():
        cmd.extend(["--block_index_path_prefix", str(block_index_dir) + "/"])
    elif method.get("entry_group_provider") == "special_block_trie" and not (
        index_dir / "special_block_trie.bin"
    ).exists():
        raise FileNotFoundError(
            f"missing independent or legacy Trie block index for {method['name']}: "
            f"{block_index_dir}"
        )
    if "scalar_els_cap" in method:
        cmd.extend(["--scalar_els_cap", str(method["scalar_els_cap"])])

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
        for method in cfg["methods"]:
            ensure_native_ung_index(cfg, dataset_cfg, method)
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
