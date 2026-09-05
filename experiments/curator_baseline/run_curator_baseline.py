#!/usr/bin/env python3

import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from curator_backend import CuratorBackend
from curator_project_adapter import (
    build_label_inverted_index,
    load_groundtruth,
    load_label_sets,
    load_project_bin,
    qualified_ids_for_containment,
    qualified_ids_for_containment_indexed,
    recall_at_k,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def configure_threads(num_threads: int) -> None:
    threads = max(1, int(num_threads))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    try:
        import faiss  # type: ignore

        if hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(threads)
    except Exception:
        return


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


def merged_build_cfg(cfg: dict, dataset_cfg: dict) -> dict:
    build_cfg = {
        "nlist": 32,
        "bf_capacity": 1000,
        "bf_error_rate": 0.001,
        "max_sl_size": 128,
        "clus_niter": 20,
        "max_leaf_size": 128,
        "prune_thres": 1.6,
        "variance_boost": 0.2,
        "search_ef": 64,
        "beam_size": 4,
        "use_temp_index_caching": False,
        "num_threads": 1,
    }
    build_cfg.update(cfg.get("build", {}))
    build_cfg.update(dataset_cfg.get("build", {}))
    return build_cfg


def merged_search_cfg(cfg: dict, dataset_cfg: dict) -> dict:
    search_cfg = {
        "K": 10,
        "num_repeats": 1,
        "num_threads": cfg.get("build", {}).get("num_threads", 1),
        "search_parameter": "search_ef",
        "ef_values": [64, 128, 256, 512, 1024],
        "filter_mode": "indexed",
    }
    search_cfg.update(cfg.get("search", {}))
    search_cfg.update(dataset_cfg.get("search", {}))
    return search_cfg


def search_budget_values(search_cfg: dict) -> list[int]:
    if "ef_values" in search_cfg:
        return [int(value) for value in search_cfg["ef_values"]]
    if all(key in search_cfg for key in ("ef_start", "ef_end", "ef_step")):
        return list(
            range(
                int(search_cfg["ef_start"]),
                int(search_cfg["ef_end"]) + 1,
                int(search_cfg["ef_step"]),
            )
        )
    if "nprobe_values" in search_cfg:
        return [int(value) for value in search_cfg["nprobe_values"]]
    return list(
        range(
            int(search_cfg.get("nprobe_start", 100)),
            int(search_cfg.get("nprobe_end", 1000)) + 1,
            int(search_cfg.get("nprobe_step", 100)),
        )
    )


def search_budget_parameter(search_cfg: dict) -> str:
    if "search_parameter" in search_cfg:
        return str(search_cfg["search_parameter"])
    return "nprobe" if any(str(key).startswith("nprobe") for key in search_cfg) else "search_ef"


def result_query_dir_name(query_task: str, search_cfg: dict) -> str:
    values = search_budget_values(search_cfg)
    if not values:
        raise ValueError("at least one Curator search budget value is required")
    parameter = search_budget_parameter(search_cfg)
    if len(values) == 1:
        return f"{query_task}_{parameter}{values[0]}"
    return f"{query_task}_{parameter}{values[0]}_{parameter}{values[-1]}"


def dataset_paths(cfg: dict, dataset_cfg: dict, search_cfg: dict) -> dict[str, Path | str]:
    dataset = dataset_cfg["dataset"]
    query_task = dataset_cfg["query_task"]
    data_dir = Path(cfg["data_root"]) / dataset
    query_dir = data_dir / query_task
    result_dataset_dir = Path(cfg["result_root"]) / dataset
    index_name = dataset_cfg.get("index_name", cfg.get("index_name", "Curator"))
    result_method = dataset_cfg.get("method_name", cfg.get("method_name", "Curator"))
    result_task = result_query_dir_name(query_task, search_cfg)
    return {
        "dataset": dataset,
        "query_task": query_task,
        "data_dir": data_dir,
        "query_dir": query_dir,
        "base_bin": Path(dataset_cfg.get("base_bin_file", data_dir / f"{dataset}_base.bin")),
        "base_fvecs": Path(dataset_cfg.get("base_fvecs", data_dir / f"{dataset}_base.fvecs")),
        "base_labels": Path(dataset_cfg.get("base_label_file", data_dir / f"{dataset}_base_labels.txt")),
        "query_bin": Path(dataset_cfg.get("query_bin_file", query_dir / f"{dataset}_query.bin")),
        "query_fvecs": Path(dataset_cfg.get("query_fvecs", query_dir / f"{dataset}_query.fvecs")),
        "query_labels": Path(dataset_cfg.get("query_label_file", query_dir / f"{dataset}_query_labels.txt")),
        "gt_file": Path(
            dataset_cfg.get(
                "groundtruth_file",
                result_dataset_dir
                / "GroundTruth"
                / query_task
                / f"{dataset}_gt_labels_containment.bin",
            )
        ),
        "index_dir": Path(
            dataset_cfg.get(
                "index_path_prefix",
                result_dataset_dir / "index" / index_name / "index_files",
            )
        ),
        "result_dir": Path(
            dataset_cfg.get(
                "result_path_prefix",
                result_dataset_dir / "results" / result_method / result_task / "results",
            )
        ),
        "other_dir": Path(
            dataset_cfg.get(
                "other_path_prefix",
                result_dataset_dir / "results" / result_method / result_task / "others",
            )
        ),
    }


def ensure_binaries(cfg: dict) -> tuple[Path, Path]:
    source_dir = Path(cfg.get("source_dir", Path(__file__).resolve().parents[2] / "UNG" / "codes"))
    build_dir = Path(cfg.get("build_dir", Path(__file__).resolve().parents[2] / "build_ung_rel"))
    auto_compile = bool(cfg.get("auto_compile", False))
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
                "compute_groundtruth",
                "fvecs_to_bin",
            ]
        )
    return build_dir / "tools" / "compute_groundtruth", build_dir / "tools" / "fvecs_to_bin"


def ensure_query_bin(fvecs_to_bin: Path, cfg: dict, paths: dict[str, Path | str]) -> Path:
    query_bin = paths["query_bin"]
    query_fvecs = paths["query_fvecs"]
    assert isinstance(query_bin, Path)
    assert isinstance(query_fvecs, Path)
    if query_bin.exists():
        return query_bin
    if not bool(cfg.get("allow_fvecs_conversion", True)):
        raise FileNotFoundError(f"missing query bin: {query_bin}")
    if not fvecs_to_bin.exists():
        raise FileNotFoundError(f"missing fvecs_to_bin binary: {fvecs_to_bin}")
    run_command(
        [
            fvecs_to_bin,
            "--data_type",
            cfg.get("data_type", "float"),
            "--input_file",
            query_fvecs,
            "--output_file",
            query_bin,
        ]
    )
    return query_bin


def ensure_gt(
    compute_gt_bin: Path, cfg: dict, paths: dict[str, Path | str], query_bin: Path, search_cfg: dict
) -> Path:
    gt_file = paths["gt_file"]
    assert isinstance(gt_file, Path)
    if gt_file.exists():
        return gt_file
    if not bool(cfg.get("allow_gt_generation", True)):
        raise FileNotFoundError(f"missing groundtruth file: {gt_file}")
    if not compute_gt_bin.exists():
        raise FileNotFoundError(f"missing compute_groundtruth binary: {compute_gt_bin}")
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
            paths["base_bin"],
            "--base_label_file",
            paths["base_labels"],
            "--query_bin_file",
            query_bin,
            "--query_label_file",
            paths["query_labels"],
            "--gt_file",
            gt_file,
        ],
        paths["other_dir"] / f"{paths['dataset']}_compute_gt.log",
    )
    return gt_file


def recall_for_query(result_ids: np.ndarray, groundtruth_ids: np.ndarray, k: int) -> float:
    limit = min(k, result_ids.shape[0], groundtruth_ids.shape[0])
    expected = {int(x) for x in groundtruth_ids[:limit] if int(x) >= 0}
    if not expected:
        return 0.0
    actual = {int(x) for x in result_ids[:limit] if int(x) >= 0}
    return len(actual.intersection(expected)) / float(len(expected))


def append_result_row(
    rows: list[dict[str, Any]],
    search_ef: int,
    qid: int,
    result_ids: np.ndarray,
    query_recall: float,
    search_time_ms: float,
    profile: dict[str, Any],
) -> None:
    rows.append(
        {
            "search_ef": int(search_ef),
            "QueryID": int(qid),
            "Recall": f"{query_recall:.6f}",
            "Search_Time_ms": f"{search_time_ms:.6f}",
            "VisitedPoints": int(profile["visited_points"]),
            "VisitedEdges": int(profile["visited_edges"]),
            "DistanceComputations": int(profile["distance_computations"]),
            "FilterLookup_Time_ms": f"{float(profile.get('filter_lookup_time_ms', 0.0)):.6f}",
            "PrepareFilter_Time_ms": f"{float(profile.get('prepare_filter_time_ms', 0.0)):.6f}",
            "BackendSearch_Time_ms": f"{float(profile.get('backend_search_time_ms', 0.0)):.6f}",
            "BuildTempIndex_Time_ms": f"{float(profile.get('build_temp_index_time_ms', 0.0)):.6f}",
            "CuratorSearch_Time_ms": f"{float(profile.get('search_time_ms', 0.0)):.6f}",
            "ResultIDs": ";".join(str(int(value)) for value in result_ids),
        }
    )


def write_result_file(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "search_ef",
                "QueryID",
                "Recall",
                "Search_Time_ms",
                "VisitedPoints",
                "VisitedEdges",
                "DistanceComputations",
                "FilterLookup_Time_ms",
                "PrepareFilter_Time_ms",
                "BackendSearch_Time_ms",
                "BuildTempIndex_Time_ms",
                "CuratorSearch_Time_ms",
                "ResultIDs",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(result_dir: Path, rows: list[dict[str, Any]]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Lsearch",
        "Recall",
        "Average_Time_ms",
        "Average_VisitedPoints",
        "Average_VisitedEdges",
        "Average_DistanceComputations",
    ]
    summary_csv = result_dir / "search_time_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    qps_rows = []
    for row in rows:
        qps_row = {key: row[key] for key in fieldnames}
        avg_ms = float(row["Average_Time_ms"])
        num_queries = int(row["Num_Queries"])
        qps_row["QPS"] = f"{(num_queries * 1000.0 / avg_ms) if avg_ms > 0 else 0.0:.6f}"
        qps_rows.append(qps_row)
    with (result_dir / "search_time_summary_qps.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + ["QPS"])
        writer.writeheader()
        writer.writerows(qps_rows)


def write_repeat_details(result_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Persist every measured batch wall time instead of only their mean.

    Keeping the raw repeats is important for highly parallel query batches: CPU
    scheduling and cache effects can otherwise make a single mean look like an
    algorithmic trend.  Warmup batches are intentionally excluded from this
    file and identified separately by ``Warmup_Repeats``.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "search_parameter",
        "Lsearch",
        "Repeat",
        "Warmup_Repeats",
        "Num_Queries",
        "Total_Time_ms",
        "QPS",
    ]
    with (result_dir / "search_time_details.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def read_curator_meta(index_dir: Path) -> dict[str, Any] | None:
    meta_path = index_dir / "curator_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def curator_meta_matches(index_dir: Path, build_cfg: dict, num_points: int, dim: int) -> bool:
    artifact_path = index_dir / "curator.index"
    meta = read_curator_meta(index_dir)
    if meta is None or not artifact_path.exists():
        return False
    expected_build = dict(build_cfg)
    actual_build = dict(meta.get("build", {}))
    return (
        meta.get("index_name") == "Curator"
        and int(meta.get("num_points", -1)) == int(num_points)
        and int(meta.get("dim", -1)) == int(dim)
        and bool(meta.get("persistent", False))
        and actual_build == expected_build
    )


def write_build_stats(build_dir: Path, build_ms: float, memory_bytes: int, disk_bytes: int, reused: bool) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    with (build_dir / "build_time.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index Name", "Build Time (ms)", "Reused Existing Index"])
        writer.writerow(["Curator", f"{build_ms:.6f}", str(bool(reused)).lower()])
    with (build_dir / "index_size.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index Name", "Memory Estimate Bytes", "Disk Bytes"])
        writer.writerow(["Curator", int(memory_bytes), int(disk_bytes)])


def write_index_meta(index_dir: Path, cfg: dict, build_cfg: dict, num_points: int, dim: int, build_ms: float, memory_bytes: int, disk_bytes: int, persistent: bool) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "index_name": "Curator",
        "num_points": num_points,
        "dim": dim,
        "build_ms": build_ms,
        "memory_bytes": int(memory_bytes),
        "disk_bytes": int(disk_bytes),
        "persistent": bool(persistent),
        "build": build_cfg,
        "curator_repo": cfg.get("curator_repo", ""),
    }
    (index_dir / "curator_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")



def run_dataset(cfg: dict, dataset_cfg: dict) -> None:
    search_cfg = merged_search_cfg(cfg, dataset_cfg)
    build_cfg = merged_build_cfg(cfg, dataset_cfg)
    paths = dataset_paths(cfg, dataset_cfg, search_cfg)
    compute_gt_bin, fvecs_to_bin = ensure_binaries(cfg)
    query_bin = ensure_query_bin(fvecs_to_bin, cfg, paths)
    gt_file = ensure_gt(compute_gt_bin, cfg, paths, query_bin, search_cfg)

    base_vectors = load_project_bin(paths["base_bin"])
    query_vectors = load_project_bin(query_bin)
    base_labels = load_label_sets(paths["base_labels"])
    query_labels = load_label_sets(paths["query_labels"])
    if len(base_labels) != base_vectors.shape[0]:
        raise RuntimeError(
            f"base vector/label count mismatch: {base_vectors.shape[0]} vectors, {len(base_labels)} labels"
        )
    if len(query_labels) != query_vectors.shape[0]:
        raise RuntimeError(
            f"query vector/label count mismatch: {query_vectors.shape[0]} vectors, {len(query_labels)} labels"
        )

    backend_cfg = dict(cfg)
    backend_cfg["build"] = build_cfg
    index_dir = paths["index_dir"]
    assert isinstance(index_dir, Path)
    build_stats_dir = index_dir.parent / "build"
    build_threads = max(1, int(build_cfg.get("num_threads", 1)))
    configure_threads(build_threads)
    backend = CuratorBackend.from_config(base_vectors.shape[1], backend_cfg)
    backend.enable_stats_tracking(True)
    reused_index = False
    build_ms = 0.0
    persistent = False
    if curator_meta_matches(index_dir, build_cfg, base_vectors.shape[0], base_vectors.shape[1]):
        backend.load(index_dir)
        reused_index = True
        existing_meta = read_curator_meta(index_dir) or {}
        build_ms = float(existing_meta.get("build_ms", 0.0) or 0.0)
    else:
        build_start = time.perf_counter()
        backend.build(base_vectors, np.arange(base_vectors.shape[0], dtype=np.uint32))
        build_ms = (time.perf_counter() - build_start) * 1000.0
        try:
            backend.save(index_dir)
            persistent = True
        except RuntimeError as exc:
            log(f"[{paths['dataset']}] Curator index persistence unavailable: {exc}")
    memory_bytes = backend.memory_usage_bytes() if hasattr(backend, "memory_usage_bytes") else 0
    if memory_bytes <= 0:
        memory_bytes = int(base_vectors.nbytes)
    write_index_meta(index_dir, cfg, build_cfg, base_vectors.shape[0], base_vectors.shape[1], build_ms, memory_bytes, 0, persistent or reused_index)
    disk_bytes = directory_size_bytes(index_dir)
    write_index_meta(index_dir, cfg, build_cfg, base_vectors.shape[0], base_vectors.shape[1], build_ms, memory_bytes, disk_bytes, persistent or reused_index)
    write_build_stats(build_stats_dir, build_ms, memory_bytes, disk_bytes, reused_index)

    k = int(search_cfg["K"])
    num_threads = max(1, int(search_cfg.get("num_threads", 1)))
    configure_threads(num_threads)
    groundtruth = load_groundtruth(gt_file, k)
    max_queries = int(search_cfg.get("max_queries", 0) or 0)
    if max_queries > 0:
        query_vectors = query_vectors[:max_queries]
        query_labels = query_labels[:max_queries]
        groundtruth = groundtruth[:max_queries]
    filter_mode = str(search_cfg.get("filter_mode", "indexed")).lower()
    if filter_mode not in {"indexed", "scan"}:
        raise ValueError(f"unsupported Curator filter_mode: {filter_mode}")
    label_index = build_label_inverted_index(base_labels) if filter_mode == "indexed" else None

    summary_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    result_dir = paths["result_dir"]
    other_dir = paths["other_dir"]
    assert isinstance(result_dir, Path)
    assert isinstance(other_dir, Path)
    other_dir.mkdir(parents=True, exist_ok=True)

    budget_parameter = search_budget_parameter(search_cfg)
    for search_budget in search_budget_values(search_cfg):
        backend.set_search_budget(search_budget, budget_parameter)
        best_results = np.full((query_vectors.shape[0], k), -1, dtype=np.int64)
        repeat_total_ms = 0.0
        best_query_times_ms = np.zeros(query_vectors.shape[0], dtype=np.float64)
        best_profiles: list[dict[str, Any]] = []
        total_visited_points = 0
        total_visited_edges = 0
        total_distance_computations = 0
        num_repeats = int(search_cfg["num_repeats"])
        warmup_repeats = max(0, int(search_cfg.get("num_warmup_repeats", 0)))
        for run_index in range(warmup_repeats + num_repeats):
            results = np.full((query_vectors.shape[0], k), -1, dtype=np.int64)

            def search_one(qid: int) -> tuple[int, np.ndarray, float, dict[str, Any]]:
                query_start = time.perf_counter()
                filter_start = query_start
                if filter_mode == "indexed":
                    assert label_index is not None
                    qualified = qualified_ids_for_containment_indexed(label_index, len(base_labels), query_labels[qid])
                else:
                    qualified = qualified_ids_for_containment(base_labels, query_labels[qid])
                filter_end = time.perf_counter()
                if hasattr(backend, "prepare_filter"):
                    qualified_filter = backend.prepare_filter(qualified)
                else:
                    qualified_filter = qualified
                prepare_end = time.perf_counter()
                result_ids, profile = backend.search(query_vectors[qid], k, qualified_filter)
                search_end = time.perf_counter()
                query_ms = (search_end - query_start) * 1000.0
                profile = dict(profile)
                profile["filter_lookup_time_ms"] = (filter_end - filter_start) * 1000.0
                profile["prepare_filter_time_ms"] = (prepare_end - filter_end) * 1000.0
                profile["backend_search_time_ms"] = (search_end - prepare_end) * 1000.0
                return qid, result_ids, query_ms, profile

            start = time.perf_counter()
            query_times_ms = np.zeros(query_vectors.shape[0], dtype=np.float64)
            query_profiles: list[dict[str, Any]] = [dict() for _ in range(query_vectors.shape[0])]
            if num_threads > 1 and query_vectors.shape[0] > 1:
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    for qid, result_ids, query_ms, profile in executor.map(search_one, range(query_vectors.shape[0])):
                        results[qid, :] = result_ids
                        query_times_ms[qid] = query_ms
                        query_profiles[qid] = profile
            else:
                for qid in range(query_vectors.shape[0]):
                    qid, result_ids, query_ms, profile = search_one(qid)
                    results[qid, :] = result_ids
                    query_times_ms[qid] = query_ms
                    query_profiles[qid] = profile
            batch_ms = (time.perf_counter() - start) * 1000.0
            if run_index < warmup_repeats:
                continue
            measured_repeat = run_index - warmup_repeats
            total_visited_points += sum(int(profile["visited_points"]) for profile in query_profiles)
            total_visited_edges += sum(int(profile["visited_edges"]) for profile in query_profiles)
            total_distance_computations += sum(
                int(profile["distance_computations"]) for profile in query_profiles
            )
            repeat_total_ms += batch_ms
            repeat_rows.append(
                {
                    "search_parameter": budget_parameter,
                    "Lsearch": int(search_budget),
                    "Repeat": measured_repeat,
                    "Warmup_Repeats": warmup_repeats,
                    "Num_Queries": query_vectors.shape[0],
                    "Total_Time_ms": f"{batch_ms:.6f}",
                    "QPS": f"{(query_vectors.shape[0] * 1000.0 / batch_ms) if batch_ms > 0 else 0.0:.6f}",
                }
            )
            best_results = results
            if measured_repeat == num_repeats - 1:
                best_query_times_ms = query_times_ms
                best_profiles = query_profiles

        avg_time_ms = repeat_total_ms / float(max(1, num_repeats))
        recall = recall_at_k(best_results, groundtruth, k)
        for qid in range(best_results.shape[0]):
            append_result_row(
                result_rows,
                search_budget,
                qid,
                best_results[qid],
                recall_for_query(best_results[qid], groundtruth[qid], k),
                float(best_query_times_ms[qid]),
                best_profiles[qid],
            )
        measurement_count = max(
            1, query_vectors.shape[0] * num_repeats
        )
        summary_rows.append(
            {
                "Dataset": paths["dataset"],
                "Method": paths.get("method_name", cfg.get("method_name", "Curator")),
                "K": str(k),
                "Lsearch": str(search_budget),
                "nprobe": str(search_budget),
                "ef": str(search_budget),
                "search_parameter": budget_parameter,
                "Recall": f"{recall:.6f}",
                "Average_Time_ms": f"{avg_time_ms:.6f}",
                "Total_Time_ms": f"{repeat_total_ms:.6f}",
                "Num_Queries": str(query_vectors.shape[0]),
                "Average_VisitedPoints": f"{total_visited_points / measurement_count:.6f}",
                "Average_VisitedEdges": f"{total_visited_edges / measurement_count:.6f}",
                "Average_DistanceComputations": f"{total_distance_computations / measurement_count:.6f}",
            }
        )

    write_result_file(result_dir / "curator_results.csv", result_rows)
    write_summary(result_dir, summary_rows)
    write_repeat_details(result_dir, repeat_rows)
    log(f"[{paths['dataset']}] Curator baseline done")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_curator_baseline.py <config.json>", file=sys.stderr)
        return 2
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    log(f"config: {cfg_path}")
    for dataset_cfg in cfg.get("datasets", []):
        run_dataset(cfg, dataset_cfg)
    log("all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
