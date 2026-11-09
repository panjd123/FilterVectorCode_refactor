#!/usr/bin/env python3

import csv
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from result_completion import describe_search_values, should_skip_completed_search


ADAPTER_FORMAT = "FilterVector RWalks adapter v1"


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def run_command(cmd, output_path: Path | None = None, env=None) -> None:
    command_line = "COMMAND: " + " ".join(str(item) for item in cmd)
    log(command_line)
    output = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = output_path.open("w", encoding="utf-8")
        output.write(command_line + "\n\n")
        output.flush()
    try:
        process = subprocess.Popen(
            [str(item) for item in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if output is not None:
                output.write(line)
                output.flush()
            print(line, end="", flush=True)
        process.stdout.close()
        returncode = process.wait()
    finally:
        if output is not None:
            output.close()
    if returncode != 0:
        detail = f", see {output_path}" if output_path is not None else ""
        raise RuntimeError(f"command failed ({returncode}){detail}")


def parse_label_line(line: str) -> list[int]:
    return [int(value) for value in re.split(r"[\s,]+", line.strip()) if value]


def labels_in_file(path: Path) -> set[int]:
    labels: set[int] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            labels.update(parse_label_line(line))
    return labels


def label_row_count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for _ in stream)


def fvecs_count(path: Path) -> int:
    with path.open("rb") as stream:
        dimension_bytes = stream.read(4)
    if len(dimension_bytes) != 4:
        raise RuntimeError(f"empty or truncated fvecs file: {path}")
    dimension = struct.unpack("<i", dimension_bytes)[0]
    if dimension <= 0:
        raise RuntimeError(f"invalid fvecs dimension {dimension}: {path}")
    record_bytes = 4 + dimension * 4
    file_bytes = path.stat().st_size
    if file_bytes % record_bytes:
        raise RuntimeError(f"fvecs size is not a whole number of records: {path}")
    return file_bytes // record_bytes


def fvecs_dimension(path: Path) -> int:
    with path.open("rb") as stream:
        dimension_bytes = stream.read(4)
    if len(dimension_bytes) != 4:
        raise RuntimeError(f"empty or truncated fvecs file: {path}")
    dimension = struct.unpack("<i", dimension_bytes)[0]
    if dimension <= 0:
        raise RuntimeError(f"invalid fvecs dimension {dimension}: {path}")
    return dimension


def prepare_rwalks_groundtruth_file(
    source_gt_file: Path,
    rwalks_gt_file: Path,
    topk: int,
    num_queries: int,
) -> Path:
    expected_ids = topk * num_queries
    expected_id_bytes = expected_ids * struct.calcsize("<i")
    expected_pair_bytes = expected_ids * struct.calcsize("<If")
    actual_bytes = source_gt_file.stat().st_size
    rwalks_gt_file.parent.mkdir(parents=True, exist_ok=True)
    if actual_bytes == expected_id_bytes:
        if source_gt_file != rwalks_gt_file:
            shutil.copyfile(source_gt_file, rwalks_gt_file)
        return rwalks_gt_file
    if actual_bytes != expected_pair_bytes:
        raise RuntimeError(
            f"groundtruth size mismatch: {source_gt_file} has {actual_bytes} bytes, "
            f"expected {expected_id_bytes} id bytes or {expected_pair_bytes} UNG pair bytes"
        )
    with source_gt_file.open("rb") as source, rwalks_gt_file.open("wb") as target:
        for _ in range(expected_ids):
            target.write(source.read(struct.calcsize("<If"))[:4])
    return rwalks_gt_file


def configured_methods(cfg: dict) -> list[dict]:
    methods = [dict(method) for method in cfg.get("methods", [])]
    if methods:
        return methods
    return [{"name": cfg.get("method_name", "rwalks"), "index_name": cfg.get("index_name", "RWalks")}]


def merged_build_cfg(cfg: dict, dataset_cfg: dict, method: dict) -> dict:
    result = {
        "num_threads": 32,
        "M": 16,
        "ef_construction": 100,
        "attr_depth": 3,
        "attr_steps": 20,
        "random_seed": 100,
        "attribute_scope": "configured_queries",
    }
    result.update(cfg.get("build", {}))
    result.update(dataset_cfg.get("build", {}))
    result.update(method.get("build", {}))
    return result


def merged_search_cfg(cfg: dict, dataset_cfg: dict, method: dict) -> dict:
    result = {
        "K": cfg.get("topk", 10),
        "num_repeats": 1,
        "num_threads": merged_build_cfg(cfg, dataset_cfg, method)["num_threads"],
        "search_mode": 0,
        "hybrid_factor": 0.0,
        "pruning_factor": 0.0,
        "lsearch_start": 10,
        "lsearch_end": 500,
        "lsearch_step": 50,
    }
    result.update(cfg.get("search", {}))
    result.update(dataset_cfg.get("search", {}))
    result.update(method.get("search", {}))
    return result


def search_values(cfg: dict, dataset_cfg: dict, method: dict, search_cfg: dict) -> list[int]:
    explicit = method.get("ef_values") or dataset_cfg.get("ef_values") or cfg.get("ef_values")
    if explicit:
        return [int(value) for value in explicit]
    return list(range(
        int(search_cfg["lsearch_start"]),
        int(search_cfg["lsearch_end"]) + 1,
        int(search_cfg["lsearch_step"]),
    ))


def result_query_dir_name(query_task: str, search_cfg: dict) -> str:
    return (
        f"{query_task}_{search_cfg['lsearch_start']}_"
        f"{search_cfg['lsearch_step']}_{search_cfg['lsearch_end']}"
    )


def resolve_dataset_files(cfg: dict, dataset_cfg: dict, method: dict) -> dict[str, Path | str]:
    dataset = dataset_cfg["dataset"]
    query_task = dataset_cfg["query_task"]
    data_dir = Path(cfg["data_root"]) / dataset
    result_root = Path(cfg["result_root"]) / dataset
    search_cfg = merged_search_cfg(cfg, dataset_cfg, method)
    result_task = result_query_dir_name(query_task, search_cfg)
    adapter_dir = (
        result_root / "results" / method.get("name", "rwalks") /
        result_task / "results"
    )
    index_name = method.get("index_name", dataset_cfg.get("index_name", cfg.get("index_name", "RWalks")))
    index_dir = result_root / "index" / index_name / "index_files"
    gt_default = result_root / "GroundTruth" / query_task / f"{dataset}_gt_labels_containment.bin"
    return {
        "dataset": dataset,
        "query_task": query_task,
        "base_fvecs": Path(dataset_cfg.get("base_fvecs", data_dir / f"{dataset}_base.fvecs")),
        "query_fvecs": Path(dataset_cfg.get("query_fvecs", data_dir / query_task / f"{dataset}_query.fvecs")),
        "base_labels": Path(dataset_cfg.get("base_labels", data_dir / f"{dataset}_base_labels.txt")),
        "query_labels": Path(dataset_cfg.get("query_labels", data_dir / query_task / f"{dataset}_query_labels.txt")),
        "gt_file": Path(dataset_cfg.get("groundtruth_file", gt_default)),
        "rwalks_gt_file": Path(dataset_cfg.get("rwalks_groundtruth_file", adapter_dir / f"{dataset}_rwalks_gt_ids.bin")),
        "index_file": Path(dataset_cfg.get("index_file", index_dir / "index.bin")),
        "index_attr_file": Path(dataset_cfg.get("index_attr_file", index_dir / "index_attr.bin")),
        "schema_file": Path(dataset_cfg.get("schema_file", index_dir / "index.rwalks.labels.txt")),
        "index_manifest": Path(dataset_cfg.get("index_manifest", index_dir / "index.rwalks.json")),
        "index_name": index_name,
    }


def ensure_input_files(paths: dict[str, Path | str]) -> None:
    for key in ["base_fvecs", "query_fvecs", "base_labels", "query_labels", "gt_file"]:
        path = paths[key]
        assert isinstance(path, Path)
        if not path.exists():
            raise FileNotFoundError(f"missing {key}: {path}")


def ensure_rwalks_binaries(cfg: dict) -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[2]
    source_dir = Path(cfg.get("rwalks_source_dir", project_root / "RWalks"))
    adapter_source_dir = source_dir / "project_adapter"
    build_dir = Path(cfg.get("rwalks_adapter_build_dir", adapter_source_dir / "build"))
    build_binary = Path(cfg.get("rwalks_build_binary", build_dir / "build_index"))
    search_binary = Path(cfg.get("rwalks_search_binary", build_dir / "search_sweep"))
    if bool(cfg.get("auto_build_rwalks", True)):
        build_dir.mkdir(parents=True, exist_ok=True)
        run_command([
            "cmake", "-S", adapter_source_dir, "-B", build_dir,
            "-DCMAKE_BUILD_TYPE=Release", f"-DRWALKS_ROOT={source_dir}",
        ])
        jobs = str(cfg.get("build_jobs", os.environ.get("BUILD_JOBS", 16)))
        run_command([
            "cmake", "--build", build_dir, f"-j{jobs}", "--target",
            "rwalks_build_index", "rwalks_search_sweep",
        ])
    if not build_binary.exists():
        raise FileNotFoundError(f"missing RWalks build binary: {build_binary}")
    if not search_binary.exists():
        raise FileNotFoundError(f"missing RWalks search binary: {search_binary}")
    return build_binary, search_binary


def file_signature(path: Path) -> dict[str, int | str]:
    status = path.stat()
    return {"path": str(path.resolve()), "size": status.st_size, "mtime_ns": status.st_mtime_ns}


def configured_schema_labels(cfg: dict, dataset_cfg: dict, method: dict) -> list[int]:
    build_cfg = merged_build_cfg(cfg, dataset_cfg, method)
    scope = build_cfg.get("attribute_scope", "configured_queries")
    if scope == "all_base_labels":
        return sorted(labels_in_file(resolve_dataset_files(cfg, dataset_cfg, method)["base_labels"]))
    if scope == "current_query":
        return sorted(labels_in_file(resolve_dataset_files(cfg, dataset_cfg, method)["query_labels"]))
    if scope != "configured_queries":
        raise ValueError(f"unsupported build.attribute_scope: {scope}")

    index_name = resolve_dataset_files(cfg, dataset_cfg, method)["index_name"]
    labels: set[int] = set()
    # A reduced search workload can retain the original index's attribute schema.
    for label_file in cfg.get("schema_query_label_files", []):
        labels.update(labels_in_file(Path(label_file)))
    for candidate in cfg["datasets"]:
        if candidate["dataset"] != dataset_cfg["dataset"]:
            continue
        candidate_paths = resolve_dataset_files(cfg, candidate, method)
        if candidate_paths["index_name"] == index_name:
            labels.update(labels_in_file(candidate_paths["query_labels"]))
    if not labels:
        raise RuntimeError("configured RWalks query-label schema is empty")
    return sorted(labels)


def schema_digest(labels: list[int]) -> str:
    payload = "".join(f"{label}\n" for label in labels).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_manifest(
    cfg: dict,
    dataset_cfg: dict,
    method: dict,
    paths: dict[str, Path | str],
    schema_labels: list[int],
) -> dict:
    build_cfg = merged_build_cfg(cfg, dataset_cfg, method)
    return {
        "format": ADAPTER_FORMAT,
        "dataset": paths["dataset"],
        "base_fvecs": file_signature(paths["base_fvecs"]),
        "base_labels": file_signature(paths["base_labels"]),
        "schema": {
            "scope": build_cfg["attribute_scope"],
            "dimension": len(schema_labels),
            "sha256": schema_digest(schema_labels),
        },
        "build": {
            "num_threads": int(build_cfg["num_threads"]),
            "M": int(build_cfg["M"]),
            "ef_construction": int(build_cfg["ef_construction"]),
            "attr_depth": int(build_cfg["attr_depth"]),
            "attr_steps": int(build_cfg["attr_steps"]),
            "random_seed": int(build_cfg["random_seed"]),
        },
    }


def index_is_current(paths: dict[str, Path | str], manifest: dict) -> bool:
    required = ["index_file", "index_attr_file", "schema_file", "index_manifest"]
    if not all(isinstance(paths[key], Path) and paths[key].exists() for key in required):
        return False
    try:
        actual = json.loads(paths["index_manifest"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if actual != manifest:
        return False
    try:
        schema_labels = sorted(labels_in_file(paths["schema_file"]))
    except (OSError, ValueError):
        return False
    return schema_digest(schema_labels) == manifest["schema"]["sha256"]


def parse_index_memory_stats(build_log: Path) -> dict[str, int | float]:
    text = build_log.read_text(encoding="utf-8")
    patterns = {
        "index_memory_bytes": r"index_memory_bytes=(\d+)",
        "index_memory_mib": r"index_memory_mib=([0-9.eE+-]+)",
        "index_level0_bytes": r"index_level0_bytes=(\d+)",
        "index_upper_level_bytes": r"index_upper_level_bytes=(\d+)",
        "index_runtime_metadata_bytes": r"index_runtime_metadata_bytes=(\d+)",
        "index_attribute_bytes": r"index_attribute_bytes=(\d+)",
    }
    stats: dict[str, int | float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise RuntimeError(f"missing RWalks memory statistic {key} in {build_log}")
        stats[key] = float(match.group(1)) if key == "index_memory_mib" else int(match.group(1))
    return stats


def append_index_build_time(
    output_path: Path,
    dataset: str,
    index_name: str,
    index_file: Path,
    build_time_seconds: float,
    memory: dict[str, int | float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "index_name", "build_time_seconds", "build_time_ms", "index_file",
        "index_memory_bytes", "index_memory_mib", "index_level0_bytes",
        "index_upper_level_bytes", "index_runtime_metadata_bytes", "index_attribute_bytes",
    ]
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "dataset": dataset,
            "index_name": index_name,
            "build_time_seconds": f"{build_time_seconds:.6f}",
            "build_time_ms": f"{build_time_seconds * 1000.0:.3f}",
            "index_file": str(index_file),
            **memory,
        })


def build_index(
    build_binary: Path,
    cfg: dict,
    dataset_cfg: dict,
    method: dict,
    paths: dict[str, Path | str],
) -> None:
    schema_labels = configured_schema_labels(cfg, dataset_cfg, method)
    manifest = expected_manifest(cfg, dataset_cfg, method, paths, schema_labels)
    if index_is_current(paths, manifest):
        log(f"[{paths['dataset']}] RWalks index exists and manifest matches: {paths['index_file']}")
        return

    build_cfg = merged_build_cfg(cfg, dataset_cfg, method)
    if str(build_cfg.get("dist_fn", "L2")).lower() not in {"l2", "euclidean"}:
        raise ValueError("the RWalks native adapter currently supports only L2 distance")
    index_file = paths["index_file"]
    index_attr_file = paths["index_attr_file"]
    schema_file = paths["schema_file"]
    index_manifest = paths["index_manifest"]
    assert all(isinstance(path, Path) for path in [index_file, index_attr_file, schema_file, index_manifest])
    index_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = index_file.with_name(index_file.stem + ".building.bin")
    temporary_attr = index_attr_file.with_name(index_file.stem + ".building_attr.bin")
    temporary_schema = schema_file.with_name(schema_file.name + ".building")
    temporary_schema.write_text("".join(f"{label}\n" for label in schema_labels), encoding="utf-8")
    build_log = index_file.parent.parent / "others" / "rwalks_build.log"
    log(
        f"[{paths['dataset']}] RWalks build start with {len(schema_labels)} compact attributes "
        f"({build_cfg['attribute_scope']})"
    )
    started = time.perf_counter()
    try:
        run_command([
            build_binary, paths["base_fvecs"], paths["base_labels"], temporary_schema,
            temporary_index, str(build_cfg["num_threads"]), str(build_cfg["M"]),
            str(build_cfg["ef_construction"]), str(build_cfg["attr_depth"]),
            str(build_cfg["attr_steps"]), str(build_cfg["random_seed"]),
        ], build_log)
        elapsed = time.perf_counter() - started
        if not temporary_index.exists() or not temporary_attr.exists():
            raise RuntimeError("RWalks build completed without both index files")
        os.replace(temporary_index, index_file)
        os.replace(temporary_attr, index_attr_file)
        os.replace(temporary_schema, schema_file)
        temporary_manifest = index_manifest.with_name(index_manifest.name + ".building")
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_manifest, index_manifest)
    finally:
        for temporary in [temporary_index, temporary_attr, temporary_schema]:
            if temporary.exists():
                temporary.unlink()
    memory = parse_index_memory_stats(build_log)
    append_index_build_time(
        index_file.parent.parent / "others" / "index_build_time.csv",
        str(paths["dataset"]), str(paths["index_name"]), index_file, elapsed, memory,
    )
    log(f"[{paths['dataset']}] RWalks build done in {elapsed:.3f}s")


def run_searches(
    search_binary: Path,
    cfg: dict,
    dataset_cfg: dict,
    method: dict,
    paths: dict[str, Path | str],
) -> None:
    search_cfg = merged_search_cfg(cfg, dataset_cfg, method)
    ef_values = search_values(cfg, dataset_cfg, method, search_cfg)
    topk = int(search_cfg["K"])
    query_count = fvecs_count(paths["query_fvecs"])
    if fvecs_dimension(paths["query_fvecs"]) != fvecs_dimension(paths["base_fvecs"]):
        raise RuntimeError("query vector dimension does not match the indexed base vectors")
    if label_row_count(paths["query_labels"]) != query_count:
        raise RuntimeError(f"query label count does not match query fvecs: {paths['query_labels']}")
    groundtruth = prepare_rwalks_groundtruth_file(
        paths["gt_file"], paths["rwalks_gt_file"], topk, query_count)
    result_task = result_query_dir_name(str(paths["query_task"]), search_cfg)
    result_dir = (
        Path(cfg["result_root"]) / str(paths["dataset"]) / "results" /
        method.get("name", "rwalks") / result_task
    )
    raw_results = result_dir / "results"
    log_path = result_dir / "others" / f"{paths['dataset']}_search_output.txt"
    summary_csv = raw_results / "search_time_summary.csv"
    if should_skip_completed_search(summary_csv, ef_values, cfg, dataset_cfg, method):
        log(
            f"[{paths['dataset']}][{method.get('name', 'rwalks')}] existing results cover "
            f"{describe_search_values(ef_values)}; skipping search: {summary_csv}"
        )
        return
    command = [
        search_binary, paths["query_fvecs"], paths["query_labels"], paths["schema_file"],
        groundtruth, paths["index_file"], raw_results, str(topk),
        str(search_cfg["num_repeats"]), str(search_cfg["num_threads"]),
        str(search_cfg["search_mode"]), str(search_cfg["hybrid_factor"]),
        str(search_cfg["pruning_factor"]),
    ]
    command.extend(str(value) for value in ef_values)
    run_command(command, log_path)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_rwalks_experiment.py <config.json>", file=sys.stderr)
        return 2
    config_path = Path(sys.argv[1])
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    log(f"config: {config_path}")
    build_binary, search_binary = ensure_rwalks_binaries(cfg)
    for dataset_cfg in cfg["datasets"]:
        for method in configured_methods(cfg):
            paths = resolve_dataset_files(cfg, dataset_cfg, method)
            ensure_input_files(paths)
            build_index(build_binary, cfg, dataset_cfg, method, paths)
            run_searches(search_binary, cfg, dataset_cfg, method, paths)
    log("all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
