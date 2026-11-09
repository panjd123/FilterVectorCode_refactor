#!/usr/bin/env python3

import csv
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


ADAPTER_FORMAT = "FilterVector Milvus HNSW adapter v1"


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def run_command(command, output_path: Path | None = None) -> None:
    command = [str(item) for item in command]
    command_line = "COMMAND: " + " ".join(command)
    log(command_line)
    output = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = output_path.open("w", encoding="utf-8")
        output.write(command_line + "\n\n")
        output.flush()
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
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
        detail = f", see {output_path}" if output_path else ""
        raise RuntimeError(f"command failed ({returncode}){detail}")


def fvecs_metadata(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        raw_dimension = stream.read(4)
    if len(raw_dimension) != 4:
        raise RuntimeError(f"empty or truncated fvecs file: {path}")
    dimension = struct.unpack("<i", raw_dimension)[0]
    if dimension <= 0:
        raise RuntimeError(f"invalid fvecs dimension {dimension}: {path}")
    record_bytes = 4 + dimension * 4
    size = path.stat().st_size
    if size % record_bytes:
        raise RuntimeError(f"fvecs size is not a whole number of records: {path}")
    return size // record_bytes, dimension


def label_row_count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for _ in stream)


def prepare_groundtruth(source: Path, target: Path, topk: int, query_count: int) -> Path:
    expected = topk * query_count
    id_bytes = expected * struct.calcsize("<i")
    pair_bytes = expected * struct.calcsize("<If")
    actual = source.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    if actual == id_bytes:
        if source != target:
            shutil.copyfile(source, target)
        return target
    if actual != pair_bytes:
        raise RuntimeError(
            f"groundtruth size mismatch: {source} has {actual} bytes; "
            f"expected {id_bytes} id bytes or {pair_bytes} UNG pair bytes"
        )
    with source.open("rb") as input_stream, target.open("wb") as output_stream:
        for _ in range(expected):
            record = input_stream.read(8)
            output_stream.write(record[:4])
    return target


def methods(cfg: dict) -> list[dict]:
    configured = [dict(item) for item in cfg.get("methods", [])]
    return configured or [{
        "name": cfg.get("method_name", "milvus_hnsw"),
        "index_name": cfg.get("index_name", "Milvus_HNSW_default"),
    }]


def build_config(cfg: dict, dataset_cfg: dict, method: dict) -> dict:
    result = {
        "data_type": "float", "dist_fn": "L2", "scenario": "containment",
        "num_threads": 32, "M": 30, "ef_construction": 360,
        "add_batch_size": 0,
    }
    result.update(cfg.get("build", {}))
    result.update(dataset_cfg.get("build", {}))
    result.update(method.get("build", {}))
    return result


def search_config(cfg: dict, dataset_cfg: dict, method: dict) -> dict:
    result = {
        "K": cfg.get("topk", 10), "num_repeats": 1,
        "num_threads": build_config(cfg, dataset_cfg, method)["num_threads"],
        "disable_fallback_brute_force": False,
        "lsearch_start": 100, "lsearch_end": 2000, "lsearch_step": 100,
    }
    result.update(cfg.get("search", {}))
    result.update(dataset_cfg.get("search", {}))
    result.update(method.get("search", {}))
    return result


def ef_values(cfg: dict, dataset_cfg: dict, method: dict, search_cfg: dict) -> list[int]:
    explicit = method.get("ef_values") or dataset_cfg.get("ef_values") or cfg.get("ef_values")
    if explicit:
        return [int(value) for value in explicit]
    return list(range(
        int(search_cfg["lsearch_start"]), int(search_cfg["lsearch_end"]) + 1,
        int(search_cfg["lsearch_step"]),
    ))


def result_task_name(query_task: str, search_cfg: dict) -> str:
    return (
        f"{query_task}_{search_cfg['lsearch_start']}_"
        f"{search_cfg['lsearch_step']}_{search_cfg['lsearch_end']}"
    )


def resolve_paths(cfg: dict, dataset_cfg: dict, method: dict) -> dict[str, Path | str]:
    dataset = dataset_cfg["dataset"]
    task = dataset_cfg["query_task"]
    data_dir = Path(cfg["data_root"]) / dataset
    result_dataset = Path(cfg["result_root"]) / dataset
    index_name = method.get("index_name", "Milvus_HNSW_default")
    index_files = result_dataset / "index" / index_name / "index_files"
    gt_default = result_dataset / "GroundTruth" / task / f"{dataset}_gt_labels_containment.bin"
    search = search_config(cfg, dataset_cfg, method)
    result_task = result_task_name(task, search)
    adapter_dir = (
        result_dataset / "results" / method.get("name", "milvus_hnsw") /
        result_task / "results"
    )
    return {
        "dataset": dataset, "query_task": task, "index_name": index_name,
        "base_fvecs": Path(dataset_cfg.get("base_fvecs", data_dir / f"{dataset}_base.fvecs")),
        "base_labels": Path(dataset_cfg.get("base_labels", data_dir / f"{dataset}_base_labels.txt")),
        "query_fvecs": Path(dataset_cfg.get("query_fvecs", data_dir / task / f"{dataset}_query.fvecs")),
        "query_labels": Path(dataset_cfg.get("query_labels", data_dir / task / f"{dataset}_query_labels.txt")),
        "groundtruth": Path(dataset_cfg.get("groundtruth_file", gt_default)),
        "adapter_groundtruth": Path(dataset_cfg.get(
            "milvus_groundtruth_file", adapter_dir / f"{dataset}_milvus_hnsw_gt_ids.bin")),
        "index_file": Path(dataset_cfg.get("index_file", index_files / "index.bin")),
        "filter_file": Path(dataset_cfg.get("filter_file", index_files / "filters.bin")),
        "manifest": Path(dataset_cfg.get("index_manifest", index_files / "index.milvus_hnsw.json")),
    }


def ensure_inputs(paths: dict[str, Path | str]) -> None:
    for key in ["base_fvecs", "base_labels", "query_fvecs", "query_labels", "groundtruth"]:
        path = paths[key]
        assert isinstance(path, Path)
        if not path.exists():
            raise FileNotFoundError(f"missing {key}: {path}")


def ensure_binaries(cfg: dict) -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[2]
    source = Path(cfg.get("knowhere_source_dir", project_root / "Knowhere"))
    faiss_build = Path(cfg.get("knowhere_faiss_build_dir", source / "thirdparty/faiss/build_adapter"))
    adapter_source = source / "project_adapter"
    adapter_build = Path(cfg.get("knowhere_adapter_build_dir", adapter_source / "build"))
    build_binary = Path(cfg.get("milvus_hnsw_build_binary", adapter_build / "build_index"))
    search_binary = Path(cfg.get("milvus_hnsw_search_binary", adapter_build / "search_sweep"))
    if bool(cfg.get("auto_build_milvus_hnsw", True)):
        faiss_build.mkdir(parents=True, exist_ok=True)
        openblas = Path(cfg.get("openblas_library", "/home/dev/graphdb/openblas/lib/libopenblas.so"))
        run_command([
            "cmake", "-S", source / "thirdparty/faiss", "-B", faiss_build,
            "-DCMAKE_BUILD_TYPE=Release", "-DFAISS_ENABLE_GPU=OFF",
            "-DFAISS_ENABLE_PYTHON=OFF", "-DFAISS_ENABLE_C_API=OFF",
            "-DFAISS_ENABLE_EXTRAS=OFF", "-DBUILD_TESTING=OFF",
            "-DFAISS_OPT_LEVEL=avx2", "-DFAISS_ENABLE_MKL=OFF",
            "-DBUILD_SHARED_LIBS=ON", f"-DBLAS_LIBRARIES={openblas}",
            f"-DLAPACK_LIBRARIES={openblas}",
        ])
        jobs = str(cfg.get("build_jobs", os.environ.get("BUILD_JOBS", 16)))
        run_command(["cmake", "--build", faiss_build, f"-j{jobs}", "--target", "faiss"])
        adapter_build.mkdir(parents=True, exist_ok=True)
        run_command([
            "cmake", "-S", adapter_source, "-B", adapter_build,
            "-DCMAKE_BUILD_TYPE=Release", f"-DKNOWHERE_ROOT={source}",
            f"-DKNOWHERE_FAISS_BUILD_DIR={faiss_build}", f"-DOPENBLAS_LIBRARY={openblas}",
        ])
        run_command([
            "cmake", "--build", adapter_build, f"-j{jobs}", "--target",
            "milvus_hnsw_build_index", "milvus_hnsw_search_sweep",
        ])
    if not build_binary.exists():
        raise FileNotFoundError(f"missing Milvus HNSW build binary: {build_binary}")
    if not search_binary.exists():
        raise FileNotFoundError(f"missing Milvus HNSW search binary: {search_binary}")
    return build_binary, search_binary


def signature(path: Path) -> dict[str, int | str]:
    status = path.stat()
    return {"path": str(path.resolve()), "size": status.st_size, "mtime_ns": status.st_mtime_ns}


def knowhere_commit(cfg: dict) -> str:
    source = Path(cfg.get("knowhere_source_dir", Path(__file__).resolve().parents[2] / "Knowhere"))
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        marker = Path(__file__).resolve().parents[2] / ".knowhere_source_commit"
        if marker.exists():
            for line in marker.read_text(encoding="utf-8").splitlines():
                if line.startswith("commit="):
                    return line.split("=", 1)[1]
        return "unknown"


def expected_manifest(cfg: dict, dataset_cfg: dict, method: dict, paths: dict) -> dict:
    build = build_config(cfg, dataset_cfg, method)
    return {
        "format": ADAPTER_FORMAT, "dataset": paths["dataset"],
        "knowhere_commit": knowhere_commit(cfg),
        "base_fvecs": signature(paths["base_fvecs"]),
        "base_labels": signature(paths["base_labels"]),
        "build": {
            "dist_fn": str(build["dist_fn"]), "num_threads": int(build["num_threads"]),
            "M": int(build["M"]),
            "ef_construction": int(build["ef_construction"]),
            "add_batch_size": int(build["add_batch_size"]),
        },
    }


def index_is_current(paths: dict, expected: dict) -> bool:
    required = [paths["index_file"], paths["filter_file"], paths["manifest"]]
    if not all(isinstance(path, Path) and path.exists() for path in required):
        return False
    try:
        actual = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected


def memory_stats(log_path: Path) -> dict[str, int | float]:
    text = log_path.read_text(encoding="utf-8")
    patterns = {
        "index_memory_bytes": r"index_memory_bytes=(\d+)",
        "index_memory_mib": r"index_memory_mib=([0-9.eE+-]+)",
        "index_level0_bytes": r"index_level0_bytes=(\d+)",
        "index_upper_level_bytes": r"index_upper_level_bytes=(\d+)",
        "index_runtime_metadata_bytes": r"index_runtime_metadata_bytes=(\d+)",
        "index_filter_bytes": r"index_filter_bytes=(\d+)",
        "vector_storage_bytes": r"vector_storage_bytes=(\d+)",
    }
    result: dict[str, int | float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise RuntimeError(f"missing memory statistic {key} in {log_path}")
        result[key] = float(match.group(1)) if key == "index_memory_mib" else int(match.group(1))
    return result


def append_build_record(
    output: Path, dataset: str, index_name: str, index_file: Path,
    elapsed: float, memory: dict,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "index_name", "build_time_seconds", "build_time_ms", "index_file",
        "index_memory_bytes", "index_memory_mib", "index_level0_bytes",
        "index_upper_level_bytes", "index_runtime_metadata_bytes", "index_filter_bytes",
        "vector_storage_bytes",
    ]
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "dataset": dataset, "index_name": index_name,
            "build_time_seconds": f"{elapsed:.6f}", "build_time_ms": f"{elapsed * 1000:.3f}",
            "index_file": str(index_file), **memory,
        })


def build_index(binary: Path, cfg: dict, dataset_cfg: dict, method: dict, paths: dict) -> None:
    expected = expected_manifest(cfg, dataset_cfg, method, paths)
    if index_is_current(paths, expected):
        log(f"[{paths['dataset']}] Milvus HNSW index exists and manifest matches: {paths['index_file']}")
        return
    build = build_config(cfg, dataset_cfg, method)
    if str(build["data_type"]).lower() != "float":
        raise ValueError("the Milvus HNSW adapter supports float vectors")
    index_file = paths["index_file"]
    filter_file = paths["filter_file"]
    manifest = paths["manifest"]
    assert isinstance(index_file, Path) and isinstance(filter_file, Path) and isinstance(manifest, Path)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = index_file.with_name(index_file.name + ".building")
    temporary_filter = filter_file.with_name(filter_file.name + ".building")
    build_log = index_file.parent.parent / "others" / "milvus_hnsw_build.log"
    started = time.perf_counter()
    try:
        run_command([
            binary, paths["base_fvecs"], paths["base_labels"], temporary_index, temporary_filter,
            build["num_threads"], build["M"], build["ef_construction"], build["dist_fn"],
            build["add_batch_size"],
        ], build_log)
        elapsed = time.perf_counter() - started
        if not temporary_index.exists() or not temporary_filter.exists():
            raise RuntimeError("Milvus HNSW build completed without both index files")
        os.replace(temporary_index, index_file)
        os.replace(temporary_filter, filter_file)
        temporary_manifest = manifest.with_name(manifest.name + ".building")
        temporary_manifest.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_manifest, manifest)
    finally:
        for temporary in [temporary_index, temporary_filter]:
            if temporary.exists():
                temporary.unlink()
    append_build_record(
        index_file.parent.parent / "others" / "index_build_time.csv",
        str(paths["dataset"]), str(paths["index_name"]), index_file,
        elapsed, memory_stats(build_log),
    )
    log(f"[{paths['dataset']}] Milvus HNSW build done in {elapsed:.3f}s")


def run_search(binary: Path, cfg: dict, dataset_cfg: dict, method: dict, paths: dict) -> None:
    search = search_config(cfg, dataset_cfg, method)
    values = ef_values(cfg, dataset_cfg, method, search)
    topk = int(search["K"])
    query_count, query_dimension = fvecs_metadata(paths["query_fvecs"])
    _, base_dimension = fvecs_metadata(paths["base_fvecs"])
    if query_dimension != base_dimension:
        raise RuntimeError("query dimension does not match base vectors")
    if label_row_count(paths["query_labels"]) != query_count:
        raise RuntimeError("query-label row count does not match query vectors")
    groundtruth = prepare_groundtruth(paths["groundtruth"], paths["adapter_groundtruth"], topk, query_count)
    result_dir = (
        Path(cfg["result_root"]) / str(paths["dataset"]) / "results" /
        method.get("name", "milvus_hnsw") /
        result_task_name(str(paths["query_task"]), search)
    )
    output = result_dir / "results"
    log_path = result_dir / "others" / f"{paths['dataset']}_search_output.txt"
    summary_csv = output / "search_time_summary.csv"
    if should_skip_completed_search(summary_csv, values, cfg, dataset_cfg, method):
        log(
            f"[{paths['dataset']}][{method.get('name', 'milvus_hnsw')}] existing results cover "
            f"{describe_search_values(values)}; skipping search: {summary_csv}"
        )
        return
    command = [
        binary, paths["query_fvecs"], paths["query_labels"], groundtruth,
        paths["index_file"], paths["filter_file"], output, topk,
        search["num_repeats"], search["num_threads"], build_config(cfg, dataset_cfg, method)["dist_fn"],
        int(bool(search["disable_fallback_brute_force"])),
    ]
    command.extend(values)
    run_command(command, log_path)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_milvus_hnsw_experiment.py <config.json>", file=sys.stderr)
        return 2
    config_path = Path(sys.argv[1])
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    log(f"config: {config_path}")
    build_binary, search_binary = ensure_binaries(cfg)
    for dataset_cfg in cfg["datasets"]:
        for method in methods(cfg):
            paths = resolve_paths(cfg, dataset_cfg, method)
            ensure_inputs(paths)
            build_index(build_binary, cfg, dataset_cfg, method, paths)
            run_search(search_binary, cfg, dataset_cfg, method, paths)
    log("all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
