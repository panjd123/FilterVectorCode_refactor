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


def read_label_rows(path: Path) -> list[list[int]]:
    return [parse_label_line(line) for line in path.read_text(encoding="utf-8").splitlines()]


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


def prepare_pathseer_groundtruth_file(
    source_gt_file: Path,
    pathseer_gt_file: Path,
    topk: int,
    num_queries: int,
) -> Path:
    expected_ids = topk * num_queries
    expected_id_bytes = expected_ids * struct.calcsize("<i")
    expected_pair_bytes = expected_ids * struct.calcsize("<If")
    actual_bytes = source_gt_file.stat().st_size

    pathseer_gt_file.parent.mkdir(parents=True, exist_ok=True)
    if actual_bytes == expected_id_bytes:
        if source_gt_file != pathseer_gt_file:
            shutil.copyfile(source_gt_file, pathseer_gt_file)
        return pathseer_gt_file
    if actual_bytes != expected_pair_bytes:
        raise RuntimeError(
            f"groundtruth size mismatch: {source_gt_file} has {actual_bytes} bytes, "
            f"expected {expected_id_bytes} id bytes or {expected_pair_bytes} UNG pair bytes"
        )

    with source_gt_file.open("rb") as source, pathseer_gt_file.open("wb") as target:
        for _ in range(expected_ids):
            pair = source.read(struct.calcsize("<If"))
            # Preserve the raw uint32 bits so UNG sentinel 0xFFFFFFFF becomes int32 -1.
            target.write(pair[:4])
    return pathseer_gt_file


def merged_search_cfg(cfg: dict, dataset_cfg: dict, method: dict) -> dict:
    build_cfg = cfg.get("build", {})
    search_cfg = {
        "K": cfg.get("topk", 10),
        "num_repeats": 1,
        "num_threads": build_cfg.get("num_threads", 32),
        "lsearch_start": 100,
        "lsearch_end": 100,
        "lsearch_step": 100,
        "M_exp_search": max(1, int(build_cfg.get("M", build_cfg.get("max_degree", 64))) // 2),
    }
    search_cfg.update(cfg.get("search", {}))
    search_cfg.update(dataset_cfg.get("search", {}))
    search_cfg.update(method.get("search", {}))
    return search_cfg


def search_values(cfg: dict, dataset_cfg: dict, method: dict, search_cfg: dict) -> list[int]:
    explicit = method.get("ef_values") or dataset_cfg.get("ef_values") or cfg.get("ef_values")
    if explicit:
        return [int(value) for value in explicit]
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


def configured_methods(cfg: dict) -> list[dict]:
    methods = [dict(method) for method in cfg.get("methods", [])]
    if methods:
        return methods
    return [{"name": cfg.get("method_name", "pathseer"), "index_name": cfg.get("index_name", "PathSeer")}]


def resolve_dataset_files(cfg: dict, dataset_cfg: dict, method: dict) -> dict[str, Path | str]:
    dataset = dataset_cfg["dataset"]
    query_task = dataset_cfg["query_task"]
    data_dir = Path(cfg["data_root"]) / dataset
    result_root = Path(cfg["result_root"]) / dataset
    search_cfg = merged_search_cfg(cfg, dataset_cfg, method)
    result_task = result_query_dir_name(query_task, search_cfg)
    adapter_dir = (
        result_root / "results" / method.get("name", "pathseer") /
        result_task / "results"
    )
    index_name = method.get("index_name", dataset_cfg.get("index_name", cfg.get("index_name", "PathSeer")))
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
        "pathseer_gt_file": Path(dataset_cfg.get("pathseer_groundtruth_file", adapter_dir / f"{dataset}_pathseer_gt_ids.bin")),
        "index_file": Path(dataset_cfg.get("index_file", index_dir / "index.bin")),
        "index_manifest": Path(dataset_cfg.get("index_manifest", index_dir / "index.pathseer.json")),
        "index_name": index_name,
    }


def ensure_input_files(paths: dict[str, Path | str]) -> None:
    for key in ["base_fvecs", "query_fvecs", "base_labels", "query_labels", "gt_file"]:
        path = paths[key]
        assert isinstance(path, Path)
        if not path.exists():
            raise FileNotFoundError(f"missing {key}: {path}")


def find_openblas(cfg: dict, project_root: Path) -> Path | None:
    if cfg.get("openblas_library"):
        library = Path(cfg["openblas_library"])
        if not library.exists():
            raise FileNotFoundError(f"configured openblas_library does not exist: {library}")
        return library
    candidates = [
        project_root.parent / "openblas" / "lib" / "libopenblas.so",
        project_root.parent / "OpenBLAS" / "libopenblas.so",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def runtime_environment(faiss_build_dir: Path, openblas_library: Path | None) -> dict[str, str]:
    environment = dict(os.environ)
    library_dirs = [str(faiss_build_dir / "faiss")]
    if openblas_library is not None:
        library_dirs.append(str(openblas_library.parent))
    previous = environment.get("LD_LIBRARY_PATH")
    if previous:
        library_dirs.append(previous)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_dirs)
    return environment


def ensure_pathseer_binaries(cfg: dict) -> tuple[Path, Path, dict[str, str]]:
    project_root = Path(__file__).resolve().parents[2]
    source_dir = Path(cfg.get("pathseer_source_dir", project_root / "PathSeer"))
    faiss_source_dir = source_dir / "faiss"
    faiss_build_dir = Path(cfg.get("pathseer_faiss_build_dir", faiss_source_dir / "build"))
    adapter_source_dir = source_dir / "project_adapter"
    adapter_build_dir = Path(cfg.get("pathseer_adapter_build_dir", adapter_source_dir / "build"))
    build_binary = Path(cfg.get("pathseer_build_binary", adapter_build_dir / "build_index"))
    search_binary = Path(cfg.get("pathseer_search_binary", adapter_build_dir / "search_sweep"))
    openblas_library = find_openblas(cfg, project_root)
    environment = runtime_environment(faiss_build_dir, openblas_library)

    if bool(cfg.get("auto_build_pathseer", True)):
        json_archive = Path(cfg.get("nlohmann_json_archive", project_root / "thirdparty" / "json-3.10.4.tar.gz"))
        if not json_archive.exists():
            raise FileNotFoundError(f"missing offline nlohmann/json archive: {json_archive}")
        faiss_build_dir.mkdir(parents=True, exist_ok=True)
        adapter_build_dir.mkdir(parents=True, exist_ok=True)
        configure = [
            "cmake", "-S", faiss_source_dir, "-B", faiss_build_dir,
            "-DFAISS_ENABLE_GPU=OFF", "-DFAISS_ENABLE_PYTHON=OFF",
            "-DBUILD_TESTING=OFF", "-DBUILD_SHARED_LIBS=ON",
            "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_FLAGS=-march=native",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            f"-DPATHSEER_NLOHMANN_JSON_ARCHIVE={json_archive}",
        ]
        if openblas_library is not None:
            configure.extend([
                f"-DBLAS_LIBRARIES={openblas_library}",
                f"-DLAPACK_LIBRARIES={openblas_library}",
                f"-DCMAKE_BUILD_RPATH={openblas_library.parent}",
            ])
        run_command(configure, env=environment)
        jobs = str(cfg.get("build_jobs", os.environ.get("BUILD_JOBS", 16)))
        run_command(["cmake", "--build", faiss_build_dir, f"-j{jobs}", "--target", "faiss"], env=environment)
        run_command([
            "cmake", "-S", adapter_source_dir, "-B", adapter_build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPATHSEER_ROOT={source_dir}",
            f"-DPATHSEER_FAISS_BUILD_DIR={faiss_build_dir}",
        ], env=environment)
        run_command([
            "cmake", "--build", adapter_build_dir, f"-j{jobs}",
            "--target", "pathseer_build_index", "pathseer_search_sweep",
        ], env=environment)

    if not build_binary.exists():
        raise FileNotFoundError(f"missing PathSeer build binary: {build_binary}")
    if not search_binary.exists():
        raise FileNotFoundError(f"missing PathSeer search binary: {search_binary}")
    return build_binary, search_binary, environment


def merged_build_cfg(cfg: dict, dataset_cfg: dict, method: dict) -> dict:
    build_cfg = dict(cfg.get("build", {}))
    build_cfg.update(dataset_cfg.get("build", {}))
    build_cfg.update(method.get("build", {}))
    return build_cfg


def parse_index_memory_stats(build_log: Path) -> dict[str, int | float]:
    text = build_log.read_text(encoding="utf-8")
    patterns = {
        "index_memory_bytes": r"index_memory_bytes=(\d+)",
        "index_memory_mib": r"index_memory_mib=([0-9.eE+-]+)",
        "index_level0_bytes": r"index_level0_bytes=(\d+)",
        "index_upper_level_bytes": r"index_upper_level_bytes=(\d+)",
        "index_runtime_metadata_bytes": r"index_runtime_metadata_bytes=(\d+)",
    }
    stats: dict[str, int | float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise RuntimeError(f"missing PathSeer memory statistic {key} in {build_log}")
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
        "index_upper_level_bytes", "index_runtime_metadata_bytes",
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
    environment: dict[str, str],
) -> None:
    index_file = paths["index_file"]
    index_manifest = paths["index_manifest"]
    assert isinstance(index_file, Path)
    assert isinstance(index_manifest, Path)
    if index_file.exists() and index_manifest.exists():
        log(f"[{paths['dataset']}] PathSeer index exists: {index_file}")
        return

    build_cfg = merged_build_cfg(cfg, dataset_cfg, method)
    num_threads = int(build_cfg.get("num_threads", 32))
    M = int(build_cfg.get("M", build_cfg.get("max_degree", 64)))
    M_exp = int(build_cfg.get("M_exp", build_cfg.get("M_exp_construct", 128)))
    ef_construction = int(build_cfg.get("ef_construction", build_cfg.get("Lbuild", M * 2)))
    ef_construction_exp = int(build_cfg.get("ef_construction_exp", M_exp * 2))
    dist_fn = str(build_cfg.get("dist_fn", "L2"))
    optimized_building = bool(build_cfg.get("optimized_building", True))
    index_file.parent.mkdir(parents=True, exist_ok=True)
    build_log = index_file.parent.parent / "others" / "pathseer_build.log"
    started = time.perf_counter()
    run_command([
        build_binary, paths["base_fvecs"], index_file, str(num_threads), str(M), str(M_exp),
        str(ef_construction), str(ef_construction_exp), dist_fn,
        "1" if optimized_building else "0",
    ], build_log, environment)
    elapsed = time.perf_counter() - started
    memory = parse_index_memory_stats(build_log)
    append_index_build_time(
        index_file.parent.parent / "others" / "index_build_time.csv",
        str(paths["dataset"]), str(paths["index_name"]), index_file, elapsed, memory,
    )
    index_manifest.write_text(json.dumps({
        "format": "PathSeer IndexPathSeerFlat",
        "base_fvecs": str(paths["base_fvecs"]),
        "build": {
            "num_threads": num_threads, "M": M, "M_exp": M_exp,
            "ef_construction": ef_construction,
            "ef_construction_exp": ef_construction_exp,
            "dist_fn": dist_fn, "optimized_building": optimized_building,
        },
    }, indent=2) + "\n", encoding="utf-8")


def run_searches(
    search_binary: Path,
    cfg: dict,
    dataset_cfg: dict,
    method: dict,
    paths: dict[str, Path | str],
    environment: dict[str, str],
) -> None:
    search_cfg = merged_search_cfg(cfg, dataset_cfg, method)
    ef_values = search_values(cfg, dataset_cfg, method, search_cfg)
    topk = int(search_cfg["K"])
    num_queries = fvecs_count(paths["query_fvecs"])
    if len(read_label_rows(paths["query_labels"])) != num_queries:
        raise RuntimeError(f"query label count does not match query fvecs: {paths['query_labels']}")
    pathseer_gt = prepare_pathseer_groundtruth_file(
        paths["gt_file"], paths["pathseer_gt_file"], topk, num_queries)
    build_cfg = merged_build_cfg(cfg, dataset_cfg, method)
    dist_fn = str(build_cfg.get("dist_fn", "L2"))
    result_task = result_query_dir_name(str(paths["query_task"]), search_cfg)
    result_dir = Path(cfg["result_root"]) / str(paths["dataset"]) / "results" / method.get("name", "pathseer") / result_task
    raw_results = result_dir / "results"
    log_path = result_dir / "others" / f"{paths['dataset']}_search_output.txt"
    summary_csv = raw_results / "search_time_summary.csv"
    if should_skip_completed_search(summary_csv, ef_values, cfg, dataset_cfg, method):
        log(
            f"[{paths['dataset']}][{method.get('name', 'pathseer')}] existing results cover "
            f"{describe_search_values(ef_values)}; skipping search: {summary_csv}"
        )
        return
    command = [
        search_binary, paths["query_fvecs"], paths["base_labels"], paths["query_labels"],
        pathseer_gt, paths["index_file"], raw_results, str(topk),
        str(search_cfg["num_repeats"]), str(search_cfg["num_threads"]), dist_fn,
        str(search_cfg["M_exp_search"]),
        str(search_cfg.get("virtual_overhead", 0)),
        str(search_cfg.get("dist_comp_ns", 0)),
        str(search_cfg.get("filter_ns", 0)),
    ]
    command.extend(str(value) for value in ef_values)
    run_command(command, log_path, environment)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_pathseer_experiment.py <config.json>", file=sys.stderr)
        return 2
    config_path = Path(sys.argv[1])
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    log(f"config: {config_path}")
    build_binary, search_binary, environment = ensure_pathseer_binaries(cfg)

    for dataset_cfg in cfg["datasets"]:
        for method in configured_methods(cfg):
            paths = resolve_dataset_files(cfg, dataset_cfg, method)
            ensure_input_files(paths)
            build_index(build_binary, cfg, dataset_cfg, method, paths, environment)
            run_searches(search_binary, cfg, dataset_cfg, method, paths, environment)
    log("all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
