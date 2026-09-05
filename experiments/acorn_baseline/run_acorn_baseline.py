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


def read_num_fvecs(fvecs_path: Path) -> int:
    with fvecs_path.open("rb") as f:
        dim_raw = f.read(4)
    if len(dim_raw) != 4:
        raise RuntimeError(f"invalid fvecs header: {fvecs_path}")
    (dim,) = struct.unpack("<I", dim_raw)
    vector_bytes = 4 + int(dim) * 4
    size = fvecs_path.stat().st_size
    if vector_bytes <= 4 or size % vector_bytes != 0:
        raise RuntimeError(f"invalid fvecs size for dim {dim}: {fvecs_path}")
    return size // vector_bytes


def bool_int(value) -> str:
    if isinstance(value, str):
        return "1" if value.lower() in {"1", "true", "yes", "on"} else "0"
    return "1" if bool(value) else "0"


def merged_build_cfg(cfg: dict, dataset_cfg: dict) -> dict:
    build_cfg = {
        "M": 32,
        "M_beta": 64,
        "gamma": 80,
        "num_threads": 32,
    }
    build_cfg.update(cfg.get("build", {}))
    build_cfg.update(dataset_cfg.get("build", {}))
    return build_cfg


def merged_search_cfg(cfg: dict, dataset_cfg: dict) -> dict:
    search_cfg = {
        "K": 10,
        "num_repeats": 1,
        "num_threads": cfg.get("build", {}).get("num_threads", 32),
        "efs_start": 100,
        "efs_end": 1000,
        "efs_step": 100,
        "if_bfs_filter": True,
    }
    search_cfg.update(cfg.get("search", {}))
    search_cfg.update(dataset_cfg.get("search", {}))
    return search_cfg


def efs_values(search_cfg: dict) -> list[int]:
    if search_cfg.get("efs_values"):
        return [int(value) for value in search_cfg["efs_values"]]
    return list(
        range(
            int(search_cfg["efs_start"]),
            int(search_cfg["efs_end"]) + 1,
            int(search_cfg["efs_step"]),
        )
    )


def result_query_dir_name(query_task: str, search_cfg: dict) -> str:
    safe_query_task = str(query_task).strip("/").replace("/", "__")
    return f"{safe_query_task}_{search_cfg['efs_start']}_{search_cfg['efs_step']}_{search_cfg['efs_end']}"


def dataset_display_name(dataset_cfg: dict) -> str:
    dataset = str(dataset_cfg["dataset"]).strip("/")
    return str(dataset_cfg.get("dataset_name", Path(dataset).name))


def result_dataset_name(dataset_cfg: dict) -> str:
    dataset = str(dataset_cfg["dataset"]).strip("/")
    return str(dataset_cfg.get("result_dataset", dataset.replace("/", "__")))


def dataset_paths(cfg: dict, dataset_cfg: dict, search_cfg: dict) -> dict[str, Path | str]:
    dataset_rel = str(dataset_cfg["dataset"]).strip("/")
    dataset = dataset_display_name(dataset_cfg)
    query_task = str(dataset_cfg["query_task"]).strip("/")
    data_dir = Path(dataset_cfg.get("data_dir", Path(cfg["data_root"]) / dataset_rel))
    query_dir = Path(dataset_cfg.get("query_dir", data_dir / query_task))
    result_dataset_dir = Path(cfg["result_root"]) / result_dataset_name(dataset_cfg)
    index_name = dataset_cfg.get("index_name", cfg.get("index_name", "ACORN"))
    result_method = dataset_cfg.get("method_name", cfg.get("method_name", "ACORN"))
    result_task = result_query_dir_name(query_task, search_cfg)
    return {
        "dataset": dataset,
        "dataset_rel": dataset_rel,
        "query_task": query_task,
        "data_dir": data_dir,
        "query_dir": query_dir,
        "base_bin": Path(dataset_cfg.get("base_bin_file", data_dir / f"{dataset}_base.bin")),
        "base_fvecs": Path(dataset_cfg.get("base_fvecs", data_dir / f"{dataset}_base.fvecs")),
        "base_labels": Path(dataset_cfg.get("base_label_file", data_dir / f"{dataset}_base_labels.txt")),
        "query_bin": Path(dataset_cfg.get("query_bin_file", query_dir / f"{dataset}_query.bin")),
        "query_fvecs": Path(dataset_cfg.get("query_fvecs", query_dir / f"{dataset}_query.fvecs")),
        "query_labels": Path(dataset_cfg.get("query_label_file", query_dir / f"{dataset}_query_labels.txt")),
        "gt_file": Path(dataset_cfg.get("groundtruth_file", result_dataset_dir / "GroundTruth" / query_task / f"{dataset}_gt_labels_containment.bin")),
        "index_dir": Path(dataset_cfg.get("index_path_prefix", result_dataset_dir / "index" / index_name / "index_files")),
        "result_dir": Path(dataset_cfg.get("result_path_prefix", result_dataset_dir / "results" / result_method / result_task / "results")),
        "other_dir": Path(dataset_cfg.get("other_path_prefix", result_dataset_dir / "results" / result_method / result_task / "others")),
    }


def discover_dataset_configs(cfg: dict) -> list[dict]:
    data_root = Path(cfg["data_root"])
    discovered = []
    for base_fvecs in sorted(data_root.rglob("*_base.fvecs")):
        dataset_name = base_fvecs.name[: -len("_base.fvecs")]
        data_dir = base_fvecs.parent
        base_labels = data_dir / f"{dataset_name}_base_labels.txt"
        if not base_labels.exists():
            continue
        dataset_rel = data_dir.relative_to(data_root).as_posix()
        for query_labels in sorted(data_dir.rglob(f"{dataset_name}_query_labels.txt")):
            query_dir = query_labels.parent
            query_fvecs = query_dir / f"{dataset_name}_query.fvecs"
            if not query_fvecs.exists():
                continue
            discovered.append(
                {
                    "dataset": dataset_rel,
                    "dataset_name": dataset_name,
                    "query_task": query_dir.relative_to(data_dir).as_posix(),
                }
            )
    return discovered


def configured_datasets(cfg: dict) -> list[dict]:
    datasets = cfg.get("datasets", [])
    if datasets == "all" or cfg.get("discover_datasets", False):
        discovered = discover_dataset_configs(cfg)
        if not discovered:
            raise RuntimeError(f"no runnable ACORN datasets discovered under {cfg['data_root']}")
        return discovered
    return list(datasets)


def default_openblas_library(source_dir: Path) -> Path | None:
    candidates = [
        source_dir.parents[1] / "openblas" / "lib",
        source_dir.parents[1] / "OpenBLAS",
        source_dir.parents[2] / "openblas" / "lib" if len(source_dir.parents) > 2 else source_dir.parents[1] / "openblas" / "lib",
    ]
    seen = set()
    for directory in candidates:
        if directory in seen or not directory.exists():
            continue
        seen.add(directory)
        libs = sorted(directory.glob("libopenblas*.so")) + sorted(directory.glob("libopenblas*.a"))
        if libs:
            return libs[0]
    return None


def acorn_cmake_args(cfg: dict, source_dir: Path) -> list[str]:
    args = list(cfg.get("cmake_args", []))
    has_blas = any(str(arg).startswith("-DBLAS_LIBRARIES=") for arg in args)
    has_lapack = any(str(arg).startswith("-DLAPACK_LIBRARIES=") for arg in args)
    blas_lapack_library = cfg.get("blas_lapack_library")
    if not blas_lapack_library:
        detected = default_openblas_library(source_dir)
        blas_lapack_library = str(detected) if detected else None
    if blas_lapack_library:
        if not any(str(arg).startswith("-DBLA_VENDOR=") for arg in args):
            args.append("-DBLA_VENDOR=OpenBLAS")
        if not has_blas:
            args.append(f"-DBLAS_LIBRARIES={blas_lapack_library}")
        if not has_lapack:
            args.append(f"-DLAPACK_LIBRARIES={blas_lapack_library}")
    return [str(arg) for arg in args]


def ensure_acorn_binary(cfg: dict) -> Path:
    source_dir = Path(cfg.get("source_dir", Path(__file__).resolve().parents[2] / "ACORN"))
    build_dir = Path(cfg.get("build_dir", Path(__file__).resolve().parents[2] / "build_acorn_rel"))
    auto_compile = bool(cfg.get("auto_compile", True))
    jobs = str(cfg.get("build_jobs", os.environ.get("BUILD_JOBS", 16)))
    if auto_compile:
        build_dir.mkdir(parents=True, exist_ok=True)
        cmake_cmd = [
            "cmake",
            "-S",
            source_dir,
            "-B",
            build_dir,
            "-DFAISS_ENABLE_GPU=OFF",
            "-DFAISS_ENABLE_PYTHON=OFF",
            "-DBUILD_TESTING=OFF",
            "-DBUILD_SHARED_LIBS=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        cmake_cmd.extend(acorn_cmake_args(cfg, source_dir))
        run_command(cmake_cmd)
        run_command(["cmake", "--build", build_dir, f"-j{jobs}", "--target", "test_acorn"])
    test_acorn = build_dir / "demos" / "test_acorn"
    if not test_acorn.exists():
        raise FileNotFoundError(f"missing required binary: {test_acorn}")
    return test_acorn


def ensure_ung_tools(cfg: dict) -> tuple[Path, Path]:
    source_dir = Path(cfg.get("ung_source_dir", Path(__file__).resolve().parents[2] / "UNG" / "codes"))
    build_dir = Path(cfg.get("ung_build_dir", Path(__file__).resolve().parents[2] / "build_ung_rel"))
    auto_compile = bool(cfg.get("auto_compile_ung_tools", cfg.get("auto_compile", True)))
    jobs = str(cfg.get("build_jobs", os.environ.get("BUILD_JOBS", 16)))
    if auto_compile:
        build_dir.mkdir(parents=True, exist_ok=True)
        run_command(["cmake", "-S", source_dir, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release"])
        run_command(["cmake", "--build", build_dir, f"-j{jobs}", "--target", "compute_groundtruth", "fvecs_to_bin"])
    compute_gt_bin = build_dir / "tools" / "compute_groundtruth"
    fvecs_to_bin = build_dir / "tools" / "fvecs_to_bin"
    for binary in [compute_gt_bin, fvecs_to_bin]:
        if not binary.exists():
            raise FileNotFoundError(f"missing required binary: {binary}")
    return compute_gt_bin, fvecs_to_bin


def ensure_query_bin(fvecs_to_bin: Path, paths: dict[str, Path | str]) -> Path:
    query_bin = paths["query_bin"]
    query_fvecs = paths["query_fvecs"]
    assert isinstance(query_bin, Path)
    assert isinstance(query_fvecs, Path)
    if query_bin.exists():
        return query_bin
    if not query_fvecs.exists():
        raise FileNotFoundError(f"missing query bin/fvecs: {query_bin} / {query_fvecs}")
    run_command(["fvecs_to_bin" if fvecs_to_bin is None else fvecs_to_bin, "--data_type", "float", "--input_file", query_fvecs, "--output_file", query_bin])
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


def acorn_index_exists(index_dir: Path) -> bool:
    required = ["acorn.index", "acorn1.index", "acorn.index.meta", "acorn.index.inverted_index"]
    if not all((index_dir / name).exists() for name in required):
        return False
    return (index_dir / "acorn.index.inverted_index").stat().st_size > 8


def acorn_cmd(
    test_acorn: Path,
    mode: str,
    cfg: dict,
    paths: dict[str, Path | str],
    build_cfg: dict,
    search_cfg: dict,
    n: int,
    gt_file: Path,
) -> list:
    values = efs_values(search_cfg)
    index_dir = paths["index_dir"]
    result_dir = paths["result_dir"]
    assert isinstance(index_dir, Path)
    assert isinstance(result_dir, Path)
    threads = build_cfg.get("num_threads", 32) if mode == "build" else search_cfg.get("num_threads", build_cfg.get("num_threads", 32))
    return [
        test_acorn,
        mode,
        str(n),
        str(build_cfg["gamma"]),
        paths["dataset"],
        str(build_cfg["M"]),
        str(build_cfg["M_beta"]),
        paths["base_fvecs"],
        paths["base_labels"],
        paths["query_dir"],
        str(result_dir) + "/",
        str(result_dir) + "/",
        gt_file if mode == "search" else "dummy_gt_path_for_build",
        str(threads),
        str(search_cfg["num_repeats"]),
        bool_int(search_cfg.get("if_bfs_filter", True)),
        ",".join(str(value) for value in values),
        index_dir / "acorn.index",
        index_dir / "acorn1.index",
        str(search_cfg["K"]),
    ]


def ensure_index(test_acorn: Path, cfg: dict, paths: dict[str, Path | str], build_cfg: dict, search_cfg: dict, n: int) -> Path:
    index_dir = paths["index_dir"]
    other_dir = paths["other_dir"]
    assert isinstance(index_dir, Path)
    assert isinstance(other_dir, Path)
    if acorn_index_exists(index_dir):
        log(f"[{paths['dataset']}] ACORN index exists: {index_dir}")
        return index_dir
    for key in ["base_fvecs", "base_labels"]:
        path = paths[key]
        assert isinstance(path, Path)
        if not path.exists():
            raise FileNotFoundError(f"missing {key}: {path}")
    index_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        acorn_cmd(test_acorn, "build", cfg, paths, build_cfg, search_cfg, n, Path("dummy_gt_path_for_build")),
        other_dir / f"{paths['dataset']}_acorn_build.log",
    )
    if not acorn_index_exists(index_dir):
        missing = [name for name in ["acorn.index", "acorn1.index", "acorn.index.meta", "acorn.index.inverted_index"] if not (index_dir / name).exists()]
        raise FileNotFoundError(f"ACORN build finished but required files are missing in {index_dir}: {missing}")
    return index_dir


def write_normalized_summary(rows: list[dict], output_dir: Path, *, time_col: str, recall_col: str, qps_col: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [
        {
            "Lsearch": row["efs"],
            "Average_Efs": row["efs"],
            "Average_Time_ms": row[time_col],
            "Average_Recall": row[recall_col],
        }
        for row in rows
    ]
    summary_csv = output_dir / "search_time_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Lsearch", "Average_Efs", "Average_Time_ms", "Average_Recall"])
        writer.writeheader()
        writer.writerows(summary_rows)
    qps_csv = output_dir / "search_time_summary_qps.csv"
    with qps_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Lsearch", "Average_Efs", "Average_Time_ms", "Average_Recall", "QPS"])
        writer.writeheader()
        for source, normalized in zip(rows, summary_rows):
            row = dict(normalized)
            row["QPS"] = source.get(qps_col, "0")
            writer.writerow(row)


def acorn1_result_dir(result_dir: Path) -> Path:
    parts = list(result_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "ACORN":
            parts[i] = "ACORN-1"
            return Path(*parts)
    return result_dir.parent.parent.parent / "ACORN-1" / result_dir.parent.name / result_dir.name


def normalize_acorn_summary(result_dir: Path) -> None:
    avg_files = sorted(result_dir.glob("avg_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not avg_files:
        raise FileNotFoundError(f"missing ACORN avg csv in {result_dir}")
    avg_csv = avg_files[0]
    with avg_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    write_normalized_summary(rows, result_dir, time_col="acorn_Time_ms", recall_col="acorn_Recall", qps_col="acorn_QPS")
    write_normalized_summary(rows, acorn1_result_dir(result_dir), time_col="acorn_1_Time_ms", recall_col="acorn_1_Recall", qps_col="acorn_1_QPS")


def run_search(
    test_acorn: Path,
    cfg: dict,
    paths: dict[str, Path | str],
    build_cfg: dict,
    search_cfg: dict,
    n: int,
    gt_file: Path,
) -> None:
    result_dir = paths["result_dir"]
    other_dir = paths["other_dir"]
    assert isinstance(result_dir, Path)
    assert isinstance(other_dir, Path)
    result_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        acorn_cmd(test_acorn, "search", cfg, paths, build_cfg, search_cfg, n, gt_file),
        other_dir / f"{paths['dataset']}_acorn_search.log",
    )
    normalize_acorn_summary(result_dir)


def dataset_n(paths: dict[str, Path | str], dataset_cfg: dict) -> int:
    if "N" in dataset_cfg:
        return int(dataset_cfg["N"])
    base_bin = paths["base_bin"]
    base_fvecs = paths["base_fvecs"]
    assert isinstance(base_bin, Path)
    assert isinstance(base_fvecs, Path)
    if base_bin.exists():
        return read_num_vectors(base_bin)
    return read_num_fvecs(base_fvecs)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_acorn_baseline.py <config.json>", file=sys.stderr)
        return 2
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    log(f"config: {cfg_path}")
    test_acorn = ensure_acorn_binary(cfg)
    compute_gt_bin, fvecs_to_bin = ensure_ung_tools(cfg)

    for dataset_cfg in configured_datasets(cfg):
        search_cfg = merged_search_cfg(cfg, dataset_cfg)
        build_cfg = merged_build_cfg(cfg, dataset_cfg)
        paths = dataset_paths(cfg, dataset_cfg, search_cfg)
        query_bin = ensure_query_bin(fvecs_to_bin, paths)
        gt_file = ensure_gt(compute_gt_bin, cfg, paths, query_bin, search_cfg)
        n = dataset_n(paths, dataset_cfg)
        ensure_index(test_acorn, cfg, paths, build_cfg, search_cfg, n)
        run_search(test_acorn, cfg, paths, build_cfg, search_cfg, n, gt_file)
        log(f"[{paths['dataset']}] ACORN baseline done")
    log("all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
