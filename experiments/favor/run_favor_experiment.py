#!/usr/bin/env python3

import csv
import json
import os
import shutil
import struct
import re
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


def parse_label_line(line: str) -> list[int]:
    line = line.strip()
    if not line:
        return []
    return [int(x) for x in re.split(r"[\s,]+", line) if x]


def read_label_rows(path: Path) -> list[list[int]]:
    return [parse_label_line(line) for line in path.read_text(encoding="utf-8").splitlines()]


def prepare_favor_condition_file(query_label_file: Path, condition_file: Path) -> None:
    query_rows = read_label_rows(query_label_file)

    condition_file.parent.mkdir(parents=True, exist_ok=True)
    with condition_file.open("w", encoding="utf-8") as f:
        for row in query_rows:
            if row:
                f.write(" AND ".join(f"label_{label} == 1" for label in row) + "\n")
            else:
                f.write("label_empty == 1\n")


def prepare_favor_groundtruth_file(
    source_gt_file: Path,
    favor_gt_file: Path,
    topk: int,
    num_queries: int,
) -> Path:
    expected_ids = topk * num_queries
    expected_id_bytes = expected_ids * struct.calcsize("<I")
    expected_pair_bytes = expected_ids * struct.calcsize("<If")
    actual_bytes = source_gt_file.stat().st_size

    favor_gt_file.parent.mkdir(parents=True, exist_ok=True)
    if actual_bytes == expected_id_bytes:
        if source_gt_file != favor_gt_file:
            shutil.copyfile(source_gt_file, favor_gt_file)
        return favor_gt_file

    if actual_bytes != expected_pair_bytes:
        raise RuntimeError(
            f"groundtruth size mismatch for FAVOR conversion: {source_gt_file} has "
            f"{actual_bytes} bytes, expected {expected_id_bytes} bytes for FAVOR ids or "
            f"{expected_pair_bytes} bytes for UNG (id, distance) pairs"
        )

    with source_gt_file.open("rb") as src, favor_gt_file.open("wb") as dst:
        for _ in range(expected_ids):
            pair = src.read(struct.calcsize("<If"))
            vector_id = struct.unpack_from("<I", pair)[0]
            dst.write(struct.pack("<I", vector_id))
    return favor_gt_file


def merged_search_cfg(cfg: dict, dataset_cfg: dict) -> dict:
    defaults = {
        "K": cfg.get("topk", 10),
        "num_repeats": 1,
        "num_threads": cfg.get("build", {}).get("num_threads", 32),
        "lsearch_start": 100,
        "lsearch_end": 100,
        "lsearch_step": 100,
    }
    if cfg.get("ef_values"):
        values = [int(v) for v in cfg["ef_values"]]
        defaults.update({"lsearch_start": values[0], "lsearch_end": values[-1]})
        if len(values) >= 2:
            defaults["lsearch_step"] = values[1] - values[0]
    search_cfg = dict(defaults)
    search_cfg.update(cfg.get("search", {}))
    search_cfg.update(dataset_cfg.get("search", {}))
    return search_cfg


def result_query_dir_name(query_task: str, search_cfg: dict) -> str:
    return (
        f"{query_task}_{search_cfg['lsearch_start']}_"
        f"{search_cfg['lsearch_step']}_{search_cfg['lsearch_end']}"
    )


def search_values(cfg: dict, dataset_cfg: dict, method: dict, search_cfg: dict) -> list[int]:
    explicit = method.get("ef_values") or dataset_cfg.get("ef_values") or cfg.get("ef_values")
    if explicit:
        return [int(v) for v in explicit]
    return list(
        range(
            int(search_cfg["lsearch_start"]),
            int(search_cfg["lsearch_end"]) + 1,
            int(search_cfg["lsearch_step"]),
        )
    )


def resolve_dataset_files(cfg: dict, dataset_cfg: dict, method: dict | None = None) -> dict[str, Path | str]:
    method = method or {}
    dataset = dataset_cfg["dataset"]
    query_task = dataset_cfg["query_task"]
    data_dir = Path(cfg["data_root"]) / dataset
    result_root = Path(cfg["result_root"]) / dataset
    favor_dir = result_root / "favor" / query_task
    gt_default = result_root / "GroundTruth" / query_task / f"{dataset}_gt_labels_containment.bin"
    index_name = method.get("index_name", dataset_cfg.get("index_name", cfg.get("index_name", "FAVOR")))
    index_dir = result_root / "index" / index_name / "index_files"
    return {
        "dataset": dataset,
        "query_task": query_task,
        "base_fvecs": Path(dataset_cfg.get("base_fvecs", data_dir / f"{dataset}_base.fvecs")),
        "query_fvecs": Path(dataset_cfg.get("query_fvecs", data_dir / query_task / f"{dataset}_query.fvecs")),
        "base_labels": Path(dataset_cfg.get("base_labels", data_dir / f"{dataset}_base_labels.txt")),
        "query_labels": Path(dataset_cfg.get("query_labels", data_dir / query_task / f"{dataset}_query_labels.txt")),
        "condition_file": Path(dataset_cfg.get("condition_file", favor_dir / "conditions.txt")),
        "gt_file": Path(dataset_cfg.get("groundtruth_file", gt_default)),
        "favor_gt_file": Path(dataset_cfg.get("favor_groundtruth_file", favor_dir / f"{dataset}_favor_gt_ids.bin")),
        "index_file": Path(dataset_cfg.get("index_file", index_dir / "index.bin")),
        "label_schema_file": Path(dataset_cfg.get("label_schema_file", str(index_dir / "index.bin") + ".labels")),
        "index_name": index_name,
    }


def threshold_name(value: float | str) -> str:
    return str(float(value)).rstrip("0").rstrip(".")


def expand_prefilter_threshold_methods(method: dict, cfg: dict) -> list[dict]:
    thresholds = method.get("prefilter_selectivity_thresholds", cfg.get("prefilter_selectivity_thresholds"))
    if not thresholds:
        return [method]
    base_name = method.get("name", cfg.get("method_name", "favor"))
    expanded = []
    for threshold in thresholds:
        threshold_label = threshold_name(threshold)
        threshold_method = dict(method)
        threshold_method["name"] = f"{base_name}_{threshold_label}"
        threshold_method["prefilter_selectivity_threshold"] = float(threshold)
        threshold_method.pop("prefilter_selectivity_thresholds", None)
        expanded.append(threshold_method)
    return expanded


def configured_methods(cfg: dict) -> list[dict]:
    methods = [dict(method) for method in cfg.get("methods", [])]
    if not methods:
        methods = [
            {
                "name": cfg.get("method_name", "favor"),
                "index_name": cfg.get("index_name", "FAVOR"),
                "ef_values": cfg.get("ef_values"),
            }
        ]
    expanded_methods = []
    for method in methods:
        expanded_methods.extend(expand_prefilter_threshold_methods(method, cfg))
    return expanded_methods


def ensure_favor_binaries(cfg: dict) -> tuple[Path, Path, Path]:
    build_dir = Path(cfg.get("favor_build_dir", Path(cfg.get("favor_root", "FAVOR")) / "build"))
    build_bin = Path(cfg.get("favor_build_binary", build_dir / "app" / "build_index"))
    search_bin = Path(cfg.get("favor_search_binary", build_dir / "app" / "search"))
    search_sweep_bin = Path(cfg.get("favor_search_sweep_binary", build_dir / "app" / "search_sweep"))
    source_dir = cfg.get("favor_source_dir") or cfg.get("favor_root")
    auto_build = bool(cfg.get("auto_build_favor", True))

    if auto_build:
        if not source_dir:
            raise FileNotFoundError("auto_build_favor is true but favor_source_dir/favor_root is not configured")
        source_dir = Path(source_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        log(f"FAVOR auto-build source={source_dir} build={build_dir}")
        run_command(["cmake", "-S", source_dir, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release"])
        run_command(
            [
                "cmake",
                "--build",
                build_dir,
                f"-j{cfg.get('build_jobs', os.environ.get('BUILD_JOBS', 16))}",
                "--target",
                "build_index",
                "search",
                "search_sweep",
            ]
        )

    if not build_bin.exists():
        raise FileNotFoundError(f"missing FAVOR build_index binary: {build_bin}")
    if not search_bin.exists():
        raise FileNotFoundError(f"missing FAVOR search binary: {search_bin}")
    if not search_sweep_bin.exists():
        raise FileNotFoundError(f"missing FAVOR search_sweep binary: {search_sweep_bin}")
    return build_bin, search_bin, search_sweep_bin


def ensure_input_files(paths: dict[str, Path | str]) -> None:
    for key in ["base_fvecs", "query_fvecs", "base_labels", "query_labels", "gt_file"]:
        path = paths[key]
        assert isinstance(path, Path)
        if not path.exists():
            raise FileNotFoundError(f"missing {key}: {path}")


def append_index_build_time(
    output_path: Path,
    dataset: str,
    index_name: str,
    index_file: Path,
    build_time_seconds: float,
    index_memory: dict[str, int | float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "index_name",
        "build_time_seconds",
        "build_time_ms",
        "index_file",
        "index_memory_bytes",
        "index_memory_mib",
        "index_level0_bytes",
        "index_upper_level_bytes",
        "index_runtime_metadata_bytes",
    ]
    existing_rows = []
    if output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open(newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            existing_rows = list(reader)
            existing_fieldnames = list(reader.fieldnames or [])
        if existing_fieldnames != fieldnames:
            with output_path.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow({field: row.get(field, "") for field in fieldnames})

    with output_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if fp.tell() == 0:
            writer.writeheader()
        writer.writerow(
            {
                "dataset": dataset,
                "index_name": index_name,
                "build_time_seconds": f"{build_time_seconds:.6f}",
                "build_time_ms": f"{build_time_seconds * 1000.0:.3f}",
                "index_file": str(index_file),
                "index_memory_bytes": index_memory["index_memory_bytes"],
                "index_memory_mib": f"{index_memory['index_memory_mib']:.3f}",
                "index_level0_bytes": index_memory["index_level0_bytes"],
                "index_upper_level_bytes": index_memory["index_upper_level_bytes"],
                "index_runtime_metadata_bytes": index_memory[
                    "index_runtime_metadata_bytes"
                ],
            }
        )


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
            raise RuntimeError(f"missing FAVOR memory statistic {key} in {build_log}")
        stats[key] = float(match.group(1)) if key == "index_memory_mib" else int(match.group(1))
    return stats


def build_index(build_bin: Path, cfg: dict, paths: dict[str, Path | str]) -> None:
    index_file = paths["index_file"]
    label_schema_file = paths["label_schema_file"]
    assert isinstance(index_file, Path)
    assert isinstance(label_schema_file, Path)
    if index_file.exists() and label_schema_file.exists():
        log(f"[{paths['dataset']}] FAVOR index exists: {index_file}")
        return
    if index_file.exists():
        log(f"[{paths['dataset']}] FAVOR label schema missing; rebuilding: {index_file}")
    build_cfg = dict(cfg.get("build", {}))
    num_threads = int(build_cfg.get("num_threads", 32))
    max_degree = int(build_cfg.get("M", build_cfg.get("max_degree", 64)))
    ef_construction = int(build_cfg.get("ef_construction", build_cfg.get("Lbuild", 200)))
    index_file.parent.mkdir(parents=True, exist_ok=True)
    log(f"[{paths['dataset']}] FAVOR build start")
    build_log = index_file.parent.parent / "others" / "favor_build.log"
    started = time.perf_counter()
    run_command(
        [
            build_bin,
            paths["base_fvecs"],
            paths["base_labels"],
            index_file,
            str(num_threads),
            str(max_degree),
            str(ef_construction),
        ],
        build_log,
    )
    build_time_seconds = time.perf_counter() - started
    index_memory = parse_index_memory_stats(build_log)
    append_index_build_time(
        index_file.parent.parent / "others" / "index_build_time.csv",
        str(paths["dataset"]),
        str(paths.get("index_name", index_file.parent.parent.name)),
        index_file,
        build_time_seconds,
        index_memory,
    )
    log(f"[{paths['dataset']}] FAVOR build done in {build_time_seconds:.3f}s")


def parse_favor_metrics(log_path: Path) -> dict[str, float]:
    text = log_path.read_text(encoding="utf-8")
    ung_style = re.search(
        r"efs\s*=\s*([0-9]+)\s*,\s*time\s*=\s*([0-9.eE+-]+)\s*ms\s*,\s*avg_recall\s*=\s*([0-9.eE+-]+)",
        text,
    )
    if ung_style:
        qps_match = re.search(r"QPS\s*=\s*([0-9.eE+-]+)", text)
        return {
            "recall": float(ung_style.group(3)),
            "average_ms": float(ung_style.group(2)),
            "qps": float(qps_match.group(1)) if qps_match else 0.0,
        }

    patterns = {
        "recall": r"recall\s*=\s*([0-9.eE+-]+)",
        "average_ms": r"average latency\s*=\s*([0-9.eE+-]+)\s*ms",
        "qps": r"QPS\s*=\s*([0-9.eE+-]+)",
    }
    metrics = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise RuntimeError(f"missing FAVOR metric {key} in {log_path}")
        metrics[key] = float(match.group(1))
    return metrics


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {
        "recall": sum(m["recall"] for m in metrics) / len(metrics),
        "average_ms": sum(m["average_ms"] for m in metrics) / len(metrics),
        "qps": sum(m["qps"] for m in metrics) / len(metrics),
    }


def metric_text(value: float) -> str:
    return str(round(value, 6)).rstrip("0").rstrip(".") if value % 1 else str(float(value)).rstrip("0").rstrip(".") + ".0"


def write_search_summaries(raw_results_dir: Path, rows: list[dict[str, float | int]]) -> None:
    raw_results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = raw_results_dir / "search_time_summary.csv"
    qps_csv = raw_results_dir / "search_time_summary_qps.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["Lsearch", "Average_Efs", "Average_Time_ms", "Average_Recall"])
        for row in rows:
            writer.writerow([
                row["Lsearch"],
                row["Average_Efs"],
                metric_text(float(row["Average_Time_ms"])),
                metric_text(float(row["Average_Recall"])),
            ])
    with qps_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["Lsearch", "Average_Efs", "Average_Time_ms", "Average_Recall", "QPS"])
        for row in rows:
            writer.writerow([
                row["Lsearch"],
                row["Average_Efs"],
                metric_text(float(row["Average_Time_ms"])),
                metric_text(float(row["Average_Recall"])),
                metric_text(float(row["QPS"])),
            ])


def run_searches(search_sweep_bin: Path, cfg: dict, dataset_cfg: dict, method: dict, paths: dict[str, Path | str]) -> None:
    search_cfg = merged_search_cfg(cfg, dataset_cfg)
    topk = int(search_cfg.get("K", cfg.get("topk", 10)))
    num_repeats = int(search_cfg.get("num_repeats", 1))
    num_threads = int(search_cfg.get("num_threads", 1))
    ef_values = search_values(cfg, dataset_cfg, method, search_cfg)
    favor_gt_file = prepare_favor_groundtruth_file(
        paths["gt_file"],
        paths["favor_gt_file"],
        topk,
        len(read_label_rows(paths["query_labels"])),
    )
    result_query_task = result_query_dir_name(str(paths["query_task"]), search_cfg)
    result_dir = Path(cfg["result_root"]) / str(paths["dataset"]) / "results" / method.get("name", "favor") / result_query_task
    raw_results_dir = result_dir / "results"
    other_dir = result_dir / "others"
    log(f"[{paths['dataset']}][{method.get('name', 'favor')}] FAVOR search sweep start")
    log_path = other_dir / f"{paths['dataset']}_search_output.txt"
    command = [
        search_sweep_bin,
        paths["base_fvecs"],
        paths["query_fvecs"],
        paths["label_schema_file"],
        str(topk),
        favor_gt_file,
        paths["condition_file"],
        paths["index_file"],
        raw_results_dir,
        str(num_repeats),
        "--num_threads",
        str(num_threads),
    ]
    if "prefilter_selectivity_threshold" in method:
        command.extend([
            "--prefilter_selectivity_threshold",
            threshold_name(method["prefilter_selectivity_threshold"]),
        ])
    command.extend(str(ef) for ef in ef_values)
    run_command(command, log_path)
    log(f"[{paths['dataset']}][{method.get('name', 'favor')}] search done")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_favor_experiment.py <config.json>", file=sys.stderr)
        return 2
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    log(f"config: {cfg_path}")
    build_bin, _search_bin, search_sweep_bin = ensure_favor_binaries(cfg)

    for dataset_cfg in cfg["datasets"]:
        for method in configured_methods(cfg):
            paths = resolve_dataset_files(cfg, dataset_cfg, method)
            ensure_input_files(paths)
            prepare_favor_condition_file(paths["query_labels"], paths["condition_file"])
            build_index(build_bin, cfg, paths)
            run_searches(search_sweep_bin, cfg, dataset_cfg, method, paths)
    log("all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
