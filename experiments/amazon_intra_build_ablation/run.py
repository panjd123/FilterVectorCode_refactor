#!/usr/bin/env python3
"""Whole-index experiment: gpulock perf per process, not a CUDA microbenchmark.

Documented timing exception: payload=1, no L2 bypass or CUDA graph capture,
because each sample includes CPU packing/assembly, CUDA allocation, and file I/O.
Use independent-process warmups/repeats and report metadata stage wall times.
"""

import argparse
import copy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shlex
import statistics
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
METRICS = {
    "intra_ms": "special_edge_intra_build_time(ms)",
    "block_index_ms": "build_time(ms)",
    "inter_ms": "special_edge_inter_build_time(ms)",
    "metadata_ms": "special_block_metadata_time(ms)",
    "save_ms": "special_blocks_save_time(ms)",
}
PIPELINE_METRICS = ("base_build_timer_ms", "total_index_build_ms", "base_wall_ms",
                    "block_wall_ms", "end_to_end_wall_ms")
IDENTITY = ("source_ung_fingerprint", "num_points", "num_groups",
            "hierarchy_layers", "special_block_count", "special_block_member_points",
            "special_block_max_degree", "special_block_num_cross_edges")


def read_meta(path):
    return dict(line.split("=", 1) for line in Path(path).read_text().splitlines() if "=" in line)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def effective_config(base, settings, variant):
    config = copy.deepcopy(base)
    config.update(build_ung=True, reuse_existing_index=False)
    config.pop("ung_index_path_prefix", None)
    env = config["ung_env"]
    # Freeze unrelated routing knobs, including those otherwise inherited from
    # the invoking shell, so every GPU variant uses the same FastGrnnd pruning.
    env.update({
        "UNG_SPECIAL_BLOCK_GPU_INTRA": str(int(variant["gpu_intra"])),
        "UNG_SPECIAL_BLOCK_GPU_INTRA_STRICT": str(int(variant["gpu_intra"])),
        "UNG_SPECIAL_BLOCK_GPU_INTRA_REQUIRE_USED": str(int(variant["gpu_intra"])),
        "UNG_SPECIAL_BLOCK_GPU_INTER": "0",
        "UNG_SPECIAL_INTRA_ROUTE": "0",
        "UNG_SPECIAL_INTRA_COMPLETE_NX": "128",
        "UNG_SPECIAL_INTRA_SAMPLE_CANDIDATES": "0",
        "UNG_SPECIAL_INTRA_SAMPLE_MODE": "topk",
        "UNG_TAGORE_BATCH_STREAMS": str(settings["streams"] if variant["load_balance"] else 1),
        "UNG_SPECIAL_BLOCK_GPU_INTRA_BATCH_POINTS": str(settings["batch_points"]),
        "UNG_TAGORE_OPTIMIZED_TOPK": str(int(variant["optimized_topk"])),
        "UNG_FAST_GRNND_BATCH_EXACT_NX": "0",
        "UNG_TAGORE_COMPACT_D2H": "1",
        "UNG_FAST_GRNND_LIGHT_PRUNE_NX": "256",
        "UNG_FAST_GRNND_FORWARD_CAP": "64",
        "UNG_FAST_GRNND_REVERSE_CAP": "32",
        "UNG_UPPER_HNSW_ENABLE": "0",
    })
    return config


def plan(config, variants, repeat, warmup):
    steps = []
    for phase, rounds in (("warmup", warmup), ("measured", repeat)):
        for r in range(rounds):
            # Rotate order across rounds to reduce systematic first/last bias.
            ordered = variants[r % len(variants):] + variants[:r % len(variants)]
            for variant in ordered:
                steps.append((phase, r + 1, variant))
    return steps


def builder_command(config, case):
    build = config["build"]
    command = [str(Path(config["build_dir"]) / "apps/build_special_block_index"),
               "--ung_index_path_prefix", str(case / "index_files"),
               "--block_index_path_prefix", str(case / "block_index_files"),
               "--result_path_prefix", str(case / "block_results")]
    for key in ("data_type", "dist_fn", "num_threads", "max_degree", "num_cross_edges", "Lbuild", "alpha"):
        command.extend(["--" + key, str(build[key])])
    command.extend(["--min_points", config["ung_env"]["UNG_SPECIAL_BLOCK_MIN_POINTS"]])
    return command


def base_builder_command(config, case):
    dataset = config["datasets"][0]
    data = Path(config["data_root"]) / dataset
    command = [str(Path(config["build_dir"]) / "apps/build_UNG_index")]
    for key in ("data_type", "dist_fn", "num_threads", "max_degree", "num_cross_edges", "Lbuild", "alpha", "scenario"):
        command.extend(["--" + key, str(config["build"][key])])
    info = next((p for p in (data / f"{dataset}_base_labels_info.log", data / "log")
                 if p.is_file()), case / "placeholder.txt")
    roots = data / "tree_roots.txt"
    command.extend(["--dataset", dataset,
                    "--base_bin_file", str(data / f"{dataset}_base.bin"),
                    "--base_label_file", str(data / f"{dataset}_base_labels.txt"),
                    "--base_label_info_file", str(info),
                    "--base_label_tree_roots", str(roots if roots.is_file() else case / "placeholder.txt"),
                    "--index_path_prefix", str(case / "index_files") + "/",
                    "--result_path_prefix", str(case / "base_results") + "/"])
    return command


def execute_case(config_path):
    """Runs inside gpulock; measure after lock acquisition, before base loading."""
    config_path = Path(config_path).resolve()
    case = config_path.parent
    config = json.loads(config_path.read_text())
    for name in ("index_files", "base_results", "block_results"):
        (case / name).mkdir(exist_ok=False)
    (case / "placeholder.txt").touch(exist_ok=False)
    base_env = dict(os.environ)
    # Match the ordinary builder: build only the base here, then one sidecar.
    base_env.pop("UNG_SPECIAL_BLOCKS", None)
    commands = {"base": base_builder_command(config, case), "blocks": builder_command(config, case)}
    (case / "phase_commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (case / "base_build.log").open("w") as base_log:
        start = time.perf_counter()
        subprocess.run(commands["base"], env=base_env, stdout=base_log,
                       stderr=subprocess.STDOUT, check=True)
        base_end = time.perf_counter()
    usage_base = resource.getrusage(resource.RUSAGE_CHILDREN)
    print("Base Trie built; starting block index", flush=True)
    block_start = time.perf_counter()
    subprocess.run(commands["blocks"], check=True)
    end = time.perf_counter()
    usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    timing = {"base_wall_ms": (base_end - start) * 1000,
              "block_wall_ms": (end - block_start) * 1000,
              "end_to_end_wall_ms": (end - start) * 1000,
              "base_cpu_ms": ((usage_base.ru_utime + usage_base.ru_stime) -
                              (usage_before.ru_utime + usage_before.ru_stime)) * 1000,
              "block_cpu_ms": ((usage_end.ru_utime + usage_end.ru_stime) -
                               (usage_base.ru_utime + usage_base.ru_stime)) * 1000,
              "block_involuntary_context_switches": usage_end.ru_nivcsw - usage_base.ru_nivcsw,
              "scope": "fresh base Trie load/build/save followed by block load/build/save; lock wait and compilation excluded"}
    (case / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    return 0


def pipeline_values(base_meta, block_meta, timing):
    base_ms = float(base_meta["index_time_add_rb(ms)"])
    values = {"base_build_timer_ms": base_ms,
              "total_index_build_ms": base_ms + float(block_meta["build_time(ms)"]),
              **{key: float(timing[key]) for key in ("base_wall_ms", "block_wall_ms", "end_to_end_wall_ms")}}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("invalid pipeline timing")
    if values["end_to_end_wall_ms"] + 0.01 < values["base_wall_ms"] + values["block_wall_ms"]:
        raise ValueError("end-to-end wall time omits a phase")
    return values


def profile_values(log):
    line = next((line for line in log.splitlines() if "[cpu_inter_profile]" in line), "")
    return {key: int(value) for key, value in re.findall(
        r"\b(graph_pairs|graph_queries|graph_pair_work|graph_edges|graph_distance_computations)=(\d+)", line)}


def validate(meta, config, variant, log, identity=None):
    observed = {key: meta[key] for key in IDENTITY}
    if identity is not None and observed != identity:
        raise ValueError("source index, block partition, or degree settings differ across cases")
    expected_streams = config["ung_env"]["UNG_TAGORE_BATCH_STREAMS"]
    if meta["gpu_intra_requested_streams"] != expected_streams:
        raise ValueError("requested stream setting did not reach the builder")
    if meta["gpu_intra_optimized_topk"] != str(int(variant["optimized_topk"])):
        raise ValueError("top-k setting did not reach the builder")
    if int(meta["gpu_intra_fallback_blocks"]) != 0:
        raise ValueError("GPU fallback invalidates the experiment")
    gpu_blocks = int(meta["gpu_intra_blocks"])
    cpu_blocks = int(meta["cpu_intra_vamana_blocks"])
    if variant["gpu_intra"]:
        if gpu_blocks != int(meta["special_block_count"]) or cpu_blocks != 0:
            raise ValueError("Amazon GPU case did not build every block on CUDA")
        if int(meta["gpu_intra_points"]) != int(meta["special_block_member_points"]):
            raise ValueError("GPU point coverage is incomplete")
    elif gpu_blocks != 0 or cpu_blocks != int(meta["special_block_count"]):
        raise ValueError("Amazon CPU case did not build every block with CPU Vamana")
    if bool("[TagoreCuda][batch_balance]" in log) != bool(variant["load_balance"]):
        raise ValueError("observed balancing path does not match the requested variant")
    profile = profile_values(log)
    if profile.get("graph_queries", 0) > 0:
        comparisons = profile.get("graph_distance_computations", 0)
        if not profile["graph_queries"] <= comparisons <= profile["graph_pair_work"] + profile["graph_queries"]:
            raise ValueError("missing or inconsistent graph distance-comparison count")
    for field in METRICS.values():
        if not math.isfinite(float(meta[field])) or float(meta[field]) < 0:
            raise ValueError("invalid timing")
    return observed


def summarize(rows):
    summary = []
    names = list(dict.fromkeys(row["variant"] for row in rows))
    for name in names:
        samples = [row for row in rows if row["variant"] == name and row["phase"] == "measured"]
        if not samples:
            continue
        item = {"variant": name, "samples": len(samples),
                "gpu_intra": samples[0]["gpu_intra"],
                "load_balance": samples[0]["load_balance"],
                "optimized_topk": samples[0]["optimized_topk"]}
        metrics = list(METRICS) + [key for key in PIPELINE_METRICS if all(key in row for row in samples)]
        for metric in metrics:
            values = [row[metric] for row in samples]
            item[metric + "_median"] = statistics.median(values)
            item[metric + "_min"] = min(values)
            item[metric + "_max"] = max(values)
        summary.append(item)
    cpu = next((row for row in summary if row["variant"] == "cpu"), None)
    for row in summary:
        row["intra_speedup_vs_cpu"] = (cpu["intra_ms_median"] / row["intra_ms_median"]
                                        if cpu and row["intra_ms_median"] else "")
    return summary


def write_summary(root, rows):
    summary = summarize(rows)
    write_csv(root / "all_runs.csv", rows)
    write_csv(root / "comparison.csv", summary)
    text = ["# Amazon 块内建图对比", "",
            "topK 指候选更新中的阈值过滤、局部 Bitonic 排序和二分排名归并；所有 GPU 组均保留 FastGrnnd 剪枝。", "",
            "每个样本重新构建相同配置的基础 Trie 索引，再构建块索引。", "",
            "端到端墙钟从基础索引进程启动前计到块索引进程退出后，包含两阶段加载、构建及保存；不含编译与等待 GPU 锁。", "",
            "总构建计时 = 基础 index_time_add_rb(ms) + 块 build_time(ms)，用于对齐现有 memory_and_build_summary 的口径，和真实端到端墙钟分列。", "",
            "各方法的近似图可能不同；这些结果仅验证构建与时间，不证明召回率相同。", "",
            "| 方法 | 样本数 | 块内/s | 块间/s | 基础构建计时/s | 总构建计时/s | 端到端墙钟/s |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        text.append(f"| {row['variant']} | {row['samples']} | {row['intra_ms_median']/1000:.3f} | "
                    f"{row['inter_ms_median']/1000:.3f} | {row['base_build_timer_ms_median']/1000:.3f} | "
                    f"{row['total_index_build_ms_median']/1000:.3f} | {row['end_to_end_wall_ms_median']/1000:.3f} |")
    (root / "comparison.md").write_text("\n".join(text) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=HERE / "config.json")
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--only", action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--execute-case", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.execute_case:
        return execute_case(args.execute_case)
    settings = json.loads(args.config.read_text())
    base_path = (args.config.resolve().parent / settings["base_config"]).resolve()
    base = json.loads(base_path.read_text())
    repeat = args.repeat if args.repeat is not None else settings["repeat"]
    warmup = args.warmup if args.warmup is not None else settings["warmup"]
    gpu = args.gpu if args.gpu is not None else settings["gpu"]
    if repeat < 1 or warmup < 0:
        parser.error("repeat must be >=1 and warmup >=0")
    variants = settings["variants"]
    if args.only:
        by_name = {variant["name"]: variant for variant in variants}
        if any(name not in by_name for name in args.only):
            parser.error("unknown variant")
        variants = [by_name[name] for name in dict.fromkeys(args.only)]
    root = (args.run_root or Path(settings["result_root"]) /
            datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")).resolve()
    root.mkdir(parents=True, exist_ok=False)
    steps = []
    for phase, r, variant in plan(settings, variants, repeat, warmup):
        case = root / f"{phase}_r{r:02d}" / variant["name"]
        case.mkdir(parents=True)
        config = effective_config(base, settings, variant)
        config["ung_index_path_prefix"] = str(case / "index_files")
        (case / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        command = ["gpulock", "perf", "--wait-gpu-idle", str(gpu), "--",
                   sys.executable, str(HERE / "run.py"), "--execute-case", str(case / "config.json")]
        steps.append((phase, r, variant, case, config, command))
    manifest = {"settings": settings, "base_config": base, "repeat": repeat,
                "warmup": warmup, "gpu": gpu, "dry_run": args.dry_run,
                "timing_scope": "fresh base Trie plus independent block index, including both load/build/save phases",
                "timing_policy": "whole-index exception: payload=1, process repeats, no L2 bypass/CUDA graphs",
                "commands": [command for *_, command in steps],
                "status": "planned" if args.dry_run else "running"}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Output: {root}", flush=True)
    if args.dry_run:
        for *_, command in steps:
            print(shlex.join(command))
        return 0
    dataset = base["datasets"][0]
    for name in (f"{dataset}_base.bin", f"{dataset}_base_labels.txt"):
        if not (Path(base["data_root"]) / dataset / name).is_file():
            raise ValueError(f"missing dataset input: {name}")
    if not args.skip_build:
        with (root / "compile.log").open("w") as out:
            subprocess.run(["cmake", "--build", base["build_dir"], "-j", "8",
                            "--target", "build_UNG_index", "build_special_block_index"],
                           stdout=out, stderr=subprocess.STDOUT, check=True)
    manifest["binary_sha256"] = {name: hashlib.sha256(
        (Path(base["build_dir"]) / "apps" / name).read_bytes()).hexdigest()
        for name in ("build_UNG_index", "build_special_block_index")}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows, identity = [], None
    for phase, r, variant, case, config, command in steps:
        print(f"{phase} r{r}: {variant['name']}", flush=True)
        env = {key: value for key, value in os.environ.items() if not key.startswith("UNG_")}
        env.update(config["ung_env"])
        env.update(OMP_PROC_BIND="close", OMP_PLACES="cores", OMP_DYNAMIC="FALSE",
                   OMP_NUM_THREADS=str(config["build"]["num_threads"]), MALLOC_ARENA_MAX="2")
        with (case / "build.log").open("w") as out:
            out.write(manifest["timing_policy"] + "\n")
            out.flush()
            subprocess.run(command, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
        meta = read_meta(case / "block_index_files/meta")
        base_meta = read_meta(case / "index_files/meta")
        if base_meta["base_group_topology"] != "trie":
            raise ValueError("base index did not use Trie topology")
        log = (case / "build.log").read_text()
        timing = json.loads((case / "timing.json").read_text())
        identity = validate(meta, config, variant, log, identity)
        row = {"variant": variant["name"], "phase": phase, "repeat": r,
               "gpu_intra": int(variant["gpu_intra"]), "load_balance": int(variant["load_balance"]),
               "optimized_topk": int(variant["optimized_topk"]),
               **{name: float(meta[key]) for name, key in METRICS.items()},
               **pipeline_values(base_meta, meta, timing),
               **{key: profile_values(log).get(key, "") for key in
                  ("graph_pairs", "graph_queries", "graph_pair_work", "graph_edges", "graph_distance_computations")},
               "block_cpu_ms": timing["block_cpu_ms"],
               "block_involuntary_context_switches": timing["block_involuntary_context_switches"],
               "gpu_blocks": int(meta["gpu_intra_blocks"]),
               "cpu_blocks": int(meta["cpu_intra_vamana_blocks"]),
               "intra_edges": int(meta["special_edge_intra_count"]), "case_dir": str(case)}
        rows.append(row)
        write_summary(root, rows)
    print((root / "comparison.md").read_text(), flush=True)
    subprocess.run([sys.executable, str(HERE / "plot.py"), str(root / "comparison.csv")], check=True)
    manifest["status"] = "complete"
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Experiment failed: {error}", file=sys.stderr)
        raise SystemExit(1)
