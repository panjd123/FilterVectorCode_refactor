#!/usr/bin/env python3
"""Trie block build timings across datasets, including fresh base Trie per sample.

Whole-index timing exception: payload=1, process repeats, no L2 bypass or CUDA
Graph capture. Every timed CPU/GPU sample runs inside gpulock perf on GPU0.
Compile, lock wait, validation, reports and optional output cleanup are excluded.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
COMMON_PATH = HERE.parent / "amazon_intra_build_ablation/run.py"
SPEC = importlib.util.spec_from_file_location("trie_ablation_common", COMMON_PATH)
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    settings["source_config_dir"] = str((Path(path).resolve().parent / settings["source_config_dir"]).resolve())
    settings["config_dir"] = str((Path(path).resolve().parent / settings["config_dir"]).resolve())
    return settings


def make_config(settings, dataset, variant):
    source = Path(settings["source_config_dir"]) / f"{dataset}_config_Trie_block.json"
    base = json.loads(source.read_text())
    if base["datasets"] != [dataset]:
        raise ValueError(f"dataset mismatch: {source}")
    config = common.effective_config(base, settings, variant)
    config.update(run_name=f"{dataset.lower()}_trie_block_build_{variant['name']}",
                  index_name=f"Trie_block_500_M_32_cross_block_4_{variant['name']}",
                  index_name_params=[])
    config["ung_env"].update({
        "UNG_BASE_GROUP_TOPOLOGY": "trie", "UNG_SPECIAL_BLOCK_PARTITION": "trie",
        "UNG_SPECIAL_BLOCK_MIN_POINTS": "500", "UNG_SPECIAL_BLOCK_MAX_DEGREE": "32",
        "UNG_SPECIAL_BLOCK_NUM_CROSS_EDGES": "4", "UNG_HIERARCHY_LAYERS": "500:trie",
        "UNG_SPECIAL_INTER_RESERVE": str(int(settings["inter_reserve"])),
        "UNG_SPECIAL_INTER_PROFILE_NO_OUTPUT": "0", "UNG_SPECIAL_INTER_MAX_PAIR_WORK": "0",
        "UNG_SPECIAL_INTER_SAMPLE_CANDIDATES": "0", "UNG_SPECIAL_INTER_ROUTE_PROFILE": "1",
        "UNG_SPECIAL_EDGE_BINARY_ONLY": "1", "UNG_SPECIAL_EDGE_BINARY": "1",
    })
    config["build"].update(max_degree=32, num_cross_edges=4)
    config["special_block_min_points_values"] = [500]
    config["special_block_max_degree_values"] = [32]
    config["special_block_num_cross_edges_values"] = [4]
    return config


def generate(settings, output):
    paths = []
    for dataset in settings["datasets"]:
        for variant in settings["variants"]:
            path = output / dataset / (variant["name"] + ".json")
            content = json.dumps(make_config(settings, dataset, variant), indent=2) + "\n"
            if path.exists() and path.read_text() != content:
                raise ValueError(f"refusing to replace edited configuration: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            paths.append(path)
    return paths


def process_env(config):
    env = {key: value for key, value in os.environ.items() if not key.startswith("UNG_")}
    # Diagnostic preload libraries must never contaminate official timings.
    env.pop("LD_PRELOAD", None)
    env.pop("AMAZON_MUTEX_PROFILE", None)
    env.update(config["ung_env"])
    env.update(OMP_PROC_BIND="close", OMP_PLACES="cores", OMP_DYNAMIC="FALSE",
               OMP_NUM_THREADS=str(config["build"]["num_threads"]), MALLOC_ARENA_MAX="2")
    return env


def load_configs(settings, datasets, variants):
    configs = {}
    factors = {"UNG_SPECIAL_BLOCK_GPU_INTRA", "UNG_SPECIAL_BLOCK_GPU_INTRA_STRICT",
               "UNG_SPECIAL_BLOCK_GPU_INTRA_REQUIRE_USED", "UNG_TAGORE_BATCH_STREAMS", "UNG_TAGORE_OPTIMIZED_TOPK"}
    for dataset in datasets:
        reference = None
        for variant in variants:
            path = Path(settings["config_dir"]) / dataset / (variant["name"] + ".json")
            config = json.loads(path.read_text())
            env = config["ung_env"]
            expected = {"UNG_SPECIAL_BLOCK_GPU_INTRA": str(int(variant["gpu_intra"])),
                        "UNG_SPECIAL_BLOCK_GPU_INTRA_STRICT": str(int(variant["gpu_intra"])),
                        "UNG_SPECIAL_BLOCK_GPU_INTRA_REQUIRE_USED": str(int(variant["gpu_intra"])),
                        "UNG_TAGORE_BATCH_STREAMS": str(settings["streams"] if variant["load_balance"] else 1),
                        "UNG_TAGORE_OPTIMIZED_TOPK": str(int(variant["optimized_topk"]))}
            if config["datasets"] != [dataset] or any(env.get(k) != v for k, v in expected.items()):
                raise ValueError(f"configuration does not match dataset/variant: {path}")
            if not config.get("build_ung") or config.get("reuse_existing_index"):
                raise ValueError(f"configuration must build a fresh base Trie: {path}")
            if env.get("UNG_SPECIAL_INTER_PROFILE_NO_OUTPUT", "0") != "0" or env.get("UNG_SPECIAL_BLOCK_GPU_INTER") != "0":
                raise ValueError(f"configuration must retain CPU inter edges: {path}")
            common_config = {k: v for k, v in config.items() if k not in ("run_name", "index_name", "ung_env")}
            common_config["ung_env"] = {k: v for k, v in env.items() if k not in factors}
            if reference is not None and common_config != reference:
                raise ValueError(f"unrelated settings differ between variants: {path}")
            reference = common_config
            configs[dataset, variant["name"]] = config
    if len({config["build_dir"] for config in configs.values()}) != 1:
        raise ValueError("all selected cases must use the same builder directory")
    return configs


def input_info(config):
    dataset = config["datasets"][0]
    data = Path(config["data_root"]) / dataset
    info = {}
    for name in (f"{dataset}_base.bin", f"{dataset}_base_labels.txt"):
        path = data / name
        stat = path.stat()
        info[name] = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    with (data / f"{dataset}_base.bin").open("rb") as stream:
        points, dim = struct.unpack("<II", stream.read(8))
    if info[f"{dataset}_base.bin"]["bytes"] != 8 + 4 * points * dim:
        raise ValueError(f"unexpected float vector file size: {dataset}")
    return {"points": points, "dim": dim, "files": info}


def collect(step, root, identity=None):
    case = root / step["case"]
    config = json.loads((case / "config.json").read_text())
    if digest(case / "config.json") != step["config_sha256"]:
        raise ValueError(f"case configuration changed: {case}")
    meta = common.read_meta(case / "block_index_files/meta")
    base_meta = common.read_meta(case / "index_files/meta")
    if base_meta["base_group_topology"] != "trie" or meta["special_block_partition"] != "trie":
        raise ValueError(f"non-Trie index: {case}")
    log = (case / "build.log").read_text()
    observed = common.validate(meta, config, step["variant"], log, identity)
    if config["ung_env"]["UNG_SPECIAL_INTER_RESERVE"] == "1" and "[inter_reserve]" not in log:
        raise ValueError(f"missing inter-edge reservation: {case}")
    timing = json.loads((case / "timing.json").read_text())
    profile = common.profile_values(log)
    variant = step["variant"]
    row = {"dataset": step["dataset"], "variant": variant["name"], "phase": step["phase"],
           "repeat": step["repeat"], "gpu_intra": int(variant["gpu_intra"]),
           "load_balance": int(variant["load_balance"]), "optimized_topk": int(variant["optimized_topk"]),
           **{name: float(meta[key]) for name, key in common.METRICS.items()},
           **common.pipeline_values(base_meta, meta, timing),
           **{key: profile.get(key, "") for key in ("graph_pairs", "graph_queries", "graph_pair_work",
                                                  "graph_edges", "graph_distance_computations")},
           "gpu_blocks": int(meta["gpu_intra_blocks"]), "cpu_blocks": int(meta["cpu_intra_vamana_blocks"]),
           "block_count": int(meta["special_block_count"]),
           "block_cpu_ms": timing["block_cpu_ms"], "case_dir": str(case)}
    return row, observed


def write_results(root, manifest):
    rows, identities = [], {}
    for step in manifest["steps"]:
        if step["status"] == "complete":
            row, observed = collect(step, root, identities.get(step["dataset"]))
            identities[step["dataset"]] = observed
            rows.append(row)
    common.write_csv(root / "all_runs.csv", rows)
    summary = []
    for dataset in manifest["datasets"]:
        summary.extend({"dataset": dataset, **row} for row in common.summarize(
            [row for row in rows if row["dataset"] == dataset]))
    common.write_csv(root / "comparison.csv", summary)
    names = [v["name"] for v in manifest["variants"]]
    text = ["# Trie block 端到端构建计时", "",
            "每个样本从原始数据重新构建基础 Trie，再构建块索引；包含两阶段加载、构建和保存。",
            "编译、GPU 锁等待、校验及汇总不计入时间。各样本完整保留块间连边及保存阶段。", "",
            "下表单位秒，显示端到端中位数；括号为正式样本数，预热已排除。空格表示尚未完成。",
            "单样本结果仅作初测，不用于断言稳定收益。", "",
            "| 数据集 | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for dataset in manifest["datasets"]:
        cells = []
        for name in names:
            row = next((r for r in summary if r["dataset"] == dataset and r["variant"] == name), None)
            cells.append(f"{row['end_to_end_wall_ms_median']/1000:.3f} ({row['samples']})" if row else "—")
        text.append("| " + dataset + " | " + " | ".join(cells) + " |")
    if len(names) == 1:
        text += ["", "阶段拆分（秒，中位数）：", "",
                 "| 数据集 | 基础 Trie 墙钟 | 块内图 | 块间边 | 块索引保存 | 块阶段墙钟 | 端到端墙钟 |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for dataset in manifest["datasets"]:
            row = next((r for r in summary if r["dataset"] == dataset), None)
            metrics = ("base_wall_ms", "intra_ms", "inter_ms", "save_ms",
                       "block_wall_ms", "end_to_end_wall_ms")
            cells = [f"{row[key + '_median']/1000:.3f}" if row else "—" for key in metrics]
            text.append("| " + dataset + " | " + " | ".join(cells) + " |")
    text += ["", f"已完成 {sum(s['status']=='complete' for s in manifest['steps'])}/{len(manifest['steps'])} 个进程样本。",
             "详细阶段时间、最小/最大值见 comparison.csv；所有失败保留在 manifest.json 和各 case 的 build.log 中。", ""]
    (root / "comparison.md").write_text("\n".join(text))


def main(argv=None, on_progress=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--variant", action="append")
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--generate-configs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--discard-index-data", action="store_true",
                        help="after timing, remove only this run's index payloads; retain meta/logs/timings")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    if args.generate_configs:
        paths = generate(settings, Path(settings["config_dir"]))
        print(f"Generated {len(paths)} configurations in {settings['config_dir']}")
        return 0
    if args.summarize:
        root = args.summarize.resolve()
        write_results(root, json.loads((root / "manifest.json").read_text()))
        print(root / "comparison.md")
        return 0
    if args.resume:
        if args.run_root or args.dataset or args.variant or args.repeat is not None or args.warmup is not None or args.dry_run:
            parser.error("resume uses the original manifest; do not combine with selection/plan arguments")
        root = args.resume.resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["dry_run"]:
            parser.error("dry-run plans cannot be resumed; start a new real run")
    else:
        datasets = list(dict.fromkeys(args.dataset or settings["datasets"]))
        names = list(dict.fromkeys(args.variant or [v["name"] for v in settings["variants"]]))
        if any(d not in settings["datasets"] for d in datasets) or any(n not in [v["name"] for v in settings["variants"]] for n in names):
            parser.error("unknown dataset or variant")
        variants = [next(v for v in settings["variants"] if v["name"] == name) for name in names]
        configs = load_configs(settings, datasets, variants)
        repeat = settings["repeat"] if args.repeat is None else args.repeat
        warmup = settings["warmup"] if args.warmup is None else args.warmup
        if repeat < 1 or warmup < 0:
            parser.error("repeat must be positive and warmup nonnegative")
        root = (args.run_root or Path(settings["result_root"]) /
                datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")).resolve()
        root.mkdir(parents=True, exist_ok=False)
        manifest = {"settings": settings, "datasets": datasets, "variants": variants,
                    "repeat": repeat, "warmup": warmup, "dry_run": args.dry_run,
                    "discard_index_data": args.discard_index_data, "timing_policy": __doc__,
                    "status": "planned" if args.dry_run else "running", "inputs": {}, "steps": []}
        for dataset in datasets:
            for phase, r, variant in common.plan(settings, variants, repeat, warmup):
                # JSON files are the source of truth. Snapshot them per case;
                # do not silently regenerate and ignore the user's edits.
                config = json.loads(json.dumps(configs[dataset, variant["name"]]))
                manifest["inputs"][dataset] = input_info(config)
                case = root / dataset / f"{phase}_r{r:02d}" / variant["name"]
                case.mkdir(parents=True)
                config["ung_index_path_prefix"] = str(case / "index_files")
                path = case / "config.json"
                path.write_text(json.dumps(config, indent=2) + "\n")
                command = ["gpulock", "perf", "--wait-gpu-idle", str(settings["gpu"]), "--",
                           sys.executable, str(COMMON_PATH), "--execute-case", str(path)]
                manifest["steps"].append({"dataset": dataset, "phase": phase, "repeat": r,
                                          "variant": variant, "case": str(case.relative_to(root)),
                                          "config_sha256": digest(path), "command": command, "status": "pending"})
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Output: {root}", flush=True)
    if args.dry_run:
        for step in manifest["steps"]:
            print(shlex.join(step["command"]))
        write_results(root, manifest)
        return 0
    first = json.loads((root / manifest["steps"][0]["case"] / "config.json").read_text())
    build_dir = Path(first["build_dir"])
    if not args.skip_build and not args.resume:
        with (root / "compile.log").open("w") as out:
            subprocess.run(["cmake", "--build", str(build_dir), "-j", "8", "--target",
                            "build_UNG_index", "build_special_block_index"], stdout=out,
                           stderr=subprocess.STDOUT, check=True)
    binaries = {name: digest(build_dir / "apps" / name) for name in ("build_UNG_index", "build_special_block_index")}
    if args.resume and binaries != manifest.get("binary_sha256", binaries):
        raise ValueError("builder binaries changed; start a new experiment")
    manifest["binary_sha256"] = binaries
    # The timed helper itself must remain fixed during a resumable experiment.
    if args.resume and digest(COMMON_PATH) != manifest.get("timing_worker_sha256", digest(COMMON_PATH)):
        raise ValueError("timing worker changed; start a new experiment")
    manifest["timing_worker_sha256"] = digest(COMMON_PATH)
    manifest["status"] = "running"
    for step in manifest["steps"]:
        config = json.loads((root / step["case"] / "config.json").read_text())
        if digest(root / step["case"] / "config.json") != step["config_sha256"]:
            raise ValueError("case configuration changed")
        if input_info(config) != manifest["inputs"][step["dataset"]]:
            raise ValueError("dataset input changed")
        if step["status"] == "complete":
            continue
        case = root / step["case"]
        # A failed/interrupted sample is never reused as a successful timing.
        # Preserve its files, then run from scratch in a distinct attempt folder.
        if any((case / name).exists() for name in ("index_files", "timing.json", "build.log")):
            old_case = case
            attempt = 2
            while case.exists():
                case = old_case.with_name(old_case.name + f"_attempt{attempt}")
                attempt += 1
            case.mkdir()
            shutil.copyfile(old_case / "config.json", case / "config.json")
            config["ung_index_path_prefix"] = str(case / "index_files")
            (case / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            step["case"] = str(case.relative_to(root))
            step["config_sha256"] = digest(case / "config.json")
            step["command"][-1] = str(case / "config.json")
        step["status"] = "running"
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{step['dataset']} {step['phase']} r{step['repeat']}: {step['variant']['name']}", flush=True)
        try:
            with (case / "build.log").open("w") as log:
                log.write(__doc__ + "\n"); log.flush()
                subprocess.run(step["command"], env=process_env(config), stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            collect(step, root)
            step["status"] = "complete"
            write_results(root, manifest)
            if manifest["discard_index_data"]:
                # These exact fresh per-case directories are owned by this run.
                # Keep metadata and timings so completed samples remain auditable.
                for name in ("index_files", "block_index_files"):
                    directory = case / name
                    for payload in directory.iterdir():
                        if payload.name != "meta":
                            if payload.is_dir(): shutil.rmtree(payload)
                            else: payload.unlink()
                step["index_payloads_discarded"] = True
        except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as error:
            step["status"] = "failed"
            step["error"] = str(error)
            print(f"FAILED: {error}; see {case / 'build.log'}", flush=True)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if on_progress is not None:
            on_progress(root, manifest)
    manifest["status"] = "complete" if all(s["status"] == "complete" for s in manifest["steps"]) else "partial"
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_results(root, manifest)
    if on_progress is not None:
        on_progress(root, manifest)
    print((root / "comparison.md").read_text(), flush=True)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Experiment failed: {error}", file=sys.stderr)
        raise SystemExit(1)
