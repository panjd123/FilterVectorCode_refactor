#!/usr/bin/env python3
"""Separate CPU inter-edge diagnostics; reuses a base Trie, NOT end-to-end timing.

No-output cases deliberately omit inter edges and must never be used as indexes.
All GPU work is serialized with gpulock. Lock sampling perturbs measurements;
these results explain bottlenecks and are not pooled with official samples.
Whole-index timing exception: payload=1, no L2 bypass or CUDA Graph capture.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

import run as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="existing index_files directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--only", choices=["topk_only", "load_balance_topk"])
    parser.add_argument("--modes", nargs="+", choices=["standard", "no_output", "arena16", "reserve"],
                        default=["standard", "no_output", "arena16"])
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat must be positive")
    source = args.source.resolve()
    if not (source / "meta").is_file():
        parser.error("source must contain a completed base Trie")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    profiler = root / "profile_mutex.so"
    subprocess.run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC",
                    str(experiment.HERE / "profile_mutex.cpp"), "-ldl", "-o", str(profiler)], check=True)
    settings = json.loads((experiment.HERE / "config.json").read_text())
    base = json.loads((experiment.HERE / settings["base_config"]).read_text())
    variants = [v for v in settings["variants"] if v["name"] in ("topk_only", "load_balance_topk")]
    if args.only:
        variants = [v for v in variants if v["name"] == args.only]
    cases = [(mode, variant) for mode in args.modes for variant in variants]
    binary = Path(base["build_dir"]) / "apps/build_special_block_index"
    manifest = {"scope": __doc__, "source": str(source), "repeat": args.repeat,
                "profiler_sha256": hashlib.sha256(profiler.read_bytes()).hexdigest(),
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "commands": []}
    rows = []
    identity = None
    for r in range(args.repeat):
        ordered = cases[r % len(cases):] + cases[:r % len(cases)]
        for mode, variant in ordered:
            case = root / f"r{r+1:02d}" / mode / variant["name"]
            case.mkdir(parents=True)
            config = experiment.effective_config(base, settings, variant)
            env = {k: v for k, v in os.environ.items() if not k.startswith("UNG_")}
            env.update(config["ung_env"])
            env.update(OMP_PROC_BIND="close", OMP_PLACES="cores", OMP_DYNAMIC="FALSE",
                       OMP_NUM_THREADS=str(config["build"]["num_threads"]),
                       MALLOC_ARENA_MAX="16" if mode == "arena16" else "2")
            # Cache-pool locking, output allocation and allocator contention are
            # tested independently. All preserve the search parameters.
            env["UNG_SPECIAL_INTER_ROUTE_PROFILE"] = "1"
            env["UNG_SPECIAL_INTER_PROFILE_NO_OUTPUT"] = str(int(mode == "no_output"))
            env["UNG_SPECIAL_INTER_RESERVE"] = str(int(mode == "reserve"))
            command = experiment.builder_command(config, case)
            command[command.index("--ung_index_path_prefix") + 1] = str(source)
            # Preload only the builder: preloading gpulock or its helper
            # processes would overwrite the same report when they exit.
            command = ["gpulock", "perf", "--wait-gpu-idle", str(settings["gpu"]), "--", "env",
                       "LD_PRELOAD=" + str(profiler), "AMAZON_MUTEX_PROFILE=" + str(case / "mutex.csv")] + command
            manifest["commands"].append({"mode": mode, "variant": variant["name"],
                                         "repeat": r + 1, "command": command,
                                         "env": {k: env[k] for k in (*config["ung_env"], "MALLOC_ARENA_MAX",
                                              "UNG_SPECIAL_INTER_PROFILE_NO_OUTPUT", "UNG_SPECIAL_INTER_RESERVE")}})
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"r{r+1}: {mode} {variant['name']}", flush=True)
            with (case / "build.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            meta = experiment.read_meta(case / "block_index_files/meta")
            log = (case / "build.log").read_text()
            identity = experiment.validate(meta, config, variant, log, identity)
            row = {"repeat": r + 1, "mode": mode, "variant": variant["name"],
                   **{name: float(meta[key]) for name, key in experiment.METRICS.items()},
                   **experiment.profile_values(log), "case_dir": str(case)}
            sites = {}
            with (case / "mutex.csv").open() as stream:
                for item in csv.DictReader(stream):
                    if Path(item["module"]).name != binary.name:
                        continue
                    key = item["offset"]
                    site = sites.setdefault(key, {"calls": 0, "samples": 0, "sampled_acquire_ns": 0})
                    for field in site:
                        site[field] += int(item[field])
            for offset, site in sites.items():
                site["symbol"] = subprocess.check_output(
                    ["addr2line", "-f", "-C", "-e", str(binary), offset], text=True).strip()
                site["estimated_acquire_thread_ms"] = (
                    site["sampled_acquire_ns"] * site["calls"] / site["samples"] / 1e6 if site["samples"] else 0)
            cache_sites = [site for site in sites.values() if "append_cross_edges_from_target_points" in site["symbol"]]
            row["cache_mutex_calls"] = sum(site["calls"] for site in cache_sites)
            row["cache_mutex_coverage"] = row["cache_mutex_calls"] / (2 * row["graph_queries"]) if row["graph_queries"] else 0
            row["cache_mutex_estimated_acquire_thread_ms"] = sum(site["estimated_acquire_thread_ms"] for site in cache_sites)
            (case / "mutex_summary.json").write_text(json.dumps(sites, indent=2) + "\n")
            rows.append(row)
            experiment.write_csv(root / "diagnostics.csv", rows)
            print(f"  intra={row['intra_ms']/1000:.3f}s inter={row['inter_ms']/1000:.3f}s "
                  f"cache-lock-acquire(thread sum)={row['cache_mutex_estimated_acquire_thread_ms']/1000:.3f}s", flush=True)
    manifest["status"] = "complete"
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
