#!/usr/bin/env python3

import argparse
import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")
DEFAULT_GENERATED_DIR = Path(__file__).with_name("generated")
DEFAULT_RUNNER = REPO_ROOT / "run_cpu_special_blocks_experiment.sh"


@dataclass(frozen=True)
class RunStep:
    name: str
    command: list[str]
    env: dict[str, str] | None = None


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def selected_variants(config: dict, only: list[str] | None = None) -> list[dict]:
    variants = config.get("variants", [])
    if not isinstance(variants, list) or not variants:
        raise ValueError("config must contain non-empty variants[]")
    if not only:
        return variants

    by_name = {variant.get("name"): variant for variant in variants}
    missing = [name for name in only if name not in by_name]
    if missing:
        raise ValueError(f"unknown variant name(s): {', '.join(missing)}")
    return [by_name[name] for name in only]


def materialize_variant(variant: dict, generated_dir: Path, *, repeat_index: int | None = None) -> Path:
    name = variant.get("name")
    config = variant.get("config")
    if not name or not isinstance(config, dict):
        raise ValueError("each variant must contain name and config object")

    config = copy.deepcopy(config)
    name_suffix = ""
    if repeat_index is not None:
        name_suffix = f"_r{repeat_index:02d}"
        config["run_name"] = f"{config.get('run_name', name)}{name_suffix}"
        config["index_name"] = f"{config.get('index_name', name)}{name_suffix}"

    generated_dir.mkdir(parents=True, exist_ok=True)
    out_path = generated_dir / f"{name}{name_suffix}.json"
    out_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return out_path


def stable_cpu_env(num_threads: int | None = None) -> dict[str, str]:
    env = {
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "OMP_DYNAMIC": "FALSE",
        "MALLOC_ARENA_MAX": "2",
    }
    if num_threads is not None:
        env["OMP_NUM_THREADS"] = str(num_threads)
    return env


def command_with_taskset(command: list[str], cpu_list: str | None) -> list[str]:
    if not cpu_list:
        return command
    return ["taskset", "-c", cpu_list, *command]


def plan_steps(
    config_path: Path,
    *,
    generated_dir: Path = DEFAULT_GENERATED_DIR,
    runner_path: Path = DEFAULT_RUNNER,
    only: list[str] | None = None,
    repeat: int = 1,
    cpu_list: str | None = None,
    stable_cpu: bool = False,
    interleave: bool = True,
) -> list[RunStep]:
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    config = load_config(config_path)
    variants = selected_variants(config, only)
    build_threads = config.get("shared", {}).get("build", {}).get("num_threads")
    try:
        build_threads = int(build_threads) if build_threads is not None else None
    except (TypeError, ValueError):
        build_threads = None
    child_env = stable_cpu_env(build_threads) if stable_cpu else None

    steps: list[RunStep] = []
    if interleave:
        sequence = [
            (variant, repeat_index)
            for repeat_index in range(1, repeat + 1)
            for variant in variants
        ]
    else:
        sequence = [
            (variant, repeat_index)
            for variant in variants
            for repeat_index in range(1, repeat + 1)
        ]
    for variant, repeat_index in sequence:
        materialized_repeat = repeat_index if repeat > 1 else None
        out_path = materialize_variant(variant, generated_dir, repeat_index=materialized_repeat)
        command = command_with_taskset([str(runner_path), str(out_path)], cpu_list)
        step_name = variant["name"] if repeat == 1 else f"{variant['name']}_r{repeat_index:02d}"
        steps.append(RunStep(step_name, command, child_env))
    return steps


def run_steps(steps: list[RunStep], *, continue_on_error: bool = False, dry_run: bool = False) -> int:
    worst_rc = 0
    for step in steps:
        print(f"[cross_edge_topk_ablation] {step.name}: {' '.join(step.command)}")
        if step.env:
            env_text = " ".join(f"{key}={value}" for key, value in sorted(step.env.items()))
            print(f"[cross_edge_topk_ablation] {step.name}: env {env_text}")
        if dry_run:
            continue
        env = {**os.environ, **step.env} if step.env else None
        completed = subprocess.run(step.command, env=env)
        if completed.returncode != 0:
            worst_rc = completed.returncode
            print(
                f"[cross_edge_topk_ablation] {step.name} failed with exit code {completed.returncode}",
                file=sys.stderr,
            )
            if not continue_on_error:
                return completed.returncode
    return worst_rc


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize and run all cross-edge top-k ablation variants."
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=DEFAULT_CONFIG,
        help="ablation config.json",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=DEFAULT_GENERATED_DIR,
        help="directory for generated per-variant configs",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=DEFAULT_RUNNER,
        help="path to run_cpu_special_blocks_experiment.sh",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="run only this variant name; can be passed multiple times",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write generated configs and print commands without executing them",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="run remaining variants after a failure",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run each selected variant this many times; repeat outputs get _rNN suffixes",
    )
    parser.add_argument(
        "--no-interleave",
        action="store_true",
        help="with --repeat, run all repeats of one variant before moving to the next",
    )
    parser.add_argument(
        "--taskset",
        metavar="CPU_LIST",
        help="prefix each run with taskset -c CPU_LIST, for example 0-59",
    )
    parser.add_argument(
        "--stable-cpu",
        action="store_true",
        help="set OpenMP binding/dynamic and malloc env vars to reduce CPU timing variance",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        steps = plan_steps(
            args.config,
            generated_dir=args.generated_dir,
            runner_path=args.runner,
            only=args.only or None,
            repeat=args.repeat,
            cpu_list=args.taskset,
            stable_cpu=args.stable_cpu,
            interleave=not args.no_interleave,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return run_steps(steps, continue_on_error=args.continue_on_error, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
