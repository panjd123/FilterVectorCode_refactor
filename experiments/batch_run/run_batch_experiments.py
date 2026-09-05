#!/usr/bin/env python3

import argparse
import copy
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

DEFAULT_METHODS = [
    "favor",
    "Navix",
    "curator",
    "UNG__hybrid",
    "cpu_bruteforce_els_special_blocks",
    "cpu_bruteforce_els_ung",
]

CPU_ELS_METHOD_NAMES = {
    "cpu_bruteforce_els_special_blocks",
    "cpu_bruteforce_els_ung",
}

CORE_UNG_INDEX_FILES = ["meta", "graph", "vecs.bin", "labels.txt"]
SPECIAL_BLOCK_INDEX_FILES = [
    "special_blocks.csv",
    "special_block_members.csv",
    "special_block_children.csv",
    "special_edges.csv",
]

COMMON_KEYS = [
    "data_root",
    "result_root",
    "build_dir",
    "source_dir",
    "selector_model_prefix",
    "build_jobs",
    "data_type",
    "dist_fn",
    "scenario",
]

GLOBAL_BUILD_KEYS_BY_TARGET = {
    "favor": {"num_threads", "max_degree", "Lbuild", "M", "ef_construction"},
    "Navix": {"num_threads", "max_degree", "Lbuild", "alpha", "index_type"},
    "curator": {"num_threads", "nlist", "bf_capacity", "bf_error_rate", "max_sl_size", "clus_niter", "max_leaf_size", "prune_thres", "variance_boost", "search_ef", "beam_size", "use_temp_index_caching"},
    "UNG__hybrid": {"data_type", "dist_fn", "scenario", "num_threads", "max_degree", "Lbuild", "alpha", "num_cross_edges"},
    "cpu_special_blocks_index": {"data_type", "dist_fn", "scenario", "num_threads", "max_degree", "Lbuild", "alpha", "num_cross_edges"},
    "search_comparison": set(),
}


@dataclass
class RunStep:
    name: str
    command: list[str]
    config: dict[str, Any]
    config_path: Path


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    cfg.setdefault("methods", DEFAULT_METHODS)
    cfg.setdefault("generated_config_dir", str(config_path.parent / "generated"))
    cfg.setdefault("continue_on_error", False)
    cfg.setdefault("ensure_special_blocks_index", True)
    cfg.setdefault("overrides", {})
    return cfg


def normalize_method_name(name: str) -> str:
    aliases = {
        "navix": "Navix",
        "NaviX": "Navix",
        "curator": "curator",
        "Curator": "curator",
        "FAVOR": "favor",
        "Favor": "favor",
        "ung_hybrid": "UNG__hybrid",
        "UNG_hybrid": "UNG__hybrid",
    }
    return aliases.get(name, name)


def method_names(cfg: dict[str, Any]) -> list[str]:
    return [normalize_method_name(str(method)) for method in cfg.get("methods", DEFAULT_METHODS)]


def dataset_objects(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = cfg.get("datasets", [])
    normalized: list[dict[str, Any]] = []
    for dataset in datasets:
        if isinstance(dataset, str):
            normalized.append(
                {
                    "dataset": dataset,
                    "query_task": cfg.get("default_query_task", "query_selected_recall_advantage"),
                }
            )
        elif isinstance(dataset, dict):
            if "dataset" not in dataset:
                raise ValueError(f"dataset entry is missing 'dataset': {dataset}")
            if "query_task" not in dataset:
                dataset = dict(dataset)
                dataset["query_task"] = cfg.get("default_query_task", "query_selected_recall_advantage")
            normalized.append(copy.deepcopy(dataset))
        else:
            raise ValueError(f"unsupported dataset entry: {dataset!r}")
    if not normalized:
        raise ValueError("config must contain at least one dataset")
    return normalized


def dataset_names(cfg: dict[str, Any]) -> list[str]:
    return [str(dataset["dataset"]) for dataset in dataset_objects(cfg)]


def common_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(cfg[key]) for key in COMMON_KEYS if key in cfg}


def template_config(relative_path: str) -> dict[str, Any]:
    return load_json(EXPERIMENTS_DIR / relative_path)


def generated_path(cfg: dict[str, Any], run_id: str, filename: str) -> Path:
    return Path(cfg["generated_config_dir"]) / run_id / filename


def apply_named_override(cfg: dict[str, Any], target: str, data: dict[str, Any]) -> dict[str, Any]:
    overrides = cfg.get("overrides", {})
    merged = deep_merge(data, overrides.get(target, {}))
    return merged


def filtered_build_overrides(cfg: dict[str, Any], target: str) -> dict[str, Any]:
    allowed = GLOBAL_BUILD_KEYS_BY_TARGET.get(target)
    build_cfg = cfg.get("build", {})
    if allowed is None:
        return copy.deepcopy(build_cfg)
    return {key: copy.deepcopy(value) for key, value in build_cfg.items() if key in allowed}


def apply_global_hyperparameters(cfg: dict[str, Any], data: dict[str, Any], target: str) -> dict[str, Any]:
    merged = copy.deepcopy(data)
    build_override = filtered_build_overrides(cfg, target)
    if build_override:
        merged["build"] = deep_merge(merged.get("build", {}), build_override)
    if "search" in cfg:
        merged["search"] = deep_merge(merged.get("search", {}), cfg["search"])
    return merged


def validate_sweep(sweep: dict[str, Any]) -> dict[str, int]:
    start = int(sweep["start"])
    end = int(sweep["end"])
    step = int(sweep["step"])
    if step <= 0:
        raise ValueError(f"sweep step must be positive: {sweep}")
    if end < start:
        raise ValueError(f"sweep end must be >= start: {sweep}")
    return {"start": start, "end": end, "step": step}


def expand_sweeps(search_cfg: dict[str, Any]) -> list[int] | None:
    if "values" in search_cfg:
        return [int(value) for value in search_cfg["values"]]
    sweeps = search_cfg.get("sweeps")
    if not sweeps:
        return None
    values: list[int] = []
    for raw_sweep in sweeps:
        sweep = validate_sweep(raw_sweep)
        values.extend(range(sweep["start"], sweep["end"] + 1, sweep["step"]))
    deduped: list[int] = []
    seen = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped




def method_search_cfg(cfg: dict[str, Any], method: str) -> dict[str, Any]:
    method_search = cfg.get("method_search", {})
    return copy.deepcopy(method_search.get(method, {}))


def clean_index_name_part(value: Any) -> str:
    text = str(value).strip().replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return text or "empty"


def nested_value(root: dict[str, Any], dotted_key: str) -> Any:
    current: Any = root
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def effective_index_name(index_cfg: dict[str, Any]) -> str:
    base = str(index_cfg.get("index_name", ""))
    parts: list[str] = []
    for item in index_cfg.get("index_name_params", []):
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if not key:
            continue
        source = item.get("source", "env")
        if source == "build":
            value = nested_value(index_cfg.get("build", {}), str(key))
        elif source == "env":
            value = index_cfg.get("ung_env", {}).get(str(key))
        else:
            value = None
        if value is None:
            continue
        label = item.get("label", key)
        parts.append(f"{clean_index_name_part(label)}{clean_index_name_part(value)}")
    return base + ("_" + "_".join(parts) if parts else "")


def index_files_exist(index_cfg: dict[str, Any], dataset: str, index_name: str, *, special_blocks: bool = False) -> bool:
    index_dir = Path(index_cfg.get("result_root", "/home/dev/graphdb/FilterVectorResult")) / dataset / "index" / index_name / "index_files"
    required = list(CORE_UNG_INDEX_FILES)
    if special_blocks:
        required.extend(SPECIAL_BLOCK_INDEX_FILES)
    return all((index_dir / name).exists() for name in required)


def missing_index_datasets(index_cfg: dict[str, Any], datasets: list[str], *, special_blocks: bool = False) -> list[str]:
    if not bool(index_cfg.get("skip_existing_index_builds", True)):
        return list(datasets)
    index_name = effective_index_name(index_cfg)
    return [dataset for dataset in datasets if not index_files_exist(index_cfg, dataset, index_name, special_blocks=special_blocks)]


def apply_method_search(cfg: dict[str, Any], method: str, data: dict[str, Any]) -> dict[str, Any]:
    search_override = method_search_cfg(cfg, method)
    if not search_override:
        return data
    merged = copy.deepcopy(data)
    explicit_values = expand_sweeps(search_override)
    clean_override = {key: value for key, value in search_override.items() if key not in {"sweeps", "values"}}
    if clean_override:
        merged["search"] = deep_merge(merged.get("search", {}), clean_override)
    if explicit_values is not None:
        if method == "favor":
            merged["ef_values"] = explicit_values
        else:
            merged.setdefault("search", {})["lsearch_values"] = explicit_values
            merged["search"]["lsearch_start"] = explicit_values[0]
            merged["search"]["lsearch_end"] = explicit_values[-1]
            merged["search"]["lsearch_step"] = explicit_values[1] - explicit_values[0] if len(explicit_values) > 1 else explicit_values[0]
    return merged




def python_step(name: str, script: Path, config_path: Path, config: dict[str, Any]) -> RunStep:
    return RunStep(name=name, command=[sys.executable, str(script), str(config_path)], config=config, config_path=config_path)


def shell_step(name: str, script: Path, config_path: Path, config: dict[str, Any]) -> RunStep:
    return RunStep(name=name, command=[str(script), str(config_path)], config=config, config_path=config_path)


def plan_steps(cfg: dict[str, Any], config_path: Path) -> list[RunStep]:
    selected = method_names(cfg)
    run_id = str(cfg.get("run_id") or time.strftime("%Y%m%d_%H%M%S"))
    datasets_for_python = dataset_objects(cfg)
    names_for_shell = dataset_names(cfg)
    common = common_overrides(cfg)
    steps: list[RunStep] = []
    selected_cpu_els = [name for name in selected if name in CPU_ELS_METHOD_NAMES]
    ung_search_index_name = "UNG__hybrid"
    special_search_index_name = None

    if "UNG__hybrid" in selected:
        ung_cfg = deep_merge(template_config("cpu_special_blocks/config_ung.json"), common)
        ung_cfg = apply_global_hyperparameters(cfg, ung_cfg, "UNG__hybrid")
        ung_cfg["run_name"] = "UNG__hybrid"
        ung_cfg["index_name"] = "UNG__hybrid"
        ung_cfg = apply_named_override(cfg, "UNG__hybrid", ung_cfg)
        ung_search_index_name = effective_index_name(ung_cfg)
        missing_datasets = missing_index_datasets(ung_cfg, names_for_shell, special_blocks=False)
        if missing_datasets:
            ung_cfg["datasets"] = missing_datasets
            steps.append(
                shell_step(
                    "UNG__hybrid",
                    REPO_ROOT / "run_cpu_special_blocks_experiment.sh",
                    generated_path(cfg, run_id, "ung_hybrid_index.json"),
                    ung_cfg,
                )
            )

    if "cpu_bruteforce_els_special_blocks" in selected_cpu_els:
        special_cfg = deep_merge(template_config("cpu_special_blocks/config.json"), common)
        special_cfg = apply_global_hyperparameters(cfg, special_cfg, "cpu_special_blocks_index")
        special_cfg["run_name"] = "cpu_special_blocks"
        special_cfg = apply_named_override(cfg, "cpu_special_blocks_index", special_cfg)
        special_search_index_name = effective_index_name(special_cfg)
        if cfg.get("ensure_special_blocks_index", True):
            missing_datasets = missing_index_datasets(special_cfg, names_for_shell, special_blocks=True)
            if missing_datasets:
                special_cfg["datasets"] = missing_datasets
                steps.append(
                    shell_step(
                        "cpu_special_blocks_index",
                        REPO_ROOT / "run_cpu_special_blocks_experiment.sh",
                        generated_path(cfg, run_id, "cpu_special_blocks_index.json"),
                        special_cfg,
                    )
                )

    if selected_cpu_els:
        search_cfg = deep_merge(template_config("search_comparison/config.json"), common)
        search_cfg = apply_global_hyperparameters(cfg, search_cfg, "search_comparison")
        search_cfg = apply_method_search(cfg, "search_comparison", search_cfg)
        search_cfg["datasets"] = datasets_for_python
        method_by_name = {method.get("name"): method for method in search_cfg.get("methods", [])}
        missing = [name for name in selected_cpu_els if name not in method_by_name]
        if missing:
            raise ValueError(f"search_comparison config is missing methods: {missing}")
        search_cfg["methods"] = [copy.deepcopy(method_by_name[name]) for name in selected_cpu_els]
        for method in search_cfg["methods"]:
            if method.get("name") == "cpu_bruteforce_els_ung":
                method["index_name"] = ung_search_index_name
            elif method.get("name") == "cpu_bruteforce_els_special_blocks" and special_search_index_name:
                method["index_name"] = special_search_index_name
        search_cfg = apply_named_override(cfg, "search_comparison", search_cfg)
        steps.append(
            python_step(
                "search_comparison_cpu_els",
                EXPERIMENTS_DIR / "search_comparison" / "run_search_comparison.py",
                generated_path(cfg, run_id, "search_comparison_cpu_els.json"),
                search_cfg,
            )
        )

    if "favor" in selected:
        favor_cfg = deep_merge(template_config("favor/config.json"), common)
        favor_cfg = apply_global_hyperparameters(cfg, favor_cfg, "favor")
        favor_cfg = apply_method_search(cfg, "favor", favor_cfg)
        favor_cfg["datasets"] = datasets_for_python
        favor_cfg = apply_named_override(cfg, "favor", favor_cfg)
        steps.append(
            python_step(
                "favor",
                EXPERIMENTS_DIR / "favor" / "run_favor_experiment.py",
                generated_path(cfg, run_id, "favor.json"),
                favor_cfg,
            )
        )

    if "Navix" in selected:
        navix_cfg = deep_merge(template_config("navix_baseline/config.json"), common)
        navix_cfg = apply_global_hyperparameters(cfg, navix_cfg, "Navix")
        navix_cfg = apply_method_search(cfg, "Navix", navix_cfg)
        navix_cfg["datasets"] = datasets_for_python
        navix_cfg = apply_named_override(cfg, "Navix", navix_cfg)
        steps.append(
            python_step(
                "Navix",
                EXPERIMENTS_DIR / "navix_baseline" / "run_navix_baseline.py",
                generated_path(cfg, run_id, "navix.json"),
                navix_cfg,
            )
        )

    if "curator" in selected:
        curator_cfg = deep_merge(template_config("curator_baseline/config.json"), common)
        curator_cfg = apply_global_hyperparameters(cfg, curator_cfg, "curator")
        curator_cfg = apply_method_search(cfg, "curator", curator_cfg)
        curator_cfg["datasets"] = datasets_for_python
        curator_cfg = apply_named_override(cfg, "curator", curator_cfg)
        steps.append(
            python_step(
                "curator",
                EXPERIMENTS_DIR / "curator_baseline" / "run_curator_baseline.py",
                generated_path(cfg, run_id, "curator.json"),
                curator_cfg,
            )
        )

    unknown = sorted(set(selected) - set(DEFAULT_METHODS))
    if unknown:
        raise ValueError(f"unsupported methods: {unknown}")
    return steps

def write_step_configs(steps: list[RunStep]) -> None:
    for step in steps:
        write_json(step.config_path, step.config)


def run_step(step: RunStep, dry_run: bool) -> int:
    command_text = " ".join(step.command)
    log(f"[{step.name}] config: {step.config_path}")
    log(f"[{step.name}] command: {command_text}")
    if dry_run:
        return 0
    return subprocess.run(step.command, cwd=REPO_ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a batch of configured experiments across datasets.")
    parser.add_argument("config", type=Path, help="batch JSON config")
    parser.add_argument("--dry-run", action="store_true", help="write generated configs and print commands only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    steps = plan_steps(cfg, args.config)
    write_step_configs(steps)
    log(f"batch config: {args.config}")
    log(f"steps: {', '.join(step.name for step in steps)}")

    overall = 0
    for step in steps:
        returncode = run_step(step, args.dry_run)
        if returncode != 0:
            overall = returncode
            log(f"[{step.name}] failed with exit={returncode}")
            if not cfg.get("continue_on_error", False):
                return overall
    log("batch done" if overall == 0 else f"batch finished with failures, exit={overall}")
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
