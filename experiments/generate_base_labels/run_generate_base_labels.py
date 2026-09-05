#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)


def require_int(cfg: dict, key: str) -> int:
    if key not in cfg:
        raise KeyError(f"missing required config key: {key}")
    return int(cfg[key])


def iter_job_configs(cfg: dict) -> list[dict]:
    datasets = cfg.get("datasets")
    if datasets is None:
        return [cfg]
    if not isinstance(datasets, list):
        raise TypeError("config key 'datasets' must be a list")

    defaults = {key: value for key, value in cfg.items() if key != "datasets"}
    return [{**defaults, **dataset_cfg} for dataset_cfg in datasets]


def build_command(cfg: dict) -> list[str]:
    build_dir = Path(cfg["build_dir"])
    tool = build_dir / "tools" / "generate_base_labels"
    if not tool.exists():
        raise FileNotFoundError(
            f"missing generate_base_labels executable: {tool}\n"
            "Build UNG first, or update build_dir in the config."
        )

    output_file = Path(cfg["output_file"])
    distribution_type = str(cfg.get("distribution_type", "zipf"))
    expected_num_label = int(cfg.get("expected_num_label", 3))
    max_num_label = int(cfg.get("max_num_label", 12))

    return [
        str(tool),
        "--output_file",
        str(output_file),
        "--num_points",
        str(require_int(cfg, "num_points")),
        "--num_labels",
        str(require_int(cfg, "num_labels")),
        "--distribution_type",
        distribution_type,
        "--expected_num_label",
        str(expected_num_label),
        "--max_num_label",
        str(max_num_label),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate base vector label sets with tools/generate_base_labels."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help=f"Path to config JSON. Defaults to {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated command without running it.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    cfg = load_config(config_path)
    for job_cfg in iter_job_configs(cfg):
        output_file = Path(job_cfg["output_file"])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_command(job_cfg)

        dataset = job_cfg.get("dataset")
        prefix = f"[{dataset}] " if dataset else ""
        print(f"{prefix}COMMAND:", " ".join(cmd), flush=True)
        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)
        print(f"{prefix}Labels written to {output_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
