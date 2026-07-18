#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def run(cmd):
    print("+ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def bool_arg(value):
    return "true" if bool(value) else "false"


def main():
    parser = argparse.ArgumentParser(
        description="Run JSON-configured query generation tasks and convert fvecs to bin."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "build_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "build_ung_rel",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    tool = args.build_dir / "tools" / "generate_mixed_queries"
    fvecs_to_bin = args.build_dir / "tools" / "fvecs_to_bin"

    if not args.config.is_file():
        raise SystemExit(f"config file not found: {args.config}")
    if not tool.is_file():
        raise SystemExit(f"generate_mixed_queries not found: {tool}")
    if not fvecs_to_bin.is_file():
        raise SystemExit(f"fvecs_to_bin not found: {fvecs_to_bin}")

    config = json.loads(args.config.read_text())
    tasks = config.get("query_tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit("config must contain a non-empty query_tasks array")

    for task in tasks:
        if not task.get("enabled", False):
            continue

        task_name = task["task_name"]
        mode = task.get("mode", "generate")
        dataset = task["dataset"]
        data_dir = Path(task["data_dir"])
        overwrite = bool(task.get("overwrite", False))

        query_dir = data_dir / f"query_{task_name}"
        query_labels = query_dir / f"{dataset}_query_labels.txt"
        query_fvecs = query_dir / f"{dataset}_query.fvecs"
        query_bin = query_dir / f"{dataset}_query.bin"
        base_labels = data_dir / f"{dataset}_base_labels.txt"
        base_fvecs = data_dir / f"{dataset}_base.fvecs"

        query_dir.mkdir(parents=True, exist_ok=True)

        if query_fvecs.exists() and not overwrite:
            print(f"skip existing query fvecs: {query_fvecs}")
        else:
            cmd = [
                tool,
                "--input_file",
                base_labels,
                "--output_file",
                query_labels,
                "--base_vectors_file",
                base_fvecs,
                "--output_vectors_file",
                query_fvecs,
                "--mode",
                mode,
            ]

            if mode in ("sub_base", "weighted_sub_base", "variable_sub_base"):
                params = task.get("sub_base_params", {})
                cmd += [
                    "--num_points",
                    params.get("num_points", 1000),
                    "--query-length",
                    params.get("query_length", params.get("min_query_length", 3)),
                    "--min-query-length",
                    params.get("min_query_length", params.get("query_length", 3)),
                    "--max-query-length",
                    params.get("max_query_length", 0),
                    "--K",
                    params.get("K", 10),
                    "--max-coverage",
                    params.get("max_coverage", 2_000_000_000),
                    "--min-children",
                    params.get("min_children", 0 if mode == "variable_sub_base" else 1),
                ]
                cache_file = params.get("cache-file")
                if cache_file:
                    cmd += ["--cache-file", cache_file]
            elif mode == "generate":
                params = task.get("generation_params", {})
                cmd += [
                    "--num_points",
                    params.get("num_points", 1000),
                    "--K",
                    params.get("K", 10),
                    "--distribution_type",
                    params.get("distribution_type", "uniform"),
                    "--truncate_to_fixed_length",
                    bool_arg(params.get("truncate_to_fixed_length", True)),
                    "--num_labels_per_query",
                    params.get("num_labels_per_query", 3),
                    "--expected_num_label",
                    params.get("expected_num_label", 3),
                ]
            else:
                raise SystemExit(f"unknown mode for task {task_name}: {mode}")

            run(cmd)

        if not query_bin.exists() or overwrite:
            run(
                [
                    fvecs_to_bin,
                    "--data_type",
                    "float",
                    "--input_file",
                    query_fvecs,
                    "--output_file",
                    query_bin,
                ]
            )

        if task.get("analysis_params", {}).get("analyze", False):
            profile = query_dir / f"profiled_{task_name}.csv"
            run(
                [
                    tool,
                    "--mode",
                    "analyze",
                    "--input_file",
                    base_labels,
                    "--candidate_file",
                    query_labels,
                    "--profiled_output",
                    profile,
                ]
            )

        print("Query task ready:")
        print(f"  name:   query_{task_name}")
        print(f"  labels: {query_labels}")
        print(f"  fvecs:  {query_fvecs}")
        print(f"  bin:    {query_bin}")


if __name__ == "__main__":
    main()
