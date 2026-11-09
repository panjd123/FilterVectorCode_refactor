#!/usr/bin/env python3
"""Run the length-group campaign serially using the existing algorithm runners.

Examples (from the project root):
  python3 experiments/query_label_length/run.py --dry-run
  python3 experiments/query_label_length/run.py --check-only
  python3 experiments/query_label_length/run.py
  python3 experiments/query_label_length/run.py --datasets Amazon --methods trie_block ung

Native runners retain their index-reuse and completed-search skipping behavior.
PYTHON_BIN selects regular runners; CURATOR_PYTHON_BIN selects Curator's venv.
Console logs and status are saved under <project>/logs/query_label_length/<timestamp>/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import struct
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from generate_configs import BUNDLE, DATASETS, METHODS, PROJECT, TASKS, transform


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--dry-run", action="store_true", help="validate and print commands; execute no algorithm")
    parser.add_argument("--check-only", action="store_true", help="validate configurations, query files, GT and launchers")
    parser.add_argument("--continue-on-error", action="store_true", help="try remaining jobs after a failure; still exit nonzero")
    parser.add_argument("--log-dir", type=Path,
                        help="log directory (default: <project>/logs/query_label_length/<timestamp>)")
    return parser.parse_args()


class Tee:
    """Write console messages to both the terminal and the campaign log."""

    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_query(cfg, dataset, task):
    """Check actual file contents without loading base vectors or running search."""
    folder = Path(cfg["data_root"]) / dataset / task
    binary = (folder / f"{dataset}_query.bin").read_bytes()
    count, dim = struct.unpack_from("<II", binary)
    check(count > 0 and dim > 0 and len(binary) == 8 + count * dim * 4, f"Invalid query bin: {folder}")
    fvecs = (folder / f"{dataset}_query.fvecs").read_bytes()
    check(len(fvecs) == count * (dim + 1) * 4, f"Invalid fvecs size: {folder}")
    labels = (folder / f"{dataset}_query_labels.txt").read_text().splitlines()
    check(len(labels) == count, f"Label count mismatch: {folder}")
    low, high = (1, 3) if task == TASKS[0] else (4, 6)
    profile_path = folder / f"profiled_len{low}to{high}_sel0to40pct.csv"
    with profile_path.open(newline="") as handle:
        profile = list(csv.DictReader(handle))
    check(len(profile) == count, f"Profile count mismatch: {folder}")
    for index, (line, row) in enumerate(zip(labels, profile)):
        ids = tuple(map(int, line.split(",")))
        check(low <= len(ids) <= high, f"Wrong label group: {folder}, row {index}")
        check(ids == tuple(map(int, row["labels"].split())), f"Profile labels mismatch: {folder}, row {index}")
        offset = index * (dim + 1) * 4
        record = fvecs[offset:offset + (dim + 1) * 4]
        check(struct.unpack_from("<I", record)[0] == dim, f"Wrong fvecs dimension: {folder}, row {index}")
        check(record[4:] == binary[8 + index * dim * 4:8 + (index + 1) * dim * 4],
              f"Bin/fvecs differ: {folder}, row {index}")
    gt = Path(cfg["result_root"]) / dataset / "GroundTruth" / task / f"{dataset}_gt_labels_containment.bin"
    check(gt.stat().st_size == count * int(cfg["search"]["K"]) * 8, f"GroundTruth size mismatch: {gt}")
    base = Path(cfg["data_root"]) / dataset
    for suffix in ("base.bin", "base.fvecs", "base_labels.txt"):
        path = base / f"{dataset}_{suffix}"
        check(path.is_file(), f"Missing base data: {path}")
    return count


def command_for(job):
    runner = PROJECT / job["runner"]
    config = BUNDLE / job["config"]
    environment = dict(os.environ)
    if job["method"] == "curator":
        python = Path(environment.get("CURATOR_PYTHON_BIN", PROJECT / ".venv_curator/bin/python"))
        check(python.is_file() and os.access(python, os.X_OK), f"Missing Curator Python: {python}")
        environment["PYTHON_BIN"] = str(python)
        command = ["bash", str(runner), str(config)]
        display = f"PYTHON_BIN={shlex.quote(str(python))} " + shlex.join(command)
    else:
        command = [sys.executable, str(runner), str(config)]
        display = shlex.join(command)
    check(runner.is_file(), f"Missing runner: {runner}")
    return command, environment, display


def validate_jobs(jobs):
    checked = {}
    for job in jobs:
        source = PROJECT / job["source_config"]
        check(hashlib.sha256(source.read_bytes()).hexdigest() == job["source_sha256"],
              f"Source template changed: {source}. Regenerate with generate_configs.py.")
        cfg = json.loads((BUNDLE / job["config"]).read_text())
        expected, _ = transform(json.loads(source.read_text()), job["dataset"])
        check(cfg == expected, f"Configuration differs from source beyond task replacement: {job['config']}")
        check(len(cfg["datasets"]) == job["workload_count"], f"Invalid workload count: {job['config']}")
        for row in cfg["datasets"]:
            key = (cfg["data_root"], cfg["result_root"], row["dataset"], row["query_task"])
            if key not in checked:
                checked[key] = validate_query(cfg, row["dataset"], row["query_task"])
        command_for(job)
    return checked


def run_job(job, log_file):
    command, environment, display = command_for(job)
    print(f"\n[{job['dataset']}/{job['method']}] {display}\nLog: {log_file}", flush=True)
    with log_file.open("w", encoding="utf-8") as handle:
        handle.write(display + "\n\n")
        handle.flush()
        with subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            return process.wait()


def run_campaign(args, log_dir=None):
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    jobs = [job for job in manifest["jobs"]
            if job["dataset"] in args.datasets and job["method"] in args.methods]
    check(jobs, "No selected jobs")
    expected_pairs = {(dataset, method) for dataset in args.datasets for method in args.methods}
    check({(job["dataset"], job["method"]) for job in jobs} == expected_pairs,
          "Manifest is missing selected dataset/method combinations")
    check(len(jobs) == len(expected_pairs), "Manifest contains duplicate jobs")
    counts = validate_jobs(jobs)
    workload_count = sum(job["workload_count"] for job in jobs)
    print(f"Validated {len(jobs)} configurations, {len(counts)} query groups, "
          f"{workload_count} task/parameter combinations.", flush=True)
    if args.check_only:
        return 0
    if args.dry_run:
        for job in jobs:
            print(f"[{job['dataset']}/{job['method']}] {command_for(job)[2]}")
        return 0
    results = []
    for job in jobs:
        log_file = log_dir / f"{job['dataset']}_{job['method']}.log"
        returncode = run_job(job, log_file)
        results.append(dict(dataset=job["dataset"], method=job["method"],
                            returncode=returncode, log=str(log_file)))
        (log_dir / "status.json").write_text(json.dumps(results, indent=2) + "\n")
        if returncode and not args.continue_on_error:
            break
    failed = [row for row in results if row["returncode"] != 0]
    print(f"\nFinished {len(results)}/{len(jobs)} jobs; {len(failed)} failed. Logs: {log_dir}")
    return 1 if failed else 0


def main():
    args = parse_args()
    if args.check_only or args.dry_run:
        return run_campaign(args)
    # Beijing time is UTC+8; a fixed offset also works on hosts without tzdata.
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    log_dir = (args.log_dir or PROJECT / "logs" / "query_label_length" / stamp).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "run.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print(f"Campaign log: {log_dir / 'run.log'}", flush=True)
            try:
                return run_campaign(args, log_dir)
            except (OSError, ValueError, KeyError, struct.error) as error:
                print(f"ERROR: {error}", file=sys.stderr, flush=True)
                return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, struct.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
