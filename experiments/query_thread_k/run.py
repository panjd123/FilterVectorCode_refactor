#!/usr/bin/env python3
"""Serial runner for threads10 (K=10) and k20 (100 threads) experiments.

Use --groups threads10 or --groups k20 to run one experiment group.
--dry-run and --check-only never compute GroundTruth or launch algorithms.
K=20 GroundTruth is computed once per dataset before any selected search.
--prepare-gt-only prepares GT without running algorithms.
PYTHON_BIN selects Python; CURATOR_PYTHON_BIN selects Curator's environment.
Logs: <project>/logs/query_thread_k/<Beijing-time timestamp>/.
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
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from generate_configs import BUNDLE, PROJECT, DATASETS, GROUPS, METHODS, TASK, transform


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--prepare-gt-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--log-dir", type=Path)
    return parser.parse_args()


def launch(job):
    env = dict(os.environ)
    config = BUNDLE / job["config"]
    runner = PROJECT / job["runner"]
    check(runner.is_file(), f"Missing runner: {runner}")
    if job["method"] == "curator":
        python = Path(env.get("CURATOR_PYTHON_BIN", PROJECT / ".venv_curator/bin/python"))
        check(python.is_file() and os.access(python, os.X_OK), f"Missing Curator Python: {python}")
        env["PYTHON_BIN"] = str(python)
        command = ["bash", str(runner), str(config)]
        display = f"PYTHON_BIN={shlex.quote(str(python))} " + shlex.join(command)
    else:
        command = [sys.executable, str(runner), str(config)]
        display = shlex.join(command)
    return command, env, display


def query_count(cfg):
    dataset = cfg["datasets"][0]["dataset"]
    data = Path(cfg["data_root"]) / dataset
    folder = data / TASK
    binary = (folder / f"{dataset}_query.bin").read_bytes()
    n, dim = struct.unpack_from("<II", binary)
    check(n > 0 and dim > 0 and len(binary) == 8 + n * dim * 4, f"Invalid query bin: {folder}")
    fvecs = (folder / f"{dataset}_query.fvecs").read_bytes()
    labels = (folder / f"{dataset}_query_labels.txt").read_text().splitlines()
    with (folder / "profiled_minlen1_sel10to40pct.csv").open(newline="") as handle:
        profile = list(csv.DictReader(handle))
    check(len(fvecs) == n * (dim + 1) * 4 and len(labels) == len(profile) == n, f"Query count mismatch: {folder}")
    for i, (line, row) in enumerate(zip(labels, profile)):
        check(tuple(map(int, line.split(','))) == tuple(map(int, row['labels'].split())), f"Profile mismatch: {folder}, {i}")
        record = fvecs[i * (dim + 1) * 4:(i + 1) * (dim + 1) * 4]
        check(struct.unpack_from('<I', record)[0] == dim and record[4:] == binary[8 + i * dim * 4:8 + (i + 1) * dim * 4], f"Vector mismatch: {folder}, {i}")
    for suffix in ("base.bin", "base.fvecs", "base_labels.txt"):
        check((data / f"{dataset}_{suffix}").is_file(), f"Missing {dataset}_{suffix}")
    return n


def gt_command(cfg, destination):
    row = cfg["datasets"][0]
    dataset = row["dataset"]
    base = Path(cfg["data_root"]) / dataset
    binary = PROJECT / "build_ung_multilevel_trie_gpu/tools/compute_groundtruth"
    check(binary.is_file() and os.access(binary, os.X_OK), f"Missing GT tool: {binary}")
    return [str(binary), "--data_type", "float", "--dist_fn", "L2", "--scenario", "containment",
            "--K", str(cfg["search"]["K"]), "--num_threads", str(cfg["search"]["num_threads"]),
            "--base_bin_file", str(base / f"{dataset}_base.bin"),
            "--base_label_file", str(base / f"{dataset}_base_labels.txt"),
            "--query_bin_file", str(base / TASK / f"{dataset}_query.bin"),
            "--query_label_file", str(base / TASK / f"{dataset}_query_labels.txt"),
            "--gt_file", str(destination)]


def preflight(jobs):
    inputs, gts = {}, {}
    for job in jobs:
        source = PROJECT / job["source_config"]
        check(hashlib.sha256(source.read_bytes()).hexdigest() == job["source_sha256"], f"Template changed; regenerate configs: {source}")
        cfg = json.loads((BUNDLE / job["config"]).read_text())
        check(cfg == transform(json.loads(source.read_text()), job["dataset"], job["method"], job["group"]), f"Unexpected configuration change: {job['config']}")
        key = (cfg["data_root"], job["dataset"])
        if key not in inputs:
            inputs[key] = query_count(cfg)
        gt = Path(cfg["datasets"][0]["groundtruth_file"])
        expected_bytes = inputs[key] * cfg["search"]["K"] * 8
        if gt.exists():
            check(gt.stat().st_size == expected_bytes, f"GT does not match query count/K: {gt}")
        else:
            gt_command(cfg, gt)
        gts[gt] = (cfg, expected_bytes)
        launch(job)
    print(f"Validated {len(jobs)} configurations; {len(inputs)} datasets; "
          f"{sum(not p.exists() for p in gts)} GroundTruth files to prepare.", flush=True)
    return gts


def execute(command, environment, log_file, display=None):
    display = display or shlex.join(command)
    print(f"\n{display}\nLog: {log_file}", flush=True)
    with log_file.open('w', encoding='utf-8') as handle:
        handle.write(display + '\n\n')
        handle.flush()
        with subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                handle.write(line)
                handle.flush()
            return process.wait()


def prepare_gt(gts, log_dir):
    for gt, (cfg, expected_bytes) in gts.items():
        if gt.exists():
            check(gt.stat().st_size == expected_bytes, f"GT changed: {gt}")
            continue
        gt.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.prepare_gt_', dir=gt.parent) as temporary:
            output = Path(temporary) / gt.name
            dataset = cfg['datasets'][0]['dataset']
            code = execute(gt_command(cfg, output), dict(os.environ),
                           log_dir / f"groundtruth_{dataset}_k{cfg['search']['K']}.log")
            check(code == 0, f"GroundTruth preparation failed: {gt}")
            check(output.is_file() and output.stat().st_size == expected_bytes, f"Incomplete GT output: {output}")
            check(not gt.exists(), f"GroundTruth appeared during preparation: {gt}")
            output.rename(gt)


def campaign(args, log_dir=None):
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    jobs = [j for j in manifest['jobs'] if j['group'] in args.groups and j['dataset'] in args.datasets and j['method'] in args.methods]
    expected = {(g, d, m) for g in args.groups for d in args.datasets for m in args.methods}
    check(len(jobs) == len(expected) and {(j['group'], j['dataset'], j['method']) for j in jobs} == expected, 'Missing or duplicate jobs')
    gts = preflight(jobs)
    if args.check_only:
        return 0
    if args.dry_run:
        for gt, (cfg, _) in gts.items():
            if not gt.exists():
                print('[prepare GT] ' + shlex.join(gt_command(cfg, gt)))
        for job in jobs:
            print(f"[{job['group']}/{job['dataset']}/{job['method']}] {launch(job)[2]}")
        return 0
    prepare_gt(gts, log_dir)
    if args.prepare_gt_only:
        print('GroundTruth preparation complete; no algorithms launched.')
        return 0
    results = []
    for job in jobs:
        command, env, display = launch(job)
        log = log_dir / f"{job['group']}_{job['dataset']}_{job['method']}.log"
        code = execute(command, env, log, display)
        results.append(dict(group=job['group'], dataset=job['dataset'], method=job['method'], returncode=code, log=str(log)))
        (log_dir / 'status.json').write_text(json.dumps(results, indent=2) + '\n')
        if code and not args.continue_on_error:
            break
    failures = sum(r['returncode'] != 0 for r in results)
    print(f"Finished {len(results)}/{len(jobs)} jobs; {failures} failed. Logs: {log_dir}")
    return int(failures > 0)


def main():
    args = arguments()
    if args.check_only or args.dry_run:
        return campaign(args)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    log_dir = (args.log_dir or PROJECT / 'logs/query_thread_k' / stamp).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / 'run.log').open('w', encoding='utf-8') as handle:
        with redirect_stdout(Tee(sys.stdout, handle)), redirect_stderr(Tee(sys.stderr, handle)):
            print(f"Campaign log: {log_dir / 'run.log'}", flush=True)
            try:
                return campaign(args, log_dir)
            except (OSError, ValueError, KeyError, struct.error) as error:
                print(f'ERROR: {error}', file=sys.stderr, flush=True)
                return 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, struct.error) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(2)
