#!/usr/bin/env python3
"""Run one command while sampling its process-tree CPU and GPU memory."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import IO, Any, Optional


def _read_kib_field(pid: int, field: str) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        pass
    return 0


def _descendants(root_pid: int) -> set[int]:
    found = {root_pid}
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        try:
            text = Path(f"/proc/{pid}/task/{pid}/children").read_text()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for token in text.split():
            try:
                child = int(token)
            except ValueError:
                continue
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def _gpu_memory_mib(pids: set[int]) -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=5, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    total = 0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            pid, used = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if pid in pids:
            total += used
    return total


def run_profiled(command: list[str], env: Optional[dict[str, str]],
                 log: IO[str], interval_seconds: float = 0.2) -> dict[str, Any]:
    """Return process wall time and sampled process-tree resource peaks."""
    start = time.monotonic()
    process = subprocess.Popen(
        command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    samples = 0
    peak_rss_kib = 0
    peak_hwm_kib = 0
    peak_gpu_mib = 0
    while process.poll() is None:
        pids = _descendants(process.pid)
        peak_rss_kib = max(
            peak_rss_kib, sum(_read_kib_field(pid, "VmRSS") for pid in pids))
        peak_hwm_kib = max(
            peak_hwm_kib, sum(_read_kib_field(pid, "VmHWM") for pid in pids))
        peak_gpu_mib = max(peak_gpu_mib, _gpu_memory_mib(pids))
        samples += 1
        time.sleep(interval_seconds)
    return {
        "returncode": process.wait(),
        "elapsed_seconds": time.monotonic() - start,
        "sample_interval_seconds": interval_seconds,
        "num_samples": samples,
        "peak_process_tree_rss_kib": peak_rss_kib,
        "peak_process_tree_hwm_kib": peak_hwm_kib,
        "peak_gpu_memory_mib": peak_gpu_mib,
        "gpu_memory_source": "nvidia-smi compute-app rows for sampled process tree",
        "rss_source": "sum of Linux /proc/<pid>/status values over sampled process tree",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=0.2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    with args.output.with_suffix(".log").open("w") as log:
        result = run_profiled(command, None, log, args.interval_seconds)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return int(result["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
