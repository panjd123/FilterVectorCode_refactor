#!/usr/bin/env python3
"""Fail-closed GPU coordination evidence for authoritative build runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_POLICY = {
    "device": 0,
    "idle_consecutive_samples": 3,
    "idle_sample_interval_seconds": 1.0,
    "idle_wait_timeout_seconds": 900.0,
    "max_idle_utilization_percent": 0,
    "max_idle_memory_mib": 16,
}


def profile_uses_gpu(case: dict[str, Any]) -> bool:
    profile = str(case.get("benchmark_profile", ""))
    return profile not in {"", "cpu", "original_cpu", "current_cpu"}


def policy_for(config: dict[str, Any]) -> dict[str, Any]:
    policy = dict(DEFAULT_POLICY)
    policy.update(config.get("gpu_isolation", {}))
    return policy


def _locked_devices() -> set[str]:
    raw = os.environ.get("GPULOCK_LOCKED_DEVICES", "")
    return {token.strip() for token in raw.replace(" ", ",").split(",")
            if token.strip()}


def has_perf_lock(device: int) -> bool:
    if os.environ.get("GPULOCK_LOCK_MODE") != "perf":
        return False
    devices = _locked_devices()
    return str(device) in devices


def _run_nvidia_smi(arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", *arguments], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=10, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"cannot inspect GPU state: {error}") from error
    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def snapshot(device: int) -> dict[str, Any]:
    gpu_fields = _run_nvidia_smi([
        f"--id={device}",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]).strip().splitlines()
    if len(gpu_fields) != 1:
        raise RuntimeError(
            f"expected one nvidia-smi row for GPU {device}, found {len(gpu_fields)}")
    values = [value.strip() for value in gpu_fields[0].split(",")]
    if len(values) != 5:
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {gpu_fields[0]}")
    applications = []
    app_text = _run_nvidia_smi([
        f"--id={device}",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    for line in app_text.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 3:
            continue
        applications.append({
            "pid": int(fields[0]),
            "process_name": fields[1],
            "used_memory_mib": int(fields[2]),
        })
    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": int(values[0]),
        "name": values[1],
        "utilization_percent": int(values[2]),
        "used_memory_mib": int(values[3]),
        "total_memory_mib": int(values[4]),
        "compute_applications": applications,
    }


def snapshot_is_idle(state: dict[str, Any], policy: dict[str, Any]) -> bool:
    return (
        not state.get("compute_applications")
        and int(state["utilization_percent"]) <=
        int(policy["max_idle_utilization_percent"])
        and int(state["used_memory_mib"]) <=
        int(policy["max_idle_memory_mib"])
    )


def prepare_case(config: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Return isolation evidence immediately before an untimed build child."""
    if not profile_uses_gpu(case):
        return {"mode": "not_required", "gpu_required": False}
    policy = policy_for(config)
    device = int(policy["device"])
    if has_perf_lock(device):
        return {
            "mode": "gpulock_perf",
            "gpu_required": True,
            "device": device,
            "lock_mode": os.environ.get("GPULOCK_LOCK_MODE"),
            "locked_devices": sorted(_locked_devices()),
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    if shutil.which("gpulock"):
        raise RuntimeError(
            "gpulock is installed but this GPU build is not inside a perf lock")

    required = int(policy["idle_consecutive_samples"])
    interval = float(policy["idle_sample_interval_seconds"])
    timeout = float(policy["idle_wait_timeout_seconds"])
    if required <= 0 or interval < 0 or timeout < 0:
        raise RuntimeError(f"invalid GPU isolation policy: {policy}")
    deadline = time.monotonic() + timeout
    accepted: list[dict[str, Any]] = []
    observations = 0
    while True:
        state = snapshot(device)
        observations += 1
        if snapshot_is_idle(state, policy):
            accepted.append(state)
            if len(accepted) >= required:
                return {
                    "mode": "idle_preflight_no_lock",
                    "gpu_required": True,
                    "device": device,
                    "policy": policy,
                    "observations": observations,
                    "accepted_snapshots": accepted,
                    "exclusive_lock": False,
                    "limitation": (
                        "gpulock unavailable; idle preflight cannot prevent a "
                        "process from starting after the final sample"),
                }
        else:
            accepted = []
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU {device} did not satisfy the idle policy before timeout; "
                f"last state={json.dumps(state, sort_keys=True)}")
        time.sleep(interval)


def validate_case_evidence(case: dict[str, Any], case_root: Path) -> None:
    if not profile_uses_gpu(case):
        return
    path = case_root / "gpu_isolation.json"
    if not path.is_file():
        raise ValueError(f"missing GPU isolation evidence: {path}")
    evidence = json.loads(path.read_text())
    if evidence.get("mode") not in {"gpulock_perf", "idle_preflight_no_lock"}:
        raise ValueError(f"invalid GPU isolation evidence: {path}")


def reexec_under_perf_lock(device: int) -> bool:
    """Re-exec the whole campaign under gpulock when the host provides it."""
    if has_perf_lock(device):
        return False
    executable = shutil.which("gpulock")
    if executable is None:
        return False
    os.execv(executable, [
        executable, "perf", str(device), "--", sys.executable,
        str(Path(sys.argv[0]).resolve()), *sys.argv[1:],
    ])
    return True
