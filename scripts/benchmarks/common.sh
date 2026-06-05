#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
default_results_root="${RESULTS_ROOT:-/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks}"

timestamp() {
  date +%Y%m%d_%H%M%S
}

make_out_dir() {
  local name="$1"
  local out="${OUTDIR:-${default_results_root}/${name}_$(timestamp)}"
  mkdir -p "$out"
  printf '%s\n' "$out"
}

require_gpu_idle() {
  local gpu_id="${GPU_ID:-0}"
  local max_mem_mb="${GPU_IDLE_MAX_MEMORY_MB:-1024}"
  local gpu_line util mem apps

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] nvidia-smi not found; cannot verify GPU idleness." >&2
    return 0
  fi

  gpu_line="$(nvidia-smi --id="$gpu_id" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F, 'NR==1{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1 "," $2}')"
  util="$(printf '%s' "$gpu_line" | awk -F, '{gsub(/ /,"",$1); print $1}')"
  mem="$(printf '%s' "$gpu_line" | awk -F, '{gsub(/ /,"",$2); print $2}')"
  apps="$(nvidia-smi --id="$gpu_id" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"

  if [[ -n "${apps//[[:space:]]/}" ]]; then
    echo "[ERROR] GPU $gpu_id has active compute apps:" >&2
    printf '%s\n' "$apps" >&2
    echo "        Re-run with WAIT_GPU_IDLE=0 to ignore this guard." >&2
    return 3
  fi
  if [[ "${util:-100}" != "0" ]]; then
    echo "[ERROR] GPU $gpu_id is busy (util=${util}%). Re-run with WAIT_GPU_IDLE=0 to ignore this guard." >&2
    return 3
  fi
  if [[ "${mem:-999999}" -gt "$max_mem_mb" ]]; then
    echo "[ERROR] GPU $gpu_id memory is not idle (used=${mem} MiB > ${max_mem_mb} MiB)." >&2
    echo "        Set GPU_IDLE_MAX_MEMORY_MB to adjust the threshold, or WAIT_GPU_IDLE=0 to ignore." >&2
    return 3
  fi
}

run_with_gpu_lock() {
  local mode="$1"
  shift
  local gpu_id="${GPU_ID:-0}"
  if command -v gpulock >/dev/null 2>&1; then
    if [[ "$mode" == "perf" ]]; then
      gpulock perf --wait-gpu-idle "$gpu_id" -- "$@"
    else
      gpulock check "$gpu_id" -- "$@"
    fi
  else
    echo "[WARN] gpulock not found; running directly: $*" >&2
    "$@"
  fi
}

log_cmd() {
  local log="$1"
  shift
  {
    echo "# cwd: $(pwd)"
    echo "# date: $(date -Is)"
    echo "# cmd: $*"
  } > "$log"
  "$@" >> "$log" 2>&1
}
