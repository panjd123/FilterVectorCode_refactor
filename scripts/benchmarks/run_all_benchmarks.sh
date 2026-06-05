#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

root_out="$(make_out_dir all)"
echo "[INFO] all benchmark output root: $root_out"

OUTDIR="$root_out/existing_summary" "$repo_root/scripts/benchmarks/run_existing_summary.sh"

if [[ "${RUN_HEAVY_UNG:-0}" == "1" ]]; then
  OUTDIR="$root_out/ung_gpu" "$repo_root/scripts/benchmarks/run_ung_cross_edge_ab.sh" gpu
  OUTDIR="$root_out/ung_fused" "$repo_root/scripts/benchmarks/run_ung_cross_edge_ab.sh" fused
fi

OUTDIR="$root_out/nx256_exact" "$repo_root/scripts/benchmarks/run_nx256_exact_benchmark.sh"

if [[ "${RUN_TAGORE:-1}" == "1" ]]; then
  OUTDIR="$root_out/tagore_nx256" "$repo_root/scripts/benchmarks/run_tagore_nx256_benchmark.sh"
fi

echo "[OK] wrote $root_out"
