#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

out="$(make_out_dir existing_summary)"
python3 "$repo_root/scripts/benchmarks/summarize_existing_results.py" > "$out/existing_results_summary.csv"
cp "$repo_root/docs/reports/CROSS_GROUP_CELEBA_RESULTS_CN.md" "$out/" 2>/dev/null || true
echo "[OK] wrote $out/existing_results_summary.csv"
