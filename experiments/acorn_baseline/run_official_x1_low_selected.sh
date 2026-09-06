#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/sunyahui/worktrees/FilterVectorCode_multilevel_special}
DATA_ROOT=${DATA_ROOT:-/home/graphdb/FilterVectorData/Amazon}
RESULT_ROOT=${RESULT_ROOT:-/home/graphdb/FilterVectorResult/Amazon}
BINARY=${BINARY:-$ROOT/thirdparty/acorn-official/build_local/demos/amazon_x1_acorn}
INDEX=${INDEX:-$ROOT/runs/external_baselines/acorn/Amazon/index/ACORN1_official_M32_g1_mb64/acorn.index}
OUTPUT=${OUTPUT:-$ROOT/runs/external_baselines/acorn_low_selectivity_formal/ACORN-1}

cases=(
  "query_minlen5_avgsel05pct:4096"
  "query_minlen5_avgsel1pct:8192"
  "query_minlen3_avgsel10pct:16384"
)

mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/.run.lock"
flock -n 9 || { echo "another selected ACORN run is active" >&2; exit 3; }

for item in "${cases[@]}"; do
  IFS=: read -r workload ef_search <<<"$item"
  out="$OUTPUT/${workload}_ef${ef_search}.csv"
  err="$OUTPUT/${workload}_ef${ef_search}.stderr"
  tmp_out="$out.tmp.$$"
  tmp_err="$err.tmp.$$"
  rm -f "$tmp_out" "$tmp_err"
  "$BINARY" \
    --mode search --index "$INDEX" \
    --base-bin "$DATA_ROOT/Amazon_base.bin" \
    --base-labels "$DATA_ROOT/Amazon_base_labels.txt" \
    --query-bin "$DATA_ROOT/$workload/Amazon_query.bin" \
    --query-labels "$DATA_ROOT/$workload/Amazon_query_labels.txt" \
    --groundtruth "$RESULT_ROOT/GroundTruth/$workload/Amazon_gt_labels_containment.bin" \
    --max-queries 1000 --ef-search "$ef_search" --threads 100 --batch-size 1000 \
    --warmups 1 --repeats 5 >"$tmp_out" 2>"$tmp_err"
  awk -F, 'NR == 1 { next } $2 == 0 { measured += 1 } $NF != 0 { bad = 1 } END { exit !(measured == 5 && !bad) }' "$tmp_out"
  mv "$tmp_out" "$out"
  mv "$tmp_err" "$err"
  printf '%s ef=%s formal complete\n' "$workload" "$ef_search"
done
