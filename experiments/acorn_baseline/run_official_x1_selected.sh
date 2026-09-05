#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/sunyahui/worktrees/FilterVectorCode_multilevel_special}
DATA_ROOT=${DATA_ROOT:-/home/graphdb/FilterVectorData/Amazon}
RESULT_ROOT=${RESULT_ROOT:-/home/graphdb/FilterVectorResult/Amazon}
BINARY=$ROOT/thirdparty/acorn-official/build_local/demos/amazon_x1_acorn
INDEX=$ROOT/runs/external_baselines/acorn/Amazon/index/ACORN1_official_M32_g1_mb64/acorn.index
OUTPUT=$ROOT/runs/external_baselines/acorn/Amazon/acorn1_selected_formal

cases=(
  "query_minlen2_avgsel25pct:16384"
  "query_minlen1_avgsel50pct:8192"
  "query_minlen1_avgsel75pct:4096"
)

mkdir -p "$OUTPUT"
for item in "${cases[@]}"; do
  IFS=: read -r workload ef <<<"$item"
  out="$OUTPUT/${workload}_ef${ef}.csv"
  err="$OUTPUT/${workload}_ef${ef}.stderr"
  "$BINARY" \
    --mode search --index "$INDEX" \
    --base-bin "$DATA_ROOT/Amazon_base.bin" \
    --base-labels "$DATA_ROOT/Amazon_base_labels.txt" \
    --query-bin "$DATA_ROOT/$workload/Amazon_query.bin" \
    --query-labels "$DATA_ROOT/$workload/Amazon_query_labels.txt" \
    --groundtruth "$RESULT_ROOT/GroundTruth/$workload/Amazon_gt_labels_containment.bin" \
    --max-queries 1000 --ef-search "$ef" --threads 100 --batch-size 1000 \
    --warmups 1 --repeats 5 >"$out.tmp" 2>"$err.tmp"
  awk -F, 'NR > 1 && $NF != 0 { bad=1 } END { exit bad }' "$out.tmp"
  mv "$out.tmp" "$out"
  mv "$err.tmp" "$err"
done
