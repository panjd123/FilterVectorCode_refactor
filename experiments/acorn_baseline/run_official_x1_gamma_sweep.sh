#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/sunyahui/worktrees/FilterVectorCode_multilevel_special}
DATA_ROOT=${DATA_ROOT:-/home/graphdb/FilterVectorData/Amazon}
RESULT_ROOT=${RESULT_ROOT:-/home/graphdb/FilterVectorResult/Amazon}
BINARY=${BINARY:-$ROOT/thirdparty/acorn-official/build_local/demos/amazon_x1_acorn}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/runs/external_baselines/acorn/Amazon/gamma_sweep}
INDEX_ROOT=${INDEX_ROOT:-$ROOT/runs/external_baselines/acorn/Amazon/index}
BUILD_THREADS=${BUILD_THREADS:-60}
SEARCH_THREADS=${SEARCH_THREADS:-100}
MAX_QUERIES=${MAX_QUERIES:-1000}
WARMUPS=${WARMUPS:-1}
REPEATS=${REPEATS:-3}

if [[ ! -x "$BINARY" ]]; then
  echo "missing ACORN benchmark binary: $BINARY" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$INDEX_ROOT"

variants=(
  "2:32:64"
  "4:32:128"
  "8:32:256"
)
workloads=(
  "query_minlen2_avgsel25pct"
  "query_minlen1_avgsel50pct"
  "query_minlen1_avgsel75pct"
)
ef_values=(1024 4096 16384)

for variant in "${variants[@]}"; do
  IFS=: read -r gamma m_beta ef_construction <<<"$variant"
  variant_name="ACORN_official_M32_g${gamma}_mb${m_beta}_efc${ef_construction}"
  variant_dir="$INDEX_ROOT/$variant_name"
  index_path="$variant_dir/acorn.index"
  build_log="$variant_dir/build.log"
  mkdir -p "$variant_dir"

  if [[ ! -s "$index_path" ]]; then
    tmp_index="$index_path.tmp.$$"
    tmp_log="$build_log.tmp.$$"
    rm -f "$tmp_index" "$tmp_log"
    /usr/bin/time -v "$BINARY" \
      --mode build \
      --base-bin "$DATA_ROOT/Amazon_base.bin" \
      --index "$tmp_index" \
      --M 32 \
      --gamma "$gamma" \
      --M-beta "$m_beta" \
      --ef-construction "$ef_construction" \
      --threads "$BUILD_THREADS" \
      >"$tmp_log" 2>&1
    mv "$tmp_index" "$index_path"
    mv "$tmp_log" "$build_log"
  fi

  if [[ ! -s "$index_path" ]]; then
    echo "index build did not produce a non-empty file: $index_path" >&2
    exit 1
  fi

  for workload in "${workloads[@]}"; do
    query_dir="$DATA_ROOT/$workload"
    gt="$RESULT_ROOT/GroundTruth/$workload/Amazon_gt_labels_containment.bin"
    result_dir="$OUTPUT_ROOT/gamma${gamma}/$workload"
    mkdir -p "$result_dir"

    for ef_search in "${ef_values[@]}"; do
      csv="$result_dir/ef${ef_search}.csv"
      stderr="$result_dir/ef${ef_search}.stderr"
      if [[ -s "$csv" ]] && awk -F, 'NR > 1 && $NF != 0 { bad=1 } END { exit bad }' "$csv"; then
        continue
      fi
      tmp_csv="$csv.tmp.$$"
      tmp_stderr="$stderr.tmp.$$"
      rm -f "$tmp_csv" "$tmp_stderr"
      "$BINARY" \
        --mode search \
        --index "$index_path" \
        --base-bin "$DATA_ROOT/Amazon_base.bin" \
        --base-labels "$DATA_ROOT/Amazon_base_labels.txt" \
        --query-bin "$query_dir/Amazon_query.bin" \
        --query-labels "$query_dir/Amazon_query_labels.txt" \
        --groundtruth "$gt" \
        --max-queries "$MAX_QUERIES" \
        --ef-search "$ef_search" \
        --threads "$SEARCH_THREADS" \
        --batch-size "$MAX_QUERIES" \
        --warmups "$WARMUPS" \
        --repeats "$REPEATS" \
        >"$tmp_csv" 2>"$tmp_stderr"
      awk -F, 'NR > 1 && $NF != 0 { bad=1 } END { exit bad }' "$tmp_csv"
      mv "$tmp_csv" "$csv"
      mv "$tmp_stderr" "$stderr"
    done
  done
done
