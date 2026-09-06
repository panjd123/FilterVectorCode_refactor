#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/sunyahui/worktrees/FilterVectorCode_multilevel_special}
DATA_ROOT=${DATA_ROOT:-/home/graphdb/FilterVectorData/Amazon}
RESULT_ROOT=${RESULT_ROOT:-/home/graphdb/FilterVectorResult/Amazon}
BINARY=${BINARY:-$ROOT/thirdparty/acorn-official/build_local/demos/amazon_x1_acorn}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/runs/external_baselines/acorn_low_selectivity_screen}
SEARCH_THREADS=${SEARCH_THREADS:-100}
MAX_QUERIES=${MAX_QUERIES:-1000}
WARMUPS=${WARMUPS:-1}
REPEATS=${REPEATS:-3}

if [[ ! -x "$BINARY" ]]; then
  echo "missing ACORN benchmark binary: $BINARY" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.run.lock"
if ! flock -n 9; then
  echo "another low-selectivity ACORN sweep owns $OUTPUT_ROOT" >&2
  exit 3
fi

variants=(
  "ACORN-1:$ROOT/runs/external_baselines/acorn/Amazon/index/ACORN1_official_M32_g1_mb64/acorn.index"
  "ACORN-gamma2:$ROOT/runs/external_baselines/acorn/Amazon/index/ACORN_official_M32_g2_mb32_efc64/acorn.index"
  "ACORN-gamma4:$ROOT/runs/external_baselines/acorn/Amazon/index/ACORN_official_M32_g4_mb32_efc128/acorn.index"
  "ACORN-gamma8:$ROOT/runs/external_baselines/acorn/Amazon/index/ACORN_official_M32_g8_mb32_efc256/acorn.index"
  "ACORN-gamma12:$ROOT/runs/external_baselines/acorn/Amazon/index/ACORN_official_M32_g12_mb32/acorn.index"
)
workloads=(
  "query_minlen5_avgsel05pct"
  "query_minlen5_avgsel1pct"
  "query_minlen3_avgsel10pct"
)
ef_values=(16 32 64 128 256 512 1024 2048 4096 8192 16384 32768)

for variant_spec in "${variants[@]}"; do
  IFS=: read -r variant index_path <<<"$variant_spec"
  if [[ ! -s "$index_path" ]]; then
    echo "missing ACORN index: $index_path" >&2
    exit 4
  fi
  for workload in "${workloads[@]}"; do
    query_dir="$DATA_ROOT/$workload"
    gt="$RESULT_ROOT/GroundTruth/$workload/Amazon_gt_labels_containment.bin"
    result_dir="$OUTPUT_ROOT/$variant/$workload"
    mkdir -p "$result_dir"
    for ef_search in "${ef_values[@]}"; do
      csv="$result_dir/ef${ef_search}.csv"
      stderr="$result_dir/ef${ef_search}.stderr"
      if [[ -s "$csv" ]] && awk -F, -v expected="$REPEATS" '
          NR == 1 { next }
          $2 == 0 { measured += 1 }
          $NF != 0 { bad = 1 }
          END { exit !(measured == expected && !bad) }
        ' "$csv"; then
        continue
      fi
      tmp_csv="$csv.tmp.$$"
      tmp_stderr="$stderr.tmp.$$"
      rm -f "$tmp_csv" "$tmp_stderr"
      "$BINARY" \
        --mode search --index "$index_path" \
        --base-bin "$DATA_ROOT/Amazon_base.bin" \
        --base-labels "$DATA_ROOT/Amazon_base_labels.txt" \
        --query-bin "$query_dir/Amazon_query.bin" \
        --query-labels "$query_dir/Amazon_query_labels.txt" \
        --groundtruth "$gt" \
        --max-queries "$MAX_QUERIES" --ef-search "$ef_search" \
        --threads "$SEARCH_THREADS" --batch-size "$MAX_QUERIES" \
        --warmups "$WARMUPS" --repeats "$REPEATS" \
        >"$tmp_csv" 2>"$tmp_stderr"
      awk -F, -v expected="$REPEATS" '
        NR == 1 { next }
        $2 == 0 { measured += 1 }
        $NF != 0 { bad = 1 }
        END { exit !(measured == expected && !bad) }
      ' "$tmp_csv"
      mv "$tmp_csv" "$csv"
      mv "$tmp_stderr" "$stderr"
      printf '%s %s ef=%s done\n' "$variant" "$workload" "$ef_search"
    done
  done
done

printf 'all low-selectivity ACORN screen cases complete\n'
