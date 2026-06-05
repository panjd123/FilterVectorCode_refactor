#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

bin="${BENCH_BIN:-}"
if [[ -z "$bin" ]]; then
  bin="$("$(dirname "$0")/build_query_entry_group_bench.sh" | tail -n 1)"
fi

out_dir="$(make_out_dir query_entry_group_bench)"
base_label_file="${BASE_LABEL_FILE:-/home/graphdb/FilterVectorResultsRefactor/datasets/amazon/labels.txt}"
query_label_file="${QUERY_LABEL_FILE:-/home/graphdb/FilterVectorResultsRefactor/datasets/amazon/query_labels.txt}"
provider="${PROVIDER:-gpu_cover_frontier}"
max_base="${MAX_BASE:-0}"
max_query="${MAX_QUERY:-2048}"
query_batch="${QUERY_BATCH:-2048}"
group_chunk="${GROUP_CHUNK:-65536}"
frontier_cover_cap="${FRONTIER_COVER_CAP:-8192}"
repeats="${REPEATS:-5}"
warmup="${WARMUP:-1}"
threads="${THREADS:-128}"
gpu_id="${GPU_ID:-0}"

args=(
  --base-label-file "$base_label_file"
  --query-label-file "$query_label_file"
  --provider "$provider"
  --max-query "$max_query"
  --query-batch "$query_batch"
  --group-chunk "$group_chunk"
  --frontier-cover-cap "$frontier_cover_cap"
  --repeats "$repeats"
  --warmup "$warmup"
  --threads "$threads"
  --gpu "$gpu_id"
  --output-csv "$out_dir/query_entry_group_bench.csv"
)

if [[ "$max_base" != "0" ]]; then
  args+=(--max-base "$max_base")
fi
if [[ "${CHECK:-1}" == "0" ]]; then
  args+=(--no-check)
fi

log="$out_dir/run.log"
{
  echo "# date: $(date -Is)"
  echo "# bin: $bin"
  echo "# base_label_file: $base_label_file"
  echo "# query_label_file: $query_label_file"
  echo "# provider: $provider"
  echo "# output_csv: $out_dir/query_entry_group_bench.csv"
} > "$log"

run_with_gpu_lock perf "$bin" "${args[@]}" >> "$log" 2>&1
cat "$log"
echo "$out_dir"
