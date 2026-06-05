#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/benchmarks/run_group_graph_router_ab.sh

Purpose:
  Sweep the FastGrnndCuda mixed exact/GNN group-graph router:

    UNG_FAST_GRNND_BATCH_EXACT_NX=0/128/256/512/...

  The experiment is reviewer-facing: it tests whether routing small/medium
  groups to the batched exact CUDA path improves build time without reproducing
  the pure-exact recall regression.

Important environment variables:
  DATASET, DATA_DIR, QUERY_DIR_NAME, GT_FILE
      Dataset/query paths. Defaults target Amazon 1% x200 coverage-query.
  ROUTER_THRESHOLDS
      Space-separated exact-router thresholds. Default: "0 128 256 512".
  RUN_CPU
      1 to include CPU Vamana group baseline. Default: 1.
  VARIANT
      fastgrnnd_cpu_fallback or fastgrnnd_complete_fallback.
      Default: fastgrnnd_cpu_fallback.
  WAIT_GPU_IDLE
      1 refuses to run if GPU utilization, memory, or compute apps are non-idle.
      Default: 1.
  GPU_IDLE_MAX_MEMORY_MB
      Maximum allowed used GPU memory when WAIT_GPU_IDLE=1. Default: 1024.
  DRY_RUN
      1 validates commands/env without launching builds or requiring idle GPU.

Outputs:
  <out_root>/summary.csv
  <out_root>/<case>/summary.csv
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

data_dir="${DATA_DIR:-/home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x200_jitter_nonempty}"
dataset="${DATASET:-Amazon_1pct_x200}"
query_dir_name="${QUERY_DIR_NAME:-query_coverage_1000}"
gt_file="${GT_FILE:-$data_dir/query_coverage_1000_gt/gt_K10_containment.bin}"
out_root="${OUTDIR:-$(make_out_dir group_graph_router_ab)}"
thresholds="${ROUTER_THRESHOLDS:-0 128 256 512}"
variant="${VARIANT:-fastgrnnd_cpu_fallback}"
run_cpu="${RUN_CPU:-1}"
wait_gpu_idle="${WAIT_GPU_IDLE:-1}"
dry_run="${DRY_RUN:-0}"

case "$variant" in
  fastgrnnd_cpu_fallback|fastgrnnd_complete_fallback) ;;
  *)
    echo "[ERROR] unsupported VARIANT=$variant" >&2
    echo "        Use fastgrnnd_cpu_fallback or fastgrnnd_complete_fallback." >&2
    exit 2
    ;;
esac

if [[ "$dry_run" != "1" && "$wait_gpu_idle" == "1" ]]; then
  require_gpu_idle
fi

mkdir -p "$out_root"
summary="$out_root/summary.csv"
echo "case,router_exact_nx,variant,index_ms,group_ms,cross_ms,lsearch,avg_efs,avg_time_ms,avg_recall,search_summary,index_log" > "$summary"

common_env=(
  DATASET="$dataset"
  DATA_DIR="$data_dir"
  QUERY_DIR_NAME="$query_dir_name"
  BASE_BIN_FILE="${BASE_BIN_FILE:-$data_dir/${dataset}_base.bin}"
  BASE_LABEL_FILE="${BASE_LABEL_FILE:-$data_dir/${dataset}_labels.txt}"
  BASE_LABEL_INFO_FILE="${BASE_LABEL_INFO_FILE:-$data_dir/info.log}"
  BASE_TREE_ROOTS_FILE="${BASE_TREE_ROOTS_FILE:-$data_dir/tree_roots.txt}"
  QUERY_BIN_FILE="${QUERY_BIN_FILE:-$data_dir/$query_dir_name/query.bin}"
  QUERY_LABEL_FILE="${QUERY_LABEL_FILE:-$data_dir/$query_dir_name/query_labels.txt}"
  QUERY_GROUP_ID_FILE="${QUERY_GROUP_ID_FILE:-$data_dir/$query_dir_name/query_group_ids.txt}"
  GT_FILE="$gt_file"
  UNG_ADDITIONAL_EDGES_IMPL="${UNG_ADDITIONAL_EDGES_IMPL:-0}"
  UNG_CROSS_EDGE_IMPL="${UNG_CROSS_EDGE_IMPL:-1}"
  UNG_GPU_TOPK_IMPL="${UNG_GPU_TOPK_IMPL:-3}"
  UNG_CROSS_EDGE_GPU_STRICT="${UNG_CROSS_EDGE_GPU_STRICT:-1}"
  UNG_TAGORE_MIN_GROUP_SIZE="${UNG_TAGORE_MIN_GROUP_SIZE:-128}"
  UNG_FAST_GRNND_LIGHT_PRUNE_NX="${UNG_FAST_GRNND_LIGHT_PRUNE_NX:-256}"
  UNG_FAST_GRNND_REPAIR_DEGREE="${UNG_FAST_GRNND_REPAIR_DEGREE:-1}"
  UNG_TAGORE_COMPACT_D2H="${UNG_TAGORE_COMPACT_D2H:-1}"
  NUM_THREADS="${NUM_THREADS:-128}"
  NUM_REPEATS="${NUM_REPEATS:-3}"
  LSEARCH_VALUES="${LSEARCH_VALUES:-100 200 500 1000}"
  DRY_RUN="$dry_run"
)

append_child_summary() {
  local case_name="$1"
  local router_nx="$2"
  local child_summary="$3"
  if [[ ! -f "$child_summary" ]]; then
    echo "[WARN] missing child summary: $child_summary" >&2
    return
  fi
  tail -n +2 "$child_summary" | awk -v case_name="$case_name" -v router_nx="$router_nx" \
    'BEGIN{FS=OFS=","} {print case_name, router_nx, $0}' >> "$summary"
}

run_case() {
  local case_name="$1"
  local router_nx="$2"
  shift 2
  local case_out="$out_root/$case_name"
  mkdir -p "$case_out"
  echo "[RUN] $case_name router_exact_nx=$router_nx -> $case_out"
  env "${common_env[@]}" "$@" OUTDIR="$case_out" scripts/benchmarks/run_end_to_end_recall_ab.sh
  append_child_summary "$case_name" "$router_nx" "$case_out/summary.csv"
}

if [[ "$run_cpu" == "1" ]]; then
  run_case cpu_vamana 0 VARIANTS=cpu_vamana_group UNG_FAST_GRNND_BATCH_EXACT_NX=0
fi

for threshold in $thresholds; do
  case_name="router_exact_nx${threshold}"
  run_case "$case_name" "$threshold" VARIANTS="$variant" UNG_FAST_GRNND_BATCH_EXACT_NX="$threshold"
done

echo "[OK] wrote $out_root"
echo "[OK] summary: $summary"
