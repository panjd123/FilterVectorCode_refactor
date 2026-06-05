#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

data_dir="${DATA_DIR:-/home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x400_jitter_nonempty}"
query_dir_name="${QUERY_DIR_NAME:-query_coverage_1000}"
gt_file="${GT_FILE:-$data_dir/query_coverage_1000_gt/gt_K10_containment.bin}"
out_root="${OUTDIR:-$(make_out_dir x400_reverse_tail_ab)}"
run_cpu="${RUN_CPU:-0}"
wait_gpu_idle="${WAIT_GPU_IDLE:-1}"
run_repair_ablation="${RUN_REPAIR_ABLATION:-1}"
run_reverse_tail="${RUN_REVERSE_TAIL:-1}"
run_compact_ablation="${RUN_COMPACT_ABLATION:-0}"
run_graph_diag="${RUN_GRAPH_DIAG:-0}"
run_only_cpu="${RUN_ONLY_CPU:-0}"

if [[ "$wait_gpu_idle" == "1" ]]; then
  require_gpu_idle
fi

mkdir -p "$out_root"

common_env=(
  DATASET=Amazon_1pct_x400
  DATA_DIR="$data_dir"
  QUERY_DIR_NAME="$query_dir_name"
  BASE_BIN_FILE="$data_dir/Amazon_1pct_x400_base.bin"
  BASE_LABEL_FILE="$data_dir/Amazon_1pct_x400_labels.txt"
  BASE_LABEL_INFO_FILE="$data_dir/info.log"
  BASE_TREE_ROOTS_FILE="$data_dir/tree_roots.txt"
  QUERY_BIN_FILE="$data_dir/$query_dir_name/query.bin"
  QUERY_LABEL_FILE="$data_dir/$query_dir_name/query_labels.txt"
  QUERY_GROUP_ID_FILE="$data_dir/$query_dir_name/query_group_ids.txt"
  GT_FILE="$gt_file"
  UNG_ADDITIONAL_EDGES_IMPL=0
  UNG_TAGORE_MIN_GROUP_SIZE=256
  UNG_DIRECT_QID_FUSED=0
  UNG_DIRECT_QID_ALL_FUSED=0
  UNG_GPU_TOPK_IMPL=3
  UNG_CROSS_EDGE_IMPL=1
  UNG_CROSS_EDGE_GPU_STRICT=1
  UNG_TAGORE_COMPACT_D2H="${UNG_TAGORE_COMPACT_D2H:-1}"
  NUM_THREADS="${NUM_THREADS:-128}"
  NUM_REPEATS="${NUM_REPEATS:-3}"
  LSEARCH_VALUES="${LSEARCH_VALUES:-1000 5000}"
)

run_case() {
  local name="$1"
  shift
  local case_out="$out_root/$name"
  mkdir -p "$case_out"
  echo "[RUN] $name -> $case_out"
  env "${common_env[@]}" "$@" OUTDIR="$case_out" scripts/benchmarks/run_end_to_end_recall_ab.sh
}

run_graph_diag_case() {
  local name="$1"
  local run_dir="$out_root/$name/fastgrnnd_cpu_fallback"
  if [[ "$name" == "cpu_vamana" ]]; then
    run_dir="$out_root/$name/cpu_vamana_group"
  fi
  if [[ ! -d "$run_dir/index_files" ]]; then
    echo "[WARN] skip graph diagnostics for $name; missing $run_dir/index_files" >&2
    return
  fi
  echo "[RUN] graph diagnostics $name"
  python3 tools/benchmarks/diagnose_ung_graph_structure.py \
    "$run_dir/index_files" \
    --out "$out_root/$name/graph_diag" \
    --reciprocal-limit 0 >/dev/null
}

if [[ "$run_cpu" == "1" ]]; then
  run_case cpu_vamana VARIANTS=cpu_vamana_group
fi

if [[ "$run_only_cpu" == "1" ]]; then
  if [[ "$run_cpu" != "1" ]]; then
    echo "[ERROR] RUN_ONLY_CPU=1 requires RUN_CPU=1" >&2
    exit 2
  fi
  run_repair_ablation=0
  run_reverse_tail=0
  run_compact_ablation=0
else

  if [[ "$run_repair_ablation" == "1" ]]; then
    run_case light512_norepair \
      VARIANTS=fastgrnnd_cpu_fallback \
      UNG_FAST_GRNND_LIGHT_PRUNE_NX=512 \
      UNG_FAST_GRNND_LIGHT_HEAD="${LIGHT_HEAD:-24}" \
      UNG_FAST_GRNND_LIGHT_REVERSE_CAP=0 \
      UNG_FAST_GRNND_REPAIR_DEGREE=0
  fi

  run_case light512_repair \
    VARIANTS=fastgrnnd_cpu_fallback \
    UNG_FAST_GRNND_LIGHT_PRUNE_NX=512 \
    UNG_FAST_GRNND_LIGHT_HEAD="${LIGHT_HEAD:-24}" \
    UNG_FAST_GRNND_LIGHT_REVERSE_CAP=0 \
    UNG_FAST_GRNND_REPAIR_DEGREE="${REPAIR_DEGREE:-1}"

  if [[ "$run_reverse_tail" == "1" ]]; then
    run_case light512_reverse_repair \
      VARIANTS=fastgrnnd_cpu_fallback \
      UNG_FAST_GRNND_LIGHT_PRUNE_NX=512 \
      UNG_FAST_GRNND_LIGHT_HEAD="${LIGHT_HEAD:-24}" \
      UNG_FAST_GRNND_LIGHT_REVERSE_CAP="${LIGHT_REVERSE_CAP:-8}" \
      UNG_FAST_GRNND_LIGHT_REVERSE_SLOTS="${LIGHT_REVERSE_SLOTS:-4}" \
      UNG_FAST_GRNND_LIGHT_REVERSE_FORWARD_CAP="${LIGHT_REVERSE_FORWARD_CAP:-24}" \
      UNG_FAST_GRNND_REPAIR_DEGREE="${REPAIR_DEGREE:-1}"
  fi

  if [[ "$run_compact_ablation" == "1" ]]; then
    run_case light512_repair_compact_off \
      VARIANTS=fastgrnnd_cpu_fallback \
      UNG_FAST_GRNND_LIGHT_PRUNE_NX=512 \
      UNG_FAST_GRNND_LIGHT_HEAD="${LIGHT_HEAD:-24}" \
      UNG_FAST_GRNND_LIGHT_REVERSE_CAP=0 \
      UNG_FAST_GRNND_REPAIR_DEGREE="${REPAIR_DEGREE:-1}" \
      UNG_TAGORE_COMPACT_D2H=0
  fi
fi

summary_args=()
if [[ -d "$out_root/cpu_vamana/cpu_vamana_group" ]]; then
  summary_args+=("cpu=$out_root/cpu_vamana/cpu_vamana_group")
fi
if [[ -d "$out_root/light512_norepair/fastgrnnd_cpu_fallback" ]]; then
  summary_args+=("light512_norepair=$out_root/light512_norepair/fastgrnnd_cpu_fallback")
fi
if [[ -d "$out_root/light512_repair/fastgrnnd_cpu_fallback" ]]; then
  summary_args+=("light512_repair=$out_root/light512_repair/fastgrnnd_cpu_fallback")
fi
if [[ -d "$out_root/light512_reverse_repair/fastgrnnd_cpu_fallback" ]]; then
  summary_args+=("light512_reverse_repair=$out_root/light512_reverse_repair/fastgrnnd_cpu_fallback")
fi
if [[ -d "$out_root/light512_repair_compact_off/fastgrnnd_cpu_fallback" ]]; then
  summary_args+=("light512_repair_compact_off=$out_root/light512_repair_compact_off/fastgrnnd_cpu_fallback")
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  if [[ "$run_graph_diag" == "1" ]]; then
    for case_name in cpu_vamana light512_norepair light512_repair light512_reverse_repair light512_repair_compact_off; do
      if [[ -d "$out_root/$case_name" ]]; then
        run_graph_diag_case "$case_name"
      fi
    done
  fi
  python3 scripts/benchmarks/summarize_search_recall.py "${summary_args[@]}" \
    -o "$out_root/search_recall_summary.csv"
  python3 tools/benchmarks/summarize_x400_ab.py "$out_root" \
    --csv "$out_root/x400_ab_summary.csv" \
    --md "$out_root/x400_ab_summary.md" >/dev/null
fi

echo "[OK] wrote $out_root"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  echo "[OK] summary: $out_root/search_recall_summary.csv"
  echo "[OK] x400 summary: $out_root/x400_ab_summary.md"
fi
