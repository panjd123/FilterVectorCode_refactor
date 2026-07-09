#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'EOF'
Usage:
  DATASET=amazon_x1 DATA_DIR=/path/to/x1 QUERY_DIR_NAME=query_coverage_1000 \
  scripts/benchmarks/run_special_block_build_query_ab.sh

This is a thin orchestrator over run_end_to_end_recall_ab.sh. It runs:
  - baseline_plain: ordinary UNG, no special blocks
  - special_T*_skip: special blocks + free-state search + trivial-source skip
  - special_T*_noskip: special blocks + free-state search without trivial-source skip

Required:
  DATASET, DATA_DIR, QUERY_DIR_NAME as accepted by run_end_to_end_recall_ab.sh

Important:
  The input must be an explicit x1/original or x1-restored dataset. This script
  sets UNG_SPECIAL_BLOCK_DATA_MODE=x1 for special variants, but it cannot prove
  the dataset is not repeated xN. Put that provenance in the report.

Optional:
  THRESHOLDS              default: "50 100 200 400 800 1600"
  SPECIAL_VARIANTS        default: "special_skip special_noskip"
  BASELINE_VARIANTS       default: "cpu_vamana_group"
  OUTDIR                  output root
  All run_end_to_end_recall_ab.sh options are forwarded.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[ERROR] missing required env: $name" >&2
    usage >&2
    exit 2
  fi
}

require_env DATASET
require_env DATA_DIR
require_env QUERY_DIR_NAME

thresholds="${THRESHOLDS:-50 100 200 400 800 1600}"
baseline_variants="${BASELINE_VARIANTS:-cpu_vamana_group}"
special_variants="${SPECIAL_VARIANTS:-special_skip special_noskip}"
out_root="${OUTDIR:-$(make_out_dir special_block_build_query_ab)}"
summary="$out_root/special_block_runs.csv"
mkdir -p "$out_root"
echo "case,threshold,skip_trivial,run_root,summary_csv" > "$summary"

run_case() {
  local case_name="$1"
  local threshold="$2"
  local skip_trivial="$3"
  local run_out="$out_root/$case_name"
  mkdir -p "$run_out"

  local -a env_args=(
    "OUTDIR=$run_out"
    "DATASET=$DATASET"
    "DATA_DIR=$DATA_DIR"
    "QUERY_DIR_NAME=$QUERY_DIR_NAME"
    "VARIANTS=${VARIANTS:-$baseline_variants}"
  )

  local name
  for name in \
    BUILD_DIR BUILD_APP SEARCH_APP GT_APP AUTO_BUILD RESULTS_ROOT K NUM_THREADS NUM_REPEATS \
    LSEARCH_VALUES NUM_ENTRY_POINTS MAX_DEGREE LBUILD ALPHA NUM_CROSS_EDGES \
    BUILD_SCENARIO SEARCH_SCENARIO GENERATE_GT GT_FILE GT_DIR BASE_BIN_FILE BASE_LABEL_FILE \
    BASE_LABEL_INFO_FILE BASE_TREE_ROOTS_FILE QUERY_BIN_FILE QUERY_LABEL_FILE QUERY_GROUP_ID_FILE \
    FORCE_USE_ALG IS_IDEA2_AVAILABLE IS_NEW_TRIE_METHOD IS_REC_MORE_START IS_UNG_MORE_ENTRY \
    ENTRY_GROUP_PROVIDER SELECTOR_MODEL_PREFIX LSEARCH_START LSEARCH_STEP EFS_START \
    EFS_STEP_SLOW EFS_STEP_FAST LSEARCH_THRESHOLD UNG_CROSS_EDGE_IMPL UNG_ADDITIONAL_EDGES_IMPL \
    UNG_GPU_TOPK_IMPL UNG_CROSS_EDGE_GPU_STRICT; do
    if [[ -n "${!name:-}" ]]; then
      env_args+=("$name=${!name}")
    fi
  done

  if [[ "$threshold" != "0" ]]; then
    env_args+=(
      "UNG_SPECIAL_BLOCKS=1"
      "UNG_SPECIAL_BLOCK_DATA_MODE=x1"
      "UNG_SPECIAL_BLOCK_MIN_POINTS=$threshold"
      "UNG_SPECIAL_BLOCK_SEARCH=1"
      "UNG_SPECIAL_BLOCK_FREE_USE_REGULAR=${UNG_SPECIAL_BLOCK_FREE_USE_REGULAR:-1}"
      "UNG_SPECIAL_BLOCK_SKIP_TRIVIAL=$skip_trivial"
    )
  fi

  echo "[RUN] case=$case_name threshold=$threshold skip_trivial=$skip_trivial"
  env "${env_args[@]}" scripts/benchmarks/run_end_to_end_recall_ab.sh
  echo "$case_name,$threshold,$skip_trivial,$run_out,$run_out/summary.csv" >> "$summary"
}

run_case "baseline_plain" 0 0

for threshold in $thresholds; do
  for variant in $special_variants; do
    case "$variant" in
      special_skip)
        run_case "special_T${threshold}_skip" "$threshold" 1
        ;;
      special_noskip)
        run_case "special_T${threshold}_noskip" "$threshold" 0
        ;;
      *)
        echo "[ERROR] unknown SPECIAL_VARIANTS item: $variant" >&2
        exit 2
        ;;
    esac
  done
done

mapfile -t run_roots < <(tail -n +2 "$summary" | awk -F, '{print $4}')
python3 tools/benchmarks/summarize_special_block_ab.py "${run_roots[@]}" --out-dir "$out_root/summary"
echo "[OK] runs: $summary"
echo "[OK] summary: $out_root/summary/special_block_ab_summary.md"
