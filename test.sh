#!/usr/bin/env bash
set -euo pipefail

# Run every search implementation with the same dataset/task coverage as Amazon.
#
# Usage:
#   ./test.sh --check              # validate every command/config without running experiments
#   ./test.sh                      # run all eight datasets
#   ./test.sh Amazon BReviews      # run only the named datasets
# Every selected dataset runs the same complete algorithm suite.
# Logs are written to logs/<run-id>/<dataset>/<algorithm>.log and streamed
# to the terminal at the same time.
# Amazon  VImg Genome Reviews BReviews Music Tiktok Laion 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ALL_DATASETS=(Laion )

declare -A SEARCH_CONFIG_SLUG=(
  [Amazon]=amazon
  [BReviews]=breviews
  [Laion]=laion
  [Music]=music
  [Tiktok]=tiktok
  [VImg]=VImg
  [Genome]=genome
  [Reviews]=reviews
)

# Every dataset builds only the cross_block_4 index consumed by search.
declare -A TRIE_BUILD_CONFIG=(
  [Amazon]=Amazon_config_Trie_block.json
  [BReviews]=BReviews_config_Trie_block.json
  [Laion]=Laion_config_Trie_block.json
  [Music]=Music_config_Trie_block.json
  [Tiktok]=Tiktok_config_Trie_block.json
  [VImg]=VImg_config_Trie_block.json
  [Genome]=Genome_config_Trie_block.json
  [Reviews]=Reviews_config_Trie_block.json
)

TRIE_BLOCK_INDEX_NAME="Trie_block_500_M_32_cross_block_4"

usage() {
  cat <<'EOF'
Usage:
  ./test.sh --check
  ./test.sh [Amazon|BReviews|Laion|Music|Tiktok|VImg|Genome|Reviews ...]
EOF
}

is_known_dataset() {
  [[ -n "${SEARCH_CONFIG_SLUG[$1]+x}" ]]
}

run_logged() {
  local dataset="$1"
  local algorithm="$2"
  shift 2

  local log_dir="${RUN_LOG_DIR}/${dataset}"
  local log_file="${log_dir}/${algorithm}.log"
  local command_status tee_status final_status
  local -a pipeline_status
  mkdir -p "$log_dir"

  {
    printf '[%s] START dataset=%s algorithm=%s\n' \
      "$(date '+%F %T')" "$dataset" "$algorithm"
    printf 'COMMAND:'
    printf ' %q' "$@"
    printf '\n'
  } | tee "$log_file"

  set +e
  "$@" 2>&1 | tee -a "$log_file"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e

  command_status="${pipeline_status[0]}"
  tee_status="${pipeline_status[1]}"
  final_status="$command_status"
  if [[ "$final_status" -eq 0 && "$tee_status" -ne 0 ]]; then
    final_status="$tee_status"
  fi

  printf '[%s] END dataset=%s algorithm=%s exit=%s\n' \
    "$(date '+%F %T')" "$dataset" "$algorithm" "$final_status" \
    | tee -a "$log_file"
  return "$final_status"
}

trie_block_index_complete() {
  local index_root="$1"
  local index_dir="${index_root}/index_files"
  local block_dir="${index_root}/block_index_files"
  local required_file

  for required_file in \
    "${index_dir}/meta" \
    "${index_dir}/vecs.bin" \
    "${index_dir}/labels.txt" \
    "${index_dir}/group_id_to_label_set" \
    "${index_dir}/group_id_to_range" \
    "${block_dir}/meta" \
    "${block_dir}/special_edges.bin" \
    "${block_dir}/special_blocks.bin" \
    "${block_dir}/special_block_trie.bin"; do
    [[ -s "$required_file" ]] || return 1
  done

  grep -qx "base_group_topology=trie" "${index_dir}/meta" &&
    grep -qx "special_block_min_points=500" "${block_dir}/meta" &&
    grep -qx "special_block_max_degree=32" "${block_dir}/meta" &&
    grep -qx "special_block_num_cross_edges=4" "${block_dir}/meta"
}

ensure_trie_block_index() {
  local dataset="$1"
  local config="experiments/cpu_special_blocks/${TRIE_BUILD_CONFIG[$dataset]}"
  local result_root index_root

  result_root="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[\"result_root\"])" "$config")"
  index_root="${result_root}/${dataset}/index/${TRIE_BLOCK_INDEX_NAME}"

  if trie_block_index_complete "$index_root"; then
    printf "[%s] Trie block index already complete: %s\n" \
      "$(date "+%F %T")" "$index_root"
    return 0
  fi

  printf "[%s] Trie block index missing or incomplete; building: %s\n" \
    "$(date "+%F %T")" "$index_root"
  run_logged "$dataset" trie_block_build \
    gpulock perf --wait-gpu-idle 0 -- \
    ./run_cpu_special_blocks_experiment.sh \
    "$config"

  if ! trie_block_index_complete "$index_root"; then
    echo "ERROR: Trie block build completed without a complete index: $index_root" >&2
    return 1
  fi
}

preflight() {
  local dataset slug trie_config runner config
  local -a configs

  command -v python3 >/dev/null
  command -v gpulock >/dev/null
  command -v tee >/dev/null

  if [[ ! -x .venv_curator/bin/python ]]; then
    echo "ERROR: missing Curator environment; run experiments/curator_baseline/setup_curator_env.sh" >&2
    return 1
  fi
  env PYTHONPATH="$SCRIPT_DIR/thirdparty/curator-v2" \
    LD_LIBRARY_PATH="/home/dev/graphdb/OpenBLAS" \
    .venv_curator/bin/python -c "import faiss; from indexes.curator import Curator"

  for runner in \
    run_cpu_special_blocks_experiment.sh \
    run_search_comparison_experiment.sh \
    experiments/curator_baseline/run_curator_baseline_with_env.sh; do
    [[ -x "$runner" ]] || { echo "ERROR: missing executable: $runner" >&2; return 1; }
    bash -n "$runner"
  done

  python3 -m py_compile \
    experiments/result_completion.py \
    experiments/search_comparison/run_search_comparison.py \
    experiments/favor/run_favor_experiment.py \
    experiments/acorn_baseline/run_acorn_baseline.py \
    experiments/curator_baseline/run_curator_baseline.py \
    experiments/navix_baseline/run_navix_baseline.py \
    experiments/milvus_hnsw/run_milvus_hnsw_experiment.py \
    experiments/pathseer/run_pathseer_experiment.py \
    experiments/rwalks/run_rwalks_experiment.py

  for dataset in "$@"; do
    is_known_dataset "$dataset" || { echo "ERROR: unknown dataset: $dataset" >&2; return 1; }
    slug="${SEARCH_CONFIG_SLUG[$dataset]}"
    trie_config="experiments/cpu_special_blocks/${TRIE_BUILD_CONFIG[$dataset]}"
    [[ -f "$trie_config" ]] || { echo "ERROR: missing config: $trie_config" >&2; return 1; }
    python3 -m json.tool "$trie_config" >/dev/null
    configs=(
      "experiments/search_comparison/config_${slug}_ung.json"
      "experiments/search_comparison/config_${slug}.json"
      "experiments/favor/${dataset}_config.json"
      "experiments/acorn_baseline/${dataset}_config.json"
      "experiments/curator_baseline/${dataset}_config.json"
      "experiments/navix_baseline/${dataset}_config.json"
      "experiments/milvus_hnsw/${dataset}_config.json"
      "experiments/pathseer/${dataset}_config.json"
      "experiments/rwalks/${dataset}_config.json"
    )

    for config in "${configs[@]}"; do
      [[ -f "$config" ]] || { echo "ERROR: missing config: $config" >&2; return 1; }
      python3 -m json.tool "$config" >/dev/null
    done
  done

  echo "Preflight passed for: $*"
}

run_dataset() {
  local dataset="$1"
  local slug="${SEARCH_CONFIG_SLUG[$dataset]}"

  ensure_trie_block_index "$dataset"

  run_logged "$dataset" native_ung \
    ./run_search_comparison_experiment.sh \
    "experiments/search_comparison/config_${slug}_ung.json"

  run_logged "$dataset" trie_block \
    ./run_search_comparison_experiment.sh \
    "experiments/search_comparison/config_${slug}.json"

  run_logged "$dataset" favor \
    python3 experiments/favor/run_favor_experiment.py \
    "experiments/favor/${dataset}_config.json"

  run_logged "$dataset" acorn \
    python3 experiments/acorn_baseline/run_acorn_baseline.py \
    "experiments/acorn_baseline/${dataset}_config.json"

  run_logged "$dataset" curator \
    experiments/curator_baseline/run_curator_baseline_with_env.sh \
    "experiments/curator_baseline/${dataset}_config.json"

  run_logged "$dataset" navix \
    python3 experiments/navix_baseline/run_navix_baseline.py \
    "experiments/navix_baseline/${dataset}_config.json"

  run_logged "$dataset" milvus_hnsw \
    python3 experiments/milvus_hnsw/run_milvus_hnsw_experiment.py \
    "experiments/milvus_hnsw/${dataset}_config.json"

  run_logged "$dataset" pathseer \
    python3 experiments/pathseer/run_pathseer_experiment.py \
    "experiments/pathseer/${dataset}_config.json"

  run_logged "$dataset" rwalks \
    python3 experiments/rwalks/run_rwalks_experiment.py \
    "experiments/rwalks/${dataset}_config.json"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--check" ]]; then
  shift
  [[ $# -eq 0 ]] && set -- "${ALL_DATASETS[@]}"
  preflight "$@"
  exit 0
fi

[[ $# -eq 0 ]] && set -- "${ALL_DATASETS[@]}"
preflight "$@"

LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
RUN_LOG_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "$RUN_LOG_DIR"
printf 'Run logs: %s\n' "$RUN_LOG_DIR"

for dataset in "$@"; do
  run_dataset "$dataset"
done
