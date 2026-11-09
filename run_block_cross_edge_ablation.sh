#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-all}"

BUILD_CONFIGS=(
  "experiments/cpu_special_blocks/Amazon_config_Trie_block_cross_block_sweep.json"
  "experiments/cpu_special_blocks/Music_config_Trie_block_cross_block_sweep.json"
  "experiments/cpu_special_blocks/VImg_config_Trie_block_cross_block_sweep.json"
)

SEARCH_CONFIGS=(
  "experiments/search_comparison/config_amazon_block_cross_block.json"
  "experiments/search_comparison/config_music_block_cross_block.json"
  "experiments/search_comparison/config_VImg_block_cross_block.json"
)

usage() {
  cat <<'EOF'
Usage:
  ./run_block_cross_edge_ablation.sh [all|build|search]

Stages:
  all     Build all cross-edge variants, then run all searches (default).
  build   Build only the cross-edge variants.
  search  Search existing cross-edge variants without rebuilding them.

Experiment matrix:
  datasets: Amazon, Music, VImg
  block threshold: 500
  block-local degree M: 32
  cross-block edges: 2, 4, 6
  query task: query_minlen1_sel10to40pct
EOF
}

run_builds() {
  local config
  for config in "${BUILD_CONFIGS[@]}"; do
    printf '\n[%s] Building indexes from %s\n' "$(date '+%F %T')" "${config}"
    "${SCRIPT_DIR}/run_cpu_special_blocks_experiment.sh" "${SCRIPT_DIR}/${config}"
  done
}

run_searches() {
  local config
  for config in "${SEARCH_CONFIGS[@]}"; do
    printf '\n[%s] Running searches from %s\n' "$(date '+%F %T')" "${config}"
    "${SCRIPT_DIR}/run_search_comparison_experiment.sh" "${SCRIPT_DIR}/${config}"
  done
}

case "${STAGE}" in
  all)
    run_builds
    run_searches
    ;;
  build)
    run_builds
    ;;
  search)
    run_searches
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    printf 'Unknown stage: %s\n\n' "${STAGE}" >&2
    usage >&2
    exit 2
    ;;
esac
