#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./run_block_presence_ablation.sh [search]

Search only the no-block variant; reuse existing with-block results.
No index build is needed: use the existing Trie base graph without its block overlay.

Experiment matrix:
  datasets: Amazon, VImg, Music (sequential)
  query task: query_minlen1_sel10to40pct
  Lsearch: 10000..100000, step 10000 (10 values; no prefix)
  base index: Trie_block_500_M_32_cross_block_4/index_files
  entry strategy: trie
  special block search: disabled; block sidecar: not loaded
  result method: Trie_no_block
EOF
}

if (( $# > 1 )); then
  usage >&2
  exit 2
fi
case "${1:-search}" in
  search) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    printf 'Unknown stage: %s\n\n' "$1" >&2
    usage >&2
    exit 2
    ;;
esac

SEARCH_CONFIGS=(
  "experiments/search_comparison/config_amazon_no_block.json"
  "experiments/search_comparison/config_VImg_no_block.json"
  "experiments/search_comparison/config_music_no_block.json"
)

for config in "${SEARCH_CONFIGS[@]}"; do
  printf '\n[%s] Running no-block search from %s\n' "$(date '+%F %T')" "${config}"
  "${SCRIPT_DIR}/run_search_comparison_experiment.sh" "${SCRIPT_DIR}/${config}"
done
