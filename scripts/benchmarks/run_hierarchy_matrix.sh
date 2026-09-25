#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'EOF'
Usage:
  DATASET=amazon DATA_DIR=/path/to/data QUERY_DIR_NAME=query_name \
  scripts/benchmarks/run_hierarchy_matrix.sh

Runs reproducible combinations of base topology, hierarchy plan, and entry
strategy through run_end_to_end_recall_ab.sh. All options accepted by that
script may also be set here.

Optional:
  MATRIX_FILE   pipe-delimited rows: name|base_topology|layers|entry_strategy
  OUTDIR        output root
  VARIANTS      graph-build variants passed to the underlying runner
                default: best_full_quality

The built-in matrix is:
  lng_0|lng||optimized_lng
  trie_0|trie||trie
  trie_1k|lng|1000:trie|trie
  trie_1k_16k|lng|1000:trie,16000:trie|trie
  mixed_1k_16k|lng|1000:trie,16000:lng|trie
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

for name in DATASET DATA_DIR QUERY_DIR_NAME; do
  if [[ -z "${!name:-}" ]]; then
    echo "[ERROR] missing required env: $name" >&2
    usage >&2
    exit 2
  fi
done

out_root="${OUTDIR:-$(make_out_dir hierarchy_matrix)}"
variants="${VARIANTS:-best_full_quality}"
mkdir -p "$out_root"
manifest="$out_root/matrix.csv"
echo "case,base_topology,hierarchy_layers,entry_strategy,run_root,summary_csv" > "$manifest"

default_matrix() {
  cat <<'EOF'
lng_0|lng||optimized_lng
trie_0|trie||trie
trie_1k|lng|1000:trie|trie
trie_1k_16k|lng|1000:trie,16000:trie|trie
mixed_1k_16k|lng|1000:trie,16000:lng|trie
EOF
}

run_case() {
  local case_name="$1"
  local base_topology="$2"
  local layers="$3"
  local entry_strategy="$4"
  local run_root="$out_root/$case_name"
  local special_blocks=0
  if [[ -n "$layers" ]]; then
    special_blocks=1
  fi

  echo "[RUN] case=$case_name base=$base_topology layers=${layers:-none} entry=$entry_strategy"
  env \
    OUTDIR="$run_root" \
    UNG_BASE_GROUP_TOPOLOGY="$base_topology" \
    UNG_HIERARCHY_LAYERS="$layers" \
    UNG_SPECIAL_BLOCKS="$special_blocks" \
    UNG_SPECIAL_BLOCK_SEARCH="$special_blocks" \
    ENTRY_GROUP_STRATEGY="$entry_strategy" \
    VARIANTS="$variants" \
    "$repo_root/scripts/benchmarks/run_end_to_end_recall_ab.sh"
  echo "$case_name,$base_topology,\"${layers:-none}\",$entry_strategy,$run_root,$run_root/summary.csv" >> "$manifest"
}

if [[ -n "${MATRIX_FILE:-}" ]]; then
  matrix_source="$MATRIX_FILE"
else
  matrix_source="$out_root/default_matrix.txt"
  default_matrix > "$matrix_source"
fi

while IFS='|' read -r case_name base_topology layers entry_strategy; do
  [[ -z "$case_name" || "$case_name" == \#* ]] && continue
  if [[ -z "$base_topology" || -z "$entry_strategy" ]]; then
    echo "[ERROR] malformed matrix row for case '$case_name'" >&2
    exit 2
  fi
  run_case "$case_name" "$base_topology" "$layers" "$entry_strategy"
done < "$matrix_source"

echo "[OK] matrix manifest: $manifest"
