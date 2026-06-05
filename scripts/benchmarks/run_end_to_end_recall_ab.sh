#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  cat <<'EOF'
Usage:
  DATASET=amazon DATA_DIR=/path/to/data QUERY_DIR_NAME=query_xxx \
  scripts/benchmarks/run_end_to_end_recall_ab.sh

Required inputs:
  DATASET             Dataset name prefix, e.g. amazon
  DATA_DIR            Directory containing <dataset>_base.bin and labels
  QUERY_DIR_NAME      Query subdirectory under DATA_DIR

Optional inputs:
  BUILD_DIR           Build dir with apps/build_UNG_index and apps/search_UNG_index
                      default: <repo>/build_mode_switch
  AUTO_BUILD          Build missing executables in BUILD_DIR, default: 1
  RESULTS_ROOT        Output root
  K                   default: 10
  NUM_THREADS         default: 128
  NUM_REPEATS         default: 3
  LSEARCH_VALUES      default: "20 50 100 200"
  NUM_ENTRY_POINTS    default: 16
  BUILD_SCENARIO      default: general
  SEARCH_SCENARIO     default: containment
  UNG_ADDITIONAL_EDGES_IMPL
                      additional-edge implementation for all variants, default: 1 (skip)
  GENERATE_GT         1 to generate GT when missing, default: 0
  DRY_RUN             1 to print commands only, default: 0

Dataset metadata:
  BASE_LABEL_FILE       default: DATA_DIR/<dataset>_base_labels.txt
  BASE_LABEL_INFO_FILE  default: DATA_DIR/<dataset>_base_labels_info.log
  BASE_TREE_ROOTS_FILE  default: DATA_DIR/tree_roots.txt
  GT_FILE               default: output/GroundTruth/<query>/K<K>/<dataset>_gt_labels_containment.bin

Variants can be overridden with VARIANTS, default:
  cpu_vamana_group fastgrnnd_cpu_fallback fastgrnnd_complete_fallback
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

build_dir="${BUILD_DIR:-$repo_root/build_mode_switch}"
build_app="${BUILD_APP:-$build_dir/apps/build_UNG_index}"
search_app="${SEARCH_APP:-$build_dir/apps/search_UNG_index}"
gt_app="${GT_APP:-$build_dir/tools/compute_groundtruth}"
auto_build="${AUTO_BUILD:-1}"

ensure_exe() {
  local path="$1"
  local target="$2"
  if [[ -x "$path" ]]; then
    return
  fi
  if [[ "$auto_build" == "1" ]]; then
    echo "[BUILD] missing $target; running cmake --build $build_dir --target $target"
    cmake --build "$build_dir" -j "${BUILD_JOBS:-32}" --target "$target"
  fi
  if [[ ! -x "$path" ]]; then
    echo "[ERROR] $target not executable: $path" >&2
    exit 2
  fi
}

ensure_exe "$build_app" build_UNG_index
ensure_exe "$search_app" search_UNG_index

k="${K:-10}"
num_threads="${NUM_THREADS:-128}"
num_repeats="${NUM_REPEATS:-3}"
lsearch_values="${LSEARCH_VALUES:-20 50 100 200}"
num_entry_points="${NUM_ENTRY_POINTS:-16}"
max_degree="${MAX_DEGREE:-32}"
lbuild="${LBUILD:-100}"
alpha="${ALPHA:-1.2}"
num_cross_edges="${NUM_CROSS_EDGES:-6}"
build_scenario="${BUILD_SCENARIO:-general}"
search_scenario="${SEARCH_SCENARIO:-containment}"
dry_run="${DRY_RUN:-0}"
additional_edges_impl="${UNG_ADDITIONAL_EDGES_IMPL:-1}"
cross_edge_impl="${UNG_CROSS_EDGE_IMPL:-1}"
gpu_topk_impl="${UNG_GPU_TOPK_IMPL:-3}"
cross_edge_gpu_strict="${UNG_CROSS_EDGE_GPU_STRICT:-1}"

base_bin="${BASE_BIN_FILE:-$DATA_DIR/${DATASET}_base.bin}"
base_label="${BASE_LABEL_FILE:-$DATA_DIR/${DATASET}_base_labels.txt}"
base_info="${BASE_LABEL_INFO_FILE:-$DATA_DIR/${DATASET}_base_labels_info.log}"
tree_roots="${BASE_TREE_ROOTS_FILE:-$DATA_DIR/tree_roots.txt}"
query_dir="$DATA_DIR/$QUERY_DIR_NAME"
query_bin="${QUERY_BIN_FILE:-$query_dir/${DATASET}_query.bin}"
query_label="${QUERY_LABEL_FILE:-$query_dir/${DATASET}_query_labels.txt}"
query_group_ids="${QUERY_GROUP_ID_FILE:-$query_dir/${DATASET}_query_source_groups.txt}"

safe_query_name="$(printf '%s' "$QUERY_DIR_NAME" | tr '/' '_')"
out="$(make_out_dir end_to_end_recall_ab)"
gt_dir="${GT_DIR:-$out/GroundTruth/GT_${safe_query_name}_K${k}}"
gt_file="${GT_FILE:-$gt_dir/${DATASET}_gt_labels_containment.bin}"
summary_csv="$out/summary.csv"

mkdir -p "$out" "$gt_dir"
echo "variant,index_ms,group_ms,cross_ms,lsearch,avg_efs,avg_time_ms,avg_recall,search_summary,index_log" > "$summary_csv"

check_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] missing $label: $path" >&2
    exit 2
  fi
}

check_file "$base_bin" "base vectors"
check_file "$base_label" "base labels"
check_file "$base_info" "base label info"
check_file "$tree_roots" "tree roots"
check_file "$query_bin" "query vectors"
check_file "$query_label" "query labels"
if [[ ! -f "$query_group_ids" ]]; then
  echo "[WARN] query source groups missing: $query_group_ids" >&2
  echo "       In the current search_UNG_index path this only matters if is_ung_more_entry is enabled; that flag is not exposed by the CLI and defaults to false." >&2
fi

run_or_print() {
  if [[ "$dry_run" == "1" ]]; then
    printf '[DRY_RUN]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ ! -f "$gt_file" ]]; then
  if [[ "${GENERATE_GT:-0}" != "1" ]]; then
    echo "[ERROR] GT missing: $gt_file" >&2
    echo "        Set GENERATE_GT=1 or provide GT_FILE." >&2
    exit 2
  fi
  if [[ ! -x "$gt_app" ]]; then
    ensure_exe "$gt_app" compute_groundtruth
  fi
  echo "[RUN] generating GT -> $gt_file"
  run_or_print "$gt_app" \
    --data_type float --dist_fn L2 --scenario containment --K "$k" --num_threads "$num_threads" \
    --base_bin_file "$base_bin" \
    --base_label_file "$base_label" \
    --query_bin_file "$query_bin" \
    --query_label_file "$query_label" \
    --gt_file "$gt_file"
fi

extract_build_metric() {
  local log="$1"
  local build_csv="$2"
  local key="$3"
  if [[ -f "$build_csv" ]]; then
    case "$key" in
      index)
        awk -F, '$1=="index_time"{print $2}' "$build_csv" | tail -n1
        return
        ;;
      cross)
        awk -F, '$1=="build_cross_edges_time"{print $2}' "$build_csv" | tail -n1
        return
        ;;
      group)
        awk -F, '$1=="build_graph_time"{print $2}' "$build_csv" | tail -n1
        return
        ;;
    esac
  fi
  case "$key" in
    index)
      rg -o "Index time: ([0-9]+)ms" -r '$1' "$log" | tail -n1 || true
      ;;
    cross)
      awk '/Building cross-group edges/{f=1;next} f&&/- Finish in/{print $4; exit}' "$log" || true
      ;;
    group)
      rg -o "build_graph_time,([0-9.]+)" -r '$1' "$log" | tail -n1 || true
      ;;
  esac
}

optional_ung_env() {
  local name
  for name in \
    UNG_FAST_GRNND_LIGHT_PRUNE_NX \
    UNG_FAST_GRNND_LIGHT_HEAD \
    UNG_FAST_GRNND_LIGHT_REVERSE_CAP \
    UNG_FAST_GRNND_LIGHT_REVERSE_SLOTS \
    UNG_FAST_GRNND_LIGHT_REVERSE_FORWARD_CAP \
    UNG_FAST_GRNND_REPAIR_DEGREE \
    UNG_TAGORE_COMPACT_D2H \
    UNG_FAST_GRNND_FORWARD_CAP \
    UNG_FAST_GRNND_REVERSE_CAP \
    UNG_TAGORE_BATCH_STREAMS \
    UNG_TAGORE_FILL_FAST \
    UNG_TAGORE_FILL_THREADS \
    UNG_FAST_GRNND_BATCH_EXACT_NX \
    UNG_FAST_EXACT_DEVICE_LOOKUP \
    UNG_FAST_EXACT_PINNED_HOST \
    UNG_FAST_EXACT_ANCHOR_TAIL \
    UNG_ADAPTIVE_EXACT_MAX_NX \
    UNG_TAGORE_OVERLAP_FALLBACK \
    UNG_TAGORE_FALLBACK_THREADS \
    UNG_TAGORE_FALLBACK_IMPL \
    UNG_GRAPH_RESERVE_CAPACITY \
    UNG_GRAPH_RESERVE_ADDITIONAL_SLACK \
    UNG_GRAPH_RESERVE_HARD_CAP \
    UNG_ADDITIONAL_DIRECT_APPEND \
    UNG_GROUP_GRAPH_BOUNDED_COMPLETE \
    UNG_GROUP_GRAPH_COMPLETE_NX \
    UNG_GPU_SOURCE_EXACT \
    UNG_GPU_SOURCE_EXACT_MODE \
    UNG_GPU_SOURCE_EXACT_WARPS \
    UNG_UNIVERSAL_GPU \
    UNG_GPU_ID_VECTOR_WRITEBACK \
    UNG_GPU_FLAT_ID_WRITEBACK \
    UNG_GPU_DB_NOSPLIT \
    UNG_GPU_DB_GLOBAL_MERGE \
    UNG_GPU_DB_SPLIT_UNSUPPORTED \
    UNG_GPU_DB_LARGE_MAX_NX \
    UNG_GPU_DB_MEDIUM_MAX_NX \
    UNG_GPU_DB_CHUNK_QUERIES \
    UNG_GPU_PREPARE_DIRECT_HOSTREG \
    UNG_GPU_PREPARE_DIRECT_PAGEABLE \
    UNG_GPU_PREPARE_HOSTREG_MAX_MB; do
    if [[ -n "${!name:-}" ]]; then
      printf '%s=%s\n' "$name" "${!name}"
    fi
  done
}

variant_env() {
  local variant="$1"
  case "$variant" in
    cpu_vamana_group)
      cat <<EOF
UNG_BUILD_PROFILE=custom
UNG_GROUP_GRAPH_IMPL=0
UNG_GET_MIN_SUPER_SETS_IMPL=0
UNG_LNG_IMPL=0
UNG_DESCENDANTS_IMPL=0
UNG_COVERAGE_IMPL=1
UNG_COVERAGE_THREADS=128
UNG_CROSS_EDGE_IMPL=${cross_edge_impl}
UNG_ADDITIONAL_EDGES_IMPL=${additional_edges_impl}
UNG_GPU_TOPK_IMPL=${gpu_topk_impl}
UNG_CROSS_EDGE_GPU_STRICT=${cross_edge_gpu_strict}
EOF
      optional_ung_env
      ;;
    fastgrnnd_cpu_fallback)
      cat <<EOF
UNG_BUILD_PROFILE=custom
UNG_GROUP_GRAPH_IMPL=3
UNG_TAGORE_MIN_GROUP_SIZE=${UNG_TAGORE_MIN_GROUP_SIZE:-128}
UNG_TAGORE_K=${UNG_TAGORE_K:-64}
UNG_TAGORE_ITER=${UNG_TAGORE_ITER:-4}
UNG_TAGORE_M=${UNG_TAGORE_M:-64}
UNG_GET_MIN_SUPER_SETS_IMPL=0
UNG_LNG_IMPL=0
UNG_DESCENDANTS_IMPL=0
UNG_COVERAGE_IMPL=1
UNG_COVERAGE_THREADS=128
UNG_CROSS_EDGE_IMPL=${cross_edge_impl}
UNG_ADDITIONAL_EDGES_IMPL=${additional_edges_impl}
UNG_GPU_TOPK_IMPL=${gpu_topk_impl}
UNG_CROSS_EDGE_GPU_STRICT=${cross_edge_gpu_strict}
EOF
      optional_ung_env
      ;;
    fastgrnnd_complete_fallback)
      cat <<EOF
UNG_BUILD_PROFILE=custom
UNG_GROUP_GRAPH_IMPL=3
UNG_TAGORE_FALLBACK_IMPL=0
UNG_TAGORE_MIN_GROUP_SIZE=${UNG_TAGORE_MIN_GROUP_SIZE:-128}
UNG_TAGORE_K=${UNG_TAGORE_K:-64}
UNG_TAGORE_ITER=${UNG_TAGORE_ITER:-4}
UNG_TAGORE_M=${UNG_TAGORE_M:-64}
UNG_GET_MIN_SUPER_SETS_IMPL=0
UNG_LNG_IMPL=0
UNG_DESCENDANTS_IMPL=0
UNG_COVERAGE_IMPL=1
UNG_COVERAGE_THREADS=128
UNG_CROSS_EDGE_IMPL=${cross_edge_impl}
UNG_ADDITIONAL_EDGES_IMPL=${additional_edges_impl}
UNG_GPU_TOPK_IMPL=${gpu_topk_impl}
UNG_CROSS_EDGE_GPU_STRICT=${cross_edge_gpu_strict}
EOF
      optional_ung_env
      ;;
    adaptive_cuda)
      cat <<EOF
UNG_BUILD_PROFILE=custom
UNG_GROUP_GRAPH_IMPL=4
UNG_TAGORE_MIN_GROUP_SIZE=${UNG_TAGORE_MIN_GROUP_SIZE:-128}
UNG_ADAPTIVE_EXACT_MAX_NX=${UNG_ADAPTIVE_EXACT_MAX_NX:-512}
UNG_TAGORE_OVERLAP_FALLBACK=${UNG_TAGORE_OVERLAP_FALLBACK:-1}
UNG_TAGORE_FILL_FAST=${UNG_TAGORE_FILL_FAST:-1}
UNG_TAGORE_FILL_THREADS=${UNG_TAGORE_FILL_THREADS:-${num_threads}}
UNG_TAGORE_COMPACT_D2H=${UNG_TAGORE_COMPACT_D2H:-1}
UNG_TAGORE_K=${UNG_TAGORE_K:-64}
UNG_TAGORE_ITER=${UNG_TAGORE_ITER:-4}
UNG_TAGORE_M=${UNG_TAGORE_M:-64}
UNG_GET_MIN_SUPER_SETS_IMPL=0
UNG_LNG_IMPL=0
UNG_DESCENDANTS_IMPL=0
UNG_COVERAGE_IMPL=1
UNG_COVERAGE_THREADS=128
UNG_CROSS_EDGE_IMPL=${cross_edge_impl}
UNG_ADDITIONAL_EDGES_IMPL=${additional_edges_impl}
UNG_GPU_TOPK_IMPL=${gpu_topk_impl}
UNG_CROSS_EDGE_GPU_STRICT=${cross_edge_gpu_strict}
EOF
      optional_ung_env
      ;;
    *)
      echo "[ERROR] unknown variant: $variant" >&2
      exit 2
      ;;
  esac
}

run_variant() {
  local variant="$1"
  local vout="$out/$variant"
  local index_dir="$vout/index_files"
  local result_dir="$vout/results"
  local log="$vout/others/build.log"
  local search_log="$vout/others/search.log"
  mkdir -p "$index_dir" "$result_dir" "$vout/others"

  variant_env "$variant" > "$vout/others/env"

  echo "[RUN] build variant=$variant"
  if [[ "$dry_run" == "1" ]]; then
    env $(tr '\n' ' ' < "$vout/others/env") "$build_app" --help >/dev/null || true
    echo "[DRY_RUN] env $(tr '\n' ' ' < "$vout/others/env") $build_app ..." | tee "$log"
  else
    env $(tr '\n' ' ' < "$vout/others/env") "$build_app" \
      --data_type float --dataset "$DATASET" --dist_fn L2 --num_threads "$num_threads" \
      --max_degree "$max_degree" --Lbuild "$lbuild" --alpha "$alpha" --num_cross_edges "$num_cross_edges" \
      --base_bin_file "$base_bin" \
      --base_label_file "$base_label" \
      --base_label_info_file "$base_info" \
      --base_label_tree_roots "$tree_roots" \
      --index_path_prefix "$index_dir/" \
      --result_path_prefix "$result_dir/" \
      --scenario "$build_scenario" > "$log" 2>&1
  fi

  echo "[RUN] search variant=$variant"
  if [[ "$dry_run" == "1" ]]; then
    echo "[DRY_RUN] $search_app ..." | tee "$search_log"
  else
    "$search_app" \
      --data_type float --dataset "$DATASET" --dist_fn L2 --num_threads "$num_threads" \
      --K "$k" --num_repeats "$num_repeats" \
      --is_new_method true \
      --force_use_alg "${FORCE_USE_ALG:-0}" \
      --is_idea2_available "${IS_IDEA2_AVAILABLE:-false}" \
      --is_new_trie_method "${IS_NEW_TRIE_METHOD:-true}" \
      --is_rec_more_start "${IS_REC_MORE_START:-false}" \
      --base_bin_file "$base_bin" \
      --query_bin_file "$query_bin" \
      --query_label_file "$query_label" \
      --query_group_id_file "$query_group_ids" \
      --gt_file "$gt_file" \
      --index_path_prefix "$index_dir/" \
      --result_path_prefix "$result_dir/" \
      --acorn_index_path "$index_dir/../acorn_output/acorn.index" \
      --acorn_1_index_path "$index_dir/../acorn_output/acorn1.index" \
      --selector_modle_prefix "${SELECTOR_MODEL_PREFIX:-$out/SelectModels}" \
      --scenario "$search_scenario" \
      --num_entry_points "$num_entry_points" \
      --Lsearch $lsearch_values \
      --lsearch_start "${LSEARCH_START:-20}" \
      --lsearch_step "${LSEARCH_STEP:-30}" \
      --efs_start "${EFS_START:-100}" \
      --efs_step_slow "${EFS_STEP_SLOW:-50}" \
      --efs_step_fast "${EFS_STEP_FAST:-20}" \
      --lsearch_threshold "${LSEARCH_THRESHOLD:-100}" > "$search_log" 2>&1
  fi

  local index_ms group_ms cross_ms summary_file build_csv
  build_csv="$result_dir/build_time.csv"
  index_ms="$(extract_build_metric "$log" "$build_csv" index)"
  group_ms="$(extract_build_metric "$log" "$build_csv" group)"
  cross_ms="$(extract_build_metric "$log" "$build_csv" cross)"
  summary_file="$result_dir/search_time_summary.csv"
  if [[ -f "$summary_file" ]]; then
    tail -n +2 "$summary_file" | while IFS=, read -r lsearch avg_efs avg_time avg_recall; do
      echo "$variant,${index_ms:-NA},${group_ms:-NA},${cross_ms:-NA},$lsearch,$avg_efs,$avg_time,$avg_recall,$summary_file,$log" >> "$summary_csv"
    done
  else
    echo "$variant,${index_ms:-NA},${group_ms:-NA},${cross_ms:-NA},NA,NA,NA,NA,$summary_file,$log" >> "$summary_csv"
  fi
}

variants="${VARIANTS:-cpu_vamana_group fastgrnnd_cpu_fallback fastgrnnd_complete_fallback}"
for variant in $variants; do
  run_variant "$variant"
done

echo "[OK] wrote $out"
echo "[OK] summary: $summary_csv"
