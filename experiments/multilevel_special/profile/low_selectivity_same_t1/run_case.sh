#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <output-dir> <query-dir> <block-index> <lsearch> <single|multi>" >&2
  exit 2
fi

output_dir=$1
query_dir=$2
block_index=$3
lsearch=$4
structure=$5
repo=/home/sunyahui/worktrees/FilterVectorCode_multilevel_special
binary=$repo/build_ung_rel/apps/search_UNG_index

mkdir -p "$output_dir"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

export CUDA_VISIBLE_DEVICES=0
export UNG_DISABLE_ENTRY_ROUTE_STATS=1
export UNG_SPECIAL_BLOCK_SEARCH=1
export UNG_SPECIAL_LIGHT_STATS=${PROFILE_LIGHT_STATS:-0}
export UNG_SPECIAL_PROFILE_TIMING=1
export UNG_VALIDATE_FILTER_RESULTS=1
unset UNG_SPECIAL_FREE_GPU_DISTANCE
unset UNG_SPECIAL_BATCH_GPU_SEARCH
if [[ $structure == single ]]; then
  export UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE=1
else
  unset UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE
fi

exec "$binary" \
  --data_type float --dataset Amazon --dist_fn L2 --num_threads 100 --K 10 \
  --num_repeats "${PROFILE_NUM_REPEATS:-2}" --is_new_method true --force_use_alg 1 \
  --is_idea2_available false --is_new_trie_method false \
  --is_rec_more_start false --is_ung_more_entry false \
  --base_bin_file /home/graphdb/FilterVectorData/Amazon/Amazon_base.bin \
  --query_bin_file /home/graphdb/FilterVectorData/Amazon/"$query_dir"/Amazon_query.bin \
  --query_label_file /home/graphdb/FilterVectorData/Amazon/"$query_dir"/Amazon_query_labels.txt \
  --query_group_id_file "$output_dir"/missing_query_source_groups.txt \
  --gt_file /home/graphdb/FilterVectorResult/Amazon/GroundTruth/"$query_dir"/Amazon_gt_labels_containment.bin \
  --index_path_prefix /home/graphdb/FilterVectorResult/Amazon/index/Trie_block/index_files/ \
  --result_path_prefix "$output_dir"/ --selector_model_prefix /nonexistent \
  --scenario containment --num_entry_points 16 --Lsearch "$lsearch" \
  --lsearch_start "$lsearch" --lsearch_step 1 --efs_start 10 \
  --efs_step_slow 10 --efs_step_fast 10 --lsearch_threshold "$lsearch" \
  --entry_group_provider cpu_bruteforce_els --graph_search_backend neighbor_list \
  --skip_query_features true --skip_bitmap_comparison true \
  --block_index_path_prefix "$block_index"/
