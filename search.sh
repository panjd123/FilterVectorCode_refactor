#!/bin/bash

# ==============================================================================
# search.sh - 负责在已有的索引和GT上执行搜索任务
# ==============================================================================

set -e # 如果任何命令失败，则立即退出

# --- Step 1: 解析命令行参数 ---
while [[ $# -gt 0 ]]; do
    if [[ $1 == --* ]]; then
        key=$(echo "$1" | sed 's/--//' | tr '[:lower:]-' '[:upper:]_')
        if [[ $key == "QUERY_DIR_NAME" ]]; then
            QUERY_DIR_NAME="$2"
            shift 2
            continue
        fi
        if [ -z "$2" ]; then
            echo "错误: 参数 $1 缺少值"
            exit 1
        fi
        declare "$key"="$2"
        shift 2
    else
        echo "未知参数: $1"; exit 1
    fi
done

# --- Step 2: 根据搜索参数构造唯一的结果输出目录 ---
SAFE_QUERY_NAME=$(echo "$QUERY_DIR_NAME" | tr '/' '_')
GT_DIR_NAME="GT_${SAFE_QUERY_NAME}_K${K}"
SEARCH_DIR_NAME="Ls${LSEARCH_START}-Le${LSEARCH_END}-Lp${LSEARCH_STEP}_efsS${EFS_START}-efss${EFS_STEP_SLOW}-efsf${EFS_STEP_FAST}-lt${LSEARCH_THRESHOLD}_K${K}_th${NUM_THREADS}"
RESULT_OUTPUT_DIR="${ALGO_RESULT_DIR}/Index[${INDEX_DIR_NAME}]_GT[${GT_DIR_NAME}]_Search[${SEARCH_DIR_NAME}]"

# --- Step 3: 创建结果目录 ---
mkdir -p "$RESULT_OUTPUT_DIR/results"
mkdir -p "$RESULT_OUTPUT_DIR/others"

# --- Step 4: 准备Lsearch参数序列 ---
LSEARCH_VALUES=$(seq "$LSEARCH_START" "$LSEARCH_STEP" "$LSEARCH_END" | tr '\n' ' ')
echo "将在以下Lsearch值上进行测试: $LSEARCH_VALUES"

# --- Step 5: 定义依赖文件和目录的路径 ---
# 根据构建模式确定索引基础目录
if [[ "$BUILD_MODE" == "parallel" ]]; then
    INDEX_BASE_DIR="Index_parallel"
else
    INDEX_BASE_DIR="Index"
fi
INDEX_PATH="${SHARED_OUTPUT_DIR}/${INDEX_BASE_DIR}/${INDEX_DIR_NAME}"
GT_PATH="${SHARED_OUTPUT_DIR}/GroundTruth/${GT_DIR_NAME}"
MODEL_PATH="${SHARED_OUTPUT_DIR}/SelectModels"
ACORN_INDEX_PREFIX="${INDEX_PATH}/acorn_output/"

QUERY_DIR="${DATA_DIR}/${QUERY_DIR_NAME}"
echo "Using query directory from: $QUERY_DIR"

ACORN_INDEX_PREFIX="${INDEX_PATH}/acorn_output/"

echo "使用索引: $INDEX_PATH"
echo "使用GT: $GT_PATH"
echo "结果将保存到: $RESULT_OUTPUT_DIR"

# --- Step 6: 执行搜索 ---
"$BUILD_DIR"/apps/search_UNG_index \
    --data_type float  --dataset "$DATASET" --dist_fn L2 --num_threads "$NUM_THREADS" --K "$K" --num_repeats "$NUM_REPEATS" \
    --is_new_method true \
    --force_use_alg "$FORCE_USE_ALG" \
    --is_idea2_available "$IS_IDEA2_AVAILABLE" \
    --is_new_trie_method "$IS_NEW_TRIE_METHOD" --is_rec_more_start "$IS_REC_MORE_START" \
    --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
    --query_bin_file "$QUERY_DIR/${DATASET}_query.bin" \
    --query_label_file "$QUERY_DIR/${DATASET}_query_labels.txt" \
    --query_group_id_file "$QUERY_DIR/${DATASET}_query_source_groups.txt" \
    --gt_file "$GT_PATH/${DATASET}_gt_labels_containment.bin" \
    --index_path_prefix "$INDEX_PATH/index_files/" \
    --result_path_prefix "$RESULT_OUTPUT_DIR/results/" \
    --acorn_index_path "$ACORN_INDEX_PREFIX/acorn.index" --acorn_1_index_path "$ACORN_INDEX_PREFIX/acorn1.index" \
    --selector_modle_prefix "${MODEL_PATH}" \
    --scenario containment \
    --num_entry_points "$NUM_ENTRY_POINTS" \
    --Lsearch $LSEARCH_VALUES \
    --lsearch_start "$LSEARCH_START" \
    --lsearch_step "$LSEARCH_STEP" \
    --efs_start "$EFS_START" \
    --efs_step_slow "$EFS_STEP_SLOW" --efs_step_fast "$EFS_STEP_FAST" --lsearch_threshold "$LSEARCH_THRESHOLD"> "$RESULT_OUTPUT_DIR/others/${DATASET}_search_output.txt" 2>&1

echo "搜索完成！"
