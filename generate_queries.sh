#!/bin/bash

# ==============================================================================
# generate_queries.sh - 独立的查询数据生成脚本
# 作用: 读取指定的JSON配置文件，并根据其内容执行一个或多个查询生成任务。
#       支持多种生成模式 (generate, sub_base, weighted_sub_base, analyze)。
# ==============================================================================

set -e # 如果任何命令失败，则立即退出

# --- 检查jq是否安装 ---
if ! command -v jq &> /dev/null; then
    echo "错误: jq 未安装。请先安装 jq (https://stedolan.github.io/jq/)"
    exit 1
fi

# --- 检查命令行参数 ---
CONFIG_FILE="$1"
BUILD_DIR="$2"

if [ -z "$CONFIG_FILE" ] || [ -z "$BUILD_DIR" ]; then
    echo "错误: 参数不足。"
    echo "用法: $0 /path/to/config.json /path/to/build_dir"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo "使用配置文件: $CONFIG_FILE"
echo "使用构建目录: $BUILD_DIR"

# ==============================================================================
# 核心逻辑: 遍历JSON文件中的所有任务
# ==============================================================================
cat "$CONFIG_FILE" | jq -c '.query_tasks[]' | while read -r task; do
    
    # --- 提取任务参数 ---
    ENABLED=$(echo "$task" | jq -r '.enabled')
    TASK_NAME=$(echo "$task" | jq -r '.task_name')

    echo -e "\n=========================================================="
    echo "正在处理任务: $TASK_NAME"
    
    if [[ "$ENABLED" != "true" ]]; then
        echo "任务被禁用，跳过。"
        continue
    fi

    # - 通用参数
    MODE=$(echo "$task" | jq -r '.mode // "generate"')
    DATASET=$(echo "$task" | jq -r '.dataset')
    DATA_DIR=$(echo "$task" | jq -r '.data_dir')
    OVERWRITE=$(echo "$task" | jq -r '.overwrite')
    
    # --- 构造文件路径 ---
    QUERY_DIR="$DATA_DIR/query_${TASK_NAME}"
    QUERY_VECTORS_FILE="$QUERY_DIR/${DATASET}_query.fvecs"
    QUERY_LABELS_FILE="$QUERY_DIR/${DATASET}_query_labels.txt"
    
    BASE_LABELS_FILE="$DATA_DIR/${DATASET}_base_labels.txt"
    BASE_VECTORS_FILE="$DATA_DIR/${DATASET}_base.fvecs"

    # --- 编译检查和自动编译 ---
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    EXECUTABLE_PATH="$BUILD_DIR/tools/generate_mixed_queries"
    
    if [ ! -f "$EXECUTABLE_PATH" ]; then
        echo "可执行文件不存在，开始编译..."
        mkdir -p "$BUILD_DIR"
        SOURCE_CODE_DIR="${SCRIPT_DIR}/UNG/codes" # 此脚本与 UNG/ACORN 目录同级
        if [ ! -d "$SOURCE_CODE_DIR" ]; then
            # 如果不是，可以做一个兼容性调整
            SOURCE_CODE_DIR="${SCRIPT_DIR}/codes" # 兼容旧的"codes"目录结构
        fi
        echo "使用源码目录: $SOURCE_CODE_DIR"
        cmake -S "$SOURCE_CODE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
        make -C "$BUILD_DIR" -j generate_mixed_queries
    else
        echo "编译产物已存在，跳过编译步骤。"
    fi

    # --- 检查是否跳过 ---
    if [[ "$OVERWRITE" != "true" ]] && [ -f "$QUERY_VECTORS_FILE" ]; then
        echo "查询文件 '$QUERY_VECTORS_FILE' 已存在且不允许覆盖，跳过任务。"
        continue
    fi
    
    echo "开始生成查询数据..."
    mkdir -p "$QUERY_DIR"

    # --- 检查依赖文件 ---
    if [ ! -f "$BASE_LABELS_FILE" ]; then echo "错误: Base labels 文件不存在: $BASE_LABELS_FILE"; exit 1; fi
    if [ ! -f "$BASE_VECTORS_FILE" ]; then echo "错误: Base vectors 文件不存在: $BASE_VECTORS_FILE"; exit 1; fi
    if [ ! -f "$EXECUTABLE_PATH" ]; then echo "错误: 可执行文件不存在: $EXECUTABLE_PATH"; exit 1; fi

    # --- 根据模式构建并执行命令 ---
    echo "任务模式: $MODE"
    
    CMD_BASE=(
        "$EXECUTABLE_PATH"
        --input_file "$BASE_LABELS_FILE"
        --output_file "$QUERY_LABELS_FILE"
        --base_vectors_file "$BASE_VECTORS_FILE"
        --output_vectors_file "$QUERY_VECTORS_FILE"
    )

    if [[ "$MODE" == "sub_base" ]]; then
        PARAMS=$(echo "$task" | jq -r '.sub_base_params')
        NUM_POINTS=$(echo "$PARAMS" | jq -r '.num_points')
        QUERY_LENGTH=$(echo "$PARAMS" | jq -r '.query_length')
        K_VAL=$(echo "$PARAMS" | jq -r '.K')
        MAX_COVERAGE=$(echo "$PARAMS" | jq -r '.max_coverage')
        MIN_CHILDREN=$(echo "$PARAMS" | jq -r '.min_children')
        CACHE_FILE=$(echo "$PARAMS" | jq -r '.["cache-file"] // ""') 

        CMD=(
            "${CMD_BASE[@]}" --mode sub_base
            --num_points "$NUM_POINTS"
            --query-length "$QUERY_LENGTH"
            --K "$K_VAL"
            --max-coverage "$MAX_COVERAGE"
            --min-children "$MIN_CHILDREN"
        )

        if [ -n "$CACHE_FILE" ]; then
            CMD+=("--cache-file" "$CACHE_FILE")
        fi

        echo "执行 sub_base 模式命令..."
        "${CMD[@]}"

    elif [[ "$MODE" == "weighted_sub_base" ]]; then
        # weighted_sub_base 使用与 sub_base 完全相同的参数块
        PARAMS=$(echo "$task" | jq -r '.sub_base_params')
        NUM_POINTS=$(echo "$PARAMS" | jq -r '.num_points')
        QUERY_LENGTH=$(echo "$PARAMS" | jq -r '.query_length')
        K_VAL=$(echo "$PARAMS" | jq -r '.K')
        MAX_COVERAGE=$(echo "$PARAMS" | jq -r '.max_coverage')
        MIN_CHILDREN=$(echo "$PARAMS" | jq -r '.min_children')
        CACHE_FILE=$(echo "$PARAMS" | jq -r '.["cache-file"] // ""')

        # 唯一的区别是 --mode 参数
        CMD=(
            "${CMD_BASE[@]}" --mode weighted_sub_base
            --num_points "$NUM_POINTS"
            --query-length "$QUERY_LENGTH"
            --K "$K_VAL"
            --max-coverage "$MAX_COVERAGE"
            --min-children "$MIN_CHILDREN"
        )

        if [ -n "$CACHE_FILE" ]; then
            CMD+=("--cache-file" "$CACHE_FILE")
        fi

        echo "执行 weighted_sub_base 模式命令..."
        "${CMD[@]}"

    elif [[ "$MODE" == "generate" ]]; then
        PARAMS=$(echo "$task" | jq -r '.generation_params')
        NUM_POINTS=$(echo "$PARAMS" | jq -r '.num_points')
        K=$(echo "$PARAMS" | jq -r '.K')
        DIST_TYPE=$(echo "$PARAMS" | jq -r '.distribution_type')
        TRUNCATE=$(echo "$PARAMS" | jq -r '.truncate_to_fixed_length')
        LABELS_PER_QUERY=$(echo "$PARAMS" | jq -r '.num_labels_per_query')
        EXPECTED_LABEL=$(echo "$PARAMS" | jq -r '.expected_num_label')

        echo "执行 generate 模式命令..."
        "${CMD_BASE[@]}" --mode generate \
            --num_points "$NUM_POINTS" \
            --K "$K" \
            --distribution_type "$DIST_TYPE" \
            --truncate_to_fixed_length "$TRUNCATE" \
            --num_labels_per_query "$LABELS_PER_QUERY" \
            --expected_num_label "$EXPECTED_LABEL"
            
    else
        echo "错误: 未知的任务模式 '$MODE' (非 generate, sub_base, 或 weighted_sub_base)。"
        exit 1
    fi

    echo "查询数据生成成功 -> $QUERY_VECTORS_FILE"

    # --- 执行分析命令 ---
    ANALYZE=$(echo "$task" | jq -r '.analysis_params.analyze // false')
    if [[ "$ANALYZE" == "true" ]]; then
        echo "开始分析生成的查询..."
        PROFILE_OUTPUT_FILE="$QUERY_DIR/profiled_${TASK_NAME}.csv"
        "$EXECUTABLE_PATH" --mode analyze \
            --input_file "$BASE_LABELS_FILE" \
            --candidate_file "$QUERY_LABELS_FILE" \
            --profiled_output "$PROFILE_OUTPUT_FILE"
        echo "分析完成 -> $PROFILE_OUTPUT_FILE"
    fi

done

echo -e "\n所有已启用的查询生成任务处理完毕！"