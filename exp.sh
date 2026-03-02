#!/bin/bash

# ==============================================================================
# exp.sh - 实验流程主控脚本 (v2.0)
# 核心逻辑:
# 遍历【所有】实验配置, 为每个配置:
#   a. 提取【全部】参数。
#   b. 调用 build_hybrid.sh (该脚本内部负责编译、数据转换、创建索引)。
#   c. 调用 generate_gt.sh 创建或确认GT文件存在。
#   d. 调用 search.sh 在对应的索引和GT上执行搜索。
# ==============================================================================

set -e # 如果任何命令失败，则立即退出

# --- 检查jq是否安装 ---
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install first. jq (https://stedolan.github.io/jq/)"
    exit 1
fi

CONFIG_FILE="experiments.json"

cat "$CONFIG_FILE" | jq -c '.experiments[]' | while read -r experiment; do
    echo -e "\n=========================================================="
    echo "Process new experiment-config..."
    echo "=========================================================="

    # --- 【步骤1】为当前实验提取【所有】参数 ---
    
    # - 通用参数
    DATASET=$(echo "$experiment" | jq -r '.dataset')
    DATA_DIR=$(echo "$experiment" | jq -r '.data_dir')
    BASE_OUTPUT_DIR=$(echo "$experiment" | jq -r '.output_dir')
    BUILD_MODE=$(echo "$experiment" | jq -r '.build_mode')
    

    # - 构建参数
    MAX_DEGREE=$(echo "$experiment" | jq -r '.max_degree')
    LBUILD=$(echo "$experiment" | jq -r '.Lbuild')
    ALPHA=$(echo "$experiment" | jq -r '.alpha')
    NUM_CROSS_EDGES=$(echo "$experiment" | jq -r '.num_cross_edges')
    NUM_ENTRY_POINTS=$(echo "$experiment" | jq -r '.num_entry_points')


    # - 搜索 & GT 参数
    K=$(echo "$experiment" | jq -r '.K')
    LSEARCH_START=$(echo "$experiment" | jq -r '.Lsearch_start')
    LSEARCH_END=$(echo "$experiment" | jq -r '.Lsearch_end')
    LSEARCH_STEP=$(echo "$experiment" | jq -r '.Lsearch_step')
    NUM_THREADS=$(echo "$experiment" | jq -r '.num_threads')
    NUM_REPEATS=$(echo "$experiment" | jq -r '.num_repeats')
    IS_NEW_TRIE_METHOD=$(echo "$experiment" | jq -r '.trie_param.is_new_trie_method')
    IS_REC_MORE_START=$(echo "$experiment" | jq -r '.trie_param.is_rec_more_start')
    IS_IDEA2_AVAILABLE=$(echo "$experiment" | jq -r '.is_idea2_available') 
    FORCE_USE_ALG=$(echo "$experiment" | jq -r '.force_use_alg')
    QUERY_DIR_NAME=$(echo "$experiment" | jq -r '.query_dir_name')

    # 新增: 提取 ACORN 参数
    ACORN_N=$(echo "$experiment" | jq -r '.acorn_params.N')
    ACORN_M=$(echo "$experiment" | jq -r '.acorn_params.M')
    ACORN_M_BETA=$(echo "$experiment" | jq -r '.acorn_params.M_beta')
    ACORN_GAMMA=$(echo "$experiment" | jq -r '.acorn_params.gamma')
    ACORN_EFS_START=$(echo "$experiment" | jq -r '.acorn_params.acorn_efs_start')
    ACORN_EFS_STEP_SLOW=$(echo "$experiment" | jq -r '.acorn_params.acorn_efs_step_slow')
    ACORN_EFS_STEP_FAST=$(echo "$experiment" | jq -r '.acorn_params.acorn_efs_step_fast')
    LSEARCH_THRESHOLD=$(echo "$experiment" | jq -r '.acorn_params.lsearch_threshold')

    export UNG_BUILD_DIR="/data/fxy/FilterVector/build_para/ung"
    export ACORN_BUILD_DIR="/data/fxy/FilterVector/build_para/acorn"
    export SCRIPT_DIR=$(pwd) # 假设脚本在当前目录

    # --- 根据开关参数确定算法名称 ---
    ALGORITHM_NAME=""
    if [[ "$FORCE_USE_ALG" == "1" ]]; then
        ALGORITHM_NAME="UNG-nTfalse"
    elif [[ "$FORCE_USE_ALG" == "2" ]]; then
        ALGORITHM_NAME="UNG-nTtrue"
    elif [[ "$FORCE_USE_ALG" == "3" ]]; then
        ALGORITHM_NAME="ACORN-gamma"
    elif [[ "$FORCE_USE_ALG" == "4" ]]; then
        ALGORITHM_NAME="ACORN-1"
    elif [[ "$IS_NEW_TRIE_METHOD" == "true" && "$IS_IDEA2_AVAILABLE" == "true" && "$FORCE_USE_ALG" == "0" ]]; then
        ALGORITHM_NAME="method3"
    elif [[ "$IS_NEW_TRIE_METHOD" == "false" && "$IS_IDEA2_AVAILABLE" == "true" && "$FORCE_USE_ALG" == "0" ]]; then
        ALGORITHM_NAME="method2"
    elif [[ "$IS_NEW_TRIE_METHOD" == "true" && "$IS_IDEA2_AVAILABLE" == "false" && "$FORCE_USE_ALG" == "0" ]]; then
        ALGORITHM_NAME="method1"
    else
        echo "Error: Unknown algorithm combination. Please check the switch parameters in the JSON configuration!"
        exit 1
    fi
    echo "Detection algorithm: $ALGORITHM_NAME"

    SHARED_DATASET_DIR="${BASE_OUTPUT_DIR}/${DATASET}"
    ALGO_RESULT_DIR="${SHARED_DATASET_DIR}/Results/${ALGORITHM_NAME}"

    # --- 【步骤2】调用 build_hybrid.sh ---
    # build_hybrid.sh 内部会负责编译、数据转换和索引构建
    echo "Preparing build index..."

    ./build_hybrid.sh \
        --build_mode "$BUILD_MODE" \
        --query_dir_name "$QUERY_DIR_NAME" \
        --dataset "$DATASET" --data_dir "$DATA_DIR" --exp_output_dir "$SHARED_DATASET_DIR" \
        --max_degree "$MAX_DEGREE" --Lbuild "$LBUILD" --alpha "$ALPHA" \
        --num_cross_edges "$NUM_CROSS_EDGES" --num_entry_points "$NUM_ENTRY_POINTS" \
        --acorn_n "$ACORN_N" --acorn_m "$ACORN_M" --acorn_m_beta "$ACORN_M_BETA" --acorn_gamma "$ACORN_GAMMA"
    
    # 新增判断：如果 build_mode 是 'parallel'，则构建任务已完成，直接跳过 GT 生成和搜索，进入下一个实验。
    if [ "$BUILD_MODE" == "parallel" ]; then
        echo "[INFO] Build mode is 'parallel'. Skipping GT generation and search steps."
        echo "--- The current experimental configuration processing has been completed (BUILD ONLY) ---"
        continue
    fi
    
    # --- 【步骤3】调用 generate_gt.sh ---
    echo "Preparing Ground Truth (K=$K)..."
    ./generate_gt.sh \
        --dataset "$DATASET" --data_dir "$DATA_DIR" --exp_output_dir "$SHARED_DATASET_DIR" --build_dir "$UNG_BUILD_DIR" \
        --query_dir_name "$QUERY_DIR_NAME" \
        --K "$K"

    # --- 【步骤4】调用 search.sh ---
    INDEX_DIR_NAME="M${MAX_DEGREE}_LB${LBUILD}_alpha${ALPHA}_C${NUM_CROSS_EDGES}_EP${NUM_ENTRY_POINTS}_AN${ACORN_N}_AM${ACORN_M}_AMB${ACORN_M_BETA}_AG${ACORN_GAMMA}"
    echo "Begin search (K=$K)..."
    ./search.sh \
        --dataset "$DATASET" --data_dir "$DATA_DIR" \
        --query_dir_name "$QUERY_DIR_NAME" \
        --shared_output_dir "$SHARED_DATASET_DIR" \
        --algo_result_dir "$ALGO_RESULT_DIR" \
        --build_dir "$UNG_BUILD_DIR" \
        --index_dir_name "$INDEX_DIR_NAME" \
        --build_mode "$BUILD_MODE" \
        --num_entry_points "$NUM_ENTRY_POINTS" \
        --Lsearch_start "$LSEARCH_START" --Lsearch_end "$LSEARCH_END" --Lsearch_step "$LSEARCH_STEP" \
        --num_threads "$NUM_THREADS" --K "$K" --num_repeats "$NUM_REPEATS" \
        --is_new_trie_method "$IS_NEW_TRIE_METHOD" --is_rec_more_start "$IS_REC_MORE_START" \
        --is_idea2_available "$IS_IDEA2_AVAILABLE" --force_use_alg "$FORCE_USE_ALG" \
        --efs_start "$ACORN_EFS_START" \
        --efs_step_slow "$ACORN_EFS_STEP_SLOW" --efs_step_fast "$ACORN_EFS_STEP_FAST" --lsearch_threshold "$LSEARCH_THRESHOLD"
         
    echo "--- The current experimental configuration processing has been completed ---"
done

echo -e "\n All experiments have been completed!"