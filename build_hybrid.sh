#!/bin/bash

# ==============================================================================
# build_hybrid.sh - 统一构建 UNG 和 ACORN 索引 (v2.0)
#
# 功能:
# 1. 编译 UNG 和 ACORN 的代码
# 2. 检查并转换 fvecs -> bin 数据格式
# 3. 根据 build_mode (parallel, serial, ung_only, acorn_only) 构建索引
# ==============================================================================

set -e # 如果任何命令失败，则立即退出

# --- Step 1: 解析命令行参数 ---
PARAMS=("$@")
while [[ $# -gt 0 ]]; do
    if [[ $1 == --* ]]; then
        key=$(echo "$1" | sed 's/--//' | tr '[:lower:]-' '[:upper:]_')
        
        # 处理模式参数
        if [[ $key == "BUILD_MODE" ]]; then
            BUILD_MODE="$2"
            shift 2
            continue
        fi

        # 处理其他参数
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

# 验证构建模式
case "$BUILD_MODE" in
    parallel|serial|ung_only|acorn_only)
        echo "[INFO] Build mode set to: $BUILD_MODE"
        ;;
    *)
        echo "错误: 无效的 build_mode '$BUILD_MODE'。可用选项: parallel, serial, ung_only, acorn_only"
        exit 1
        ;;
esac

# --- Step 2: 编译代码 ---
# UNG 编译
UNG_EXECUTABLE="${UNG_BUILD_DIR}/apps/build_UNG_index"
if [ ! -f "$UNG_EXECUTABLE" ]; then
    echo "[INFO] UNG executable not found. Compiling..."
    mkdir -p "$UNG_BUILD_DIR"
    cmake -S "${SCRIPT_DIR}/UNG/codes" -B "$UNG_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    make -C "$UNG_BUILD_DIR" -j
else
    echo "[INFO] UNG executable found."
fi
# ACORN 编译
ACORN_EXECUTABLE="${ACORN_BUILD_DIR}/demos/test_acorn"
if [ ! -f "$ACORN_EXECUTABLE" ]; then
    echo "[INFO] ACORN executable not found. Compiling..."
    mkdir -p "$ACORN_BUILD_DIR"
    cmake -S "${SCRIPT_DIR}/ACORN" -B "$ACORN_BUILD_DIR" -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
    make -C "$ACORN_BUILD_DIR" -j test_acorn
else
    echo "[INFO] ACORN executable found."
fi

# --- Step 2.5: 检查并转换数据格式 ---
echo "[INFO] Checking and converting data format (if necessary)..."
FVECS_TO_BIN_TOOL="${UNG_BUILD_DIR}/tools/fvecs_to_bin"
BASE_FVECS_FILE="${DATA_DIR}/${DATASET}_base.fvecs"
BASE_BIN_FILE="${DATA_DIR}/${DATASET}_base.bin"

if [ ! -x "$FVECS_TO_BIN_TOOL" ]; then
    echo "Error: Conversion tool not found after compilation at: ${FVECS_TO_BIN_TOOL}"
    exit 1
fi

if [ ! -f "$BASE_BIN_FILE" ]; then
    if [ -f "$BASE_FVECS_FILE" ]; then
        echo "Target file '${BASE_BIN_FILE}' not found. Starting conversion from '${BASE_FVECS_FILE}'..."
        "$FVECS_TO_BIN_TOOL" --data_type float --input_file "$BASE_FVECS_FILE" --output_file "$BASE_BIN_FILE"
        echo "Conversion complete."
    else
        echo "Warning: Source file '${BASE_FVECS_FILE}' not found. Skipping conversion. Build process might fail if base .bin is also missing."
    fi
else
    echo "Target file '${BASE_BIN_FILE}' already exists. Skipping conversion."
fi


# --- Step 3: 构造与 search.sh 兼容的输出目录 ---
# 根据构建模式确定基础目录名
if [ "$BUILD_MODE" == "parallel" ]; then
    INDEX_BASE_DIR="Index_parallel"
else
    INDEX_BASE_DIR="Index"
fi
echo "[INFO] Index base directory set to: $INDEX_BASE_DIR"
INDEX_DIR_NAME="M${MAX_DEGREE}_LB${LBUILD}_alpha${ALPHA}_C${NUM_CROSS_EDGES}_EP${NUM_ENTRY_POINTS}_AN${ACORN_N}_AM${ACORN_M}_AMB${ACORN_M_BETA}_AG${ACORN_GAMMA}"
INDEX_OUTPUT_DIR="${EXP_OUTPUT_DIR}/${INDEX_BASE_DIR}/${INDEX_DIR_NAME}"


# 如果索引已存在且模式不是 only 模式，则可以考虑跳过
if [ -d "$INDEX_OUTPUT_DIR/index_files" ] && [ -d "$INDEX_OUTPUT_DIR/acorn_output" ] && [ "$BUILD_MODE" != "ung_only" ] && [ "$BUILD_MODE" != "acorn_only" ]; then
    echo "[INFO] 索引目录 '$INDEX_OUTPUT_DIR' 已完整存在，跳过构建。"
    exit 0
fi

echo "[INFO] Preparing index directory: $INDEX_OUTPUT_DIR"
mkdir -p "$INDEX_OUTPUT_DIR/index_files"
mkdir -p "$INDEX_OUTPUT_DIR/acorn_output"
mkdir -p "$INDEX_OUTPUT_DIR/others"

# --- Step 4: 定义构建函数 ---
# 将具体的构建命令封装成函数，便于调用

build_ung() {
    echo "[UNG] Build process started."
    "$UNG_EXECUTABLE" \
        --data_type float --dist_fn L2 --num_threads 32 \
        --max_degree "$MAX_DEGREE" --Lbuild "$LBUILD" --alpha "$ALPHA" --num_cross_edges "$NUM_CROSS_EDGES" \
        --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
        --base_label_file "$DATA_DIR/${DATASET}_base_labels.txt" \
        --base_label_info_file "$DATA_DIR/${DATASET}_base_labels_info.log" \
        --base_label_tree_roots "$DATA_DIR/tree_roots.txt" \
        --index_path_prefix "$INDEX_OUTPUT_DIR/index_files/" \
        --result_path_prefix "$INDEX_OUTPUT_DIR/results/" \
        --scenario general --dataset "$DATASET" \
        > "$INDEX_OUTPUT_DIR/others/ung_build.log" 2>&1
    echo "[UNG] Build process finished."
}

build_acorn() {
    echo "[ACORN] Build process started."
    # --- 根据 BUILD_MODE 决定基础向量文件路径 ---
    local acorn_base_fvecs_path
    local acorn_base_label_path
    if [[ "$BUILD_MODE" == "serial" ]]; then
        acorn_base_fvecs_path="${INDEX_OUTPUT_DIR}/index_files/reordered_vecs.fvecs"
        acorn_base_label_path="${INDEX_OUTPUT_DIR}/index_files/reordered_labels.txt"
        echo "[ACORN] Serial mode detected. Using reordered base vectors: $acorn_base_fvecs_path"
        echo "[ACORN] Serial mode detected. Using reordered base labelss: $acorn_base_label_path"
    else
        acorn_base_fvecs_path="${DATA_DIR}/${DATASET}_base.fvecs"
        acorn_base_label_path="${DATA_DIR}/${DATASET}_base_labels.txt"
        echo "[ACORN] Non-serial mode. Using original base vectors: $acorn_base_fvecs_path"
        echo "[ACORN] Non-serial mode. Using original base labels: $acorn_base_label_path"
    fi

    "$ACORN_EXECUTABLE" build \
        "$ACORN_N" "$ACORN_GAMMA" "$DATASET" "$ACORN_M" "$ACORN_M_BETA" \
        "$acorn_base_fvecs_path" "$acorn_base_label_path" "$query_path" \
        "dummy_csv_dir" "dummy_avg_csv_dir" "dummy_dis_path" \
        32 1 true "10" \
        "$INDEX_OUTPUT_DIR/acorn_output/acorn.index" \
        "$INDEX_OUTPUT_DIR/acorn_output/acorn1.index" \
        0 \
        > "$INDEX_OUTPUT_DIR/others/acorn_build.log" 2>&1
    echo "[ACORN] Build process finished."
}


# --- Step 5: 根据构建模式执行任务 ---
start_time=$(date +%s)

if [ "$BUILD_MODE" == "parallel" ]; then
    echo "--- [Executing in PARALLEL mode] ---"
    build_ung &
    ung_pid=$!
    build_acorn &
    acorn_pid=$!
    wait $ung_pid
    wait $acorn_pid
elif [ "$BUILD_MODE" == "serial" ]; then
    echo "--- [Executing in SERIAL mode] ---"
    build_ung
    build_acorn
elif [ "$BUILD_MODE" == "ung_only" ]; then
    echo "--- [Executing in UNG_ONLY mode] ---"
    build_ung
elif [ "$BUILD_MODE" == "acorn_only" ]; then
    echo "--- [Executing in ACORN_ONLY mode] ---"
    build_acorn
fi

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "--- Build Summary ---"
echo "[SUCCESS] All build tasks complete. Total time: $duration seconds."
echo "Indexes are saved in: $INDEX_OUTPUT_DIR"