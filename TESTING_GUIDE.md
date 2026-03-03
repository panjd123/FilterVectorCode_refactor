# 测试文档（FilterVectorCode_refactor）

本文档用于统一“怎么测”，覆盖功能正确性、GPU/CPU对照、性能A/B回归。  
推荐优先使用脚本：`scripts/run_ung_tests.sh`。

## 0. 一键脚本（推荐）

脚本路径：`scripts/run_ung_tests.sh`

```bash
cd /home/graphdb/FilterVectorCode_refactor
scripts/run_ung_tests.sh --help
```

常用命令：

```bash
# 单次 GPU 构建测试（含日志摘要）
scripts/run_ung_tests.sh gpu --label quick_gpu

# 单次 CPU 对照测试
scripts/run_ung_tests.sh cpu --label quick_cpu

# A/B 回归（HEAD^ vs HEAD，各 3 轮）
scripts/run_ung_tests.sh ab --label ab_check --runs 3 --before-ref HEAD^ --after-ref HEAD

# GPU/CPU 结果一致性对比
scripts/run_ung_tests.sh compare --gpu-dir /path/to/gpu_run --cpu-dir /path/to/cpu_run
```

说明：

- `ab` 模式会自动切换 git 版本并恢复现场，要求工作区干净（无未提交改动）。
- 输出默认写到：`/home/graphdb/FilterVectorResultsRefactor`
- 如果你已经编译好，可加 `--skip-build` 跳过构建步骤。

## 1. 环境与前置条件

### 1.1 路径约定

- 代码仓库：`/home/graphdb/FilterVectorCode_refactor`
- 构建目录：`/home/graphdb/FilterVectorCode_refactor/build_ung_test_225902`
- 可执行文件：`/home/graphdb/FilterVectorCode_refactor/build_ung_test_225902/apps/build_UNG_index`
- 数据集（CelebA）：
  - `/home/graphdb/aaaGPU/FilterVectorData/celeba/celeba_base.bin`
  - `/home/graphdb/aaaGPU/FilterVectorData/celeba/celeba_base_labels.txt`
- 结果目录根：`/home/graphdb/FilterVectorResultsRefactor`

### 1.2 编译

```bash
cd /home/graphdb/FilterVectorCode_refactor/build_ung_test_225902
cmake --build . -j 32 --target build_UNG_index
```

### 1.3 临时必填参数文件

当前 `build_UNG_index` 需要 `--base_label_info_file` 和 `--base_label_tree_roots` 参数。若本地没有真实文件，可用空文件占位：

```bash
mkdir -p /home/graphdb/FilterVectorCode_refactor/.tmp
: > /home/graphdb/FilterVectorCode_refactor/.tmp/dummy_base_labels_info.log
: > /home/graphdb/FilterVectorCode_refactor/.tmp/dummy_tree_roots.txt
```

## 2. 单次构建测试（GPU）

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT=/home/graphdb/FilterVectorResultsRefactor/manual_gpu_${TS}
mkdir -p "$OUT/index_files" "$OUT/results" "$OUT/others"

UNG_CROSS_EDGE_BACKEND=1 \
UNG_COVERAGE_IMPL=1 \
UNG_COVERAGE_THREADS=32 \
/home/graphdb/FilterVectorCode_refactor/build_ung_test_225902/apps/build_UNG_index \
  --dataset celeba --data_type float --dist_fn L2 --num_threads 32 \
  --max_degree 64 --Lbuild 100 --alpha 1.2 --num_cross_edges 6 \
  --base_bin_file /home/graphdb/aaaGPU/FilterVectorData/celeba/celeba_base.bin \
  --base_label_file /home/graphdb/aaaGPU/FilterVectorData/celeba/celeba_base_labels.txt \
  --base_label_info_file /home/graphdb/FilterVectorCode_refactor/.tmp/dummy_base_labels_info.log \
  --base_label_tree_roots /home/graphdb/FilterVectorCode_refactor/.tmp/dummy_tree_roots.txt \
  --index_path_prefix "$OUT/index_files/" \
  --result_path_prefix "$OUT/results/" \
  --scenario general \
  > "$OUT/others/ung_build.log" 2>&1

echo "$OUT"
```

提取关键耗时：

```bash
LOG=$OUT/others/ung_build.log
rg -n "Finished building LNG|Calculating descendants info|Calculating coverage ratio|Building cross-group edges|\\[cross_edges\\]|\\[GPU GEMM\\]|Finish in|Index time" "$LOG"
```

## 3. 单次构建测试（CPU对照）

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT=/home/graphdb/FilterVectorResultsRefactor/manual_cpu_${TS}
mkdir -p "$OUT/index_files" "$OUT/results" "$OUT/others"

UNG_CROSS_EDGE_BACKEND=0 \
UNG_COVERAGE_IMPL=1 \
UNG_COVERAGE_THREADS=32 \
/home/graphdb/FilterVectorCode_refactor/build_ung_test_225902/apps/build_UNG_index \
  --dataset celeba --data_type float --dist_fn L2 --num_threads 32 \
  --max_degree 64 --Lbuild 100 --alpha 1.2 --num_cross_edges 6 \
  --base_bin_file /home/graphdb/aaaGPU/FilterVectorData/celeba/celeba_base.bin \
  --base_label_file /home/graphdb/aaaGPU/FilterVectorData/celeba/celeba_base_labels.txt \
  --base_label_info_file /home/graphdb/FilterVectorCode_refactor/.tmp/dummy_base_labels_info.log \
  --base_label_tree_roots /home/graphdb/FilterVectorCode_refactor/.tmp/dummy_tree_roots.txt \
  --index_path_prefix "$OUT/index_files/" \
  --result_path_prefix "$OUT/results/" \
  --scenario general \
  > "$OUT/others/ung_build.log" 2>&1

echo "$OUT"
```

## 4. 正确性校验（推荐流程）

### 4.1 日志结构一致性

对比以下统计是否一致：

- `Average number of descendants per group`
- `LNG 边 (Edges) 总数`
- `[cross_edges] ... target_groups=...`

```bash
for f in GPU_LOG_PATH CPU_LOG_PATH; do
  echo "==== $f"
  rg -n "Average number of descendants|LNG 边 \\(Edges\\)|target_groups=" "$f"
done
```

### 4.2 关键索引文件 md5 对比（GPU vs CPU）

```bash
GPU=/path/to/gpu_run/index_files
CPU=/path/to/cpu_run/index_files

for f in lng_descendants_rb.bin covered_sets_rb.bin vector_attr_graph lng_descendants_num lng_coverage_ratio; do
  echo "== $f"
  md5sum "$GPU/$f" "$CPU/$f"
done
```

## 5. A/B 性能回归测试（两版本代码）

目标：比较 `HEAD^` 与 `HEAD`，每个版本至少跑 2~3 轮，避免单次波动误判。

### 5.1 执行脚本（2轮示例）

```bash
set -euo pipefail
REPO=/home/graphdb/FilterVectorCode_refactor
BUILD=/home/graphdb/FilterVectorCode_refactor/build_ung_test_225902
BIN=$BUILD/apps/build_UNG_index
DATA_BIN=/home/graphdb/aaaGPU/FilterVectorData/celeba/celeba_base.bin
DATA_LBL=/home/graphdb/aaaGPU/FilterVectorData/celeba/celeba_base_labels.txt
DUMMY_INFO=$REPO/.tmp/dummy_base_labels_info.log
DUMMY_ROOT=$REPO/.tmp/dummy_tree_roots.txt
RES_ROOT=/home/graphdb/FilterVectorResultsRefactor
TS=$(date +%Y%m%d_%H%M%S)
CSV=$RES_ROOT/ab_manual_${TS}.csv
echo "variant,commit,run,log_path,lng_ms,desc_ms,cov_ms,cross_ms,index_ms" > "$CSV"

run_one () {
  local variant=$1
  local commit=$2
  local runid=$3
  local out=$RES_ROOT/ab_manual_${variant}_${commit}_r${runid}_${TS}
  mkdir -p "$out/index_files" "$out/results" "$out/others"
  UNG_CROSS_EDGE_BACKEND=1 UNG_COVERAGE_IMPL=1 UNG_COVERAGE_THREADS=32 \
  "$BIN" --dataset celeba --data_type float --dist_fn L2 --num_threads 32 \
    --max_degree 64 --Lbuild 100 --alpha 1.2 --num_cross_edges 6 \
    --base_bin_file "$DATA_BIN" --base_label_file "$DATA_LBL" \
    --base_label_info_file "$DUMMY_INFO" --base_label_tree_roots "$DUMMY_ROOT" \
    --index_path_prefix "$out/index_files/" --result_path_prefix "$out/results/" --scenario general \
    > "$out/others/ung_build.log" 2>&1

  local log="$out/others/ung_build.log"
  local lng=$(rg -o "Finished building LNG in ([0-9.]+) ms" -r '$1' "$log" | tail -n1)
  local desc=$(awk '/Calculating descendants info/{f=1;next} f&&/- Finish in/{print $4; exit}' "$log")
  local cov=$(awk '/Calculating coverage ratio/{f=1;next} f&&/- Finish in/{print $4; exit}' "$log")
  local cross=$(awk '/Building cross-group edges/{f=1;next} f&&/- Finish in/{print $4; exit}' "$log")
  local index=$(rg -o "Index time: ([0-9]+)ms" -r '$1' "$log" | tail -n1)
  echo "$variant,$commit,$runid,$log,$lng,$desc,$cov,$cross,$index" >> "$CSV"
}

cd "$REPO"
CUR_BRANCH=$(git rev-parse --abbrev-ref HEAD)
AFTER=$(git rev-parse --short HEAD)
BEFORE=$(git rev-parse --short HEAD^)

for variant in before after; do
  if [ "$variant" = "before" ]; then
    git checkout "$BEFORE" >/dev/null 2>&1
    COMMIT="$BEFORE"
  else
    git checkout "$AFTER" >/dev/null 2>&1
    COMMIT="$AFTER"
  fi
  cmake --build "$BUILD" -j 32 --target build_UNG_index >/dev/null
  for i in 1 2; do
    run_one "$variant" "$COMMIT" "$i"
  done
done

git checkout "$CUR_BRANCH" >/dev/null 2>&1
echo "$CSV"
```

### 5.2 汇总中位数

```bash
python - <<'PY'
import csv,statistics
p="CSV_PATH_HERE"
rows=list(csv.DictReader(open(p)))
metrics=['lng_ms','desc_ms','cov_ms','cross_ms','index_ms']
for v in ['before','after']:
    part=[r for r in rows if r['variant']==v]
    print(v)
    for m in metrics:
        vals=[float(r[m]) for r in part]
        print(m, 'median=', statistics.median(vals), 'vals=', vals)
    print()
PY
```

## 6. 结果解读建议

- 优化是否“有效”：优先看 `Index time` 中位数。
- 优化是否“定位准确”：看目标阶段耗时是否下降（如 LNG、cross-group edges）。
- 优化是否“正确”：至少满足以下两项：
  - 结构统计不异常（例如 LNG 边数、平均后代数、target_groups 不突变）
  - 关键文件 md5 一致（GPU/CPU 对照）

## 7. 常见问题

- 看到某次跑得异常慢：先重跑 2~3 次再下结论，避免系统噪声误判。
- 如果构建失败：先确认当前 `git` 分支下重新 `cmake --build ... --target build_UNG_index`。
- 如果参数缺失报错：确认 dummy 文件存在（第 1.3 节）。
