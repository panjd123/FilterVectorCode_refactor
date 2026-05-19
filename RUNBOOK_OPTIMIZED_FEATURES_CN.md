# 优化功能运行手册

本文和 `CURRENT_OPTIMIZATION_STATUS_CN.md` 配套，目标是回答：

```text
这些优化怎么跑起来？
每个功能对应哪些环境变量？
结果看哪里？
怎么判断跑对了？
```

如果你只想了解“目前做到了什么、效果如何、哪些方向失败了”，先看：

```text
CURRENT_OPTIMIZATION_STATUS_CN.md
```

如果你要复现实验、改参数、跑 benchmark，就看本文。

## 0. 功能到命令速查

| 目标 | 入口章节 | 主要命令/变量 | 主要结果 |
| --- | --- | --- | --- |
| 编译当前 UNG 构建程序 | 1 | `cmake -S UNG/codes -B build_ung_test_225902` | `build_ung_test_225902/apps/build_UNG_index` |
| 跑一次完整 UNG build | 2 | `build_UNG_index ...` | `$OUT/results/build_time.csv` |
| 对比 CPU/GPU cross-edge | 4 | `UNG_CROSS_EDGE_BACKEND=0/1` | `build_cross_edges_time` |
| 开启 fused/custom GPU topK | 5 | `UNG_SMALL_GROUP_FUSED=1`, `UNG_LARGE_GROUP_FUSED_MODE=2` | 日志中的 `[PROF] cross_edges` |
| 开启 descendants/coverage 优化 | 6 | `UNG_COVERAGE_IMPL=1` | `cal_descendants_time`, `cal_coverage_ratio_time` |
| 复现 FixedPoolGPU 失败路径 | 7 | `UNG_FIXED_POOL_L`, `UNG_FIXED_POOL_ITERS` | `build_graph_time` |
| 跑 Tagore GPU 建图 | 8 | `run_tagore_all_large.py` | `tagore_all_large_build_times.csv` |
| 跑 Tagore 图 + GPU ANN 查询 | 9 | `tagore_index_gpu_query_bench` | `recall_at_k`, `ann_e2e_reuse_ms` |
| 生成 power-Zipf 标签数据 | 10 | `tools_generate_powerzipf_labels.py` | `.txt` 标签文件和 `.manifest.json` |
| 统计 group / nq / nx 分布 | 11 | 内联 Python 统计脚本 | `p50/p90/p95/p99/max`, `nq/nx` |
| 跑封装 build 脚本 | 12 | `build_hybrid.sh` | 完整结果目录 |

默认路径：

```text
REPO=/home/graphdb/FilterVectorCode_refactor
RESULT_ROOT=/home/graphdb/FilterVectorResultsRefactor
SIFT_DIR=/home/graphdb/aaaGPU/FilterVectorData/sift
BUILD=/home/graphdb/FilterVectorCode_refactor/build_ung_test_225902
BIN=/home/graphdb/FilterVectorCode_refactor/build_ung_test_225902/apps/build_UNG_index
```

## 1. 编译与基础检查

### 1.1 编译 UNG

```bash
cd /home/graphdb/FilterVectorCode_refactor
cmake -S UNG/codes -B build_ung_test_225902 -DCMAKE_BUILD_TYPE=Release
cmake --build build_ung_test_225902 -j 32 --target build_UNG_index
```

必要工具：

```bash
ls -lh build_ung_test_225902/apps/build_UNG_index
ls -lh build_ung_test_225902/tools/fvecs_to_bin
ls -lh build_ung_test_225902/tools/generate_base_labels
ls -lh build_ung_test_225902/tools/generate_query_labels
```

### 1.2 GPU 空闲检查

当前机器没有可用 `gpulock` 时，至少手工检查：

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
```

性能测试建议只在 `utilization.gpu=0%` 时跑。

## 2. 直接跑一次 UNG 构建

### 2.1 通用构建模板

这个模板直接调用 `build_UNG_index`，适合复现实验和调环境变量。

```bash
set -euo pipefail
REPO=/home/graphdb/FilterVectorCode_refactor
BIN=$REPO/build_ung_test_225902/apps/build_UNG_index
DATA_DIR=/home/graphdb/aaaGPU/FilterVectorData/sift
RESULT_ROOT=/home/graphdb/FilterVectorResultsRefactor
DATASET=sift_powerzipf30
TS=$(date +%Y%m%d_%H%M%S)
OUT=$RESULT_ROOT/manual_${DATASET}_${TS}
mkdir -p "$OUT/index_files" "$OUT/results" "$OUT/others"

# build_UNG_index 当前要求这两个文件存在；SIFT 自构造标签可用空文件占位。
: > "$DATA_DIR/${DATASET}_base_labels_info.log"
: > "$DATA_DIR/tree_roots.txt"

UNG_CROSS_EDGE_BACKEND=1 \
UNG_CROSS_EDGE_GPU_STRICT=1 \
UNG_COVERAGE_IMPL=1 \
UNG_COVERAGE_THREADS=32 \
"$BIN" \
  --dataset "$DATASET" \
  --data_type float \
  --dist_fn L2 \
  --num_threads 32 \
  --max_degree 32 \
  --Lbuild 100 \
  --alpha 1.2 \
  --num_cross_edges 6 \
  --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
  --base_label_file "$DATA_DIR/${DATASET}_base_labels.txt" \
  --base_label_info_file "$DATA_DIR/${DATASET}_base_labels_info.log" \
  --base_label_tree_roots "$DATA_DIR/tree_roots.txt" \
  --index_path_prefix "$OUT/index_files/" \
  --result_path_prefix "$OUT/results/" \
  --scenario general \
  > "$OUT/others/ung_build.log" 2>&1

echo "$OUT"
cat "$OUT/results/build_time.csv"
```

### 2.2 判断是否跑对

看日志：

```bash
OUT=/home/graphdb/FilterVectorResultsRefactor/manual_sift_powerzipf30_YYYYMMDD_HHMMSS
rg -n "Loading data|Number of points|Number of labels|Number of groups|Building graph|Building label navigation|Average number of descendants|Building cross-group edges|\[cross_edges\]|Saving" "$OUT/others/ung_build.log"
cat "$OUT/results/build_time.csv"
```

应看到：

```text
Number of points: 1000000
Number of labels: 与数据集一致
Number of groups: 与标签构造统计大致一致
[cross_edges] backend=GPU
results/build_time.csv 存在
```

## 3. 跑不同数据集

### 3.1 SIFT30 原始 Zipf 风格数据

软链接或原始文件名通常已经存在：

```text
sift_base_30_labels_zipf_origstyle.txt
```

如果要按 `DATASET=sift30_zipf_origstyle` 运行，需要准备软链接：

```bash
D=/home/graphdb/aaaGPU/FilterVectorData/sift
ln -sfn sift_base.bin "$D/sift30_zipf_origstyle_base.bin"
ln -sfn sift_base.fvecs "$D/sift30_zipf_origstyle_base.fvecs"
ln -sfn sift_query.bin "$D/sift30_zipf_origstyle_query.bin"
ln -sfn sift_query.fvecs "$D/sift30_zipf_origstyle_query.fvecs"
ln -sfn sift_groundtruth.ivecs "$D/sift30_zipf_origstyle_groundtruth.ivecs"
ln -sfn sift_base_30_labels_zipf_origstyle.txt "$D/sift30_zipf_origstyle_base_labels.txt"
: > "$D/sift30_zipf_origstyle_base_labels_info.log"
```

然后把通用模板里的：

```bash
DATASET=sift30_zipf_origstyle
```

### 3.2 新 power-Zipf 数据

已经准备好：

```text
DATASET=sift_powerzipf30
```

对应文件：

```bash
ls -l /home/graphdb/aaaGPU/FilterVectorData/sift/sift_powerzipf30_*
```

标签统计：

```bash
cat /home/graphdb/aaaGPU/FilterVectorData/sift/sift_base_30_labels_powerzipf_a0.85_c0.75_p0.9_seed20260519.manifest.json
```

### 3.3 hiermedium 均衡数据

已有标签文件：

```text
sift_base_hier_medium_1000x3_g128_420_seed20260519.txt
```

可准备软链接：

```bash
D=/home/graphdb/aaaGPU/FilterVectorData/sift
ln -sfn sift_base.bin "$D/sift_hiermedium_base.bin"
ln -sfn sift_base.fvecs "$D/sift_hiermedium_base.fvecs"
ln -sfn sift_query.bin "$D/sift_hiermedium_query.bin"
ln -sfn sift_query.fvecs "$D/sift_hiermedium_query.fvecs"
ln -sfn sift_groundtruth.ivecs "$D/sift_hiermedium_groundtruth.ivecs"
ln -sfn sift_base_hier_medium_1000x3_g128_420_seed20260519.txt "$D/sift_hiermedium_base_labels.txt"
: > "$D/sift_hiermedium_base_labels_info.log"
```

然后：

```bash
DATASET=sift_hiermedium
```

## 4. CPU/GPU cross-edge A/B

### 4.1 GPU cross-edge

```bash
UNG_CROSS_EDGE_BACKEND=1 \
UNG_CROSS_EDGE_GPU_STRICT=1 \
UNG_COVERAGE_IMPL=1 \
UNG_COVERAGE_THREADS=32 \
... build_UNG_index ...
```

### 4.2 CPU cross-edge baseline

```bash
UNG_CROSS_EDGE_BACKEND=0 \
UNG_COVERAGE_IMPL=1 \
UNG_COVERAGE_THREADS=32 \
... build_UNG_index ...
```

### 4.3 看结果

```bash
cat "$OUT/results/build_time.csv"
rg -n "\[cross_edges\]|Building cross-group edges|Finish in" "$OUT/others/ung_build.log"
```

重点看：

```text
build_cross_edges_time
index_time
```

历史 SIFT30 对比：

```text
CPU cross-edge: 44560 ms
GPU cross-edge: 5002 ms
cross-edge 约 8.9x
```

## 5. GPU cross-edge 内部实现开关

这些变量控制 `UNG/codes/src/gpu_gemm_topk.cu` 的精确 topK 路径。

### 5.1 后端选择

```bash
export UNG_FORCE_CUSTOM_KERNEL=1
export UNG_GEMM_IMPL=2
```

含义：

```text
UNG_GEMM_IMPL=0  cublasLt
UNG_GEMM_IMPL=1  cublasSgemmStridedBatched
UNG_GEMM_IMPL=2  naive/custom CUDA-core path
UNG_FORCE_CUSTOM_KERNEL=1 会强制走 custom path
```

### 5.2 大 group 分流到 SGEMM

```bash
export UNG_NAIVE_HEAVY_SGEMM=1
export UNG_NAIVE_HEAVY_NX=256
export UNG_NAIVE_HEAVY_NQ=256
export UNG_NAIVE_HEAVY_WORK_M=1
```

含义：

```text
当 nx >= 256 且 nq >= 256 或 work >= 1M 时，重负载可以分流到 cuBLAS/SGEMM 路径。
```

### 5.3 小/中/大 group fused kernel

```bash
export UNG_SMALL_GROUP_FUSED=1
export UNG_SMALL_GROUP_MAX_NX=8
export UNG_SMALL_GROUP_WARPS=8

export UNG_MEDIUM_GROUP_FUSED=1
export UNG_MEDIUM_GROUP_MAX_NX=64
export UNG_MEDIUM_GROUP_WARPS=8

export UNG_LARGE_GROUP_FUSED=1
export UNG_LARGE_GROUP_MIN_NX=128
export UNG_LARGE_GROUP_MAX_NX=1023
export UNG_LARGE_GROUP_WARPS=2
export UNG_LARGE_GROUP_FUSED_MODE=2
```

`UNG_LARGE_GROUP_FUSED_MODE`：

```text
0 = one CTA/query CUDA-core fused
1 = one warp/query CUDA-core fused
2 = TF32 Tensor Core 16x16 WMMA fused topK
```

### 5.4 推荐起点

对 SIFT30 长尾数据：

```bash
export UNG_CROSS_EDGE_BACKEND=1
export UNG_CROSS_EDGE_GPU_STRICT=1
export UNG_COVERAGE_IMPL=1
export UNG_COVERAGE_THREADS=32
export UNG_SMALL_GROUP_FUSED=1
export UNG_MEDIUM_GROUP_FUSED=1
export UNG_BUCKET_GROUP_FUSED=1
export UNG_LARGE_GROUP_FUSED=1
export UNG_LARGE_GROUP_FUSED_MODE=2
export UNG_NAIVE_HEAVY_SGEMM=1
export UNG_NAIVE_HEAVY_NX=256
export UNG_NAIVE_HEAVY_NQ=256
```

看日志中的 `[PROF]`：

```bash
rg -n "\[PROF\] cross_edges|heavy_sgemm|large_group_fused|topk" "$OUT/others/ung_build.log"
```

## 6. Coverage / descendants 开关

### 6.1 推荐配置

```bash
export UNG_COVERAGE_IMPL=1
export UNG_COVERAGE_THREADS=32
```

含义：

```text
UNG_COVERAGE_IMPL=1  descendants_direct
UNG_COVERAGE_IMPL=0  legacy_topological_merge
```

通常推荐 `1`，它配合 vector layout 和 Roaring addMany 是当前稳定优化路径。

### 6.2 校验 coverage 是否合理

```bash
rg -n "Calculating descendants info|Average number of descendants|Calculating coverage ratio|coverage impl|Finish in" "$OUT/others/ung_build.log"
ls -lh "$OUT/index_files/lng_descendants_rb.bin" "$OUT/index_files/covered_sets_rb.bin"
```

如果改动 coverage 代码，应和 CPU/旧版本对比：

```bash
md5sum "$GPU/index_files/lng_descendants_rb.bin" "$CPU/index_files/lng_descendants_rb.bin"
md5sum "$GPU/index_files/covered_sets_rb.bin" "$CPU/index_files/covered_sets_rb.bin"
md5sum "$GPU/index_files/lng_descendants_num" "$CPU/index_files/lng_descendants_num"
md5sum "$GPU/index_files/lng_coverage_ratio" "$CPU/index_files/lng_coverage_ratio"
```

## 7. FixedPoolGPU 实验路径

这条路径目前不是推荐主线，只用于复现实验和对照。

### 7.1 开启方式

查看当前 `uni_nav_graph.cpp` 中 build graph backend 的开关。如果使用 FixedPoolGPU，需要确保已编译：

```bash
cmake --build build_ung_test_225902 -j 32 --target build_UNG_index
```

可调变量：

```bash
export UNG_FIXED_POOL_L=64
export UNG_FIXED_POOL_ITERS=8
export UNG_FIXED_POOL_SEED=20260519
export UNG_FIXED_POOL_THREADS=128
```

含义：

```text
UNG_FIXED_POOL_L       固定邻居池容量，范围 [max_degree, 128]
UNG_FIXED_POOL_ITERS   邻居的邻居 refine 轮数
UNG_FIXED_POOL_SEED    初始化随机种子
UNG_FIXED_POOL_THREADS refine kernel 每点线程数
```

### 7.2 已知结论

历史对比：

```text
CPU Vamana: build_graph_time = 2773.15 ms
FixedPoolGPU: build_graph_time = 129447 ms
```

结果目录：

```text
/home/graphdb/FilterVectorResultsRefactor/pg_build_compare_20260519_172008
```

所以：

```text
不要把 FixedPoolGPU 当生产路径。
它主要说明 naive per-group GPU builder 会被 malloc/copy/launch 开销打爆。
```

## 8. Tagore GPU 建图 benchmark

Tagore 仓库位置：

```text
/home/graphdb/Tagore
```

Python module：

```text
/home/graphdb/Tagore/build/Tagore.cpython-310-x86_64-linux-gnu.so
```

### 8.1 单独跑 Tagore 中大 group 建图

已有脚本和结果目录：

```text
/home/graphdb/FilterVectorResultsRefactor/tagore_build_bench_20260519_172638
```

重跑全部 `nx>=1024` group：

```bash
OUT=/home/graphdb/FilterVectorResultsRefactor/tagore_build_bench_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
cp /home/graphdb/FilterVectorResultsRefactor/tagore_build_bench_20260519_172638/run_tagore_all_large.py "$OUT/"
OUTDIR="$OUT" python3 "$OUT/run_tagore_all_large.py" > "$OUT/run_all_large.log" 2>&1

grep '^SUMMARY' "$OUT/run_all_large.log"
tail -20 "$OUT/tagore_all_large_build_times.csv"
```

### 8.2 结果解读

看：

```text
time of GNN-Descent
time of pruning
SUMMARY groups ... wall_total ... mean ... p50 ... max
```

历史结果：

```text
groups=119, nx>=1024
wall_total=3.434s
sum_group_total=3.246s
p50=13.97ms
max=427.15ms
```

适用判断：

```text
Tagore 对中大 group 有潜力
但当前 Python API + 文件 I/O + index 落盘，不适合直接嵌入主线
```

## 9. Tagore 图 + GPU ANN batch query benchmark

benchmark 文件：

```text
/home/graphdb/FilterVectorResultsRefactor/tagore_build_bench_20260519_172638/tagore_index_gpu_query_bench.cu
```

可执行：

```text
/home/graphdb/FilterVectorResultsRefactor/tagore_build_bench_20260519_172638/tagore_index_gpu_query_bench
```

### 9.1 编译

```bash
OUT=/home/graphdb/FilterVectorResultsRefactor/tagore_build_bench_20260519_172638
nvcc -O3 -std=c++17 -arch=sm_86 \
  "$OUT/tagore_index_gpu_query_bench.cu" \
  -o "$OUT/tagore_index_gpu_query_bench"
```

### 9.2 跑 10 个 `nx≈1k` group，每组 4096 queries

```bash
OUT=/home/graphdb/FilterVectorResultsRefactor/tagore_build_bench_20260519_172638
python3 - <<'PY'
import csv, pathlib, shlex, subprocess
root=pathlib.Path('/home/graphdb/FilterVectorResultsRefactor/tagore_build_bench_20260519_172638')
manifest=pathlib.Path('/home/graphdb/FilterVectorResultsRefactor/tagore_sift30_large_groups_20260519_1402/all_large_manifest.csv')
paths={}
for r in csv.DictReader(manifest.open()):
    paths[int(r['gid'])]=r['path']
rows=[]
for r in csv.DictReader((root/'tagore_all_large_build_times.csv').open()):
    n=int(r['n'])
    if 1024 <= n < 1200:
        gid=int(r['gid'])
        rows.append((gid,n,paths[gid],str(root/f'all_large_group_{gid}_n{n}.vamana.index'),float(r['total_s_wall'])*1000))
rows=rows[:10]
cmd=[str(root/'tagore_index_gpu_query_bench'),'128','4096','6','64']
for gid,n,p,idx,ms in rows:
    cmd += [p, idx, f'{ms:.6f}']
print(' '.join(shlex.quote(x) for x in cmd))
subprocess.run(cmd, check=True)
PY
```

### 9.3 跑更大 query batch

把命令中的 `4096` 改成 `32768`：

```text
... tagore_index_gpu_query_bench 128 32768 6 64 ...
```

### 9.4 结果字段解释

输出 CSV 字段：

```text
build_ms              Tagore 建图总耗时
h2d_ms                数据和图拷到 GPU
exact_search_ms       exact GPU topK search-only
ann_search_ms         graph ANN search-only
d2h_ms                输出拷回
exact_e2e_ms          h2d + exact + d2h
ann_e2e_once_ms       build + h2d + ann + d2h
ann_e2e_reuse_ms      h2d + ann + d2h
recall_at_k           相对 exact topK 的 recall
```

判断：

```text
如果 ann_e2e_once_ms > exact_e2e_ms，说明一次性 build+query 不划算
如果 ann_e2e_reuse_ms < exact_e2e_ms，说明图复用时 ANN 有价值
```

历史高 recall 结果：

```text
nq=32768/group, degree64, entries256, etop8
recall@6=0.958
exact_e2e=115.49ms
ann_e2e_reuse=84.55ms
reuse speedup=1.37x
ann_e2e_once=373.82ms
```

## 10. 生成新的 power-Zipf 标签数据集

生成脚本：

```text
/home/graphdb/FilterVectorCode_refactor/tools_generate_powerzipf_labels.py
```

### 10.1 生成当前版本

```bash
python3 /home/graphdb/FilterVectorCode_refactor/tools_generate_powerzipf_labels.py \
  --output /home/graphdb/aaaGPU/FilterVectorData/sift/sift_base_30_labels_powerzipf_a0.85_c0.75_p0.9_seed20260519.txt \
  --manifest /home/graphdb/aaaGPU/FilterVectorData/sift/sift_base_30_labels_powerzipf_a0.85_c0.75_p0.9_seed20260519.manifest.json \
  --num-points 1000000 \
  --num-labels 30 \
  --alpha 0.85 \
  --c 0.75 \
  --pmax 0.9 \
  --seed 20260519
```

公式：

```text
P(label i) = min(pmax, c / i^alpha)
```

当前版本统计：

```text
num_groups=145017
singleton_groups=92521
group p50=1
p90=7
p95=14
p99=80
max=26664
top1_share=2.67%
top10_share=10.03%
```

### 10.2 创建 dataset 软链接

```bash
D=/home/graphdb/aaaGPU/FilterVectorData/sift
ln -sfn sift_base.fvecs "$D/sift_powerzipf30_base.fvecs"
ln -sfn sift_base.bin "$D/sift_powerzipf30_base.bin"
ln -sfn sift_query.fvecs "$D/sift_powerzipf30_query.fvecs"
ln -sfn sift_query.bin "$D/sift_powerzipf30_query.bin"
ln -sfn sift_groundtruth.ivecs "$D/sift_powerzipf30_groundtruth.ivecs"
ln -sfn sift_base_30_labels_powerzipf_a0.85_c0.75_p0.9_seed20260519.txt "$D/sift_powerzipf30_base_labels.txt"
: > "$D/sift_powerzipf30_base_labels_info.log"
: > "$D/tree_roots.txt"
```

### 10.3 调参建议

如果 singleton 太多：

```text
降低 num_labels
提高 alpha
降低平均 label 数
或引入受控 label-set 模板，而不是完全独立 Bernoulli
```

如果头部 group 太大：

```text
降低 c
降低 pmax
降低 alpha 但注意会显著增加 singleton
```

每次生成后必须看：

```text
num_groups
singleton ratio
group_size p50/p90/p95/p99/max
top1/top10 share
```

完整 UNG build 后还要看：

```text
nq/nx p50/p95/p99
work=nq*nx top1/top10 share
build_LNG_time
build_cross_edges_time
```

## 11. 统计 group / nq / nx 分布

目前还没有独立脚本落盘，可直接用下面的 Python 片段分析一个已构建 index：

```bash
python3 - <<'PY'
from pathlib import Path
import statistics, math, sys
root=Path('/home/graphdb/FilterVectorResultsRefactor/sift30_zipf_origstyle_current_gpu_20260519_133252')

def pct(vals,q):
    vals=sorted(vals)
    pos=(len(vals)-1)*q/100
    lo=math.floor(pos); hi=math.ceil(pos)
    if lo==hi: return vals[lo]
    return vals[lo]*(hi-pos)+vals[hi]*(pos-lo)

def parse_ranges(p):
    xs=[]
    for line in p.read_text().splitlines():
        if line.strip():
            a,b=map(int,line.split())
            xs.append((a,b))
    if xs and xs[-1] == (0,0): xs=xs[:-1]
    return xs

def parse_out(p):
    return [list(map(int,line.split())) if line.strip() else [] for line in p.read_text().splitlines()]

ranges=parse_ranges(root/'index_files/group_id_to_range')
outs=parse_out(root/'index_files/lng_out_neighbors.dat')
n=len(ranges)
sizes=[b-a for a,b in ranges]
ins=[[] for _ in range(n)]
for u,vs in enumerate(outs[:n]):
    for v in vs:
        if 0 <= v < n: ins[v].append(u)

targets=[]
for v in range(n):
    if ins[v]:
        nx=sizes[v]
        nq=sum(sizes[u] for u in ins[v])
        targets.append((v,nq,nx,nq/nx if nx else 0,nq*nx))

for name,vals in [('nx',[x[2] for x in targets]),('nq',[x[1] for x in targets]),('nq/nx',[x[3] for x in targets]),('work',[x[4] for x in targets])]:
    print(name, 'mean', statistics.mean(vals), 'p50', pct(vals,50), 'p95', pct(vals,95), 'p99', pct(vals,99), 'max', max(vals))
print('top work')
for x in sorted(targets,key=lambda t:t[4],reverse=True)[:10]:
    print(x)
PY
```

把 `root` 改成你自己的结果目录。

## 12. 用 `build_hybrid.sh` 跑完整 build 流程

如果希望沿项目原有脚本跑：

```bash
cd /home/graphdb/FilterVectorCode_refactor
export UNG_BUILD_DIR=/home/graphdb/FilterVectorCode_refactor/build_ung_test_225902
export ACORN_BUILD_DIR=/home/graphdb/FilterVectorCode_refactor/build_acorn_test
export SCRIPT_DIR=/home/graphdb/FilterVectorCode_refactor

UNG_CROSS_EDGE_BACKEND=1 \
UNG_CROSS_EDGE_GPU_STRICT=1 \
UNG_COVERAGE_IMPL=1 \
UNG_COVERAGE_THREADS=32 \
./build_hybrid.sh \
  --build_mode ung_only \
  --query_dir_name dummy \
  --dataset sift_powerzipf30 \
  --data_dir /home/graphdb/aaaGPU/FilterVectorData/sift \
  --exp_output_dir /home/graphdb/FilterVectorResultsRefactor/sift_powerzipf30_build_hybrid \
  --max_degree 32 \
  --Lbuild 100 \
  --alpha 1.2 \
  --num_cross_edges 6 \
  --num_entry_points 16 \
  --acorn_n 1000000 \
  --acorn_m 32 \
  --acorn_m_beta 64 \
  --acorn_gamma 80
```

输出目录形如：

```text
/home/graphdb/FilterVectorResultsRefactor/sift_powerzipf30_build_hybrid/Index/M32_LB100_alpha1.2_C6_EP16_AN1000000_AM32_AMB64_AG80
```

注意：

```text
build_hybrid.sh 仍会检查/编译 ACORN，哪怕 ung_only；直接调用 build_UNG_index 更轻量。
```

## 13. 常见问题

### 13.1 为什么 `build_time.csv` 里 cross_edge_step 字段是乱码极小值

这些字段目前不是可靠计时字段，主要看：

```text
build_cross_edges_time
build_graph_time
build_LNG_time
cal_descendants_time
cal_coverage_ratio_time
index_time
```

### 13.2 为什么 GPU strict 要开

```bash
UNG_CROSS_EDGE_GPU_STRICT=1
```

可以避免 GPU 失败后静默回退 CPU，导致你以为测的是 GPU，实际测的是 CPU。

### 13.3 什么时候不应该跑 Tagore/ANN

```text
nx 很小，特别是 nx=1/几十
图只构建后查一次
目标 recall 必须接近 exact
数据分布被 singleton 主导
```

### 13.4 什么时候应该考虑 Tagore/ANN

```text
nx >= 1024 的中大 group
图可以复用
每个 group 有大量 query
允许 recall/performance tradeoff
```

## 14. 推荐实验组合

### 14.1 验证 GPU cross-edge 是否工作

```text
数据集：sift30_zipf_origstyle
对比：UNG_CROSS_EDGE_BACKEND=0 vs 1
指标：build_cross_edges_time
预期：GPU 明显快，SIFT30 历史约 8.9x
```

### 14.2 验证 small/medium fused kernel

```text
数据集：sift_hiermedium
指标：build_cross_edges_time
原因：nq≈nx≈250，适合测中等均衡任务
```

### 14.3 验证 Tagore 建图

```text
数据集：SIFT30 nx>=1024 groups
命令：run_tagore_all_large.py
指标：wall_total / p50 / max
预期：119 groups 约 3.4s wall
```

### 14.4 验证 ANN graph query 是否有潜力

```text
数据集：Tagore 已构建的 10 个 nx≈1k groups
命令：tagore_index_gpu_query_bench
指标：ann_e2e_reuse_ms vs exact_e2e_ms，recall_at_k
预期：图复用时可快；build+单次 query 不划算
```

### 14.5 验证新数据分布

```text
数据集：sift_powerzipf30
先看 manifest，再跑完整 build
指标：group p50/p95/p99/max、nq/nx、work top share
当前风险：singleton 仍然过多
```
