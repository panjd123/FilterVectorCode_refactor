# 两级 Special Block 复现实验手册

本文给出当前论文结果的最短、可审计复现路径。当前实现是固定的两级 overlay：中层阈值 `T1` 与上层阈值 `T2`，查询权限严格按普通图 `0` → 中层 `1` → 上层 `2` 单调升级；不是任意 N 层实现。

## 1. 三种“复现”必须分开

| 路径 | 能证明什么 | 是否需要未提交的大文件 | 是否重做计时 |
|---|---|---:|---:|
| 论文结果闭包审计 | 已报告表格确实来自已记录的 compact source、manifest 和统一历史 binary | 否 | 否 |
| 当前源码新鲜复测 | 当前源码在同一 Amazon x1 输入上重新构建、查询，并生成一套新的结果 | 是 | 是 |
| 历史 raw run 精确重放 | 使用论文实验当时的 immutable binary 和原有索引重跑同一 config | 是 | 是 |

仓库提交中不包含约 25 GB 的 `runs/`、数据集、索引和两个第三方 nested clone。因此，仅克隆 Git 仓库可以完整审计论文结果闭包，但不能凭空重做 raw benchmark。新鲜复测必须生成新的 output root 和 manifest，不能覆盖论文 source CSV。

## 2. 已验证环境与输入

- 隔离仓库：`/home/sunyahui/worktrees/FilterVectorCode_multilevel_special`。
- 实现/结果 checkpoint：`7a2bf4635eac43d174100770838daf1b3a10fa58`；最终交付文档在其后续提交。
- CPU：2 × Intel Xeon Platinum 8360Y，72 个物理核、144 个逻辑 CPU、2 个 NUMA node。
- GPU：NVIDIA L20 46 GiB，driver 550.54.14。
- 工具链：Python 3.10.18、CMake 3.31.11、GCC 10.3.0。
- Amazon 原始 x1：602,453 points、768D、482,387 groups、30,723 labels。
- base-label SHA-256：`aec768bba7092af445252835be3f1ef7f708305ea646af19ae72738df3dd2f96`。
- main-index label SHA-256：`ddb3f616c27626afe6b20bf633aca5e9a1efd31d82bd505b163f59fc79f4dd56`。
- source UNG fingerprint：`91d78580ae29f468`。
- 论文实验 immutable search binary SHA-256：`f078e1744775a3aefab6cc670b4a72e02b8d7d7118a6d34a6df76cb591287b11`。

当前源码重新编译出的 binary hash 可以不同；这表示“新鲜复测”，不能把其结果追加到上述历史 binary 的正式 manifest。内部 runner 会在每个 output root 中创建只读、内容寻址的 binary snapshot，并把 hash 写入 manifest。

本次已验证的当前源码 binary SHA-256 为 `88d7dba189478cd11402a8433076d220c7ad68ca9f9a6366118f78430752f37f`。六档 24 点 fresh regression 的 compact 输出被单独保存为 `results_summary/current_source_regression.csv`，不参与 `paper_results.csv` 的 operating-point 选择。

## 3. 只审计论文结果闭包

这是最快且不依赖 raw runs 的路径。

```bash
cd /home/sunyahui/worktrees/FilterVectorCode_multilevel_special/experiments/multilevel_special

python3 -m unittest -v test_generate_paper_results.py
python3 generate_paper_results.py
cd ../..

tail -n +2 experiments/multilevel_special/results_summary/artifact_manifest.csv | \
while IFS=, read -r path expected; do
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  test "$actual" = "$expected" || exit 1
done
```

预期结果：14 项结果生成/provenance 测试通过；生成 60 行 selected results、209 个内部 canonical measured points、447 个外部 canonical measured points 和 7 行构建记录；artifact manifest 无 hash mismatch。这里验证的是提交中的证据闭包，不是重新计时。

## 4. 构建与代码回归

```bash
cd /home/sunyahui/worktrees/FilterVectorCode_multilevel_special
cmake --build build_ung_rel -j16 --target \
  test_special_edge_io build_special_block_index search_UNG_index

cd build_ung_rel
ctest --output-on-failure -R \
  'ung_build_config|special_block_trie|special_block_free_state|special_candidate_queue|special_edge_io'
cd ..

cd experiments/multilevel_special
python3 -m unittest -v \
  test_multilevel_selection.py test_generate_paper_results.py
cd ../..
git diff --check
```

最终 checkpoint 上的预期结果为 C++ focused tests 5/5、Python tests 28/28。

## 5. 从当前源码重新构建 Special Block

以下命令只接受 Amazon x1；runner 会校验 point/group 数、label hash 和主图 fingerprint，并对输出加锁、先写 staging、成功后原子发布。runner 还会把 builder 复制成只读、内容寻址的 snapshot，并把 SHA-256 写入 manifest；这样构建期间重新编译工作目录中的 binary 不会造成混合版本。

```bash
cd /home/sunyahui/worktrees/FilterVectorCode_multilevel_special

# 单层 T1=1k baseline
python3 experiments/multilevel_special/run_build_sweep.py \
  experiments/multilevel_special/config.amazon_x1_fresh_single_build.json

# T1=1k，独立扫描 T2=4k/10k/25k/50k
python3 experiments/multilevel_special/run_build_sweep.py \
  experiments/multilevel_special/config.amazon_x1_build_sweep.json

# 固定 T2=25k 的 T1=500 与 T1=2000 构建敏感性
python3 experiments/multilevel_special/run_build_sweep.py \
  experiments/multilevel_special/config.amazon_x1_t1_build_sensitivity.json
python3 experiments/multilevel_special/run_build_sweep.py \
  experiments/multilevel_special/config.amazon_x1_t1_2000_build_sensitivity.json
```

不带 `--force` 时，已存在且通过 metadata/sidecar 校验的 case 会安全跳过。可先加 `--dry-run` 检查命令；不要对正式 output root 使用 `--force`，除非明确接受旧 case 被移入 quarantine。

当前源码的 tuned 配置可用 `config.amazon_x1_current_source_build_t1_2000_t2_25000.json` 重建。每次独立重建必须改为新的 `output_root`；三次已审计构建的 compact 结果由下式生成：

```bash
python3 experiments/multilevel_special/summarize_rebuild_regression.py
```

三次 builder wall 为 60.250 / 56.285 / 56.916 s。block metadata、trie 与 regular-edge sidecar 的 hash 一致；近似 GPU intra graph 使用并行原子更新，因此 `special_edges.bin` 不要求 bitwise identical。正确验收是 topology/provenance validator 加端到端 Recall，而不是 edge 文件 hash 相同。

CPU Vamana large-block 对照总耗时 1131.863 s，约为 GPU tuned 构建中位数的 19.89 倍；intra stage 1059.060 s，约为 GPU 中位 intra 的 101.32 倍。它只说明退回 CPU 的构建代价，不能作为同图质量的速度比较。

## 6. 内部六档查询复测

每个 config 都通过同一个 runner 执行；每轮完成后必须运行 validator，再运行 summarizer。论文主结论按同一 Recall 门槛选择最快的离散实测点，不插值。

```bash
cd /home/sunyahui/worktrees/FilterVectorCode_multilevel_special

run_one () {
  config=$1
  baseline=${2:-single_1k}
  python3 experiments/multilevel_special/run_selection_sweep.py "$config"
  python3 experiments/multilevel_special/validate_selection_sweep.py "$config"
  python3 experiments/multilevel_special/summarize_selection_sweep.py \
    "$config" --baseline "$baseline"
}

# 低选择率 0.499% / 0.903% / 9.907% 与 crossing 补点
run_one experiments/multilevel_special/config.amazon_x1_low_selectivity_formal.json
run_one experiments/multilevel_special/config.amazon_x1_low_sel1_crossing_formal.json
run_one experiments/multilevel_special/config.amazon_x1_t1_2000_low_formal.json

# 高选择率 plain、T2 family、T1 sensitivity 和 upper-off/on paired ablation
for s in 25 50 75; do
  run_one experiments/multilevel_special/config.amazon_x1_plain_formal_sel${s}.json plain
  run_one experiments/multilevel_special/config.amazon_x1_current_binary_sel${s}.json
  run_one experiments/multilevel_special/config.amazon_x1_t1_formal_sel${s}.json multi_t1_1000_t2_25k
  run_one experiments/multilevel_special/config.amazon_x1_paired_formal_sel${s}.json
done
```

上述 high-selectivity configs 为历史 raw replay，`search_app` 指向 `runs/` 中的 immutable binary snapshot。若做当前源码的新鲜复测，应复制 config 到新文件，至少同时修改：

1. `search_app` 为当前 `build_ung_rel/apps/search_UNG_index`；
2. `output_root` 为一个全新的目录；
3. `block_index` 为第 5 节新建的对应 sidecar；
4. 保留 Amazon x1 hashes、K=10、100 threads、1000 queries 和原 L-grid。

不得把新 binary 的 aggregate 覆盖到 `results_summary/source/`，除非对六档正式矩阵全部重跑、更新所有 manifest，并重新做同 Recall 选择。

仓库已提供当前源码六档配置和聚合审计入口：

```bash
cd /home/sunyahui/worktrees/FilterVectorCode_multilevel_special

# 每个配置只跑冻结主表中四个内部 operating points；sel_0p5 使用 21 repeats，
# 其余五档使用 7 repeats。output_root 必须是新的 raw 目录。
for s in 0p5_repeat21 1 10 25 50 75; do
  python3 experiments/multilevel_special/run_selection_sweep.py \
    experiments/multilevel_special/config.amazon_x1_current_source_main_sel${s}.json
done

python3 experiments/multilevel_special/audit_current_source_regression.py
```

审计器 fail closed 地检查 24/24 点、budget、repeat 数和单一 binary hash，并输出每点 Recall delta、median、min/max 与 CV。当前结果为：Special Recall 最大漂移 0；fresh/frozen timing ratio 中位数 1.0195，范围 0.9781--1.2188。0.499% 的 20 个 warm samples 仍存在明显调度长尾，尤其 single-level CV=0.680，故短延迟档不能作为严格的 timing reproduction gate。

上段只隔离验证 search-code 漂移，因为它复用了冻结 sidecar。完整 rebuild-to-query 回归另读取三份 fresh bundle。原主表 tuned operating points 在前两份 bundle 上仅有 50% `L=500` 的一份构建低于 R=.85（R=.8495）；对三份 bundle 扫描 `L=500/550/600/650/700` 后，最小共同达标点为 `L=550`，Recall 范围 .8540--.8582，median latency 范围 372.561--379.648 ms。部署复现建议使用这个稳健点；冻结主表仍保留其特定 sidecar 上的 `L=500` 结果。

## 7. 外部系统位置对照

外部方法共享 Amazon x1 query、exact filtered GT、K=10、1000 queries 和 100 threads，但 timing boundary 与 runner 不同，因此只能称为系统位置比较，不能称为统一内核排名。运行前必须准备相应第三方源码/环境与索引。

```bash
cd /home/sunyahui/worktrees/FilterVectorCode_multilevel_special

# NaviX 与 FAVOR：低三档、再高三档
python3 experiments/navix_baseline/run_navix_baseline.py \
  experiments/multilevel_special/config.amazon_x1_navix_low.json
python3 experiments/navix_baseline/run_navix_baseline.py \
  experiments/multilevel_special/config.amazon_x1_navix_final.json
python3 experiments/favor/run_favor_experiment.py \
  experiments/multilevel_special/config.amazon_x1_favor_low.json
python3 experiments/favor/run_favor_experiment.py \
  experiments/multilevel_special/config.amazon_x1_favor_final.json

# Curator；需要 thirdparty/curator-v2、.venv_curator 和 OpenBLAS
OPENBLAS_LIB=/home/graphdb/OpenBLAS/libopenblas.so \
PYTHON_BIN=$PWD/.venv_curator/bin/python \
bash experiments/curator_baseline/run_curator_baseline_with_env.sh \
  experiments/multilevel_special/config.amazon_x1_curator_low.json
OPENBLAS_LIB=/home/graphdb/OpenBLAS/libopenblas.so \
PYTHON_BIN=$PWD/.venv_curator/bin/python \
bash experiments/curator_baseline/run_curator_baseline_with_env.sh \
  experiments/multilevel_special/config.amazon_x1_curator_rebuild_final.json

# 官方 ACORN adapter：gamma/ef screening 后复测选中点
bash experiments/acorn_baseline/run_official_x1_low_selectivity.sh
bash experiments/acorn_baseline/run_official_x1_low_selected.sh
bash experiments/acorn_baseline/run_official_x1_gamma_sweep.sh
bash experiments/acorn_baseline/run_official_x1_selected.sh
```

FAVOR 的 1% prefilter threshold 会使低两档混合 prefilter/graph route；ACORN total 包含 lookup、filter materialization 和 ANN search；Curator 是 Python thread-pool orchestrated batch total；NaviX 使用其 runner 的 neighbor-list 路径。这些差异必须和数值一起报告。

## 8. 从 raw runs 生成 compact evidence

内部每个 sweep 先由 `summarize_selection_sweep.py` 生成 `summary/all_points.csv`，然后把正式 aggregate 和对应 `manifest.json` 复制到 `results_summary/source/` 的既定文件名。外部低选择率用：

```bash
python3 experiments/multilevel_special/summarize_low_external.py \
  --snapshot-root /path/to/external-results-snapshot \
  --output experiments/multilevel_special/results_summary/source/external_low_robust.csv
```

更新 source 后必须执行：

```bash
cd experiments/multilevel_special
python3 generate_paper_results.py
python3 -m unittest -v test_generate_paper_results.py
cd ../..
git diff --check
```

生成器会拒绝混用 search binary、缺失/额外 L-grid、错误 workload/query/selectivity、错误 repeat 数、重复 canonical key 或不完整 run。主报告读取的是生成后的 `paper_results.csv`，不是人工抄写的散点。

## 9. 结果判定与禁止事项

- 唯一质量标准是对相同 exact filtered GT 的端到端 Recall；coverage、block 数和局部 top-K overlap 仅作诊断。
- 固定 L 的耗时不能直接形成加速结论；比较的是达到预声明 Recall 门槛的最快离散实测点。
- `Upper-off`/`Upper-on` paired 对照度量新增第二层本身；T1 调优与 T1/T2 联合最优是另外两种结论。
- 不得把 overlay builder wall 当成完整 from-scratch UNG 建图时间。
- 不得把 repeat/hybrid 数据或 21,834-label 历史主图混入当前 30,723-label Amazon x1 结果。
- raw runs、数据集和第三方索引不提交；compact CSV、manifest、配置、生成器和报告才构成可审计论文闭包。
