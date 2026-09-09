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
- 最新公平 0/1/2 层 formal search binary SHA-256：`6fa4082dae77e99b6c1b3c84c75d324d57cc25e0da32149011379453cfa40087`；builder SHA-256：`c305f48751d990715dbcd448185a65e8391302f38ff6a9410a982c0d5e0b295b`。
- 旧的固定阈值/外部系统论文表使用历史 search binary `f078e1744775a3aefab6cc670b4a72e02b8d7d7118a6d34a6df76cb591287b11`，只作补充证据，不得与最新公平层数表拼接。

当前源码重新编译出的 binary hash可以不同；这表示“新鲜复测”，不能把其结果追加到任一已冻结正式 manifest。内部 runner 会在每个 output root 中创建只读、内容寻址的 binary snapshot，并把 hash 写入 manifest。

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

预期结果：15 项结果生成/provenance 测试通过；生成 60 行 selected results、209 个内部 canonical measured points、447 个外部 canonical measured points 和 7 行构建记录；artifact manifest 无 hash mismatch。这里验证的是提交中的证据闭包，不是重新计时。

## 4. 构建与代码回归

```bash
cd /home/sunyahui/worktrees/FilterVectorCode_multilevel_special
cmake --build build_ung_rel -j16 --target \
  test_special_edge_io test_filter_validation build_special_block_index search_UNG_index

cd build_ung_rel
ctest --output-on-failure -R \
  'ung_build_config|special_block_trie|special_block_free_state|special_candidate_queue|special_edge_io|filter_validation'
cd ..

cd experiments/multilevel_special
python3 -m unittest -v \
  test_multilevel_selection.py test_generate_paper_results.py
cd ../..
git diff --check
```

最终 checkpoint 上的预期结果为 C++ focused tests 7/7、Python tests 29/29。
Python 测试必须从 `experiments/multilevel_special` 目录启动，因为测试模块使用同目录 import；从仓库根直接执行模块路径会因找不到 `run_selection_sweep` 而失败，这不表示算法回归。

### 4.1 Special edge 持久化语义审计

loader 不只检查 CSR offset、ID 范围和 metadata topology，还对每条持久化 special edge 做 owner/child 校验：

- intra edge：source 与 target 都必须是声明 owner 在相应层的 direct member；
- inter edge：source 必须属于声明 parent，target 必须属于其同层 direct child；
- source/target/owner 越界、owner=0、未知 kind 或跨层/非直接 child 目标均拒绝；legacy binary 先读入 staging，整份验证成功后才发布，避免半加载状态；light/heavy sidecar 分流和查询阶段 per-target-block 限流都按 edge owner 的 level 选择中层或上层 ownership，不能固定使用中层 map。
- legacy CSV conversion 与运行时 fallback 共用同一个严格 parser：字段必须恰为 `source,target,block,kind`，kind 只能是 `intra`/`inter`；少字段、未知 kind 和尾随列都会拒绝。显式独立 bundle 仍强制使用 binary light sidecar，light CSV 仅为旧内嵌索引兼容路径。requested heavy binary 若损坏，只能回退到合法 heavy CSV；二者均不可用时 fail closed。
- metadata 还会由 root-label path 重新推导层级关系：`child_block_ids` 必须是最近的同层 block 祖先关系，middle `parent_block_id` 必须是其根路径上的最近 upper block。检查使用按 `(level, root path)` 建立的索引并逐级缩短前缀，不做平方级 block 两两扫描。

不能额外假设 `T2>T1` 会让 upper partition 成为 middle partition 的严格粗化。对一般树形的反例搜索表明，两次独立 uncovered partition 可能使一个 middle block 的直接成员跨越 upper ownership 边界；因此 loader 有意不要求一个 middle block 的全部 direct members 共享同一 upper owner。查询依靠两套 point ownership 在到达 covered upper member 时升级，语义不依赖这个数据特例。当前 Amazon tuned 索引本身恰好完全嵌套：103 个 middle block、8 个 upper block、108 条同层 child edge；102 个 middle block 有最近 upper ancestor，1 个确实位于所有 upper block 之外，逐项重推导均无错挂或漏挂。

`test_special_edge_io` 包含合法 intra/inter、middle/upper target owner 解析、最近同层/上层祖先、允许跨 upper ownership 的独立 partition 反例、upper direct region 必须存在根位于其子树内的 middle-owned 激活点、仅由更大 middle 祖先相交的不可达负例、公开 graph-aware validator 直接拒绝非法 level、converter/runtime CSV 一致性、损坏 heavy binary 的合法 fallback 与无 fallback fail-closed；`test_special_block_free_state` 另外直接覆盖普通点不能越级、middle 到 upper-owned point 的 `1 -> 2` membership activation、未覆盖 upper 与非 containment 不升级。真实数据审计还用当前 binary 完整加载 fresh `T1=2k,T2=25k` sidecar 的 62,938,887 条 special edges，并在 50% workload、L=550 上得到 Recall=.8575、10,000 个返回点、0 个 filter violation。最终审计 binary SHA-256 为 `3947f4389fc23e6111aa17f2a43383170edaefa714c6ba1f45277590b9c11ba3`。

构造 validator 必须遵循对象生命周期。`build_special_blocks()` 完成后，block partition、source layout、ownership 和层级关系已经固定，但 `entry_point_id` 仍为 invalid sentinel；它要等 block-local intra graph 构造后才产生。因此 builder 先调用 `validate_special_block_partition_semantics()` 做一次完整 preflight；intra 完成后只校验本阶段新增的 entry point，再进入昂贵 inter-edge 阶段，避免重复扫描 482,387 groups / 602,453 points。持久化 loader 面对不可信输入时仍调用完整 `validate_special_block_graph_semantics()`。单测同时断言“无 entry point 的合法 partition 通过前者、拒绝于后者”，防止两个 gate 再次被误合并。

最终顺序已用 runner 的只读、内容寻址 snapshot 从头验证：builder SHA-256 `c5cee68dc2dab7c409270eb37bbd73e903a425fbedd6a7bf943a4c52c0653a8f`，T1=2000/T2=25000，57.313 s 完成，输出 111 blocks（8 upper）和 62,941,206 special edges。随后 search snapshot `af73a74de3cac02be1b4e9ae44c2d6eaa68cf73d463d43ddc1f3dc721064990b` 完整加载该 bundle；50%/L550 的 Recall=.8593，检查 10,000 个返回点、0 filter violation。它比既有三次独立重建区间 Recall=.8540--.8582 的上界高 .0011，仍属于 GPU approximate intra 的跨构建波动并超过 .85 门槛。服务器没有 `gpulock`，运行前确认 L20 空闲但没有外部锁，因此 57.313/1.225 s 都只作 fresh correctness/provenance audit，不进入冻结性能主表。

同一最终 bundle 还按六档推荐 L 各跑 1 次计时外过滤审计；0.499%/0.903%/9.907%/24.915%/49.971%/74.994% 的 Recall 分别为 .9101/.9165/.9009/.9023/.8593/.8745。60,000 个结果槽中实际返回 59,991 个点，9 个空槽全部来自 24.915% 档，59,991 个返回点的 filter violation 为 0。compact 证据见 `results_summary/source/final_fresh_six_workload_audit.csv`；由于没有 repeats，这些查询时间不进入性能主表。

最终 source-layout 审计又把门禁扩大到完整 UNG source：group 0 必须是空 sentinel；每个 group 的 label path 必须非空、升序、去重且全局唯一；group ranges 必须按 group id 构成覆盖全部点的连续非空分区；`point_to_group` 必须逐点与 range 双向一致。对每一层还会从所有 block root 重建最近祖先 ownership，核对 direct members、`common_labels` 和完整 Trie 子树点数。构建时只做一次全量 preflight，intra 后只检查新增 entry point；最终内容寻址 builder `c305f487...e0b295b` 从头构建成功（runner wall 59.038 s，metadata/partition 2.659 s，111 blocks/8 upper、62,941,289 edges）。search binary `052e4cc3...88be2a` 在该 bundle 的 50%/L550 上 Recall=.8587、10,000 results、0 violations，并成功加载旧单层 bundle（Recall=.8523、10,000 results、0 violations）。这些仍是单次 correctness/provenance audit，不更新冻结性能表；compact 记录见 `results_summary/source/source_layout_validator_audit.csv`。

最新 binary 的 light-stats 审计中 sidecar 冷加载为 1.835 s，查询 batch 为 0.855 s；此前相邻正确性审计的 load/query 分别在 1.85--2.10 s / 0.97--2.27 s 间波动。这是单次、负载敏感的正确性运行：load 不在 query batch timer 内，query 也没有正式 repeats，因此两者都不更新冻结性能表，更不能用相邻运行之差声称 metadata validator 或 parser 的精确 overhead。

若要审计层级路径本身，在 method env 中额外设置 `UNG_SPECIAL_LIGHT_STATS=0`。同一 binary、sidecar、workload 和 `L=550` 的详细统计运行得到：533/1000 条 query 使用 special graph，512/1000 条搜索 upper block，累计扫描 15,622,167 条 middle edge 和 15,419,434 条 upper edge，发生 57,897 次 upper activation；Recall=.8575，10,000 个结果中 0 个过滤违规。详细统计运行耗时 1029.07 ms，包含 instrumentation overhead，仅证明机制被真实执行。

### 4.2 过滤结果合法性审计

性能 Recall 只能说明返回点与 exact GT 的重合率；还应独立确认每个实际返回点都满足 query filter。`search_UNG_index` 提供默认关闭的 `UNG_VALIDATE_FILTER_RESULTS=1`：搜索 batch 计时停止后，它将 original result id 映射到 reordered graph id，并验证 point labels 包含全部 query labels。审计输出为 `filter_validation.csv`；有任一违规时进程返回 2。

复用第 6 节内部搜索命令时额外设置：

```bash
export UNG_VALIDATE_FILTER_RESULTS=1
```

正确性运行和性能运行必须分开：审计逻辑虽位于 batch timer 外，但会额外读取标签、生成文件，不应把这种单次诊断运行的冷启动时间写入性能主表。已在同一个 fresh-built T1=2k/T2=25k bundle 上覆盖全部六档 workload，L 分别为 1500/3000/10000/10000/550/1000；每档 1000 queries、K=10、100 threads。60,000 个结果槽中实际返回 59,991 个点，9 个 missing 槽全部来自 24.915% 档；所有实际返回点的 `FilterViolations=0`，六档 Recall 分别为 .9101/.9165/.9009/.9023/.8575/.8732，均达到预声明门槛。missing 表示未填满 K，不等于非法结果。审计 binary SHA-256 为 `378e71d77f93c891f67660d33cce36213290dbadee312ee3cac769d6fb3f0313`，逐档 compact 证据见 `results_summary/filter_validation_audit.csv`。

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

这三次历史 fresh build 发生在 runner 增加 immutable builder snapshot 之前，所以旧 manifest 本身没有 `build_binary_sha256`。`current_source_rebuilds.csv` 中的 hash 来自同一审计会话记录，并以 `builder_sha256_source=historical_audit_record` 明确标记，不能误写成逐份旧 manifest 原生签名。现在重新运行 builder 时会得到 `builder_sha256_source=manifest_snapshot`，其 provenance 更强。

构建日志中的 `gpu_intra_enabled=0` 只对应旧的全局 `UNG_SPECIAL_BLOCK_GPU_INTRA` 路径；本实验使用新的 size-routed intra path。判断实际后端应同时查看 `intra_route=1`、`intra_large_backend=jasper_style` 和 `gpu_intra_blocks=24`：3 个小 block 走 exact top-K，84 个中等 block 走 sampled Vamana，24 个大 block 走 FastGrnnd CUDA。不能仅凭旧开关字段断言本次没有使用 GPU。

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

公平 0/1/2 层主结论使用独立的 coarse -> formal 流程。coarse 配置与 1506-point 汇总的 SHA-256 会写入 formal 配置；formal 再显式加入首轮 7-repeat 暴露出的 4 个边界 guard。已有 case 只有在 binary、命令、环境、L-grid 和 7 repeats 全部一致时才复用：

```bash
python3 experiments/multilevel_special/generate_layer_tuning_config.py \
  --output experiments/multilevel_special/config.amazon_x1_layer_tuning_query_extended.json

python3 experiments/multilevel_special/generate_layer_tuning_formal.py \
  experiments/multilevel_special/config.amazon_x1_layer_tuning_query_extended.json \
  --points runs/layer_tuning_query_coarse_amazon_x1/summary/all_points.csv \
  --output experiments/multilevel_special/config.amazon_x1_layer_tuning_query_formal.json

python3 experiments/multilevel_special/run_selection_sweep.py \
  experiments/multilevel_special/config.amazon_x1_layer_tuning_query_formal.json
python3 experiments/multilevel_special/validate_selection_sweep.py \
  experiments/multilevel_special/config.amazon_x1_layer_tuning_query_formal.json
python3 experiments/multilevel_special/summarize_selection_sweep.py \
  experiments/multilevel_special/config.amazon_x1_layer_tuning_query_formal.json
python3 experiments/multilevel_special/select_layer_tuning.py \
  experiments/multilevel_special/config.amazon_x1_layer_tuning_query_formal.json
python3 experiments/multilevel_special/generate_layer_tuning_report.py \
  experiments/multilevel_special/config.amazon_x1_layer_tuning_query_formal.json \
  --build-manifest runs/layer_tuning_build_amazon_x1/manifest.json
```

预期最终规模是 32 个 formal 结构、92 个 structure/workload case、552 个 L 点、18 个 oracle 行、3 个 shared 配置和 0 个开放边界命中。共享阈值结果写入 `results_summary/layer_tuning_shared_summary.csv`，阶段数据写入 `layer_tuning_stage_breakdown.csv`，oracle 上界写入 `layer_tuning_oracle_summary.csv`；raw 文件哈希记录在 `layer_tuning_evidence_manifest.csv`。

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

## 9. 75% 以上正式实验

高选择率结果由 95-case 正式 shortlist 与 20-case 边界补测组成，均为 7 repeats；合并器要求 `(workload, method, Lsearch)` 键互斥，避免静默覆盖。`Lsearch` 的合法下界为 `K=10`。

```bash
# 验证两个 sweep
python3 experiments/multilevel_special/validate_selection_sweep.py \
  experiments/multilevel_special/config.amazon_x1_high_selectivity_formal.json
python3 experiments/multilevel_special/validate_selection_sweep.py \
  experiments/multilevel_special/config.amazon_x1_high_selectivity_formal_boundary_extension.json

# 分别汇总，然后合并唯一实测点
python3 experiments/multilevel_special/summarize_selection_sweep.py \
  experiments/multilevel_special/config.amazon_x1_high_selectivity_formal.json
python3 experiments/multilevel_special/summarize_selection_sweep.py \
  experiments/multilevel_special/config.amazon_x1_high_selectivity_formal_boundary_extension.json
python3 experiments/multilevel_special/combine_selection_points.py \
  experiments/multilevel_special/config.amazon_x1_high_selectivity_formal.json \
  experiments/multilevel_special/config.amazon_x1_high_selectivity_formal_boundary_extension.json \
  --output runs/high_selectivity_formal_boundary_extension_amazon_x1/summary/all_points_combined.csv

# 使用包含结构端点的配置做最终共享/oracle 选择与边界审计
python3 experiments/multilevel_special/select_layer_tuning.py \
  experiments/multilevel_special/config.amazon_x1_high_selectivity_formal_boundary_extension.json \
  --points runs/high_selectivity_formal_boundary_extension_amazon_x1/summary/all_points_combined.csv \
  --boundary-source config
```

预期结果：主 sweep 95/95 cases、570 points；boundary extension 20/20 cases、149 points；合并后 719 个唯一点、18 个 oracle 行、3 个共享配置。共享结构/Lsearch 边界命中均为 0；`selected_boundary_audit.csv` 中仅保留 2 个逐 workload oracle 结构轴提示。正式查询 binary SHA-256 为 `6fa4082dae77e99b6c1b3c84c75d324d57cc25e0da32149011379453cfa40087`。

可提交 evidence 位于：

- `results_summary/source/high_selectivity_formal_all_points.csv`
- `results_summary/source/high_selectivity_formal_manifest.json`
- `results_summary/source/high_selectivity_boundary_manifest.json`
- `results_summary/high_selectivity_shared_summary.csv`
- `results_summary/high_selectivity_shared_points.csv`
- `results_summary/high_selectivity_oracle_summary.csv`
- `results_summary/high_selectivity_oracle_boundary_audit.csv`

100% 是空 predicate 的独立 unfiltered control；不得与 96.7% 连续插值，也不得据此外推 97%--99%。

## 10. 结果判定与禁止事项

- 唯一质量标准是对相同 exact filtered GT 的端到端 Recall；coverage、block 数和局部 top-K overlap 仅作诊断。
- 固定 L 的耗时不能直接形成加速结论；比较的是达到预声明 Recall 门槛的最快离散实测点。
- 公平层数主结论使用每层独立调优后的共享配置；逐 workload oracle 只能作为上界。旧的 `Upper-off`/`Upper-on` paired 对照可用于机制说明，但不替代该主结论。
- 不得把 overlay builder wall 当成完整 from-scratch UNG 建图时间。
- 不得把 repeat/hybrid 数据或 21,834-label 历史主图混入当前 30,723-label Amazon x1 结果。
- raw runs、数据集和第三方索引不提交；compact CSV、manifest、配置、生成器和报告才构成可审计论文闭包。
