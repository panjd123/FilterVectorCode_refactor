# 多层 Special Block：方法、构建代价与端到端查询性能

更新时间：2026-09-06

## 摘要

我们把原来的单层 Special Block 扩展成固定的 two-level 分层图 overlay：保留中层 block，再独立增加覆盖更大合法子树的上层 block。当前实现只支持这两个 block 层级，不宣称任意 N 层。搜索候选携带单调递增的层级状态，只能按 `普通图 -> 中层 -> 上层` 逐级获得使用更高层边的权限。它不是简单地把阈值从 1000 改成 10000，也不是在一张图上无条件增加长边。

实验使用 Amazon 原始 100% x1 数据（602,453 个 768 维向量，482,387 个 group，30,723 个实际 label），六档真实查询平均选择率为 0.499%、0.903%、9.907%、24.915%、49.971% 和 74.994%。每档 1000 条 query，K=10，100 个搜索线程，质量由同一 exact filtered ground truth 的端到端 Recall 定义。报告统一采用 `.90/.90/.90/.90/.85/.87` Recall 门槛；所有性能比较均选择达到对应门槛的最快实测点，不插值、不外推。若扫描内未达门槛，则选择最高 Recall，并在并列时取更快点，明确标记质量上限。

核心结论有三点：

1. 新增上层本身确实有效。统一搜索 binary 后，固定 `T1=1k`，原始多层方法族相对单层在 24.915%、49.971%、74.994% 三档分别加速 **1.289x、1.709x、2.320x**。前两档赢家是 paired `T2=25k` 点；75% 的方法族赢家来自独立但同 binary 的 `T2=50k` sweep。固定 `T2=25k` 的同轮受控比较为 **1.289x、1.709x、2.229x**。
2. `T1` 调优是另一项独立收益。同轮 paired formal 中，最终 `T1=2k,T2=25k` 相对 `T1=1k,T2=25k` 在上述三档再加速 **1.514x、1.067x、1.154x**；相对 plain UNG 在 49.971% 和 74.994% 达到 **14.48x 和 57.38x**。25% 及以下并非稳定优势区间。
3. 方法有清楚的适用边界：0.903%、9.907%、24.915% 上 plain UNG 更快；0.499% 的 **1.044x** 小于测量波动尺度，应视为基本持平，尚未建立统计显著优势。多层的论文价值集中在宽过滤条件下以更少 `Lsearch` 达到同 Recall，而不是宣称对所有选择率全面更快。

机器生成的完整表见 `results_summary/paper_results.md`，选择结果见 `paper_results.csv`，构建数据见 `build_results.csv`。

## 1. 问题：为什么需要第二层 block

原始查询图包含普通组内边和跨组边。单层 Special Block 又把 label trie 上一个完整合法子树对应的点组织成一张附加图；查询完整覆盖该 block 时，可以使用这些 special edges 快速在合法区域内移动。

单层只有一个空间尺度。若 `T1=1000`，它擅长在约千点规模的合法子树内导航，却缺少跨越更大合法区域的长程结构。直接把 T1 改成 10000 会丢掉原来的中层结构，因此不等价于“中层 + 上层”。多层方法的目标是同时保留局部导航和更长尺度导航。

## 2. 方法

### 2.1 分层构建

构建器在同一 group trie 上执行两次独立的 bottom-up uncovered partition：

- 中层以阈值 `T1` 构建，保留原单层 block 的语义；
- 上层以更大的 `T2>T1` 独立构建；
- 一个 group/point 可以同时属于一个中层 block 和一个上层 block，因此维护两套 ownership；
- 普通图、中层 special graph、上层 special graph 同时存在；上层不会替换中层。

这里的“两次独立”还有一个容易误解的边界：`T2>T1` 不构成“上层一定是中层的严格粗化”的一般性定理。某个中层 block 的直接成员理论上可以跨过一个位于其内部的 upper block 边界；因此实现维护两套逐点 ownership，并按候选点当前所在层判断是否升级，不能只保存一张 middle-to-upper 映射来替代它们。`parent_block_id` 仅记录一个 middle block 根路径的最近 upper 祖先，`child_block_ids` 则记录同一层中的最近 block 祖先关系。Amazon 当前 tuned 索引恰好呈完全嵌套形态，但论文方法与正确性不能依赖这个数据特例。

Special Block 被视为一种特殊 group，因此其图构建复用既有 group 图路径：小候选域走 exact top-K，较大候选域走 sampled Vamana 或 GPU 路径，跨 block 边走与普通跨组边相同的启发式 dispatch。正式 T1/T2 索引的实际日志显示：`n<=2048` 的 block 使用 CPU exact top-K，`2048<n<8192` 使用 CPU sampled-Vamana，`n>=8192` 使用 `jasper_style`/FastGrnnd CUDA；本批 block 间 pair 均走 CPU graph search。因此下文不能误称为 GPU/GPU 构建结果。

### 2.2 逐层查询授权

每个候选点维护 `activation_level in {0,1,2}`：

```text
level 0: 只能走普通边
   | 完整进入合法中层 block
   v
level 1: 可走普通边和中层 special edge
   | 到达查询完整覆盖的上层 block 根/成员
   v
level 2: 可走普通边、中层边和上层边
```

状态只能上升，且一条 transition 最多提升一级，所以普通候选不能越过中层直接使用上层边。同一点若通过更高层路径再次到达，候选队列原位升级其 level 并允许重新展开，但不会重复占用 `Lsearch` 或 top-K 槽。主扩展与 pre-expand 共用同一个 edge-transition helper，避免两条执行路径语义漂移。

持久化加载器会 fail closed 地重建并核对两类结构不变量：同层 child 必须指向最近的同层 Trie block 祖先；middle 的 `parent_block_id` 必须指向其根路径上最近的 upper block（若不存在则为 0）。它不强制每个 middle block 的全部成员共享同一个 upper owner，因为独立 partition 在一般树形上并不保证该性质。真实 `T1=2k,T2=25k` Amazon 索引包含 103 个 middle block、8 个 upper block、108 条同层直接父子边；所有关系均通过从 root-label path 重新推导的全量审计。

加载器还验证每个 upper block 至少有一个 direct-member group 被“根位于该 upper 子树内”的 middle block 拥有。这不是要求两层 partition 严格嵌套，而是逐级状态机的可达性条件：若 upper 的整个直接区域都没有这种 middle-owned point，候选就不存在合法的 `1 -> 2` 激活位置。仅由更大的 middle 祖先拥有并不充分，因为查询完整覆盖 upper 并不保证覆盖该祖先。该性质由 `T2>T1` 的 bottom-up uncovered 构造保证；损坏或手工拼接但不可达的 sidecar 会被拒绝。

这套结构的预期收益不是“相同 L 下每次展开更便宜”。special edge 会增加可扫描邻居，相同 L 下甚至可能更慢。真正目标是让 Recall-L 曲线左移，以更小 L 达到相同 Recall。

## 3. Baseline、超参数与公平性

### 3.1 内部 baseline

| 方法 | 图结构 | 主要超参数 | 回答的问题 |
|---|---|---|---|
| Plain UNG | 只有普通组内/跨组图 | `Lsearch` | 不使用 Special Block 时的系统性能 |
| Single-level | 普通图 + 中层 overlay | `T1=1k`, `Lsearch` | 一层 block 能带来什么 |
| Original Multi-level | 普通图 + 中层 + 上层 | `T1=1k`, `T2=4k/10k/25k/50k`, `Lsearch` | 只增加第二层是否有效 |
| Tuned Multi-level | 同上 | `T1=2k,T2=25k`, `Lsearch` | 联合调优后的最佳当前版本 |
| Upper-off | 与多层使用同一索引但禁用 level 2 | 查询开关 | 排除重建随机性，验证上层路径的机制作用 |

原始多层与单层的比较固定 T1=1k，因而能单独归因到上层。最终调优多层包含 T1 改动，不能把它相对单层或 plain 的全部收益都归因于第二层。

### 3.2 外部 baseline

| 系统 | 实现与本实验角色 | 搜索参数 | 计时边界 |
|---|---|---|---|
| FAVOR | 项目已接入的官方式 filtered-search 路径，按选择率在 prefilter 与 graph route 间分流 | `Lsearch` | 1000-query batch total；过滤条件构造与选择率估计在 timed batch 外 |
| NaviX | 项目 runner 中的 NaviX 系统路径，使用 `cpu_min_super_sets` 等既有入口流程 | `Lsearch` | 1000-query batch total |
| Curator v2 | 独立官方仓库/FAISS 扩展重建的 x1 索引 | `search_ef` | 包含 filter preparation 与搜索的 batch total |
| Official ACORN | 官方 commit `c259f11c` 加 Amazon x1 adapter；显式传入 containment bitmap | `gamma`, `efSearch` | 同时报 lookup、materialize、ANN search 和 total |

ACORN adapter 修复了官方 hybrid search 初始 candidate 未检查 `filter_map` 的过滤泄漏；所有纳入数据均为零 filter violations。五档 gamma（1/2/4/8/12）均实际扫描，不能假设 gamma 越大越好。

## 4. 数据和指标

| 项目 | 口径 |
|---|---|
| 数据 | Amazon 原始 100% x1；不使用 xN repeat/hybrid |
| 规模 | 602,453 points，768 dimensions |
| 标签结构 | 482,387 groups，30,723 个实际 labels |
| Workload | 6 档，每档 1000 条真实 query |
| 平均选择率 | 0.499%、0.903%、9.907%、24.915%、49.971%、74.994% |
| 搜索 | K=10，100 threads |
| 质量 | 对同一 exact filtered GT 的端到端 Recall |
| 内部时间 | 7 repeats；排除 cold repeat 后 6 个 warm batch 的 median，同时保留 mean/CV |
| 外部时间 | 5 measured repeats（历史高选择率部分为 cold 后 4 warm repeats）；以 total median 为主 |

Block 覆盖率、入口数、边数、局部 top-K overlap 只用于解释机制，不是准确率。唯一论文级质量标准是完整 filtered query 的 Recall。

内部正式结果全部由同一个 immutable 搜索 binary 产生，其 SHA-256 为 `f078e174...287b11`。结果生成器逐份检查 run manifest；若出现不同 binary、未完成运行或非零退出会直接拒绝生成主表。高选择率另做了一轮 paired formal：Single、同索引 upper-off、upper-on 和 T1=2k 在同一批次中依次运行，降低跨时段系统噪声。25%/50% 的方法族赢家来自该 paired 轮；75% 的方法族赢家是同 binary 的独立 T2=50k sweep，而 fixed-T2 因果比较使用 paired T2=25k。

## 5. 六档内部主性能

下表各方法独立选择达到门槛的最快实测点。`未达` 表示给定扫描上限内未达到门槛，此时只显示最高 Recall 点。

| 选择率 / Recall门槛 | Plain UNG | Single T1=1k | Original Multi T1=1k | Tuned Multi T1=2k,T2=25k | Tuned / Plain |
|---|---|---|---|---|---:|
| 0.499% / .90 | L1500, R=.9134, 44.706 ms | L2000, R=.9133, 48.049 ms | T2=10k L2000, R=.9133, 48.224 ms | L1500, R=.9101, 42.829 ms | **1.044x** |
| 0.903% / .90 | L2000, R=.9271, 60.656 ms | L4500, R=.9043, 152.858 ms | T2=50k L4500, R=.9043, 141.684 ms | L3000, R=.9169, 108.401 ms | **0.560x** |
| 9.907% / .90 | L15000, R=.9016, 675.142 ms | 未达：L20000, R=.8948, 3455.295 ms | T2=10k L15000, R=.9088, 2669.605 ms | L10000, R=.9009, 1793.005 ms | **0.377x** |
| 24.915% / .90 | L22000, R=.9026, 1648.470 ms | L18000, R=.9037, 5828.470 ms | T2=25k L14000, R=.9012, 4520.320 ms | L10000, R=.9023, 2986.040 ms | **0.552x** |
| 49.971% / .85 | L40000, R=.8561, 5551.180 ms | L1800, R=.8523, 698.601 ms | T2=25k L600, R=.8584, 408.749 ms | L500, R=.8518, 383.260 ms | **14.484x** |
| 74.994% / .87 | L110000, R=.8737, 44264.200 ms | L4000, R=.8703, 1983.815 ms | T2=50k L1200, R=.8725, 854.962 ms | L1000, R=.8729, 771.469 ms | **57.377x** |

### 5.1 第二层的独立贡献

在 T1 固定为 1k 时，Original Multi 相对 Single：

| 选择率 | Single | Original Multi | 加速 | 解释 |
|---:|---:|---:|---:|---|
| 0.499% | 48.049 ms | 48.224 ms | 0.996x | 基本持平，上层没有可见收益 |
| 0.903% | 152.858 ms | 141.684 ms | 1.079x | 小幅收益 |
| 9.907% | 未达 R=.90 | 2669.605 ms 达标 | 不计算 | 第二层改善质量可达性，但耗时仍差于 plain |
| 24.915% | 5828.470 ms | 4520.320 ms (T2=25k) | **1.289x** | 上层开始稳定减少所需 L |
| 49.971% | 698.601 ms | 408.749 ms (T2=25k) | **1.709x** | 宽查询更常授权上层边 |
| 74.994% | 1983.815 ms | 854.962 ms (T2=50k) | **2.320x** | 上层长程导航价值最大；paired T2=25k 为 2.229x |

上表比较的是每个方法族达到 Recall 门槛的最快实测点，用来回答“增加第二层后系统能否用更小预算达标”。更强的因果消融是在同一 `T1=1k,T2=25k` 索引、同一 L 下只切换上层权限：

| 选择率 | L | Upper-off Recall / 时间 | Upper-on Recall / 时间 | 结论 |
|---:|---:|---:|---:|---|
| 24.915% | 14000 | .8946 / 4200.080 ms | .9012 / 4520.320 ms | 上层以 7.6% 当轮时间代价跨过 .90 门槛 |
| 49.971% | 600 | .7633 / 338.707 ms | .8584 / 408.749 ms | 上层带来 +9.51 个 Recall 百分点并跨过 .85 |
| 74.994% | 1200 | .7678 / 672.272 ms | .8759 / 890.059 ms | 上层带来 +10.81 个 Recall 百分点并跨过 .87 |

不能把 upper-on 的固定 L 时间增加误判为方法变慢：它实际扫描了新增的合法上层边，因此单次预算略贵；收益体现在 Recall-L 曲线左移，使方法能以远小于 Single 的 L 达到质量门槛。

### 5.2 T1 调优的独立贡献

固定 T2=25k 的 7-repeat 正式结果：

| 选择率 | T1=500 | T1=1000 | T1=2000 | 最佳结论 |
|---:|---|---|---|---|
| 24.915%, R>=.90 | 未达（最高 .8905） | L14000, R=.9012, 4520.320 ms | L10000, R=.9023, 2986.040 ms | T1=2k 比 T1=1k 快 **1.514x** |
| 49.971%, R>=.85 | L600, R=.8594, 426.165 ms | L600, R=.8584, 408.749 ms | L500, R=.8518, 383.260 ms | T1=2k 快 **1.067x** |
| 74.994%, R>=.87 | L1200, R=.8747, 863.141 ms | L1200, R=.8759, 890.059 ms | L1000, R=.8729, 771.469 ms | T1=2k 快 **1.154x** |

T1 越小并不越好：T1=500 产生更多中层 blocks 和边管理开销，25% 档甚至在 L=30k 仍未达到 .90。T1=500 来自同 binary 的独立正式运行；T1=1000/2000 数字来自最新 paired formal，因此只把后两者的比值作为稳定的同轮加速结论。

### 5.3 跨独立建图的质量鲁棒性

冻结主表固定一份 immutable sidecar，适合比较方法，但不能回答“重新构建近似图后 operating point 是否仍过线”。为此，当前源码用同一 builder binary 独立构建三次 `T1=2k,T2=25k`，再用同一 search binary 查询。block partition、trie 和 regular overlay 的 SHA-256 三次完全一致；`special_edges.bin` 不一致。差异来自 large-block GPU FastGrnnd 的并行近似 intra graph，并沿 child graph search 传到 inter edge 目标；它不是输入、block 定义或持久化错误。

冻结主表的六档 tuned operating points 在前两份 fresh bundle 上共 12 个结果，只有第二份 bundle 的 49.971% `L=500` 未达到门槛（R=.8495）；其余 11 个均达标。对该敏感档追加三份 bundle、`L=500/550/600/650/700`、每点 20 个 warm repeats：

| Lsearch | rebuild 1 Recall / median | rebuild 2 Recall / median | rebuild 3 Recall / median | 三次均达 R>=.85 |
|---:|---:|---:|---:|---:|
| 500 | .8529 / 353.308 ms | .8495 / 356.679 ms | .8537 / 358.820 ms | 否 |
| **550** | **.8575 / 372.561 ms** | **.8540 / 373.969 ms** | **.8582 / 379.648 ms** | **是** |
| 600 | .8619 / 391.094 ms | .8585 / 390.760 ms | .8629 / 396.928 ms | 是 |
| 650 | .8651 / 410.408 ms | .8623 / 411.116 ms | .8661 / 414.754 ms | 是 |
| 700 | .8699 / 426.075 ms | .8662 / 429.209 ms | .8691 / 433.291 ms | 是 |

因此，`L=500` 保留为冻结 sidecar 上的最快论文点；若部署要求跨重建仍稳健达标，推荐 `L=550`。后者三次 Recall 下界为 .8540，median latency 上界为 379.648 ms。这里的稳健性是经验性的三次独立构建结果，不是对所有 GPU 调度的形式化保证。

## 6. 构建成本

`total builder wall` 是在已有 base UNG index 上运行完整独立 Special Block builder 的墙钟时间；`special edge stage` 只包含 special intra/inter edge 阶段。两者不能混称。

| 配置 | 中层块 | 上层块 | total builder wall | special edge stage | loaded allocated bytes |
|---|---:|---:|---:|---:|---:|
| Single T1=1k | 210 | 0 | 72.927 s | 53.472 s | 未记录 |
| Multi T1=1k,T2=4k | 210 | 46 | 94.425 s | 73.003 s | 未记录 |
| Multi T1=1k,T2=10k | 210 | 22 | 89.051 s | 68.690 s | 未记录 |
| Multi T1=1k,T2=25k | 210 | 8 | 92.024 s | 70.157 s | 864,059,967 B |
| Multi T1=1k,T2=50k | 210 | 5 | 94.396 s | 70.898 s | 未记录 |
| Multi T1=500,T2=25k | 442 | 8 | 102.373 s | 未单列 | 918,423,543 B |
| Tuned Multi T1=2k,T2=25k | 111 | 8 | **58.760 s** | 未单列 | **812,811,383 B** |

原始 T1=1k 的第二层使 builder wall 相对 single 增加 22.1%--29.5%。T1=2k 则减少中层 block 数，使构建时间比 T1=1k,T2=25k 低 36.1%，甚至低于单层 T1=1k；这是阈值调优和工作量变化的结果，不能解释成“多构建一层本身更便宜”。

表中的 loaded bytes 来自修复 ownership-map 计数遗漏之前的运行日志，T1=1k/T2=25k 与 T1=2k/T2=25k 均约低估 4.1 MiB；sidecar 文件大小和构建墙钟不受影响。代码现已把 upper group/point ownership maps 纳入统计，但在不重建相同索引前，论文不应把旧 loaded-byte 数写成精确峰值。

当前源码对 tuned 配置又做了三次完整 fresh build：60.250 / 56.285 / 56.916 s，中位数 56.916 s，相对冻结记录 58.760 s 为 0.969x（即快约 3.1%），没有构建性能回归。三次均产生 111 个中层块和 8 个上层块；partition/trie/regular overlay 逐字节一致，inter edge 数固定为 14,965,020。GPU large-block intra 的原子并行更新不是 bitwise deterministic，intra edge 数在 47,973,867--47,978,904 间变化（跨度 0.0105%）；因此论文质量复现应使用端到端 Recall 和跨重建稳健点，而不能要求 approximate edge sidecar 字节完全相同。

这三次运行早于 builder immutable-snapshot 功能，旧 manifest 未原生携带 builder hash；compact CSV 将审计会话记录的 `6ff471a8...50692` 标为 `builder_sha256_source=historical_audit_record`。后续 fresh run 会在 manifest 内记录 `manifest_snapshot`。另需注意，日志的 `gpu_intra_enabled=0` 是旧全局开关状态；本配置实际启用了 `intra_route=1`，其中 24 个大 block 走 `jasper_style` FastGrnnd CUDA。

为判断是否值得用确定性换取字节级复现，还做了 CPU Vamana large-block 对照：总 builder wall 为 1131.863 s，其中 intra 为 1059.060 s；GPU FastGrnnd 三次中位 total/intra 分别为 56.916/10.453 s。按完整 builder wall，GPU 路径快 **19.89x**；按被替换的 intra stage，约快 **101.32x**。因此默认回退 CPU 不可取；论文与部署更合理的策略是保留 GPU approximate build，并为质量门槛设置跨重建余量。CPU 与 GPU 会构造不同的近似图，该对照只度量构建代价，不能声称图质量完全等价。

作为量级参照，base UNG 历史 metadata 为 index build 211.819 s、含 additional edges 为 215.406 s。该时间与独立 overlay builder 边界不同，不能直接相加后声称严格的 from-scratch speedup。外部完整索引构建记录：FAVOR 76.174 s，Curator 102.207 s，NaviX 1316.957 s；ACORN gamma=1/2/4/8/12 分别约 17.268/53.413/80.696/135.577/212.311 s core。跨系统索引语义和计时边界不同，因此只报告量级，不计算构建加速比。

## 7. 外部系统位置

下面使用各 runner 自己定义的 total batch median；由于过滤准备边界不同，这是系统位置比较，不是严格统一的端到端计时排名。ACORN 括号内补充 core ANN search；未达门槛者只报告扫描内质量上限，不参与达标性能排序。

| 选择率 / 门槛 | Plain | Tuned Multi | FAVOR | NaviX | Curator | ACORN |
|---|---|---|---|---|---|---|
| 0.499% / .90 | R=.9134, **44.706 ms** | R=.9101, **42.829 ms** | R=.9224, 258.117 ms | 未达：R=.7649 | R=.9430, 2624.251 ms | 未达：R=.7391, 1188.030 ms (core 876.904) |
| 0.903% / .90 | R=.9271, **60.656 ms** | R=.9169, 108.401 ms | R=.9094, 403.783 ms | 未达：R=.7899 | R=.9582, 2886.603 ms | 未达：R=.7907, 1790.202 ms (core 1556.979) |
| 9.907% / .90 | R=.9016, **675.142 ms** | R=.9009, 1793.005 ms | R=.9204, 2641.120 ms | R=.9036, 2848.300 ms | R=.9816, 1420.208 ms | 未达：R=.8968, 7873.439 ms (core 7642.445) |
| 24.915% / .90 | R=.9026, **1648.470 ms** | R=.9023, 2986.040 ms | R=.9133, 1754.680 ms | R=.9153, 2413.740 ms | R=.9744, 1838.554 ms | R=.9011, 8207.948 ms (core 7943.332) |
| 49.971% / .85 | R=.8561, 5551.180 ms | R=.8518, 383.260 ms | R=.8644, **200.930 ms** | R=.8695, 2612.825 ms | R=.9742, 1590.117 ms | R=.8531, 3106.804 ms (core 2864.419) |
| 74.994% / .87 | R=.8737, 44264.200 ms | R=.8729, 771.469 ms | R=.8920, **265.841 ms** | R=.9179, 2065.955 ms | R=.9629, 1942.862 ms | R=.8710, 1148.216 ms (core 843.998) |

这里有三个必须保留的解释边界：

- 外部系统的 Recall 高于门槛幅度不同，因此这是“达到门槛的最快离散实测点”，不是连续曲线上的精确同 Recall 插值。
- FAVOR 在 0.499% 和 0.903% 分别有 88.3% 和 79.9% query 走 prefilter；9.907% 有 9.5% prefilter，其余走 graph。因此低选择率 FAVOR 是混合系统结果，不是纯图搜索。
- Curator 的低选择率 total 被 filter preparation 主导；小 `search_ef` 不一定更快。ACORN 的 total 包含合法集合 lookup/materialization，不能只拿 core search 与其他系统 total 比。

## 8. 机制解释

上层只在 query 完整覆盖 block 时授权。25%/50%/75% workload 中，完整覆盖上层 block 的 query 比例约为 26.0%/51.3%/77.3%，因此宽查询更常利用长程边，Recall-L 曲线更明显左移。

但选择率不是唯一变量。真实 query 的 label 组合、覆盖 block 的位置、入口分布与普通图可达性都会影响效果；这解释了 0.499% 的小幅正收益以及 0.903%/9.907% 的负收益并存。论文应表述为“收益与查询对层级 block 的完整覆盖结构相关，并在高选择率 workload 上最稳定”，而不是简单声称随选择率单调增长。

同一多层索引的 upper-on/off 消融在已测点上显示启用 level 2 提高 Recall，运行计数也观察到上层激活、节点展开和边扫描；因此第二层收益来自真实上层路径，而不只是重建随机性。

最终详细计数回归进一步直接验证了这条路径。在 50% workload、`L=550` 的 1000 条 query 中，533 条使用 special graph，512 条实际搜索 upper block；累计搜索 778 个 middle block 和 3426 个 upper block，扫描 15,622,167 条 middle special edge 与 15,419,434 条 upper special edge，并记录 57,897 次 upper activation。该诊断将 `UNG_SPECIAL_LIGHT_STATS=0`，因此其 1029.07 ms 单次耗时包含详细计数开销，只用于机制验证，不进入性能主表。Recall 仍为 .8575，10,000 个返回点的过滤违规为 0。

## 9. 论文可主张内容与局限

可以主张：

- 一种保留已有中层、叠加更大合法子树导航层的层级 Special Block overlay；
- 逐级 activation、per-point 去重和原位 level upgrade 保证搜索权限语义；
- 在 Amazon x1 的 50%/75% 宽过滤 workload，方法族最佳相对单层分别达到 1.709x/2.320x；固定 T2=25k 的 paired 结果为 1.709x/2.229x；联合调优后相对 plain 达 14.48x/57.38x；
- 通过六档选择率、T1/T2 调参和外部系统比较展示适用区间与负结果。

不能主张：

- 不能把最终 tuned-vs-single 的全部提升归因于新增第二层；
- 不能说方法在所有选择率优于 plain 或 FAVOR；
- 不能用固定 L 耗时、coverage、入口数或局部 top-K overlap 代替同 Recall 端到端结果；
- 不能把独立 overlay builder wall 写成完整 from-scratch index build；
- 不能把 21,834-label hybrid 主图的历史结果与当前 30,723-label x1 结果混用。

当前实现还有一个性能限制：多层 metadata 尚未进入 GPU free-distance batch scratch，因此发现 upper blocks 时查询会回落到 activation-gated CPU path，保证语义正确。若未来把 per-edge level 一并带入 batch scratch，有机会继续降低查询成本，但必须重新做端到端 Recall 验证。

## 10. 复现与证据

- 隔离仓库：`/home/sunyahui/worktrees/FilterVectorCode_multilevel_special`；分支 `codex/multilevel-special-block-20260905`。服务器 Git 过旧不支持 native worktree，故使用 `git clone --shared`，原始脏仓库未被修改。
- 实现与结果检查点：`7a2bf4635eac43d174100770838daf1b3a10fa58`；查询二进制 SHA-256：`f078e1744775a3aefab6cc670b4a72e02b8d7d7118a6d34a6df76cb591287b11`。
- 输入 provenance：base labels SHA-256 `aec768bba7092af445252835be3f1ef7f708305ea646af19ae72738df3dd2f96`，main-index labels SHA-256 `ddb3f616c27626afe6b20bf633aca5e9a1efd31d82bd505b163f59fc79f4dd56`，source fingerprint `91d78580ae29f468`；逐轮完整值与 binary hash 见 `experiments/multilevel_special/results_summary/source/*_manifest.json`。
- focused C++ tests 7/7 通过；多层 Python tests 14/14 通过；结果生成/provenance tests 17/17 通过；所有正式 selection sweep validator、六档 current-source validator 与 rebuild validator 通过。
- 持久化边界还逐条验证 special edge 的归属语义：intra edge 的 source/target 必须同属声明 block 的 direct members；inter edge 的 source 必须属于声明 parent，target 必须属于该 parent 的同层 direct child。越界 ID、owner=0、未知 kind 或 owner/child 不一致均 fail closed，legacy binary 采用 staged load，失败时不会发布部分结果；light/heavy 分流与查询 per-target-block 限流也按 edge owner 的层级选择对应 ownership map。legacy CSV converter 和 runtime fallback 共用严格 parser；损坏的 requested heavy binary 只有存在合法 CSV 时才允许回退。该校验已完整加载一份 fresh `T1=2k,T2=25k` sidecar 的 62,938,887 条边；随后 50% workload、L=550 返回 10,000 个点，过滤违规为 0、Recall=.8575。最新正确性运行 binary SHA-256 为 `550f04c4...204c`，sidecar load/query 单次为 1.893/0.973 s；这些负载敏感的单次数据只证明持久化语义和查询正确性，不更新冻结性能表。
- 最终可达性审计进一步拒绝“upper block 没有可由其子树内部 middle block 激活的 direct member”这一类 metadata；公开 graph-aware validator 也会自行先检查 level、ID、ownership 等基础格式，不能被独立调用绕过。最终 binary SHA-256 为 `3947f4389fc23e6111aa17f2a43383170edaefa714c6ba1f45277590b9c11ba3`，完整加载真实 sidecar 的 62,938,887 条边；light-stats 正确性运行 Recall=.8575、10,000 个结果、0 违规；紧邻版本的详细统计运行则实证 512/1000 条 query 搜索到 upper block。两次均为单次审计，不替换冻结性能表。
- 构建端采用两阶段语义门禁。`partition preflight` 紧接两次 bottom-up partition，只检查此时已经确定的 block ID/level、direct members、point count、最近同层/上层祖先和 upper 激活可达性；intra graph 随后为每个 block 选择局部入口点，完整 graph validator 在进入昂贵 inter-edge 阶段前检查 entry point，并在整个 special overlay 后再次防御性复核，之后才构建 regular overlay 和发布 sidecar。这样既能尽早拒绝坏状态，也不会错误要求尚未产生的 entry point。最终内容寻址 builder `c5cee68dc2dab7c409270eb37bbd73e903a425fbedd6a7bf943a4c52c0653a8f` 从头构建 T1=2k/T2=25k 成功（57.313 s，111 blocks、其中 8 upper，62,941,206 special edges）；对应 current search binary `af73a74de3cac02be1b4e9ae44c2d6eaa68cf73d463d43ddc1f3dc721064990b` 在 50%/L550 得到 Recall=.8593、10,000 个结果、0 违规。该 Recall 比此前三次重建区间 .8540--.8582 的上界高 .0011，符合 GPU approximate intra 图的已知跨构建波动，且稳健超过 .85 门槛。两次运行均未由 `gpulock` 隔离，单次时间只作构造/正确性证据，不替换冻结性能表。
- 同一最终 builder/search/bundle 的六档正确性回归也全部达到预声明门槛：Recall 依次为 .9101/.9165/.9009/.9023/.8593/.8745；60,000 个结果槽中实际返回 59,991 个点，过滤违规为 0。各档只运行一次，作用是证明最终代码在完整选择率范围内仍满足端到端语义，不用于更新主性能表。
- 最终 source-layout gate 不再只检查 block 引用到的 group：它验证 group-zero sentinel、label path 规范与唯一性、所有 group ranges 对点空间的无缝非重叠覆盖、逐点 ownership、每层最近 block-root direct ownership、`common_labels` 和完整 Trie 子树点数。当前 builder `10de4d7e...55f290f` 从头构建成功（runner wall 62.352 s，metadata 2.420 s，111 blocks/8 upper、62,941,374 edges）；search binary `60220cbc...a82e18` 在该 bundle 的 50%/L550 上 Recall=.8565、10,000 results、0 violations，并以同一 binary 加载旧单层 bundle得到 Recall=.8523、10,000 results、0 violations。单次时间仅作正确性/兼容性证据，不替换冻结性能表。
- 当前源码提供默认关闭的 `UNG_VALIDATE_FILTER_RESULTS=1` 审计开关：它在搜索计时结束后，把返回的 original point id 映射回图内 reordered id，并逐点检查其 label set 是否包含 query labels。使用同一个 fresh-built `T1=2k,T2=25k` bundle、同一 audit binary，在六档 workload 的推荐 operating point 上共审计 60,000 个结果槽；实际返回 59,991 个点，9 个槽为空，59,991 个返回点的过滤违规均为 0，且六档 Recall 都达到预声明门槛。空槽只表示该 query 未填满 K，不是非法返回。该运行使用审计 binary `378e71d7...0313`，只证明过滤合法性；单次冷启动时间不进入性能主表，也不替换冻结 binary 的结果。逐档 compact 记录见 `results_summary/filter_validation_audit.csv`。
- `generate_paper_results.py` 从 compact source CSV 和每轮 manifest 重建 `paper_results.csv`、`paper_results.md`、`build_results.csv`、`internal_canonical_measured_points.csv` 与 `external_canonical_measured_points.csv`。每个内部 aggregate 的 method/workload/L 网格和 warm-repeat 数都必须与 manifest 完全一致；构建数据来自 `results_summary/source/build_results_source.csv`，不再硬编码在生成器中。Canonical aggregate 按 `(workload, method family, variant, budget)` 去重；相同点的 paired rerun 以更高优先级覆盖旧统计，因此它不是历史执行次数的逐行并集。raw `runs/` 和大型第三方索引不提交。
- `results_summary/artifact_manifest.csv` 保护生成器、测试、关键配置、主报告与生成表组成的论文结果闭包；`AGENT_KANBAN.md` 和 `WORKTREE_HANDOFF.md` 是会随 checkpoint 更新的运维状态文档，故不纳入该闭包。
- 从 compact evidence 审计、当前源码 fresh rerun 到历史 raw replay 的精确命令与边界见 `docs/reports/MULTILEVEL_SPECIAL_BLOCK_REPRODUCE_CN.md`。

当前源码另用 binary `88d7dba189478cd11402a8433076d220c7ad68ca9f9a6366118f78430752f37f` 重跑了六档主表的 24 个内部 operating points。它是“新 search binary + 冻结 sidecar”的版本漂移回归，不替换冻结论文 binary 和主表：18 个 Special 点的 Recall 最大漂移为 0；24 点 fresh/frozen 耗时比中位数为 1.0195，范围为 0.9781--1.2188。0.499% 档采用 21 次运行（1 次 warm-up、20 次计时），仍可观察到短任务调度长尾，因此该档保留 min/max/CV，只作为回归审计，不设严格 timing gate。逐点结果见 `results_summary/current_source_regression.csv`。从头重建的独立结果另见 `current_source_rebuilds.csv`、`current_source_rebuild_query_regression.csv` 和 `current_source_rebuild_sel50_l_sweep.csv`，不得与前者混为一个确定性结论。后续 `378e71d7...0313` 仅增加默认关闭、计时外的结果合法性审计，不构成新的性能 binary。

最终展示时，建议把“第二层独立贡献”作为主要创新结果，把最终 tuned 配置作为系统最佳结果，再用六档表明确展示边界。
