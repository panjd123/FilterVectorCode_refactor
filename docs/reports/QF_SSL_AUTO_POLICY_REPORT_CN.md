# QF-SSL：无需查询校准的 Special Block 自动分层

更新时间：2026-09-17

## 摘要

本文提出 **Query-Free Structural Scale Ladder（QF-SSL）**：仅使用索引构建参数和标签 trie 的静态基数，自动确定 Special Block 的层数与各层 block 阈值。它不读取 query workload，不运行 latency/Recall 校准，也不在数据集间拟合参数。当前实现最多支持两层，但规则本身可自然延伸到任意层。

对当前固定构建参数 `M=64`、`L_build=100`、每 block 跨边预算 `C=4`，QF-SSL 在所有数据集上生成统一尺度 `T1=8192`、`T2=131072`；层数由每个尺度是否还能产生非空 trie partition 自动决定。Amazon、Reviews 和 VariousImg 选择两层，Genome 自动停在一层。

Amazon x1 的九档等 Recall 正式实验表明，自动两层相对相同 `T1` 的自动单层在 0.5%--10% 四档总体近似持平，在 30%--99% 五档明显更快。15-repeat 关键复核的分段几何平均分别为 **1.029x** 和 **1.985x**；其中 0.5%、5% 有小幅正收益，1%、10% 与持平相容，而 30%--99% 的 paired-bootstrap 95% CI 全部高于 1。相对 plain，7-repeat 正式表中的自动两层在 30% 为 1.237x，在 60%--99% 为 52.36x--811.83x。

跨数据集结果支持“静态结构能够给出合理层数与尺度”，但不支持“QF-SSL 对每个 workload 都最优”：Genome 两个 workload 中一快一慢；Reviews 的 0.200% workload 两层比 plain 快 1.540x，但 4.115% workload 仍慢于 plain；VariousImg 10.252% 上两层比 plain 快 1.596x，并比同 `T1` 单层快 1.059x。因而 QF-SSL 应被表述为一套无需 query calibration、简洁可解释且有竞争力的结构启发式，而不是最优性定理。

## 1. 创新点

传统配置需要同时选择层数 `H` 和阈值序列 `(T1,...,TH)`。QF-SSL 将它们压缩为由已有构建预算导出的确定性尺度阶梯：

[
T_1=2^{\lceil\log_2(M L_{build})\rceil},\qquad
\rho=2^{\lceil\log_2(\max(2,M/C))\rceil},\qquad
T_l=T_1\rho^{l-1}.
]

对每个 `T_l`，在原始 trie 上独立执行既有 bottom-up uncovered-mass partition。若该 partition 产生至少一个非根 block，则加入第 `l` 层；首个产生零 block 的尺度立即停止。由此：

- `M L_build` 表示一个值得独立建图的局部搜索工作尺度；
- `M/C` 表示从层内连接密度到跨 block 连接预算的尺度比；
- 2 的幂取整使布局稳定、可复现，并避免伪精确阈值；
- trie 本身决定层数，参数不会随 query workload 改变。

`M`、`L_build` 和 `C` 都是已有图构建参数，因此 QF-SSL 没有新增需训练或搜索的自由超参数。这里的解释是设计动机与结构启发式，不是理论最优性证明。

### 自动停止算法

```text
T <- next_power_of_two(M * L_build)
rho <- next_power_of_two(max(2, M / C))
levels <- []
while len(levels) < implementation_limit:
    blocks <- partition(original_trie, threshold=T)
    if blocks is empty: break
    levels.append((T, blocks))
    T <- T * rho
return levels
```

每一层都对原始 trie 独立 partition，而不是从上一层再次切分。这样与当前双 ownership 索引格式一致，也不假定不同阈值的 block 必然严格嵌套。

## 2. 自动产生的配置

固定 `M=64, L_build=100, C=4` 后，`T1=8192`、`rho=16`、`T2=131072`。

| 数据集 | points | groups | trie nodes | 自动层数 | 每层阈值 | block 数 | 覆盖率 | 停止原因 |
|---|---:|---:|---:|---:|---|---|---|---|
| Amazon | 602,453 | 482,387 | 2,800,516 | 2 | 8,192 / 131,072 | 23 / 1 | 98.74% / 91.05% | `T3=2,097,152` partition 为空 |
| Genome | 108,077 | 108,055 | 3,544,633 | 1 | 8,192 | 9 | 99.65% | `T2` partition 为空 |
| Reviews | 288,065 | 264,602 | 734,573 | 2 | 8,192 / 131,072 | 16 / 1 | 96.18% / 45.52% | `T3=2,097,152` partition 为空 |
| VariousImg | 758,935 | 293,903 | 633,417 | 2 | 8,192 / 131,072 | 27 / 2 | 85.80% / 73.16% | `T3=2,097,152` partition 为空 |

静态分析的 safety cap 为 64 层，四个数据集均在触及 cap 前因空 partition 自然停止。因此此处的 1/2 层是规则的输出，而非当前两层索引格式的截断结果。当前实现仍只能物化两层；若未来数据集产生三层以上，需先扩展索引格式。

## 3. 实验协议

### 3.1 公平性

- K=10，100 search threads；所有方法使用相同 query、精确 GT、主图与 `cpu_bruteforce_els`。
- 禁用 ELS query-result reuse 与隐式 CPU ELS warmup。
- 每个正式点 7 repeats；repeat 0 为 cold，报告 repeats 1--6 的 warm median、mean、CV 和 p95。
- Amazon 的关键“同 T1 单层 vs 两层”比较额外运行 15 repeats，报告 14 个 warm repeats。
- 每个方法独立选择最小的**实测** `Lsearch` 使预声明 Recall 门槛达标；不插值、不外推。
- paired-bootstrap 95% CI 在同序号 warm repeats 上重采样。仅 6 或 14 个 warm batch，CI 反映重复批次稳定性，不等同于跨查询或跨数据集总体推断。

### 3.2 数据与门槛

Amazon x1 包含 602,453 个 768D 向量、482,387 groups。0.5%、1%、5%、10% 使用 Recall@10 >= 0.90；30%、60%、80%、95%、99% 使用 >= 0.87。99% workload 是 303 个 label-1 query 与 697 个 empty-predicate query 的 synthetic batch mixture，不是单个自然 predicate。

跨数据集 workload 使用其真实平均选择率：Genome 3.365%/6.292%，Reviews 0.200%/4.115%，VariousImg 10.252%；门槛均为 Recall@10 >= 0.90。VariousImg plain 的首次 `L=4250` 正式重复中有一个 repeat 略低于门槛，故按预先离散网格前进到下一实测点 `L=4500` 后重跑；该修正同时记录在配置中。

## 4. Amazon x1 完整等 Recall 结果

表中延迟为 1000-query batch 的 warm median。`2L/1L` 是自动两层相对相同 `T1=8192` 单层的加速；95% CI 来自 15-repeat 关键复核，以降低 6-repeat 正式表在短 workload 上的偶然波动。

| 实际选择率 | Recall 门槛 | plain L / ms | 1层 L / ms | 2层 L / ms | 2层 vs plain | 2层 vs 1层（15-repeat） | paired 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.499% | .90 | 1400 / 47.805 | 1400 / 50.639 | 1400 / 48.093 | 0.994x | 1.092x | [1.064, 1.158] |
| 0.903% | .90 | 1600 / 55.087 | 1600 / 70.329 | 1600 / 71.491 | 0.771x | 0.965x | [0.936, 1.022] |
| 5.038% | .90 | 2500 / 161.502 | 1750 / 155.088 | 1600 / 149.505 | 1.080x | 1.057x | [1.002, 1.110] |
| 9.907% | .90 | 15000 / 731.797 | 8500 / 1039.625 | 8500 / 1066.150 | 0.686x | 1.005x | [0.995, 1.020] |
| 30.027% | .87 | 17000 / 1533.305 | 7000 / 1524.580 | 6000 / 1239.835 | 1.237x | 1.289x | [1.279, 1.326] |
| 60.047% | .87 | 75000 / 17340.500 | 1700 / 629.011 | 500 / 331.152 | 52.364x | 1.976x | [1.955, 2.007] |
| 80.024% | .87 | 120000 / 57321.800 | 2600 / 1162.035 | 500 / 445.216 | 128.751x | 2.661x | [2.634, 2.715] |
| 95.020% | .87 | 160000 / 137927.000 | 900 / 711.283 | 75 / 313.251 | 440.309x | 2.202x | [2.168, 2.249] |
| 99.001% | .87 | 240000 / 322303.000 | 1000 / 865.207 | 60 / 397.006 | 811.834x | 2.067x | [2.027, 2.094] |

7-repeat 正式表的九档几何平均中，自动单层和自动两层相对 plain 分别为 **7.224x** 和 **10.602x**；两层相对同 `T1` 单层为 **1.468x**。该聚合受高选择率的极大倍率影响，必须与逐点表同时报告。更稳健的 15-repeat 同 T1 复核为：

- 0.5%--10%：两层相对单层几何平均 **1.029x**，可概括为总体持平；
- 30%--99%：两层相对单层几何平均 **1.985x**；五档 95% CI 均完全高于 1。

### 与 prior-tuned control 的距离

历史人工选择的 `(T1,T2)=(32000,200000)` 只作为 prior-tuned control，不称 oracle。自动 `(8192,131072)` 的九档几何平均延迟比该 control 高 **3.28%**；逐点除 10% 外均在约 -2.21% 到 +5.65% 内，10% 慢 26.95%。这说明自动规则总体接近已有人工配置，同时也暴露了一个明确的局部失配点。

## 5. 跨数据集等 Recall 验证

| 数据集 / 实际选择率 | 自动层数 | plain L / ms | 1层 L / ms | 2层 L / ms | 自动配置 vs plain | 2层 vs 同 T1 单层 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Genome / 3.365% | 1 | 10 / 122.642 | 10 / 108.415 | — | 1.131x | — | [1.085, 1.148] |
| Genome / 6.292% | 1 | 10 / 196.810 | 10 / 234.973 | — | 0.838x | — | [0.828, 0.886] |
| Reviews / 0.200% | 2 | 125 / 116.035 | 125 / 117.437 | 125 / 75.341 | 1.540x | 1.559x | [1.219, 2.640] |
| Reviews / 4.115% | 2 | 225 / 221.645 | 125 / 259.780 | 125 / 258.104 | 0.859x | 1.006x | [0.991, 1.043] |
| VariousImg / 10.252% | 2 | 4500 / 2272.995 | 3750 / 1509.320 | 3750 / 1424.600 | 1.596x | 1.059x | [1.050, 1.071] |

最后一列对两层数据集报告 `2层/1层` paired CI；Genome 则报告自动一层相对 plain 的 CI。Reviews 0.200% 的 CV 较高（plain 32.6%，两层 33.7%），因此虽 CI 高于 1，也应视为需要更多 repeats 的提示性结果。

## 6. 如何解释结果

QF-SSL 自动决定的是**结构容量层级**，不是“哪个 workload 必然会使用哪一层”。在 exact-level 实现中，候选一旦处于某层，只扫描该层拥有的边；不能混扫低层边。合法候选可晋升到高层，而不适用的查询可以不进入高层。因此，保持 `T1` 不变时，第二层不会改变第一层 block 的定义。

但“未进入高层”并不严格推出 wall-time 完全不变：索引加载、membership/activation 检查、内存布局和 CPU 调度仍可能造成小幅波动。Amazon 的 15-repeat 数据支持更准确的说法：在中低选择率，第二层整体近似无害；一旦查询能充分授权大 block，第二层显著降低达到相同 Recall 所需的 `Lsearch`，收益自然出现。

静态“存在非空 partition”只说明某一尺度在数据结构上存在，不保证给定 workload 会从中获益。因此 QF-SSL 的两层输出有时比 plain 慢。这不是规则失效的隐藏例外，而是 query-free 配置的可预期边界：没有查询分布，就不可能同时保证每个 workload 的最优路由。若系统允许 query-time routing，可让不满足高层授权条件的 query 保持在低层；但本报告没有用额外校准训练该路由器。

## 7. 构建与存储代价

下表报告 overlay builder 内部时间和持久化/加载后的增量结构规模，不包含基础 UNG 主图的 from-scratch 构建。一次构建测量不足以比较构建时间显著性，因此主要将磁盘、内存和 edge 数视为确定性的容量代价。

| 数据集 | 结构 | builder time | disk | loaded alloc. | special edges | blocks（upper） |
|---|---|---:|---:|---:|---:|---:|
| Amazon | 1层 8,192 | 37.22 s | 444.7 MiB | 535.4 MiB | 31,306,508 | 23 (0) |
| Amazon | 2层 8,192/131,072 | 49.94 s | 613.8 MiB | 704.8 MiB | 53,253,411 | 24 (1) |
| Genome | 1层 8,192 | 6.49 s | 124.0 MiB | 184.9 MiB | 3,569,630 | 9 (0) |
| Reviews | 1层 8,192 | 5.21 s | 106.7 MiB | 138.5 MiB | 10,553,905 | 16 (0) |
| Reviews | 2层 8,192/131,072 | 5.92 s | 142.7 MiB | 174.6 MiB | 15,220,148 | 17 (1) |
| VariousImg | 1层 8,192 | 96.94 s | 642.9 MiB | 678.8 MiB | 27,046,197 | 27 (0) |
| VariousImg | 2层 8,192/131,072 | 89.93 s | 780.0 MiB | 816.3 MiB | 44,904,138 | 29 (2) |

以相同 `T1` 比较，第二层使 Amazon/Reviews/VariousImg 的磁盘占用分别增加约 38.0%/33.8%/21.3%，loaded allocated memory 增加约 31.6%/26.1%/20.3%。这说明高选择率查询的延迟收益不是免费获得的；是否值得启用自动产生的高层仍取决于部署对内存和索引大小的约束。

## 8. 论文中建议的表述

可主张：

- QF-SSL 用已有图构建参数和静态 trie mass 自动产生层数及每层阈值，无需查询 workload、latency 或 Recall 校准。
- 在四个数据集上，同一公式生成一层或两层结构；Amazon 自动两层在 30%--99% 五个测点相对同 `T1` 单层获得 1.289x--2.661x 加速，并在中低选择率整体近似持平。
- 自动配置在 Amazon 九档的几何平均延迟仅比 prior-tuned control 高 3.28%，显示其无需人工搜索仍具有竞争力。
- VariousImg 和 Reviews 的部分 workload 也观察到第二层增益；Genome 能依据静态结构自动停止在一层。

不可主张：

- QF-SSL 是最优参数求解器，或存在全数据集、全选择率的性能支配关系。
- “构建了更高层”必然加速所有查询；非空结构是可用性条件，不是 workload 收益保证。
- 99% 是单一真实 predicate；它是明确构造的混合 batch。
- 6 个 warm repeats 的 bootstrap CI 能替代跨机器、跨查询集的统计推断。
- 当前实验已经验证任意多层；当前实现上限仍是两层。

建议论文中将 QF-SSL 定位为 **query-free structural prior**：它消除了层数与 block 阈值的人工网格搜索，并把运行时搜索预算调优留给常规 ANN Recall--latency 选择。

## 9. 复现入口

- 静态策略：`experiments/multilevel_special/analyze_auto_layer_policy.py`
- Amazon 正式配置：`experiments/multilevel_special/config.auto_policy_formal_exact_level.json`
- Amazon 15-repeat 配置：`experiments/multilevel_special/config.auto_policy_critical_exact_level.json`
- 跨数据集配置生成：`experiments/multilevel_special/generate_auto_policy_cross_dataset_configs.py`、`generate_auto_policy_external_crossing.py`、`generate_auto_policy_cross_dataset_formal.py`
- 汇总器：`experiments/multilevel_special/summarize_auto_policy_formal.py`、`summarize_auto_policy_cross_dataset.py`
- compact evidence：`experiments/multilevel_special/results_summary/auto_policy_cross_dataset_formal.csv`
- 执行 provenance snapshots：`experiments/multilevel_special/results_summary/auto_policy_manifests/`
- raw 运行产物位于 `runs/`，不纳入 Git。

## 10. 结论

最终建议采用 QF-SSL：

[
T_1=\operatorname{pow2ceil}(M L_{build}),\quad
T_l=T_1\operatorname{pow2ceil}(\max(2,M/C))^{l-1},
]

并在首个空 partition 前停止。它比手工指定“层数 + 每层 block 大小”更简洁，也比基于 query latency 的校准更容易部署和解释。现有证据显示，它抓住了 Amazon 高选择率下第二层的主要价值，同时在低选择率总体保持接近单层；跨数据集存在正例、持平和负例，因此论文应强调它是合理且有效的结构先验，而非普适最优定理。
