# QF-SSL：无需查询校准的 Special Block 自动分层

更新时间：2026-09-17

## 摘要

本文提出 **Query-Free Structural Scale Ladder（QF-SSL）**：仅使用 Special Block 构建参数和标签 trie 的静态基数，自动确定 Special Block 的层数与各层 block 阈值。它不读取 query workload，不运行 latency/Recall 校准，也不在数据集间拟合参数。当前实现最多物化两层，但规则本身可自然延伸到任意层。

对当前固定构建参数 `M_sb=64`、`L_build,sb=100`、每 block 跨边预算 `C_sb=4`，QF-SSL 在所有数据集上生成候选尺度 `T1=8192`、`T2=131072`。候选层数由每个尺度是否还能产生非空 trie partition 决定；只有至少两个候选尺度时才物化层级，否则回退 plain UNG。最终 Amazon、Reviews 和 VariousImg 选择两层，Genome 选择零层。

Amazon x1 的九档等 Recall 正式实验表明，带结构路由的自动两层相对 plain 在 0.5%--10% 的 15-repeat 复核中总体近似持平，在 30%--99% 明显更快；在统一 current binary 的严格对照中，相对同 `T1` 单层，高选择率五档的加速为 **1.408x--2.664x**，几何平均 **1.992x**。结构路由只在至少一个 upper block 的 root-label 集合包含 query labels、从而该 block 对 containment query 合法时启用 overlay；无需 latency、Recall 或选择率校准。

相对人工固定的 prior-tuned 两层 `(32000,200000)`，自动 `(8192,131072)+router` 在 Amazon 九档的几何平均为 **1.010x**，逐档范围 **0.932x--1.212x**。因此自动方案的核心结果是：在免除层数与阈值网格搜索后，整体性能与该人工调优结构相当，而不是逐档支配人工参数。

跨数据集结果支持“静态结构能够给出合理层数与尺度”，但不支持“QF-SSL 对每个 workload 都最优”：Genome 因只有一个候选尺度而由最终 stop rule 回退 plain；Reviews 4.115% 与 plain 统计持平；VariousImg 10.252% 相对 plain 为 **2.317x**，相对同 `T1` 单层为 **1.539x**。Reviews 0.200% 的 100-thread batch wall-time 对执行顺序高度敏感，不能判定正负收益。因而 QF-SSL 应被表述为一套无需 query calibration、简洁可解释且有竞争力的结构启发式，而不是最优性定理。

## 1. 创新点

传统配置需要同时选择层数 `H` 和阈值序列 `(T1,...,TH)`。QF-SSL 将它们压缩为由已有构建预算导出的确定性尺度阶梯：

[
T_1=2^{\lceil\log_2(M_{sb} L_{build,sb})\rceil},\qquad
\rho=2^{\lceil\log_2(\max(2,M_{sb}/C_{sb}))\rceil},\qquad
T_l=T_1\rho^{l-1}.
]

对每个 `T_l`，在原始 trie 上独立执行既有 bottom-up uncovered-mass partition。若该 partition 产生至少一个非根 block，则加入第 `l` 层；首个产生零 block 的尺度立即停止。由此：

- `M_sb L_build,sb` 表示一个值得独立建图的局部搜索工作尺度；
- `M_sb/C_sb` 表示从层内连接密度到跨 block 连接预算的尺度比；
- 2 的幂取整使布局稳定、可复现，并避免伪精确阈值；
- trie 本身决定层数，参数不会随 query workload 改变。

`M_sb`、`L_build,sb` 和 `C_sb` 都是 Special Block overlay 已有的图构建参数，不应与基础 UNG 主图的 degree/build-width 混淆。因此 QF-SSL 没有新增需训练或搜索的自由超参数。这里的解释是设计动机与结构启发式，不是理论最优性证明。

### 自动停止算法

```text
T <- next_power_of_two(M_sb * L_build_sb)
rho <- next_power_of_two(max(2, M_sb / C_sb))
candidates <- []
while len(candidates) < implementation_limit:
    blocks <- partition(original_trie, threshold=T)
    if blocks is empty: break
    candidates.append((T, blocks))
    T <- T * rho
if len(candidates) < 2: return plain_UNG
return candidates
```

每一层都对原始 trie 独立 partition，而不是从上一层再次切分。这样与当前双 ownership 索引格式一致，也不假定不同阈值的 block 必然严格嵌套。单个候选尺度不构成 hierarchy，且已观察到只有单层而无 coarse bypass 时可能显著变慢；因此最终规则要求至少两个非空候选尺度，否则不物化 overlay。

## 2. 自动产生的配置

固定 `M_sb=64, L_build,sb=100, C_sb=4` 后，`T1=8192`、`rho=16`、`T2=131072`。

| 数据集 | points | groups | trie nodes | 自动层数 | 每层阈值 | block 数 | 覆盖率 | 停止原因 |
|---|---:|---:|---:|---:|---|---|---|---|
| Amazon | 602,453 | 482,387 | 2,800,516 | 2 | 8,192 / 131,072 | 23 / 1 | 98.74% / 91.05% | `T3=2,097,152` partition 为空 |
| Genome | 108,077 | 108,055 | 3,544,633 | 0（plain） | — | 候选 9 / 最终 0 | 候选 99.65% / 最终 — | `T2` 为空，仅一个非空候选尺度，不物化 |
| Reviews | 288,065 | 264,602 | 734,573 | 2 | 8,192 / 131,072 | 16 / 1 | 96.18% / 45.52% | `T3=2,097,152` partition 为空 |
| VariousImg | 758,935 | 293,903 | 633,417 | 2 | 8,192 / 131,072 | 27 / 2 | 85.80% / 73.16% | `T3=2,097,152` partition 为空 |

静态分析的 safety cap 为 64 层，四个数据集均在触及 cap 前因空 partition 自然停止；Genome 在候选第一层之后、其余数据在候选第二层之后停止。因此此处的 0/2 层是最终规则的输出，而非当前两层索引格式的截断结果。当前实现仍只能物化两层；若未来数据集产生三层以上，需先扩展索引格式。

## 3. 实验协议

### 3.1 公平性

- K=10，100 search threads；所有方法使用相同 query、精确 GT、主图与 `cpu_bruteforce_els`。
- 禁用 ELS query-result reuse 与隐式 CPU ELS warmup。
- 每个正式点 7 repeats；repeat 0 为 cold，报告 repeats 1--6 的 warm median、mean、CV 和 p95。
- Amazon 低选择率 router/plain 比较额外运行 15 repeats，报告 14 个 warm repeats。结构因果对照（单层、不路由二层、prior-tuned）与 router 均使用 SHA-256 为 `2e69823e...7edb26` 的 current binary，各 7 repeats。
- 每个方法独立选择最小的**实测** `Lsearch` 使预声明 Recall 门槛达标；不插值、不外推。
- paired-bootstrap 95% CI 在同序号 warm repeats 上重采样。仅 6 或 14 个 warm batch，CI 反映重复批次稳定性，不等同于跨查询或跨数据集总体推断。

### 3.2 数据与门槛

Amazon x1 包含 602,453 个 768D 向量、482,387 groups。0.5%、1%、5%、10% 使用 Recall@10 >= 0.90；30%、60%、80%、95%、99% 使用 >= 0.87。99% workload 是 303 个 label-1 query 与 697 个 empty-predicate query 的 synthetic batch mixture，不是单个自然 predicate。

跨数据集 workload 使用其真实平均选择率：Genome 3.365%/6.292%，Reviews 0.200%/4.115%，VariousImg 10.252%；门槛均为 Recall@10 >= 0.90。VariousImg plain 的首次 `L=4250` 正式重复中有一个 repeat 略低于门槛，故按预先离散网格前进到下一实测点 `L=4500` 后重跑；该修正同时记录在配置中。

## 4. Amazon x1 完整等 Recall 结果

最终部署方法记作 `QF-SSL + structural router`：索引由 QF-SSL 自动产生两层；只有 containment query 的 label 集合被至少一个 upper block 的 root-label 集合包含时才进入 overlay，否则执行与 0 层相同的 plain 路径。换言之，该 upper block 的所有后代都满足 query，而不是 query 包含 block root。表中延迟均为 1000-query batch 的 warm median。plain 列来自冻结的 exact-level 对照；单层与 router 使用同一 current binary，因而 `router/单层` 是关键结构因果比较。由于短 workload 的 6 个 warm batch 波动较大，0.5%--10% 的 router/plain 主结论优先采用独立 15-repeat 复核。

| 实际选择率 | Recall 门槛 | plain L / ms | 同 T1 单层 L / ms | 自动2层+router L / ms | router 比 plain | router 比单层 | router 实际启用率 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.499% | .90 | 1400 / 47.805 | 1400 / 56.137 | 1400 / 47.613 | 1.004x | 1.179x | 0.0% |
| 0.903% | .90 | 1600 / 55.087 | 1600 / 73.482 | 1600 / 68.105 | 0.809x | 1.079x | 0.0% |
| 5.038% | .90 | 2500 / 161.502 | 1750 / 164.611 | 1600 / 134.723 | 1.199x | 1.222x | 4.3% |
| 9.907% | .90 | 15000 / 731.797 | 8500 / 1025.955 | 15000 / 710.928 | 1.029x | 1.443x | 0.0% |
| 30.027% | .87 | 17000 / 1533.305 | 7000 / 1560.580 | 6000 / 1108.310 | 1.383x | 1.408x | 29.6% |
| 60.047% | .87 | 75000 / 17340.500 | 1700 / 608.461 | 600 / 338.479 | 51.231x | 1.798x | 61.4% |
| 80.024% | .87 | 120000 / 57321.800 | 2600 / 1151.635 | 550 / 432.303 | 132.597x | 2.664x | 82.2% |
| 95.020% | .87 | 160000 / 137927.000 | 900 / 698.801 | 75 / 318.707 | 432.771x | 2.193x | 98.2% |
| 99.001% | .87 | 240000 / 322303.000 | 1000 / 858.507 | 60 / 404.740 | 796.322x | 2.121x | 100.0% |

低档 15-repeat router/plain 复核为：0.499% `0.965x [0.903,1.003]`，0.903% `0.985x [0.929,1.029]`，5.038% `1.091x [1.022,1.114]`，9.907% `1.011x [0.991,1.022]`，四档点估计的几何平均为 `1.012x`。因此可严谨概括为：0.5%--10% 整体近似持平，仅 5% 有稳定小幅收益；不能根据 7-repeat 的 1% 偶然慢值宣称系统性退化。

高档 router 相对 current-binary 同 `T1` 单层分别为 **1.408x、1.798x、2.664x、2.193x、2.121x**，几何平均 **1.992x**；五档的 paired-bootstrap 95% CI 均严格高于 1。相对 current-binary 不路由二层为 **1.081x、1.047x、0.906x、1.000x、0.987x**，几何平均 **1.002x**。该对照支持“高档收益来自第二层”，而 router 自身在不同档位有小幅正负扰动；不能宣称 router 在每个 workload 都优于无路由二层。

### 4.1 自动分层与人工调优的同 binary 对比

下表把论文最需要的三个参照放在一起。`自动单层` 是 QF-SSL 的第一尺度 `T1=8192`，用于隔离第二层的增量；`自动两层` 是最终的 `(8192,131072)+router`；`人工两层` 是历史人工选择的固定结构 `(32000,200000)`。三者均由同一 current binary 运行，并分别选择满足 Recall 门槛的最小实测 `Lsearch`。相对 plain 的加速使用第 4 节冻结的 exact-level plain 对照；因此它适合呈现端到端量级，但最严格的结构因果结论应优先看同 binary 的 `自动两层/自动单层` 和 `自动/人工` 两列。

| 实际选择率 | plain ms | 自动单层 ms / 比 plain | 自动两层 ms / 比 plain | 人工两层 ms / 比 plain | 自动两层比自动单层 | 自动比人工 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.499% | 47.805 | 56.137 / 0.852x | 47.613 / 1.004x | 46.250 / 1.034x | 1.179x | 0.971x |
| 0.903% | 55.087 | 73.482 / 0.750x | 68.105 / 0.809x | 68.001 / 0.810x | 1.079x | 0.998x |
| 5.038% | 161.502 | 164.611 / 0.981x | 134.723 / 1.199x | 144.104 / 1.121x | 1.222x | 1.070x |
| 9.907% | 731.797 | 1025.955 / 0.713x | 710.928 / 1.029x | 861.740 / 0.849x | 1.443x | 1.212x |
| 30.027% | 1533.305 | 1560.580 / 0.983x | 1108.310 / 1.383x | 1191.475 / 1.287x | 1.408x | 1.075x |
| 60.047% | 17340.500 | 608.461 / 28.499x | 338.479 / 51.231x | 315.345 / 54.989x | 1.798x | 0.932x |
| 80.024% | 57321.800 | 1151.635 / 49.774x | 432.303 / 132.597x | 408.101 / 140.460x | 2.664x | 0.944x |
| 95.020% | 137927.000 | 698.801 / 197.377x | 318.707 / 432.771x | 312.983 / 440.686x | 2.193x | 0.982x |
| 99.001% | 322303.000 | 858.507 / 375.423x | 404.740 / 796.322x | 377.886 / 852.911x | 2.121x | 0.934x |

自动两层相对人工两层的九档几何平均为 **1.010x**，范围为 **0.932x--1.212x**：自动规则总体相当，在 5%--30% 更快，在 60% 及以上略慢。换言之，QF-SSL 的主要价值不是声称逐 workload 胜过人工调参，而是在**完全不搜索层数和阈值**的条件下达到与该人工固定结构相当的整体性能，同时保留高选择率相对单层的明确收益。

若只问“当前九档的已测结果里，最快双层相对单层和 plain 如何”，可将三个同 binary 双层候选（自动无路由、自动路由、人工固定）逐档取最低延迟。这里的单层只有自动 `T1=8192` 一个 current-binary 候选，因此不能称其为单层网格最优。

| 实际选择率 | plain ms | 当前单层候选 ms / 比 plain | 当前已测双层最低 ms（来源）/ 比 plain | 双层最低比单层候选 |
|---:|---:|---:|---:|---:|
| 0.499% | 47.805 | 56.137 / 0.852x | 46.250（人工固定）/ 1.034x | 1.214x |
| 0.903% | 55.087 | 73.482 / 0.750x | 68.001（人工固定）/ 0.810x | 1.081x |
| 5.038% | 161.502 | 164.611 / 0.981x | 134.723（自动路由）/ 1.199x | 1.222x |
| 9.907% | 731.797 | 1025.955 / 0.713x | 710.928（自动路由）/ 1.029x | 1.443x |
| 30.027% | 1533.305 | 1560.580 / 0.983x | 1108.310（自动路由）/ 1.383x | 1.408x |
| 60.047% | 17340.500 | 608.461 / 28.499x | 315.345（人工固定）/ 54.989x | 1.930x |
| 80.024% | 57321.800 | 1151.635 / 49.774x | 391.781（自动无路由）/ 146.311x | 2.939x |
| 95.020% | 137927.000 | 698.801 / 197.377x | 312.983（人工固定）/ 440.686x | 2.233x |
| 99.001% | 322303.000 | 858.507 / 375.423x | 377.886（人工固定）/ 852.911x | 2.272x |

### 4.2 已测候选中的单层最优与双层最优（历史 sweep）

为回答“双层最优是否优于单层最优”，下表单独列出旧参数 sweep 在每个 workload 的预定义候选网格内选出的最低等 Recall 延迟。它只覆盖 0.5%、1%、10%、25%、50%、75%，使用的是 router/exact-level 最终实现之前的历史 binary，故只能作为参数空间形态证据，不能与上表做跨行数值合并，也不能称为连续参数空间的全局最优。

| 实际选择率 | plain ms | 已测单层最优（参数）ms / 比 plain | 已测双层最优（参数）ms / 比 plain | 双层最优比单层最优 |
|---:|---:|---:|---:|---:|
| 0.499% | 39.201 | `T1=32000` 40.658 / 0.964x | `(32000,200000)` 40.268 / 0.973x | 1.010x |
| 0.903% | 51.531 | `T1=16000` 64.679 / 0.797x | `(64000,200000)` 62.142 / 0.829x | 1.041x |
| 9.907% | 656.284 | `T1=32000` 790.144 / 0.831x | `(32000,100000)` 776.964 / 0.845x | 1.017x |
| 24.915% | 1533.380 | `T1=128000` 1447.450 / 1.059x | `(64000,64001)` 1713.795 / 0.895x | 0.845x |
| 49.971% | 5072.390 | `T1=32000` 295.583 / 17.161x | `(16000,200000)` 310.929 / 16.314x | 0.951x |
| 74.994% | 41353.500 | `T1=32000` 574.830 / 71.940x | `(16000,200000)` 490.104 / 84.377x | 1.173x |

该 sweep 显示双层并非在每个选择率都支配最优单层：在 25% 和 50% 的已测网格中单层更快，在 75% 双层优势最明显。这也说明不能只凭“层数更多”推出延迟必然更低；最终方法使用静态尺度生成加结构路由，并将论文主结论建立在上表的 current-binary 对照，而不是从历史 sweep 外推。

### 4.3 与 prior-tuned control 的距离

历史人工选择的 `(T1,T2)=(32000,200000)` 只作为 prior-tuned control，不称 oracle。统一 current binary 后，最终自动方法 `(8192,131072)+router` 的九档几何平均加速为 **1.010x**；逐档 speedup 为 **0.932x--1.212x**。这属于不同结构加独立等 Recall `Lsearch` crossing 的经验比较：总体相当、部分档位更优，但不构成逐 workload 或全局最优性证明。

## 5. 跨数据集等 Recall 验证

| 数据集 / 实际选择率 | 最终自动层数 | plain L / ms | 同 T1 单层 L / ms | 最终方法 L / ms | 最终方法比 plain | 比同 T1 单层 | router 启用率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Genome / 3.365% | 0 | 10 / 122.642 | 历史候选：10 / 108.415 | 等同 plain | 1.000x（定义） | — | — |
| Genome / 6.292% | 0 | 10 / 196.810 | 历史候选：10 / 234.973 | 等同 plain | 1.000x（定义） | — | — |
| Reviews / 0.200% | 2 | 125 / 不稳定 | 历史：125 / 117.437 | 125 / 不稳定 | 不可判定 | 不可判定 | 0.0% |
| Reviews / 4.115% | 2 | 225 / 212.898 | 125 / 259.780 | 225 / 212.016 | 1.004x [0.982,1.039] | 非同 L，不作因果比 | 0.0% |
| VariousImg / 10.252% | 2 | 4500 / 2272.995 | 3750 / 1509.320 | 3500 / 980.925 | 2.317x [2.300,2.355] | 1.539x [1.526,1.550] | 14.5% |

Genome 的一层数据是最终 singleton stop rule 形成前的历史候选实验：它在两个 workload 中一正一负，且 CelebA 原冻结规则的一层实验为 plain 1023.94 ms、单层 2840.23 ms（`0.361x [0.356,0.362]`）。这两组负例是引入“至少两个候选尺度才物化”的开发依据，不能再称作最终方法的 held-out 验证。最终方法在 Genome/CelebA 上均输出 0 层，因此按定义等同 plain；CelebA 尚未生成最终静态 JSON 工件，只能由已冻结候选结构推出该结果。

Reviews 0.200% 需要单独解释。7-repeat 历史 plain/router 为 116.035/118.042 ms（`0.983x [0.595,1.128]`）；15-repeat 顺序实验为 29.766/128.486 ms；反转顺序后为 128.358/78.701 ms；12 对独立进程交错实验又为 124.045/100.700 ms，而逐 query total 中位数是 0.725/0.856 ms，方向相反。由于该 workload 有 3000 queries、单 query 图搜索极短，100-thread batch wall-time 被调度和运行顺序主导。故本报告不选择任何一个有利数字，而将它标为 **wall-time 不可判定**。

## 6. 如何解释结果

QF-SSL 自动决定的是**结构容量层级**，不是“哪个 workload 必然会使用哪一层”。在 exact-level 实现中，候选一旦处于某层，只扫描该层拥有的边；不能混扫低层边。合法候选可晋升到高层，而不适用的查询可以不进入高层。因此，保持 `T1` 不变时，第二层不会改变第一层 block 的定义。

router 未授权时不会调用 Special Block entry preparation，也不会调用 Special Block graph backend；它执行 plain UNG query path。尽管如此，“未进入高层”仍不严格推出整批 wall-time 数值完全相同：sidecar 在进程初始化时已加载，而极短的 100-thread batch 还受到线程池调度、缓存与系统状态影响。Amazon 的 15-repeat 数据支持更准确的说法：在中低选择率，最终方法整体近似无害；一旦某个大 block 的 root labels 包含 query labels，该 block 对 query 合法，第二层便可显著降低达到相同 Recall 所需的 `Lsearch`。

静态“存在非空 partition”只说明某一尺度在数据结构上存在，不保证给定 workload 会从中获益。因此 QF-SSL 还配套一个纯结构 query-time router：不满足 upper root-label containment 的 query 直接走 plain path。该判定不使用历史 workload、latency、Recall 或选择率模型。它仍不保证每个 workload 全局最优，因为已授权 query 也可能收益不足，且 sidecar 有常驻内存代价；这正是 query-free 配置的边界。

## 7. 构建与存储代价

下表报告 overlay builder 内部时间和持久化/加载后的增量结构规模，不包含基础 UNG 主图的 from-scratch 构建。一次构建测量不足以比较构建时间显著性，因此主要将磁盘、内存和 edge 数视为确定性的容量代价。

| 数据集 | 结构 | builder time | disk | loaded alloc. | special edges | blocks（upper） |
|---|---|---:|---:|---:|---:|---:|
| Amazon | 1层 8,192 | 37.22 s | 444.7 MiB | 535.4 MiB | 31,306,508 | 23 (0) |
| Amazon | 2层 8,192/131,072 | 49.94 s | 613.8 MiB | 704.8 MiB | 53,253,411 | 24 (1) |
| Genome | 候选1层 8,192（最终不物化） | 6.49 s | 124.0 MiB | 184.9 MiB | 3,569,630 | 9 (0) |
| Reviews | 1层 8,192 | 5.21 s | 106.7 MiB | 138.5 MiB | 10,553,905 | 16 (0) |
| Reviews | 2层 8,192/131,072 | 5.92 s | 142.7 MiB | 174.6 MiB | 15,220,148 | 17 (1) |
| VariousImg | 1层 8,192 | 96.94 s | 642.9 MiB | 678.8 MiB | 27,046,197 | 27 (0) |
| VariousImg | 2层 8,192/131,072 | 89.93 s | 780.0 MiB | 816.3 MiB | 44,904,138 | 29 (2) |

以相同 `T1` 比较，第二层使 Amazon/Reviews/VariousImg 的磁盘占用分别增加约 38.0%/33.8%/21.3%，loaded allocated memory 增加约 31.6%/26.1%/20.3%。这说明高选择率查询的延迟收益不是免费获得的；是否值得启用自动产生的高层仍取决于部署对内存和索引大小的约束。

## 8. 论文中建议的表述

可主张：

- QF-SSL 用已有图构建参数和静态 trie mass 自动产生层数及每层阈值，无需查询 workload、latency 或 Recall 校准。
- 在四个数据集上，同一公式最终生成零层或两层结构；Amazon 自动两层加结构路由在 30%--99% 五个测点相对同 `T1` 单层获得 1.408x--2.664x 加速，并在中低选择率相对 plain 整体近似持平。
- 统一 current binary 后，自动方法在 Amazon 九档相对 prior-tuned control 的几何平均加速为 1.010x，显示其无需人工结构搜索仍具有竞争力。
- VariousImg 10.252% 也观察到稳定的二层增益；Reviews 4.115% 与 plain 持平；Genome 能依据静态结构自动回退 plain。

不可主张：

- QF-SSL 是最优参数求解器，或存在全数据集、全选择率的性能支配关系。
- “构建了更高层”必然加速所有查询；非空结构是可用性条件，不是 workload 收益保证。
- 99% 是单一真实 predicate；它是明确构造的混合 batch。
- 6 个 warm repeats 的 bootstrap CI 能替代跨机器、跨查询集的统计推断。
- 当前实验已经验证任意多层；当前实现上限仍是两层。
- 最终 singleton stop rule 是未触碰 held-out 结果的预注册规则；它是在观察 CelebA/Genome 单层负例后加入的开发修正。

建议论文中将 QF-SSL 定位为 **query-free structural prior**：它消除了层数与 block 阈值的人工网格搜索，并把运行时搜索预算调优留给常规 ANN Recall--latency 选择。

## 9. 复现入口

- 静态策略：`experiments/multilevel_special/analyze_auto_layer_policy.py`
- Amazon router 正式配置：`experiments/multilevel_special/config.auto_policy_structural_router_formal.json`
- Amazon 低档 15-repeat 配置：`experiments/multilevel_special/config.auto_policy_structural_router_low_critical.json`
- Amazon current-binary 结构因果对照：`experiments/multilevel_special/config.auto_policy_current_binary_controls.json`、`results_summary/auto_policy_current_binary_controls.csv`
- 历史单层/双层候选网格最优：`experiments/multilevel_special/results_summary/layer_tuning_oracle_summary.csv`（文中仅称“已测候选最优”）
- 冻结的 plain/单层/不路由二层对照：`experiments/multilevel_special/config.auto_policy_formal_exact_level.json`、`config.auto_policy_critical_exact_level.json`
- 跨数据集配置生成：`experiments/multilevel_special/generate_auto_policy_cross_dataset_configs.py`、`generate_auto_policy_external_crossing.py`、`generate_auto_policy_cross_dataset_formal.py`
- router 汇总器：`experiments/multilevel_special/summarize_auto_policy_router.py`、`summarize_auto_policy_router_low_critical.py`、`summarize_auto_policy_router_cross_dataset.py`
- 核心 compact evidence：`experiments/multilevel_special/results_summary/auto_policy_structural_router_formal.csv`、`auto_policy_structural_router_low_critical.csv`、`auto_policy_current_binary_controls.csv`、`auto_policy_structural_router_cross_dataset.csv`
- Reviews 0.200% 稳定性复核：`runs/auto_policy_structural_router_reviews_critical`、`runs/auto_policy_structural_router_reviews_reverse_probe`、`runs/auto_policy_structural_router_reviews_interleaved_probe`
- 执行 provenance snapshots：`experiments/multilevel_special/results_summary/auto_policy_manifests/`
- raw 运行产物位于 `runs/`，不纳入 Git。

## 10. 结论

最终建议采用 QF-SSL：

[
T_1=\operatorname{pow2ceil}(M_{sb} L_{build,sb}),\quad
T_l=T_1\operatorname{pow2ceil}(\max(2,M_{sb}/C_{sb}))^{l-1},
]

在首个空 partition 前停止，并仅在至少存在两个候选尺度时物化；运行时再以 upper root-label containment 作零校准结构路由。它比手工指定“层数 + 每层 block 大小”更简洁，也比基于 query latency 的校准更容易部署和解释。现有证据显示，它抓住了 Amazon 高选择率下第二层的主要价值，同时在中低选择率总体接近 plain；跨数据集包含稳定正例、统计持平点和不可判定的短任务点，因此论文应强调它是合理且有效的结构先验，而非普适最优定理。
