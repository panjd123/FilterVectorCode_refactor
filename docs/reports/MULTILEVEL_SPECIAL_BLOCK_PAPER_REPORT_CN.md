# 两级 Special Block：系统组成、适用区间与公平性能评估

更新时间：2026-09-07

## 摘要

本文评估一套用于 filtered graph search 的两级 Special Block overlay。系统在既有 UNG 主图之上保留中层 block，并增加覆盖更大合法 Trie 子树的上层 block；查询只有完整覆盖某个 block 的根标签时才能使用该层 special edges。当前实现是固定的两级 overlay，不宣称支持任意 N 层。

我们在 Amazon 原始 100% x1 数据上，按同一 Recall 门槛、同一 ELS provider 和同一查询 binary，分别独立调优 0 层、1 层和 2 层。论文主结论使用“一层一套跨六档共享阈值”的可部署配置；逐 workload 换阈值的结果只作为 oracle 上界。正式评估包含 92 个 structure/workload case、552 个离散 `Lsearch` 点，每点 7 repeats；丢弃 cold repeat 0 后，以 6 个 warm batch 的中位数为主指标。所有共享与 oracle 赢家均已通过开放边界审计。

主要结论如下：

1. 跨六档共享配置中，单层 `T1=32k` 的几何平均加速为 **3.034x**，两层 `T1=16k,T2=200k` 为 **3.019x**。两者总体差异仅约 0.5%，且未做显著性检验，因此不能声称两层整体优于单层。
2. 优势高度依赖查询环境。50%/75% 选择率下，单层相对 0 层分别为 **17.16x/71.94x**，两层为 **16.31x/84.38x**；75% 档两层又比单层快 **1.173x**。
3. 低到中选择率不是该 overlay 的优势区间：0.499%、0.903%、9.907% 上两种 Special 配置都慢于 0 层；24.915% 上单层基本持平，两层慢 18.6%。
4. 性能跃迁主要来自图搜索预算下降，而不是 ELS 或 Block Authorization。75% 档每查询 Graph Search 从 0 层的 3822.66 ms 降至单层 26.68 ms、两层 20.14 ms；Authorization 仅约 0.005 ms/query。

因此，严谨的表述是：Special Block 对宽过滤条件、且查询能够完整授权大 block 的环境具有大幅性能优势；独立调优后的两层配置进一步改善了最高选择率 workload，但当前证据不支持“两层配置在所有 workload 或总体上都优于独立调优后的单层”。

## 1. 系统包含什么

系统由五部分组成：

- **基础 UNG 主图**：所有 0/1/2 层方法共享，不因实验方法而改变。
- **ELS**：固定使用 `cpu_bruteforce_els` 计算可用入口 group；本实验不评估 ELS 算法优劣。
- **Special Block overlay**：单层保存一套中层 ownership 和 special edges；两层额外保存独立的 upper ownership、上层 block 与上层 special edges。
- **逐级授权搜索**：候选状态为 `activation_level in {0,1,2}`，只能按普通图到中层、再到上层单调升级；同一点可原位升级，但不重复占用搜索预算。
- **构建、验证与实验工具**：包含 source-layout/ownership/祖先关系/upper 可达性验证、fail-closed sidecar loader、可恢复 build/query runner、Recall 门槛选择器、边界审计器和阶段 breakdown。

系统不包含以下主张：任意 N 层、对所有过滤选择率都加速、GPU query batch 上完整支持 upper-level metadata，或通过加速 ELS 获得本文结果。发现 upper block 时当前查询优先保证 activation 语义正确，使用 CPU neighbor-list 路径；GPU 主要用于构建大 block 的近似 intra graph。

## 2. 方法与正确性边界

### 2.1 两次独立 partition

构建器在同一 label trie 上分别以 `T1` 和 `T2>T1` 执行 bottom-up uncovered partition。上层不会替换中层；一个 point 可以同时有 middle owner 和 upper owner。一般树形上，较大阈值的 partition 不保证严格嵌套，因此实现维护两套 ownership，不能简化为一张 middle-to-upper 映射。

### 2.2 授权语义

查询完整覆盖某 block 的 root-label prefix 后，才允许扫描该 block 的 special edges。状态转移为：

```text
普通图 (level 0) -> 中层 (level 1) -> 上层 (level 2)
```

一条 transition 最多提升一级，普通候选不能跳过中层直接使用上层边。持久化验证器重新核对 direct members、point ownership、同层最近祖先、最近 upper 祖先和 upper 激活可达性；不满足条件的 sidecar 被拒绝。

### 2.3 预期性能机制

Special edges 会增加单次展开的邻居数，因此“相同 `Lsearch` 更快”不是目标。方法希望把 Recall–`Lsearch` 曲线左移，使宽过滤查询以更小搜索预算达到同一 Recall。性能比较必须在各方法独立调优后、按同一 Recall 门槛进行。

## 3. 实验协议

| 项目 | 设置 |
|---|---|
| 数据 | Amazon 原始 100% x1；602,453 points，768D |
| 标签 | 482,387 groups，30,723 labels |
| Workload | 平均选择率 0.499%、0.903%、9.907%、24.915%、49.971%、74.994% |
| 查询 | 每档 1000 queries，K=10，100 CPU threads |
| Recall 门槛 | .90 / .90 / .90 / .90 / .85 / .87 |
| 调优 | 0 层调 `Lsearch`；1 层调 `T1 × Lsearch`；2 层调 `T1 × T2 × Lsearch`，且 `T2>T1` |
| ELS | 所有方法固定 `cpu_bruteforce_els`；`UNG_DISABLE_ELS_REUSE=1` |
| 重复 | 7 repeats；repeat 0 为 cold，主指标为后 6 次 warm batch median |
| 可行性 | 每个 repeat 的最低 Recall 达门槛；不插值 |
| 主选择 | 每层一套跨六档共享阈值；`Lsearch` 可按 workload 调整 |
| Oracle | 每个 workload 可独立换结构，只表示上界 |

环境为双路 Intel Xeon Platinum 8360Y（72 physical cores、144 hardware threads）、NVIDIA L20 46 GiB、driver 550.54.14。正式查询 binary SHA-256 为 `6fa4082d...a40087`，builder 为 `c305f487...0b295b`。机器资源未由 `gpulock` 独占，因此本文报告重复中位数，不把小差异解释为统计显著。

阶段定义彼此互斥：`ELS`、`Entry Point Setup`、`Block Authorization`、`Graph Search` 和 `Residual` 在每个 repeat 的原始精度下闭合到平均单查询总时间。CSV 输出精度与聚合会造成极小的表面差值；它们是并行 batch 内累计的 per-query work，不能与 1000-query batch wall 相加或直接换算。

## 4. 公平 0/1/2 层主结果

共享配置为：0 层 plain、1 层 `T1=32k`、2 层 `T1=16k,T2=200k`。下表的每个点都是达到对应 Recall 门槛的最快离散实测点。

| 选择率 | Recall门槛 | 0层：L / batch ms | 1层：L / batch ms / vs 0 | 2层：L / batch ms / vs 0 | 2层 vs 1层 |
|---:|---:|---:|---:|---:|---:|
| 0.499% | .90 | 1375 / 39.201 | 1375 / 40.658 / 0.964x | 1375 / 40.932 / 0.958x | 0.993x |
| 0.903% | .90 | 1625 / 51.531 | 1750 / 65.520 / 0.786x | 1625 / 62.526 / 0.824x | 1.048x |
| 9.907% | .90 | 14750 / 656.284 | 7000 / 790.144 / 0.831x | 6875 / 795.111 / 0.825x | 0.994x |
| 24.915% | .90 | 21000 / 1533.380 | 8000 / 1528.690 / 1.003x | 8125 / 1817.060 / 0.844x | 0.841x |
| 49.971% | .85 | 37500 / 5072.390 | 550 / 295.582 / **17.161x** | 450 / 310.929 / **16.314x** | 0.951x |
| 74.994% | .87 | 106250 / 41353.500 | 1150 / 574.830 / **71.940x** | 625 / 490.104 / **84.377x** | **1.173x** |

六档延迟比的几何平均结果为：0 层 1.000x、1 层 **3.034x**、2 层 **3.019x**。该平均值被 50% 和 75% 两档的大收益主导，不能解读为任意 workload 都有约 3x 加速。单层与两层总体差异约 0.5%，小于本实验足以支持稳定排序的幅度；当前只能报告离散实测排名。

边界审计覆盖完整 coarse 结构空间、formal guard 以及 `T2=T1+1` 和“无 upper block”的合法域端点。最终 shared 与 oracle 赢家均无开放轴命中，因此只声称“预声明离散网格内实测最优”，不声称连续参数空间全局最优。

## 5. 优势在哪种环境出现

### 5.1 宽过滤条件

49.971% 和 74.994% 是明确优势区间。plain 为达到 Recall 门槛需要 `L=37,500/106,250`；Special 配置只需数百到约一千的 `L`。这说明收益来自更高效地在大合法区域中导航，而不是减小 ELS 开销。

### 5.2 两层配置何时额外有价值

两层配置相对单层最清晰的收益出现在 74.994%：共享两层比共享单层快 17.3%，Graph Search work 从 26.68 降到 20.14 ms/query。0.903% 上两层也比单层快 4.8%，但两者都慢于 plain；这不足以构成部署优势。50% 上单层反而比两层快 4.9%，说明上层并非越多越好。由于两套共享配置的 `T1` 不同，这里比较的是各自独立调优后的方法族，不是只切换第二层的纯因果消融。

### 5.3 不适合的环境

在稀疏过滤或中等选择率下，block 授权机会和长程导航收益不足以抵消额外邻居扫描与路径管理。0.499%、0.903%、9.907% 应优先使用 plain；24.915% 下单层与 plain 基本持平，两层不合适。实际系统宜根据过滤结构选择路由，而不是全局强制开启两层。

## 6. 阶段 breakdown

| 选择率 | 层数 | ELS ms/q | Entry ms/q | Auth ms/q | Graph ms/q | Residual ms/q | Total ms/q |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 49.971% | 0 | 8.6011 | 33.9676 | 0 | 423.3185 | 0.2311 | 466.1180 |
| 49.971% | 1 | 7.7619 | 10.4071 | 0.0037 | 8.9378 | 0.0369 | 27.1474 |
| 49.971% | 2 | 7.6127 | 10.1650 | 0.0041 | 10.4937 | 0.0357 | 28.3111 |
| 74.994% | 0 | 17.7751 | 48.6266 | 0 | 3822.6600 | 0.6227 | 3889.6800 |
| 74.994% | 1 | 11.6486 | 14.9059 | 0.0051 | 26.6774 | 0.0526 | 53.2896 |
| 74.994% | 2 | 11.5288 | 14.4443 | 0.0051 | 20.1397 | 0.0411 | 46.1590 |

Authorization 本身几乎可忽略。50% 时两层的 Graph Search 比单层略高，与 batch 上单层更快一致；75% 时两层的 Graph Search 数值明显更低，与其额外收益一致。ELS 在三种方法中口径相同，并非本文优化目标。

## 7. 构建与存储代价

以下是选中共享配置在既有 UNG 主图上构建 overlay 的增量成本。复用历史 artifact 的 manifest 没有保留 runner 外层 wall，因此这里只报告 builder 内部 `total_time`，不将其冒充完整 from-scratch index build。

| 配置 | builder total | blocks / upper | special edges | disk |
|---|---:|---:|---:|---:|
| 单层 T1=32k | 42.932 s | 7 / 0 | 28,518,091 | 0.413 GiB |
| 两层 T1=16k,T2=200k | 68.641 s | 15 / 1 | 51,853,327 | 0.589 GiB |

两层配置的 builder total、edge 数和磁盘分别约为单层的 1.60x、1.82x 和 1.43x。由于两套共享赢家使用不同 `T1`，这是一项部署成本对比，不是“单独增加第二层”的纯因果消融。0 层不需要 overlay，但基础 UNG 的构建成本不应记为 0 或与上述增量直接相加。

## 8. 可主张内容与限制

可以主张：

- 实现了保留中层并增加上层的真实两级 Special Block，以及逐级授权、双 ownership 和 fail-closed 验证；
- 在 Amazon x1、预声明 Recall 门槛和离散参数网格内，单层/两层共享配置相对 0 层的六档几何平均加速分别为 3.034x/3.019x；
- 在 50%/75% 宽过滤 workload 上取得 16.31–84.38x 相对 plain 的加速；75% 时第二层相对独立调优单层再快 1.173x；
- 通过负结果明确界定方法适用区间。

不能主张：

- 两层整体优于单层，或所有选择率都受益；
- 单层与两层几何平均加速约 0.5% 的相对差异具有统计显著性；
- oracle 阈值是一套可部署配置；
- 阶段 per-query work 可以相加成 batch wall；
- 当前离散网格最优等于连续空间全局最优；
- overlay builder 时间等于完整索引 from-scratch 时间。

## 9. 证据与复现入口

- 正式配置：`experiments/multilevel_special/config.amazon_x1_layer_tuning_query_formal.json`
- 共享 compact 表：`experiments/multilevel_special/results_summary/layer_tuning_shared_summary.csv`
- 共享阶段表：`experiments/multilevel_special/results_summary/layer_tuning_stage_breakdown.csv`
- Oracle compact 表：`experiments/multilevel_special/results_summary/layer_tuning_oracle_summary.csv`
- 构建 compact 表：`experiments/multilevel_special/results_summary/layer_tuning_build_summary.csv`
- 完整 raw 结果：`runs/layer_tuning_query_formal_fair_amazon_x1/`（不提交）
- 构建 manifest：`runs/layer_tuning_build_amazon_x1/manifest.json`（不提交）
- 实验验证：formal 92/92 case，552 points，boundary hits=0；Python 56/56。
- 详细复现步骤：`docs/reports/MULTILEVEL_SPECIAL_BLOCK_REPRODUCE_CN.md`。

旧报告中的固定 `T1=1k` 单层、`T1=2k,T2=25k` 两层及外部系统表仍可用于历史机制研究，但不再作为公平 0/1/2 层主结论。
