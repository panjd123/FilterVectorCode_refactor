# 75% 以上 Special Block 实验计划

本文档记录新增高选择率实验的预声明协议、已完成结果与边界闭合过程；用于挑选赢家的质量口径未在观察结果后修改。

## 目标

回答独立调优后的两层配置在 75% 以上是否持续优于单层，以及接近或达到 100% 时是否因过滤约束消失而回落。75% 旧点只作已知锚点；新增点必须使用 Amazon 原始 x1、独立 query workload 和 exact filtered ground truth。

## 固定公平条件

- 0 层只调 `Lsearch`；1 层调 `T1 × Lsearch`；2 层调 `T1 × T2 × Lsearch` 且 `T2>T1`。
- 所有方法使用同一 search binary、`cpu_bruteforce_els`、1000 queries、K=10、100 threads。
- 设置 `UNG_DISABLE_ELS_REUSE=1`；不复用整条 query 的 ELS 结果。
- 正式点运行 7 repeats，repeat 0 为 cold；主指标为后 6 个 warm batch 的 median。
- 以每个 repeat 的最低端到端 Recall 达到预声明门槛为可行，不插值。
- 共享结构主结果和逐 workload oracle 分开；若赢家触及结构轴或 L 轴边界，必须补测或明确降级结论。
- 100% 是 unfiltered control，必须单列，不能与有过滤 workload 混为连续区间。

## 预声明档位与构造

固定复用 75% workload 的 1000 个 query vectors。以随机种子 20260908 打乱原来不是 label 1 的 233 个 query，再按固定前缀逐步替换为 label 1，得到嵌套 workload：

| 名称 | 实际平均选择率 | 替换 query 数 |
|---|---:|---:|
| sel_80 | 80.02369928% | 55 |
| sel_85 | 84.96712723% | 108 |
| sel_90 | 90.04633639% | 162 |
| sel_95 | 95.01978063% | 215 |
| sel_967 | 96.70165142% | 233，即所有 query 均为 label 1 |
| sel_100 | 100% | 所有 query 使用空 containment predicate |

label 1 是 Amazon x1 中覆盖最广的真实标签，覆盖 582,582/602,453 points；因此 96.70% 是单标签 containment 的自然上限，严格 100% 必须单列为空过滤 control。

## GT 与参数网格

- exact GT：分别对全 label-1 与空 containment 执行完整 filtered brute-force；80–95% 的每个 query GT 行按其标签从旧 75% GT 或 label-1 GT 逐字节组合。90% 组合 GT 已用独立完整 brute-force 复算，SHA-256 一致。
- Recall 门槛：六个新增档位统一为 0.87，与原 75% 档一致。
- coarse 结构：0 层；单层 T1={16k,32k,64k}；两层使用 T1={8k,16k,32k} 和 T2={100k,200k,400k} 的 8 个已构建有效组合，共 12 个方法。
- filtered coarse L-grid：0 层 80k/95k/102.5k/110k/125k；单层 500/750/1k/1.25k/1.5k/2k/3k；两层 250/375/500/625/750/1k/1.5k。
- 100% coarse L-grid：各层统一 50/100/200/400/800/1600/3200/6400，避免先验假定 plain 或 overlay 必然胜出。
- coarse 为 3 repeats；正式 shortlist 在 crossing 邻域运行 7 repeats。共享部署、原六档共享配置延伸和逐 workload oracle 分开报告。

## 当前状态

- 六个 workload 与 exact GT 已生成并验证；每档 10,000 个 GT 邻居均有效，ID 范围合法、距离有限且非负。
- Coarse、plain crossing、结构扩展和 7-repeat formal shortlist 均已完成。正式主 sweep 为 95/95 cases、570 points；共享赢家触边后又完成 20-case、149-point 7-repeat boundary extension。两批正式点通过重复键拒绝式合并，共 719 个唯一测点。
- 最终共享结构为：0 层 plain、1 层 `T1=128k`、2 层 `T1=16k,T2=400k`。共享选择的结构轴和 Lsearch 轴均无开放边界；逐 workload oracle 仍有 2 个结构轴命中，仅作为离散网格上界。

## 正式结果

下表使用一层一套跨六档共享结构；各 workload 只独立选择达到 `Recall>=0.87` 的最快实测 `Lsearch`。时间为排除 cold repeat 0 后六次 warm batch 的中位数。

| Workload | 0层 ms | 1层 `T1=128k` ms | 2层 `16k/400k` ms | 2层 vs 1层 |
|---|---:|---:|---:|---:|
| 80.024% | 58,192.9 | 404.634 | 389.620 | 1.039x |
| 84.967% | 70,540.1 | 365.249 | 320.900 | 1.138x |
| 90.046% | 99,821.0 | 335.249 | 297.188 | 1.128x |
| 95.020% | 130,241.5 | 318.280 | 290.285 | 1.096x |
| 96.702% | 138,067.0 | 315.044 | 295.908 | 1.065x |
| 100% control | 356,311.0 | 438.012 | 366.174 | 1.196x |

跨六档相对 0 层的几何平均加速为：单层 **326.276x**，两层 **361.894x**；两层相对单层为 **1.109x**。若只统计五个 filtered workload，则对应为单层 271.790x、两层 296.940x，两层相对单层 **1.093x**。这说明在当前 Amazon x1 高选择率样本中，80.0%--96.7% 均观察到两层收益，但不能把五个离散点宣称为连续区间定理。

100% 的空 predicate 是单独的 unfiltered control：两层相对单层为 1.196x，且两层最佳 `L=15` 的下侧已测到合法端点 `L=K=10`。它不能用于插值 96.7%--100%，也不能证明 97%--99% 的行为。

## 证据入口

- 正式配置：`experiments/multilevel_special/config.amazon_x1_high_selectivity_formal.json`
- 边界补测配置：`experiments/multilevel_special/config.amazon_x1_high_selectivity_formal_boundary_extension.json`
- 可提交汇总：`experiments/multilevel_special/results_summary/high_selectivity_{shared_summary,shared_points,oracle_summary}.csv`
- 完整 719 点：`experiments/multilevel_special/results_summary/source/high_selectivity_formal_all_points.csv`
- Raw runs：`runs/high_selectivity_formal_amazon_x1/` 与 `runs/high_selectivity_formal_boundary_extension_amazon_x1/`（不提交）
