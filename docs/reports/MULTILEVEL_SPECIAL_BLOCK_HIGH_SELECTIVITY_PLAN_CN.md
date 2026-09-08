# 75% 以上 Special Block 实验计划

本文档记录新增高选择率实验的预声明协议和恢复状态。结果产生后可以追加实测表，但不得回溯修改用于挑选赢家的质量口径。

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

- 六个新增 workload 与 exact GT 已生成；每档 10,000 个 GT 邻居均有效，ID 范围合法、距离有限且非负。
- Coarse 配置 dry-run 覆盖 12 methods × 6 workloads = 72 cases；尚未运行性能实验。
