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

## 待固化内容

- 新增选择率档位及可实现的实际均值。
- query-label 生成算法、随机种子、query 向量来源和 exact GT 生成命令。
- 依据 75% crossing 预声明的各层 coarse L-grid。
- 新 workload 的 Recall 门槛；原则上沿用 75% 的 0.87，除非生成工具或 ground-truth 语义证明 100% control 需单独定义。

## 当前状态

- 已确认现有 75% workload 为 `query_minlen1_avgsel75pct`，实际平均选择率 74.99381960%。
- 尚未生成 75% 以上 workload 或运行性能实验。
