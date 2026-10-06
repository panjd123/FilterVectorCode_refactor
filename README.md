# FilterVectorCode / ML-UNG

这个仓库研究标签 AND 过滤下的近似最近邻搜索：先确定满足查询标签的向量区域，再在这些区域内寻找相近向量。当前研究主线是 **ML-UNG 的前缀入口与多粒度 block 导航**，基于 UNG 的 exact-label groups 和向量图实现。

ML-UNG 将三个选择分别配置：

| 模块 | 作用 | 当前选项 |
|---|---|---|
| 分层与分区 | 将 exact groups 聚合成可由公共前缀授权的 block | 独立阈值序列；支持多个 overlay |
| 层内连接 | 决定 group/block 之间允许的导航关系 | 每个物理层分别选择 LNG 或 Trie |
| 入口组寻找 | 为指定连接方式提供搜索入口 | 原始 LNG、优化 LNG、Trie frontier |

查询状态只使用所属物理层的边。`2L[T|LT]` 表示 Trie 基层 L0、LNG block 层 L1 和 Trie block 层 L2；`2L` 是两个 overlay，共三个物理层。阈值控制何时发出 block，不是 block 点数上限。入口算法与拓扑的覆盖条件见[方法说明](docs/papers/multilevel_ung/ADVISOR_NOTE_CN.md)。

## 从哪里开始

| 目的 | 阅读入口 |
|---|---|
| 向老师说明问题、方法和结果 | [中文汇报说明](docs/papers/multilevel_ung/ADVISOR_NOTE_CN.md) |
| 阅读完整研究论证与图示 | [论文 PDF](docs/papers/multilevel_ung/main.pdf) / [LaTeX 与编译说明](docs/papers/multilevel_ung/README.md) |
| 查看当前九档实验和全部配置 | [当前结果报告](docs/reports/MULTILEVEL_SPECIAL_BLOCK_AUTHORITATIVE_RESULTS_CN.md) / [126 格原始汇总](docs/papers/multilevel_ung/generated_data/factorial_equal_recall.csv) |
| 配置、运行或检查实验 | [实验系统](experiments/multilevel_special/README.md) |
| 查找代码、运行手册及历史研究 | [文档索引](docs/README.md) |
| 接续当前优化任务 | [工作看板](AGENT_KANBAN.md) |

## 如何理解当前结果

当前 Amazon topology 比较包括 14 种配置与 9 个查询批次，使用 100 个 CPU query workers、1 次 cold 和 2 次 warm，在每种方法达到 Recall@10 ≥ 0.90 的最小实测搜索预算处比较 QPS。选择率标注为**批次平均值**；几个批次由选择性谓词和高频 singleton 混合而成。

L0 的 LNG/Trie 比较分别使用与其匹配的入口算法；固定 L0 和入口、只替换一个 upper-layer topology 的对照用于分析该层连接方式。不同工作负载的最佳配置会变化。完整报告保留未达到 Recall 门槛的 `NC` 和自动方案的退化结果。

当前论文的查询在 CPU 上执行；GPU 用于所声明的构建后端。构建表同时区分 hierarchy 时间与 base-plus-hierarchy 的分阶段组合时间。

## 代码与实验位置

- [`UNG/codes`](UNG/codes)：分组、入口 provider、LNG/Trie 连接、block 构建与查询实现。
- [`experiments/multilevel_special`](experiments/multilevel_special)：配置、运行、验证、结果汇总与论文生成。
- [`docs/papers/multilevel_ung`](docs/papers/multilevel_ung)：当前论文、中文汇报、冻结数据与审阅记录。
- [`UNG/codes/tools`](UNG/codes/tools)：数据转换和 ground-truth 工具。
- [`ACORN`](ACORN)、[`thirdparty`](thirdparty)：对照实现及依赖。

在已配置的实验机上，从仓库根目录查看配置语义：

```bash
python3 experiments/multilevel_special/experiment_cli.py matrix \
  experiments/multilevel_special/config.authoritative_amazon_base_trie_query_grid.json
```

实际数据、索引与 raw runs 不包含在 Git 仓库中；配置中的路径需对应实验环境。编译、数据要求、dry-run 和结果验证见实验系统文档。

## 相关工作与历史入口

- [UNG: Navigating Labels and Vectors](https://doi.org/10.1145/3698822)
- [ACORN: Performant and Predicate-Agnostic Search](https://arxiv.org/abs/2403.04871)

早期 GPU cross-edge、group-graph、入口 microbenchmark 和原始 UNG/ACORN 脚本记录保存在[历史版本的仓库指南](https://github.com/panjd123/FilterVectorCode_refactor/blob/8c0b768d39b8f5040d995d60fb312e6e0607ca74/README.md)及[文档索引](docs/README.md)。这些记录对应各自保存的版本、数据和计时边界。
