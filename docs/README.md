# 文档索引

当前研究交付是 ML-UNG：前缀入口、多粒度 block 及逐层 LNG/Trie 连接。第一次阅读可从中文汇报开始，再根据目的进入论文、数据或实验系统。

## 当前论文与结果

| 文档 | 适合解决的问题 |
|---|---|
| [中文汇报说明](papers/multilevel_ung/ADVISOR_NOTE_CN.md) | 我们解决什么问题？贡献与主要证据是什么？ |
| [论文 PDF](papers/multilevel_ung/main.pdf) | 完整定义、方法、机制图和实验论证 |
| [LaTeX 说明](papers/multilevel_ung/README.md) | 如何编译、从已有 raw evidence 重新生成论文 |
| [九档权威结果报告](reports/MULTILEVEL_SPECIAL_BLOCK_AUTHORITATIVE_RESULTS_CN.md) | 当前 Amazon、held-out 自动方案及构建实验 |
| [完整 126 格数据](papers/multilevel_ung/generated_data/factorial_equal_recall.csv) | 两种 L0、0/1/2 个 overlay 的全部 crossing/NC 结果 |
| [实验系统](../experiments/multilevel_special/README.md) | 如何声明组合、检查配置、执行与验证实验 |

当前比较使用 100 个 CPU query workers，在各方法达到 Recall@10 ≥ 0.90 的最小实测预算处比较 QPS。1 次 cold 与 2 次 warm 的结果属于 screen；正式重复协议和历史 cohort 在其对应配置中单独定义。批次平均选择率、各计时阶段及 `NC` 的定义见论文实验节和附录。

## 实现与运行资料

| 入口 | 内容 |
|---|---|
| [`UNG/codes`](../UNG/codes) | 当前算法实现及入口 provider、block、搜索与构建模块 |
| [实验 CLI](../experiments/multilevel_special/experiment_cli.py) | `check / matrix / run / validate / summarize` |
| [构建开关](runbooks/UNG_BUILD_MODE_SWITCHES_CN.md) | 已有 CPU/GPU 构建路径的配置说明 |
| [方法注册表](runbooks/UNG_METHOD_REGISTRY_CN.md) | 构建与搜索实现的名称、入口函数和选项 |
| [测试说明](runbooks/TESTING_GUIDE.md) | 代码测试和既有验证工具 |

实验机的数据、向量图和 raw runs 通过配置指定；它们不随源码下载。对于现成配置，先使用 `matrix` 查看语义，再按照实验系统文档检查环境和路径。

## 历史研究与对应证据

这些文档保留了早期实现、参数搜索和研究过程。表中的“历史”表示其数据、搜索语义或计时协议与当前论文不同。

| 研究阶段 | 文档 |
|---|---|
| 早期 GPU 构建与入口优化 | [三条主线性能报告](reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md)、[GPU 工作展示](reports/WORK_PRESENTATION_DEEPRESEARCH_CN.md) |
| 早期单层/双层参数搜索 | [完整历史结果](reports/MULTILEVEL_SPECIAL_BLOCK_COMPLETE_RESULTS_CN.md)、[历史论文报告](reports/MULTILEVEL_SPECIAL_BLOCK_PAPER_REPORT_CN.md) |
| 六档与早期高选择率复现 | [对应复现记录](reports/MULTILEVEL_SPECIAL_BLOCK_REPRODUCE_CN.md) |
| 早期自动配置研究 | [QF/SSL 自动方案](reports/QF_SSL_AUTO_POLICY_REPORT_CN.md) |
| 早期交接与代码结构 | [交接记录](reports/REVIEW_READY_HANDOFF_CN.md)、[代码深读](REFACTOR_DEEP_DIVE_CN.md) |
| 原始版本、迁移与补丁 | [`archive/`](archive/) |

跨报告比较时，以配置、binary hash、索引、query 文件和计时协议识别 cohort。历史文档中的“当前”和“最优”指其记录时的研究阶段。

## 作者与维护者记录

- [当前工作看板](../AGENT_KANBAN.md)：进行中的任务、检查点和恢复入口。
- [论文大纲](papers/multilevel_ung/PAPER_OUTLINE_CN.md)：论证结构和段落职责。
- [审阅与证据审计](papers/multilevel_ung/review/)：独立审阅输入、问题处理和来源核对。
- [早期投稿证据矩阵](papers/EVIDENCE_MATRIX_CN.md)：早期 GPU 主线的 claim 记录。

审阅记录用于追溯修改依据；论文与中文汇报自身提供方法和实验论证。
