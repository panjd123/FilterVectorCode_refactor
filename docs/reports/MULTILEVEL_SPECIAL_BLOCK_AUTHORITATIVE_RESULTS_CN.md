# ML-UNG 权威实验报告

> Pending: 最终 fail-closed 流水线将在 Amazon、held-out、GPU build 和
> instrumented profile 全部通过验证后，原子替换本文件。历史 mixed-edge 与
> 人工调参结果不属于本报告的证据范围。

最终版本将由
`experiments/multilevel_special/generate_authoritative_paper_results.py`
从经过验证的 CSV、配置、冻结 policy 和 run manifest 生成，包含：

- Amazon 九档选择率上的零层、单层、双层和 gated DRH 等 Recall-QPS 结果；
- LNG/Trie topology 与三种 entry strategy 的正交消融；
- Genome、Reviews、VariousImg 上 DRH 与冻结人工 oracle 的差距；
- entry-group、entry-point、authorization、graph search 阶段耗时；
- visited points、base/special edges 和 distance calculations；
- GPU 构建时间、资源、索引大小和下游查询质量；
- 六类完整 Recall-QPS 曲线及负结果、适用区间和限制。
