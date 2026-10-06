# 文献与写作备查

本文保存相关出版记录与写作借鉴；面向导师的核心方法比较见 [中文汇报](ADVISOR_NOTE_CN.md)。具体外部主张的来源见 [文献审计](review/literature_claim_audit.md)。

## 已核对的相关工作

本文采用“具体瓶颈 → 结构观察 → 算法 → 有条件性质 → 对应实验证据”的组织，而不复制原论文表述。下列论文均已纳入引用；年份按正式出版记录。

| 已发表相关论文 | 出版记录 | 本文借鉴或区别 |
|---|---|---|
| UNG | PACMMOD 2(6), 2024 | entry、label plane 与 vector graph 必须一起解释 |
| ACORN | PACMMOD 2(3), 2024 | predicate-agnostic expansion；有低选择率 fallback，不能笼统说不适用 |
| SeRF | PACMMOD 2(1), 2024 | 先定义不可直接物化的结构，再说明压缩组织 |
| iRangeGraph | PACMMOD 2(6), 2024 | 区分离线结构与在线组合 |
| UNIFY | PVLDB 18(4), 2024 | 多策略统一索引；其区间结构不同于集合偏序 |
| Dynamic Range-Filtering ANNS | PVLDB 18(10), 2025 | 动态约束须有独立算法与测量 |
| Efficient Dynamic Indexing for RF-ANNS | PACMMOD 3(3), 2025 | 区分查询、空间、更新的权衡 |
| Starling | PACMMOD 2(1), 2024 | block 与数据布局服务 I/O，不等同于我们的授权 block |
| ELPIS | PVLDB 16(6), 2023 | 构建、内存和查询应共同报告 |
| LSH-APG | PVLDB 16(8), 2023 | 按构建阶段解释复杂度与收益 |
| tau-MNG | PACMMOD 1(1), 2023 | 可证明性质与实际近似版本分开 |
| Revisiting PG-based ANNS Construction | PVLDB 18(6), 2025 | 构建加速须检查查询质量 |
| Elastic Index Selection | PVLDB 19(4), 2025 | 共享部分索引及选择问题，不等同单索引 block scales |
| Curator | PACMMOD 4(1), 2026 | 其 AND 临时索引组织不需要距离计算，不能宣称必然昂贵 |
| FAVOR | PACMMOD 4(3), 2026 | 低选择率 brute-force，其余 exclusion-distance HNSW；不是仅暴力扫描 |
| SIEVE | PVLDB 18(11), 2025 | collection of indexes，与独立导航 overlays 区分 |

还补充 LSSG（arXiv:2609.15058）及 Query-aware Routing（arXiv:2606.19898），明确标为预印本。前者已有 label-similarity tiers，所以“首次多层标签图”不成立；我们的区别是 direct-member aggregation、root-prefix authorization 与 level-local states。后者用离线性能表和 Recall 模型，DRH 则放弃这类校准，也承担适应性不足。

完整逐条核查见 `review/literature_claim_audit.md`。UNG 的 ACM 全文拉取返回 403，其机制通过作者官方代码核验。
