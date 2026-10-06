# 论文大纲与论证设计

## 中心论点

集合过滤的近邻搜索有两个相互作用的成本：找到覆盖全部合法区域的入口，以及在合法区域内进行几何导航。基于前缀的入口 frontier 可以省去 minimal-superset 归约；安全 block 在谓词能整体授权时将多个 exact groups 合并成更大的导航单元。前者减少入口发现工作，后者改变搜索尺度，但稀疏连接和额外尺度均存在性能权衡。

方法核心按“入口覆盖 → 安全聚合 → 层内搜索”展开。自动阈值是配置方法，GPU 是构建方法；两者服务于核心索引，而非取代它成为论文主线。

## 建议正文结构（双栏研究稿的目标篇幅）

| 节 | 目标篇幅 | 应回答的问题 | 主图/证据 |
|---|---:|---|---|
| 1 Introduction | 1.25–1.5 页 | 为什么 eligibility 与 navigation 必须同时设计？为什么前缀入口与 block 互补？ | 三项贡献；谨慎引用实测收益与反例 |
| 2 Problem and Motivation | 1–1.25 页 | AND-label 模型、选择率方向、现有路线各承担什么成本？ | 同一应用例；机制面板 + 实测 0L 阶段分解 |
| 3 Prefix and Block Index | 1.5–2 页 | 如何定义 terminal forest、frontier 和 direct-member block？为什么必须保留 finer representation？ | LNG/Trie/授权块三联图；独立 partition |
| 4 Querying Certified Blocks | 1.5–2 页 | 合法入口如何覆盖前缀分支？候选如何标记层？为什么不计算非法点的距离？ | frontier 与搜索伪代码；两层 seed/淘汰例；覆盖、安全性和层隔离证明 |
| 5 Construction and Configuration | 1.5–2 页 | 如何批量构建图并限制显存？层数/阈值如何无需查询校准？ | GPU 描述符流水线；输出敏感成本；DRH；维护边界 |
| 6 Evaluation | 3–4 页 | 系统/入口/层数/拓扑/参数/GPU 各贡献多少？在何处失败？ | 0L、14×9 factorial、固定与逐档 oracle、held-out、build |
| 7 Discussion and Related Work | 0.75–1 页 | 与 UNG、Curator、ACORN、FAVOR、HNSW 的准确区别和局限 | 能力对照与待验证项 |
| 8 Conclusion | 0.2 页 | 证据支持的设计结论 | 不加入新结果 |

详细配置表、共享 Lsearch cap、完整 126 格数据和构建验证流程可放附录；正文必须保留负结果与最关键对照。上述篇幅是编辑目标，不能通过删去质量边界硬凑页数。

## 对原大纲的关键修正

| 原主张 | 审计判断 | 可用于正文的表述 |
|---|---|---|
| 全局图策略只适合中高选择率 | 过强，ACORN 等方法专门处理过滤稀疏性 | 合法点在图中稀疏/分散时，维持连通性和控制无效工作存在权衡；具体能力取决于构图、查询和数据相关性 |
| Curator 中低选择率不行 | 当前证据不支持，且历史外部比较有反例 | 分区/虚拟索引为 AND 提供另一组织方式；临时构造成本只有在对应原始论文机制或实测分解支持时才量化 |
| FAVOR 即暴力路由，因此不行 | 简化过度，需按一手论文定义 | 精确扫描的成本取决于 eligible cardinality 而非仅 selectivity；不把该模型推导变成对 FAVOR 的失败断言 |
| 我们不需要 ELS | 名称容易掩盖真实工作 | Trie 路径不需要 LNG minimal-superset reduction，但仍需要 prefix-frontier discovery 与向量 seed setup |
| 超低选择率 ELS 占比 90+% | 0.499% 的三个已报告阶段约 90%，非无仪表总耗时 | LNG 优化入口仍占该 profile 已列阶段约 90%；主 QPS 单独测量 |
| 所有层均无 LNG | 与最优混合配置不符 | prefix-only 是一个实例；整体方法允许逐层选择，当前最好固定配置为 T\|LT |
| Trie n-1，LNG 7n/O(n²)，所以 Trie 更快 | 混淆森林、典型值/最坏界与向量边 | prefix terminal forest 有 G-r 条组级边；LNG cover DAG 最坏 Θ(G²)；较少组边不推出更低 ANN latency |
| HNSW 自顶向下，我们自底向上 | 与当前 level-local 实现不符 | 本方法按谓词授权多个尺度并标记 seed，搜索不沿层间父子关系迁移 |
| 每层一样大 | 数据宇宙相同不等于节点数、覆盖、边数相同 | 每层是同一向量宇宙上的不同 label-block 组织；未覆盖残余留在 base，大小单独计量 |
| 一个静态索引所有选择率更好 | 与 10/30% 和 VariousImg 反例冲突 | 同一框架支持多个场景，当前无静态支配配置；automatic policy 的跨数据集差距如实报告 |
| 更新更容易 | 合理设计方向，当前未测在线更新 | prefix metadata 更新可局部描述，但 block repartition 和 proximity-edge repair 仍需处理；仅讨论，不称为已实现贡献 |

## 三项贡献建议

1. **前缀入口与连接的共同设计**：以每条兼容前缀分支上的首个合法 terminal 为入口，省去跨分支 minimal-superset 归约；说明为什么入口数可能增加，以及为什么不能只换边不换入口。
2. **谓词认证的多粒度 block 导航**：用公共前缀证明 direct members 合法；细层保留难以整体授权的区域，粗层提供跨 exact-group 的局部向量图；层标记约束搜索。单 overlay 的 0–1 物理层是讲解主例，2L 作为可扩展和消融实例。
3. **可构建、可配置的实现与证据**：分组 GPU 距离/top-k 批处理，显式 host 输出边界；DRH 从 N/R/C 生成计划；用完整 factorial、held-out 与负结果检验适用性。

## 投稿证据门槛

当前稿件能说明机制和已测 regime，尚不能以“高分录用”作为已证明状态。以下关键缺口需要单独完成：

- 当前固定协议下与原版 UNG、Curator、ACORN、FAVOR 的同质量跨选择率比较；历史 25/50/75% 表不可拼入新 9 档表。
- 外部系统的 filter preparation、index/partition work 和 search 分解，以支持各自 bottleneck 主张。
- prefix-only、mixed topology 和 L0=Trie 的跨数据集实验；当前完整 factorial 仅为 Amazon。
- 参数选择的层数变化实验；现有 held-out 自动计划均为两个 upper overlays。
- 构建加速赢家的最终查询质量门禁；backend execution 和 round-trip 校验不替代 equal-Recall quality。
- 正式重复和误差分析；当前 query 1 cold + 2 warm 只支持 screening 结论。

这些缺口不应伪装成已验证，也不通过文学化叙述消除。独立审阅记录随修订一起交付。

## 当前写作纪律

每段服务定义、推导、证据或解释。删除只为防止被质疑的重复自我免责；必要的测量口径与证明条件只在对应位置说明一次。结果按 matched topology、发现成本转移、自动方案转移、构建成本组织，构建 backend 清单移到附录。

本轮新增 workload 元数据核对：选择率是混合谓词批次均值，多个高档位通过频繁标签替换获得；方法效果只能先对应这些 workload families。人工候选明确列出，单层候选在统一 presence gate 下回退 base。
