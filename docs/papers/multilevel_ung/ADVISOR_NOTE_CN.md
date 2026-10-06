# ML-UNG 方法与论文汇报说明

> 这份 note 对应持续润色后的正文。阅读顺序：本文 → `main.pdf` 的方法图与结果 → `generated_data/` 完整数据。

## 1. 先讲清楚我们解决什么

我们研究标签 AND 过滤的近似最近邻搜索。查询标签为 Q，向量标签为 A；只有 Q ⊆ A 的向量可以进入结果。选择率是符合条件的向量数占全库向量数的比例，低选择率表示可用向量少。

UNG 把标签集合完全相同的向量分为一个 group，用最小严格超集关系连接 groups（LNG），再把这些关系落成向量边。它已经使用 Trie 辅助寻找入口；我们的改变是用 terminal-prefix 关系连接 groups，并设计与之配套的入口 frontier。

我们的主线有两个相互配合的机制：**把 terminal-prefix 连接与覆盖它的入口 frontier 一起设计；再把可整体证明合法的多个 groups 聚成 block，改变向量搜索的粒度。** GPU 批处理降低额外图的构建成本；自动参数 DRH 提供一个可解释的配置实例。

## 2. 用一个例子解释入口机制

设标签顺序 a < b < c，已观察到 {b}、{a,b}、{a,b,c}、{a,c}。查询 Q={b} 时，前三个 group 合法。

- LNG 可以从 {b} 通过集合包含关系走到 {a,b}，再到 {a,b,c}，因此只需 {b} 作为标签平面的入口。
- Trie 的 {b} 和 {a,b} 位于不同前缀分支；必须保留这两个入口。{a,b,c} 已能从 {a,b} 到达，不必再保留。
- 因此 prefix frontier 保留“合法且没有合法 terminal 前缀祖先”的 groups。它省去了不同分支之间的最小超集筛选，但可能产生更多向量入口。

实际计时的 Trie provider 使用 label-to-group 压缩位图交集，再查询缓存的 terminal 祖先；并非只做一次理想化 Trie 遍历。优化后的 LNG provider 已采用 cardinality buckets 和缓冲复用，但仍执行成对的集合包含判断。

论文 Proposition 1 证明：从完整 prefix frontier 出发，所有合法 terminal 在标签平面可达。它**不保证有限搜索预算下的 Recall**：入口裁剪、局部有向图可达性、向量 cross edges 和候选队列都可能限制覆盖。含 r 个根的 prefix forest 有 G−r 条 group 边；它不一定是 LNG cover edges 的子集。

## 3. 为什么还需要 block

以 Rb={a,b} 为根的 block 可以包含 {a,b}、{a,b,c} 的 direct members。若 Q ⊆ Rb，整个 block 都合法，内部向量边可以直接跨越 exact-group 边界。若 Q={b,c}，该 block 不能整体授权，但其中部分向量仍合法，必须依靠更细的表示。

每一层从 canonical Trie 独立做 uncovered-mass 聚合，非根节点累计未覆盖质量严格超过 T 时发出 block。T 是发出阈值，不是 block 最大点数；可能残留未被任何 block 覆盖的 groups。每层 direct members 互不重叠，不同层的 direct-member 分区不保证严格嵌套。

| 记号 | 物理 L0 | 物理 L1 | 物理 L2 |
|---|---|---|---|
| `0L[L]` | exact groups，LNG 连接 | 无 | 无 |
| `0L[T]` | exact groups，Trie 连接 | 无 | 无 |
| `1L[T|L]` | Trie groups | 阈值 T1 的 LNG blocks | 无 |
| `2L[T|LT]` | Trie groups | 阈值 T1 的 LNG blocks | 阈值 T2 的 Trie blocks |

`mL` 的 m 是 upper-overlay 数，物理层共 m+1 层。`L/T` 表示该层的 LNG/Trie 连接。旧图中 `2L-LT` 默认省略 LNG base，完整记号为 `2L[L|LT]`。L0 可以自由选 LNG 或 Trie，不存在方法定义要求它必须是 LNG。

查询状态为 `(vector_id, distance, physical_level)`，只扩展所属层的边；顺序按 distance、vector ID、level 依次比较。不同层共享有界候选队列，但状态不晋级、不向下 fall through。入口 group 会被标记为拥有它的最细合法 overlay；不保留一份重复 L0 seed。所有 seeds 先计距离并标为已见，再优先保留 block seeds。队列满时，新候选会淘汰更差的状态；被淘汰状态的 seen 标记保留，最终答案只从终止时仍在队列中的状态按 vector ID 去重产生。正文新增九个 singleton groups 的两层例子，展示 `(y1,L1)` 和 `(y1,L2)` 同时入队后，高层状态在首次扩展前被淘汰。

Proposition 2 的谓词安全依赖：seed 合法、overlay seed/edge 的 owner 被授权、层内 direct membership 正确、跨 block 边沿根标签超集方向。实现依靠这些不变量，而不是在每条邻边上补一次过滤。Proposition 3 证明层隔离；回退只保证相同 base 搜索路径，不包含额外授权和路由时间。

## 4. 当前数据说明什么

Amazon：602,453 个 768 维向量，482,387 个 exact groups；每档 1,000 queries，100 query threads，1 cold + 2 warm。Recall@10 crossing 要求两次 warm 均达到 0.90，取最小实测 Lsearch，不插值。L2 距离作用于已有浮点坐标，Trie 按存储的整数 label ID 升序组织。5/30/60/80/95% 档通过将部分查询谓词替换为高频 `{1}` 得到；这是混合查询的平均选择率，例如 5% 档单次查询覆盖率可从 0.0033% 到 96.702%。因此不能把档位当作每条查询的固定选择率。完整 factorial 为 14 种 topology × 9 个选择率，T1=1024、T2=16384、ungated。跨 L0 使用各自 matched provider，是系统比较；同一 L0/provider 下只改变一个 upper 字母才隔离该层 topology。

| 平均选择率 | `0L[T] / 0L[L]` QPS | 完整 factorial 最优配置 | 最优 QPS |
|---:|---:|---|---:|
| 0.499% | 6.03× | `2L[T|LT]` | 21707.89 |
| 0.903% | 2.77× | `2L[T|LT]` | 14278.50 |
| 5.038% | 18.43× | `2L[T|LT]` | 11718.35 |
| 9.907% | 0.60× | `0L[L]` | 450.17 |
| 30.027% | NC | `0L[L]` | 52.77 |
| 60.047% | 0.62× | `1L[T|L]` | 1613.68 |
| 80.024% | 0.33× | `1L[T|L]` | 648.35 |
| 95.020% | 0.33× | `2L[T|LT]` | 564.63 |
| 99.001% | 0.56× | `2L[T|TT]` | 267.67 |

NC 表示共享实测预算内没有 crossing，不代表任意预算都不能达到目标。最优配置是逐档事后选择，不是已经实现的自动 selector。固定 `2L[T|LT]` 的全网格 oracle-normalized 几何均值为 0.868，最差档为 0.482，说明它也不支配所有场景。

受控上层替换：固定 Trie base、Trie entry、L1=LNG，仅把 L2 从 LNG 换成 Trie，8/9 档更快，最大 1.251×，80% 为 0.998×。这支持“coarse L2 采用 Trie 值得优先考虑”的经验建议，不能推出每层都用 Trie 更优。

独立 profile 可以解释入口与搜索的成本转移：

| 平均选择率 | 系统 | ELS ms/query | 向量 seed setup ms/query | Graph ms/query |
|---:|---|---:|---:|---:|
| 0.499% | LNG + optimized LNG entry | 20.352 | 0.195 | 2.076 |
| 0.499% | Trie + prefix entry | 0.645 | 1.769 | 0.868 |
| 9.907% | LNG + optimized LNG entry | 36.631 | 0.646 | 98.747 |
| 9.907% | Trie + prefix entry | 3.778 | 14.744 | 323.081 |

这些是带 instrumentation 的多线程 per-query 阶段均值，不能求和后作为 QPS 的 batch latency。QPS 是 B / warm-median batch seconds；B 在 held-out 中为 2,864 或 3,000，不全是 1,000。论文将 CSV 的 `Visited` 显示为 **Encounters**。Encounters、edges、distances 分别为通过邻接首次发现的状态数、扫描邻接项数和距离计算次数。Visited 在距离计算/队列准入前增加，拒绝入队的候选也计数；它不是 pop/expansion 次数。初始化 seed 计距离，不计这次邻接发现。`timeout` 专指 0L-Trie/95% 详细 profile 超过 3300 秒，与性能表 NC 分开。

## 5. 自动方案与 GPU 构建怎样定位

DRH 用 T+N/T 的对称结构代理选择 T1≈sqrt(N)，取最近二次幂；rho=max(2,round(R/C))，下一层阈值乘 rho；当 N/T<C 时停止。预计 block 数 N/T>R 则该层选 LNG，否则选 Trie。N/T 只是代理，不能当作实际 block 数或已证明的延迟模型。

N=602453、R=64、C=4 得到 1024:LNG、16384:Trie 两层。DRH-v1 在存在合法物理层 h≥2 时进入多层后端，DRH-v2 还要求最高合法 h≥2 的 direct mass 达到下一未物化尺度。当前 held-out DRH 固定 LNG base，不自动选择全部系统组件；不读取查询频率、计时或 Recall，但单次查询授权自然要读取 Q。

Amazon 60%–99% 下自动方案为 Plain 的 14.40–26.70×；VariousImg 有明确退化。四个数据集都导出两个 overlays，所以还未验证自动深度变化后的性能。人工候选以自动 `(t,16t)` 为中心：`1L[L|L](t)`、`2L[L|LL](t,16t)`、`2L[L|TT](t,16t)`，以及 `2L[L|LT](t/2,8t)` 和 `2L[L|LT](2t,32t)`。三组 t 分别为 256/512/1024。它们使用相同 presence gate，因此单层候选始终回到 base。

GPU 把 irregular local-graph 与 cross-edge tasks 组织成批，融合 distance/top-k，并在需要时只回传 IDs。小 block 默认 exact top-R，中 block sampled CPU Vamana，大 block CUDA candidate refinement；metadata 和最终 adjacency 仍在 host。当前 fastest hybrid sidecar 90.49s，独立 stage medians 组合总计 141.27s，原 CPU base 190.38s，即约 1.35×。这不是单次联合构建 wall time，也未证明最快构建索引保持同等查询质量。论文主表已移除仅两次 repeats 的 bootstrap CI 列；原始统计及中文实验报告保留该数值，正文按实际 timing 和 resource 解释。

## 6. 文献定位与写法

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

## 7. 还差什么才能形成更强投稿证据

| 项目 | 当前状态 | 剩余工作 |
|---|---|---|
| 正文、机制图、定义和实现一致性 | 已重写并独立审阅 | 根据导师意见继续压缩与取舍 |
| Amazon 完整 topology 网格 | 126 单元完整保留 | 更多数据集上的同类 factorial |
| 自动参数 vs 人工 | 有冻结候选集与反例 | 覆盖自动导出不同深度的数据形态 |
| 入口机制因果归因 | 有 matched-system profile | 同 topology 的三 provider 受控延迟对照 |
| 同协议 SOTA 全选择率 | 尚不完整 | 不混用历史外部对照，补齐当前协议 |
| GPU 完整构建耗时 | 有 repeated timing/resource | 最快 hybrid 等各配置的等 Recall 质量验证 |
| 置信区间 | screen-level / 探索性 | 更多交错重复，控制竞争与机器状态 |

向老师汇报时建议先讲例子和两张机制图，再给低选择率入口/搜索 breakdown、完整 topology 最优表，最后介绍自动方案、GPU 与尚缺证据。论文定位应建立在可验证的区别上，而不是“所有选择率都优于已有工作”。

数据来源核对：原始 embedding 模型、存储前归一化和部分旧 query 文件的生成记录尚未找全；当前实验使用已保存的预处理快照。来源记录见 `review/workload_metadata_audit.json`。
