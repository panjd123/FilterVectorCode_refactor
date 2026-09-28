# ML-UNG 教师汇报说明

## 一句话结论

ML-UNG 保留 UNG 的 exact-label-group 分组，但把 L0 的组间 topology 显式开放为 LNG 或 Trie，并在其上增加按标签 Trie 子树质量独立划分的多粒度 overlay；查询只在被精确授权的 block 上运行，并且每个搜索状态只扫描自身物理层的边。当前无需校准的 DRH 默认保留 LNG 作为稳健 L0，再用数据规模和已有图度数预算自动决定 overlay 数量、阈值、topology 和路由门限，不读取查询分布、Recall 或延迟。L0 是否也应选 Trie 由完整 factorial 单独检验，而不是预先假设。

## 统一术语

- `mL` 中的 `m` 是 level 0 之上的 upper-overlay 数量，不是物理层总数。
- `0L`：只有物理 level 0。该层由 exact-label groups 和它们之间的 group topology 构成。
- `1L`：物理 level 0 加一个 level 1 overlay。level 1 使用阈值 `T1` 独立划分，是较细的 coarse-grained overlay。
- `2L`：物理 level 0、1、2 共三层。level 1 使用 `T1`，level 2 使用更大的 `T2`，因此 level 2 更粗。
- `L` 表示 LNG topology，`T` 表示 materialized prefix-Trie topology。完整记号写作 `mL[B|U]`：竖线左侧 `B` 是 L0，右侧 `U` 按从细到粗的顺序列出 upper levels；例如 `2L[T|LT]` 表示 L0=Trie、L1=LNG、L2=Trie。
- `2L-XY` 的两个字母只描述 upper overlays：`X` 对应 level 1，`Y` 对应 level 2；level 0 不编码在这两个字母里。
- `2L-LL`、`2L-LT`、`2L-TL`、`2L-TT` 分别表示 level 1/2 为 LNG/LNG、LNG/Trie、Trie/LNG、Trie/Trie。
- 原权威 Amazon 自动方案 `2L-LT(T1=1024,T2=16384)` 的完整写法是 `2L[L|LT]`：level 0 为 exact-group LNG；level 1 为 threshold-1024 LNG overlay；level 2 为 threshold-16384 Trie overlay。
- 实现允许 level 0 独立选择 LNG 或 Trie，也允许它与任意 upper-overlay 组合。新增的完整二元 topology factorial 覆盖 `0L` 的 2 种、`1L` 的 4 种和 `2L` 的 8 种组合，不再把 L0=LNG 当作默认事实。
- 固定 L0 的比较用于隔离 upper topology；跨 L0 比较采用各自能够达到 Recall 的 matched entry provider（LNG base 配 optimized-LNG entry，Trie base 配 Trie entry），因此是端到端系统比较，不应写成纯 base-edge 因果效应。
- `entry=orig-LNG`、`entry=opt-LNG`、`entry=Trie` 是入口组算法，与 graph topology 是不同实验维度。
- `ungated` 仍执行逐 block 的精确授权，只是不做“是否进入多层后端”的全局前置回退。
- 当前 2L 实验中的 `DRH-v1`，要求至少有一个合法 level-2 block 才进入多层后端，否则直接调用不变的 0L 搜索。
- `DRH-v2` 在 DRH-v1 之上，还要求合法 level-2 blocks 的 direct-member mass 总和至少达到自动推导的 `T3`。

## 方法主线

### 1. 现有瓶颈

UNG 已经解决“标签可达性”和“向量邻近性”如何合并的问题，但它的每个导航单元仍是一个 exact-label group。宽谓词会命中大量小 group；即使谓词完全合法，搜索也要重复扫描大量细粒度 group 和 cross-group edges。

### 2. 核心观察

标签 Trie 中，以前缀 `Rb` 为根的子树内，每个 label set 都包含 `Rb`。当查询标签 `Q` 满足 `Q subseteq Rb` 时，该 block 的所有 direct members 都满足谓词。这个包含关系给出了不依赖查询样本的安全粗化条件。

### 3. 三个彼此分离的设计维度

1. Hierarchy：用 `T1 < T2 < ...` 独立划分每个 overlay，增加粗粒度导航尺度，但不改写 level 0 或更细 overlay。
2. Topology：每个物理层内部可使用 LNG 或 prefix-Trie 连接；Trie topology 指实际 materialized inter-block edges，不是“用 Trie 找入口”。
3. Entry and routing：入口 provider 只决定初始 group；router 决定调用多层后端还是 0L fallback；二者都不改变某一层内部的边。

### 4. 查询不变量

搜索状态是 `(vector_id, distance, physical_level)`。level 0 状态只扫描 base edges；level 1/2 状态只扫描对应 overlay 拥有的边。不同层可以向同一个 bounded queue 提供 seed，但状态不会跨层晋级，也不会在缺边时 fall through 到另一层。构建时的父子/归属关系不是查询边。

### 5. 自动参数 DRH

DRH 取 `T1` 为最接近 `sqrt(N)` 的二次幂，以 `rho=max(2,round(R/C))` 递增阈值，并在 `N/Tl < C` 时停止；预计 block 数大于局部度预算 `R` 时使用 LNG，否则使用 Trie。它因此一次性给出 overlay 数量、每层阈值和拓扑。DRH-v2 再把下一个未物化尺度作为 direct-mass gate，仍然不需要 query calibration。

## 当前结果应该怎样表述

- 查询结果是 screen-level evidence：每个点 1 次 cold、2 次 warm；100 个 query workers；固定 1,000 条查询。
- Mean selectivity 是查询批次中 `|eligible(Q)|/N` 的均值，不是 DRH 输入。
- Recall@10 是每条查询 Recall@10 的批次均值。equal-Recall crossing 是两次 warm 都达到 0.90 的最小实测 `Lsearch`，不插值。
- Plain QPS 是 0L-LNG 的原始吞吐；表中 `A/plain` 是各自 equal-Recall crossing 上的 `QPS_A/QPS_plain`。
- `NC` 只表示在 workload 共享的实测 `Lsearch` 上限内没有 crossing，不表示无限增大预算也无法达到目标。
- 完整的 `base x overlay` factorial 已覆盖 14 种 topology、9 个选择率，共 126 个等 Recall 单元；两半使用同一个查询二进制和同一个 hierarchy-builder 二进制。逐档 QPS 最优依次为：0.5/1/5% 的 `2L[T|LT]`，10/30% 的 `0L[L]`，60/80% 的 `1L[T|L]`，95% 的 `2L[T|LT]`，以及 99% 的 `2L[T|TT]`。不存在支配全部 workload 的静态 topology。
- 在只比较 0L 的 matched-provider 系统时，`0L[T]` 在 0.5/1/5% 是 `0L[L]` 的 6.03/2.77/18.43x；10% 为 0.60x，30% 在共享预算内 NC（`Rmax=0.8882`），60/80/95/99% 分别为 0.62/0.33/0.33/0.56x。因此不能写成“Trie base 普遍优于 LNG base”，也不能把这些差值全部归因于边 topology，因为入口 provider 同时不同。
- 加入 overlay 后，所有 Trie-base 组合都在 9 档达到 Recall 门槛。按逐 workload oracle 归一化后的全网格几何均值，固定配置 `2L[T|LT]` 最高（0.868），但其最差档只有 oracle 的 0.482（30%）；这说明它是当前最好的单一固定组合，却仍不能替代 query routing。
- 最干净的 upper-topology 对照显示：固定 Trie base、Trie entry 和 L1=LNG，只把 L2 从 LNG 换成 Trie（`2L[T|LL] -> 2L[T|LT]`），9 档中 8 档加速，最高 1.251x；唯一未胜的 80% 为 0.998x，近似持平。相反，只把更细的 L1 从 LNG 换成 Trie（`2L[T|LL] -> 2L[T|TL]`）虽在 5/9 档胜出，却在 10/60/80/95% 慢 40.7%--52.9%。当前证据更支持“Trie 用作稀疏的 coarse L2，LNG 保留在 fine L1”，而不是所有层统一 Trie。
- Amazon 的自动 2L 方案在 60%--99% selectivity 相对 0L baseline 为 14.40--26.70x；这是目标宽谓词区间的证据，不是所有数据集、所有选择率上的普遍优势。
- 0L-Trie 在 Amazon 0.5%、1%、5% 为 Plain 的 6.03x、2.77x、18.43x，但在 10% 只有 0.60x。这是 level-0 topology 与配套 Trie entry 的联合效果，不应写成“多一层带来的收益”。
- DRH-v1 在 Genome 和 Reviews 接近 Plain 及冻结的人工候选集，但在 VariousImg 明显退化；DRH-v2 缓和但没有消除该反例。
- 重复构建实验中最快的 composed base-plus-hierarchy 路径为原始 CPU base builder 的 1.35x。该数值是独立 stage medians 的组合，不冒充一次联合 wall-clock。

## 与 UNG 及相邻工作的区别

ML-UNG 的贡献不是“第一次在入口处使用 Trie”。UNG 已经使用 Trie 找 minimal-superset entry groups。本文的 Trie topology 是实际物化的 inter-group/inter-block connectivity。ML-UNG 也不是 HNSW 式的随机抽样层级：HNSW 的层用于向量空间导航，ML-UNG 的 overlay 聚合标签空间区域，并由精确谓词包含关系授权。

Range-filtered ANNS 工作同样关注不同选择范围对应不同图粒度，但它们通常利用一维有序区间的分解、压缩或拼接。集合 containment 是偏序关系；两个合法 supersets 不一定落在同一个 prefix branch，因此不能把 range index 的论证直接移植到这里。

## 检索并核验的 SIGMOD/PVLDB 论文

以下 12 篇论文的题目、venue、年份和 DOI 已核验。前 6 篇直接讨论 filtered/range-filtered vector search，后 6 篇用于借鉴图索引、构建和系统论文的论证方式。

| Paper | Venue | 与本文的关系及写作借鉴 |
|---|---|---|
| UNG, *Navigating Labels and Vectors* | PACMMOD/SIGMOD 2024, DOI `10.1145/3698822` | 最近基线。先列过滤向量搜索的四个困难，再用一个统一抽象串起 label plane 与 vector plane；本文沿用这种“问题 -> 抽象 -> 性质 -> 端到端结果”的主线。 |
| ACORN | PACMMOD/SIGMOD 2024, DOI `10.1145/3654923` | 从不可实现的理想 predicate subgraph 出发，再说明可实现机制如何逼近；启发本文先写安全 coarse navigation 的理想条件，再落到授权和 level-local expansion。 |
| SeRF | PACMMOD/SIGMOD 2024, DOI `10.1145/3639324` | 先指出“为每个范围建图”的二次爆炸，再给出可压缩结构和复杂度；启发本文明确 exact-group 单尺度瓶颈和独立 overlay 的结构代价。 |
| iRangeGraph | PACMMOD/SIGMOD 2024, DOI `10.1145/3698814` | 把 materialized elemental graphs 与 query-time composition 分开；启发本文明确 offline partition/topology 与 online authorization/search 两条路径。 |
| UNIFY | PVLDB 2024, DOI `10.14778/3717755.3717770` | 系统化区分 pre/post/hybrid filtering，再说明统一索引如何支持三者；启发本文把 hierarchy、topology、entry、routing 明确拆成正交接口。 |
| Dynamic Range-Filtering ANNS | PVLDB 2025, DOI `10.14778/3748191.3748193` | 先说明静态方法在 arrival order 上为何失效，再给动态 segment graph；启发本文把“不适用边界”写成具体结构差异，而不是泛泛局限。 |
| RangePQ, *Efficient Dynamic Indexing for RF-ANNS* | PACMMOD/SIGMOD 2025, DOI `10.1145/3725401` | 围绕空间、查询和更新三方权衡组织设计；其 two-layer 是空间组织，不等同于本文的 label-navigation overlay，报告时必须避免混淆。 |
| Starling | PACMMOD/SIGMOD 2024, DOI `10.1145/3639269` | 用“data layout + block search”两个组件解释 I/O 收益，并给出明确资源约束；启发本文把 GPU kernel、stage wall time 和 full-index cost 分开。 |
| ELPIS | PVLDB 2023, DOI `10.14778/3583140.3583166` | 强调 strong baseline，并同时报告 build time、memory 和 query performance；启发本文不只报告查询加速，还保留构建和资源边界。 |
| LSH-APG, *Towards Efficient Index Construction...* | PVLDB 2023, DOI `10.14778/3594512.3594527` | 先定位 proximity-graph 构建超线性成本，再分别给 entry 与 pruning 优化；启发本文将 GPU batching 的收益归因到具体 construction stage。 |
| tau-MNG, *Efficient ANN Search in Multi-dimensional Databases* | PACMMOD/SIGMOD 2023, DOI `10.1145/3588908` | 先定义结构性质和可证明边界，再给可构建近似版本；启发本文把 authorization safety、level isolation、fallback equivalence 单列。 |
| *Revisiting the Index Construction of PG-based ANNS* | PVLDB 2025, DOI `10.14778/3725688.3725709` | 从构建流程中定位瓶颈，声明加速不能牺牲查询性能；启发本文避免用 kernel speedup 替代 end-to-end builder 结论。 |

## 本轮论文写作调整

- 方法段现在先给一个 invariant，再给 offline/online 总览图，然后分别展开 partition、topology、entry、routing 和 bounded search。
- 新增统一 notation table，直接列出 0L、1L、2L 时物理 level 0/1/2 分别是什么。
- 所有结果表和图例统一使用 `0L-LNG`、`1L-LNG`、`2L-LT` 等形式，`LL/LT/TL/TT` 不再脱离 `2L` 单独出现。
- 新增 measurement-definition table，明确 selectivity、Recall、`Lsearch`、batch latency、QPS、speedup、NC、`Rmax`、阶段时间和工作计数。
- Related Work 明确区分 set-containment hierarchy 与 ordered-range hierarchy，并扩展构建文献，避免只围绕 UNG 一篇论文叙述。

## 建议汇报顺序

1. 先讲 UNG 的单尺度瓶颈：宽谓词下 exact groups 太碎。
2. 再讲一条安全粗化观察：`Q subseteq block root` 保证 direct members 合法。
3. 用总览图讲清楚 0L/1L/2L、LT 顺序、entry 和 routing 的分离。
4. 强调 level-local invariant：同一队列不等于混扫边。
5. 给出 DRH 的三个输入 `N/R/C` 和完全不使用 query calibration 的边界。
6. 最后先报 Amazon 高选择率收益，再主动报告 VariousImg 反例和 screen-level 证据限制。
