# ML-UNG 论证与阅读设计

## 中心问题

标签过滤搜索既要覆盖合法标签分支，又要在这些分支内进行有效的向量导航。改变连接关系会改变所需入口；将入口交给粗粒度 block 又会改变仍可使用的基层路径。因此入口、block 授权和搜索状态必须共同设计。

Prefix frontier 为每个合法前缀子树保留入口。Prefix certificate 允许 block 内的向量边跨越 exact-group 边界。Level-local 状态明确每次扩展使用哪个图。论文用覆盖分析和固定配置实验解释这三个选择之间的关系。

## 读者应当形成的推理链

1. Exact-label groups 提供过滤语义，却可能产生昂贵的入口归约和细碎导航。
2. Prefix forest 减少标签连接关系，因此必须保留不同前缀分支的入口。
3. 若 block 根前缀包含查询标签，其 direct members 都合法，可以在其中按几何邻近关系导航。
4. 转交入口给 block 会替换其 L0 状态；LNG 最小入口依赖的非前缀路径可能永久丢失。
5. 完整 prefix frontier 在明确的入口保留与有向可达性条件下可以保留覆盖。有限队列随后决定实际参加搜索的状态。
6. 固定配置在不同批次有收益和退化；搜索预算、入口设置和图工作共同影响吞吐。批次Recall门槛掩盖的子群差异决定这些吞吐结果如何解读。
7. 自动参数和 GPU 构建是支持性研究，分别需要迁移性能与输出质量证据。

## 正文各部分的职责

| 部分 | 核心问题 | 对应证据或图示 |
|---|---|---|
| Introduction | 为什么 entry 与 navigation 必须共同设计？ | 三项贡献；固定开发配置的收益和反例 |
| Problem and Motivation | AND 标签语义、选择率、两类组织成本是什么？ | 应用示例与两档阶段 profile |
| Prefix-Frontier Entry Discovery | 为什么 prefix frontier 覆盖全部合法 terminal？实际怎样计算？ | Proposition 1；bitmap/ancestor 算法和成本 |
| Predicate-Certified Block Navigation | 独立分区生成什么？阈值控制什么？L0/L1/L2 如何命名？ | LNG/prefix/block 三联图；分区算法；block 数上界 |
| Querying Certified Blocks | seed 如何标层？安全与覆盖各由什么条件保证？ | 搜索算法；retagging 反例图；Propositions 2–4 |
| Construction and Hierarchy Configuration | 怎么构建不规则图？DRH 中哪些量有结构依据？ | 构建数据流；复杂度；N/T 上界与启发式选择 |
| Experimental Methodology | 实际数据、批次、执行与测量对象是什么？ | Amazon 和 held-out 描述；重复与 Recall 规则 |
| Results | 固定配置怎样表现？哪些查询仍失败？入口节省去了哪里？ | 固定配置主表；子群Recall；profile；自动方案反例；构建计时 |
| Discussion and Related Work | 与直接相关工作的结构区别是什么？ | UNG、Curator、ACORN、FAVOR、LSSG 等 |
| Conclusion | 从现有分析和测量可以形成什么设计认识？ | 入口覆盖、几何导航和预算共同决定效果 |

附录包含分区单调性证明、交叉分区例子、有限队列的两层执行示例、全网格赢家、126 格 QPS/预算图、详细计量与构建 profiles、Recall 曲线。

## 三项贡献的对应关系

| 贡献 | 方法内容 | 目前的证据 |
|---|---|---|
| 配套的入口与标签连接 | terminal-prefix forest、bitmap-backed frontier | 标签平面覆盖证明；实际入口与后续成本 profile |
| 谓词认证的 block 导航 | 独立 direct-member 分区、root authorization、level-local 搜索 | 谓词安全；retagging 反例；完整 prefix 入口的条件覆盖；分区结构性质 |
| 入口与导航的性能交换分析 | 固定配置、受控 upper 替换、完整 factorial | Amazon 开发集九个批次；高搜索预算与 NC；支持性的 policy/build 研究 |

## 数据展示顺序

主表先给 Plain、Trie base、固定一层和固定两层，分别沿九个批次保持结构不变。逐档赢家用于了解配置空间，放在完整分析中。跨 L0 比较注明各自 provider；同 L0/provider 下改变一个 upper 字母才隔离该层 topology。

高平均选择率的数百倍比率与 260k–400k 对 1k–5k 的搜索预算差异一起解释。发现、seed、graph 的 instrumented ms/query 与主 QPS 分开定义。实验方法用图区分标签数与精确选择率；Results紧接吞吐表解释5%与30%批次的子群Recall，附录列出完整标签数分组和两档真实选择率分组。这样将批次平均门槛的含义落实到具体证据。

## 后续研究需要补齐的证据

| 问题 | 所需证据 |
|---|---|
| 固定方案的竞争力 | 同协议外部强基线、原版 UNG 对照及高效 exact scan |
| 核心机制的普适性 | 跨数据集的 prefix/blocks 组合；标签顺序和查询分层 |
| NC 与大预算的来源 | 逐查询可达性、保持覆盖的入口控制及队列成本 |
| 空间换吞吐是否划算 | 加载索引、每层边数、bitmap/ancestor caches、并发 workspace |
| 自动方案与加速构建是否可用 | 真正执行的一层候选、自动深度变化、构建输出的查询质量 |

## 写作纪律

每段承担定义、推导、证据或解释之一。正式条件放在证明处，测量口径放在实验方法处；结果段解释这些条件如何影响观察。导师 note 用同一示例和少量固定配置表，完整文献出版记录放入备查材料。作者的评分、修订轮数与执行日志留在 review 目录。
