# SpecialBlockTrieIndex Amazon 实验分析

## 结论

`SpecialBlockTrieIndex` 已完成构建、持久化、加载和 special-block 搜索接入。
入口仍使用最大 query label 定位 posting，向上验证其余 labels，向下收集每个结构
分支的首个 terminal group，不做全局最小超集过滤。

本轮先在 block-local graph 入口之上增加 query-aware block router，再压缩 free
block 内重复的 group 入口。每个 free block 从直接 `member_group_ids` 中确定性采样
16 个候选 landmarks；普通 child block 保留最近的 1 个，free 子树的前沿 block
保留最近的 8 个。每 block 最多强制预展开最近的 2 个，其余保留点按全局 beam 的
距离顺序正常展开。

同一 free block 内的普通 group 入口点限制为 256 个；与 query labels 精确相等的
terminal group 单独保留最多 16 个点。非 free group 和 block 外 group 不受该策略
影响。最终每 query 的 free-group 入口从 4,413.5 降到 348.8，初始入口距离计算量
从 7,661.8 降到 3,597.1。

free 状态仍沿 Trie block 层级单调传播：只在 regular 状态首次进入 block 时检查
coverage；父 block 一旦 free，所有后继 child block 自动 free。最终入口压缩方案在
四档 Lsearch 的 recall 均不低于 uncapped router，并减少
15.9%/6.7%/4.5%/2.8% 的总距离计算。稳态后两次查询的 QPS 分别提高
12.1%/3.2%/1.1%/1.8%。

## 当前默认搜索逻辑

当前 `special_block_trie` provider 默认采用 terminal-only 语义，不再在命中
terminal 后继续寻找后代 block root：

1. 使用上游已经严格递增且去重的 query labels，并选择最大 query label 作为 pivot。
2. 读取 pivot label 的 Trie posting；对每个 pivot 节点向上验证其路径包含其余
   query labels，路径中允许存在额外 labels。
3. 从每个匹配 pivot 分别向下遍历。某条结构分支遇到第一个 terminal 时，输出其
   `terminal_group_id` 并立即停止该分支。
4. 不跨 Trie 分支做全局最小超集过滤；provider 最后只对相同 group ID 排序去重。
5. 搜索后端把这些 terminal groups 转换成入口点，并按现有 block coverage 判定
   free/regular 状态后执行 graph search。
6. 对 Trie provider，如果 terminal group 属于 query-covered block，则把该 block
   的持久化局部图入口点作为额外 free 入口；regular graph 首次跨入 covered block
   时执行同样的 portal 激活。legacy LNG regular 可用
   `UNG_SPECIAL_TRIE_LEGACY_LNG_REGULAR=1` 保持旧路径，portal 可用
   `UNG_SPECIAL_TRIE_DISABLE_BLOCK_PORTALS=1` 做同拓扑消融；
   `UNG_SPECIAL_TRIE_BLOCK_PORTAL_MIN_LSEARCH` 可设置 portal 的 Lsearch 门槛。
7. block coverage 使用当前 block 的直接 `member_group_ids` 标签集合的精确交集。
   该交集只负责从 regular 状态首次激活 free；一旦父 block 已经 free，Trie 后继
   child block 自动继承 free，跨 block 边不再重复检查 child 的公共标签。状态转移
   等价于 `next_free = current_free || target_block_covers_query`。
8. 对所有 free blocks，从局部图入口和直接 member groups 中采样最多 16 个
   query-independent landmarks，并计算 query 距离。
9. `free-frontier block` 定义为自身 free、但父 block 不 free 或不存在父 block 的
   block。普通 free child block 保留最近的 1 个 landmark，frontier block 保留最近
   的 8 个。
10. group 入口循环保留所有 regular/non-block group；query 精确 terminal group
    最多保留 16 个点；其余 free group 在同一 block 内合计最多保留 256 个点。配额
    只按实际成功加入的点计数。
11. 每 block 最多强制预展开最近的 2 个 retained landmarks。其邻居进入普通
    distance-ordered beam；未预展开的 retained landmarks 也在 beam 中按距离竞争，
    后续继续使用原有 block special edges 搜索。

Amazon 主配置对应的参数为：

```text
UNG_SPECIAL_TRIE_SEED_FREE_BLOCKS=1
UNG_SPECIAL_TRIE_BLOCK_SEEDS_PER_BLOCK=16
UNG_SPECIAL_TRIE_BLOCK_SEEDS_RETAIN_PER_BLOCK=1
UNG_SPECIAL_TRIE_FRONTIER_BLOCK_SEEDS_RETAIN_PER_BLOCK=8
UNG_SPECIAL_TRIE_PREEXPAND_BLOCK_SEEDS=1
UNG_SPECIAL_TRIE_BLOCK_SEEDS_PREEXPAND_PER_BLOCK=2
UNG_SPECIAL_TRIE_FREE_GROUP_ENTRY_CAP=256
UNG_SPECIAL_TRIE_EXACT_GROUP_ENTRY_CAP=16
```

`UNG_SPECIAL_TRIE_FREE_GROUP_ENTRY_CAP` 未设置时保持 uncapped 兼容行为；显式设置
为 `0` 才表示删除普通 free-group 入口。frontier retain 和 preexpand cap 未设置时
也保持旧 Router 行为。

因此，terminal-only 仍不会把后代 groups 加回 `entry_group_ids`；后代覆盖由独立的
block router 完成。查询明细 CSV 中保留的 `SpecialInterEdgesCoverageRejected` 仅用于
兼容旧实验格式，新逻辑下应恒为 0。

因此，如果 query 为 `{1}` 且节点 `{1}` 本身就是 terminal，默认只返回该节点的
terminal group，不再访问 `{1, ...}` 后代。若 `{1}` 不是 terminal，则继续沿它的
各个子分支寻找每条分支的首个 terminal。这里描述的是 entry-group provider；后续
block router 仍会为 free descendant blocks 选择 metric landmarks。

实验性的后代 block frontier 现在必须显式设置
`UNG_SPECIAL_TRIE_BLOCK_FRONTIER=1` 才会启用；
`UNG_SPECIAL_TRIE_TERMINAL_ONLY=1` 可强制关闭它。Amazon 实验中 block frontier
没有改善 recall，因此不再作为默认行为。

## 实验环境

- Dataset: Amazon，602,453 vectors，510,639 groups，dimension 768。
- Query task: `query_selected_recall_advantage`，1,971 queries，225 个唯一 label set。
- `K=10`，100 search threads，`num_entry_points=16`。
- Lsearch: 1,000 到 20,000，步长 1,000。
- 所有方法使用同一个新构建的 index：`UNG_special_block_trie_regular`。
- 搜索后端、special edges、block 参数和 ground truth 完全相同。

实验配置：

- `experiments/cpu_special_blocks/config_special_block_trie.json`
- `experiments/cpu_special_blocks/config_special_block_trie_regular.json`
- `experiments/search_comparison/config_amazon_special_block_trie_regular_ab.json`
- `experiments/search_comparison/config_amazon_special_block_router_final.json`
- `experiments/search_comparison/config_amazon_special_block_group_router_final.json`

## 索引构建

| 指标 | 结果 |
|---|---:|
| 全量构建 wall time | 186 s |
| UNG index time（含 Roaring） | 159.11 s |
| 保存时间 | 23.43 s |
| Special Trie 构建时间 | 1.113 s |
| Trie regular overlay 构建时间 | 8.022 s |
| Special Trie 保存时间 | 0.136 s |
| Special Trie 加载时间 | 0.178-0.279 s |
| Trie nodes | 2,108,923 |
| Child references | 2,108,922 |
| `special_block_trie.bin` | 59,049,888 bytes（56.31 MiB） |
| Blocks | 170 |
| Block member groups | 492,152 |
| Block member points | 581,472 |
| Child block edges | 156 |
| Block local entry points | 170/170 已持久化 |
| Direct-member `common_labels == root_labels` | 170/170 |

block 数、成员数、成员点数和 child block 数与旧临时 Trie 实现一致，说明将临时
构建树提升为正式索引没有改变阈值切分结果。

## 查询性能

下表是 3 次重复的平均结果；所有方法使用同一索引和同一 1,971 条 query。

| Lsearch | Terminal baseline Recall / QPS | 16-to-1 Router Recall / QPS | CPU ELS Recall / QPS |
|---:|---:|---:|---:|
| 1,000 | 0.7166 / 2,220.7 | 0.9076 / 1,994.9 | 0.8992 / 2,057.6 |
| 5,000 | 0.8470 / 983.2 | 0.9782 / 896.4 | 0.9651 / 891.9 |
| 10,000 | 0.8907 / 647.7 | 0.9898 / 619.5 | 0.9798 / 609.8 |
| 20,000 | 0.9199 / 417.4 | 0.9936 / 407.6 | 0.9882 / 403.1 |

Router 相对 terminal baseline 的 recall 增益为
`+0.1910/+0.1312/+0.0991/+0.0737`。相对 CPU ELS 的 recall 增益为
`+0.0084/+0.0130/+0.0100/+0.0054`。L=1,000 QPS 比 CPU ELS 低约 3.0%；
L>=5,000 则高约 0.5%-1.6%。

最主要的失败查询已经被修复。L=20,000 时，query `{2}` 从 0.5688 提升到
0.9936，query `{1,2}` 从 0.6607 提升到 1.0000；对应的 95/94 个 free blocks
均至少预展开一次。

### Free-group 入口压缩

下表比较相同 Router16 的 uncapped 路径与最终入口压缩路径，均为 3 次重复平均。
QPS 汇总包含每个进程的首次冷运行。

| Lsearch | Uncapped Recall / QPS | 压缩后 Recall / QPS | Recall 变化 | QPS 变化 |
|---:|---:|---:|---:|---:|
| 1,000 | 0.9076 / 1,725.3 | 0.9099 / 2,141.4 | +0.0023 | +24.1% |
| 5,000 | 0.9782 / 890.8 | 0.9785 / 904.6 | +0.0003 | +1.6% |
| 10,000 | 0.9898 / 611.2 | 0.9911 / 609.0 | +0.0013 | -0.3% |
| 20,000 | 0.9936 / 402.7 | 0.9940 / 406.2 | +0.0004 | +0.9% |

L=1,000 的首次 uncapped repeat 为 1,507.7 ms，明显慢于后两次的约 960 ms，
因此三次平均夸大了该档收益。只比较 repeat 2/3 的稳态 QPS，uncapped 到压缩后为：

| Lsearch | Uncapped 稳态 QPS | 压缩后稳态 QPS | 变化 |
|---:|---:|---:|---:|
| 1,000 | 2,053.7 | 2,302.4 | +12.1% |
| 5,000 | 891.2 | 919.6 | +3.2% |
| 10,000 | 610.5 | 617.1 | +1.1% |
| 20,000 | 402.7 | 409.9 | +1.8% |

距离计算减少是确定性的，不受运行时噪声影响：

| Lsearch | Uncapped DistCalcs/query | 压缩后 DistCalcs/query | 减少 |
|---:|---:|---:|---:|
| 1,000 | 23,215.7 | 19,515.8 | 15.9% |
| 5,000 | 46,651.5 | 43,537.2 | 6.7% |
| 10,000 | 62,250.6 | 59,470.7 | 4.5% |
| 20,000 | 80,799.1 | 78,530.2 | 2.8% |

## ELS 与入口统计

预热前，Trie 会重复遍历较大的结构分支，平均 ELS 为 2-3.4 ms/query，p95 为
8-18 ms。加入与 CPU ELS 同等的单次计算 cache 后，两者稳态 ELS 都约为
40-50 us/query。Trie 的 225 个唯一 label set 预热耗时为 2.71 s，CPU ELS 为
1.50 s。

Terminal baseline 的入口统计与 Lsearch 无关：

| 指标（每 query 平均） | CPU ELS | Special Trie | 变化 |
|---|---:|---:|---:|
| Entry groups | 5,945.9 | 6,328.2 | +6.4% |
| Free entry points | 6,898.7 | 4,430.7 | -35.8% |
| Regular entry points | 637.4 | 2,456.4 | +285.4% |

Special Trie 的结构遍历分布：

| 指标 | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| Pivot postings | 1,037.6 | 2 | 7,358 | 11,214 |
| Matching pivots | 997.0 | 2 | 7,249 | 11,214 |
| Upward nodes | 2,794.4 | 0 | 13,341 | 118,557 |
| Downward nodes | 40,120.3 | 19,279 | 171,201 | 200,133 |
| Terminal entries | 6,328.2 | 5,264 | 15,512 | 18,686 |

Router 不改变 6,328.2 个平均 terminal entry groups。最终方案每 query 仍打分
791.9 个 block landmarks；平均 free-frontier block 数为 1.84，最终保留 62.4 个
landmarks，强制预展开 51.3 个。普通 free child block 仍只预展开 1 个，frontier
block 最多预展开 2 个。

入口压缩的平均统计为：free-group points `4,413.5 -> 348.8`，regular-group
points 保持 2,456.4，策略跳过 4,078.1 个点；所有初始入口点
`7,661.8 -> 3,597.1`，减少 53.1%。

## Block 搜索路径统计

`SpecialBlocksSearched` 统计实际被 free 候选展开过的唯一 block。Router 保证每个
free block 至少预展开一个 landmark，因此该统计不再随 Lsearch 波动：

| Lsearch | 使用 block 搜索的查询 | Mean | Median | P95 | Max |
|---:|---:|---:|---:|---:|---:|
| 全部档位 | 1,769 / 1,971（89.8%） | 49.49 | 50 | 97 | 97 |

regular/free 边按所有 query 的 scanned-edge 总量加权。这里 regular 指 Trie
terminal groups 之间的 regular graph 边，free 指 free block graph 的 special
edges：

| Lsearch | 平均 regular edges/query | 平均 free edges/query | Regular 占比 | Free 占比 |
|---:|---:|---:|---:|---:|
| 1,000 | 855.6 | 50,138.4 | 1.68% | 98.32% |
| 5,000 | 3,583.1 | 239,247.0 | 1.48% | 98.52% |
| 10,000 | 6,485.8 | 479,193.4 | 1.34% | 98.66% |
| 20,000 | 9,598.5 | 956,988.1 | 0.99% | 99.01% |

四档 Lsearch 的 `SpecialInterEdgesCoverageRejected` 总和均为 0。exact-common 与
root-prefix 两个模式除计时字段外逐 query 完全相同，说明新状态传播没有隐藏的
child coverage 分叉。

## 原因分析

结构首个 terminal 与整个后代数据区域不是同一个概念。terminal 只代表真实存在的
group，并不天然是所有 descendant blocks 的聚合导航入口。

`cpu_bruteforce_els` 的入口倾向于落在能够通过 LNG descendants 表达大范围
coverage 的 group。Special Trie 则严格按前缀树结构停止在首个 terminal。即使
terminal group 自身满足 query，它所属 block 的直接成员公共标签也可能不包含完整
query，于是该入口只能作为 regular 状态启动。Amazon 当前数据上公共标签与 block
root path 相同，但搜索逻辑不依赖这个数据集特例。

旧路径依赖一个 terminal/group 入口和有限的 parent-child 图导航。query `{1,2}`
只有一个 terminal entry group，却对应 272,737 个点和 94 个 free blocks；全局 beam
无法保证这些 block 都获得搜索机会。原始 16-to-1 router 用多个 landmarks 估计
每个 block 的 query 距离，但只实际展开最佳一个，解决了覆盖问题且没有显著增加
扫边。

简单按遍历顺序设置 free-group cap=0/8/16/32 会把 L=1,000 recall 降到
0.8896-0.8922，因为这些入口同时承担了小 beam 下的多起点覆盖。对所有 49.49 个
free blocks 都扩大 retained landmarks 又会增加过多 special-edge 扫描。最终方案
只对平均 1.84 个 free-frontier blocks 扩大到 8 个 query-nearest landmarks，并把
每 block 强制预展开限制为 2；再保留 256 个 group 点补足局部多样性，从而同时保住
recall 和降低距离计算。

## 后续建议

保留当前 Trie terminal-only 入口语义，不重新引入旧 ELS，也不增加全局最小超集
过滤。当前入口压缩已经把初始距离计算减半；L>=5,000 时 98.5%-99.0% 的扫描仍是
free special edges，下一步应优化 block 图内部遍历和 special-edge 访存，而不是继续
压低入口 cap。还可以把 16 个 landmarks 在构建期直接持久化，并复用每 query 的
block routing scratch，减少运行时 member-group 采样和临时向量分配。

回归测试确保：

1. terminal 自身数据仍由 regular 搜索负责；
2. query-covered block 可通过真实局部入口点激活为 free；
3. 父 block free 后，child block 无条件继承 free；
4. 普通 free child block 保留 1 个、frontier block 保留 8 个 query-nearest
   landmarks，每 block 最多强制预展开 2 个；
5. 精确 query terminal、regular group 和 block 外 group 不被 free-group cap 误删；
6. recall 提升不依赖 LNG descendants 或旧 provider。

Amazon 主 A/B 配置已将压缩后的 frontier-aware Router16 作为
`special_trie_regular`，旧路径保留为 `special_trie_terminal_baseline`。
