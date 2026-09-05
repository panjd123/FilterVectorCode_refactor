# Handoff：复现当前 Special Block，并准备多层实现

更新时间：2026-09-05

远端仓库：`/home/graphdb/FilterVectorCode_refactor`

当前观察到的分支 / HEAD：`shopai8/special-block-e2e-opt` / `dda63bd`

## 1. 接手目标

下一位 agent 不要直接开始改多层结构。正确顺序是：

1. 理解远端工作树中的最新 Special Block 构造、持久化、入口组提取和查询路径；
2. 在当前机器上复现已有测试和至少一条端到端 recall/QPS 曲线，确认代码、索引、配置和结果口径一致；
3. 基于复现结果写出一个简洁的多层 Special Block 设计，再开始实现。

目标不是继续旧的 7 月 sidecar 原型，而是在当前 `SpecialBlockTrie + 独立 block sidecar + free-state search` 代码上增加真正的多层语义。

## 2. 开始前必须知道的状态

- 远端工作树很脏，包含大量已修改和 untracked 的核心代码。不要 reset、checkout 或覆盖这些文件。
- 当前 HEAD 不能完整代表运行中的实现；很多最新 Special Block 文件仍是 untracked。接手时应以工作树源码和现有实验产物为事实来源。
- 当前机器的有效数据/结果根目录是 `/home/graphdb/FilterVectorData` 和 `/home/graphdb/FilterVectorResult`。部分 README、JSON 和历史日志仍写 `/home/dev/graphdb/...`，不能原样运行。
- 历史结果已迁移到 `Amazon_hybrid/results/test-20260813/`。先核对现有产物，再建立新的、路径正确的复现配置。
- 连接偶发不稳定时短间隔重试，当前可用连接为：

```bash
ssh -J W300-pub sunyahui@10.77.110.170
```

## 3. 当前实现究竟是什么

### 3.1 构造与存储

当前主线是 trie-partitioned Special Block：

1. 用所有 group label sets 构建并持久化 `SpecialBlockTrieIndex`；
2. 自底向上统计每个 trie 子树的 `subtree_points` 和尚未被子 block 吸收的 `uncovered_points`；
3. 当 `uncovered_points > UNG_SPECIAL_BLOCK_MIN_POINTS` 时建立 block，并把该节点的 uncovered count 清零；
4. 收集 block 的直接 member groups；遇到已有子 block 时停止下钻，把它记录进 `child_block_ids`；
5. 为 block 构建 intra special edges，为父子 block 构建 inter special edges，并构建 trie regular/portal edges；
6. block sidecar 独立保存 `special_blocks.bin`、`special_block_trie.bin`、`special_edges.bin` 和 `special_trie_regular_edges.bin`。

这意味着当前构造已经产生 block 父子关系，并非完全扁平分区。但运行时主要仍把每个 group/point 映射为单个 `block_id`，候选状态主要是一个 `free` 布尔量。多层任务首先要判断：已有嵌套结构和目标中的“多层”还差哪些显式语义。

### 3.2 查询与入口组

- `special_block_trie` provider 查询持久化 trie，但对外输出的是 **group IDs**，不是 trie-node 对象。
- 当前 ELS 契约已经统一为可被 `search_UNG_index` 直接消费的 group-id entry groups；不要恢复 trie-node 粒度输出。
- provider 通过 first-terminal frontier、group caps、block seeds、frontier seeds 和 block portals 生成入口。
- 查询热路径使用 free-state：普通点受普通边约束；free point 可以访问 special edges，并可通过 covered block/portal 激活额外入口。
- 当前状态和映射包括 `_point_to_special_block`、`_group_id_to_special_block`、`child_block_ids` 和候选的 `bool free`。这些是多层化最可能需要重新审视的接口。

### 3.3 最新代码地图

| 主题 | 主要文件 | 先理解什么 |
|---|---|---|
| block 数据结构 | `UNG/codes/include/ung_special_blocks.h` | `SpecialBlock`、direct members、`child_block_ids`、entry point、special edge owner |
| partition / 构图 / 保存加载 | `UNG/codes/src/uni_nav_graph_special_blocks.cpp` | bottom-up partition、intra/inter overlay、regular portal、single-block maps |
| 持久化 trie | `UNG/codes/include/ung_special_block_trie.h`、`UNG/codes/src/ung_special_block_trie.cpp` | flat trie、block ID、first-terminal group-id provider、serialization |
| 独立 sidecar builder | `UNG/codes/include/special_block_index_builder.h`、`UNG/codes/src/special_block_index_builder.cpp` | 纯 UNG 与 block sidecar 的边界 |
| entry provider | `UNG/codes/src/uni_nav_graph_entry_provider.cpp` | provider 输出、warm-up/cache、统计 |
| search 热路径 | `UNG/codes/src/uni_nav_graph_search_backend.cpp` | free-state、block seeds/portal、special edge 扫描、candidate queue |
| 激活语义 | `UNG/codes/include/ung_special_block_activation.h` | covered block 和 child-block closure |
| candidate queue | `UNG/codes/include/ung_special_candidate_queue.h` | ordered/heap 实验边界；heap 不是当前默认最佳 |
| 构建入口 | `UNG/codes/apps/build_special_block_index.cpp` | 独立 block sidecar 构建 CLI |
| 搜索入口 | `UNG/codes/apps/search_UNG_index.cpp` | provider、sidecar path、Lsearch 和 query 流程 |

优先阅读的设计与结果文档：

1. `docs/superpowers/specs/2026-08-10-special-block-trie-entry-provider-design.md`
2. `docs/superpowers/plans/2026-08-10-special-block-trie-entry-provider.md`
3. `docs/reports/SPECIAL_BLOCK_TRIE_AMAZON_BASELINE_COMPARISON_CN.md`
4. `docs/reports/SPECIAL_BLOCK_UNG_SEARCH_OPTIMIZATION_CN.md`
5. `experiments/search_comparison/README_SPECIAL_BLOCK_TRIE_CAP16_CN.md`

早期背景才看 `SPECIAL_BLOCK_GRAPH_QUERY_DESIGN_CN.md` 和 `SPECIAL_BLOCK_BUILD_QUERY_AB_CN.md`；不要让旧实现描述覆盖当前代码事实。

## 4. 已有结果：先复现什么

现有结果有两个不同目的的代表配置，不能拼成一个“最佳结果”：

| 结果族 | 配置重点 | 已记录结果 | 用途 |
|---|---|---|---|
| Router 高 recall 曲线 | router16、free-group cap256、frontier8、preexpand2 | L=1000/5000/10000/20000：`0.909944/2141.36`、`0.978488/904.63`、`0.991071/609.03`、`0.993962/406.24`（Recall/QPS） | 验证高 recall 的当前路由能力 |
| cap16 热路径曲线 | ordered queue、free intra-edge scan cap=16 | L=2000：`0.943176/2200.60`；L=2500：`0.952714/2100.59` | 验证约 0.95 recall 下的热路径吞吐 |

真实 CSV 位于：

```text
/home/graphdb/FilterVectorResult/Amazon_hybrid/results/test-20260813/
  special_trie_router16_group_cap256_frontier8_preexpand2_repeat3/
  special_trie_ordered_intra_cap16_reverse_repeat3/
```

当前可用的代表索引是：

```text
/home/graphdb/FilterVectorResult/Amazon_hybrid/index/Trie_block_hybrid/
├── index_files/
└── block_index_files/
```

其 block metadata 显示：602,453 points、510,639 groups、170 blocks、156 child-block edges、`min_points=1000`、约 2.109M trie nodes。

## 5. 推荐复现顺序

### 5.1 先做源码和 focused-test 基线

```bash
cd /home/graphdb/FilterVectorCode_refactor
git rev-parse --abbrev-ref HEAD
git rev-parse --short HEAD
git status --short

cmake --build build_ung_rel -j16 --target \
  test_special_block_trie \
  test_special_block_free_state \
  test_special_candidate_queue \
  build_special_block_index \
  search_UNG_index

(cd build_ung_rel && ctest -R 'special_block_trie|special_block_free_state|special_candidate_queue' --output-on-failure)
```

如果编译失败，先区分是当前脏工作树本身的问题、旧 build cache，还是代码回归；不要为了“通过”而清理用户改动。

### 5.2 验证已有结果文件

先直接读取上述两条 CSV 和对应 `query_details_repeat3.csv`，确认文档数字、查询数、Lsearch、repeat 和统计字段仍一致。这一步只验证历史证据可读，不算新复跑。

### 5.3 做一次当前代码搜索复跑

以 `experiments/search_comparison/test-20260813/` 下的配置为模板，新建一个不会覆盖历史结果的配置：

- 将所有 `/home/dev/graphdb` 改为当前 `/home/graphdb`；
- 将 dataset/result layout 对齐现有 `Amazon_hybrid`；
- 将 `index_name` 对齐实际可用索引，或明确复制/迁移关系；
- 使用新的 method/result 名称；
- 先跑一个 Lsearch smoke，再跑 3-repeat 代表曲线；
- 保存最终展开后的命令、环境变量、stdout、summary 和 query details。

不要直接相信 `README_SPECIAL_BLOCK_TRIE_CAP16_CN.md` 中的旧绝对路径；其中提到的 `config_amazon_special_block_cap16_best.json` 当前并不存在。

### 5.4 最后才做从零构建复现

从零构建会消耗较多时间和存储。先把配置中的路径和 index name 校准，使用全新的输出目录，不移动或覆盖已有 `Trie_block_hybrid`。构建复现至少记录：

- UNG build 与 block sidecar build 分项时间；
- block/trie/intra/inter/regular-portal 分项时间；
- block 数、父子边数、direct member points、special edge 数；
- sidecar 磁盘大小和加载内存；
- 同一索引上的 recall/QPS 曲线。

## 6. 多层 Special Block：实现前必须回答的问题

当前源码已经允许父 block 引用 child block，但还没有完整、显式的多层查询模型。接手 agent 应先从源码和复现结果回答：

1. “层”按 trie 深度、block 祖先关系、point-count threshold，还是查询时的粒度定义？
2. 一个 group/point 是否仍只属于一个 direct block，还是同时属于多级祖先 block？
3. 每层构建什么图：direct-member intra graph、父子 block portal、同层 inter-block graph，分别由谁拥有？
4. special edge 的 `special_block_id` 是否足够，还是需要 owner level / source scope；如何避免多层重复边？
5. `bool free` 是否仍能表达权限；如果不能，是维护最高已激活层、active block chain，还是其他紧凑状态？
6. query 完整覆盖父 block、只覆盖 child block、只覆盖普通 group 时，各自允许走哪些边？必须保持 containment 正确。
7. `special_block_trie` 仍应输出 group IDs；多层 routing/portal 应在后续查询阶段解释这些入口，不能把公共 provider 接口改回 trie nodes。
8. 保存格式如何升级并兼容 v1 sidecar；旧索引加载时是降级成单层还是明确拒绝？
9. 质量比较必须使用端到端 filtered-search Recall/QPS；block 数、入口数、扫边数只能解释性能，不能代替质量。

建议先写一个短设计文档和最小语义测试，再改生产路径。设计应允许 agent 根据 profiling 选择实现，不必预先锁死为固定层数或固定数据结构。

## 7. 多层实现应保留的基线和消融

至少保留：

- 当前单层/现状语义，同一索引参数和同一 search 参数；
- 多层 metadata only（验证分区和存储，不启用多层搜索）；
- 多层 portal/routing 开启；
- 必要时分别关闭祖先 portal、子 block portal或某层 special edges，定位收益来源。

报告时同时给出构建成本和查询结果：

| 类别 | 必要指标 |
|---|---|
| 构造 | 总时间及 partition/intra/inter/portal/save 分项 |
| 结构 | 每层 block 数、direct/recursive points、父子边、special edges、磁盘/内存 |
| 查询 | Recall、QPS/latency、Lsearch、distance calculations、free/regular expansions、special edges scanned |
| 公平性 | 同数据、同 queries、同线程、同 K；按相同 recall 比性能 |

## 8. 已知边界与不要重复的方向

- 全局把 `MIN_POINTS` 从 1000 提到 2000/4000，在固定 L 下会提速，但为恢复 Recall 需要更高 Lsearch，`Recall >= 0.95` 时反而更慢；不要把“多层”简化成全局大 block。
- candidate heap/lazy 路线已有负结果或边界，当前 cap16 最佳点使用 ordered queue；重新启用前先读优化报告。
- 当前 L>=5000 时约 98.5%-99.0% 的扫边来自 free special edges。多层设计若不能减少这些扫描或提高其导航价值，很可能只增加元数据和分支开销。
- trie-node cover 不是当前 ELS 输出目标；入口接口是 group IDs。
- 局部 topK overlap、block coverage 或入口数都不是最终质量标准，必须跑端到端 recall。

## 9. 接手后的合理停止点

完成以下内容后，再进入多层编码比较稳妥：

- 最新代码路径图与现有层级语义已经确认；
- focused tests 通过，或失败原因有清楚记录；
- 至少一条当前代码端到端曲线完成复跑，并与历史结果解释了差异；
- 多层设计明确了层级、membership、edge ownership、query activation 和兼容策略；
- 建立了不会覆盖现有结果的实验命名和 baseline。

## 10. 旧项目总览入口

如果还需要 broader project 背景，再读：

- `docs/reports/WORK_PRESENTATION_DEEPRESEARCH_CN.md`
- `docs/reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md`
- `docs/reports/REVIEW_READY_REQUIREMENTS_MATRIX_CN.md`
- `AGENT_KANBAN.md`

这份 handoff 以最新 Special Block 接手任务为主；旧的 cross-edge、PG、additional/output 总览不是本轮第一优先级。
