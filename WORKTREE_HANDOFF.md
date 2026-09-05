# Handoff: Multi-level Special Block

## 目标与结论

本分支在隔离 checkout 中实现并验证了真正的两层 Special Block overlay：保留阈值 `T1=1000` 的中层 block，再叠加可配置的 `T2` 上层 block。已在 Amazon 原始 100% x1 上构建 `T2=4k/10k/25k/50k`。查询候选的激活状态由布尔值改为有序层级：

```text
0: 普通图  ->  1: 中层 Special Block  ->  2: 上层 Special Block
```

候选只能沿已激活层级允许的边扩展，普通候选不能直接跳到上层。Amazon x1 的 dense-L、7-repeat 正式矩阵已经完成：多层的主要价值是改善导航质量，以更小 `Lsearch` 达到单层相同 Recall；共同 Recall 门槛下 25%/50%/75% 三档相对单层分别为 `1.222x/1.530x/2.345x`，单层最高质量附近为 `1.222x/4.925x/6.290x`。本文后面保留的早期 1.124x/1.331x 数字来自旧 hybrid 主图，只能作为机制历史，禁止作为最终 Recall/QPS 结果。

当前 ELS 输出契约仍是 `search_UNG_index` 可直接消费的 group IDs，不是 trie-node cover。质量只以端到端 filtered-search Recall 判定；block 数、入口数和边数仅用于诊断。

## 隔离环境与 Git 状态

- 原始 checkout：`/home/graphdb/FilterVectorCode_refactor`
- 原始分支：`shopai8/special-block-e2e-opt`
- 创建隔离目录时及 2026-09-05 收尾时原始 HEAD：`dda63bd7663b06dce0ce3a977b81268df09b2d01`
- 隔离 checkout：`/home/sunyahui/worktrees/FilterVectorCode_multilevel_special`
- 实现分支：`codex/multilevel-special-block-20260905`
- 原始脏工作树快照：`781184607b921b63e71d1ff1d0550aef46b0c4f7`
- C++17 兼容修复：`ab984da`
- 初版 handoff：`cea584c`
- 多层实现：`1f0c1e6`
- 候选去重与 loader 修复：`368227e`
- 外部 baseline runners：`fb9baa2`、`a8ce77e`、`8e21b83`
- ACORN gamma/正式重复 runner：`f2360f1`、`ca8fd41`
- 论文级结果与方法报告：`5cff515`
- 官方 ACORN x1 可复现 patch：`4169be6`

服务器 Git 1.8.3.1 不支持 `git worktree`，因此这里使用 `git clone --shared` 创建等价隔离目录：branch/index 独立，但共享对象库。仓库没有配置 submodule。

隔离分支的 tracked 文件在最终验证时为 clean；`runs/`、`thirdparty/acorn-official/`、`thirdparty/curator-v2/` 为未跟踪实验产物/nested clones，不应提交。`UNG/codes/third_party/CRoaring/build/src/libroaring.a` 是 ignored、可丢弃的构建缓存，也不应提交。

原始 checkout 的 HEAD 没有前进，但仍有大量用户的 tracked/untracked 改动。检查显示其文件内容与隔离分支的快照提交 `7811846` 一致。不要在原始 checkout 直接执行 merge、reset、checkout 或覆盖文件。安全回合并方式是：先由用户把原始脏状态形成自己的 checkpoint commit，再仅 cherry-pick `ab984da` 之后的实现提交，并处理 handoff 文档是否需要进入主分支。

## 实现语义

### 1. 两层独立分区

构建器对同一棵 group trie 运行两次独立的 bottom-up uncovered partition：

- 中层：`UNG_SPECIAL_BLOCK_MIN_POINTS=1000`；
- 上层：`UNG_SPECIAL_BLOCK_UPPER_MIN_POINTS=10000`，启用时必须大于中层阈值。

独立分区保证开启上层后，中层 block 的 ID 和成员不变。同一 group/point 可同时属于一个中层和一个上层 block，因此运行时维护两套 ownership map，而不是用单一 owner 覆盖。中层 `parent_block_id` 指向最近的上层容器；原来的 `child_block_ids` 继续表示层内 block hierarchy。部分中层 block 没有上层容器是允许的。

### 2. 图与边

普通图、中层 special graph、上层 special graph 以 overlay 形式共存。每条 special edge 带所属 block；block 的 `level` 决定这条边需要的 activation level。两层 block 可共享同一个 point，这些 point 上的不同层 special edges 构成中层到上层的真实可达路径。

持久化格式为 `special_block_trie_multilevel_v1`，并保留对旧 `special_block_trie_v1/v2` 的读取兼容。多层 metadata 记录 `level`、`parent_block_id`、上下层阈值和计数。

### 3. 查询状态机

候选状态为 `activation_level in {0,1,2}`：

- level 0 只能走普通边；
- level 1 可走普通边和中层 special edge；
- level 2 可走普通边、中层和上层 special edge；
- 状态只能单调提升，不能降级；
- 完整覆盖判断在多层模式使用 trie root prefix，避免 `common_labels` 对仅部分覆盖的子树错误授权；
- 当前 GPU free-distance batch scratch 不携带 activation level，多层模式暂时禁用该 batch path，防止 level 2 被错误降级。

同一 point 在候选队列中必须唯一。若更高层路径到达已存在 point，则原位升级其 activation level；若该候选此前已扩展，则重新放回 expansion heap，但不额外占用 `Lsearch` 或最终 top-K 槽。

## 两个关键正确性修复

1. 初版索引可构建但查询 loader 拒绝 `special_block_trie_multilevel_v1`。原因是公开 loader 的格式白名单遗漏新格式；现已抽为 `is_supported_special_block_index_format()` 并有单测覆盖。
2. 初版查询把同一个 point 的 level 0/1/2 当成三个候选，重复消耗 `Lsearch` 和最终 top-K 槽，导致 L=1000 Recall 仅 0.635718，增至 L=20000 仍约 0.625。修复为按 point 去重、原位升级后，L=1000 Recall 恢复到 0.938153，L=20000 达到 0.993151。

这两个问题说明不能用结构计数代替端到端 Recall 验证。

## 当前正式数据与实验口径

- 数据集：Amazon，原始 100% x1 label/group 口径；
- points：602,453；groups：482,387；labels：30,723；维度：768；
- workload：25%/50%/75% 三档真实 query 集，每档 1,000 queries；K=10；
- Ground Truth：`/home/graphdb/FilterVectorResult/Amazon/GroundTruth/<query_dir>/Amazon_gt_labels_containment.bin`；
- 100 search threads，CPU brute-force ELS，neighbor-list backend；
- 初筛每个工作点 3 repeats；最终候选已经完成 7 repeats。表中 batch time 是整批 1,000 queries 的 wall time，不是单 query latency；
- 单层查询显式启用 root-label coverage，使覆盖语义与多层一致。

构建共用参数：max degree 64、cross edges 4、Lbuild 100、alpha 1.2；intra route 的 small/mid/large 分界为 2048/8192，large backend 为 `jasper_style`，mid sampling 256；CPU inter graph route 的强制阈值为 pair-work 10000，`inter_search_ef=32`。

正式构建日志确认：虽然 route 统计字段名仍叫 `gpu_intra_blocks`，本轮没有设置全局 `UNG_SPECIAL_BLOCK_GPU_INTRA/INTER`；分流本身会让 `n>=8192` 的 large blocks 走 `jasper_style`/FastGrnnd CUDA，small/mid 分别走 CPU exact-topK / sampled-Vamana。inter 的 200--220 个 pair 全部走 CPU graph search，`gpu_inter_used=0`。因此 72--94 秒表应称为 hybrid-intra + CPU-inter overlay，而不是 GPU/GPU。

## 当前 Amazon x1 构建结果

| T2 | upper blocks | total build s | overlay s | intra s | inter s | edges | disk MiB | loaded MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4k | 46 | 94.425 | 73.003 | 44.523 | 28.475 | 72,733,732 | 762.6 | 850.2 |
| 10k | 22 | 89.051 | 68.690 | 40.847 | 27.838 | 71,753,752 | 755.1 | 842.4 |
| 25k | 8 | 92.024 | 70.157 | 40.599 | 29.554 | 69,343,169 | 736.7 | 824.0 |
| 50k | 5 | 94.396 | 70.898 | 43.289 | 27.604 | 66,543,978 | 715.2 | 802.2 |

以下旧 1k/10k hybrid 表只用于保留实现过程证据，不进入最终论文主表。

| 指标 | 单层 T1=1k | 多层 T1=1k, T2=10k | 多层代价 |
|---|---:|---:|---:|
| block 数 | 170 | 188 = 170 中层 + 18 上层 | +18 |
| 中层 member points | 581,472 | 581,472 | 不变 |
| 上层 member points | 0 | 547,160 | +547,160 |
| special edges | 35,445,996 | 61,771,891 | +74.3% |
| intra build | 32.358 s | 34.334 s | +6.1% |
| inter build | 12.591 s | 20.703 s | +64.4% |
| trie-regular build | 9.430 s | 9.957 s | +5.6% |
| save | 2.449 s | 3.654 s | +49.2% |
| total build | 63.180 s | 74.784 s | +11.604 s / +18.4% |
| compact reload | 449.2 ms | 666.4 ms | +48.4% |
| disk bytes | 512,448,677 | 729,864,536 | +217,415,859 / +42.4% |

多层中 159/170 个中层 block 有合法上层父块，11 个没有上层容器。两层的 member-point 计数有意重叠，不应相加后解释为不同数据点覆盖数。

构建 artifacts：

- 单层：`runs/formal_single_1k_current/`；
- 多层：`runs/formal_1k_10k_hybrid/`。

## 查询结果

### 当前正式 Amazon x1 主表

以下结果使用正确的 30,723-label Amazon x1 主图。每行在预声明的共同 Recall 门槛下，从各方法实测点中选择最快 warm median，不插值、不外推。

| 选择率 | Recall 门槛 | Single T1=1k | Multi-level | speedup |
|---:|---:|---|---|---:|
| 24.915% | >=.90 | L20000, R=.9057, 6520.180 ms | T2=25k, L16000, R=.9062, 5335.580 ms | 1.222x |
| 49.971% | >=.85 | L2000, R=.8640, 673.019 ms | T2=50k, L700, R=.8661, 439.856 ms | 1.530x |
| 74.994% | >=.87 | L5000, R=.8865, 2508.740 ms | T2=25k, L1600, R=.8871, 1069.745 ms | 2.345x |

在单层最高质量附近，多层分别达到 `1.222x/4.925x/6.290x`。完整构建表、外部系统对照和 claim 边界见 `docs/reports/MULTILEVEL_SPECIAL_BLOCK_PAPER_REPORT_CN.md`。FAVOR 在当前三档共同门槛下均比多层快，因此不能声称系统全面领先。

### 历史 hybrid 机制实验（不得进入最终性能主表）

#### 固定 Lsearch

| Lsearch | 单层 Recall | 单层 batch ms | 多层 Recall | 多层 batch ms |
|---:|---:|---:|---:|---:|
| 500 | 0.821664 | 696.697 | 0.899087 | 851.338 |
| 600 | 0.842669 | 650.190 | 0.908929 | 808.761 |
| 750 | 0.865398 | 767.823 | 0.920548 | 913.236 |
| 1000 | 0.902080 | 879.104 | 0.938153 | 1085.730 |
| 1500 | 0.922171 | 1146.220 | 0.946575 | 1383.760 |
| 2000 | 0.934196 | 1389.820 | 0.951700 | 1643.460 |

同 L 下多层做了更多有效搜索，因此耗时更高、Recall 也更高；不能据此宣称查询加速。

#### 相同 Recall 的公平比较

| 目标质量 | 单层 | 多层 | Recall 差 | batch speedup | L 缩减 |
|---|---:|---:|---:|---:|---:|
| Recall 约 0.902 | L1000: 0.902080, 879.104 ms | L532: 0.901674, 782.128 ms | -0.000406 | 1.124x | 46.8% |
| Recall 约 0.934 | L2000: 0.934196, 1389.820 ms | L940: 0.934805, 1044.000 ms | +0.000609 | 1.331x | 53.0% |

每个进程的 repeat 0 有冷启动波动。warm repeats 1--4 的近似结果分别为 1.11x 和 1.32x，与 all-5 mean 结论一致。

查询 artifacts：

- 单层 sweep：`runs/query_current_repeat5/single/`；
- 多层 sweep：`runs/query_current_repeat5/multilevel/`；
- L532：`runs/query_current_repeat5/multilevel_match_532/`；
- L520/L940：`runs/query_current_repeat5/multilevel_exact_match/`；
- detailed diagnostics：`runs/query_diagnostics/multilevel_l532/`。

`UNG_SPECIAL_LIGHT_STATS=0` 才会记录 edge counters；默认 light stats 中这些计数为 0，不能解释为 special path 没有执行。开启 detailed stats 会显著扰动时间，因此只用于机制解释，不作为性能表数字。L532 诊断中，每 query 平均 118.31 次 level upgrade、39,164.26 条 intra special edge scan、7,272.47 条 inter special edge scan。

## 验证与复现入口

构建目录：`/home/sunyahui/worktrees/FilterVectorCode_multilevel_special/build_ung_rel`。首次配置需要：

```bash
cmake -S UNG/codes -B build_ung_rel \
  -DCMAKE_BUILD_TYPE=Release \
  -DTAGORE_ROOT=/home/graphdb/Tagore
cmake --build build_ung_rel -j16 --target \
  test_special_block_trie test_special_block_free_state \
  test_special_candidate_queue test_special_edge_io \
  build_special_block_index search_UNG_index
```

Focused tests：

```bash
cd build_ung_rel
ctest -R 'special_block_trie|special_block_free_state|special_candidate_queue|special_edge_io' \
  --output-on-failure
```

最新结果为 focused C++ 4/4、multilevel Python 13/13、Curator 12/12；ACORN adapter/build/smoke 通过，所有纳入正式表的 ACORN 结果均为 0 filter violations。论文主表可由 `generate_paper_results.py` 重生成，最终 SHA-256 为 `74fb87c5dfc94af9ceb926cb1d4c1370ae2a8f1e76f37a5fe2802ec139012516`。官方 ACORN patch 已在干净 `c259f11c` archive 上通过 `git apply --check` 和实际应用验证；`git diff --check` 通过。重点测试覆盖普通->中层->上层激活约束、候选按 point 唯一及 level upgrade 后重新扩展、metadata v3 round-trip、新旧 index format 白名单。

## Merge-back 提示

不要把 `runs/` 或本机构建缓存提交到主分支。原始工作树仍是大型 dirty tree，即使 HEAD 未前进，也不满足直接 merge 条件。建议：

1. 在原始分支先审阅并提交当前用户改动，确保形成与 `7811846` 等价的基线；
2. 从该 clean checkpoint cherry-pick `ab984da` 之后的意图提交；核心实现是 `1f0c1e6`、`368227e`，实验/报告提交延续到 `4169be6`；`cea584c` 仅是旧 handoff，可跳过；
3. 对涉及 `search_cache.h`、`uni_nav_graph*.cpp`、special block headers/tests 的文件人工审阅冲突；
4. 合并后重跑上述 4 个 focused tests，并至少复跑单层 L1000 与多层 L532 的端到端 Recall；
5. 保持单层默认兼容：未设置 `UNG_SPECIAL_BLOCK_UPPER_MIN_POINTS` 时不得生成上层 block。

此 handoff 记录的是当前实测状态，不是不可修改的设计合同；若实现或实验口径变化，应同步更新。
