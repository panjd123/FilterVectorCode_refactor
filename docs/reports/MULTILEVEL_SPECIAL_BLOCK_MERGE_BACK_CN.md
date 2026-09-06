# Multi-level Special Block 合并风险清单

最后审计：2026-09-06。此文件只描述如何把隔离分支交回原始脏 checkout；它不是自动合并授权。

## 当前边界

- 隔离分支：`codex/multilevel-special-block-20260905`；最终源码与证据检查点 `cb72797af5af615ca196d3618903417a54b0da2b`，状态文档提交见分支 HEAD。
- 原始 checkout：`/home/graphdb/FilterVectorCode_refactor`，HEAD 与共同基线均为 `dda63bd7663b06dce0ce3a977b81268df09b2d01`。
- 隔离分支领先共同基线 70+ 个提交，改动 410 个路径。
- 原始 checkout 有 150 条 status 记录；其中 128 个路径也被隔离分支修改。
- 128 个重叠路径中，101 个当前结果相同（含双方都删除的 `experiments/search_comparison/config_amazon.json`），27 个内容不同。没有“只存在一侧”的未解释路径。

因此，分支可以独立审阅和复现，但原始 checkout 尚不适合直接 merge 或整段 cherry-pick。先保存原始改动，再对下列 27 个路径进行三方整合：

```text
UNG/codes/apps/build_special_block_index.cpp
UNG/codes/apps/search_UNG_index.cpp
UNG/codes/include/search_cache.h
UNG/codes/include/special_block_index_builder.h
UNG/codes/include/ung_build_config.h
UNG/codes/include/ung_query_route.h
UNG/codes/include/ung_query_stats.h
UNG/codes/include/ung_special_block_activation.h
UNG/codes/include/ung_special_blocks.h
UNG/codes/include/ung_special_candidate_queue.h
UNG/codes/include/ung_special_edge_io.h
UNG/codes/include/uni_nav_graph.h
UNG/codes/src/special_block_index_builder.cpp
UNG/codes/src/ung_build_config.cpp
UNG/codes/src/ung_special_candidate_queue.cpp
UNG/codes/src/ung_special_edge_io.cpp
UNG/codes/src/uni_nav_graph_batch_gpu_search.cpp
UNG/codes/src/uni_nav_graph_io.cpp
UNG/codes/src/uni_nav_graph_query_features.cpp
UNG/codes/src/uni_nav_graph_query_route.cpp
UNG/codes/src/uni_nav_graph_search.cpp
UNG/codes/src/uni_nav_graph_search_backend.cpp
UNG/codes/src/uni_nav_graph_special_blocks.cpp
UNG/codes/test/CMakeLists.txt
UNG/codes/test/test_special_block_free_state.cpp
UNG/codes/test/test_special_candidate_queue.cpp
UNG/codes/test/test_special_edge_io.cpp
```

## 建议合并顺序

1. 在原始 checkout 上先提交、stash 或另建备份分支，确保现有 150 条改动可恢复。
2. 以 `dda63bd...` 为 merge base，对上述 27 个文件逐项做三方合并；其余同内容路径不需要手工改写。
3. 不带入隔离目录中的 `runs/`、`thirdparty/acorn-official/`、`thirdparty/curator-v2/`。
4. 合并后至少重跑生产 targets、CTest 14/14、Python 32/32、结果生成器，以及一轮 Amazon x1 fresh build + filtered query validator。
5. 只有合并后的 source fingerprint、T1/T2、binary hash 和 query/GT provenance 都一致，才能引用现有论文表。
