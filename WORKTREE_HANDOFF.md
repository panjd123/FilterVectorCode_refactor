# Handoff: Multi-level Special Block

## User intent

Implement and evaluate a real hierarchical Special Block graph in an isolated checkout. Keep the existing roughly 1k-point Special Blocks as a middle layer, add a larger roughly 10k-point layer above them, and let search expand progressively from the ordinary graph into middle blocks and then larger blocks. This is a graph/search hierarchy; it must not change the active ELS contract from search-consumable group IDs back to trie-node outputs.

## Isolated checkout

- Source checkout: `/home/graphdb/FilterVectorCode_refactor`
- Source branch at creation: `shopai8/special-block-e2e-opt`
- Source HEAD at creation: `dda63bd7663b06dce0ce3a977b81268df09b2d01`
- Isolated checkout: `/home/sunyahui/worktrees/FilterVectorCode_multilevel_special`
- Implementation branch: `codex/multilevel-special-block-20260905`
- Baseline snapshot: `781184607b921b63e71d1ff1d0550aef46b0c4f7`
- C++17 build fix checkpoint: `ab984da`
- Intended merge-back target: the source checkout/branch above, after carefully reconciling its large dirty working tree. Do not blindly merge or overwrite it.

The server has Git 1.8.3.1 and no `git worktree` command. This directory is therefore a `git clone --shared` equivalent: it has an independent branch and index while sharing Git objects with the source repository. There are no configured submodules.

## Current verified state

The baseline snapshot contains the source repository's current uncommitted Special Block implementation rather than only its older HEAD. The existing implementation has one configured threshold, persistent trie/block metadata, parent-child block relations, group-ID entry providers, special intra/inter edges, and a boolean free-state search path. Point/group ownership is still represented primarily as one direct block, so explicit multi-level activation and edge ownership need careful design.

Fresh build setup:

```bash
cmake -S UNG/codes -B build_ung_rel \
  -DCMAKE_BUILD_TYPE=Release \
  -DTAGORE_ROOT=/home/graphdb/Tagore
cmake --build build_ung_rel -j16 --target \
  test_special_block_trie test_special_block_free_state \
  test_special_candidate_queue build_special_block_index search_UNG_index
```

Focused validation last run:

```bash
cd build_ung_rel
ctest -R 'special_block_trie|special_block_free_state|special_candidate_queue' \
  --output-on-failure
```

Result: 3/3 passed on 2026-09-05 after commit `ab984da`.

`UNG/codes/third_party/CRoaring/build/src/libroaring.a` is an ignored, disposable copy from the source checkout. It was reused because rebuilding bundled CRoaring fails on this host's assembler for AVX-512 `vpcompressb`/`vpopcntq`. Do not commit this cache. `FAVOR/` is also locally excluded and was intentionally not included in the baseline snapshot.

## Design constraints and evidence

- Preserve the current single-level behavior as the baseline and backward-compatible default.
- A 10k threshold must add an upper layer; it must not replace the 1k layer. Prior single-threshold 2k/4k tests gained fixed-L speed but lost at equal Recall.
- Search quality is established only by end-to-end filtered-search Recall/QPS. Block counts, entry counts, or top-k overlap are diagnostics, not quality substitutes.
- Existing representative Amazon index metadata: 602,453 points, 510,639 groups, 170 blocks, 156 child-block relations, threshold 1000, and 2,108,923 trie nodes.
- Existing query coverage is propagated downward through `child_block_ids`; the current runtime uses one direct point/group block mapping and mostly a `bool free` state.
- At large L, free special-edge scans dominate observed edge scans. The upper layer is useful only if it improves navigation enough to reduce work at equal Recall; merely adding more edges can regress performance.

Start with `ung_special_blocks.h`, `uni_nav_graph_special_blocks.cpp`, `ung_special_block_trie.*`, `uni_nav_graph_entry_provider.cpp`, `uni_nav_graph_search_backend.cpp`, `ung_special_block_activation.h`, `uni_nav_graph.h`, and `ung_build_config.*`. Keep facts, hypotheses, and measured results separate.

## Historical baseline for orientation

Amazon Router results include L=1000 Recall 0.909944 / QPS 2141.36, L=5000 Recall 0.978488 / QPS 904.63, L=10000 Recall 0.991071 / QPS 609.03, and L=20000 Recall 0.993962 / QPS 406.24. The cap16 path includes L=2000 Recall 0.943176 / QPS 2200.60 and L=2500 Recall 0.952714 / QPS 2100.59. These are historical references; reruns must use corrected `/home/graphdb/...` paths and record fresh configs.

## Working discipline

Read this file first, but treat it as live context rather than an immutable design. Update it when implementation semantics, environment assumptions, validation, or branch state materially change. Commit verified milestones before risky restructuring. Before reporting completion, clean temporary artifacts, rerun relevant tests, compare end-to-end Recall/QPS at equal quality, inspect the dirty source merge target, and perform a conflict/readiness check without modifying the user's source worktree. Final reporting should be in Chinese.
