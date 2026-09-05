# Special Block Trie Entry Provider Implementation Plan

**Goal:** Persist the special-block construction trie and use its first-terminal
frontier as the only entry-group source for trie-partitioned special-block search.

## Constraints

- Preserve all unrelated dirty-worktree changes.
- Do not change LNG-partitioned block construction or existing providers.
- Do not perform global minimal-superset filtering.
- Do not move free/regular classification into the trie provider.
- Fail explicitly when the selected provider and loaded index are incompatible.

## Tasks

- [x] Add `SpecialBlockTrieIndex` with flat construction, first-terminal query,
      validation, binary save/load, checksum, and query statistics.
- [x] Replace the temporary `SpecialTrieNode` vector in trie block partitioning
      with the formal index and transient point-count arrays.
- [x] Persist `special_block_trie.bin`, load it only for trie-partitioned indexes,
      and report build/save/load/size metadata.
- [x] Add `special_block_trie` to provider parsing, dispatch, runtime validation,
      CLI help, and batch special search.
- [x] Add focused semantic and binary-corruption tests plus provider parsing tests.
- [x] Build the affected targets and run the focused CTest set.
- [x] Build or reuse the Amazon trie-partitioned index and run the
      `query_selected_recall_advantage` A/B evaluation.
