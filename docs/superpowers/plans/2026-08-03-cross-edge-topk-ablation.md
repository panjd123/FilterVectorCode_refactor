# Cross-Edge TopK Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config file for ordinary UNG cross-edge top-k ablations and a special-block GPU inter-edge control.

**Architecture:** Use a single JSON document under `experiments/cross_edge_topk_ablation/` with shared defaults plus fully expanded `variants[]`. Each variant mirrors the existing `run_cpu_special_blocks_experiment.sh` config schema so it can be materialized as an individual runner config.

**Tech Stack:** JSON experiment config, existing `run_cpu_special_blocks_experiment.sh`, UNG environment variables.

## Global Constraints

- Keep changes limited to docs and experiment configuration.
- Do not modify existing experiment runners.
- Do not claim ordinary bucket-fused cross-edge kernels are applied to special-block inter edges; label the special-block path as source-exact/WMMA top-k.

---

### Task 1: Add Ablation Config

**Files:**
- Create: `experiments/cross_edge_topk_ablation/config.json`

**Interfaces:**
- Consumes: Existing shell runner config fields: `run_name`, `output_layout`, `index_name`, `data_root`, `result_root`, `build_dir`, `datasets`, `build`, `ung_env`, `index_name_params`.
- Produces: `variants[]` entries that can each be materialized into one shell-runner config.

- [x] **Step 1: Identify existing config schema**

Read `experiments/cpu_special_blocks/config.json` and `experiments/cpu_special_blocks/config_ung.json`.

- [x] **Step 2: Define ordinary matrix-style variant**

Use `UNG_CROSS_EDGE_IMPL=1`, `UNG_GPU_TOPK_IMPL=2`, disable fused bucket kernels, and disable double-buffer routing.

- [x] **Step 3: Define ordinary on-chip top-k variant**

Use `UNG_CROSS_EDGE_IMPL=1`, `UNG_GPU_TOPK_IMPL=3`, enable bucket fused kernels, direct qid fused mode, id-only writeback, GPU global merge, and disable double-buffer routing.

- [x] **Step 4: Define special-block GPU inter control**

Use `UNG_SPECIAL_BLOCKS=1`, `UNG_SPECIAL_BLOCK_GPU_INTER=1`, and keep the label explicit: `special_inter_source_exact_topk`.

- [x] **Step 5: Validate JSON**

Run `python3 -m json.tool experiments/cross_edge_topk_ablation/config.json`.

### Task 2: Report Special-Block Applicability

**Files:**
- No code files modified.

**Interfaces:**
- Consumes: Code findings from `gpu_build_special_inter_edges`.
- Produces: Final answer explaining that special-block inter edges have a GPU batched on-chip top-k path, but not the ordinary small/medium descriptor bucket route.

- [x] **Step 1: Cite code path**

Reference `gpu_build_special_inter_edges` and its source-exact/WMMA kernel launches.

- [x] **Step 2: State next implementation option**

Explain that applying ordinary bucket descriptors to special inter edges would require a separate C++ route that converts child block member groups into `UngGroupQueryDesc`/`UngGroupTileDesc` batches keyed by source block queries.
