# Cross-Edge TopK Ablation Design

## Goal

Add an experiment configuration for comparing UNG cross-edge construction variants:

1. Small-graph task batching with a matrix-style SGEMM/topK path.
2. Small-graph task batching with fused on-chip top-k.
3. `special_blocks+UNG` inter-block GPU construction as a control for whether the cross-block path uses the same optimization family.

## Current Code Findings

Ordinary UNG cross-group edges use `UNG_CROSS_EDGE_IMPL=1` for GPU batched construction. The two ordinary variants can be selected through existing environment knobs:

- Matrix-style path: `UNG_GPU_TOPK_IMPL=2`, fused bucket kernels disabled, double-buffer disabled so the regular SGEMM/topK route is exercised.
- On-chip top-k path: `UNG_GPU_TOPK_IMPL=3`, bucket fused kernels enabled, direct qid fused and GPU global merge enabled, double-buffer disabled so the ordinary descriptor bucket route is visible in logs.

Special-block inter-block edges are controlled by `UNG_SPECIAL_BLOCK_GPU_INTER=1`. That path already batches special inter-block queries and uses `ung_source_exact_topk_global_kernel` or `ung_source_tf32_wmma_topk_global_kernel`, both of which maintain top-k on the GPU. It is not the same ordinary UNG small/medium `UngGroupQueryDesc` bucket route, so the config labels it as a separate control rather than claiming direct reuse.

## Deliverable

Create `experiments/cross_edge_topk_ablation/config.json` with:

- Shared dataset/build defaults matching existing experiment configs.
- `variants[]` containing three complete runner configs.
- Clear variant names and notes describing the intended code path.
- `run_command_template` showing that each generated variant can be passed to `run_cpu_special_blocks_experiment.sh` after materialization as a single config JSON.

## Non-Goals

Do not add a new runner or C++ implementation in this step. The config should be useful for manual materialization or for a later runner, and should not change existing build behavior.
