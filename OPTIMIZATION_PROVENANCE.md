# Optimization Provenance

- Optimization reference repo: `/home/graphdb/aaaGPU/FilterVectorCode`
- Source branch/commit: `gpu-opt-correctness-loop` (see `.optimization_source_aaagpu_commit`)
- Imported optimization files:
  - `UNG/codes/include/uni_nav_graph.h`
  - `UNG/codes/src/uni_nav_graph.cpp`
  - `UNG/codes/src/gpu_gemm_topk.cu`
  - `UNG/codes/src/CMakeLists.txt`
- Baseline CUDA file is preserved as `UNG/codes/src/uni_nav_graph_cuda.baseline.cu` for diff/audit.
