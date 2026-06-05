# 优化来源追踪（Optimization Provenance）

本文档用于回答两个问题：

1. 当前 `FilterVectorCode_refactor` 的优化代码来自哪里。
2. 每一轮优化/回退在历史上如何定位、如何审计。

## 1. 优化来源仓库

- 参考仓库：`/home/graphdb/aaaGPU/FilterVectorCode`
- 主要参考分支/提交：`gpu-opt-correctness-loop`（见 `.optimization_source_aaagpu_commit`）

## 2. 首轮迁移范围

最初从参考仓库迁移进入 refactor 的核心文件：

- `UNG/codes/include/uni_nav_graph.h`
- `UNG/codes/src/uni_nav_graph.cpp`
- `UNG/codes/src/gpu_gemm_topk.cu`
- `UNG/codes/src/CMakeLists.txt`

同时保留了基线 CUDA 文件用于审计对比：

- `UNG/codes/src/uni_nav_graph_cuda.baseline.cu`

## 3. 增量迁移（2026-03-02）

- 目标提交（本仓库）：`076d990`
- 额外参考提交（aaaGPU）：`eb6d892`
- 迁移内容（`UNG/codes/src/uni_nav_graph.cpp`）：
  - `get_min_super_sets`：容器复用 + 桶化遍历
  - `get_descendants_info`：线程本地 BFS 缓冲复用
  - `cal_f_coverage_ratio`：线程化 + `descendants_direct` 分支 + 环境变量开关

## 4. 已回退实验（2026-03-03）

- 试验提交：`ff47638`
- 回退提交：`14a0fa9`
- 试验目标：通过覆盖率阶段线程上限与调度策略降低抖动
- 回退原因：在复现实验中出现可重复的端到端回退（详见 `HISTORY_MAP.md`）

## 5. 高影响稳定优化（2026-03-03）

### 5.1 LNG/覆盖集合容器重构

- 提交：`216cd6c`
- 文件：
  - `UNG/codes/include/label_nav_graph.h`
  - `UNG/codes/src/uni_nav_graph.cpp`
- 核心变更：
  - `_lng_descendants` / `covered_sets` 从哈希集合布局迁移为连续向量布局
  - descendants/coverage 构建路径改为顺序写入，减少哈希插入热点
  - legacy 覆盖路径保留 `sort + unique` 保证语义
  - Roaring 初始化改为 `addMany` 批量写入 + OMP 并行

### 5.2 Trie 与 LNG Phase1 低风险提速

- 提交：`d68f3a0`
- 文件：
  - `UNG/codes/src/trie.cpp`
  - `UNG/codes/src/uni_nav_graph.cpp`
- 核心变更：
  - Phase1 入口超集结果容器改为线程本地复用 + `swap`
  - trie 插入去掉冗余 resize
  - superset 候选遍历改为 `vector + head`，去重改为 `unordered_set`

## 6. 追踪建议

若要继续优化并保持可审计，请遵守以下节奏：

1. 每一轮性能改动单独 commit（不要把多类改动揉在一起）。
2. 每轮至少保留一份 A/B CSV 与对应日志路径。
3. 每轮标注“是否语义等价、如何验证正确性”。
4. 回退失败实验也要记录，不要删除证据链。
