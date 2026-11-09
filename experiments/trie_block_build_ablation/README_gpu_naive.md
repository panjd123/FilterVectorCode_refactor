# 独立 GPU-Naive 构建对照

GPU-Naive 的定义为：CUDA Core 基础 local-join 距离内核、单 stream、完整候选排序和完整邻接行回传。
`UNG_TAGORE_NAIVE=1` 显式启用；默认值为 0，不改变现有四组实验入口与配置。

第一阶段用一个线程直接扫描一个 16 候选 tile，以 FP32 FMA 累加 FP16 输入的内积；
沿用 Tagore 的 new-old 双向局部最小候选和 new-new 上三角 tile 候选策略，
使用相同的 norm-dot 距离表达式、无效距离阈值、采样集与迭代次数，不使用 WMMA。
第二阶段沿用已有 CUDA Core 距离计算；候选更新使用现有 full-sort 对照内核。
FastGrnnd 剪枝仍使用正向 64/反向 32 的预算、alpha=1.2 和最大出度 32。
K 请求仍为 128、编译上限仍为 96，回传每点全部 96 槽位。

本版本没有增加人为延迟、无用迭代或额外候选工作；距离累加顺序和近似图可不同，
不声称与 Tensor Core/CPU Vamana 结果逐位相同或具有相同召回率。
分块、基础 Trie、块间 CPU 连边、输出内容和计时范围与已有实验一致。

运行配置：`config_gpu_naive_eight.json`，八份独立配置位于 `configs_gpu_naive/`。
编译目录：`build_ung_trie_gpu_naive`。结果根目录：`FilterVectorResult/trie_block_gpu_naive/`。
不写入已有 `current_summary.md`、四组 CSV 或其绘图目录。

从项目根目录配置和编译独立目录：

```bash
cmake -S UNG/codes -B build_ung_trie_gpu_naive \
  -DCMAKE_BUILD_TYPE=Release -DUNG_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DUNG_BUILD_APPS=ON \
  -DUNG_BUILD_TOOLS=OFF -DUNG_BUILD_TESTS=OFF
cmake --build build_ung_trie_gpu_naive -j 8 \
  --target build_UNG_index build_special_block_index
python3 experiments/trie_block_build_ablation/run_gpu_naive.py --dry-run
python3 experiments/trie_block_build_ablation/run_gpu_naive.py --skip-build
```

默认每个数据集测量一次，无独立预热，与已有初测口径一致。
每个样本从原始数据重新构建基础 Trie，再构建块索引；包含加载、构建、保存及进程开销。
所有计时经 `gpulock perf --wait-gpu-idle 0 -- ...` 串行执行。
完整索引计时沿用已有记录的例外：payload=1，不做 L2 bypass、不捕获 CUDA Graph；
编译、GPU 锁等待、正确性校验和报告生成均在计时之外。

运行器检查所有块确实走 CUDA、无 CPU fallback、Naive 内核覆盖所有块、
逐块 trace 点数覆盖完整、回传 stride=96，并保存二进制/配置/输入的审计信息。
CPU 参考测试覆盖 36 组维数/空候选/候选规模组合及默认路径与 Naive 开关约束。

```bash
python3 -m unittest discover -s experiments/trie_block_build_ablation -p 'test*.py'
gpulock check 0 -- /tmp/test_tagore_naive_distance
```

保留独立结果与已有四组的参考对比时，须注明跨批次、构建器哈希不同和单样本限制；
不能把新增 Naive 的测量冒充旧 `gpu` 组或覆盖历史结果。

本轮额外交叉校验：以相同 CPU 参考分别检查 Naive 与原 Tensor Core 内核，共 72 例通过。

本轮正式八数据集结果（8/8 完成，预检无效轮未计入）：

- /home/dev/graphdb/FilterVectorResult/trie_block_gpu_naive/20261020_153118_339915_gpu_naive_eight/naive_vs_existing.md
- 审计：/home/dev/graphdb/FilterVectorResult/trie_block_gpu_naive/20261020_153118_339915_gpu_naive_eight/verification/audit.json
