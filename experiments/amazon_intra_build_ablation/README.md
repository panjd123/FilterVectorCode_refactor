# Amazon 块内图构建时间对比

五组只改变块内图构建方式（CPU 参考组和 GPU 的完整 2×2 对照）。这里的 **topK 优化**专指 Tagore 候选更新中的
“第 K 距离阈值过滤 + 局部 Bitonic 排序 + 二分排名并行归并”，不指纯 kNN
图、不指 FastGrnnd 剪枝，也不指跨组边的 fused top-k。

| 名称 | 块内后端 | 均衡调度 | 候选更新 |
|---|---|---|---|
| `cpu` | CPU Vamana | 无 | CPU 原实现 |
| `gpu_baseline` | GPU GNN-Descent + FastGrnnd | 单 stream | 完整排序对照 |
| `load_balance_only` | GPU GNN-Descent + FastGrnnd | 4 stream | 完整排序对照 |
| `topk_only` | GPU GNN-Descent + FastGrnnd | 单 stream | 原有优化 topK |
| `load_balance_topk` | GPU GNN-Descent + FastGrnnd | 4 stream | 原有优化 topK |

CPU 是外部参考基线；“仅负载均衡”仍包含 GPU 候选生成、FP16/Tensor Core 和
FastGrnnd 等公共能力，不能将它相对 CPU 的全部收益归因于负载均衡。要独立
计算负载均衡的收益，可比较 `gpu_baseline` 和 `load_balance_only`，或比较
`topk_only` 和 `load_balance_topk`；固定均衡调度
后的 topK 收益，应比较 `load_balance_only` 和 `load_balance_topk`。

`UNG_TAGORE_OPTIMIZED_TOPK=0` 使用独立对照 kernel：在 GNN-Descent 第一阶段更新、
第二阶段更新、最终反向边归并三处，对旧候选和新候选完整去重、按距离排序后截取 K。
不使用第 K 距离阈值淘汰，也不使用二分排名归并。完整排序仍使用标准 Bitonic
排序网络；第二阶段的采样、距离计算前 ID 去重及最终 FastGrnnd 剪枝不变。
独立 kernel 避免给原优化 kernel 增加共享内存、改变资源占用。默认开关为 1。

已有的 Amazon 串行 GPU 结果确实对应 `topk_only` 的算法组合；此前的四 stream
验证对应 `load_balance_topk`。它们未在同一批独占性能实验中运行，因此只作历史
记录，本实验重新计时，不混入旧样本。

从项目根目录运行：

```bash
# 只生成五组配置并查看命令。
python3 experiments/amazon_intra_build_ablation/run.py --dry-run

# 正式实验：默认每组 1 次独立进程预热、3 次计时，轮换组间顺序。
python3 experiments/amazon_intra_build_ablation/run.py

# 快速检查五组是否完整跑通：每组一次，无预热，不作为稳定性能结论。
python3 experiments/amazon_intra_build_ablation/run.py --repeat 1 --warmup 0

# 补测关闭两项优化的 GPU 基线，仍从头构建基础 Trie 和块索引。
python3 experiments/amazon_intra_build_ablation/run.py --only gpu_baseline --skip-build
```

配置位于 `config.json`。每个预热/计时样本都从 Amazon 原始数据重新构建基础 Trie，
再构建块索引；基础构建配置相同，索引文件各自独立，不复用已有基础索引跳过计时。
分块阈值 500、块内出度 32、跨块边数 4、K 请求值 128（当前编译上限为 96）、
GNN-Descent 迭代 4、Lbuild=100、alpha=1.2、CPU 线程设置 60 均一致。CPU Vamana
沿用当前块构建器的逐块、每块单线程调用；60 主要用于其他 CPU 阶段，不能描述
为“60 线程 CPU 块内建图”。GPU 块间连边关闭，所有组使用同样的 CPU 块间连边配置。

完整索引时间有两列，禁止混用：

- `total_index_build_ms` = 基础 `index_time_add_rb(ms)` + 块 `build_time(ms)`。
  与现有 `memory_and_build_summary.csv` 的 PTNC 总构建时间口径一致。
- `end_to_end_wall_ms` 从基础索引进程启动前计到块索引进程退出后，包含两阶段数据
  加载、构建、保存及进程开销；排除编译、等待 GPU 锁、结果分析和画图。
  由同一锁内的 worker 通过单调时钟实测，不能用历史计时简单相加来冒充。

`base_wall_ms`、`block_wall_ms` 分别记录两阶段进程墙钟。`block_index_ms` 保留块侧
自身计时，`intra_ms`、`inter_ms`、`save_ms` 等用于解释差异。所有总时间先按样本
逐次相加/实测，再取中位数，避免将两个阶段的独立中位数相加。

CPU 块间连边使用块内图搜索，因此不同近似图可能改变实际距离计算次数。
新版 `[cpu_inter_profile] graph_distance_computations=` 和慢 pair 日志记录实际工作量，
不再仅凭 `graph_queries`、`graph_pair_work` 推断搜索成本。`timing.json` 还记录块阶段
进程 CPU 时间和非自愿上下文切换。实验仅比较构建时间，不证明召回率相同。

计时遵循 `cuda-operator-benchmark` 的 GPU 协调要求，每个 CPU/GPU 样本均通过
`gpulock perf --wait-gpu-idle 0 -- ...` 串行运行；仅有 GPU0，默认使用 GPU0。
这是完整索引构建的明确计时例外：payload=1，不做 L2 bypass、不捕获 CUDA Graph，
因为流程包含 CPU 数据准备、动态图分配和文件输出；采用进程级预热与重复计时。
功能正确性测试单独通过 `gpulock check` 运行，测试时间不进入统计。

每次运行创建新的结果目录，绝不复用完成的索引跳过计时，也不覆盖现有正式索引：

```text
FilterVectorResult/Amazon/intra_build_ablation/<北京时间>/
  manifest.json                 # 命令、源配置、两个构建器哈希、计时范围
  warmup_r01/<variant>/         # 预热数据，排除在最终统计之外
  measured_r01/<variant>/
    config.json                # 每组实际环境及构建参数
    phase_commands.json        # 基础构建及块构建的实际命令
    base_build.log
    build.log
    index_files/meta           # 本样本新构建的基础 Trie
    block_index_files/meta
    timing.json                # 真实端到端墙钟、分阶段墙钟和 CPU 使用计数
  all_runs.csv                 # 原始分阶段时间及路由计数
  comparison.csv               # 中位数、最小、最大值
  comparison.md                # 可阅读的对比表
  comparison.png / .pdf        # 两个时间口径的柱状图，误差线为最小/最大值
```

结果逐样本写入。发现 GPU 回退、块数/点数不完整、topK 或均衡开关没有生效，
或不同组的源指纹、分块、出度不一致时直接失败，不纳入该样本。

功能测试：

```bash
python3 -m unittest discover -s experiments/amazon_intra_build_ablation -p 'test_*.py'
cmake -S UNG/codes -B build_ung_multilevel_trie_gpu -DUNG_BUILD_TESTS=ON
cmake --build build_ung_multilevel_trie_gpu -j8 --target test_tagore_topk_ablation build_UNG_index build_special_block_index
gpulock check 0 -- build_ung_multilevel_trie_gpu/test/test_tagore_topk_ablation
gpulock check 0 -- env \
  UNG_GPU_INTRA_TEST_BINARY="$PWD/build_ung_multilevel_trie_gpu/apps/build_special_block_index" \
  python3 tools/tests/test_special_block_gpu_intra.py
```

分析历史块侧试验并补上基础构建计时（保留原始 CSV 不变）：

```bash
python3 experiments/amazon_intra_build_ablation/analyze.py \
  /home/dev/graphdb/FilterVectorResult/Amazon/intra_build_ablation/20261018_224645_773560 \
  --base-meta /home/dev/graphdb/FilterVectorResult/Amazon/index/Trie_block_500_M_32_cross_block_4/index_files/meta
```

输出 `comparison_with_base.csv`、`analysis.md`、`analysis.json`。历史数据没有新的
`timing.json` 时，只补齐“基础＋块”的构建计时，不声称恢复了真实端到端墙钟。
分析新的完整样本不需要 `--base-meta`。

块间阶段的独立诊断（与正式端到端样本分开）：

```bash
python3 experiments/amazon_intra_build_ablation/diagnose_inter.py \
  /path/to/completed/index_files /path/to/new/diagnostic_directory --repeat 3
```

该脚本复用基础 Trie，比较单 stream topK 和组合方式；默认分别保留原实现、
跳过块间结果写入、将 `MALLOC_ARENA_MAX` 从 2 增至 16。可用 `--modes reserve`
诊断预分配边容量的影响。诊断中的 `no_output` 索引缺少块间边，**不可用于查询**。
`profile_mutex.cpp` 仅通过 LD_PRELOAD 采样每个线程/调用位置的 1/64 次锁获取，
不修改正式构图二进制。`mutex_summary.json` 的 `estimated_acquire_thread_ms` 是
各线程锁获取耗时的估算总和，含调度延迟和无竞争获取开销，不能当作墙钟直接相减。
采样会扰动运行，诊断时间不并入 `comparison.csv` 的正式统计。
