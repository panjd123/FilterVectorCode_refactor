# 七数据集 Trie block 四组构建对照

数据集：Genome、Reviews、VImg、Music、BReviews、Tiktok、Laion，不包含已测试的 Amazon。
实际配置在 `configs/<数据集>/<方式>.json`，共 28 份；批量脚本直接读取这些文件，
并为每个样本保存独立快照，不会静默忽略编辑重新生成。

| 文件名 | 块内后端 | 负载均衡 | 优化 topK |
|---|---|---|---|
| `cpu.json` | CPU Vamana | 无 | 无 |
| `gpu.json` | GPU GNN-Descent + FastGrnnd | 单 stream | 关闭，完整候选排序 |
| `gpu_balance.json` | 同上 | 4 stream，按块点数均衡 | 关闭，完整候选排序 |
| `gpu_balance_topk.json` | 同上 | 4 stream，按块点数均衡 | 开启 |

topK 指第 K 距离阈值过滤、局部 Bitonic 排序和二分排名并行归并。
所有 GPU 组保留共同的候选生成和 FastGrnnd 剪枝，不能将 CPU→GPU 的收益
全部归因于负载均衡或 topK。

四组统一：单层 `500:trie`、出度 32、跨块边数 4、Lbuild=100、alpha=1.2、
float/L2、CPU 线程设置 60、K 请求 128（当前编译上限 96）、M=64、迭代 4、
批次点数上限 65536。CPU 块内沿用逐块单线程 Vamana，60 线程用于其他 CPU 阶段。
块间统一为 CPU 搜索，并启用 `UNG_SPECIAL_INTER_RESERVE=1`，完整保存结果边。
禁止采样、跳过块关系、诊断 no-output、GPU 失败回退混入对照。

每个样本从原始数据重新构建基础 Trie，再构建块索引。**主指标
`end_to_end_wall_ms`** 从基础进程启动前计到块进程退出后，包含两阶段加载、
构建、保存和进程开销，不含编译、等待 GPU 锁、校验、统计及可选输出清理。
不复用基础索引绕过计时。另存内部 timer 之和 `total_index_build_ms`，
用于对齐原有汇总口径，不能把它当作实测端到端墙钟。

2026-10-19 起，基础索引构建已删除未被固定图搜索使用的全量后代、覆盖集合、
位图和选择器特征预计算；配置不再设置这些旧开关。端到端计时范围仍包含完整
基础 Trie 构建、块索引构建及两阶段保存。此前 `20261018_234449_171644` 的数据
属于旧实现，文件保持原样；新实现的性能需另行运行同一矩阵测量，不能直接把
删除前后的样本混作负载均衡/topK 消融。

从项目根目录运行：

```bash
# 查看全部 28 组计划，生成实际 case 配置，不执行构建。
python3 experiments/trie_block_build_ablation/run.py --dry-run

# 完整初测：每组 1 次、无独立预热，共 28 次完整流程。
python3 experiments/trie_block_build_ablation/run.py

# 更稳定的正式对照：每组 1 次独立预热、3 次计时。
python3 experiments/trie_block_build_ablation/run.py --warmup 1 --repeat 3

# 单独选择数据集或方式；参数可重复，顺序按传入顺序。
python3 experiments/trie_block_build_ablation/run.py --dataset Genome --dataset Reviews
python3 experiments/trie_block_build_ablation/run.py --dataset Music --variant gpu --variant gpu_balance_topk

# 编译器/二进制已验证时跳过编译，仍会记录两个构建器哈希。
python3 experiments/trie_block_build_ablation/run.py --skip-build

# 仅需要计时、磁盘不保留索引正文时显式添加此选项。
python3 experiments/trie_block_build_ablation/run.py --discard-index-data

# 中断后继续：完成样本重新校验、失败样本保存原目录并从头重测。
python3 experiments/trie_block_build_ablation/run.py --resume /path/to/existing/run

# 只重新汇总已完成的样本。
python3 experiments/trie_block_build_ablation/run.py --summarize /path/to/existing/run
```

默认初测的单样本不能用于断言稳定加速。多次重复时轮换四组顺序，报告中位数、
最小/最大值，预热不纳入统计。新运行总是创建新目录；旧正式索引不会被覆盖。
Laion 为 1515 万条 768 维向量，原有单组构建计时约一小时，全量初测需要较长时间。
保留 28 组完整索引会消耗数百 GB；多次重复可选择 `--discard-index-data`，
仅删除本次新建的 case 索引正文，meta、命令、日志、计时和统计仍保留。
该选项在全部计时结束后才清理对应样本，不能用于查询这些已清理的索引。

结果默认写入 `FilterVectorResult/trie_block_build_ablation/<北京时间>/`：

```text
manifest.json                    # 输入大小/mtime、命令、配置/构建器哈希、每组状态和失败原因
comparison.md                    # 七数据集 × 四方式的端到端秒数，逐样本更新
comparison.csv                   # 每数据集/方式的各阶段中位数与 min/max
all_runs.csv                     # 原始样本，含预热标记
Genome/measured_r01/cpu/
  config.json
  phase_commands.json
  base_build.log
  build.log
  timing.json
  index_files/meta               # 本样本新构建的基础 Trie
  block_index_files/meta
```

不同数据集的分区标识独立检查。任何组出现 GPU 回退、GPU 块覆盖不完整、
开关未生效、分块/源指纹不一致或时间非法都会失败并保留日志；批次继续其他组，
最终状态 `partial`，不会把失败记成 0 秒。恢复时拒绝输入、配置、构建器或计时
worker 已改变的批次，避免合并不同版本。

遵循 `cuda-operator-benchmark` 的独占要求，每个 CPU/GPU 计时样本均通过
`gpulock perf --wait-gpu-idle 0 -- ...` 串行执行。完整索引流程采用已记录的例外：
payload=1、不做 L2 bypass、不捕获 CUDA Graph，使用进程级预热和重复计时。
默认 OMP 绑定和 `MALLOC_ARENA_MAX=2` 固定一致，不载入诊断用 LD_PRELOAD。

```bash
python3 -m unittest discover -s experiments/trie_block_build_ablation -p 'test_*.py'
```

`--generate-configs` 可从七个现有 `cpu_special_blocks/*_config_Trie_block.json`
重新生成 28 份配置；已有文件与生成结果不一致时拒绝覆盖，保留人工修改。
同一数据集四份配置除两项优化和 GPU 开关之外必须一致，否则拒绝运行。

八数据集仅重测 GPU＋负载均衡＋topK（包含 Amazon，使用删除辅助预计算后的构建器）：

```bash
python3 experiments/trie_block_build_ablation/run.py \
  --config experiments/trie_block_build_ablation/config_gpu_balance_topk_eight.json \
  --discard-index-data
```

该配置每个数据集测量 1 次，无独立预热；各样本重新构建基础 Trie，按数据集顺序
串行运行。计时完成后只清理本次新建的索引正文，元数据、配置、日志和阶段计时保留。
单方式报告额外列出基础阶段、块内图、块间边、块保存、块阶段和端到端墙钟。

补测新版前三组，并与同版本第四组自动汇总：

```bash
python3 experiments/trie_block_build_ablation/run_graph_only_four_method_comparison.py
```

该入口核对第四组的构建器、计时 worker、输入和公共配置，为本批次复制独立的
构建器和计时 worker，防止后台运行期间其他编译影响测量。八数据集 × 前三组共
24 个进程样本，通过 gpulock perf 串行执行；每个样本重建基础 Trie。
每组每数据集测 1 次，沿用第四组的无独立预热设置。仅清理本次新生成的索引正文。

每个样本结束后，在计时区间外自动更新：

- `tools/pic-draw/trie_block_build_optimizations/current_summary.md` 和 `summary.md`。
- `current_four_methods.csv`、`current_four_methods_wide.csv` 和 `current_speedups.csv`。
- `current_status.json`，记录进度与失败；缺失单元格留空。
- 本批次结果目录的 `four_method_comparison.md` 和 `four_method_comparison.csv`。

32 个组合完成且验证通过后，自动更新三张 PNG/PDF 和 `current_source_manifest.json`。
历史摘要保留为 `historical_summary.md`，历史图片保留在绘图目录的 `historical/`；
四组未齐前不生成新版完整图。运行完成后可手动重绘：

```bash
python3 tools/pic-draw/plot_trie_block_build_optimizations.py --cohort graph_only
```

仅重绘旧版数据时用 `--cohort historical`，并通过 `--data-dir` 和 `--output-dir`
指定另外的输出目录，避免覆盖当前新版汇总。
