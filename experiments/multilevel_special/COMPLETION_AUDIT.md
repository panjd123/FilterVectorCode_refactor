# 多层 Filtered-ANN 权威实验完成审计

本文件把当前用户目标映射到可检查的实现、原始实验、聚合产物和论文内容。
历史六档结果、修复前 binary 和部分 screen 均不能证明当前目标完成。只有下表
所有必需项均有当前证据并通过对应 validator，任务才可标记完成。

## 固定口径

- 开发数据集：Amazon x1，602,453 points、768 dimensions、482,387 groups。
- 选择率：0.499249%、0.903055%、5.038117%、9.906615%、30.027242%、
  60.047491%、80.023699%、95.019781%、99.000600%。
- 查询：每档 1,000 queries，K=10，100 threads，exact containment GT。
- crossing：所有 warm repeats 的 Recall@10 均不低于 0.90 的最小实测
  Lsearch；不插值、不外推、不按延迟回选更大的 L。
- performance：1 cold + 15 warm；profile：1 cold + 3 warm；screen 仅用于
  确定 refinement 区间，不进入最终性能主表。
- 公平比较固定 data/query/labels/GT/K/threads/binary/backend。自动 DRH 和
  held-out manual grid 使用相同 require_upper_authorization gate。

## 目标与证据

| 显式要求 | 完成所需的权威证据 | 当前状态 |
|---|---|---|
| 任意多层统一结构 | hierarchy、base/per-layer topology、entry strategy、routing 四维独立；候选只扫描自身层边；无隐式晋级或 edge fallthrough | **实现完成**：experiment_core.py、uni_nav_graph_search_backend.cpp 及 C++/Python 回归测试 |
| Trie vs LNG 公平消融 | Amazon 九档、六个零层 topology-entry 组合，以及固定 entry 下的逐层 topology 曲线；同一 immutable performance binary | **采集中**：44 methods x 9 workloads screen 正在串行运行 |
| 0/1/2 层与参数消融 | 每档 plain、best one-layer、best two-layer、ungated DRH、gated DRH；同时给单一固定配置的九档 geomean | **脚本完成，结果待采集**：summarize_depth_ablation.py |
| Recall-QPS 曲线 | screen 的全部离散实测点，缺失 case 必须失败；PDF/PNG 与输入 hash manifest；完整曲线嵌入最终论文 | **脚本与论文接口完成，结果待采集**：plot_authoritative_recall_qps.py；paper generator 拒绝 partial/stale plot manifest |
| 查询阶段 breakdown | entry-group、entry-point setup、完整 block authorization、graph search、residual，单位统一为 ms/query，并验证 closure | **采集链完成，正确口径待重跑**：旧 immutable binary 未把 exact gate 计入 authorization；源码已修复 |
| 查询工作量 breakdown | visited points、base/special intra/special inter edges、entry/graph/total distance calculations，以及 layered-path activation rate | **采集链完成，结果待 profile**：summarize_selection_sweep.py |
| 自动决定层数、阈值、topology | DRH 仅从 N、R、C 推导，无 query distribution、latency 或 Recall 校准；exact gate 只读当前 predicate 和持久化 root labels | **方法与协议完成**：derive_static_hierarchy.py、AUTOMATIC_HIERARCHY_PROTOCOL.md |
| held-out 自动方案 vs 手工 oracle | Genome、Reviews、VariousImg；manual grid 在看结果前冻结；自动与 35 个 manual candidates 使用同一 gate；报告逐 workload 和单一配置 oracle gap | **配置/预检完成，查询待运行**：summarize_heldout_oracle.py |
| GPU 建图 | kernel 与端到端 wall time 分开；1 cold + 至少 5 measured；CPU/GPU 交错；RSS/GPU memory/index bytes；下游 Recall 复验 | **runner 完成，权威 timing/resource/quality 待运行** |
| 完整可展示报告 | 方法、baseline、数据集、Recall crossing、QPS/CI、适用区间、负结果、机制解释和限制均由当前证据生成 | **同源自动生成接口完成，数字待 validator**：`MULTILEVEL_SPECIAL_BLOCK_AUTHORITATIVE_RESULTS_CN.md` |
| SIGMOD/VLDB LaTeX | 算法定义、正确性、复杂度、伪代码、实验方法、表图、讨论、限制、引用完整；无 pending；可编译 | **正文与原子结果接口完成；placeholder 骨架已通过 Tectonic 0.15.0 musl 编译，最终数字版验收待完成** |

## 必须通过的最终门禁

1. validate_selection_sweep.py 分别通过 Amazon screen、crossing、formal、
   instrumented profile，以及三个 held-out formal 和 build-quality formal。
2. Amazon formal 的 equal_recall_conservative.csv、depth_by_workload.csv
   和 depth_global.csv 与 config 的 method/workload/L/结构元数据完全一致。
3. Instrumented profile 从 formal 的同一实测 L 派生，manifest 中所有 case 使用
   新 binary；BlockAuthorizationTime_ms 包含 exact gate 与 coverage，stage
   closure 不超过 1e-6 ms/query。
4. Held-out policy 从记录的 N/R/C 重推导后与 formal config 的自动层级一致；
   35 个冻结 manual 候选、自动方法、ungated ablation、统一 gate 和 workload
   元数据全部匹配。输出恰好覆盖 Genome 2、Reviews 2、VariousImg 1 个
   workload；允许并显式保留 automatic_no_crossing，不得删掉失败点。
5. 四个 build manifest 全部完成；每个 timing profile 至少 5 个 measured
   repeats；GPU case 有外部锁或至少三次连续 idle-preflight 证据。
6. generate_authoritative_paper_results.py 成功运行并原子替换
   generated_results.tex；生成文件记录所有输入/config/binary/policy hash，且
   没有 pending。
7. Python 实验测试、C++ tests、git diff --check 和 LaTeX 编译全部通过。

## 当前不可使用的证据

- runs/multilevel_selection/ 及旧六档 paper_results.csv 仅为历史结果。
- Trie 空谓词修复前 binary 3ae4fe9a... 的所有数据均已 invalidated。
- 当前 performance binary 4c99a51c... 的 Recall/QPS 可用于本轮主表，但其
  profile authorization 子项口径不完整；该 profile 只能作诊断。
- 任何固定 L 延迟、mean-Recall crossing、插值 crossing、缩短 query 数或减少
  正式重复的结果均不能替代上述权威口径。
- GPU kernel microbenchmark 不能替代 base-plus-hierarchy 端到端 build wall time。
- 无锁机器上的 idle preflight 不能表述为独占 GPU 测量。

## 当前执行状态

- Amazon 44 x 9 screen：运行中，由既有 runner 和 watcher 串行推进。
- Amazon crossing/formal/profile：等待 screen validator。
- Held-out：配置与冻结 manual grid 已完成，等待 Amazon query gate。
- Build：等待 query gate；continue_after_query.py 将 held-out 与 build 串联。
- 完整 authorization profile：run_authoritative_instrumented_profile.py 已就绪，
  会在所有实验进程退出后使用独立 build 目录执行。
- 论文结果：generated_results.tex 保持 fail-closed placeholder；摘要、结论和
  Results 由同一生成文件统一更新。
- 最终 provenance：finalization manifest v2 会逐项核对并记录结果 CSV、配置、
  冻结 policy、validator、query/build manifest、论文源文件、Tectonic、编译日志、
  generated_results.tex 与 PDF 的 SHA-256；输入在编译期间变化或日志含 fatal、
  undefined citation/reference、BibTeX warning 时拒绝发布。
- 论文编译：2026-09-27 使用 Tectonic 0.15.0 musl 对当前 placeholder 骨架执行
  `tectonic main.tex --keep-logs --keep-intermediates` 成功并生成 PDF；这仅证明
  LaTeX 工程可编译，不替代最终 `generated_results.tex` 生成后的再次验收。

## 完成判定

当前目标**尚未完成**。代码、协议和论文正文骨架已具备，但 Amazon formal、
held-out oracle、GPU build 和修复后 profile 的权威数据尚未全部生成并验证。
