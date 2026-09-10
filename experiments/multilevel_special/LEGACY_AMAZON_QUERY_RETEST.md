# Amazon 历史查询集兼容性与重测

## 结论

当前 Amazon x1 base/labels 下，`old_query_1000/query_minlen1_cov10k` 可以重算 exact GT 后用于同口径复测。其 1000 条查询的当前平均选择率为 28.610%，不是目录名可以直接推出的固定选择率。

在 `Recall_min >= 0.90`、K=10、100 threads、7 repeats（去掉 cold repeat 0）下：

| 方法 | 阈值 | L | Recall min | warm mean ms | vs plain | vs 固定 T1 单层 |
|---|---|---:|---:|---:|---:|---:|
| 0 层 plain | - | 50,000 | 0.9079 | 5469.83 | 1.000x | 1.742x |
| 1 层 | T1=1k | 22,000 | 0.9027 | 9530.47 | 0.574x | 1.000x |
| 2 层 | T1=1k,T2=4k | 18,000 | 0.9017 | 8442.51 | 0.648x | 1.129x |
| 2 层 | T1=1k,T2=10k | 18,000 | 0.9020 | 7969.53 | 0.686x | 1.196x |
| 2 层 | T1=1k,T2=25k | 18,000 | 0.9013 | 7984.39 | 0.685x | 1.194x |
| 2 层 | T1=1k,T2=50k | 18,000 | 0.9013 | 7873.90 | 0.695x | 1.210x |
| 1 层 | T1=128k | 14,000 | 0.9023 | 3831.68 | 1.428x | 2.487x |
| 2 层 | T1=16k,T2=400k | 14,000 | 0.9143 | 4487.59 | 1.219x | 2.124x |

因此，这个 workload 同时给出两个不同层次的结论：

- 固定 `T1=1k` 时，只增加第二层可带来 1.13--1.21x 的增量收益；
- 各方法独立调优时，单层 `T1=128k` 是本次预声明候选中的最快点，比调优双层快 1.171x。

前者是第二层的局部因果消融，后者才对应当前候选集合中的部署选择。两者不能互相替代。

## 数据集审计

| 目录 | 当前 x1 状态 | 用途 |
|---|---|---|
| `old_query_1000/query_minlen1_cov10k` | 1000 queries，768D；可重算 exact GT | 本轮正式复测 |
| `hybrid_query/query_minlen1_cov10k` | 3000 queries；当前平均选择率 38.035%，旧 profile 的 18.845% 已漂移 | 可重标定后另做实验 |
| `hybrid_query/query_minlen1_cov10k_zipf` | 当前 K=10 下平均仅 0.839 个匹配点/query，482 条零匹配 | 与当前 x1 label 语义不兼容，不做同口径 Recall 实验 |
| `hybrid_query/query_selected_recall_advantage` | 1971 queries；128 条零匹配；按历史方法结果后验筛选 | 仅作诊断，不能作无偏主结果 |

目录名、旧 coverage 或旧 profile 均不作为当前选择率证据；选择率和 GT 均由当前 x1 base labels 重新计算。

## 协议与证据

- Search binary SHA-256：`6fa4082dae77e99b6c1b3c84c75d324d57cc25e0da32149011379453cfa40087`。
- Query binary SHA-256：`e07980a3146cbd57fc2fa78350d1d6aafa77c58a0a8aecf1f0a0cf7e8c3c1ae9`。
- Exact GT SHA-256：`d5c20e27d67c12ef4d9a73dd27f85b4b8de031eadeff299d652a3275fec2e5ec`。
- 8/8 cases、24 个 L 点均通过 sweep validator；最大 stage closure error 为 `2.794e-12 ms/query`。
- 本轮未开启逐结果 filter validator，因此正确性证据是 exact-GT Recall 与既有索引结构验证，不能额外声称本轮检查了所有返回点的标签合法性。
- `gpulock` 不可用；仅做进程协调和运行前 `nvidia-smi` 检查，因此不声称锁级独占。

复现入口：

```bash
python3 experiments/multilevel_special/run_selection_sweep.py \
  experiments/multilevel_special/config.amazon_x1_legacy_cov10k_formal.json
python3 experiments/multilevel_special/validate_selection_sweep.py \
  experiments/multilevel_special/config.amazon_x1_legacy_cov10k_formal.json
python3 experiments/multilevel_special/summarize_legacy_cov10k.py \
  experiments/multilevel_special/config.amazon_x1_legacy_cov10k_formal.json \
  --output-dir experiments/multilevel_special/results_summary/legacy_cov10k
```

Raw output 保留在未跟踪的 `runs/legacy_query_retest_amazon_x1/`；提交中只保留配置、compact equal-recall 表、execution manifest snapshot 和 hash 清单。
