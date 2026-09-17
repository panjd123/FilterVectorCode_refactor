# UNG 原始入口与 Plain 入口路由的受控对比

## 结论

在 Amazon x1、同一搜索二进制、同一 UNG 主向量图索引、同一查询与
Ground Truth、100 线程和同一 `neighbor_list` graph backend 下，将入口组生成器
从 UNG 原始的 `cpu_min_super_sets` 替换为
`cpu_bruteforce_els`，在本次完成正式复验的 0.5%--30% 选择率上获得
3.33--19.20 倍端到端加速。

这个结果应表述为 **entry-group provider ablation**，不能表述成两套主向量图之间
的比较。两种方法读取同一份 index；Trie 和 LNG 在这里服务于不同的入口集合计算
步骤，而不是分别构造了两个被独立搜索的 vector graph。

## 实验协议

- Dataset：Amazon x1，602,453 个向量、482,387 个 label groups。
- Query：每个选择率 1,000 条，`K=10`，containment 场景。
- 选择率：0.5%、1%、5%、10%、30%。
- Recall 门槛：0.5%--10% 为 0.90，30% 为 0.87。
- 参数选择：coarse screen、refined crossing、formal 三阶段；每个方法采用其
  measured repeats 全部通过门槛的最小已测 `Lsearch`。
- 正式测量：1 个 cold repeat（只保留作审计）加 6 个 measured repeats。
- 端到端数值：1,000-query batch wall time 的 warm median；加速比置信区间是
  100,000 次 independent bootstrap 得到的 median-ratio 95% CI。
- ELS/graph 数值：并发查询的 per-query CPU time warm median，仅用于阶段间相对
  对比；不能与 batch wall time相加，也不能据此计算端到端耗时占比。
- ELS query-result reuse 被关闭；binary snapshot SHA-256 为
  `2e69823e43f32b4bb2531a9cc99ee6ef6f50a6f144112987a44d3139887edb26`。

## 正式结果

| 选择率 | UNG L / min Recall | Plain L / min Recall | UNG batch ms | Plain batch ms | 端到端加速 | 95% CI | ELS 加速 | Graph 加速 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5% | 1680 / 0.9117 | 1400 / 0.9019 | 702.83 | 45.47 | 15.46x | [12.57, 31.27] | 103.78x | 1.04x |
| 1% | 1920 / 0.9093 | 1600 / 0.9039 | 360.52 | 59.14 | 6.10x | [3.73, 11.34] | 25.29x | 0.86x |
| 5% | 2500 / 0.9030 | 2000 / 0.9054 | 2560.18 | 133.35 | 19.20x | [13.42, 28.85] | 65.45x | 1.59x |
| 10% | 24000 / 0.9013 | 15000 / 0.9008 | 2355.75 | 708.27 | 3.33x | [3.21, 4.24] | 16.85x | 1.36x |
| 30% | 61200 / 0.8712 | 17000 / 0.8707 | 7842.56 | 1541.66 | 5.09x | [5.02, 5.22] | 30.53x | 3.68x |

这些是相同目标门槛下、各自最小已测可行 L 的比较，不是数值完全相同 Recall 的
插值比较。0.5%、1% 的 UNG Recall 余量更大，因此端到端加速存在一部分保守偏差；
论文中应同时列出实际 Recall，不能只列加速比。低选择率的 CI 较宽也说明运行方差
较大，应保留区间而不是只报告点估计。

## 加速来源

### 1. 已由源码和计时共同支持：入口集合计算更便宜

原始 UNG provider 先在 label Trie 中枚举满足 query labels 的 super-set candidates，
再按 label-set size 排序或桶化，并反复执行 `std::includes` 以得到 exact minimal
super sets。该路径输出 `exact_minimal=true`。

Plain provider 使用预构建的 inverted label bitsets 做按位交集，在
`query_label_count ... query_label_count + 2` 的 bounded frontier 中选择最多 8192 个
groups，再利用 LNG descendant Roaring sets 消除已覆盖 descendants。该路径输出
`coverage_correct=true`、`exact_minimal=false`。

正式数据中 ELS 阶段加速为 16.85--103.78 倍，且所有选择率都显著快于原始
provider。这是端到端加速最一致、最直接的来源。

### 2. 高一些的选择率出现第二个来源：更小 L 和更少图搜索工作

| 选择率 | Plain/UNG entry 数 | distance calcs 变化 | node visits 变化 | 解释 |
|---:|---:|---:|---:|---|
| 0.5% | 2.45x | +10.73% | -8.64% | ELS 主导，图工作近似持平 |
| 1% | 3.24x | +12.84% | -6.64% | 更多入口使 graph 阶段慢 16.7% |
| 5% | 1.43x | +16.65% | +2.96% | ELS 主导；graph 加速与计数器不完全同向 |
| 10% | 3.51x | -11.59% | -16.94% | L 从 24000 降到 15000，图阶段开始获益 |
| 30% | 2.21x | -33.61% | -38.81% | L 从 61200 降到 17000，graph 获得 3.68x 加速 |

Plain 并不是通过返回更少的 entry groups 获胜；它在全部五个 workload 中都返回
更多入口。中高选择率的额外收益来自这些入口改变了 Recall--L 曲线，使目标 Recall
能在更小 L 达到，从而减少后续 graph search 工作。

1% 是重要反例：Plain 的 graph 阶段为 0.86x，即约慢 16.7%，但 ELS 快 25.29x，
所以端到端仍快 6.10x。5% 下 graph time 加速而 distance/node counters 没有下降，
说明计数器不能完整解释单次访问成本和并发调度；这里应标为现象，而不应强行给出
单一因果解释。

## 可以与不可以从本实验推出的结论

可以推出：

1. 对当前 Amazon workload，bitset + bounded frontier + LNG descendant coverage 是比
   原始 Trie exact-minimal provider 更高效的 entry routing 实现。
2. 主要收益是入口计算成本下降；10% 和 30% 还叠加了更小 L 带来的图搜索收益。
3. Coverage-correct、非 exact-minimal 的更大入口集合在这些 workload 上没有破坏目标
   Recall，反而改善了部分 Recall--L trade-off。

不能推出：

1. 不能推出“Trie-built vector graph 优于 LNG-built vector graph”或反之，因为主图
   artifact 没有改变。
2. 不能推出 Plain provider 对所有数据集和所有选择率都更快；60%--99% 尚未完成
   同等正式协议，高选择率只能另行做受控扩展性实验。
3. 不能将 `coverage_correct` 等同于 `exact_minimal`；两种 provider 的输出语义不同，
   这里只验证了搜索质量门槛和性能，而不是入口集合逐项相等。

## 复现入口

```bash
python3 experiments/multilevel_special/generate_ung_plain_comparison.py --phase screen
python3 experiments/multilevel_special/experiment_cli.py run experiments/multilevel_special/config.ung_plain_screen.json
python3 experiments/multilevel_special/generate_ung_plain_comparison.py --phase crossing
python3 experiments/multilevel_special/experiment_cli.py run experiments/multilevel_special/config.ung_plain_crossing.json
python3 experiments/multilevel_special/generate_ung_plain_comparison.py --phase formal
python3 experiments/multilevel_special/experiment_cli.py run experiments/multilevel_special/config.ung_plain_formal.json
python3 experiments/multilevel_special/experiment_cli.py validate experiments/multilevel_special/config.ung_plain_formal.json
python3 experiments/multilevel_special/analyze_ung_plain_attribution.py experiments/multilevel_special/config.ung_plain_formal.json
```

原始实验产物位于 `runs/ung_plain_formal_amazon_x1/`，不纳入 Git；可提交的配置、
执行器、validator、汇总与归因脚本位于 `experiments/multilevel_special/`。
