# ML-UNG 截止版实验报告

> 证据级别：screen-level。每点 1 次 cold + 2 次 warm；100 个 query worker；固定查询集；Recall@10 crossing 是两次 warm 都达到 0.90 的最小实测 Lsearch，不插值。人工比较只含 5 个预注册替代方案，不称为 35-case oracle。

## 方法与自动参数

ML-UNG 将三个维度解耦：层数与阈值、每层 group topology（LNG 或 Trie）、入口组方法（原始 LNG、优化 LNG、Trie）。候选只扫描其 activation level 拥有的边，不跨层混扫，也不隐式晋级。DRH 不读取查询分布、Recall 或延迟：$T_1$ 取最接近 $\sqrt{N}$ 的二次幂，$\rho=\max(2,\mathrm{round}(R/C))$，$T_{l+1}=\rho T_l$，当 $N/T_l<C$ 时停止；预计 block 数大于 $R$ 时用 LNG，否则用 Trie。DRH-v2 再以 $T_{L+1}$ 作为最高授权层的 direct-mass gate。

数据集包括 Amazon（602,453 个 768 维向量，九个选择率档位）以及 held-out 的 Genome（108,077 x 512）、Reviews（288,065 x 384）和 VariousImg（758,935 x 512）。所有查询使用 $K=10$ 和精确 containment ground truth。

## 核心结论

- Amazon 在 60%--99% 选择率出现明确多层收益，最佳已测加速为 27.99x；30% 的多层方法在共同预算内未 crossing。
- 0 层 Trie 在 0.5%、1%、5% 分别达到 6.03x、2.77x、18.43x，但 10% 仅 0.60x，说明收益来自 group topology 与入口覆盖的组合，而不是层数单调性。
- DRH-v1 在 Genome/Reviews 为 plain 的 0.954--1.014x，且为最佳人工配置的 0.980--1.008；VariousImg 只有 plain 的 0.205x，是必须保留的反例。
- DRH-v2 当前结果：Genome 3.365%: v2/plain=0.992x; Genome 6.292%: v2/plain=1.005x; Reviews 0.200%: v2/plain=1.007x; Reviews 4.115%: v2/plain=0.972x; VariousImg 10.252%: v2/plain=0.336x.
- DRH-v2 将 Genome 两档恢复到 plain 的 0.992--1.005x，也把 VariousImg 从 v1 的约 0.21x 提升到 0.336x；但它在 Reviews 4.115% 过度回退到 0.972x，说明该 gate 能限制灾难性开销，却仍不能保证逐 workload 单调更优。
- GPU base-stage 最佳初步时间可从论文构建段落读取。hierarchy cold sidecar 中 CPU 为 986.12 s，hybrid GPU intra 为 85.75 s（11.50x），hybrid GPU intra+inter 为 136.08 s，full GPU 为 105.83 s；这些均为单次 cold screen，repeats 尚未完成，因此不宣称端到端 GPU 多层构建加速。

## Amazon：0/1/2 层与 topology

| Selectivity | Plain QPS | 0-layer Trie | 1-layer T1=1024 | 2-layer ungated | 2-layer gated |
|---:|---:|---:|---:|---:|---:|
| 0.499% | 2476.35 | 6.03x | NC | NC | 0.86x |
| 0.903% | 4191.04 | 2.77x | NC | NC | 0.99x |
| 5.038% | 544.93 | 18.43x | NC | NC | 0.85x |
| 9.907% | 450.17 | 0.60x | NC | NC | 0.64x |
| 30.027% | 52.77 | NC | NC | NC | NC |
| 60.047% | 4.07 | 0.62x | 13.89x | 15.09x | 14.40x |
| 80.024% | 1.80 | 0.33x | 20.33x | 27.57x | 24.22x |
| 95.020% | 1.50 | 0.33x | 27.99x | 26.46x | 26.70x |
| 99.001% | 0.79 | 0.56x | 19.57x | 21.34x | 18.78x |


NC 表示在该 workload 的实测共同预算内未达到 Recall@10 >= 0.90，不表示算法无法在任意更大预算下达到该质量。低选择率下 gated 两层为 plain 的 0.86x、0.99x、0.85x、0.64x，因此“额外层不用时必然无成本”在当前共享入口、授权和队列实现上不成立。

## 自动 DRH 与人工调优

| Dataset | Selectivity | Plain QPS | DRH QPS | DRH/plain | DRH/best manual |
|---|---:|---:|---:|---:|---:|
| Genome | 3.365% | 153.65 | 146.52 | 0.954x | 1.008 |
| Genome | 6.292% | 85.45 | 83.88 | 0.982x | 1.007 |
| Reviews | 0.200% | 19319.56 | 19594.52 | 1.014x | 0.997 |
| Reviews | 4.115% | 354.78 | 356.35 | 1.004x | 0.980 |
| VariousImg | 10.252% | 2657.04 | 543.74 | 0.205x | 0.218 |


按 dataset 固定一个配置后，DRH/最佳人工配置的几何平均 QPS 比分别为 Genome 1.013、Reviews 0.989、VariousImg 0.218。前两者说明无需查询校准的规则可以接近小型人工候选集；VariousImg 说明它尚不是普适的自适应最优规则。

## Detailed profile 机制解释

| Dataset | Method | Graph ms/query | Visited | Scanned edges | Distances | Layer active |
|---|---|---:|---:|---:|---:|---:|
| Genome | plain | 0.124 | 123.8 | 154.3 | 4538.9 | 0.0% |
| Genome | DRH-v1 | 0.199 | 185.1 | 591.8 | 4601.4 | 1.8% |
| Reviews | plain | 6.756 | 4415.8 | 15126.2 | 8239.3 | 0.0% |
| Reviews | DRH-v1 | 6.504 | 3500.1 | 25605.4 | 7312.0 | 18.2% |
| VariousImg | plain | 35.098 | 18656.0 | 119045.5 | 18673.1 | 0.0% |
| VariousImg | DRH-v1 | 172.065 | 27356.1 | 251144.0 | 27393.7 | 37.4% |


Reviews 的较宽 workload 中 DRH 将 visited 从 4415.8 降至 3500.1、distance calculations 从 8239.3 降至 7312.0，但扫描边从 15126.2 增至 25605.4；其图时间仍从 6.756 ms 降至 6.504 ms。VariousImg 则把 visited 从 18656.0 增至 27356.1、扫描边从 119045.5 增至 251144.0，graph time 从 35.10 ms 增至 172.06 ms，直接解释负收益。Genome 的 ELS 占总时间主体，图阶段从 0.124 ms 增至 0.199 ms，总体 QPS 略降。

## 构建证据

自动 DRH 的 CPU sidecar wall time 分别为 Genome 44.2 s、Reviews 120.4 s、VariousImg 633.2 s。独立 base-index timing 的原始 CPU 中位数为 190.61 s，最快 GPU profile 为 53.18 s（3.58x），但这只证明 base stage。另一个 Amazon hierarchy cold sidecar 完成了 CPU 986.12 s、hybrid GPU intra 85.75 s、hybrid GPU intra+inter 136.08 s 和 full GPU 105.83 s；最快完成项相对 CPU 为 11.50x。该横向比较只有单次 cold run，且不含 base-index construction，因此只能作为 GPU hierarchy 可行性证据，不能当作重复测量的端到端构建加速比。

## 学术边界

当前结论不包含正式置信区间，不把 light-stats 的缺失边计数解释为 0，也不把 CPU sidecar 构建或 base-only GPU timing 当作完整多层 GPU 构建结果。完整 396-case 生成器仍保持 fail-closed；本报告来自单独、显式缩小的 deadline evidence contract。所有负结果与 NC 均保留。
