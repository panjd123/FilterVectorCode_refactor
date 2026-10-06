# ML-UNG 导师汇报备查

[返回主汇报](ADVISOR_NOTE_CN.md)。这里保留完整网格、结构规律、参数规则与构建组合计时，供讨论时展开。

## 完整网格与观测波动

![全配置相对吞吐](generated_figures/topology/topology_relative_qps.png)

每列按该批次实测最大QPS归一化；红框为事后赢家，灰色为共享实测预算内的NC。14种配置在九档上共126格，95格crossing、31格NC。图中拓扑、入口和阈值口径见[全部数据](generated_data/factorial_equal_recall.csv)。

固定Trie基层与L1=LNG，只将L2从LNG改为Trie，点估计在8/9档较高。但30%、80%两档的两次warm观测范围跨过1；95%虽点估计1.251×，四种跨repeat比值范围为1.004–1.502，不能描述为稳定提高25.1%。这与“加一个L2”的TLT/TL对照不同，后者95%范围为1.072–1.098。

[原始warm计时摘录](generated_data/warm_repeats/warm_repeat_points.csv)保留95个crossing的两次时间、Recall和原文件hash。[比较表](generated_data/warm_repeats/warm_repeat_comparisons.csv)可由仓库脚本重建。四种组合不是四次独立实验；范围不是置信区间。

| 平均选择率 | Plain容量 | 固定1L容量 | 固定2L容量 |
|---:|---:|---:|---:|
| 60.047% | 260,000 | 1,000 | 1,000 |
| 80.024% | 320,000 | 2,500 | 2,500 |
| 95.020% | 320,000 | 2,500 | 2,500 |
| 99.001% | 400,000 | 5,000 | 5,000 |

## 分区与覆盖性质

每个overlay独立从canonical Trie累计未覆盖点数；非根节点累计值严格超过非负整数T时发出block。direct members排除同层已发出的后代blocks。T是发出阈值，不是最大点数。

增大T不会增加block数，且B(T)≤floor(N/(T+1))。当两阈值的block总数相同时，更大阈值的全局覆盖点集包含较小阈值的覆盖点集；但成员分区可能交叉，query授权的direct mass也可能下降。因此block计数单调不推出Recall或耗时单调。证明和例子见论文“Partition and Search Details”。

完整prefix frontier在retag后保留穷尽覆盖，需要完整保留frontier和所有授权block入口、局部有向图从portal可达全部direct members、基层入口可达自身exact group及其terminal prefix后代，并且搜索不剪枝。这个充分条件可用于两种基层拓扑；不等于当前有界ANN的实际Recall保证。

## DRH自动规则

R为局部图度预算，C为跨组度预算。DRH用T+N/T的对称代理选T1≈sqrt(N)，取最近二次幂；下一层阈值乘rho=max(2,round(R/C))，N/T<C时停止。N/T是block数的宽松上界，T不是局部搜索工作量上界，这不是延迟最优推导。

Amazon的N=602453、R=64、C=4给出1024:LNG、16384:Trie。DRH-v1仅在存在合法物理层h≥2时进入多层后端；v2再检查最高合法层的direct mass。规则固定LNG base，没有选择完整系统的全部组件。四个数据集均导出两个overlays，未验证随数据形态选择不同深度的有效性。

Genome/Reviews的DRH-v1为Plain的0.954–1.014×，VariousImg仅0.205×。在另一个独立routing cohort中，v2为其同cohort Plain的0.336×；不能用0.336/0.205当作严格配对改进倍数。五个人工候选使用同一presence gate，单层候选始终执行base；它们没有测试正常启用的一层导航。

## GPU与完整构建预算

Amazon同一声明层级的CPU hierarchy builder为1000.64s，最快hybrid为90.49s，即11.06×。每个后端测量两次warm，并另做资源pass。输出索引的同Recall查询质量尚未验证，查询在CPU上执行。

独立base与hierarchy阶段中位数之和为141.27s，原CPU base-only为190.38s；约1.35×是两个不同构建任务之间的组合计时之比，尚不能视为同质量完整索引的加速比。

## 文献与复现

相关方法的原文依据与写作观察见[文献备查](LITERATURE_NOTES_CN.md)。[实验入口](../../../experiments/multilevel_special/README.md)区分冻结数据重画、现有run汇总和新实验声明；完整计时复现还需要配置中列出的外部向量、标签、查询、索引与raw-run文件。
