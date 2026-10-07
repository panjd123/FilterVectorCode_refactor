C2 PDF 定向版面与表述检查

检查对象：`/tmp/mlung-c2-review-20261007/paper.pdf`，27 页。

实测 SHA256：`fdf4f22c3f97cdfa13420b2ac482290955f00d24d65f060cece0efbdd51e167a`，与指定值一致。

**结论：本次指定检查项通过，未发现需要阻止 C2 checkpoint 的版面或表述问题。** 这是延续 C1 的局部检查，不是新 cold review、总体评分或科学证据验收。检查了 pp. 10、13–16 的文本与页面渲染，并查看 p. 9 的 §8.1，以核对 8/9 和 1.251× 的实际正文措辞。未打开其他 review、仓库或运行数据，也未重复主线程的数据和测试检查。

| 检查项 | 位置 | 结果 |
|---|---|---|
| 原 DOI 孤行页 | p. 13 | 已修正。现为参考文献 [17]–[24] 的正常两栏续页，[24] DOI 与条目相邻。页下部留白属于参考文献末页留白，不再是单行占整页。 |
| 长路径越界 | p. 15，A.2 | 已修正。`docs/papers/multilevel_ung/` 与 `generated_data/` 分行显示，未侵入右侧行号区；新增 `generated_data/warm_repeats/` 也正常显示。 |
| 附录标题顺序 | pp. 14–16 | Appendix A 在 Table 7 前；Appendix B 在其分析内容前；Appendix C 在 Table 11 前，章节归属清楚。 |
| 有限预算例子与表相邻 | pp. 15–16，B.2、Table 10 | B.2 的设定在 p. 15 末尾，Table 10 位于紧接的 p. 16 顶部，驱逐过程解释紧随表后。虽跨页，但关系明确，没有无关材料插入。 |
| 新 Table 9 的统计含义 | pp. 14–15，A.1、Table 9 | 明确由每个配置的两次原始 warm measurements 派生四个 cross-repeat ratios；明确不是 confidence bounds。 |
| 8/9 与 1.251× 的正文措辞 | p. 9，§8.1 | 已适当降调为“higher point estimates”，并同时披露 overlap 和最大 point ratio 的宽 observed range，未宣称统计显著或跨新运行稳定。 |

**Observed ranges 的说明正确且足够明确。** A.1 给出等 batch size 下 QPS 比率的四组合包络：

`[min(t_B)/max(t_A), max(t_B)/min(t_A)]`。

这是两组各两个正 batch times 所形成的四个 cross-repeat QPS ratios 的最小值与最大值，公式方向正确。文字进一步说明不假定 paired executions，范围与 1 分离也不构成 significance test 或新 process runs 的稳定性证据。Table 9 图注重复“not confidence bounds”，因此没有把四个派生比率说成四次独立实验或置信区间。

§8.1 的“八个 workload 点估计更高、最大为 1.251×”紧接着指出 30% 与 80% 的两次 warm timing ranges 重叠，最大 point ratio 的 observed range 很宽。A.1 和 Table 9 给出该 95% TLT/TLL 范围为 `[1.004, 1.502]`，并直接说明点排名可能夸大精度。该组合表述已经回应 C1 对小样本差异的疑虑，不能解读为稳定的 25.1% 提升保证。这里未发现新的过强结论。

同一段对 TLT/TL 的描述也保持在观测层面：60% 和 80% ranges 重叠；95% 的四个 cross-repeat ratios 为 `[1.072, 1.098]`；正文再次说明不是 confidence intervals。无需为本次定向检查继续修改措辞。

**其他已修正的相关细节。** Table 3（p. 10）图注已明确 ungated routing、不同 base 的 entry provider，以及最后两列由未四舍五入的 QPS 计算，解决此前用显示值回算产生表观差异的问题。Table 9 同时列出 Plain、TL、TLT 的精确 selected capacities，补充了预算热图只能显示大致量级的不足。重点页面未见文本遮挡、表格截断或路径溢出。

**本次结论的范围。** Table 9 及相关文字是对原始两次 warm 测量的描述性再分析，没有被表述成新增 vector benchmarks。本文档未独立复算 frozen data、95 个 crossing 的 CV 或源 timing hashes，也没有验证主线程报告的测试数量。此前外部基线、主机制跨数据集验证、构建图质量和完整复现输入等科学证据缺口，不因本次版面与措辞通过而关闭。本次没有总体接收建议或评分变更。
