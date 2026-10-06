# C2 source-only 再生与读者入口复核

**结论：文档给出的两个 source-only 入口均可运行；warm comparison CSV、TeX 和 manifest 与发布文件逐字节一致。未发现阻断再生的文档命令或相对链接错误。** 有两处低成本的说明完善建议，见末尾。

本次只读取 `/tmp/filtervector-paper-18h-20261006` 中的源码、发布输入和文档，在临时目录写入再生输出。未修改仓库、论文或脚本，未运行 query/build benchmark；远程 full finalizer、LaTeX 编译和渲染由主线程处理，不属于本次通过结论。

## 实际执行

临时输出根目录：

`/var/folders/7s/5ws1jcr15r18t9m_xh8cg5f40000gn/T/mlung-c2-source-repro-sq5npuej/`

从仓库根目录执行实验 README 中的命令，仅将输出参数替换为以上目录内的 `warm/`、`topology/`。设置 `PYTHONDONTWRITEBYTECODE=1`，避免生成仓库缓存文件。

| 入口 | 执行结果 | 产物核对 |
|---|---|---|
| `summarize_warm_repeat_ranges.py <published points.csv> <factorial.csv> <temp>/warm` | 默认本机 `python3` 成功；95 crossings、35 comparisons | comparison CSV、9 行数据的 TeX、summary manifest 均与 `generated_data/warm_repeats/` 发布文件逐字节一致 |
| `plot_topology_factorial.py <factorial.csv> <temp>/topology` | `/tmp/filtervector-paper-tools/bin/python` 成功；126 格、95 crossings、31 NC | 生成两张 PDF、两张 PNG 和 manifest；输入 hash、网格顺序、选择率、Recall 门槛和归一化定义与发布 manifest 一致 |

默认本机 `python3` 的绘图首次尝试因缺少 NumPy 失败。实验 README `61–62` 已明确要求 NumPy 和 Matplotlib ≥3.6，因此这属于本机依赖缺项，不是文档命令错误。改用主线程提供的已安装环境后成功。

绘图文件不与发布图片字节相同：发布 manifest 记录 Matplotlib 3.10.7，本次为 3.11.2。两个 manifest 除输入绝对路径、Matplotlib 版本及输出 hash 外全部一致。本次确认数据与声明语义再生，**没有声称跨版本渲染字节一致**，也未代替主线程做 PDF 版面审查。

## Warm 再生的精确证据

- `warm_repeat_comparisons.csv` SHA256：`6011c3c85613eeee14dd31d606e805c0c2c3f1671879953cc2b847b899f11682`
- `generated_warm_repeat_ranges.tex` SHA256：`d894011a5e3103de0455d8d3b26bad314940a2ebb694707aaf4f858a2165629f`
- `summary_manifest.json` SHA256：`7f035724e1947e58aa5837905ef8d60dc64838b7a29fa22ea704e6fbb34963fb`
- 冻结 factorial SHA256：`9e56a1ed859152382b829e612c5e62ff4f90d50a5ce3b9a6a3e8230c426b6505`
- 发布 points SHA256：`9410e2b3348e927c993f20115899c7fde8937d3031db87a74454137bd11b9eea`

脚本对完整14×9网格、95个已选 crossing 的 method/Lsearch、两次warm Recall、median-time QPS、CV和选择率进行核验；35行比较的三个数值列和 overlap 标记独立比对全部一致。TeX 由 `main.tex:31` 从 `generated_data/warm_repeats/generated_warm_repeat_ranges` 加载，`sections/measurement_details.tex:46` 调用对应宏，路径闭合。

脚本明确限定冻结 Amazon 每批1,000条查询，QPS为 `1,000,000 / median(Time_ms)`。范围是四种跨repeat比率的min/max，不是配对统计、四次独立实验或CI。这与实验README `76–89`、paper README `94–98`、主汇报 `58–60`、备查 `11–13` 相符。来源hash是发布输入与先前提取的证据；source-only命令不重新读取远程raw timing，其原始文件核查已在前一次定向数值审计中完成。

## 读者入口与方法口径

- 主汇报→论文、主汇报→备查、备查→主汇报/文献/完整数据/图片/warm数据/实验入口，以及实验README中当前论文/结果/历史报告的本地相对链接均解析成功。
- 备查 `46` 的 `../../../experiments/multilevel_special/README.md` **正确**。实际父目录为 `/private/tmp/filtervector-paper-18h-20261006/docs/papers/multilevel_ung`，解析目标为 `/private/tmp/filtervector-paper-18h-20261006/experiments/multilevel_special/README.md`，存在。先前口头心算提出的四层路径建议已撤回，不能采用。
- `AUTHORITATIVE_BUILD_PROTOCOL.md` 是已告知的本地snapshot缺项；远程版本存在，不作为断链报告。本次未再下载它或扩大到构建协议审查。
- 主汇报固定四配置表明确采用 ungated，TLT来自Amazon开发集观察；备查另外说明DRH的h≥2 gate、固定LNG base及单层人工候选退化到base。没有把固定配置结果与DRH结果拼接。
- 阈值是未覆盖整数质量严格超过T时发出block，不是最大块大小；实验JSON要求正且递增的阈值，备查数学定义允许非负整数T。这是实现配置与抽象定义的范围区别，当前文字不冲突。
- Recall是每次warm的批次平均Recall≥0.90，配置实际Recall可不同；NC与profile timeout分开。主汇报明确阶段均值不能相加替代并发batch时间。固定2L/1L与TLT/TLL范围区别、95%宽范围及非CI限定均与已核验数据一致。
- source-only、现有raw-run finalizer、需要外部数据的新实验在实验README中分开，旧formal生成器和历史promotion语义也有明确标记。

## 两处非阻断说明建议

1. **实验README `157–159`：**“sequential method runs are independent samples”比实际已知条件强。顺序执行只说明不能按相同repeat编号视为配对，不能保证统计独立。建议改为“sequential method runs are summarized as unpaired observations; statistical independence is not established by the schedule.” 与后文四组合非独立实验的限定保持一致。
2. **备查 `30–36`：**README说备查包含threshold rules，但该段虽给出T1/rho/停止规则和Amazon具体计划，未写如何选择每层LNG/Trie，也未明确v2 direct mass的门槛。可补一行：`N/T_l > R`选LNG，否则Trie；v2比较最高合法h≥2的direct mass与下一未物化阈值`T_{m+1}`。现有文字没有给出错误阈值，这是自包含程度的补充。

本次没有发现需要修改再生脚本的阻断错误。实验缺口仍按现有材料保留：两次warm仅为观测，外部基线、跨数据集、加载内存/队列代价和加速构建输出质量尚不足；source-only成功不改变这些边界。
