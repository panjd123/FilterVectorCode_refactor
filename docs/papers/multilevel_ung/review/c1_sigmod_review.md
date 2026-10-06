C1 定向修订复核：ML-UNG

复核文件：`/tmp/mlung-c1-review-20261007/paper.pdf`，26 页。

已核对 SHA256：`17457b1362f6791fabc2e183d770690340c21f46668ad4e0c2bf0dcb4a3e7009`。

本报告承接此前 C0 独立审稿 `/tmp/mlung-cold-c0-sigmod.md`，属于 revision review，不是重新开展的 cold review。页码均指本次 C1 PDF。重点检查新增覆盖分析、分区性质、DRH 解释、固定配置结果和图表排版；没有检查作者代码、运行 vector benchmark 或读取其他 reviewer 的意见。

**总体判断：本轮分析修订有效，关键论证未发现新的实质性错误，但尚不足以改变此前因科学证据不足而给出的 Reject 建议。** 三组 retagging 反例准确；Proposition 4 在明确列出的条件下成立；分区单调性证明及例子成立。固定配置主表与容量图使结果更容易评估。本轮应获得“技术解释得到实质澄清”的认可，不能被视为已经修复已测 LNG-overlay 算法、解释实际 NC 的成因比例，或补齐竞争性、泛化和构建质量证据。本次不重新给出一组数值评分。

| C0 问题或本次核查项 | C1 判断 | 仍需区分的边界 |
|---|---|---|
| LNG retagging 的结构性覆盖损失未被说明 | 分析层面已准确回应 | 已测算法仍会发生此损失；没有新实验衡量影响 |
| Prefix entries 能否在 retagging 后保留覆盖 | Proposition 4 提供了正确的条件性充分保证 | 不保证有限队列或现有构建器一定满足条件 |
| 独立分区的计数、覆盖集合与授权质量 | 新证明与反例成立，明显更清楚 | count 单调不推出授权质量、时延或 Recall 单调 |
| DRH 的平方根尺度依据 | 已降为恰当的结构 surrogate 解释 | 没有性能最优性或有效迁移保证 |
| 逐 workload oracle 掩盖固定方案行为 | Table 3 已显著改善 | 固定结构仍来自 development study，且各批次单独选择搜索容量 |
| 大额 QPS 比率与队列预算关系 | Figure 5 和 §8.1 已改善解释 | 没有分离队列实现、几何导航和覆盖因素的因果贡献 |
| 外部基线、主机制跨数据集验证、构建图质量 | 未补齐，原意见保留 | 本轮没有新增相应 benchmark |

**1. 三组反例准确，而且比 C0 评审所给四组构造更简洁。**

§5.3（p. 6）采用 `{b}`、`{b,c}`、`{a,b}` 三个 singleton groups，`a<b<c`，`T1=1`。按 Algorithm 2，`{b}` 子树累计两个点，生成 root 为 `{b}` 的 block；`{a,b}` 所在分支累计一个点，不生成 block，成为 residual。Q=`{b}` 的 LNG-minimal entry 只有 `{b}`。该入口被 retag 到 level 1 后，没有 level-0 seed；overlay 状态又不能使用原来的 base edge，所以 residual `{a,b}` 不可达。

这严格满足构造算法和查询规则，不依赖队列截断、局部图不连通或不够大的搜索预算。Prefix frontier 包含 `{a,b}`，而该组没有授权 owner，所以其 level-0 seed 得以保留。Figure 3（p. 7）正确区分“存在的 base edge”和“L1 状态不能使用的 base edge”，并明确写出所有点都 eligible。图中 grey 的含义是 unreachable，不是 predicate-ineligible，这一点图注已说明。

因此，C0 关于“结构覆盖损失没有被解释”的问题可以在**论证层面关闭**。不过，C1 没有改变该 LNG-minimal-entry 查询路径，也没有提供 coverage-preserving LNG control；“问题得到准确说明”和“测量系统已修复”是两件事。§8.1（p. 9）明确承认现有 screen 没有测量逐查询 exhaustive reachability，不能把 NC 分摊为结构损失和有限预算损失，措辞正确。

**2. Proposition 4 成立，前提也基本写得充分。**

§5.3、Proposition 4（p. 6）的逻辑是：对任意 eligible terminal S，取其第一个 eligible terminal prefix ancestor E。若 E 保留在 base，命题假设的 base 可达路径覆盖 S。若 E 被 retag 到某层的授权直接 owner B，则 `R_B` prefixes E，E prefixes S。S 一定仍位于 B 的 prefix subtree 内；按该层的直接成员划分，它属于 B 或 B 下某个已标记 descendant owner C，而不可能变成“没有 marked ancestor 的 residual”。C 的 root 包含 `R_B`，因此也授权；由于每个授权 block 都有保留的 portal，且 portal 可达该 block 的全部 direct members，S 的全部点可达。

证明不需要不同 overlay 的直接成员嵌套，也不需要 overlay cross edges；针对每个 retagged entry 使用它所选择的一个物理层即可。即便 base topology 为 LNG，只要额外供给完整 prefix frontier 并满足命题中的 base-path 假设，结论仍成立。因此“适用于两种 base topology、但取决于 entry contract”的解释是准确的。

需要继续保留以下边界：

- 这是**覆盖保持的充分条件**，不是 frontier 最小性或必要性的结论。
- 所有授权 block 的 portal 都被保留、无 seed/candidate/edge pruning、局部有向图从 portal 可达所有 direct members，是实质性前提。
- 当前有限容量初始化会裁剪 seed，候选会被拒绝或驱逐，构建器也没有在本轮证明上述局部可达条件。因此命题不能直接转写成“已测 ML-UNG 对所有 query 都完整覆盖”或“必能跨越 Recall target”。

摘要、引言和结论（pp. 1、12）目前基本保留了这些限定，未发现把该命题升级成无条件实际检索保证的表述。若继续压缩正文，建议优先保留这些条件。

**3. 分区性质的新增证明正确，DRH 的解释也更恰当。**

§4.2（p. 4）与 Appendix B.1（p. 15）证明的重点是 block count 单调，而不是 block membership 嵌套。其加强归纳不变量为：低阈值的 subtree block count 不小于高阈值；当 counts 相等时，低阈值返回父节点的 residual set 包含高阈值 residual set。

证明中 `d+e1−e2` 的分情况完整：若 children 的 count 差 `d≥1`，父节点最多抵消一个；若 `d=0`，child residual containment 使 high-threshold-only emission 不可能。最终 count 相等时，要么两边 emission 相同，要么由 `d=1,e1=0,e2=1` 使高阈值 residual 为空。synthetic root 不 emission，因此不变量传到整棵树。对 root residual 取补集，得到“全局 block counts 相等时 represented set 只能增长”。未发现逻辑缺口。

我也手动核对了三个类型的例子：

- `a:2, ab:1, ac:1, abc:2` 在 T=1 和 T=2 都产生两个 block、覆盖六点，但 direct-member partitions crossing，成立。
- `a:1, ab:2, abc:3, ac:2` 在 T=2、4 的 represented mass 分别为 8、5；`a:1, ab:3, ac:3` 在 T=2、6 的 represented mass 分别为 6、7，均成立。
- `a:1, ab:3`、Q=`{b}` 在 T=2 时授权 mass=3，在 T=3 时唯一 block root 变为 `{a}`、授权 mass=0，说明 count plateau 不保持 query-authorized mass，成立。

一个小的数学表述建议：Eq. (5) 的 `B(T)(T+1)≤M(T)` 应明确限定 **T 为非负整数**。本稿实际使用的阈值满足这一条件；若允许实数阈值，strict `>T` 的整数点数下界应写作 `floor(T)+1`。这是条件声明的补全，不影响当前实验或新证明的主要结论。

§6.3（pp. 7–8）现在明确：`N/T` 是 block count 的 loose upper envelope；T 是 emission scale，不能作为最大 block size 或 local-search work 的界；平方根仅提供 structural reference scale。这样使用 `T+N/T` 是可接受的 heuristic 解释。它没有因 Eq. (5) 而变成真实搜索成本函数，文稿当前没有作出这种升级。DRH 的性能迁移失败仍在 §8.3 保留，恰当。

**4. 固定配置主表和 QPS/预算图确实帮助评估，但没有产生新的实验支持。**

Table 3（p. 10）把 0L[L]、0L[T]、1L[T|L]、2L[T|LT] 作为固定结构沿九个 Amazon batches 展开，优于 C0 以逐 workload 最佳配置为主的展示。最后两列把“相对 Plain”和“增添第二层的边际影响”分开，使读者能看到 10% 和 30% 的退化，以及第二层并非总有收益。摘要改成固定 Trie/LNG/Trie 在七批次胜出、30% 仅为 Plain 的 0.482×，与主表一致。

§8.1（p. 9）明确这是 development-dataset 分析，held-out 实验使用另一条固定 LNG-base rule，避免把它包装成 out-of-sample 固定系统结果。Table 3 图注也正确表明各 method 在各批次取其最小 measured crossing capacity。因此“固定配置”在这里指 topology/thresholds 固定，不是所有查询参数固定。这个限定需要保留。

Figure 5（p. 17）把完整 14×9 grid 的 normalized QPS 与 crossing capacity 上下对应，清楚展示 Plain 在高均值批次需要 260k–400k capacity，而固定 2L[T|LT] 只需 1k–5k。分开 LNG-base 和 Trie-base 的分组线、灰色 NC 和两面板相同的 winner 红框都便于比较。它让巨大 QPS 比率的预算背景更透明；§8.1 同时承认 array queue 的容量相关工作，方向正确。

该图仍不能确定“速度提升中多少来自图质量、多少来自队列实现”，也没有为 NC 提供根因分解。现有表中观察到的低百分比差异仍缺少足够重复来判定显著性。这里不应把新的数据呈现称为新的向量检索实验。

两点小建议：一是在 Figure 5 或邻近表中保留关键固定配置的**精确** selected capacities，热图颜色不能替代数值；二是 Table 3 注明比率是否来自未四舍五入的原始 QPS。由于 Plain 在 99% 仅显示 0.79，读者用显示值计算 `257.77/0.79` 会得到约 326.29，而表为 327.562；这完全可能来自原始值精度，不构成我已确认的数值错误，但一句 rounding 注释可以消除疑惑。

**5. 新版有两处明确排版瑕疵，应修正。**

第一，p. 13 除页眉、行号和页码外，只有参考文献 [24] 从 p. 12 溢出的 DOI 尾行 `10.1145/3639324`，造成近乎整页空白。这是显眼的 pagination 问题，不是内容需要一页。应调整参考文献断行或相关浮动体布局，使该条目完整留在正常参考文献页。

第二，p. 14 Appendix A.1 的长 monospaced 路径 `docs/papers/multilevel_ung/generated_data/` 超出正文右边界并伸入右侧红色行号区。应允许路径内部换行，或把它单独放到可换行的位置。复核中按正文边界提取文本会截断该路径，render 则确认它实际越界，故不是单纯 PDF extraction 假象。

其余重点页面基本清楚：p. 6 的反例与 Proposition 4 在同一栏连续呈现，p. 7 Figure 3 易读，p. 10 固定配置表足够清晰，p. 15 的 partition proof 虽信息密集但没有发现遮挡。Figure 5 放在附录使读者要从 §8.1 翻到 p. 17；考虑它在解释大比率中的价值，可以优先于部分次要 construction material 进入正文，但这属于编辑建议。

**6. 可复现性说明有改善，完整复现实验仍存在边界。**

Appendix A.1（p. 14）现在提供了“source repository”的可点击仓库链接，指出 frozen CSV 的文件名和所含 selected capacity、minimum warm Recall、QPS、status、method identifier，解释 source-only plotting 与 existing-run finalization 的差别。这比 C0 只有 manifest/hash 的表述具体，应予认可。本次仅确认 PDF 内链接和说明，没有打开仓库或核查 CSV 内容。

文稿也明确声明 vector、label、query、index 和 raw-run 等大型输入不随 source repository 分发。因此可重画 frozen summaries 与可重跑计时仍不同。现阶段不能把 A.1 视为端到端 reproducibility 已验证；同样，不应继续说“PDF 完全没有 artifact access link”。

**7. 原科学证据意见的保留范围。**

以下问题未被本轮新增分析解决：

- 缺少竞争性外部 filtered-ANN 方法和高效 exact-filter/scan baseline，无法判断对真实强基线的优势。
- 最主要的 prefix-frontier/Trie-base/block 组合仍仅在 Amazon 上系统验证；held-out 三数据集试验针对固定 LNG-base DRH。
- selectivity mixture、predicate composition、频繁 singleton/empty predicates 与 batch-mean Recall 的混杂仍在；没有 subgroup recall 结果。
- 没有量化结构覆盖损失在实际 NC 中的比例，也没有检验保留必要 base entries 后的速度与质量。
- GPU builder 输出图之间的 query-quality equivalence 尚未验证；loaded-index size、per-worker workspace 与完整构建/检索质量折衷仍缺失。
- 固定拓扑是更清楚的 development comparison，还不是独立 workload 上验证的部署选择规则。

因此，本轮值得肯定的 delta 是：**明确承认并解释 retagging 的结构性风险，给出 prefix-entry 合同下的正确充分条件，厘清 partition 与 DRH 的数学含义，并更诚实地展示固定方案及搜索预算。** 我不会再将这些已修正的解释缺口当作“作者未回应”。但当前 PDF 尚未提供改变竞争性与泛化判断所需的新证据，原 Reject 建议的主要依据仍成立。对新增反例、Proposition 4 和分区证明正确性的判断置信度较高；对未公开原始运行数据的精确预算和计时比率，本复核仅评价 PDF 中的呈现与内部一致性。
