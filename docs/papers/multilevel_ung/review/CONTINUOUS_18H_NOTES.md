# 18小时论文与读者材料优化记录

本文件用于恢复每轮输入、观察和修订决定，不作为导师汇报正文。
本次至少18小时的时限见同目录CONTINUOUS_18H_MANIFEST.json。

## C0 输入与独立性

- 论文基线：14c213e，冻结PDF SHA见manifest。
- 新建cold_c0_sigmod与cold_c0_vldb，均fork_turns=none。
- 唯一论文输入：/tmp/mlung-cold-c0-20261006/paper.pdf。
- 不提供旧审稿、旧得分、整改清单；要求完整论文评审，非定向验收。
- 完整评审区分新颖性/价值、技术可靠性、证据和表达；不设目标分数。

## 导师材料的首个调整

移出自评与修订历史，保留实验来源事实；删除已不在论文讨论的动态维护行。
这些自评信息保存在历史POLISH_ROUNDS.md与下方归档中，不进入首次阅读包。

### 移出的旧状态说明

## 8. 本轮写作与评审状态

主线固定为“入口与连接共同设计 → 前缀授权跨组导航 → 有界队列中的实际收益和代价”。删去只防御质疑的重复句，正文每段尽量只承担定义、推导、实验证据或解释之一。结果部分重新按 factorial、阶段成本、自动转移、构建成本组织。

三轮修改与独立模拟审阅已完成。第三轮 SIGMOD/VLDB 均确认表格布局、测量术语和回退行为说明通过，主线、方法与人类可评审性均为 4/5；实验支撑仍为 2/5。当前 PDF 共 23 页，结论在第 11 页；完整结果与测量定义在附录。手调结果已标明单层方案实际执行 base fallback。每轮独立意见与处理记录见 `review/POLISH_ROUNDS.md`。模拟审阅不等于导师或会议审稿人认可。

原始 embedding 模型、存储前是否归一化、部分旧 query 文件的生成记录仍未找全；正文按已核实的预处理快照描述，不推测其来源。只读审计见 `review/workload_metadata_audit.json`。

## 待审的结构与展示改进

- 根README和docs索引仍将六月GPU入口/输出边界工作标为当前唯一入口，
  与多层论文及九档权威报告不一致。单独开展文档结构审计，保留历史入口。
- 准备完整126格的QPS归一化图和对应crossing Lsearch图，只读取冻结CSV。
  目标是同时展示优势区间、NC与达到质量门槛所需的搜索预算。
- 从现有CSV可见，高平均选择率的L0[L]需要260k--400k队列容量，而逐档赢家
  为1k--5k。这是后续成本解释的重要观测，尚不能单独归因于队列实现。
- 对postorder阈值分区开展独立数学核查：block数上界、覆盖质量非单调、
  跨尺度direct members非嵌套，以及N/T在DRH中的可解释范围。

## 读者入口与评审 provenance 修订

- root README 与 docs/README 改为当前 ML-UNG 阅读路径，保留历史资料链接。
- 所有本地 Markdown 链接以 remote tracked inventory 检查；原 DataTools 目录
  不在当前 tracked tree 中，入口改为 UNG/codes/tools。
- paper README 与 POLISH_ROUNDS 明确 R1 新建 reviewer、R2/R3 复用历史；
  4/5 是表达评分，证据仍为 2/5，不表示无上下文的总审稿分数。
- 导师说明移除“正文新增”“主表已移除”等修改历史及自评状态，保留测量事实。
- 新图生成脚本只读 frozen factorial CSV，完整保留 95 crossings 与 31 NC；
  正式整合前还需 corruption checks、源 hash 校验及版面选择。
- 待独立核验：LNG-minimal group entry 被 retag 后可能失去 L0 跨前缀路径。
  三个 singleton groups {b},{b,c},{a,b}、Q={b}、T=1 提供候选反例；
  在实现与图可达性核对前，不将其写成已证实的 campaign NC 成因。

## 文档发布与后续工作

- ed4c48c 已在 W300 非强制 push 到 GitHub main，remote ref 回读相符。
- ShareLaTeX 的 ADVISOR_NOTE_CN.md、README.md、review/POLISH_ROUNDS.md
  三份文档先比对上次发布 SHA，再上传并逐字节回读；receipt 位于
  /tmp/mlung18h-doc-sync-receipt.json。TeX/PDF 尚未修改。
- plot_topology_factorial.py 已完整生成126格两幅图；5类坏输入均拒绝。
  输出和验证位于 /tmp/mlung-18h-validated-figures，尚未选入正文。
- 实验 README 由400余行压缩为 current bounded / new experiments / formal
  design 三个阅读用途；历史两级promotion复现报告已显式标注阶段。
- ENTRY_RETAGGING_COVERAGE_WORKING.md 记录三group反例、源码证据和
  候选正面覆盖性质。toy穷举图可达性检查通过，独立理论核验仍在进行。
- cold C0 原始审阅报告仍未返回；不得以旧R3评分预报本轮结论。

远程 Matplotlib 3.10.7 亦完成126格图生成，git diff --check通过。
进一步的待审数学归纳：T1<T2 时对子树定义 block数 b 和上交 uncovered u，
候选不变量为 b1>=b2，且 b1=b2 时 u1>=u2。此不变量可由postorder emission
分情况归纳；若成立，则 block数随T不增，可用实际B(T)作未来结构选择依据。
尚未将此候选性质或未来配置方案写成论文已证明/已测试的主张。

## C1 已编译并进入定向复核

C0原始意见全部归档，两位全文reviewer都建议Reject。C1加入新覆盖分析、
三group图、partition证明附录，固定配置主表与全126格QPS/预算图；
正文、摘要、结论和导师材料对齐。257 tests通过，26页Tectonic PDF，
无undefined refs/citations，CSV SHA仍为9e56a1ed...。新图和固定表已目视检查。
C1冻结输入 /tmp/mlung-c1-review-20261007；返回reviewer做定向复核，不称cold。
较早的两份retagging working scratch已删除，其完整独立审计和枚举脚本保留。

## C1 发布检查点

99bfe4c 已由 W300 非强制推送 GitHub main 并回读确认。ShareLaTeX在全部46个旧文件通过哈希预检后，上传并回读验证63个文件。服务端编译25页，仓库Tectonic编译26页；固定配置表及关键覆盖文本验证通过，两份PDF未发现未解析引用标记。receipt见SHARELATEX_C1_PUBLICATION.json。18小时时间窗口和实验证据缺口保持不变。

## C2：计时范围、数学条件与短汇报

C1定向复核全部返回并原样归档；核心证明通过，补明整数阈值、自身group与ungated语境。全部95个crossing的190条warm Recall和35组范围已由另一个reviewer对原始文件与hash核验。C2新增source-only范围生成器与5项输入/数学测试，完整测试共262项通过。主汇报由6147字符压缩到约4100字符，完整细节保留在ADVISOR_BACKUP_CN.md，定向复核认为适合12–15分钟。

修正了长路径越界、参考文献孤立DOI尾行和附录标题/表格顺序；C2共27页，完整保留原126格实验结果。新的正文说明60/80%的TLT/TL观测范围重叠、95%的TLT/TLL点估计1.251对应宽范围，未称CI或稳定改进。

C3开始只读审计旧query_details。已发现CandSize不是eligible向量数；LIGHT_STATS下的零work counters不能用于扫描量归因。QuerySize分层Recall分析正在进行，不重跑query/build campaign。

C2发布：25afcd7已由W300推送main并回读确认，80份ShareLaTeX源逐字节验证，服务端26页PDF内容检查通过。最终定向PDF复核通过；不产生新的总评分。CSV保留已审计的CRLF原字节，在目录.gitattributes中将CR识别为行结束，避免改变数据hash。

C3已重分组190000条query记录，所有190次warm均值匹配，5%混合批次43个单标签query的TLT Recall仅0.74419。精确coverage审计重算9000条query/3239个唯一谓词，与全部profile一致；base/query无行内重复标签。下一步将这些新派生证据整合到正文与可再生附录，仍不运行ANN benchmark。

## C3：查询组成与子群Recall

C3已写入正文、附录和中文材料。新图同时展示标签数与逐查询精确选择率；完整summary保留14配置、9批次和两次warm。独立实现重算190000条记录、9000条coverage、190个batch均值、570个旧直方图、756条size summary和2268条selectivity summary，未发现数值或join差异；source-only六个产物逐字节一致。

5%批次中的43条是同一{1}谓词；30%批次80--95%区间的226条是同一{1,2}谓词。正文与导师材料已明确这一重复谓词边界，不将它们外推为完整查询族。零Recall计数仅描述选定吞吐点的质量尾部，不用于推断结构原因或子群QPS。

最终远程finalizer通过271项测试并编译28页PDF，SHA为ce12f3edb743bab3fd4600a91557820f54ea6eb7e0fbf3dbf5c0812d336d6ca9；无undefined refs/citations。C3使用返回reviewer做数值和导师材料定向复核，不产生新的全文总评分。18小时时间边界在发布审计前已经满足。
