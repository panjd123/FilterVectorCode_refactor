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
