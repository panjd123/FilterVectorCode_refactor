# Agent 看板

## 当前状态

三轮论文修改与 SIGMOD/VLDB 独立模拟审阅完成。两位最终审阅均确认
本轮三组编辑问题通过，可进行实质人类评审；实验证据强度仍为 2/5。
发布产物包括源码、PDF、中文 note 和审阅记录；授权发布目标为
GitHub main 与 ShareLaTeX。

## 已完成

- 主线聚焦入口覆盖、前缀授权与跨组几何导航；GPU/DRH 为支撑性研究。
- 核对路由、seed 去重和裁剪、有界队列、同层边及两层执行例。
- 统一 mL/L0/LL/LT、平均选择率、Encounters、等 Recall 和实际回退行为。
- 删除防御性重复；整理三幅机制图；主要空白浮动页已消除。
- 256/256 tests；Tectonic 编译 23 页，结论第 11 页；无 undefined refs/citations。
- 数值、UNG 算法源码和冻结 factorial 数据未改；未重跑旧 campaign。

## 发布验证

`docs/papers/multilevel_ung/review/VALIDATION.json` 记录本轮源文件与产物。
`review/POLISH_ROUNDS.md` 和六份 polish_round 审阅记录可恢复全过程。
出版用 PDF 与第三轮冻结审阅 PDF 一致；中文 note 补充最终评审状态。

## 剩余科研工作

同协议外部基线、入口受控归因、跨数据集与变深度、GPU 构建质量，
以及支持微小性能差异的更多重复。95% 的 0L-Trie 详细 profile 仍为
3300 秒超时，未将其混作 Recall NC。轻微附录浮动排版留待投稿格式整理。

## 工作范围

仅 `/home/sunyahui/worktrees/FilterVectorCode_multilevel_special`；禁止修改
`/home/graphdb/FilterVectorCode_refactor`。不将模拟审稿当真人认可或录用。
