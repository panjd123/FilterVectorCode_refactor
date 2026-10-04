# Agent 看板

分支：`codex/multilevel-special-block-20260905`
本轮起点：`fe46b43`；论文修订提交：`e957dcb`。正文、验证与发布已完成。

## 当前交付

- 论文主线：prefix-frontier entry → predicate-certified block → level-local query。
- main.tex 拆为七份 sections；四张 TikZ 机制图；完整矩阵/profile/曲线放附录。
- 中文汇报：`docs/papers/multilevel_ung/ADVISOR_NOTE_CN.md`。
- 大纲：同目录 `PAPER_OUTLINE_CN.md`；独立审阅与处理记录在 `review/`。
- 原 126-cell topology、41-case build 证据与唯一 profile timeout 保留；不重跑旧 396-case campaign。

## 审阅与修正

- 两轮结构审阅、源码/证明审阅、文献核查；不预设审阅好评。
- 修正 bitmap frontier 实现、G−r forest 边数、conditional safety、direct membership、seed trimming、array queue/storage 复杂度。
- 统一 0L/1L/2L 和完整 L0|upper topology 记号。
- profile timeout 与 Recall NC 分离；批大小和 DRH-v2 baseline cohort 明确。
- 论文不宣称 Trie 普遍优于 LNG、层数单调加速、所有 SOTA 失败或 GPU 保持查询质量已证明。

## 验证

- Remote finalizer 已跑通 256 tests、证据再生成、diff check、Tectonic，无未解析引用。
- 21 页 PDF（含引用和附录）完成渲染审查；24 个实际文献引用均解析；source/output hashes 已核对。
- GitHub main 已由 W300 远程推送，包含 `e957dcb`。ShareLaTeX 项目 `6aba85f3680204171d471aa7` 的 34 文件逐项回读 hash 一致，在线编译成功。
- 发布记录：`runs/paper_revision_20261004/`；条件证明和文稿审阅见 `review/`。本轮完成的是正文重写，不宣称下列投稿实验缺口已完成。

## 明确保留的证据缺口

- 当前协议外部 SOTA 对照、同 topology 三入口 provider 受控实验。
- Trie/block 跨数据集转移，自动推导不同深度后的性能。
- 最快 hybrid 等 GPU 构建索引的等 Recall 查询质量。
- 更充分重复、资源/多 Recall 目标与动态维护研究。

## 恢复

1. 读取 `docs/papers/multilevel_ung/review/RESOLUTION.md` 及 README。
2. 本轮只改文稿、生成/验证脚本、相关测试、看板；不改查询算法。
3. 只操作本 worktree，禁止修改 `/home/graphdb/FilterVectorCode_refactor`。
4. 保留无关 untracked 文件和第三方目录。
