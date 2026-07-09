# Agent 看板

最后更新：`2026-07-08`
分支：`optimized-clean-v2`
检查点：`64a18ec`（push：`待确认`）

## 目标

把 `/home/graphdb/FilterVectorCode_refactor` 按 `review-ready-project` 标准整理到可交给人类 reviewer 审阅的状态，重点覆盖近期 special block、GPU intra/inter、性能文档和复现实验证据。

## 当前状态

- 总体：`验证中`
- 摘要：已新增 review-ready handoff、requirements matrix、special 当前口径速览和 reviewer reproduction checklist；交付物审查第一轮和结构 follow-up 均为 `nearly review-ready`，主要 blocker 已处理。剩余风险是工作树仍不是干净 release branch。

## 进行中

- 形成 final review-ready 报告；若继续推进，下一步是提交前的完整结构审查和干净 diff 拆分。

## 完成历史

- 完成 GPU special intra 分界线调参：CPU default exact-topK `th=2048`，GPU default `th=128`；证据见 `docs/reports/SPECIAL_BLOCK_BUILD_QUERY_AB_CN.md`。
- 完成 GPU special inter kernel warps 调参：默认 `2` warps/query，kernel `~5.34s -> ~1.87s`，coverage L50/L200 `0.344/0.403`；证据见 `docs/reports/SPECIAL_BLOCK_BUILD_QUERY_AB_CN.md`。
- 更新主性能总表：`docs/reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md` 增加 `## 0. 最终性能大表` 和 `## 8. Reviewer Reproduction Checklist`。
- 新增 review-ready 入口文档：`docs/reports/REVIEW_READY_HANDOFF_CN.md`、`docs/reports/REVIEW_READY_REQUIREMENTS_MATRIX_CN.md`；claim-language 检查 5 个文件通过。
- 清理 review 噪声：`._*` AppleDouble 文件已移动到 `/tmp/fv_appledouble_20260708`，`.gitignore` 已加入忽略规则。
- 最近验证：`cmake --build /home/graphdb/FilterVectorCode_refactor/build_mode_switch -j 16 --target build_UNG_index search_UNG_index` 通过；`python3 tools/benchmarks/check_paper_claim_language.py docs/reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md` 通过。

## 下一步

当前 review-ready 收口已完成；后续若要进入 release-style review，应拆分/提交当前大规模脏工作树。

## 阻塞与问题

- SSH 跳板 `W300-pub` 偶发 kex reset；短间隔重试通常恢复。
- 工作树已有大量历史修改和 untracked 文件；不要回滚或覆盖无关改动。
- `._*` AppleDouble 临时文件已移动到 `/tmp/fv_appledouble_20260708`；当前 `find . -name "._*" | wc -l` 为 `0`。

## 验证

- `cmake --build ... --target build_UNG_index search_UNG_index` — `通过`：近期 GPU inter warps=2 后已编译。
- `python3 tools/benchmarks/check_paper_claim_language.py docs/reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md` — `通过`：主性能表无 over-strong claim。
- `python3 tools/benchmarks/check_paper_claim_language.py docs/reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md docs/reports/SPECIAL_BLOCK_BUILD_QUERY_AB_CN.md docs/reports/SPECIAL_BLOCK_GRAPH_QUERY_DESIGN_CN.md docs/reports/REVIEW_READY_HANDOFF_CN.md docs/reports/REVIEW_READY_REQUIREMENTS_MATRIX_CN.md` — `通过`：5 个 review 文档无 over-strong claim。
- `cmake --build /home/graphdb/FilterVectorCode_refactor/build_mode_switch -j 16 --target build_UNG_index search_UNG_index` — `通过`：review-ready 文档修正和 AppleDouble 清理后重新验证。
- `find . -name "._*" | wc -l` — `通过`：输出 `0`，AppleDouble review 噪声已移走。

## 仅在需要时阅读的细节

- `docs/reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md` — reviewer 第一入口，总览五条主线与最终性能大表。
- `docs/reports/SPECIAL_BLOCK_BUILD_QUERY_AB_CN.md` — 需要审查 special block 构建/查询、CPU/GPU intra/inter 和 kernel tuning 证据时阅读。
- `docs/reports/SPECIAL_BLOCK_GRAPH_QUERY_DESIGN_CN.md` — 需要理解 special block 定义、sidecar/free-state 查询语义和默认参数时阅读。
- `docs/reports/REVIEW_READY_HANDOFF_CN.md` — 本轮 review-ready handoff，完成后作为 reviewer 导航入口。
- `docs/reports/REVIEW_READY_REQUIREMENTS_MATRIX_CN.md` — 本轮需求-证据矩阵，完成后用于审查用户历史要求覆盖情况。

## 恢复说明

1. 先读本看板。
2. 运行 `git status --short`，确认不要误回滚无关改动。
3. 从 `docs/reports/REVIEW_READY_HANDOFF_CN.md` 继续；子代理反馈已整合到 handoff/requirements matrix/方法注册表。
4. 做实质修改前更新本看板；验证后同步记录验证结果。

## 清理提示

- 可压缩或移出 repo 的噪声：`tmp/`、`run.log`、`repo_state.txt` 等历史临时文件，提交前应单独处理。
- 可从看板移除的已尘埃落定事项：无。
