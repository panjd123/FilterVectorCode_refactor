# Agent 看板

最后更新：`2026-09-10 Asia/Shanghai`
分支：`codex/multilevel-special-block-20260905`
检查点：`9ba5ce8`（push：`未执行；origin 指向用户脏工作树`）

## 目标

在隔离仓库中实现并严谨评估真正的两级 Special Block；明确系统组成、相对 0/1 层的优势区间、固定 T1 只增加第二层的增量效果，以及 Amazon 历史查询集的可用边界。

## 当前状态

- 总体：`完成`
- 摘要：实现、高选择率实验、低选择率固定 T1 消融、Nsys CPU profile 和历史 `cov10k` 重测均已完成。当前证据显示：低于 1% 时第二层与同 T1 单层基本同量级但不能声称严格不变；约 10% 时第二层相对同 T1 单层快 2.46--2.70x；历史 28.61% workload 上固定 T1 双层快 1.13--1.21x，但独立调优单层仍是候选中最快；两层相对单层最稳定的系统级优势出现在已测高选择率点。

## 进行中

- 无。

## 完成历史

- 实现两次独立 bottom-up partition、双 ownership、`activation_level=0/1/2`、逐级授权与 fail-closed 验证；完整实现/验证历史已包含在本分支早期提交中。
- Amazon x1 六档公平调优完成：0/1/2 层分别独立调参；92/92 formal cases、552 points 通过验证。共享配置六档 geomean：单层 3.034x、双层 3.019x，不能声称双层总体胜出。
- 高选择率评估完成：80.024%--96.702% 五个 filtered 点上，双层相对单层逐点更快，geomean 1.093x；100% 空 predicate control 为 1.196x。100% 不用于外推 97%--99%。证据提交 `42c21c1`。
- 固定 `T1=1k` 低选择率消融完成：0.499%/0.903% 无一致收益，9.907% 双层相对单层 2.46--2.70x；18/18 cases、所有 selected results 0 violations。证据提交 `e71079a`。
- 同 T1 profile 完成：0.903% formal 的孤立 8.83% 慢点未在 Nsys 重跑复现；搜索无 CUDA activity，middle/upper 执行计数均为 0，CPU 热点组成近似。当前只能归因为未稳定复现的 CPU 波动；overlay cache/内存效应尚未证实。
- Amazon 查询目录审计完成：`old_query_1000/query_minlen1_cov10k` 可重算 GT 后正式使用；普通 hybrid 需重标定；Zipf 与当前 x1 label 语义不兼容；selected-recall 集存在后验选择偏差。
- 历史 `cov10k` 正式复测完成：8/8 cases、24 points、7 repeats，validator 通过。固定 T1 双层相对单层 1.13--1.21x；独立调优单层 `T1=128k` 为 3831.68 ms，优于双层 `16k/400k` 的 4487.59 ms。profile 与历史查询 compact evidence 提交 `3bd438c`。

## 下一步

如需整合回原始脏仓库，先阅读 merge-back 报告并人工处理重叠路径；不要直接 merge 或 cherry-pick 整个分支。

## 阻塞与问题

- 无结果阻塞。`gpulock` 不存在，实验只能声明进程协调与 `nvidia-smi` preflight，不能声明锁级独占。
- 原始 `/home/graphdb/FilterVectorCode_refactor` 有大量用户改动；禁止直接 merge/reset/cherry-pick。
- `runs/` 和 nested third-party clone 不提交；大型 Nsys/SQLite 文件保留在测量机上并由 profile 目录 `.gitignore` 排除。
- 两层当前发现 upper blocks 时走 CPU neighbor-list 路径；本轮 query profile 无 CUDA API/kernel activity。

## 验证

- `validate_selection_sweep.py config.amazon_x1_low_same_t1_formal.json` — `通过`：18/18 cases。
- `validate_selection_sweep.py config.amazon_x1_legacy_cov10k_formal.json` — `通过`：8/8 cases、24 points，最大 stage closure error `2.794e-12 ms/query`。
- 高选择率主实验与 boundary extension — `通过`：95/95 + 20/20 cases，合并 719 个唯一点。
- `python3 -m py_compile summarize_legacy_cov10k.py summarize_query_details.py` — `通过`。
- compact evidence 重建与 `cmp` — `通过`：equal-recall CSV 和 execution manifest snapshot 与提交版本一致。
- `python3 -m unittest -q test_multilevel_selection.py test_generate_paper_results.py` — `通过`：62/62。
- `git diff --check` 与 artifact manifest 更新 — `通过`。

## 仅在需要时阅读的细节

- `docs/reports/MULTILEVEL_SPECIAL_BLOCK_PAPER_REPORT_CN.md` — 面向展示的主结论。
- `experiments/multilevel_special/LOW_SAME_T1_EXPERIMENT.md` — 固定 T1 低选择率协议与结果。
- `experiments/multilevel_special/profile/low_selectivity_same_t1/PROFILE_REPORT.md` — 解释低选择率慢点时阅读。
- `experiments/multilevel_special/LEGACY_AMAZON_QUERY_RETEST.md` — 查询目录兼容性与历史 cov10k 结果。
- `docs/reports/MULTILEVEL_SPECIAL_BLOCK_REPRODUCE_CN.md` — 重跑正式实验时阅读。
- `docs/reports/MULTILEVEL_SPECIAL_BLOCK_MERGE_BACK_CN.md` — 准备人工整合回原始脏仓库时阅读。

## 恢复说明

1. 先读本看板并运行 `git status --short`。
2. 不要暂存 `runs/`、`thirdparty/` 或 profile 下的大型 raw 文件。
3. 核对 HEAD 至少包含 `9ba5ce8`；新数值必须先进入 compact evidence，再更新主报告。
4. 从“下一步”继续，不修改 `/home/graphdb/FilterVectorCode_refactor`。

## 清理提示

- 保留 compact CSV、execution manifest snapshot、hash 清单和 profile 摘要。
- 大型 raw benchmark、`.nsys-rep`、SQLite、逐查询 profile output 均保持未跟踪。
