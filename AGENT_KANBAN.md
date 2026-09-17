# Agent 看板

最后更新：`2026-09-17 Asia/Shanghai`
分支：`codex/multilevel-special-block-20260905`
检查点：本轮提交（exact-level router、singleton stop rule、current-binary 对照与最终报告；push：`不执行，origin 指向用户脏工作树`）

## 目标

实现 exact-level 多层 Special Block，并以无需 query latency/Recall 校准的静态规则自动确定层数及各层 block 阈值；在 Amazon 九档和三个外部数据集上按等 Recall 验证。

## 当前状态

- 总体：`完成`
- 摘要：exact-level router、Amazon 九档正式/15-repeat 关键实验和 Reviews/VariousImg 跨数据集实验完成。QF-SSL 由 `M_sb=64,Lbuild_sb=100,C_sb=4` 推导 `T1=8192,rho=16,T2=131072`；只有至少两个非空候选尺度才物化层级，因此 Amazon/Reviews/VariousImg 选两层，Genome 自动回退 0 层。该 singleton stop rule 是观察 Genome/CelebA 单层负例后的开发修正，不称 held-out。
- 当前最佳判断：QF-SSL 是 query-free structural prior，不是性能最优性定理。Amazon 低档 router/plain 总体持平；统一 current binary 后，高档 router/同 T1 单层 geomean 1.992x，九档 router/prior-tuned 两层 geomean 1.010x。VariousImg 为稳定正例，Reviews 4.115% 与 plain 持平，Reviews 0.200% wall-time 因执行顺序与线程调度不稳定而不可判定。

## 进行中

- 无；等待用户审阅。

## 完成历史

- exact-level gate、upper membership promotion、upper portal seed 已提交：`4585099`；候选只消费当前层拥有的边。
- 九档 workload 生成器已提交：`073a775`；Amazon 实际平均选择率为 0.499/0.903/5.038/9.907/30.027/60.047/80.024/95.020/99.001%。
- QF-SSL 静态规则已实现：`T1=pow2ceil(M_sb*Lbuild_sb)`、`rho=pow2ceil(max(2,M_sb/C_sb))`、`Tl=T1*rho^(l-1)`；首个空 partition 前停止，少于两个候选尺度则回退 plain。
- Amazon router 9 workload x 7 repeats 已完成；低档 router/plain 15-repeat 复核总体持平；统一 current binary 后 30%--99% router/同 T1 单层 geomean 1.992x。
- Reviews/VariousImg router 正式实验完成；VariousImg 10.252% 相对 plain 2.317x、相对单层 1.539x。
- Reviews 0.200% 已完成原顺序、反转顺序及 12 对独立进程交错复核；wall-time 方向翻转且与逐 query 分解矛盾，最终标为不可判定，不挑选有利数字。
- QF-SSL 主报告已按最终 singleton stop rule、结构路由和新结果重写。
- 主报告已补充自动分层对人工 prior-tuned 的九档同 binary 对比，以及历史候选网格内单层最优/双层最优/plain 的六档对比；后者明确标为历史 sweep，不与 current-binary 主结论混合。

## 下一步

用户可从 `docs/reports/QF_SSL_AUTO_POLICY_REPORT_CN.md` 审阅最终方案；若准备写论文，直接引用摘要及第 1、4、5、8 节。

## 阻塞与问题

- 无实验阻塞。Reviews 0.200% 属测量不可判定点，已作为限制保留。
- 原始 `/home/graphdb/FilterVectorCode_refactor` 是用户脏工作树，不修改、不 merge。
- `runs/` 和 nested `thirdparty/` 不提交。
- 当前实现最多物化两层；静态分析 safety cap 已放宽到 64，四个数据集均在 1/2 层后因下一尺度 partition 为空自然停止，未被实现上限截断。未来数据若输出三层以上则需扩展索引格式。

## 验证

- `cmake --build build_ung_rel -j16 --target test_special_block_free_state search_UNG_index` — `通过`。
- `ctest --test-dir build_ung_rel --output-on-failure -R special_` — `通过`：4/4。
- `ctest --test-dir build_ung_rel --output-on-failure -R filter_validation` — `通过`：1/1。
- `validate_selection_sweep.py config.auto_policy_formal_exact_level.json` — `通过`：Amazon 36 cases。
- `validate_selection_sweep.py config.auto_policy_critical_exact_level.json` — `通过`：Amazon 18 cases、15 repeats。
- 三个 `config.auto_policy_cross_dataset_*_formal.json` — `通过`：13 cases；所有正式 repeat 达 Recall 门槛。
- `python3 -m unittest -v test_auto_layer_policy.py test_multilevel_selection.py test_generate_paper_results.py` — `通过`：66/66。
- `cmake --build build_ung_rel --target test_special_block_free_state search_UNG_index -j4` — `通过`。
- `ctest --test-dir build_ung_rel --output-on-failure -R special_block_free_state` — `通过`：1/1。
- current-binary 汇总从 raw manifests 重生成并与 committed compact CSV 逐字节一致。
- `git diff --check` — `通过`；`runs/` 与两个 nested `thirdparty/` 未提交。

## 仅在需要时阅读的细节

- `docs/reports/QF_SSL_AUTO_POLICY_REPORT_CN.md` — 自动参数方法、完整新结果与学术边界。
- `docs/reports/MULTILEVEL_SPECIAL_BLOCK_PAPER_REPORT_CN.md` — 旧手工调参与系统背景；不要把其中旧语义结果当作新 exact-level 数据。
- `experiments/multilevel_special/EXPERIMENT_STATE.md` — 历史 issue ledger；以本看板和 QF-SSL 报告中的最新状态为准。

## 恢复说明

1. 先读本看板。
2. 运行 `git status --short`，不要暂存 `runs/` 或 `thirdparty/`。
3. 核对 HEAD 和最新验证；从“下一步”继续。
4. 不修改 `/home/graphdb/FilterVectorCode_refactor`。

## 清理提示

- `runs/` 保存 raw evidence，但不纳入 Git。
- 旧报告保留作历史对照；论文引用以 QF-SSL 报告为准。
