# Agent 看板

最后更新：`2026-09-17 Asia/Shanghai`
分支：`codex/multilevel-special-block-20260905`
检查点：`073a775`（QF-SSL 脚本、报告与 compact evidence 待 commit；push：`不执行，origin 指向用户脏工作树`）

## 目标

实现 exact-level 多层 Special Block，并以无需 query latency/Recall 校准的静态规则自动确定层数及各层 block 阈值；在 Amazon 九档和三个外部数据集上按等 Recall 验证。

## 当前状态

- 总体：`验证中`
- 摘要：exact-level 实现、Amazon 九档正式/15-repeat 关键实验、Genome/Reviews/VariousImg 跨数据集实验均完成。QF-SSL 由 `M=64,Lbuild=100,C=4` 推导 `T1=8192,rho=16,T2=131072`，并以静态 trie partition 是否非空自动停止；Amazon/Reviews/VariousImg 选两层，Genome 选一层。正在进行最终文档、测试和 Git checkpoint。
- 当前最佳判断：QF-SSL 是 query-free structural prior，不是性能最优性定理。Amazon 中低选择率两层相对同 T1 单层总体持平，高选择率显著获益；跨数据集同时存在正、平、负例。

## 进行中

- 审核论文级报告与 compact evidence，运行最终验证并提交。

## 完成历史

- exact-level gate、upper membership promotion、upper portal seed 已提交：`4585099`；候选只消费当前层拥有的边。
- 九档 workload 生成器已提交：`073a775`；Amazon 实际平均选择率为 0.499/0.903/5.038/9.907/30.027/60.047/80.024/95.020/99.001%。
- QF-SSL 静态规则已实现：`T1=pow2ceil(M*Lbuild)`、`rho=pow2ceil(max(2,M/C))`、`Tl=T1*rho^(l-1)`，首个空 partition 前停止。
- Amazon 正式 4 方法 x 9 workload x 7 repeats 已通过 validator；自动两层相对 plain 九档 geomean 10.602x，相对同 T1 单层 geomean 1.468x。
- Amazon 同 T1 关键对照 2 方法 x 9 workload x 15 repeats 已通过 validator；0.5%--10% 两层/单层 geomean 1.029x，30%--99% 为 1.985x。
- Genome/Reviews/VariousImg 等 Recall 正式实验完成并通过 validator。VariousImg plain 首次 `L=4250` 有一个 repeat 低于 0.90，已前进到下一实测点 `L=4500` 重跑并通过。
- 已生成跨数据集与 15-repeat compact CSV，并撰写 `QF_SSL_AUTO_POLICY_REPORT_CN.md`。

## 下一步

运行 py_compile、单元测试、C++ focused tests、`git diff --check`；确认只暂存 QF-SSL 相关脚本/配置/compact evidence/报告/看板，然后 commit。

## 阻塞与问题

- 无实验阻塞。
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
