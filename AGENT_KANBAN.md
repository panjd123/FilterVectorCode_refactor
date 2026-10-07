# Agent 看板

最后更新：2026-10-07 15:08（北京时间）
权威分支：codex/multilevel-special-block-20260905
已发布内容检查点：6cc09d6；C3已由W300推送main，108份ShareLaTeX文件回读通过。

## 目标和时间

18小时从2026-10-06 20:52:13开始，最早完成2026-10-07 14:52:13；目标已在边界后完整完成。

## 完成历史

- C0两份全新PDF评审均Reject；4/5是clarity，2/5是evidence。
- C1新增覆盖/分区分析、固定表和全网格图并已发布。定向复核接受核心解释，保留实验证据缺口。
- C2补齐数学前提、发布95组原始warm timing及35组描述性范围；262 tests通过。
- 短汇报约4100字符，详细备查独立；source-only CSV/TeX再生和链接检查通过。
- C2 PDF 27页，无undefined refs；参考文献尾行、路径和附录表格顺序已检查。
- C3加入查询组成、子群Recall和精确选择率证据；271项测试与独立source-only复现通过。
- GitHub main内容检查点为6cc09d6；ShareLaTeX 108份文件逐字节回读通过，服务端成功编译28页PDF。

## 当前状态

C3内容、完整验证和发布审计均通过。仓库Tectonic PDF为28页，SHA
ce12f3edb743bab3fd4600a91557820f54ea6eb7e0fbf3dbf5c0812d336d6ca9；
271项测试通过且无undefined refs/citations。ShareLaTeX服务端PDF同为28页，SHA
eed4744c1b48a1c1ddb5d82ae5fd49989d21a653c7cb8cd6812735e52ff07943。
18小时边界已于14:52:13达到。

## 进行中

无。C3已完成内容、独立数据复核、远程编译、GitHub发布和ShareLaTeX发布审计。

### 已完成的输入分析

- method_proof_audit：/tmp/mlung-c3-query-fields-audit.md；字段语义、历史binary与query文件映射。
- paper_structure_review：/tmp/mlung-c3-query-size-strata.py 和 .md；只读95个crossing的query_details，输出remote runs/paper_c3_20261007/query_size_strata/。
- CandSize不是eligible向量数；LIGHT_STATS work零值不可归因；只在核实真实query metadata后研究单查询选择率。

## 恢复说明

本地 /tmp/filtervector-paper-18h-20261006；远程 /home/sunyahui/worktrees/FilterVectorCode_multilevel_special。
不修改/home/graphdb/FilterVectorCode_refactor，不重跑396-case campaign，不增加未声明的benchmark。
外部基线、核心跨数据集、label order、loaded memory、GPU质量仍为证据缺口。
恢复看review/CONTINUOUS_18H_MANIFEST.json、C3_RESOLUTION.md、VALIDATION_C3.json及runs/paper_c3_20261007/finalization_manifest.json。

C3关键证据：190k旧query记录/570分组全部与190次batch均值一致；5%中43个singletons的TLT Recall仅0.74419。602453条base和9000条query无重复标签；3239个唯一predicate的精确覆盖逐行匹配原profile（0 mismatch）。
主文已解释批次均值会掩盖子群和高选择率的zero-Recall尾部；保留crossing/QPS协议，未虚构子群同Recall加速比。

## 下一步

无待办。发布回执位于review/SHARELATEX_C3_PUBLICATION.json；开放证据缺口继续明确保留，不以本轮结果填补。

验证：新增9项语义测试及完整271项测试通过。独立实现逐字段核对756条标签数和2268条选择率summary；六个source-only产物逐字节一致。复核发现5%中43条与30%中226条分别重复{1}和{1,2}，正文已明确，禁止外推为整个选择率区间规律。
