# C3：冻结 crossing 按 QuerySize 的 Recall 分层

**关键发现：全批 Recall crossing 会掩盖未达标子群。** 5%批次的43个单标签查询，固定TLT均值只有0.74419，而该配置整体已达到0.90门槛。另一方面，0.5%与1%批次全部是至少两个载入标签，固定TLT均值分别为0.95920与0.95430，不能把全部收益解释为只服务单标签。

本次只重分组已存在的query details，没有重新运行ANN或构建。TLT为开发集观察选出的固定`2L[T|LT]`；L/T/TL分别为`0L[L]`、`0L[T]`、`1L[T|L]`。不同配置使用各自冻结的最小实测crossing容量，以下不是相同Lsearch或相同实际Recall下的比较，也没有产生子群QPS。

## 产物与完整性

- 可复现脚本：`/tmp/mlung-c3-query-size-strata.py`；同内容远程脚本为允许输出目录内的`extract_query_size_strata.py`。
- 远程输出仅在 `/home/sunyahui/worktrees/FilterVectorCode_multilevel_special/runs/paper_c3_20261007/query_size_strata/`。
- 本地字节保真副本：`/tmp/mlung-c3-query-size-strata-artifacts/`。
- `query_size_strata.csv`：95 crossing × 2 warm × 3 strata = **570行**，包含空层；`fixed_configuration_strata.csv`为L/T/TL/TLT的**210行**（T在30%没有crossing）。
- 每行包括count、mean Recall、Recall<0.9数量、Recall=0数量、0.0至1.0共11格精确Recall@10直方图，并分别保留Repeat1/2。空层mean留空、count及histogram为0，不伪造Recall=0。
- 同时保存Time_ms、ELS、seed、authorization、graph、residual的per-query均值；`repeat_recall_checks.csv`含190个全批重算检查，`workload_query_sizes.csv`含9000个workload/QueryID标签长度映射。
- 每个选中repeat恰有QueryID 0..999；同workload的QueryID→QuerySize在全部方法及repeat间一致；190000条选中记录的190个总体Recall与warm points一致，最大误差为 **1.1102e-16**。
- 所有95个query details输入以及warm points均记录SHA256、原路径，输出和脚本也记录hash。下载后额外验证输出字节hash、570行直方图count/均值恒等式、190个分层加权总体均值，全部通过。

## 查询组成与固定配置的分层均值

表中`a / b`分别是Repeat1 / Repeat2；相同值只显示一次，但CSV仍分别保存。QuerySize是载入标签向量长度，不是eligible向量数或真实selectivity。

| Workload | QuerySize | 每repeat数量 | L mean Recall | T mean Recall | TL mean Recall | TLT mean Recall |
|---|---|---:|---:|---:|---:|---:|
| sel_0p5 | >=2 | 1000 | 0.91680 / 0.91700 | 0.95980 / 0.96090 | 0.95920 | 0.95920 |
| sel_1 | >=2 | 1000 | 0.91060 / 0.91080 | 0.94610 / 0.94600 | 0.95430 | 0.95430 |
| sel_5 | 1 | 43 | 0.28837 | 0.14884 | 0.73256 | 0.74419 |
| sel_5 | >=2 | 957 | 0.93762 / 0.93783 | 0.93710 / 0.93647 | 0.92644 | 0.92644 |
| sel_10 | >=2 | 1000 | 0.90350 | 0.93240 / 0.93180 | 0.92250 | 0.90700 |
| sel_30 | 1 | 70 | 0.71714 / 0.71857 | NC | 0.94286 | 0.94571 |
| sel_30 | >=2 | 930 | 0.92516 / 0.91871 | NC | 0.90258 | 0.90538 |
| sel_60 | 1 | 1000 | 0.92830 / 0.92950 | 0.91430 / 0.91250 | 0.90110 | 0.90230 |
| sel_80 | 1 | 1000 | 0.92570 / 0.92710 | 0.96990 | 0.91670 | 0.91600 |
| sel_95 | 1 | 1000 | 0.91360 / 0.91340 | 0.96660 | 0.90980 | 0.90910 |
| sel_99 | 0 | 697 | 0.93788 / 0.93845 | 0.96585 | 0.92568 | 0.92539 |
| sel_99 | 1 | 303 | 0.94752 / 0.94620 | 0.96502 | 0.91485 | 0.91254 |

## 关键正负结果

1. **5%混合批次的全局门槛掩盖单标签失败。** 43个QuerySize=1与957个QuerySize≥2混合。L/T/TL/TLT在单标签层的均值为0.28837/0.14884/0.73256/0.74419；低于0.9的查询数分别为39/41/21/19，零Recall分别为10/23/5/4，两次warm相同。TLT相对L/T改善明显，但该子群依然没有达到0.90。它的957个多标签查询均值为0.92644、零Recall1个，不能用整体crossing宣称每类查询都达到目标。
2. **多标签场景存在正面结果。** 0.5%与1%批次均1000个QuerySize≥2。TL和TLT分别达到0.95920与0.95430，均无零Recall查询；L相应约0.9168–0.9170和0.9106–0.9108，并有3–4与4个零Recall。这里无需借助单标签或空谓词解释，但这些仍是各方法自己所选预算处的质量。
3. **30%存在子群之间的质量交换。** L的70个单标签均值0.71714–0.71857，TLT为0.94571；TLT的930个多标签均值0.90538，低于L的0.91871–0.92516。各方法总体都跨线不意味着对子群的质量排序相同。T在此没有crossing，本次不为它构造分层行。
4. **10%全部为多标签，增层并不提高质量。** TL均值0.92250、13个零Recall；TLT为0.90700、16个零Recall；两次warm相同。L为0.90350、15个零Recall；T为0.93180–0.93240、7个零Recall。这是冻结预算处的观测，不能推导固定预算下的因果效应。
5. **60/80/95%三个批次全部为单标签，快速crossing仍有失败尾部。** TLT的零Recall数依次为31/29/33，L为11–12/12/14；TL为32/27/31。对应TLT均值仍有0.90230/0.91600/0.90910。主文若保留这些批次的大吞吐倍率，应同时区分“批次均值≥0.90”与“所有查询均有较高Recall”，并避免把这三批当作多标签AND的直接证据。
6. **99%分开空谓词与单标签。** 697个QuerySize=0，303个QuerySize=1。TLT均值为0.92539/0.91254，零Recall16/13；L约0.93788–0.93845/0.94620–0.94752，零Recall8/4。空谓词表示全体向量eligible，不是空结果。

## 时间与字段解释边界

- `Time_ms`是并发执行条件下单查询wall duration；输出仅称 observed per-query cost。均值不能变成子群QPS或batch latency，重复同一1000条查询不构成1000次独立时间实验。
- 字段审计指出：GraphSearchTime_ms和ResidualTime_ms由其他计时差值计算并截零，不能作为独立精密测量，也不保证每行阶段严格闭合。已有标量Recall只被重分组，没有重算ground truth或ANN。
- loader排序但不去重，因此QuerySize严格说是载入标签向量长度；本轮尚未独立核查每个查询标签是否重复。method_proof_audit正在做单独标签/真实coverage审核，未来只能在其通过后再按真实selectivity关联，不能现在用QuerySize或CandSize替代。
- 未读取CandSize、LIGHT_STATS零work counters作任何selectivity或work归因。没有依据这些数据诊断零Recall原因。
- 历史记录的具体字段解释依据当前源码及独立字段审计；历史binary快照一致性与精确源码build关联是不同事项，后者仍有限。

**可用于C3的谨慎表述：**“按载入查询标签数重分组后，固定TLT在低平均选择率的纯多标签批次保持较高平均Recall，但在5%混合批次的单标签子群仍仅约0.744；高平均选择率的三个批次全部为单标签，整体均值达标同时保留零Recall尾部。批次级同门槛吞吐不能替代子群质量报告。”

脚本SHA256：`d7fbb5fa37077245e704dbd5664ace366a8a916fe03058a3635f4a292545f4cb`。完整输入/输出provenance见上述产物目录的`manifest.json`。
