# C3 可复用 Recall 记录导出

完成现有95个全局crossing的导出，包含190次warm、恰好190,000条原始Recall标量记录。没有ANN查询、索引构建或精确近邻重算；没有修改仓库源码、论文或现有生成器。

## 入口和产物

独立CLI原稿：`/tmp/mlung-c3-extract-query-recall.py`。
本目录中的相同脚本：`extract_query_recall.py`。
远程目录：`/home/sunyahui/worktrees/FilterVectorCode_multilevel_special/runs/paper_c3_20261007/query_recall_records/`。

```bash
python3 extract_query_recall.py \
  --repo /home/sunyahui/worktrees/FilterVectorCode_multilevel_special \
  --output /home/sunyahui/worktrees/FilterVectorCode_multilevel_special/runs/paper_c3_20261007/query_recall_records
```

该提取步骤需要原始宽表query details；产出的紧凑CSV可独立用于后续source-only重分组。

- `query_recall_records.csv.gz`：恰好190,000行数据，字段为`workload,method,topology_code,lsearch,repeat,QueryID,QuerySize,Recall`。
- `repeat_recall_checks.csv`：190个全批Recall重建检查。
- `query_size_strata.csv`：570行原有标签数分层及11格Recall直方图（含空层）。
- `workload_query_sizes.csv`：9000个唯一workload/QueryID的标签数映射。
- `manifest.json`：97个输入（95份query details、warm points、factorial）的读取前/后/最终SHA256，脚本及输出SHA256。

## 已完成校验

1. 冻结14×9表与warm points对应95个唯一complete crossing，排除31个NC，未生成不存在的crossing。
2. 每个crossing的method、Lsearch、QPS与Recall门槛与冻结表一致；仅取Repeat1/2。
3. 每个warm batch恰有QueryID0..999，跨方法与repeat的QuerySize映射一致。
4. 所有Recall在[0,1]且落在0.1网格；190个总体均值与warm points一致，最大误差1.1102230246251565e-16。
5. 97个输入读取前、读取后以及最终发布前哈希稳定。
6. gzip固定mtime=0、空filename、压缩级别9、UTF-8/LF及显式行顺序；在同一环境独立序列化两次字节一致。190,000条记录解压逐值回读一致。
7. 远程产物按字节复制到本目录后，再次检查所有输出哈希、gzip字段和行数、190批ID范围、9000个标签数映射及190个总体Recall，全部通过。

压缩大小：**614,942字节**。
压缩CSV SHA256：`e9b95f3e4e64061df030db2b4f3aa251c00aeb0455e3c14c43d84b2e0c77a657`。
未压缩CSV SHA256：`e41a4c219ac653ff53effc22cd5697ec2aa8cf68f44f6ae375f45a47fdc23ac9`。
脚本SHA256：`ecc8e48ff87c4c9b6fcc6979e3dd287e203c7c7dc970e6c3e11681b8f412c16f`。

## 解释边界

这是对历史已导出的Recall标量进行保真导出和重新分组，不是对答案ID或exact neighbors的独立验证。每个方法使用其全局crossing，不是子群单独选择的预算；总体达到0.90不意味着每个子群均达到。两次warm重复同一查询集合，不构成190,000次独立统计实验。

QuerySize是载入标签向量长度；独立coverage审计已提供9000条审计标签/真实选择率，后续可按`(workload,QueryID)`连接。本次没有通过CandSize推断eligible数量，也没有使用LIGHT_STATS零计数做work归因；紧凑记录不包含时间列，因此不生成子群QPS。

本次远程只在上述NEW runs目录写入脚本与产物；没有写入`/home/graphdb`。
