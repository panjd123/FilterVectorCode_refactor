# Hybrid Base Label 生成工具

这个目录里的工具用于生成 Amazon 数据集的混合版 base label，主要在两份 label 之间做实验：

- old 复杂 label：`/home/dev/graphdb/FilterVectorData/Amazon/old_query/Amazon_base_labels.txt`
- Zipf label：`/home/dev/graphdb/FilterVectorData/Amazon/zipf_query/Amazon_base_labels.txt`

当前推荐使用 `row-mix` 模式：按连续行块混合，而不是随机抽行混合。默认推荐配置是“前 50% 行使用 old label，后 50% 行使用 Zipf label”。每一行都会整行复制，不会对行内 label 做注入、截断、排序或随机替换。

工具仍然保留较早的 `tail-inject` 模式，但当前实验不推荐优先使用它。

## 为什么使用连续块混合

old label 搜索慢，主要是因为 `1`、`2` 这类高频公共 label 下面挂了大量不同的长尾 labelset。查询阶段 ELS 需要花很多时间寻找 containment/min-super-set entry groups，然后才进入图搜索。

Zipf label 更短、更规则。即使查询 `(1,2,3,4)` 这种高频组合，也常常只对应很少的 entry group，所以 `MinSupersetT_ms` 很小。

连续块混合的目标是保留两种真实行结构：

- old 区间保留 old label 的完整复杂度
- Zipf 区间保留 Zipf label 的规则性
- `--old-row-ratio 0.5` 表示 old 占 50%，Zipf 占 50%
- `--old-row-placement first` 表示 old 放在前半段，Zipf 放在后半段
- `--no-manifest` 表示不输出 JSON manifest

## 推荐命令

生成 50/50 连续块混合 label：

```bash
python3 /home/dev/graphdb/FilterVectorCode_refactor/tools/change_base_labels/generate_hybrid_labels.py \
  --zipf-base /home/dev/graphdb/FilterVectorData/Reviews/zipf_query/Reviews_base_labels.txt \
  --old-base /home/dev/graphdb/FilterVectorData/Reviews/old_query/Reviews_base_labels.txt \
  --output /home/dev/graphdb/FilterVectorData/Reviews/Reviews_base_labels.txt \
  --mode row-mix \
  --old-row-ratio 0.5 \
  --old-row-placement first \
  --fill-empty-with-previous \
  --compact-label-ids \
  --no-manifest
```

这条命令的效果是：

```text
前 50% 行：来自 old_query/Amazon_base_labels.txt
后 50% 行：来自 zipf_query/Amazon_base_labels.txt
空行：用前一行填充
label id：按来源分段重映射。old 区间先映射为连续的 `1..old_max`，Zipf 区间再从 `old_max + 1` 开始继续映射
manifest：不输出
```

## 如何修改 old 和 Zipf 的占比

只需要改 `--old-row-ratio`。

`--old-row-ratio` 表示 old label 占总行数的比例，剩下的部分自动使用 Zipf label。

例如：

```text
old:zipf = 2:8  ->  --old-row-ratio 0.2
old:zipf = 3:7  ->  --old-row-ratio 0.3
old:zipf = 5:5  ->  --old-row-ratio 0.5
old:zipf = 7:3  ->  --old-row-ratio 0.7
old:zipf = 8:2  ->  --old-row-ratio 0.8
```

如果使用：

```bash
--old-row-ratio 0.3 --old-row-placement first
```

则输出文件为：

```text
前 30% 行：old label
后 70% 行：Zipf label
```

如果使用：

```bash
--old-row-ratio 0.3 --old-row-placement last
```

则输出文件为：

```text
前 70% 行：Zipf label
后 30% 行：old label
```

也就是说：

- 想让 old 多一些，就增大 `--old-row-ratio`
- 想让 Zipf 多一些，就减小 `--old-row-ratio`
- 想让 old 在前面，用 `--old-row-placement first`
- 想让 old 在后面，用 `--old-row-placement last`

## 常用配置

old 25%，Zipf 75%：

```bash
--mode row-mix --old-row-ratio 0.25 --old-row-placement first --fill-empty-with-previous --compact-label-ids --no-manifest
```

old 50%，Zipf 50%：

```bash
--mode row-mix --old-row-ratio 0.5 --old-row-placement first --fill-empty-with-previous --compact-label-ids --no-manifest
```

old 75%，Zipf 25%：

```bash
--mode row-mix --old-row-ratio 0.75 --old-row-placement first --fill-empty-with-previous --compact-label-ids --no-manifest
```

## 重要参数说明

- `--mode row-mix`：使用连续块混合模式。
- `--old-row-ratio`：old label 占比，取值范围 `[0,1]`。
- `--old-row-placement`：old 区间的位置，可选 `first` 或 `last`。
- `--fill-empty-with-previous`：如果输出中出现空行，用前一行内容填充。
- `--compact-label-ids`：在 `row-mix` 模式下按来源分段重映射 label id。old 区间先按原 id 从小到大映射为连续的 `1..old_max`；Zipf 区间再按原 id 从小到大映射为 `old_max+1..old_max+zipf_max`。例如 old label 映射后为 `1..1000`，Zipf 原 label `1,2,3` 会映射为 `1001,1002,1003`。
- `--no-manifest`：不输出 manifest JSON 文件。
- `--seed`：只影响 `tail-inject` 这类随机模式；当前连续块 `row-mix` 模式是确定性的，不依赖 seed。

## 生成后需要核对什么

建议生成后至少核对三件事：

```text
1. 行数是否等于 base 向量数
2. 是否没有空行
3. 全局 label id 是否从 1 到 max 连续无缺口
4. old 区间和 Zipf 区间的新 label id 范围是否不重叠
```

可以用下面这个检查逻辑：

```bash
python3 - <<'CHECK_LABELS'
from pathlib import Path
import re

p = Path('/home/dev/graphdb/FilterVectorData/Amazon/Amazon_base_labels_hybrid_medium.txt')
split = 301226
labels = set()
old_labels = set()
zipf_labels = set()
empty = []
rows = 0

for row_id, line in enumerate(p.open(), 1):
    rows = row_id
    vals = [int(x) for x in re.findall(r'\d+', line)]
    if not vals:
        empty.append(row_id)
    labels.update(vals)
    if row_id <= split:
        old_labels.update(vals)
    else:
        zipf_labels.update(vals)

mn = min(labels) if labels else 0
mx = max(labels) if labels else 0
missing = [x for x in range(1, mx + 1) if x not in labels]

print('rows', rows)
print('empty_rows', len(empty))
print('global_unique_labels', len(labels), 'min', mn, 'max', mx)
print('continuous', len(missing) == 0 and mn == 1 and mx == len(labels))
print('missing_count', len(missing))
print('old_unique', len(old_labels), 'old_min', min(old_labels), 'old_max', max(old_labels))
print('zipf_unique', len(zipf_labels), 'zipf_min', min(zipf_labels), 'zipf_max', max(zipf_labels))
print('ranges_disjoint', max(old_labels) < min(zipf_labels))
CHECK_LABELS
```

当前已生成的 50/50 文件检查结果是：

```text
rows 602453
empty_rows 0
global_unique_labels 21834 min 1 max 21834
continuous True
missing_count 0
old_unique 20834 old_min 1 old_max 20834
zipf_unique 1000 zipf_min 20835 zipf_max 21834
ranges_disjoint True
```

## 测试

运行工具测试：

```bash
python3 /home/dev/graphdb/FilterVectorCode_refactor/tools/change_base_labels/test_generate_hybrid_labels.py
```
