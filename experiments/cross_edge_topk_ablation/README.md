# Cross-Edge TopK Ablation 实验说明

这个目录用于公平测试 UNG 跨组边构建优化。普通 UNG 的 ablation 配置只改 cross-edge 相关环境变量，其他构建设置保持和 `experiments/cpu_special_blocks/config_ung.json` 一致。

脚本默认跑 4 个配置：

- `baseline_ung_hybrid_control`：同批次默认 UNG control，等价于 `config_ung.json` 的默认构建设置，只换了 index name。
- `ordinary_batch_matrix_sort`：普通 UNG 跨组边，小图任务组合 batch + SGEMM/topK baseline。
- `ordinary_batch_onchip_topk`：普通 UNG 跨组边，小图任务组合 batch + descriptor-fused 片上 top-k。
- `special_blocks_gpu_inter_source_exact_topk`：`special_blocks+UNG` 的 GPU inter-block 对照。

## 1. 进入项目目录

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
```

## 2. 先 dry-run 看将要执行的命令

```bash
python3 experiments/cross_edge_topk_ablation/run_cross_edge_topk_ablation.py --dry-run
```

脚本会自动从 `config.json` 生成单独配置到：

```text
experiments/cross_edge_topk_ablation/generated/
```

并打印将要调用的 `run_cpu_special_blocks_experiment.sh` 命令。

## 3. 直接运行全部配置

```bash
python3 experiments/cross_edge_topk_ablation/run_cross_edge_topk_ablation.py
```

默认如果某个配置失败，脚本会停下来。想让后面的配置继续跑：

```bash
python3 experiments/cross_edge_topk_ablation/run_cross_edge_topk_ablation.py --continue-on-error
```

## 4. 只跑某一个配置

只跑默认 control：

```bash
python3 experiments/cross_edge_topk_ablation/run_cross_edge_topk_ablation.py \
  --only baseline_ung_hybrid_control
```

只跑 matrix/SGEMM baseline：

```bash
python3 experiments/cross_edge_topk_ablation/run_cross_edge_topk_ablation.py \
  --only ordinary_batch_matrix_sort
```

只跑片上 top-k：

```bash
python3 experiments/cross_edge_topk_ablation/run_cross_edge_topk_ablation.py \
  --only ordinary_batch_onchip_topk
```

只跑 special_blocks GPU inter 对照：

```bash
python3 experiments/cross_edge_topk_ablation/run_cross_edge_topk_ablation.py \
  --only special_blocks_gpu_inter_source_exact_topk
```

也可以多次传 `--only` 指定顺序：

```bash
python3 experiments/cross_edge_topk_ablation/run_cross_edge_topk_ablation.py \
  --only baseline_ung_hybrid_control \
  --only ordinary_batch_onchip_topk \
  --only ordinary_batch_matrix_sort
```

## 5. 输出位置

默认数据集是 `Reviews`。输出会写到：

```text
/home/dev/graphdb/FilterVectorResult/Reviews/index/<index_name>/
```

普通 UNG 的三个主要 index name：

```text
UNG_cross_edge_baseline_hybrid_control
UNG_cross_edge_batch_matrix_sort
UNG_cross_edge_batch_onchip_topk
```

构建日志通常在：

```text
/home/dev/graphdb/FilterVectorResult/Reviews/index/<index_name>/others/build.log
```

## 6. 检查是否排除了非 cross-edge 干扰

普通 UNG 三个配置都应该看到：

```text
[UNG config] lng_impl=optimized_phase1
[UNG config] descendants_impl=optimized_epoch_bfs
```

如果看到下面这些，说明配置又污染到了非 cross-edge 阶段，需要停止比较：

```text
[UNG config] lng_impl=legacy_allocating
[UNG config] descendants_impl=legacy_hash_bfs
```

## 7. 检查日志确认跑到了目标 cross-edge 路径

默认 control 应看到类似：

```text
[cross_edges] gpu_topk_impl=auto
```

普通 matrix/SGEMM baseline 应看到类似：

```text
[cross_edges] gpu_topk_impl=sgemm_topk
[PROF] cross_edges.bucket_group_fused enabled=0
```

普通片上 top-k 应看到类似：

```text
[cross_edges] gpu_topk_impl=fused_group_topk
[PROF] cross_edges.bucket_group_fused enabled=1
[PROF] cross_edges.id_only_writeback
```

special_blocks GPU inter control 应看到类似：

```text
[special_edges][gpu_inter]
mode=cuda_core
queries=
segments=
```

## 8. 对比指标

要体现跨组边优化，优先比较这些字段：

- `build_cross_edges_time`
- `cross_edges breakdown generate(ms)`
- `cross_edges breakdown additional(ms)`
- `index_time`
- `wall_seconds`

普通 UNG 的公平结论应该基于同批次这三个目录：

```text
/home/dev/graphdb/FilterVectorResult/Reviews/index/UNG_cross_edge_baseline_hybrid_control/
/home/dev/graphdb/FilterVectorResult/Reviews/index/UNG_cross_edge_batch_matrix_sort/
/home/dev/graphdb/FilterVectorResult/Reviews/index/UNG_cross_edge_batch_onchip_topk/
```

注意：`special_blocks_gpu_inter_source_exact_topk` 已经使用 batched GPU inter 和片上 top-k，但它不是普通 UNG 的 small/medium descriptor bucket fused 路线。

## 9. 降低 CPU 波动的单次跑法

`additional_edges` 是 CPU 重阶段，单次运行仍然会有波动；但为了节省时间，建议至少固定 CPU 和 OpenMP 绑定：

```bash
python3 experiments/cross_edge_topk_ablation/run_cross_edge_topk_ablation.py \
  --only baseline_ung_hybrid_control \
  --only ordinary_batch_matrix_sort \
  --only ordinary_batch_onchip_topk \
  --taskset 0-59 \
  --stable-cpu
```

这会按指定顺序各跑一次，并加上：

```text
taskset -c 0-59
OMP_PROC_BIND=close
OMP_PLACES=cores
OMP_DYNAMIC=FALSE
OMP_NUM_THREADS=60
MALLOC_ARENA_MAX=2
```

如果机器上 0-59 这些核会被别人占用，可以换成一段更空闲的核，例如 `--taskset 4-63`，但最好和配置里的 `build.num_threads=60` 保持数量一致。

只想先看将要跑什么：

```bash
python3 experiments/cross_edge_topk_ablation/run_cross_edge_topk_ablation.py \
  --only baseline_ung_hybrid_control \
  --only ordinary_batch_matrix_sort \
  --only ordinary_batch_onchip_topk \
  --taskset 0-59 \
  --stable-cpu \
  --dry-run
```

如果之后需要更稳的统计结果，再临时加 `--repeat N`；单次对比时不要加。
