# PathSeer Experiment

This adapter runs the official [PathSeer](https://github.com/lizhiyuell/PathSeer) implementation on this project's existing filtered-ANN datasets and writes the same result layout used by UNG and FAVOR. The vendored source is pinned at commit `6e0cd1df08927651013823f1f5458e6fb4a2336f`.

## Run

`Amazon_config.json` mirrors the two query tasks in `experiments/favor/Amazon_config.json`. In the current workspace snapshot those task files are absent; `Amazon_current_data_config.json` targets the available `query_minlen1_targetsel25pct` data and ground truth for an immediate local run.


```bash
cd /home/dev/graphdb/FilterVectorCode_multilevel_special
# Available in this workspace now
python3 experiments/pathseer/run_pathseer_experiment.py \
  experiments/pathseer/Amazon_current_data_config.json

# Same two query tasks as FAVOR (after those data files are restored)
python3 experiments/pathseer/run_pathseer_experiment.py \
  experiments/pathseer/Amazon_config.json
```

The runner incrementally builds PathSeer's FAISS fork and the two project adapters. It uses the repository's `thirdparty/json-3.10.4.tar.gz` archive, so CMake does not fetch nlohmann/json from the network. It also auto-detects `/home/dev/graphdb/openblas/lib/libopenblas.so`; set `openblas_library` when OpenBLAS is elsewhere.

## Mapping

- Existing `.fvecs` files are consumed directly.
- Existing base/query label rows are interpreted with containment semantics: a base vector passes when it contains every query label.
- `build.M` and `build.M_exp` map to PathSeer's distance-neighbor and expanding-neighbor construction widths.
- `search.M_exp_search` controls the expanding-neighbor width used at search time. A positive value selects the directly runnable fixed-parameter mode. Set it to `0` and provide positive `virtual_overhead`, `dist_comp_ns`, and `filter_ns` values from the upstream profiler to run full-tech adaptive mode.
- Each `Lsearch` value maps directly to PathSeer's `efSearch`.
- `dist_fn` supports `L2`, `IP`, and `cosine`; cosine normalizes base and query vectors before using inner product.
- The shared UNG `(uint32 id, float distance)` ground truth is converted once to PathSeer's contiguous ID format.

The default config deliberately uses a nonzero `M_exp_search=32`. This is the fixed-parameter mode documented by PathSeer and does not require a machine-specific virtual-overhead profile. According to the upstream README, this mode exercises PathSeer's first two techniques; full automatic virtual-overhead selection requires separate profiling.

## Outputs

For method `<method>` and sweep `<start>_<step>_<end>`:

```text
<result_root>/<dataset>/index/<index_name>/index_files/index.bin
<result_root>/<dataset>/index/<index_name>/index_files/index.pathseer.json
<result_root>/<dataset>/index/<index_name>/others/pathseer_build.log
<result_root>/<dataset>/index/<index_name>/others/index_build_time.csv
<result_root>/<dataset>/results/<method>/<query_task>_<start>_<step>_<end>/others/<dataset>_search_output.txt
<result_root>/<dataset>/results/<method>/<query_task>_<start>_<step>_<end>/results/<dataset>_pathseer_gt_ids.bin
<result_root>/<dataset>/results/<method>/<query_task>_<start>_<step>_<end>/results/search_time_details.csv
<result_root>/<dataset>/results/<method>/<query_task>_<start>_<step>_<end>/results/query_details_repeat<N>.csv
<result_root>/<dataset>/results/<method>/<query_task>_<start>_<step>_<end>/results/search_time_summary.csv
<result_root>/<dataset>/results/<method>/<query_task>_<start>_<step>_<end>/results/search_time_summary_qps.csv
```

PathSeer performs each configured `efSearch` as one OpenMP batch. Consequently, `Average_Time_ms` is batch wall time, matching the FAVOR adapter. Per-query recall is exact; per-query time and traversal counters in `query_details` are batch averages because the upstream FAISS search API exposes aggregate counters. `Selectivity` is written as `-1` to avoid an extra full dataset scan for every query.

An index is reused across query tasks with the same `index_name`. Change `index_name` after changing build parameters, or remove the old index and manifest before rebuilding.
