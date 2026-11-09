# FAVOR Experiment

This runner adapts the upstream FAVOR command line workflow to this repo's UNG-style experiment layout.

FAVOR is vendored in this project at `FAVOR/`. The expected binaries are:

- `FAVOR/build/app/build_index`
- `FAVOR/build/app/search`
- `FAVOR/build/app/search_sweep`

By default, `run_favor_experiment.py` runs an incremental CMake configure/build before the experiment, so `FAVOR/build/app/build_index`, `FAVOR/build/app/search`, and `FAVOR/build/app/search_sweep` stay up to date. Set `auto_build_favor` to `false` to skip this step.

## Run

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
python3 experiments/favor/run_favor_experiment.py experiments/favor/config.json
```

## Config Layout

`experiments/favor/config.json` mirrors the UNG experiment style:

- `result_root`, `data_root`: same meaning as UNG configs.
- `favor_source_dir`, `favor_build_dir`: project-local FAVOR source/build paths.
- `build`: build-time hyperparameters. `num_threads`, `M`/`max_degree`, and `ef_construction`/`Lbuild` are passed to `FAVOR/build/app/build_index`. UNG-compatible fields such as `data_type`, `dist_fn`, `scenario`, `alpha`, and `num_cross_edges` are kept in the config for experiment parity and logging context.
- `search`: FAVOR search hyperparameters. `K`, `num_repeats`, `num_threads`, `lsearch_start`, `lsearch_step`, and `lsearch_end` drive the FAVOR search sweep. The runner maps each `Lsearch` value directly to FAVOR's `ef` argument.
- `datasets`: each item needs `dataset` and `query_task`; it may override `search` or input/output file paths.
- `methods`: each item needs `name` and `index_name`; it may also set `ef_values` to override the `lsearch_*` sweep.
- `prefilter_selectivity_thresholds`: optional list of selectivity thresholds used to expand FAVOR runs. A value such as `0.05` writes results under method name `favor_0.05` and passes that threshold to `search_sweep`. Without this list, FAVOR keeps its default `0.01` threshold.

## Output Layout

For dataset `<Dataset>`, method `<Method>`, and query task `<QueryTask>` with search range `<start>_<step>_<end>`, outputs match the UNG result layout:

```text
<result_root>/<Dataset>/favor/<QueryTask>/<Dataset>_favor_gt_ids.bin
<result_root>/<Dataset>/index/<index_name>/index_files/index.bin
<result_root>/<Dataset>/index/<index_name>/index_files/index.bin.labels
<result_root>/<Dataset>/index/<index_name>/others/favor_build.log
<result_root>/<Dataset>/index/<index_name>/others/index_build_time.csv
<result_root>/<Dataset>/results/<Method>/<QueryTask>_<start>_<step>_<end>/others/<Dataset>_search_output.txt
# With threshold sweep, <Method> is favor_0.05, favor_0.1, etc.
<result_root>/<Dataset>/results/<Method>/<QueryTask>_<start>_<step>_<end>/results/search_time_details.csv
<result_root>/<Dataset>/results/<Method>/<QueryTask>_<start>_<step>_<end>/results/query_details_repeat<num_repeats>.csv
<result_root>/<Dataset>/results/<Method>/<QueryTask>_<start>_<step>_<end>/results/search_time_summary.csv
<result_root>/<Dataset>/results/<Method>/<QueryTask>_<start>_<step>_<end>/results/search_time_summary_qps.csv
```

`search_time_summary.csv` uses the same columns as UNG:

```text
Lsearch,Average_Efs,Average_Time_ms,Average_Recall,Average_VisitedPoints,Average_VisitedEdges,Average_DistanceComputations
```

`search_time_summary_qps.csv` adds:

```text
QPS
```

For FAVOR, `Lsearch` and `Average_Efs` both record the FAVOR `ef` value used for that row.

`index_build_time.csv` records one row for each completed FAVOR index build with build time and the resident index-memory accounting: `index_memory_bytes`, `index_memory_mib`, `index_level0_bytes`, `index_upper_level_bytes`, `index_runtime_metadata_bytes`, and `index_attribute_roaring_bytes`. The Roaring value includes bitmap-object storage plus each bitmap's portable serialized size as a stable compressed-size estimate; these fields are not the on-disk file size.

FAVOR search is executed as a single sweep process: vectors, attributes, conditions, ground truth, and index are loaded once, then all configured `ef` values are queried in-process.
The per-query details CSV contains `QueryID,Lsearch,SearchTime_ms,Recall,Route,Selectivity,VisitedPoints,VisitedEdges,DistanceComputations`. `Route` is `prefilter` when FAVOR uses the brute-force filtered path at or below the configured selectivity threshold and `graph` otherwise. `VisitedPoints` counts data-vector visits, `VisitedEdges` counts scanned adjacency entries (zero for prefilter brute force), and `DistanceComputations` counts vector-distance evaluations. `Average_Time_ms` is the end-to-end wall time of the complete query batch, averaged across repeats when configured, and `QPS` is computed from that batch time. Recall and visited/distance fields are per-query averages. `query_details_repeat*.csv` `SearchTime_ms` is the measured time of the individual query; those times overlap in a multithreaded run and must not be summed to obtain the batch wall time.


The shared UNG ground-truth files store `(id, distance)` pairs. Before invoking FAVOR search, the runner materializes `<Dataset>_favor_gt_ids.bin` with only the vector IDs because FAVOR's native evaluator reads ground truth as contiguous integer IDs.

## Label Conversion And Roaring Index

The runner builds one shared schema from every label appearing in the dataset's base-label file. FAVOR stores attributes as an inverted CRoaring index: each schema label owns a compressed bitmap of matching HNSW internal IDs. Labels are no longer materialized as an `num_vectors × num_labels` dense table and are not embedded in each level-0 HNSW record.

During parallel graph construction, internal IDs can differ from base-row IDs. The builder records the returned internal ID for every row and remaps each label bitmap after graph construction, one bitmap at a time. This keeps peak attribute memory close to one compressed index instead of retaining dense build and node copies. Low-selectivity prefilter searches use Roaring intersection/difference to produce candidate IDs directly; graph search uses bitmap membership checks for visited nodes.

The sidecar `index.bin.labels` persists the sorted label IDs and is loaded by every query task, so a `FAVOR` index is reusable across query tasks for the same dataset. The runner writes one filter expression per query from this project's containment labels:

- base label `1,3` inserts the vector ID into the Roaring bitmaps for labels `1` and `3`.
- query label `1,3` becomes `label_1 == 1 AND label_3 == 1`.

The `FAVROA01` format is intentionally incompatible with legacy dense/bitset FAVOR indexes. The runner checks the file magic and rebuilds an obsolete index instead of attempting to load it.
