# RWalks baseline

This adapter runs the official RWalks HNSW implementation on the project's
`.fvecs`, label-text, and containment-ground-truth files.

```bash
python3 experiments/rwalks/run_rwalks_experiment.py \
  experiments/rwalks/Amazon_config.json
```

The runner configures and builds the native adapter, checks the index plus its
attribute sidecar and manifest, builds only when one is missing or incompatible,
then runs the configured `ef` sweep. Index artifacts are written below
`<result_root>/<dataset>/index/<index_name>/`; search outputs use the normal
`<result_root>/<dataset>/results/<method>/<query_task>_.../results/` layout.
The converted ground-truth ID file is kept in that same per-run `results/`
directory; the runner does not create a dataset-level `rwalks/` directory.

The defaults match the official reproduction code: `M=16`,
`ef_construction=100`, random-walk depth 3, 20 walks, search mode 0, and pruning
factor 0.0. Build logs and `others/index_build_time.csv` record wall-clock build
time and an allocation-based in-memory index estimate after construction,
including RWalks' raw and diffused attribute arrays.

The build adapter reloads both serialized files before reporting the memory fields,
so the recorded size describes the resident index representation used by search.
The random-walk pass is deterministic and thread-safe; it preserves the upstream
diffusion/normalization rules while fixing its neighbor-array offset bug and
per-walk `random_device` bottleneck.

## Compact attribute schema

The upstream index stores every attribute twice as dense arrays (`int` plus
`float`). Amazon has 30,723 attributes, which would require about 148 GiB just
for those two arrays. The default `attribute_scope=configured_queries` therefore
uses the union of labels referenced by every configured query task sharing an
index. Removing labels that no configured filter reads does not change those
queries' containment predicate or their RWalks scores. The exact label list and
SHA-256 digest are saved in the index schema/manifest, so adding a task or label
invalidates and rebuilds the index automatically.

Set `attribute_scope` to `current_query` for one workload per index, or to
`all_base_labels` only when the dataset has a small attribute vocabulary and
enough memory.

The native search adapter writes zeros for graph-internal visited/distance
counters because the upstream multithreaded API does not expose thread-safe
per-query counters. Its timing fields follow the FAVOR adapter's convention:
`search_time_details.csv` `Time_ms` is the end-to-end wall time for one complete
query batch at an `ef` value, and `search_time_summary.csv` `Average_Time_ms`
averages that batch wall time across repeats. `QPS` is computed as
`query_count * 1000 / batch_time_ms`. The timed region includes parallel search,
per-query recall evaluation, and staging the per-query result rows; it excludes
input/index loading and CSV output. `query_details_repeat*.csv` `SearchTime_ms`
is the measured `searchKnn` time of that individual query. Those per-query times
overlap in a multithreaded run and therefore must not be summed to obtain the
batch wall time.
