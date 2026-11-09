# Milvus / Knowhere HNSW baseline

This adapter runs the current HNSW implementation bundled by Knowhere, the C++
vector-search core used by Milvus.  It uses Knowhere's filter-aware HNSW
searcher and the Milvus bitset convention: a set bit excludes that vector ID.

Run the Amazon configuration from the repository root:

```bash
python3 experiments/milvus_hnsw/run_milvus_hnsw_experiment.py \
  experiments/milvus_hnsw/Amazon_config.json
```

The runner builds Knowhere's bundled CPU FAISS and the thin standalone adapter,
using the pinned upstream commit recorded in `.knowhere_source_commit`,
then checks the index manifest.  A missing or stale index is rebuilt; a matching
index is reused.  Knowhere's current defaults are preserved in the sample
configuration: `M=30` and `ef_construction=360`.  `add_batch_size=0` adds the
complete base set in one call, matching the normal Knowhere build path.

For containment queries the adapter persists a compact inverted-label sidecar.
It creates a per-query exclusion bitset and searches with Knowhere's current
`kAlpha = filter_ratio * 0.7` policy.  Unless
`disable_fallback_brute_force=true`, it also reproduces the production KNN
fallback thresholds: brute force at 93% or more filtered vectors, or when top-k
is at least half of the eligible or total vectors.

Artifacts follow the existing baseline layout:

```text
<result_root>/<dataset>/index/<index_name>/index_files/index.bin
<result_root>/<dataset>/index/<index_name>/index_files/filters.bin
<result_root>/<dataset>/index/<index_name>/index_files/index.milvus_hnsw.json
<result_root>/<dataset>/index/<index_name>/others/index_build_time.csv
<result_root>/<dataset>/results/<method>/<query_task>_<start>_<step>_<end>/results/
<result_root>/<dataset>/results/<method>/<query_task>_<start>_<step>_<end>/results/<dataset>_milvus_hnsw_gt_ids.bin
```

The runner does not create a dataset-level `milvus_hnsw/` directory.

`index_build_time.csv` records wall-clock build time and resident-memory
accounting after load: vector storage, level-0 links, upper-level links, runtime
graph metadata, and the filter posting index.  Search outputs include the same
summary/QPS/detail CSV family as the FAVOR adapter. `VisitedPoints` is reported as
the core searcher's distance-evaluated point count because this Knowhere searcher does
not expose a separate unique-node counter.  Query details additionally
record `Route` (`hnsw` or `bruteforce`) and exact selectivity.

The standalone build supports `L2`, inner product, and cosine (cosine vectors
are normalized). Jaccard is intentionally rejected because this experiment
adapter only targets Milvus HNSW on dense float vectors.

Search timing includes containment-filter preparation. For every query in every
`ef`/repeat batch, the adapter intersects the label postings and materializes
the Milvus exclusion bitset after both the per-query and batch timers start.

