# UNG Search Comparison

This experiment compares four search routes on the existing `UNG` and
`UNG_special_blocks` indexes for Amazon, Genome, Reviews, and VariousImg.

Run from the repository root:

```bash
./run_search_comparison_experiment.sh
```

The runner checks whether each query task has a containment ground-truth file
under:

```text
/home/dev/graphdb/FilterVectorResult/<Dataset>/GroundTruth/<QueryTaskName>/
```

If the GT file is missing, it computes it first. Search outputs are written to:

```text
/home/dev/graphdb/FilterVectorResult/<Dataset>/results/<Method>/<QueryTaskName>/
```

Each method result contains the raw `search_UNG_index` CSV files plus
`search_time_summary_qps.csv`, which adds `QPS = num_queries * 1000 /
Average_Time_ms`.
