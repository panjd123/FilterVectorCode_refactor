# Query composition and Recall strata

These artifacts regroup existing query results at the 95 frozen crossings.
No ANN search or index construction was rerun. Each crossing contributes two
warm executions of the same 1,000 queries, giving 190,000 exported Recall
scalars. A separate label-only pass computes exact eligible counts for the
9,000 workload/query-ID pairs.

## Reproduce without external data

From the repository root, with Python ≥3.9, NumPy and Matplotlib ≥3.6:

```bash
python3 experiments/multilevel_special/summarize_query_strata.py \
  --data-dir docs/papers/multilevel_ung/generated_data/query_strata \
  --factorial docs/papers/multilevel_ung/generated_data/factorial_equal_recall.csv \
  --warm-points docs/papers/multilevel_ung/generated_data/warm_repeats/warm_repeat_points.csv \
  --output-dir /tmp/mlung-query-strata \
  --figure-dir /tmp/mlung-query-strata-figures
```

The command validates frozen input hashes, joins by workload and zero-based
QueryID, checks the selected method/capacity/repeat, and reconstructs every
warm batch mean from integer Recall@10 hit counts. It also compares all 570
predicate-size histograms with a separate earlier extraction, including
empty subsets. CSV/TeX outputs are deterministic; PDF/PNG rendering can vary
with Matplotlib or font versions. The output manifest records exact hashes.

## Contents and meanings

| File | Meaning |
|---|---|
| `query_recall.csv.gz` | 190,000 selected rows: workload, method, topology, capacity, repeat, QueryID, loaded label count, exported Recall. Deterministic gzip, with filename and timestamp omitted. |
| `query_coverage_exact.csv` | 9,000 exact AND coverages over 602,453 base label rows. Empty predicates admit the entire corpus. Every eligible count is at least 20. |
| `query_size_reference.csv` | Byte-preserved earlier extraction of 570 size histograms. Its concurrent per-query timing columns are not used in the current analysis and do not measure subgroup QPS. |
| `query_size_summary.csv` | All 14 configurations × 9 workloads × 2 repeats × 3 size strata. |
| `query_selectivity_summary.csv` | The same grid, with nine right-closed bins of exact individual-query selectivity. |
| `repeat_reconstruction.csv` | The 190 reconstructed means and differences from the frozen batch summaries. |
| `generated_query_strata.tex` | Complete fixed-configuration size table and selectivity tables for the two mixed batches discussed in the paper. |
| `input_hashes.json` | Expected hashes of the three immutable reanalysis inputs. |
| `recall_extraction_manifest.json` | Original raw-query file hashes, extraction checks and compressed-output hash. The original output basename was `query_recall_records.csv.gz`; published bytes are identical. |
| `size_extraction_manifest.json` | Provenance for the separate earlier size extraction. |
| `coverage_audit.json` | Exact label-coverage validation, historical base identity, source hashes and available workload-generation checks. |
| `summary_manifest.json` | Current source-only reproduction checks and output hashes. |

In the summary CSVs, `workload_stratum_count` always records the query subset
size. A successful crossing has `complete` rows for populated strata and
`empty_stratum` rows for unpopulated ones. An empty stratum has count zero
and blank mean Recall. `no_crossing` retains NC without inventing a capacity,
observations, or zero-Recall result. This distinction is essential when a
method fails the batch threshold.

`mean_recall` is a fraction, while the TeX tables display percent.
`recall_zero` and `recall_below_0p9` are query counts within one repeat.
Histograms have the eleven Recall@10 values 0.0 through 1.0. A query belongs
to the same subset across methods and repeats; capacities remain those
selected by each method's original **batch-mean** crossing. These results do
not establish subgroup-matched Recall or subgroup throughput. A zero-Recall
query has no returned true top-10 neighbor; it does not necessarily return
no vectors. Query labels in this corpus have no duplicates, so loaded label
count equals distinct predicate size.

## Re-extract from external raw artifacts

`experiments/multilevel_special/extract_query_recall.py --repo REPO --output NEW_DIR`
reads the existing raw query-detail files under that repository's `runs/`.
It rejects an existing output directory and does not launch search.
`experiments/multilevel_special/verify_query_coverage.py --help` describes the
separate label-only coverage audit; it requires the original label, workload,
profile, ground-truth-file and provenance paths, but does not evaluate vector
distances or recompute nearest neighbors. The original extraction manifests
identify the exact inputs used for this publication.

The coverage pass matches the base-label hash in all 95 historical crossing
records. Six workload-generation manifests corroborate 24 input hashes.
The pre-existing 0.5%, 1%, and 10% workloads lack generation manifests;
current labels and historical commands establish the files audited here,
but not a complete historical query-content hash chain. Regrouping exported
Recall also does not independently validate the ANN outputs against exact
nearest neighbors.
