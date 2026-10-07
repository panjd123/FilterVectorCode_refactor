# Independent C3 data and semantics check

**Scope:** I am a returning targeted auditor checking C3 data, joins, subgroup claims, and source-only reproduction. This is not an overall paper review, score, or new overall recommendation. I did not edit repository sources or frozen data and did not run ANN search, index construction, or the full repository test suite.

**Result:** The C3 scientific checks pass. I independently recomputed the 190,000 exported observations, all 190 batch means, all 570 earlier size histograms, and every metric/status in the 756 size-summary and 2,268 selectivity-summary rows. No numerical or join mismatch was found. The complete source-only command reproduced all six current CSV/TeX/PDF/PNG outputs byte for byte. Two important predicate-diversity qualifications were identified during review and are now present in the Results text.

Repository inspected: `/tmp/filtervector-paper-18h-20261006`.

Independent checker: `/tmp/mlung-c3-independent-data-check.py`.

Machine-readable results, including exact histograms, subgroup counts, zero-ID intersections/unions, source hashes, and reproduction details: `/tmp/mlung-c3-independent-data-check.json`.

## Independent calculation and identity checks

I used a separate standard-library checker that does not import the supplied summarizer. It reads the compressed per-query records directly, validates the Recall values against the 0.0–1.0 tenth-step grid, and accumulates integerized Recall histograms. Selectivity bins were assigned through integer rational comparisons of `eligible_count` and N=602453, independently of the generator's float/bisect implementation.

Verified properties:

- All 9,000 `(workload,QueryID)` coverage keys are unique and exactly cover nine workloads × QueryID 0–999. All predicates are sorted, contain unique labels, and have QuerySize equal to distinct-label count. Repeated predicates have identical coverage values. There are 3,239 distinct predicates across workloads.
- All eligible counts are in [20,602453]. Empty predicates have count N. Selectivity equals eligible_count/N, and all nine exact workload means reproduce the frozen batch-mean selectivity to its recorded precision.
- The full factorial has 126 unique cells: 95 complete crossings and 31 NC cells. The warm-point file contains exactly those 95 complete cells, with matching method and Lsearch. Both warm Recalls at every selected crossing are at least 0.90.
- All 190,000 records match their selected method, Lsearch, repeat, workload, QueryID, and query size. Each of the 190 selected batches contains QueryID 0–999 exactly once. There are no observations invented for NC cells.
- All 190 integer-histogram reconstructions match the frozen warm Recall **exactly**, with maximum absolute error 0.0. The stored `repeat_reconstruction.csv` agrees field by field.
- All 570 old reference histograms agree, including counts, mean Recall, zero counts, and below-0.9 counts. This check also verifies the reference method/capacity metadata. The old reference's timing columns were not recomputed and are not used by the current generator.
- Every current summary row agrees in method/capacity, workload-stratum size, status, mean Recall, zero/below-target counts, and all eleven histogram buckets.
- The published coverage CSV matches the SHA in the earlier exact-coverage audit. The compressed Recall export matches its input hash and the extraction manifest's decompressed-content hash. The two separately created extraction manifests identify the same 95 raw query files with identical SHA256 values. I did not reread those remote wide raw files in this C3 turn.

The three immutable reanalysis input hashes and scientific summary CSVs remained unchanged during this review. The parent concurrently added a presentation table and accompanying prose; the final reproduction comparison uses those updated presentation outputs.

## Five-percent batch: 43 singleton rows

All 43 rows use **the same predicate `{1}`**. Each has eligible_count=582582, or 96.7016514151% individual selectivity, despite the batch mean being 5.0381166664%.

All values below are identical in warm repeats 1 and 2. Zero counts refer to query rows within one repeat.

| Configuration | Selected Lsearch | Mean Recall | Zero Recall | Recall below 0.9 |
| --- | ---: | ---: | ---: | ---: |
| L | 3200 | 0.28837209302325584 | 10 | 39 |
| T | 800 | 0.14883720930232558 | 23 | 41 |
| TL | 400 | 0.7325581395348837 | 5 | 21 |
| TLT | 400 | 0.7441860465116279 | 4 | 19 |

For each method, the zero-Recall IDs themselves also agree across the two executions. TLT's remaining 957 multiple-label queries have mean Recall 0.9264367816091954 and one zero-Recall row per repeat. Thus the batch crossing can coexist with the reported 0.74419 subgroup mean; the whole-batch threshold must not be restated as a per-subgroup guarantee.

The current generated size table (`generated_data/query_strata/generated_query_strata.tex:12–13`) and Results text (`sections/evaluation.tex:162–169`) match these values. The Results now explicitly identifies `{1}`.

## Thirty-percent batch: the 226 rows in (80%,95%]

Exactly 226 query rows fall in this right-closed individual-selectivity bin. **All 226 use the same two-label predicate `{1,2}`**, with eligible_count=548561 and individual selectivity **91.05457189191521%**. This is a repeated-predicate subset at a single selectivity value, not coverage of a diverse set of predicates throughout the 80–95% interval.

| Configuration | Selected Lsearch | Mean Recall, repeat 1 / 2 | Zero Recall, repeat 1 / 2 | Below 0.9, repeat 1 / 2 |
| --- | ---: | --- | --- | --- |
| L | 120000 | 0.7429203539823008 / 0.7163716814159292 | 10 / 9 | 106 / 115 |
| T | NC | unavailable | unavailable | unavailable |
| TL | 60000 | 0.6641592920353983 / same | 8 / same | 150 / same |
| TLT | 60000 | 0.668141592920354 / same | 8 / same | 145 / same |

The eight TL/TLT zero-Recall IDs are persistent within each method across the repeats. L has eight IDs that are zero in both repeats and eleven distinct IDs that are zero in at least one repeat. Therefore the table's 9–10 count range should not be interpreted as exactly the same failed rows in both executions.

The 70 singleton rows elsewhere in this batch have TLT mean Recall 0.9457142857142857, with one zero per repeat. These verified values support the text's statement that QuerySize alone does not locate the difficult subset in these observations.

The current exact-selectivity table (`generated_query_strata.tex:42–43`) and Results (`sections/evaluation.tex:171–182`) match the calculations. Results now names `{1,2}`, gives 91.055%, and explicitly limits inference to other predicates in the same bin.

## Broad-batch zero-Recall tails

Counts below are per 1,000-query warm execution; a slash gives repeat 1 / repeat 2 when they differ.

| Batch | L zeros | T zeros | TL zeros | TLT zeros | TLT batch mean Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| sel_60 | 12 / 11 | 13 | 32 | 31 | 0.9023 |
| sel_80 | 12 | 10 | 27 | 29 | 0.9160 |
| sel_95 | 14 | 12 | 31 | 33 | 0.9091 |
| sel_99 | 12 | 12 | 28 | 29 | 0.9215 |

The requested 31/29/33 TLT versus 11–12/12/14 L comparison is correct. For TLT, all zero-ID sets in these four batches are identical between its two executions. These are descriptive persistent failures in the recorded runs, not estimates of failure probabilities over independent experiments.

The 60%, 80%, and 95% batches consist entirely of single-label predicates, with respectively 388, 179, and 19 distinct predicates. The 99% batch has 697 empty predicates and 303 `{1}` predicates. Within its empty-predicate subset, TLT has mean Recall 0.9253945480631277 and 16 zero-Recall rows in each repeat. Empty query predicates therefore must not be equated with either empty eligible sets or automatically perfect retrieval.

“Zero Recall” here means the exported scalar is zero, i.e. no returned true top-10 neighbor under the recorded Recall computation. It does not establish that the method returned zero vectors, nor diagnose why the failure occurred.

## Empty and NC semantics

The summaries correctly keep two different absence cases:

| Summary | Populated complete rows | Empty-stratum rows | No-crossing rows | Total |
| --- | ---: | ---: | ---: | ---: |
| QuerySize | 248 | 322 | 186 | 756 |
| Exact selectivity | 782 | 928 | 558 | 2268 |

An empty stratum of a completed batch has count=0, blank mean Recall, and zero histogram/count metrics. An NC cell retains the workload's known stratum size but has blank observed count, Recall, and histogram metrics. It is not a zero-Recall experiment. In particular, T at sel_30 supplies no chosen crossing and cannot receive an invented subgroup Recall.

The source implements these distinctions at `experiments/multilevel_special/summarize_query_strata.py:166–187`; the complete input and record joins are at `:66–135`. Histogram cross-checking is at `:139–155`.

## What the source-only reproduction actually establishes

I ran the supplied command against the frozen local files, writing exclusively under `/tmp/mlung-c3-independent-reproduction-complete/`:

```sh
/tmp/filtervector-paper-tools/bin/python -B \
  experiments/multilevel_special/summarize_query_strata.py \
  --data-dir docs/papers/multilevel_ung/generated_data/query_strata \
  --factorial docs/papers/multilevel_ung/generated_data/factorial_equal_recall.csv \
  --warm-points docs/papers/multilevel_ung/generated_data/warm_repeats/warm_repeat_points.csv \
  --output-dir /tmp/mlung-c3-independent-reproduction-complete/data \
  --figure-dir /tmp/mlung-c3-independent-reproduction-complete/figures
```

Environment: Python 3.14.6, NumPy 2.5.3, Matplotlib 3.11.2. The system-Python attempt first completed the scientific CSVs but stopped at the plotting import because Matplotlib was absent; the existing plotting interpreter satisfied the README's stated dependencies and completed successfully without installation or environment changes.

All six reproduced outputs are **byte-identical** to the current published files:

| Output | SHA256 |
| --- | --- |
| `query_size_summary.csv` | `c765fc2d6f2fe4e99092fe70b3b320e45ff01eb6120d9ca09f145ddd1ea40be6` |
| `query_selectivity_summary.csv` | `03590bbe37ed2feff19de9c50002126e06599b636c61b50a1c0aecc2ae387b2f` |
| `repeat_reconstruction.csv` | `00583465f8077e90778021d8f2d2a16779c247b18a5fe0a45c144c7da193bf84` |
| `generated_query_strata.tex` | `80b4d2680ac9f26b07ce064c59dcacbd5458c7d859b9ed9028da9ff93784d755` |
| `query_composition.pdf` | `988b9b917e884c8c10c7181927669b63b799cdbaa0686fd7af0abfd836f64325` |
| `query_composition.png` | `e300456ba652cb81f6cdd7d53f4d55a23baf45148f82ce79a6b7da7ce9834ac7` |

Generator SHA at the successful reproduction:
`886fda37c4573aa5f8eec9830ab66d039728b34dc7c07e08425f3aec0f0e70b2`.

The generated TeX includes both the complete fixed-configuration size table and the two mixed-batch selectivity tables. The selectivity CSV contains the full 14×9×2×9 grid, not just the rows selected for display. The composition figure counts query rows by size/selectivity; it is not itself a Recall histogram or a count of distinct predicates.

This command reproduces regrouping, summaries, and figures **from frozen exported Recall and coverage**. It does not rerun ANN, independently validate returned neighbor IDs, recompute the exact coverage from raw base labels, reproduce subgroup timing from the older reference, or restore missing historical query-content hashes. Those limits are appropriately stated in `generated_data/query_strata/README.md:53–80` and the generated summary manifest. The original 190,000 records are repeated method/execution observations of 9,000 workload/query rows, not 190,000 independent queries or experiments.

## Remaining interpretation limits and small documentation finding

The scientific joins and reported key numbers need no correction based on this audit. The two predicate-diversity cautions are now explicit in Results. Continue to preserve the following scope:

- Grouping by QuerySize or eligible fraction is descriptive. It does not hold predicate identity, vector difficulty, group topology, or search budget fixed, and cannot by itself identify the cause of a Recall loss or a general selectivity crossover rule.
- Every method keeps its own batch-selected Lsearch. These are query-quality distributions at the published operating points, not subgroup-matched Recall, matched-cost comparisons, or subgroup throughput results.
- The source-only corpus validates which frozen point is selected and reproduces its means. It does not contain every budget's query records, so this C3 check alone does not independently prove that the chosen point was the smallest executed budget satisfying the original search protocol.
- At least 20 eligible vectors excludes a true fewer-than-K workload branch here, but the exported scalars and label counts do not independently validate GT-file content. Treat the integerized Recall histograms as a faithful regrouping of the recorded accuracy metric, not a new nearest-neighbor correctness test.
- Current label coverage and historical base SHA agreement do not supply missing historical query-content hashes for sel_0p5, sel_1, and sel_10.

One non-scientific documentation mismatch was reported to the parent: the artifact README at line 67 said `extract_query_recall.py` rejects an existing output directory, whereas the inspected extractor at line 162 used `mkdir(..., exist_ok=True)` and subsequently wrote existing output names. This does not affect the frozen C3 values or source-only summarizer. The audit did not change either file; any parent-side resolution is outside the numerical checks above.

Only the audit script, machine summary, this report, and temporary reproduction outputs were created by this reviewer. No new overall paper recommendation is issued.
