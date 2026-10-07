# Historical per-query CSV audit and stratum-analysis feasibility

Audit date: 2026-10-07. Read-only inspection; no benchmark, historical executable run, manuscript edit, or production-code edit. The only new artifact is this report.

Remote repository: `/home/sunyahui/worktrees/FilterVectorCode_multilevel_special`, reached via `ssh -J W300-pub sunyahui@10.77.110.170`. Current source HEAD inspected: `99bfe4c26f812a0c9be3959c7708007d6448e934`. All source locations below are relative to that repository unless an absolute data path is given.

## Main findings

1. Historical CSVs support retrospective grouping by query length, empty/single/multiple-label predicates, and observed per-query Recall/time/distance cost. Use the manifest-selected search budget and warm repeats; do not pool every budget in the CSV.
2. `CandSize` is a trie-node posting-list length for the final sorted query label, not the number of eligible vectors and not exact per-query selectivity. It cannot supply eligible-point strata.
3. All nine workload directories have a `profiled_*.csv` with `coverage_count,labels`. Each has 1,000 rows and its labels match the current query-label file in row order. Joining those annotations by workload identity plus `QueryID` is mechanically feasible. Exact eligible-count claims additionally require the annotation/base-data provenance to be established; this audit did not recompute all intersections.
4. All 95 checked historical command/environment records use the same executable SHA256, `4c99a51c0f74b187d37f7070f73017b6230935126275560fff1061fa90830a81`, through two copied snapshot paths. Both snapshot hashes were verified. This establishes executable identity, not an exact historical source/build association.
5. All 95 environments have `UNG_SPECIAL_LIGHT_STATS=1` and `UNG_DISABLE_ENTRY_ROUTE_STATS=1`. Gated edge-work and entry-coverage counters cannot be interpreted as observed zero work. Treat unavailable instrumentation as missing, not numerical zero.
6. Per-query time is wall duration under the recorded execution conditions. It must not be transformed into batch QPS. QPS must continue to come from the original batch timing records.

## Field definitions and boundaries

The definitions below are directly established from the current source. Their schema and sampled value invariants agree with the historical CSVs. An exact source-to-historical-binary build link was not found, so do not describe all implementation details as independently proven historical binary behavior.

| Field | Definition supported by source | Consequence for analysis |
| --- | --- | --- |
| `Repeat` | Zero-based executable repeat-loop index; writer iterates `repeat < num_repeats`. | `query_details_repeat3.csv` means three repetitions. It does not mean `Repeat=3`. In the inspected manifests, 0 is cold and 1/2 are warm. |
| `Lsearch` | Actual budget from the executable's `Lsearch_list`, repeated for every query. | Select the budget recorded for that manifest crossing before comparing methods. |
| `QueryID` | Zero-based loaded query row index. Writer emits loop index `i`; execution uses that index for query vector, labels, stats, and output. | Joins to line `QueryID+1` of that command's query-label file and same-index vector/GT rows. It is not a dataset-global ID or the value in a `query_group_id_file`. |
| `QuerySize` | `stats.query_length = query_labels.size()`. Labels are loaded from comma-separated rows and sorted. | Empty row gives 0; single/multiple-label grouping is available. Loader does not deduplicate, so unique-label cardinality additionally assumes valid set-valued input. |
| `CandSize` | Empty query: 0. Otherwise `get_candidate_count_for_label(query_labels.back())`, which returns `_label_to_nodes[label].size()`. Trie insertion adds one posting when a trie node is created. | A posting-list node count for the largest sorted label. It neither intersects all query labels nor weights nodes/groups by vector population. Empty query's 0 is especially not eligible-point count. |
| `Recall` | Current helper forms a unique set from the first K GT IDs, excluding sentinel -1, counts returned valid IDs present in that set, then divides by the GT-set size. | Usually Recall@K when there are K distinct valid GT IDs. For fewer than K eligible vectors, normalization equals eligible count only if GT correctly enumerates/pads them. Empty GT has an unguarded zero denominator. CSV stores a scalar, not returned IDs, so regrouping is possible but independent Recall recomputation is not. |
| `Time_ms` | Query wall duration around execution, recorded with `high_resolution_clock`. | Observed per-query cost, including effects of concurrent execution and contention. Not reciprocal throughput. |
| `EntryGroupSearchTime_ms` | `stats.get_min_super_sets_time_ms`. | Recorded entry-group search stage. |
| `EntryPointSetupTime_ms` | `stats.entry_point_setup_time_ms`. | Recorded entry-point setup stage. |
| `BlockAuthorizationTime_ms` | `stats.special_cover_time_ms`. | Recorded authorization stage. |
| `GraphSearchTime_ms` | `max(stats.core_search_time_ms - stats.entry_point_setup_time_ms, 0)`. | Derived stage duration; retain this definition when interpreting it. |
| `ResidualTime_ms` | `max(time_ms - get_min_super_sets_time_ms - special_cover_time_ms - core_search_time_ms, 0)`. | Clipped residual, not a separately timed algorithmic stage. |
| `TotalDistanceCalcs`, `DistCalcs` | Both emit `stats.num_distance_calcs`. | Duplicate total-distance-count representations. Do not add them. |
| `GraphSearchDistanceCalcs` | `min(stats.num_nodes_visited, stats.num_distance_calcs)`. | Derived partition, not an independently logged counter. |
| `EntryPointDistanceCalcs` | Total distance calculations minus the above graph-search portion. | Paired derived partition; do not add it on top of the total. |
| `EntryGroupMatchedPoints` | Optional entry-route coverage statistic. | Unavailable for the checked historical environments because entry-route statistics are disabled. |
| Edge-scan counters and derived ratios | Depend on detailed search instrumentation. | Relevant counters are gated by light-statistics mode; zeros and zero ratios cannot support “no edges scanned” claims. Check individual instrumentation sites before using any other detail field. |

Source evidence:

- `UNG/codes/apps/search_UNG_index.cpp:35–57`: per-query Recall helper. Returned IDs are not deduplicated by the helper itself; valid search-result uniqueness is a separate condition.
- `UNG/codes/apps/search_UNG_index.cpp:60–98`: timing and distance-count derived fields.
- `UNG/codes/apps/search_UNG_index.cpp:346–347`: query data load.
- `UNG/codes/apps/search_UNG_index.cpp:601–677`: repeat/search loop, batch timing, and per-query/batch Recall calculation.
- `UNG/codes/apps/search_UNG_index.cpp:922–1004`: CSV schema and writing order, including Repeat/Lsearch/QueryID and QuerySize/CandSize.
- `UNG/codes/src/uni_nav_graph_query_route.cpp:161–162`: query length and candidate statistic assignments.
- `UNG/codes/src/uni_nav_graph_query_features.cpp:632–634`: candidate accessor forwarding.
- `UNG/codes/include/trie.h:98–104` and `UNG/codes/src/trie.cpp:24–41`: node-posting count and posting construction.
- `UNG/codes/src/storage.cpp:113–145`: sequential vector/label loading and label sorting without deduplication.
- `UNG/codes/src/uni_nav_graph_search.cpp:30–38,124–140`: same-index query access, results, and per-query timing.
- `UNG/codes/src/uni_nav_graph_entry_provider.cpp:13`, `UNG/codes/src/uni_nav_graph_query_features.cpp:468–472`, and `UNG/codes/src/uni_nav_graph_search_backend.cpp:337`: statistics switches and relevant instrumentation.

## Historical evidence checked

Manifest used: `docs/papers/multilevel_ung/generated_data/warm_repeats/warm_repeat_points.csv`.

- All 95 referenced per-query CSVs exist and have the same 86-column schema, including `Repeat`.
- All 95 original batch `search_time_details.csv` SHA256 hashes match those recorded in the warm manifest. This hash check is for batch detail files, not a claim that every per-query CSV has a historical recorded hash.
- All 95 command records specify K=10 and `num_repeats=3`.
- Within each of the nine workloads, query-binary, query-label, and GT paths are consistent across all methods.
- 41 commands use the authoritative-run snapshot path and 54 use the factorial-run snapshot path. Both actual files have the same verified SHA256 above:
  - `runs/authoritative_multilevel_20260926_emptyfix/search/amazon_screen/.binary_snapshots/search_UNG_index.4c99a51c0f74b187d37f7070f73017b6230935126275560fff1061fa90830a81`
  - `runs/base_topology_factorial_20260929/search/amazon_base_trie_4c99/.binary_snapshots/search_UNG_index.4c99a51c0f74b187d37f7070f73017b6230935126275560fff1061fa90830a81`
- Both copies were hashed, never executed. Snapshot mtime and size were inspected but are not source-build proof.
- `runs/base_topology_factorial_20260929/search/amazon_base_trie_4c99/manifest_performance.json` records binary identity and repeat metadata. A concrete matching command/environment/log is under `performance/l1_base_trie_t1024_lng_entry_trie/sel_0p5/`.
- Source history inspection found app commit `7f723b6` on 2026-09-27. Its app changes are only two comment lines. That commit does **not** establish a functional CSV/timing change after the snapshot. Other instrumentation history and the exact source revision used to build the snapshot remain unproven.

Sample CSV: `runs/base_topology_factorial_20260929/search/amazon_base_trie_4c99/performance/l1_base_trie_t1024_lng_entry_trie/sel_0p5/query_details_repeat3.csv`.

- Repeat values are 0,1,2; budgets are 1000,1400,2000.
- Every sampled `(Repeat,Lsearch)` block has 1,000 rows with unique QueryID 0–999. QuerySize/CandSize are invariant across this sample's repeats and budgets.
- First row: QueryID=0, QuerySize=5, CandSize=13780, Recall=1, DistCalcs=1738, EntryPointDistanceCalcs=899, GraphSearchDistanceCalcs=839.
- The aligned existing profile gives coverage_count=2794 for that first query, concretely different from CandSize=13780.
- At Repeat=1/Lsearch=1000, all 1,000 sample records have EntryGroupMatchedPoints=0, RegularEdgesScanned=0, and FreeEdgesScanned=0, consistent with the disabled instrumentation flags.

The historical CSVs total about 485 MB. This was a bounded audit of all headers/command/environment records plus selected data slices, **not** a complete row-integrity scan of all 95 files.

## Existing annotation joins

Base population used in the workload annotations: N=602453. Data root: `/home/graphdb/FilterVectorData/Amazon/`. Every directory below has `Amazon_query_labels.txt`, `Amazon_query.bin`, and the listed `profiled_*.csv`. Corresponding GT is recorded under `/home/graphdb/FilterVectorResult/Amazon/GroundTruth/<query-directory>/Amazon_gt_labels_containment.bin`.

| Workload | Query directory / profile suffix | Coverage-count range | Empty / single / multiple-label rows |
| --- | --- | --- | --- |
| sel_0p5 | `query_minlen5_avgsel05pct` / `profiled_minlen5_avgsel05pct.csv` | 20–10810 | 0 / 0 / 1000 |
| sel_1 | `query_minlen5_avgsel1pct` / `profiled_minlen5_avgsel1pct.csv` | 20–29771 | 0 / 0 / 1000 |
| sel_5 | `query_nested_avgsel5pct` / `profiled_nested_avgsel5pct.csv` | 20–582582 | 0 / 43 / 957 |
| sel_10 | `query_minlen3_avgsel10pct` / `profiled_minlen3_avgsel10pct.csv` | 6045–143527 | 0 / 0 / 1000 |
| sel_30 | `query_nested_avgsel30pct` / `profiled_nested_avgsel30pct.csv` | 8865–582582 | 0 / 70 / 930 |
| sel_60 | `query_nested_avgsel60pct` / `profiled_nested_avgsel60pct.csv` | 794–582582 | 0 / 1000 / 0 |
| sel_80 | `query_minlen1_nested_avgsel80pct` / `profiled_minlen1_nested_avgsel80pct.csv` | 5759–582582 | 0 / 1000 / 0 |
| sel_95 | `query_minlen1_nested_avgsel95pct` / `profiled_minlen1_nested_avgsel95pct.csv` | 5783–582582 | 0 / 1000 / 0 |
| sel_99 | `query_label1_empty_avgsel99pct` / `profiled_label1_empty_avgsel99pct.csv` | 582582–602453 | 697 / 303 / 0 |

Checks completed for all nine workloads:

- Exactly 1,000 current label rows and 1,000 profile rows; labels match in corresponding row order.
- For one manifest-selected warm-repeat crossing per workload, historical QuerySize matches current label-row length for all 1,000 queries.
- The mean historical CSV Recall at each of these nine selected crossings matches the warm manifest to its printed precision.
- Current label/profile files were hashed during inspection. Current row agreement and length agreement do not independently prove that the files have unchanged full label contents since the historical binary run. A recorded historical input hash or archived file remains the strongest identity check.

Annotation-generation evidence:

- `experiments/multilevel_special/generate_intermediate_workloads.py:66–82` validates row-count/label alignment and reads existing `coverage_count` values.
- Its `nested_substitute` and writer at `:86–129` carry those counts forward; `:192–211` takes the label-1 anchor from another existing profile. Thus this script does not independently recompute general multi-label intersections.
- Its `:212–229` constructs empty predicates with count N and label-1 predicates with the anchor's count. Empty predicate means all base points eligible, not zero eligible points.
- `experiments/multilevel_special/generate_high_selectivity_workloads.py:57–67,160–186` counts occurrences of each label across base rows and uses those counts for single-label predicates; empty predicates receive N. These are eligible-point counts if each base label row contains each label at most once. The counting function itself does not deduplicate a row.
- `tools/datasets/select_queries_by_coverage.py:119–142,206–233` reads, filters, and copies pre-existing coverage values; its output retains source profile/row metadata. It does not validate counts by recomputing eligibility.

Accordingly, `coverage_count/N` can be described immediately as **existing annotated selectivity**. To present it as independently verified exact selectivity, establish the upstream profiler's containment semantics, base-file identity, absence of duplicate labels where occurrence counting is used, and relevant historical input hashes, or perform a separately authorized exact-count validation. No such recomputation was performed here.

All inspected annotated counts are at least 20, exceeding K=10. Therefore the current nine annotated workloads do not appear to exercise fewer-than-K eligible vectors or empty GT. This is conditional on annotation correctness; it is not a binary-level proof. The 697 empty predicates at sel_99 have full-dataset eligibility and must not be classed as empty eligible sets.

## Safe, bounded analysis design

1. Use the warm manifest to identify method, workload, detail-file path, and selected crossing `Lsearch`. Form identity from workload/query-file provenance plus QueryID; QueryID alone is unsafe across workloads.
2. Filter each selected CSV to that Lsearch and Repeat 1/2. Validate unique `(Repeat,Lsearch,QueryID)` keys, expected 1,000 IDs, query-length stability, and no missing joined annotations for the actual analysis inputs. Do not treat this audit's sample checks as an all-row validation.
3. Derive empty/single/multiple-label groups from QuerySize, retaining the set-input caveat. For finer label-count bins, use the same definition consistently across methods.
4. If needed, join coverage annotations by query row after verifying their labels and provenance, then compute annotated selectivity as coverage_count/602453. CandSize and truncated top-K GT cannot substitute for exact eligible counts.
5. Aggregate the stored Recall scalar per stratum and repeat. Report stratum sizes. This describes logged retrieval quality; it does not independently re-evaluate returned IDs. A method's globally selected Recall crossing may fall below that threshold in an individual stratum.
6. Summarize observed time/stage/distance costs per stratum and repeat. Keep disabled counters unavailable. Avoid summing duplicate fields and avoid treating derived partitions as independent measurements.
7. Retain batch-QPS results from the original search-time detail files. Never derive QPS from means/sums/reciprocals of per-query duration or stage time.
8. Keep workload-level results separate before any explicit weighted aggregate. Most workloads lack one or more length strata, so an absent stratum is not a zero-cost or zero-Recall result. There are only two measured warm repeats; query count does not create additional independent repeated executions.

No further benchmark or code change is required to establish the feasibility findings in this report. Exact historical source association, historical full input identity, and upstream annotation exactness remain explicit provenance boundaries.
