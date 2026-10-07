# C3 exact query-coverage validation

Completed 2026-10-07 07:44 Asia/Shanghai. Status: **PASS**. This independently resolves the prior audit's unverified `profiled_*.csv` coverage-count boundary for the current nine query-label inputs and the historically matched Amazon base-label file.

All **9,000 query rows** match their profile's `coverage_count` exactly. They contain **3,239 unique AND predicates**. None of the **602,453 base rows** or **9,000 query rows** contains duplicate labels. The current base-label SHA256 matches the provenance recorded for **all 95 frozen crossings**. No vector distances or ANN searches were computed; existing input, profile, manuscript, and implementation files were not modified.

## Result entry point

Remote output directory:

`/home/sunyahui/worktrees/FilterVectorCode_multilevel_special/runs/paper_c3_20261007/coverage_validation/`

The join-ready file is **`query_coverage_exact.csv`**, containing:

```text
workload,QueryID,QuerySize,labels,eligible_count,true_selectivity
```

- Join with historical per-query observations on `(workload, QueryID)`, after choosing the frozen crossing's Lsearch and warm Repeat 1/2.
- `QueryID` is zero-based row order within each workload's recorded query file.
- `QuerySize` preserves the original token count. All query rows have unique labels, so it also equals distinct-label count in these inputs.
- `labels` is the sorted, comma-separated predicate, properly CSV-quoted when necessary. Empty predicates have an empty field.
- `eligible_count` is independently computed exact AND coverage over base-label sets.
- `true_selectivity = eligible_count / 602453`, expressed as a fraction in [0,1], not percent.
- The output has exactly 9,000 unique `(workload,QueryID)` keys, and every workload contains QueryID 0–999 exactly once. These properties and all selectivity ratios were rechecked after writing.

Join CSV SHA256:

`29c00ef5f85a7ae61a5565f85e4362f575e4c236ca5c05c1fc11dda213a294a8`

Both `/root` and `/root/paper_structure_review` have been notified of this entry point and its passing status.

## Per-workload results

Each workload contains 1,000 queries. “Duplicate rows” counts query-label rows for which `len(tokens) != len(set(tokens))`; all are zero. Coverage mismatches include either unequal exact/profile counts or unequal predicate labels; all are zero.

| Workload | Unique predicates | Query duplicate rows | Empty / single / multi | Exact eligible-count range | Exact mean selectivity (%) | Profile mismatches |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| sel_0p5 | 1000 | 0 | 0 / 0 / 1000 | 20–10810 | 0.4992490701 | 0 |
| sel_1 | 930 | 0 | 0 / 0 / 1000 | 20–29771 | 0.9030548441 | 0 |
| sel_5 | 888 | 0 | 0 / 43 / 957 | 20–582582 | 5.0381166664 | 0 |
| sel_10 | 660 | 0 | 0 / 0 / 1000 | 6045–143527 | 9.9066152878 | 0 |
| sel_30 | 706 | 0 | 0 / 70 / 930 | 8865–582582 | 30.0272422911 | 0 |
| sel_60 | 388 | 0 | 0 / 1000 / 0 | 794–582582 | 60.0474906756 | 0 |
| sel_80 | 179 | 0 | 0 / 1000 / 0 | 5759–582582 | 80.0236992761 | 0 |
| sel_95 | 19 | 0 | 0 / 1000 / 0 | 5783–582582 | 95.0197806302 | 0 |
| sel_99 | 2 | 0 | 697 / 303 / 0 | 582582–602453 | 99.0006003788 | 0 |

Unique predicate counts are per workload and therefore do not sum to the cross-workload count of 3,239. There are 5,761 cached predicate reuses across 9,000 rows.

No query has zero eligible points or fewer than K=10 eligible points; the overall minimum is 20. The 697 empty predicates all have eligible_count=602453. Empty predicates must continue to be distinguished from empty eligible sets.

## Input identity and provenance

Base input: `/home/graphdb/FilterVectorData/Amazon/Amazon_base_labels.txt`, 21,404,200 bytes, 602,453 rows, 30,723 distinct labels, zero empty rows, zero duplicate-label rows or duplicate occurrences.

Base SHA256:

`aec768bba7092af445252835be3f1ef7f708305ea646af19ae72738df3dd2f96`

For every frozen crossing, the audit read its historical command, located the matching run-directory record in `manifest_performance.json`, and compared that record's `provenance.base_labels_sha256` with the current base file. All **95/95 matched**. The two source manifests are:

- `runs/authoritative_multilevel_20260926_emptyfix/search/amazon_screen/manifest_performance.json`
- `runs/base_topology_factorial_20260929/search/amazon_base_trie_4c99/manifest_performance.json`

Paths are relative to `/home/sunyahui/worktrees/FilterVectorCode_multilevel_special`. Full per-crossing evidence is in `historical_provenance.csv`, including manifest hashes, expected/observed base hashes, command-derived query/GT paths, and matching record counts.

Query directories, relative to `/home/graphdb/FilterVectorData/Amazon/`, are:

| Workload | Query directory | Current generation manifest available |
| --- | --- | --- |
| sel_0p5 | `query_minlen5_avgsel05pct` | No |
| sel_1 | `query_minlen5_avgsel1pct` | No |
| sel_5 | `query_nested_avgsel5pct` | Yes |
| sel_10 | `query_minlen3_avgsel10pct` | No |
| sel_30 | `query_nested_avgsel30pct` | Yes |
| sel_60 | `query_nested_avgsel60pct` | Yes |
| sel_80 | `query_minlen1_nested_avgsel80pct` | Yes |
| sel_95 | `query_minlen1_nested_avgsel95pct` | Yes |
| sel_99 | `query_label1_empty_avgsel99pct` | Yes |

Every directory's `Amazon_query_labels.txt` was checked against its single `profiled_*.csv` in row order. Historical commands agree on query-label, query-vector, and GT paths within each workload. All **24/24 available generation-manifest hash comparisons** passed: query labels, profile, query binary, and GT, for each of the six workloads with manifests. Query binary and GT files were hashed only, not used to calculate coverage or distances.

`source_hashes.json` records **142 source artifacts**, including the base labels, all nine query labels/profiles, companion binaries and GT, generation manifests, 95 historical commands, two historical run manifests, the frozen-crossing CSV, and the audit script. Every recorded source was hashed before use and rehashed after the audit; **none changed**. All source SHA256 values are retained there rather than abbreviated in this report.

The residual historical boundary is specific: current query labels are proven to have the counts in this output, and the base bytes match original run provenance. The available generation manifests corroborate six workloads' input bytes, but a complete set of query-content hashes tied directly to every historical search execution was not found. In particular, sel_0p5, sel_1, and sel_10 lack a current generation manifest. This audit does not manufacture that missing archival link. The earlier uncertainty about which exact source revision built the historical executable also remains separate from this label-only validation.

## Exact-count method and checks

The script reads query rows first, canonicalizes each predicate as a sorted set, and allocates one base-row bitmap for each of the 443 labels used by these queries. It scans every base row once, checks duplicate tokens, and sets a bit in each applicable label's bitmap. Each unique query's eligible count is the population count of the bitwise AND of its label bitmaps. The empty AND is the entire base population. Results are cached by the full canonical predicate and expanded back to each workload/QueryID row.

This is exact Boolean containment. It uses no statistical estimate, top-K truncation, vector distance, graph traversal, or ANN result. Bitmap construction uses 33,361,001 bytes. The process peak RSS was 68,832 KiB (about 67.2 MiB).

Checks performed:

- All 443 single-label bitmap population counts match separately accumulated base-row label frequencies.
- Twelve deterministic predicates spread across the unique-predicate list were also counted using direct Python set inclusion over **all** base rows, independently of bitmap intersections; all match. This includes the empty predicate.
- All 9,000 exact counts and predicate labels match their respective profile rows.
- All 9,000 output identities and ratios pass the post-write integrity checks noted above.
- Mismatch evidence is retained in `profile_mismatches.csv`, which contains only its header because there were no mismatches.

The actual audit process took **6.61 seconds**. It had an enforced 1,800-second overall timeout, within the 3,300-second per-case limit. Fast completion reflects label-only set operations and cached predicates; this duration is not a search performance benchmark.

## Artifacts and reproduction

The original script is `/tmp/mlung-c3-verify-query-coverage.py` both locally and on the remote host. Copies of the script and this report are also archived in the remote output directory as `mlung-c3-verify-query-coverage.py` and `mlung-c3-coverage-validation.md`. Its SHA256 is:

`4d7daae141b21e1aa1725ff1169840fd17158bddfaa2f7570c68fea3aeb6d232`

Other output files:

- `audit_summary.json`: status, counts, resource use, input references, and output hashes.
- `source_hashes.json`: full source identity and pre/post hash stability.
- `historical_provenance.csv`: 95 frozen-crossing provenance comparisons.
- `generation_manifest_checks.json`: 24 recorded-input hash comparisons.
- `profile_comparison.csv`: all 9,000 exact/profile pairs and deltas.
- `profile_mismatches.csv`: preserved mismatch rows, empty in this run.
- `workload_summary.csv`: numeric per-workload results.
- `duplicate_label_audit.json`: complete base/query duplicate findings.
- `independent_direct_checks.csv`: twelve direct full-scan cross-checks.

`audit_summary.json` SHA256: `875cf69db2329a5c496561e2e189db036b0baef9cd11e8447177eca9a0f6b58e`.

Reproduce on the remote host with the original script and a **new output directory**; it deliberately refuses to overwrite an existing directory:

```sh
python3 /tmp/mlung-c3-verify-query-coverage.py \
  --repo /home/sunyahui/worktrees/FilterVectorCode_multilevel_special \
  --data-root /home/graphdb/FilterVectorData/Amazon \
  --gt-root /home/graphdb/FilterVectorResult/Amazon/GroundTruth \
  --output /home/sunyahui/worktrees/FilterVectorCode_multilevel_special/runs/paper_c3_20261007/coverage_validation_recheck \
  --timeout-seconds 1800
```

The new exact coverage file can now support actual per-query selectivity strata. This does not alter the interpretation of existing time or Recall observations: single-query time must not become batch QPS, and a globally selected Recall crossing may still miss the threshold for a subgroup. The separately reported TLT singleton result at sel_5 should therefore remain explicit in the C3 analysis; coverage validation does not remove that subgroup discrepancy.
