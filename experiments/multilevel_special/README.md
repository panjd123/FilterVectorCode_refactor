# Multilevel Special Block experiments

Use `experiment_cli.py` as the stable front door. New experiments use the
`orthogonal_v2` schema and declare three independent dimensions: the base and
per-layer group topology, the hierarchy thresholds, and the entry-group
strategy. The older `generate_*` and `summarize_*` scripts remain only as
reproducibility adapters for historical experiment families.

```bash
# Fail fast on method/protocol inconsistencies.
python3 experiment_cli.py check config.json

# Print orthogonal method semantics (base topology, layers, entry strategy).
python3 experiment_cli.py matrix config.json

# Execute or inspect commands without executing them.
python3 experiment_cli.py run config.json
python3 experiment_cli.py run config.json --dry-run

# Audit manifest, binary hash, input provenance, repeats, Recall, and stages.
python3 experiment_cli.py validate config.json

# Select the smallest measured Recall crossing and report warm statistics.
python3 experiment_cli.py summarize config.json --baseline method_name

# Inspect useful completed cases while a long matrix is still in progress.
python3 experiment_cli.py summarize config.json --baseline method_name --allow-partial

# Decompose UNG/plain speedup into ELS, graph work, entries, and visits.
python3 analyze_ung_plain_attribution.py config.ung_plain_formal.json
```

## Configuration contract

Every new config should include a top-level `protocol` object:

```json
{
  "phase": "screen",
  "cold_repeats": 1,
  "measured_repeats": 2,
  "recall_rule": "all_repeats",
  "bootstrap_samples": 10000,
  "paired_repeats": false
}
```

`paired_repeats=true` is accepted only when an explicit interleaved
`execution_blocks` schedule is declared. Sequential method runs are independent
samples and must not be described as paired.
Both performance statistics and the configured Recall rule apply only to
measured repeats; declared cold repeats are retained for audit but excluded.

An `orthogonal_v2` method must explicitly name `main_index`, `base_topology`,
`entry_strategy`, `hierarchy_layers`, and `special_block_search`. Each hierarchy
layer has a positive, strictly increasing `min_points` threshold and an
independent `lng` or `trie` topology. The only legal entry strategies are
`original`, `optimized_lng`, and `trie`. Do not infer semantics from names such
as `plain`, `UNG`, or `Trie`; use `experiment_cli.py matrix` in reports and
audits.

The search invariant is per candidate state: a base candidate scans only base
edges, and a candidate activated at materialized layer `i` scans only edges
owned by layer `i`. There is no edge fallthrough or implicit promotion. All
query-authorized layers are seeded independently into one bounded candidate
queue, so this invariant does not imply that adding a layer is latency-neutral.

## Authoritative Amazon campaign

Build the two zero-layer base topologies and the declared hierarchy grid first.
The campaign driver then runs a broad Recall screen, measured crossing
refinement, a 1-cold plus 15-warm formal pass, and an independent instrumented
profile pass. Every stage is resumable and validated before the next begins.

```bash
python3 run_base_topology_build.py config.authoritative_amazon_base_topologies.json
python3 run_build_sweep.py config.authoritative_amazon_hierarchy_grid.json
python3 run_authoritative_campaign.py
```

If a screen was intentionally launched with `--stop-after screen`, run
`continue_after_screen.py` as a detached guard. It waits for that tmux session
to exit, validates the complete screen, refuses to continue while any campaign
process remains, and only then starts crossing through profile in a fresh
session. This avoids injecting a command into a pane that disappears when its
original non-interactive shell exits.

Campaign config paths are explicit overrides on both the driver and watcher.
Use `generate_authoritative_campaign.py --output-root ...` to start a clean
evidence directory while reusing immutable base and hierarchy indexes; pass the
matching `--screen-config`, `--crossing-config`, `--formal-config`, and
`--profile-config` paths to continuation jobs. This prevents a corrected binary
from silently resuming results captured by an older executable.

The generated screen contains all six zero-layer combinations of two base
topologies and three entry strategies. Each declared hierarchy plan is also
crossed with all three entry strategies. Two additional, separately labelled
controls apply the query-independent `require_upper_authorization` route to
the DRH-v1 and mass-ladder plans; they are not counted as factorial cells.
Performance runs set
`UNG_SPECIAL_LIGHT_STATS=1`; profile runs set `UNG_SPECIAL_LIGHT_STATS=0` and
`UNG_SPECIAL_PROFILE_TIMING=1`. Profile timing explains mechanism and is not
mixed into the primary QPS table. With `pass_subdirs=true`, derived tables are
also isolated under `summary/performance` and `summary/profile` so the profile
pass cannot overwrite the primary performance summary.

Crossing performs one declared measured refinement (or one bounded upper-grid
extension). A method/workload that still has no all-repeat Recall crossing is
excluded from formal timing and recorded with its maximum measured L and
Recall; results are never interpolated or extrapolated. Selected zero-layer
Trie high-selectivity guards reach `L=N`, allowing those cases to distinguish
a genuinely exhausted full-search budget from an ordinary bounded screen.

`summarize_selection_sweep.py` writes both machine-readable CSVs and a Markdown
report. The Markdown operating-point table uses the conservative crossing and
includes the warm layered-path activation rate, stage timing, and
visited-point, scanned-edge, distance-calculation, and entry-count breakdowns.
Recall/QPS figures are generated only from measured screen points:

```bash
python3 plot_authoritative_recall_qps.py \
  config.authoritative_amazon_screen_emptyfix.json \
  ../../runs/authoritative_multilevel_20260926_emptyfix/search/amazon_screen/summary/performance/all_points.csv \
  ../../runs/authoritative_multilevel_20260926_emptyfix/search/amazon_screen/summary/performance/figures
```

The plotter produces PDF and PNG panels for the principal zero-layer
comparison, fixed-entry topology, depth, two-layer topology, entry strategy,
and upper-authorization ablations. It fails closed on missing
method/workload results unless `--allow-partial` is explicitly supplied and
records input hashes and missing cases in `plot_manifest.json`.

After the formal pass, generate the requested plain versus best one-layer
versus best two-layer table with conservative crossings and bootstrap
intervals:

```bash
python3 summarize_depth_ablation.py \
  config.authoritative_amazon_formal_emptyfix.json \
  --output-dir results_summary/authoritative_depth
```

Per-workload best rows are measured oracles. The companion global table only
accepts a single unchanged method that crosses on every workload; routed DRH
is reported separately from the always-layered two-layer factorial.

## Query-free held-out hierarchy study

`generate_auto_policy_cross_dataset_configs.py` derives DRH-v1 only from
`N`, `R`, and `C` and combines it with the fixed exact upper-authorization
gate. It freezes a 36-case one/two/three-layer manual grid before reading
held-out query results. Every manual candidate uses the same gate, so the
oracle tunes only depth, thresholds, and per-layer topology; ungated DRH is a
separate ablation. The generator emits build, screen, and policy files for
Genome, Reviews, and VariousImg. Build commands can be preflighted without
starting construction:

```bash
python3 generate_auto_policy_cross_dataset_configs.py
python3 run_build_sweep.py config.authoritative_heldout_genome_build.json --dry-run
python3 run_heldout_campaign.py
```

After the held-out screen/crossing/formal pipeline completes, compare DRH with
the fastest predeclared manual hierarchy at each workload and with the best
single hierarchy used across all workloads:

```bash
python3 summarize_heldout_oracle.py \
  config.authoritative_heldout_genome_formal.json \
  config.authoritative_heldout_reviews_formal.json \
  config.authoritative_heldout_variousimg_formal.json \
  --output-dir results_summary/heldout_oracle
```

The summarizer selects the smallest measured `Lsearch` whose every warm repeat
meets the target, uses independent-sample bootstrap intervals for sequential
method runs, and never interpolates a crossing.

For unattended serial execution, `continue_after_query.py` waits until the
Amazon profile config exists and its full validator passes, then waits for all
query processes to exit before starting held-out work. The optional build step
is joined with `&&`, so it starts only if every held-out dataset and oracle
summary succeeds:

```bash
python3 continue_after_query.py --run-build-after-heldout
```

This supervisor gates on the profile manifest and validator rather than a tmux
session transition, preventing an accidental launch during the brief
screen-to-crossing handoff.

## Authoritative build campaign

Run this only after the query campaign has frozen the automatic and manual
structures. `AUTHORITATIVE_BUILD_PROTOCOL.md` defines the claim boundary and
required quality checks. The generator creates separate unmonitored timing and
monitored resource passes; backend variants are interleaved by repeat.
Hierarchy cases consume measured repeat 0 from this campaign's
`accelerated_gpu` base build rather than an unrelated pre-existing index.

```bash
python3 generate_authoritative_build_configs.py --repeats 5
python3 run_authoritative_build_campaign.py --repeats 5
```

The base matrix compares `original_cpu`, `current_cpu`, the named GPU profiles,
and the exact accelerated profile used to build the query baseline. The
hierarchy matrix holds thresholds and topology fixed while changing only
layer-local and inter-block construction backends. The runner rejects a GPU
case if its persisted profile differs, a requested GPU stage is not observed,
or a CPU fallback is observed. `process_resource_probe.py` samples process-tree
RSS and per-process GPU memory only in the resource pass.

After timing and resource measurement, the driver searches representative
repeat-0 CPU/GPU outputs through `quality_screen`, `quality_crossing`, and
`quality_formal`. Hierarchy outputs use the final
`require_upper_authorization` routing policy, and all checks use the same
Amazon query/GT and measured Recall-crossing rule as the main query study.
`summarize_authoritative_build.py` reports component times and paired
base-plus-hierarchy totals; it never substitutes a CUDA kernel timer for full
process wall time.

## Authoritative paper result section

`generate_authoritative_paper_results.py` is the fail-closed generator for the
current nine-selectivity study. It is intentionally separate from the legacy
six-workload `generate_paper_results.py`. The generator reruns the Amazon
formal/profile, all three held-out formal, and build-quality validators; checks
all four raw build manifests; verifies the formal/profile operating-point
alignment and required stage/work counters; and only then atomically replaces
`docs/papers/multilevel_ung/generated_results.tex`.

```bash
python3 generate_authoritative_paper_results.py \
  --amazon-formal-config config.authoritative_amazon_formal_emptyfix.json \
  --amazon-profile-config config.authoritative_amazon_profile_emptyfix.json \
  --heldout-formal-config config.authoritative_heldout_genome_formal.json \
  --heldout-formal-config config.authoritative_heldout_reviews_formal.json \
  --heldout-formal-config config.authoritative_heldout_variousimg_formal.json \
  --build-quality-formal-config config.authoritative_amazon_build_quality_formal.json \
  --build-config config.authoritative_amazon_base_build_timing.json \
  --build-config config.authoritative_amazon_base_build_resource.json \
  --build-config config.authoritative_amazon_hierarchy_build_timing.json \
  --build-config config.authoritative_amazon_hierarchy_build_resource.json \
  --amazon-formal ../../runs/authoritative_multilevel_20260926_emptyfix/search/amazon_formal/summary/performance/equal_recall_conservative.csv \
  --amazon-depth ../../runs/authoritative_multilevel_20260926_emptyfix/search/amazon_formal/summary/performance/depth_ablation/depth_by_workload.csv \
  --amazon-depth-global ../../runs/authoritative_multilevel_20260926_emptyfix/search/amazon_formal/summary/performance/depth_ablation/depth_global.csv \
  --amazon-profile ../../runs/authoritative_multilevel_20260926_emptyfix/search/amazon_profile/summary/profile/equal_recall_conservative.csv \
  --heldout-workload results_summary/heldout_oracle/heldout_oracle_by_workload.csv \
  --heldout-global results_summary/heldout_oracle/heldout_oracle_global.csv \
  --build-summary results_summary/authoritative_build/build_summary.csv \
  --build-end-to-end results_summary/authoritative_build/build_end_to_end.csv
```

There is no partial-output flag. Missing inputs, incomplete manifests, a failed
validator, fewer than the declared formal/build repeats, mismatched profile
operating points, missing activation/work counters, or an incomplete held-out
matrix leave the existing LaTeX file unchanged.

## UNG versus plain provider study

`generate_ung_plain_comparison.py` creates the controlled comparison. It holds
the binary, Amazon x1 index, query/GT, graph backend, threads, and Special Block
state fixed. The only changed dimension is the entry-group provider:

- `ung_original_entry`: label-trie supersets followed by exact minimality checks.
- `plain_bitset_lng_entry`: inverted label bitsets followed by LNG-descendant
  coverage elimination.

Run `screen`, then `crossing`, then `formal`. The crossing phase inserts a fine
integer-spaced L grid between the last failed and first successful coarse point;
the formal generator freezes each method's refined smallest Recall crossing.
This comparison is an entry-routing ablation. It is
not evidence that the two methods use independently built vector graphs.
For an interrupted or deliberately bounded screen, `--allow-partial` on the
formal generator retains only workloads with a valid crossing for both methods;
the resulting config records the common workload set explicitly.
