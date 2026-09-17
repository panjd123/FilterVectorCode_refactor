# Multilevel Special Block experiments

Use `experiment_cli.py` as the stable front door. The older `generate_*` and
`summarize_*` scripts remain as reproducibility adapters for already published
experiment families; new experiments should declare methods and protocol in one
JSON config and reuse the shared core.

```bash
# Fail fast on method/protocol inconsistencies.
python3 experiment_cli.py check config.json

# Print orthogonal method semantics (main graph, entry provider, overlay).
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

Methods must explicitly name `entry_group_provider`, `layer_count`, and
`special_block_search`. Layered methods additionally require `block_index` and
legal `t1`/`t2` values. Do not infer method semantics from names such as
`plain`, `UNG`, or `Trie`: use `experiment_cli.py matrix` in reports and audits.

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
