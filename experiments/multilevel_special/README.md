# Multilevel Special Block experiments

This directory configures, runs, validates, and summarizes ML-UNG experiments.
The current paper uses the completed bounded evidence described below. The
larger formal campaign is a separate protocol with additional requirements.

## Current results and their source

| Study | Recorded evidence | Reader entry |
|---|---|---|
| Amazon topology | 14 configurations × 9 query batches; 1 cold + 2 warm repeats; 100 CPU workers | [Full 126-cell table](../../docs/papers/multilevel_ung/generated_data/factorial_equal_recall.csv) |
| Automatic hierarchy | Genome, Reviews, VariousImg; DRH versus five frozen manual alternatives | [Chinese result report](../../docs/reports/MULTILEVEL_SPECIAL_BLOCK_AUTHORITATIVE_RESULTS_CN.md) |
| Construction | Five hierarchy backends; two measured timing repeats and a separate resource pass | Same report, construction tables |
| Detailed query profiles | Separate instrumented runs; 0L-Trie at 95% exceeded the 3300-second case limit | Same report, stage and counter tables |

A crossing is the smallest **measured** search capacity for which both warm
repeats reach Recall@10 ≥ 0.90. The topology table retains 95 crossings and
31 measured no-crossing cells (`NC`). A profile timeout is a separate status.
Amazon selectivities are batch means; several batches mix selective predicates
with a frequent singleton. The [paper](../../docs/papers/multilevel_ung/main.pdf)
and [advisor note](../../docs/papers/multilevel_ung/ADVISOR_NOTE_CN.md) define
the workload composition, metrics, and topology notation.

Construction totals compose independently timed base and hierarchy medians.
The timing/resource study does not establish matched query quality for every
accelerated output graph.

## Regenerate the current paper from existing runs

On the experiment host, from the repository root:

```bash
/home/lijiakang/miniconda3/bin/python3 \
  experiments/multilevel_special/finalize_complete_evidence.py
```

The finalizer validates existing outputs, rebuilds the factorial and paper
summaries, runs the relevant Python tests, checks the diff, and compiles with
Tectonic. It launches no query or construction benchmark. Raw vectors,
indexes, and runs are outside the Git source distribution; this command
requires the experiment host's recorded input paths.

The active generation chain is:

```text
existing raw manifests and summaries
  → validate_selection_sweep.py / summarize_selection_sweep.py
  → summarize_base_topology_factorial.py
  → plot_topology_factorial.py
  → generate_deadline_paper_results.py
  → generated tables, Chinese report, figures, compiled PDF
```

`generate_*` and `summarize_*` therefore include active paper components as
well as historical adapters. The finalizer is the complete invocation for the
current artifact. Input hashes are recorded in
`runs/deadline_paper_20260928/manifest.json`; final output hashes are in
`runs/complete_evidence_20260928/finalization_manifest.json`.

To redraw the complete topology and search-capacity panels from the checked-in
CSV alone, use Python with NumPy and Matplotlib ≥ 3.6, from the repository root:

```bash
python3 experiments/multilevel_special/plot_topology_factorial.py \
  docs/papers/multilevel_ung/generated_data/factorial_equal_recall.csv \
  /tmp/mlung-topology-figures
```

This writes PDF/PNG panels and a source/output hash manifest. It rejects missing
or duplicate cells, sub-target crossings, and NC cells populated with QPS.
The QPS panel normalizes each configuration by the best measured QPS in the
same batch. A red border identifies that observed winner; grey cells remain
NC. The companion panel shows the selected search capacities.

## Declare and run a new experiment

Use `experiment_cli.py` as the stable front door. The commands below run from
`experiments/multilevel_special`; `config.json` is the chosen experiment file.
Set data/index/output paths for the target host before running.

```bash
# Inspect the configuration and expand its method semantics.
python3 experiment_cli.py check config.json
python3 experiment_cli.py matrix config.json

# Inspect commands, then execute the selected experiment.
python3 experiment_cli.py run config.json --dry-run
python3 experiment_cli.py run config.json

# Validate binary/input provenance, repeats, Recall, and required stages.
python3 experiment_cli.py validate config.json
python3 experiment_cli.py summarize config.json --baseline method_name
```

`run` accepts repeatable `--method` and `--workload` filters. Summarization
accepts `--allow-partial` for inspecting an unfinished campaign; those outputs
retain unavailable rows. Use a new output root for a changed executable or
protocol so that immutable snapshots and old results remain identifiable.

An `orthogonal_v2` method declares `main_index`, `base_topology`,
`entry_strategy`, `hierarchy_layers`, and `special_block_search`. Each overlay
has a positive, strictly increasing `min_points` threshold and `lng` or `trie`
topology. Entry strategies are `original`, `optimized_lng`, and `trie`.
Nonempty `hierarchy_layers` also requires `block_index`; a zero-overlay method
omits it. `special_block_search` must agree with whether overlays are present.
Names such as `plain` or `Trie` do not define these fields; inspect `matrix`.

`mL` counts overlays above L0. For example, `2L[T|LT]` means a Trie base,
LNG at physical L1, and Trie at physical L2. A threshold triggers emission
when uncovered mass strictly exceeds it; it is not a maximum block size.

`check` and `matrix` check or display schema semantics; they do not reject
entry/topology pairs lacking coverage. For example, an
LNG-minimal entry set can omit an eligible prefix branch when used with a
Trie base. The measured full factorial pairs each base with its corresponding
provider. Coverage assumptions and seed selection are specified in the paper.

Each candidate state uses only its physical level's edges. Authorized blocks
are independently seeded into one bounded queue; a group seed can be retagged
to its finest authorized direct owner. There is no implicit promotion or
edge fallthrough. Thus adding an overlay also changes seed participation and
queue competition.

A top-level `protocol` object declares repeat counts and the Recall rule;
top-level `num_repeats` is their sum:

```json
{
  "num_repeats": 3,
  "protocol": {
    "phase": "screen",
    "cold_repeats": 1,
    "measured_repeats": 2,
    "recall_rule": "all_repeats",
    "bootstrap_samples": 10000,
    "paired_repeats": false
  }
}
```

Only warm repeats determine the operating point and timing summary. Paired
statistics require an explicit interleaved `execution_blocks` schedule;
sequential method runs are independent samples. The current paper reports
observed query throughput without confidence intervals from its two repeats.

## Summary and construction tools

| Task | Tool and outputs |
|---|---|
| Full base/overlay factorial | `summarize_base_topology_factorial.py`: 126 cells, controlled upper replacements, matched-provider base comparisons, per-batch and fixed-configuration rankings |
| Fixed-depth comparison | `summarize_depth_ablation.py`: per-workload winners and one unchanged configuration across workloads |
| Held-out auto/manual study | `summarize_heldout_oracle.py`: compare only the candidates and protocol present in its input configs |
| Base topology construction | `run_base_topology_build.py`: separately materialize LNG and Trie bases |
| Declared overlay plans | `run_build_sweep.py`: build each plan, record provenance, support `--dry-run` |
| Construction timing/resources | `summarize_authoritative_build.py`; [build protocol](AUTHORITATIVE_BUILD_PROTOCOL.md) |
| Selected instrumented profiles | `prepare_selected_profile_campaign.py`: derive operating points and hashes from completed performance results |

A per-workload winner is selected after measurement. The fixed-configuration
ranking accepts only configurations that cross on every declared workload and
uses QPS normalized to each workload's measured winner. A one-letter upper
replacement holds base/provider fixed; a cross-base result changes both the
base topology and its matched provider.

Keep timing passes free of profiling instrumentation. Construction and query
timing must not overlap. The bounded campaign applies a 3300-second case cap;
GPU construction additionally checks that the device is idle before timing.

## Larger formal protocol and historical studies

The full pipeline remains available through `run_authoritative_campaign.py`,
`run_heldout_campaign.py`, and `run_authoritative_build_campaign.py`. Its
screen/crossing/formal/profile stages, 35 manual alternatives, larger repeat
counts, and build-quality checks are **protocol requirements**, not a list of
completed evidence in the current paper.

`generate_authoritative_paper_results.py` validates that larger contract and
refuses incomplete inputs. It retains an older LaTeX macro layout; a future
completed formal campaign needs its rendering contract aligned with the
current sectioned manuscript. Use `finalize_complete_evidence.py` for the
currently published artifact.

Detailed historical invocations and continuation supervisors are preserved in
[the earlier experiment guide](https://github.com/panjd123/FilterVectorCode_refactor/blob/ed4c48cab3ac56f6bc36e7ef895d6afd3a18c989/experiments/multilevel_special/README.md).
The [older reproduction report](../../docs/reports/MULTILEVEL_SPECIAL_BLOCK_REPRODUCE_CN.md)
describes the earlier two-overlay promotion semantics and six-workload cohort.
The old `generate_ung_plain_comparison.py` study changes entry providers on a
shared vector index; it is separate from the current base-topology comparison.
