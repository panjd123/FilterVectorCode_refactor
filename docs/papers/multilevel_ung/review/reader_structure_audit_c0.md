# ML-UNG reader-documentation audit

**Verdict:** the latest local navigation and experiment README resolve the main current-versus-historical ambiguity. The advertised CLI commands checked in this audit are valid. Three small configuration/coverage clarifications remain in the experiment guide; no implementation change or benchmark is needed for them.

Scope: read-only audit of the latest documentation under `/tmp/filtervector-paper-18h-20261006`, checked against parsers and finalizer in `/home/sunyahui/worktrees/FilterVectorCode_multilevel_special` over the authorized SSH route. Remote HEAD reported `ed4c48c`, newer than the initially supplied `14c213e` baseline. No writes were made to either repository, no benchmark/campaign was run, and `/home/graphdb/FilterVectorCode_refactor` was not modified. Remote `rg` was unavailable; targeted `sed`/`grep` reads were used. No further literature review was performed.

## Documentation ownership and navigation

| Owner | Reader contract |
|---|---|
| Root `README.md` | One current project identity and links by reader intent: advisor explanation, paper, current data, experiment operation, implementation/history. Keep technical invocation details in the experiment guide. |
| `docs/README.md` | Current paper/results first; implementation runbooks second; explicitly historical cohorts third; author/review records last. Historical “current/best” must be scoped to the recorded version. |
| `docs/papers/multilevel_ung/README.md` | Manuscript layout, checked-in-source compilation, host-only regeneration, evidence limits, and review provenance. |
| `ADVISOR_NOTE_CN.md` | Self-contained explanation of frontier, certified blocks, level-local search, measured results, and unresolved evidence. It should not own runner instructions. |
| `experiments/multilevel_special/README.md` | Current evidence identity, active regeneration chain, stable experiment CLI and configuration contracts, then larger formal protocol and historical adapters. |
| `docs/reports/MULTILEVEL_SPECIAL_BLOCK_REPRODUCE_CN.md` | Historical checkpoint reproduction only. Its new banner correctly scopes old promotion semantics, six-workload results, hashes, and “current” language. |

The latest root/docs links now implement this shape. The old June GPU-entry/full-quality material is retained through immutable links and a historical section rather than presented as the active entry point. The reproduction report’s historical banner is essential and should remain.

## Verified command and configuration facts

- Root README’s `experiment_cli.py matrix experiments/multilevel_special/config.authoritative_amazon_base_trie_query_grid.json` names a real config. It declares `method_schema: orthogonal_v2` and the expected Trie-base cohort. `matrix` loads/validates the config and displays method semantics; it launches no search.
- The experiment guide’s `check`, `matrix`, `run --dry-run`, `run`, `validate`, and `summarize --baseline` commands match `experiment_cli.py`. Repeatable `--method`/`--workload` are supported on `run`; `--allow-partial` is supported on `summarize`.
- `finalize_complete_evidence.py` accepts the documented no-argument invocation and the pinned Tectonic path. Its code validates/summarizes existing evidence, regenerates files, runs Python tests, checks whitespace, and compiles. It does not invoke query/build campaigns. Recorded `runs/...` dependencies justify the guide’s host-only prerequisite.
- The new local `plot_topology_factorial.py` parser takes the two documented positional arguments, `factorial_csv` and `output_dir` (`157–160`). The standalone invocation is valid. I did not rerun the reported plotting validation. The inspected remote finalizer does not invoke this new script, so keep the independent redraw command explicit; if those panels become required regenerated paper outputs, describe the two-step generation path or integrate it separately.
- Experiment config uses `entry_strategy`; the runner translates it to binary flag `--entry_group_strategy` (`run_selection_sweep.py:502–503`). The binary defaults to `optimized_lng`; `--entry_group_provider` is a deprecated alias, with a conflict check when both are supplied (`UNG/codes/apps/search_UNG_index.cpp:293–315`). Do not use the binary flag name as the new JSON field name.

## Remaining small edits

1. **Say that provider/base compatibility is not checked.** Experiment README `109–112` correctly warns that orthogonality does not prove coverage, but readers may still expect `check` to reject the bad pair. `experiment_core.py:181–191` validates allowed `base_topology` and `entry_strategy` values independently; it contains no compatibility rejection. The binary parser independently parses the strategy as well. Recommended wording:

   > `check` validates the declared schema and structural requirements; it does not certify entry/topology coverage. For containment queries, `original` or `optimized_lng` with a Trie base may omit an eligible prefix branch. A complete Trie frontier covers the label plane with either base relation, but seed trimming, vector-edge realization, and finite search budgets can still reduce Recall. The measured factorial uses matched providers.

   The concrete example is `Q={b}`, observed groups `{b}`, `{a,b}`, `{a,b,c}`: LNG-minimal entry `{b}` reaches the other groups through containment edges, but cannot cross into their prefix branch in a Trie base. The Trie frontier keeps `{b}` and `{a,b}`. Covering both *label relations* does not make the two systems equally fast or guarantee vector-level Recall.

2. **Include `block_index` in the overlay configuration contract.** README `99–103` lists new-schema fields but omits the required sidecar path. The validator requires `block_index` when `hierarchy_layers` is nonempty and rejects it when there are no layers (`experiment_core.py:174–180`). Also state `special_block_search == bool(hierarchy_layers)`. Otherwise a reader can follow the listed method fields and still fail `check`.

3. **Label the JSON as the nested `protocol` value.** README `120–130` shows phase/repeats directly in a JSON object. `protocol_for` reads these under top-level `protocol`, and validation requires top-level `num_repeats == cold_repeats + measured_repeats` (`experiment_core.py:66–77,137–139`). Add “Set `num_repeats: 3` and place this object under `protocol`,” or show the wrapper. These are a configuration-fragment ambiguity, not invalid CLI syntax.

No additional broken executable command was found in the bounded examples inspected. The current guide honestly distinguishes the formal generator’s older rendering contract from the bounded finalizer. Historical reproduction commands should stay under the historical banner rather than be silently modernized against a different cohort.

## Minimum change set

For the original navigation correction, the appropriate files are root `README.md`, `docs/README.md`, `experiments/multilevel_special/README.md`, and the historical reproduction report’s banner. The parent’s local changes already address those responsibilities. The paper README changes properly identify the bounded artifact and review lineage.

**Remaining mandatory edits from this audit: only `experiments/multilevel_special/README.md`**, for the three precise clarifications above. An advisor-note cross-link is optional because its frontier example and finite-budget qualification already communicate the scientific boundary. Preserve the documented experimental blockers: external baselines, controlled entry attribution, transfer beyond Amazon, changing-depth behavior, and accelerated-build query quality.
