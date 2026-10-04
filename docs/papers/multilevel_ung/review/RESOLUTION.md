# Review resolution — manuscript revision

Starting commit: `fe46b438131c9444570d1a1114f8b7708b8f6dd5`.

| Review finding | Resolution |
|---|---|
| Old hierarchy/DRH-centered narrative | Rewritten around the prefix frontier and certified blocks; DRH is a secondary instantiation. |
| Trie is not an LNG edge subgraph | Corrected to containment reachability; exact forest bound is G−r. |
| Production entry algorithm mismatch | Bitmap intersection and ancestor filtering replace pivot-walk pseudocode. LNG candidates and eligible groups have distinct cost variables. |
| Block coverage and nesting assumptions | Strict emission threshold, self-ownership, partial direct coverage and independent partitions are explicit. |
| Safety theorem missing owner condition | Authorized direct owners required for overlay seeds and scanned edges. |
| Query seed/queue semantics | All seeds marked seen and scored before trimming; retagging and block priority explicit; expanded candidates remain retained. |
| Incorrect queue/storage complexity | Array insertion-attempt scans, dense ownership maps, per-level visited state and seed setup are included. |
| Old results ordering | Winners and topology findings first; full matrix, controlled tables, profiles and curves in appendix. |
| NC overloaded with timeout | Generator and existing regression test now use timeout for the one declared profile failure. |
| Metric and timing boundary ambiguity | Workload-specific batch size, Active denominator, v2 cohort Plain QPS, supervisor versus builder process wall, exploratory intervals specified. |
| Literature strawmen | Curator/FAVOR/ACORN scoped to primary evidence; LSSG and learned routing identified as preprints. |
| Build speed interpreted as quality equivalence | Time/resource claim only; missing fastest-build query-quality evidence explicit. |
| Monolithic TeX provenance | Nested section inputs included in finalization hashes; macro-contract test checks assembled sections. |

## Accepted limits

The structure and proof reviewers consider the central conditional arguments
credible, but do not approve submission-strength empirical claims. Missing
controlled entry attribution, cross-dataset prefix/block results, current
external baselines, changing-depth policy validation, and accelerated-index
quality checks remain named gaps. No new benchmark was run to conceal these
gaps. The old 396-case campaign was not rerun.

The bounded generator is the validated renderer for this revision. The legacy
formal generator has its own older rendering contract and remains fail-closed;
formal-evidence migration is not claimed complete.

## Validation record

The remote finalizer regenerates all bounded evidence, runs 256 Python tests,
checks whitespace, and compiles with Tectonic. Rendered-page and provenance
checks are recorded in the final revision audit. Literature source snapshots
are retained as audit inputs; the committed manifest identifies their hashes.

Final local rendering: 21 pages including references and appendices. Four
mechanism figures, representative main tables, full topology matrices and
Recall curves were inspected. The only remaining horizontal overfull warning
is 1.42 pt in the frontier cost paragraph; it does not clip content. Tectonic
reports its six-pass bibliography consistency warning, but all cited entries
and cross-references resolve.

Primary-source snapshots are retained on the experiment host under
`runs/paper_revision_20261004/literature_sources/`, outside the published paper
sources; the checked-in manifest records their hashes.
