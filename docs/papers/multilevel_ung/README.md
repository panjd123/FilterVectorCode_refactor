# ML-UNG research manuscript

Start with `ADVISOR_NOTE_CN.md` for the Chinese advisor briefing and `main.pdf`
for the English manuscript. The paper leads with prefix-frontier entries and
predicate-certified blocks. DRH is a secondary configuration heuristic.
`ADVISOR_BACKUP_CN.md` holds the full grid, threshold rules, and construction
details for questions after the short briefing.

`main.tex` assembles the main sections: introduction, motivation,
frontier, block index/query and seed coverage, construction/configuration, evaluation, and
related work/discussion. Four TikZ mechanism figures are embedded in those
sections. Full partition proofs, finite-budget examples, metric definitions, search caps, provenance, detailed factorial tables,
frozen construction profiles, and Recall curves are in the appendix.
`PAPER_OUTLINE_CN.md` explains the argument and evidence boundaries.

## Evidence and scope

The frozen Amazon topology campaign covers 14 configurations and 9 workloads,
with 100 query threads, one cold run and two warm repeats. `generated_data/`
contains all 126 crossing/no-crossing cells and their provenance. Cross-base
comparisons use matched providers; upper-level one-letter replacements hold
the base/provider fixed. These are screening measurements, not formal CIs.

`generated_results.tex` and `generated_topology_factorial.tex` are generated
from validated source artifacts. Do not change their numerical entries by
hand. The one declared incomplete detailed profile is 0L-Trie at 95%
(3300-second timeout); `timeout` is distinct from a Recall `NC`.

Construction timing/resource results are complete for the declared campaign.
The composed total sums independently timed base and sidecar medians; the
supervisor envelope includes additional orchestration. Two timing repeats
support exploratory bootstrap intervals only. Query-quality equivalence for
all accelerated indexes, particularly the fastest hybrid build, is not
established by the construction timing table.

The current manuscript does not claim submission-ready evidence. Required
extensions include current-protocol external baselines, controlled entry
attribution, prefix/block transfer beyond Amazon, changing-depth DRH tests,
and build-quality checks. See the final section in `ADVISOR_NOTE_CN.md`.

## Regenerate and validate on the experiment host

Use the authorized worktree, not `/home/graphdb/FilterVectorCode_refactor`:

```bash
cd /home/sunyahui/worktrees/FilterVectorCode_multilevel_special
/home/lijiakang/miniconda3/bin/python3 \
  experiments/multilevel_special/finalize_complete_evidence.py
```

This validates and summarizes existing campaign outputs, regenerates tables
and figures, runs the relevant Python tests, checks Git whitespace, and
compiles the PDF with the pinned Tectonic binary. It does not rerun the query
or build campaigns. Source inputs and hashes are recorded in
`runs/deadline_paper_20260928/manifest.json`; final outputs are recorded in
`runs/complete_evidence_20260928/finalization_manifest.json`.

To compile checked-in paper sources alone:

```bash
cd docs/papers/multilevel_ung
/home/sunyahui/.local/opt/tectonic-0.15.0-musl/tectonic \
  main.tex --keep-logs --keep-intermediates
```

An ACM-compatible local TeX installation can also use `latexmk -pdf main.tex`.
The legacy formal publication gate `generate_authoritative_paper_results.py`
remains fail-closed and uses its older macro layout. Migrating a completed
formal campaign to this sectioned draft requires updating that rendering
contract; this revision validates the bounded-evidence path only. Successful
draft compilation does not satisfy the formal experiment contract.

## Independent review

`review/` records the first and second structure passes, implementation/proof
review, and primary-source literature audit. The reviewers found the revised
core arguments credible under their stated assumptions while identifying
experimental blockers. `review/RESOLUTION.md` records addressed and remaining
items. Citation metadata includes at least 16 related SIGMOD/PVLDB papers and
two recent works explicitly labeled as preprints. The UNG ACM PDF access
failure is disclosed; the mechanism was checked against official source code.

Review input and reviewer history are recorded in `review/POLISH_ROUNDS.md`
and `review/CONTINUOUS_18H_MANIFEST.json`. In the earlier three-round polish,
round 1 used newly created reviewers with no inherited conversation; rounds
2 and 3 reused those reviewers for targeted revision checks. Their final
4/5 readability ratings were not a first-read overall assessment of the
final manuscript. The C0 full-manuscript review used newly created reviewers given only a frozen
PDF, without previous scores or revision lists. Both recommended Reject; both
rated empirical support 2/5 and clarity 4/5. Their complete reports and the
subsequent C1 response are in `review/`. These are simulated agent judgments.

C1's returning reviewers accepted the core coverage and partition arguments
but retained the missing-evidence basis for Reject. C2 makes the integer
threshold and own-group reachability premises explicit and adds reproducible
observed timing ranges. The 95 timing pairs and 35 comparisons are under
`generated_data/warm_repeats/`; the four combinations per comparison are not
independent experiments or confidence intervals.

C3 adds the query-composition figure and subgroup Recall tables, reconstructed
from 190,000 existing per-query records and 9,000 independently counted label
coverages. [The source-only reproduction guide](generated_data/query_strata/README.md)
explains the inputs, NC/empty-stratum distinction, and remaining provenance
limits. The 5% singleton subset repeats predicate `{1}`; the 30% subset at
80–95% individual selectivity repeats `{1,2}`. These are query-subset results,
not evidence about many different predicates in those bins. C1–C3 checks use
returning reviewers for stated, bounded tasks and do not provide a new cold
overall score. The original C0 launch arguments were rechecked against tool
records in `review/C0_REVIEW_SETUP_VERIFIED.json`.
