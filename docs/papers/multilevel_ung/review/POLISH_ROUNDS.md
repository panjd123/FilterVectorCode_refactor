# Iterative writing and reviewer ledger

Objective: continuously improve exposition, narrative and reader experience,
with SIGMOD-style and VLDB-style reviews after each revised draft.
Reviewers are simulated agents; acceptance is never predetermined.

## Reviewer context and score interpretation

| Round | Reviewer history | Review scope |
|---|---|---|
| 1 | New SIGMOD and VLDB agents, `fork_turns="none"` | First reading of the round-1 manuscript |
| 2 | Same agents retained round-1 history and received revision requests | Review of changes and remaining issues |
| 3 | Same agents retained earlier history and received three editorial checks | Targeted verification of the final polish |

The two reviewer roles were separate from one another, but rounds 2 and 3
were not fresh reviews without project history. The final 4/5 ratings refer
to readability or clarity; empirical support remained 2/5. They do not imply
an overall 4/5 conference recommendation. The subsequent C0 review of the
published final manuscript uses new agents and PDF-only input, as recorded
in `CONTINUOUS_18H_MANIFEST.json`. Original review reports are retained.

## Baseline

Published commit: `637226a` (paper change `e957dcb`). Prior turn was an
assessment without manuscript mutation; this round remedies that lack of
state progress by revising the authoritative manuscript.

## Round 1 — entry coverage and authorized geometric freedom

Edits: shorter abstract/introduction; one running example connects LNG entries,
prefix entries and a physical L1 block; replace repeated defensive statements
with positive mechanism explanations while retaining formal assumptions and
an explicit evaluation-limitations section. Numeric evidence is unchanged.

Compilation succeeded after the round-1 changes. A redundant unverified
assertion that every query has at least ten eligible points was removed; the
conditional Recall definition and measured K=10 protocol remain.

Completed: rendered and frozen inputs; independent SIGMOD and VLDB reports
and their resolution table appear below.

## Editorial acceptance criteria

- A new reader can state the problem, the joint entry/block insight, and the
  distinction from UNG after the abstract and introduction.
- The running figure uses consistent labels and vectors and agrees with the
  actual seeding, partition, and level-local search contracts.
- Each paragraph advances one role; evidence caveats are placed where the
  scope is defined rather than repeated as rebuttal throughout the narrative.
- Contributions, method order and evaluation questions correspond.
- No review conclusion is inflated into an empirical result or actual human
  approval; unresolved evidence gaps remain explicit.

### Parent layout audit of round 1

- Figure 3 is readable and its singletons, T1=1 emission, eligibility and
  level tags agree with the example. The explanatory line in panel (c) spills
  beyond the block rectangle; wrap it in the next round.
- The Results section has a real ordering defect: the manual paragraph under
  “Automatic Configuration and Transfer” is immediately followed by the
  generated “Construction” subsection, then “Automatic Versus Manual
  Hierarchies.” Remove the orphan heading and order the results coherently.
- Reviewers see frozen `/tmp/mlung-review-round1`; subsequent editing must not
  change their review input. Two independent reports are running.

## Round 1 reviewer decisions and round 2 changes

Two independent reviewers completed the frozen 21-page manuscript. SIGMOD:
story 4/5, method 3/5, evidence 2/5, submission readiness 2/5. VLDB:
story 3/5, method 4/5, evidence 2/5, submission readiness 2/5. Both judged
it usable for a substantive human review. These are simulated judgments.

| Finding | Action in round 2 | Status |
|---|---|---|
| Orphan Results headings and drifting tables | Main section owns four result subsections and their interpretation; table-only generated macros; float barriers; backend catalogs moved to appendix | Validated in round 2; float layout revised again in round 3 |
| Secondary threads dilute novelty | Abstract/contributions center frontier plus certified blocks; GPU/DRH remain supporting studies | Implemented |
| LNG fallback confuses oracle with router | Replace with measured LNG-base wins and explicit fixed-base policy | Implemented in generator |
| Routing variants and queue semantics underspecified | Explicit v1/v2 branches; checked actual comparator, visited marking, eviction and final-active result extraction | Implemented |
| No two-overlay execution example | Nine singleton groups, descendant exclusion, seed retagging/deduplication, two same-vector level states, and eviction before coarse expansion | Implemented |
| Workload and manual plans obscure interpretation | Checked raw predicates/manifests and source; document mixed batch means, L2, stored-ID ordering, and all five manual perturbations | Implemented; embedding preprocessing provenance remains unavailable |
| Repeated defensive prose and weak two-repeat CI presentation | User explicitly asked to delete sentences serving only as defenses; remove duplicates/maintenance digression; keep measured definitions and assumptions once; remove paper CI column while retaining source and report values | Implemented |
| Old names hide L0 | Canonical names in generated tables; frozen curve legends still need their documented alias mapping | Tables implemented |

No new benchmark was launched and no frozen numeric inputs were altered.
An early rsync directory call omitted recursive mode and skipped the directory;
this was detected from its output and corrected with `rsync -a` before the
round-2 finalizer. Its earlier successful validation is not used as round-2
validation evidence.

Additional source check: the previous definition of Visited as popped states
was incorrect. Both base and overlay implementations increment it at first
unseen-neighbor discovery, before queue admission (backend lines 317–319 and
817–834; CSV output apps/search_UNG_index.cpp:1000). The paper and Chinese
note now define this counter correctly. Values and performance conclusions
are unchanged; new Results prose uses distance and edge counts explicitly.

## Round 2 review decisions and round 3 changes

Both returning reviewers judge the main story and method clear enough for
substantive human review and recommend no further broad conceptual rewrite.
SIGMOD rates story/method/evidence/submission readiness 4/4/2/3; VLDB rates
4/4/2/2 and human-review readability 4/5. These are simulated evaluations.

Round 3 moves the complete metric definitions, search caps and provenance to
the appendix; relaxes the pre-Results float barrier; requests tables before
their interpretation; enlarges the factorial winner table; consistently
labels batch-mean selectivity and all-query counters; identifies one-overlay
manual winners as base fallback; and removes the optional-heap caution and
redundant preprint/bookkeeping statements. Numerical inputs remain fixed.
The small result-language corrections preserve the common Recall target and
explain why an overlay rejected by the gate can win the manual comparison.

The first round-3 render still had a single-column table blocking later
double-column floats, including one almost empty page. The repaired layout
keeps the central factorial, depth, manual-policy and hierarchy timing tables
in the main text, while placing global-manual, mass-gate, detailed profiles,
and composed-build tables in the appendix. All numbers remain available.
Dataset tables retain their natural font size rather than being enlarged to
fill the page width. Validation and targeted round-3 PDF reviews by returning reviewers are complete.

## Round 3 final review and validation

Both reviewers pass all three requested editorial groups and request no
further broad rewrite. Each rates human-review readability 4/5 and method
clarity 4/5; empirical adequacy remains 2/5. SIGMOD submission readiness is
3/5 and VLDB 2/5. These are simulated opinions from returning reviewers, not human approval.

The final 23-page PDF puts Results on page 9, the main query tables on page
10, and construction evidence and the conclusion on page 11. The sparse
main-body pages are resolved. Supporting tables remain in the appendix;
minor appendix heading/float placement and legacy curve-label consistency
can be handled during venue-specific typesetting. Reviewers do not consider
those items barriers to a substantive human review.

Validation: 256 Python tests pass; Tectonic compiles; no undefined references
or citations; git diff --check passes. UNG algorithm sources and all frozen
factorial data are unchanged, and no benchmark campaign was rerun. The only
declared missing detailed profile remains 0L-Trie at 95% (3300-second cap).
Current output fingerprints are recorded in VALIDATION.json. The historical
six-pass bibliography warning remains nonfatal with resolved references.

Final metadata and Chinese briefing are aligned with this review; publication
uses the already-authorized remote GitHub push and ShareLaTeX synchronization.
