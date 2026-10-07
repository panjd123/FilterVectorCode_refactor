# C3 query-composition evidence: resolution

Scope: C3 is a post-hoc analysis of the existing 95 selected crossings. It
does not add ANN or build measurements and does not change the frozen primary
throughput table.

## What changed

- Exported 190,000 per-query Recall scalars from the two warm executions of
  all 95 crossings into a deterministic gzip CSV. Every selected batch has
  QueryID 0--999 exactly once and reproduces its frozen batch mean.
- Independently recomputed the exact AND-filter coverage of the 9,000 Amazon
  workload/query-ID pairs from 602,453 base label rows. All 3,239 unique
  predicates match the saved coverage profiles; every query has at least ten
  eligible vectors, and no base or query row contains duplicate labels.
- Added one composition figure, full source CSVs, a predicate-size Recall
  table, and exact-selectivity rows for the two mixed batches discussed in the
  main text. NC and empty-stratum rows remain distinct.
- Added a source-only generator, nine focused semantic tests, a raw-record
  extractor, and a label-only coverage verifier. The finalizer now regenerates
  and hashes the C3 artifacts.
- Revised the Results, metric definition, conclusion, Chinese briefing and
  backup note so a batch-mean Recall crossing is not read as a per-query or
  per-subgroup guarantee.

## Claims that changed

The 5.038% batch contains 43 repeated `{1}` queries. At each method's own
batch-selected crossing, their mean Recall is 28.84% for L, 14.88% for T,
73.26% for TL and 74.42% for TLT. TLT reaches 92.64% on the other 957
queries. The 30.027% batch contains 226 repeated `{1,2}` queries at exact
selectivity 91.0545718919%; their Recall is 71.64--74.29% for L, NC for T,
66.42% for TL and 66.81% for TLT. These are results for two repeated
predicate subsets, not general estimates over all predicates of the same size
or selectivity.

At the 60%, 80% and 95% batch-selected crossings, TLT records 31, 29 and 33
zero-Recall queries per execution; L records 11--12, 12 and 14. These counts
describe persistent observations for the same queries in two executions. They
do not identify structural cause, failure probability, or subgroup throughput.

## Independent checks and corrections

The returning C3 data auditor used a separate standard-library implementation
and found no numerical or join mismatch: 190/190 batch means, 570/570 earlier
size histograms, 756 size-summary rows and 2,268 selectivity-summary rows agree.
Its source-only regeneration produced six byte-identical CSV/TeX/PDF/PNG
outputs. The audit identified that the 43- and 226-query subsets repeat one
predicate each; that qualification is now explicit in the paper and Chinese
documents.

The audit also found that the initially copied raw extractor could overwrite
an existing output directory while the README said it refused one. The
extractor now fails before reading inputs when the target exists, uses
`exist_ok=False`, and has a regression test. This implementation change does
not affect the frozen exported bytes.

The C0 full-manuscript review setup was separately checked against the
original tool-call record. Both reviewers used `fork_turns="none"` and received
only the frozen PDF and review instructions. Both recommended Reject; 4/5 was
clarity and reviewer confidence, not an overall paper score. C1--C3 checks are
returning, targeted audits and are not represented as new cold reviews.

## Remaining evidence boundaries

- Each method keeps its own globally selected batch crossing. C3 supplies no
  subgroup-matched-Recall throughput.
- Exported Recall is regrouped rather than recomputed from returned IDs and
  exact neighbors. The label-only audit verifies filter coverage, not ANN
  correctness.
- The pre-existing 0.5%, 1%, and 10% query files lack generation manifests;
  current file identity and historical commands do not form a complete
  historical query-content hash chain.
- External competitive baselines, principal-system transfer beyond Amazon,
  label-order sensitivity, loaded query memory, changing-depth DRH validation,
  and accelerated-build output quality remain open.

No new overall paper score or acceptance recommendation is issued in C3.
