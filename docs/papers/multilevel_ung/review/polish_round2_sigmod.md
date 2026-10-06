# Independent simulated SIGMOD review — frozen round 2

Reviewed `/tmp/mlung-review-round2/main.pdf` (24 pages) and its corresponding manuscript TeX. I read the main manuscript, checked the new ownership/queue example against the stated algorithms, inspected the metric definitions and central tables, and visually inspected the method and table layout. I did not read an author resolution ledger, inspect implementation code, or run experiments.

## Verdict and ratings

The content is substantially more reviewable than round 1. The paper now has a coherent central contribution, an executable description of routing and queue behavior, and results paragraphs that interpret the actual findings. A human reviewer can assess the technical proposal without reconstructing it from disclaimers or historical implementation details.

The PDF is not yet editorially finished: the attempt to contain floats has produced several sparse pages and separated almost all results prose from the results tables. A small number of measurement terms also need to be aligned with the quantities actually reported. These are bounded editorial repairs, not reasons for another broad rewrite.

I would still not recommend SIGMOD acceptance on the present empirical case. That judgment is separate from the improvement in readability: external competitiveness, transfer of the main prefix-base design, and accelerated-build query quality remain insufficiently established.

| Dimension | Round 1 | Round 2 | Assessment |
|---|---:|---:|---|
| Problem/story clarity | 4/5 | 4/5 | The two primary mechanisms are clear; supporting studies have a more appropriate role. |
| Method clarity | 3/5 | 4/5 | Routing variants, seed selection, eviction, result retention, and a concrete trace are now specified. |
| Evidence adequacy | 2/5 | 2/5 | The evidence is better described, but its scope has not expanded. |
| Overall submission readiness | 2/5 | 3/5 | Technically reviewable, with remaining layout/wording work and substantial empirical limitations for acceptance. |

## Resolution of the earlier editorial defects

- **Results narrative: substantially resolved; layout only partially resolved.** Sections 8.1–8.4 now answer concrete questions with numbers. The policy regression is interpreted, construction timings are discussed, and the former empty subsection stubs are gone. Tables no longer spill past the conclusion. Their placement now causes a different reading problem, detailed below.
- **Search specification: resolved at manuscript level.** Section 5.1 gives three routing rules, and Algorithm 4 applies the mass condition only to v2. Section 5.2 specifies ordering, block-seed priority, persistent seen marks, admission/eviction, and answers from the final active queue. I am assessing the description's internal consistency, not certifying implementation equivalence.
- **Contribution and deployment claim: resolved.** The abstract identifies a per-workload winner rather than implying an automatic oracle. Section 8.1 replaces the misleading LNG-fallback implication with the actual result and states the policy's fixed-base scope. The third contribution now centers on analysis, with GPU construction and configuration as supporting studies.
- **Experimental context: substantially improved, with provenance still incomplete.** Section 7.3 now identifies L2, ascending integer-label order, the predicate mixtures, and all five manual alternatives. The one-overlay alternative's inevitable base fallback is explicit. “The repository's preprocessed ... datasets” still does not identify a source repository/citation, embedding/label provenance, or precisely what was held out during policy design. Those missing facts should be documented if available; another defensive paragraph would not replace them.
- **Figures and naming: largely resolved.** The redundant relation sketch is removed, the new example explains the difficult ownership/queue behavior, and primary result tables use canonical topology names. The residual mapping to historical appendix plot names is understandable, though it remains an avoidable decoding step.

## Check of the new two-overlay example

Section 5.2 and Table 2 (PDF p. 5; `sections/index_query.tex:252–285`) are internally consistent.

For nine singleton groups with the listed labels, threshold 1 emits the `ab` subtree with three vectors and the `ac` subtree with four. Their mass is removed from their parent, leaving `a` with `y0,y8`, whose mass two also exceeds one. With threshold 4, neither the size-three `ab` nor size-four `ac` subtree is emitted because the condition is strictly greater than the threshold. The `a` root consequently receives all nine points and is emitted.

For query `{a}`, all four blocks are authorized. Retagging the frontier group `a` to level 1 duplicates its block seed. With capacity two, the two distance-one block seeds `(y1,1)` and `(y1,2)` are retained ahead of the distance-two and distance-three seeds. The level tie-break expands `(y1,1)` first; offering `(y2,1)` at distance 0.5 evicts `(y1,2)`. This follows the stated queue rule and demonstrates representation competition without a cross-level traversal. Keeping this example as a compact table and trace is sufficient; it does not need another figure or a more elaborate hypothetical case.

## Three further changes that materially improve reviewability

### 1. Repair float placement without isolating prose and evidence into separate page blocks

**Evidence:** PDF p. 9 contains only Tables 3–4 with a large empty lower half; p. 10 contains only the measurement-definition table with most of the page empty. All four Results subsections occupy p. 11, followed by tables on pp. 12–14. Page 14 contains two short construction tables and extensive blank space. Construction prose on p. 11 is therefore three pages from its central evidence. Table 6 remains unusually small despite abundant surrounding space. The `FloatBarrier` immediately before Results is visible at `sections/evaluation.tex:127`.

**Change:** Interleave the primary results tables with the corresponding subsection, using local float control rather than flushing whole sections into table-only runs. Fit dataset information and essential metric definitions beside or close to their methodological text; move the exhaustive search-cap table or less central definitions to an appendix if necessary. Enlarge Table 6 to use its available width. The objective is proximity and legibility, not simply a shorter page count. A reader should be able to read a finding and inspect its table without traversing several unrelated tables or a mostly blank page.

### 2. Align the remaining result language with batch means, all-query counters, and per-method operating points

The actual definitions in Table 5 are now useful: Recall is a query mean for each repeat; both warm repeats must cross the target; QPS uses the warm-median batch time; `Visited` counts first adjacency encounters before admission and excludes seed scoring. This last quantity is not a count of expanded states. I found no internal contradiction in these definitions. Three nearby uses still invite the wrong interpretation:

- **Conclusion, last sentence** (`sections/discussion_related.tex:79–80`, PDF p. 15): “at a fixed search budget” does not describe the primary comparisons. Sections 7.4–7.5 select each method's smallest executed budget reaching the Recall target. Replace it with “at a common Recall target” or “with an explicit search-budget tradeoff.” This is a correction to the conclusion, not a request for another disclaimer.
- **Section 8.3** (`sections/evaluation.tex:175–180`, PDF p. 11): “profiles explain where the admitted searches spend work” suggests statistics conditioned on admission. Table 5 says counters are averaged over all queries, and Table 11 supplies the admission fraction separately. Write “profiles show how the routed configuration changes average query work.” The existing numbers then support exactly what the sentence says. Labeling the `Visited` column “Adjacency encounters” would further reduce reliance on the definition table.
- **Abstract, results headings/captions, and Table 3:** Section 7.3 now establishes that nominal selectivities are batch means; the 5% batch contains queries ranging from 0.0033% to 96.702%. Use “mean selectivity” consistently in the headline comparison and table headers. Table 3's “3.365% eligible” workload label simply repeats the Selectivity column and can be dropped. Where a manual winner is a one-overlay plan, a short table label such as “base fallback (1L gate)” would make its executed behavior visible. The paper should describe measured workload families and executed paths directly rather than letting readers infer homogeneous predicates or active one-overlay navigation.

These changes use existing measurement facts. Isolating selectivity causally from predicate composition would require a separate experiment and is not an editorial repair.

### 3. Make one final targeted cut of sentences that only describe the authors' caution or bookkeeping

Most of the earlier defensive scaffolding is gone. The assumptions supporting coverage and safety, the mixture description, and the construction quality boundary all advance interpretation and should remain. The following remnants can be removed or relocated without losing a scientific fact needed at that point:

- Section 5.3's optional lazy-heap correction about what its complexity is “not correctly summarized as” (PDF p. 6) discusses an unevaluated alternative and interrupts the cost argument for the actual array queue. Keep the array bound and its direct consequences; move the heap note to implementation documentation if needed.
- Section 9.2's “These preprints are distinguished from accepted SIGMOD/PVLDB publications” (`sections/discussion_related.tex:57`) merely announces a distinction already made by calling each work a preprint. Delete it.
- The aggregate digest and “deadline artifact manifest” paragraph immediately after the conclusion (p. 15) belongs in reproducibility material. It should not be the paper's final substantive paragraph.

I would not expand this into a general deletion of qualifications. “The one-overlay alternative always falls back,” “query-quality equivalence ... has not been measured,” and the additional assumptions needed to lift label coverage to vector reachability explain what the algorithms or measurements mean.

## Empirical limitations, separate from editorial readiness

The internal Amazon factorial remains useful evidence of an entry/navigation tradeoff. Acceptance-level generality and competitiveness still require representative external baselines under the same protocol, representative prefix-versus-LNG comparisons beyond Amazon, and query-quality checks for any quality-preserving GPU construction claim. More repetitions are needed to interpret percent-level differences robustly.

The newly documented workload mixtures also sharpen the scope: the present sweep compares mixtures whose selectivity and predicate composition change together. It does not identify a universal selectivity crossover. Likewise, a single-overlay manual candidate that always routes to the base does not test one-overlay navigation quality. These are properties of the current experimental design; the manuscript can accurately report them, but prose cannot turn them into controlled tests.

The paper is ready for a useful technical human review after the bounded layout and terminology pass above. It is not yet supported strongly enough for a positive SIGMOD acceptance recommendation. No further large conceptual rewrite is needed to make that distinction clear.
