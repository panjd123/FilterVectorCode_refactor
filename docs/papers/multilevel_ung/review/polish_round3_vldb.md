# Independent simulated VLDB review — round 3

Reviewed the actual frozen `/tmp/mlung-review-round3/main.pdf` (23 pages), using PDF text extraction and rendered pages, with targeted TeX checks. I did not read author ledgers, edit manuscript sources, audit code, or run experiments. This is a bounded verification of the three round-2 editorial requests.

## Verdict

**All three request groups pass. No remaining material editorial blocker warrants another broad rewrite.** The manuscript is ready for meaningful human review. The central design, execution semantics, measurement definitions, and results can now be assessed without repairing the narrative on the author's behalf.

Conference evidence readiness is a separate matter and remains unchanged. My ratings are story clarity **4/5**, method clarity **4/5**, empirical support **2/5**, overall conference submission readiness **2/5**, and writing readiness for productive human review **4/5**.

## Verification

| Requested fix | Result | Concrete PDF evidence |
|---|---|---|
| Sparse float pages, evidence proximity, winner-table legibility | **Pass** | The former mostly empty methods-table pages have disappeared from the main argument. Dataset Table 3 shares p.8 with methods prose. Results are on p.9; winner Table 4, comparison Table 5, and primary policy Table 6 are on the facing/next p.10. Table 7 appears on p.11 before the conclusion. I rendered pp.8–11 and could read the winner table's configuration/QPS pairs and grouped headers without overlap or clipping. |
| Batch-mean selectivity | **Pass** | The abstract on p.1 says “workloads with mean selectivity below 6%.” Section 7.3 on p.8 explains singleton replacement, the empty-predicate mixture, and the wide individual-query range within the 5% batch. The results opening on p.8 explicitly defines the sweep as selective-predicate batches and mixtures. Section 8.2 on p.9 refers to batches with larger mean selectivity. Tables 3–6 use “Mean sel.” and Table 4 explicitly says “batch-mean selectivity.” |
| All-query encounter counters | **Pass** | Section 8.3 on p.9 explicitly says profile work is “averaged over all queries.” Appendix Table 8 on p.13 defines `Encounters`, maps it to the CSV `Visited` counter, states that first encounters occur before scoring/queue admission, includes rejected candidates, and distinguishes seed scoring. Its caption states the all-query averaging order. Table 17 on p.17 visibly labels the column “Encounters” and separately defines the active-query fraction. The presentation no longer invites reading these numbers as expanded states or conditional means over admitted queries. |
| Common Recall target | **Pass** | Section 7.4 on p.8 defines each method's operating point as its smallest executed search capacity for which both warm repeats reach Recall@10 ≥ 0.90. Table 5 on p.10 repeats the relevant comparison boundary. The conclusion on p.11 now says “at a common Recall target,” matching the measurement protocol rather than implying identical queue capacities. |
| One-overlay manual winners visibly marked base fallback | **Pass** | Table 6 on p.10 labels the final column “Best manual plan / executed route.” Both `256:lng` for Genome and `1024:lng` for VariousImg explicitly carry “(base fallback).” Section 8.3 on p.9 states that the VariousImg winner's advantage comes from rejecting overlays. Section 7.3 on p.8 explains why the shared presence gate gives the one-overlay plan this behavior. |
| Caution-only prose removed while necessary facts remain | **Pass** | The results and discussion on pp.9–11 explain mechanisms, outcomes, and the policy failure directly. The previous repetitive audit/deadline and guarantee-disclaimer passages are absent from the main narrative. Remaining statements about label-plane reachability, seed eviction, fixed routing, timing boundaries, and construction-quality measurement have a concrete interpretive role. They should remain. |

## Readability assessment

The central results still use a dense table page, but this is now a reasonable tradeoff: the tables are legible and one page from their interpretation. Construction Table 7 is two pages after its discussion, with the intervening page carrying the principal query results; this is ordinary float displacement rather than the prior sparse-page failure. The appendix appropriately carries metric details, profiles, routing ablations, fixed-manual aggregation, and composed construction times. The main paper reaches its conclusion on p.11 instead of p.15 in round 2.

The two-overlay worked example remains on p.5 beside Algorithm 4 and the queue description. It continues to make level ownership, retagging, duplicate states, and eviction understandable. No expansion of that example is needed for this review milestone.

I would send this version to colleagues for substantive review now. The remaining table density and routine appendix float placement do not justify another editorial cycle.

## Unchanged empirical gaps

The manuscript still needs current-protocol external comparisons to establish competitiveness beyond its internal refactored UNG-style baseline. Two warm repeats provide limited evidence for close performance differences. Quality-preserving construction acceleration requires query-quality measurements on the corresponding output graphs. Policy transfer remains largely neutral on four held-out workloads and poor on VariousImg. The mixture-based Amazon sweep does not isolate selectivity from predicate composition.

These are experimental questions, not unresolved prose defects. The present manuscript describes its measured behavior clearly enough for reviewers to decide which additional evidence matters. This editorial pass does not change my conference-evidence rating or imply an acceptance recommendation.

Created `/tmp/mlung-r3-vldb.md` and temporary PDF-page renders only; no manuscript source changes.
