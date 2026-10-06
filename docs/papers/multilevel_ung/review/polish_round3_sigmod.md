# Independent simulated SIGMOD review — frozen round 3

Reviewed artifact: `/tmp/mlung-review-round3/main.pdf`, 23 pages, with corresponding TeX consulted only to locate the requested changes. This was a bounded check of the three round-2 editorial requests, using actual rendered pages. I did not inspect implementation code or author ledgers, edit manuscript sources, or run experiments.

## Final judgment

**The three requested editorial fixes pass at the level needed for a useful human review.** The main manuscript is readable and its principal comparisons are interpretable. I found no remaining material editorial defect in the scope of this round that warrants another rewrite.

This is an editorial-readiness judgment, not an acceptance recommendation. The existing empirical limitations remain, and I would still not recommend SIGMOD acceptance on the present evidence alone.

## 1. Float pages, evidence proximity, and winner-table legibility — pass

The rendered main body no longer has the sparse methodology and construction table pages identified in round 2. Methodology now flows through pages 7–8, including the dataset table and essential measurement protocol. Results occupy page 9, the three main query-result tables appear on page 10, and the hierarchy construction table appears at the top of page 11. The conclusion also fits on page 11, with references continuing through page 12.

**Table 4 on page 10 is materially improved:** it uses the text width, and the configuration names and QPS values are legible. Tables 5 and 6 are readable beneath it. The reader can inspect the main query results one page after their interpretation, rather than pass through several sparse or unrelated table pages.

Some float placement is still imperfect: Discussion starts at the bottom of page 9, before the primary tables on page 10, and Table 7 appears above Related Work on page 11. This is a modest interruption rather than the former structural problem. I would not hold the manuscript for another float-only iteration on that basis.

The supporting material is separated appropriately: measurement definitions and budgets are on page 13; detailed policy/profile and composed-construction tables are on pages 17–18. These longer lookups no longer interrupt the primary results. Appendix A's heading appears below Tables 8–9 on page 13, and Appendix C's heading follows Table 15 on page 17. Placing those headings above their own tables would be a small finishing improvement, but is not a material barrier to review.

## 2. Batch means, all-query encounter counters, and the common Recall target — pass

The terms now match the reported experiment:

- **Page 1, abstract:** the headline speedup is explicitly for workloads “with mean selectivity below 6%.”
- **Page 8, Section 7.3 and Results introduction:** predicate mixtures and batch-mean selectivity are explained. The example showing the 5% batch spans 0.0033% to 96.702% per query remains visible. The results introduction identifies the sweep as a comparison of selective-predicate batches and mixtures containing a frequent singleton.
- **Page 10, Tables 4–6:** headers use “Mean sel.”; Table 4 explicitly says “batch-mean selectivity.” Table 3 on page 8 no longer repeats a redundant workload column.
- **Page 9, Section 8.3:** the profile discussion now says work is “averaged over all queries,” avoiding the previous implication that the statistics were conditional on overlay admission.
- **Page 13, Table 8:** `Encounters` is explicitly mapped to the CSV counter `Visited`, counted on first encounter through adjacency before queue admission, with rejected candidates included and initial seed scoring excluded. Tables 14 and 17 on pages 16–17 use the `Encounters` label. It is clear that this is not a count of expanded states.
- **Page 11, Conclusion:** “at a common Recall target” replaces “at a fixed search budget.” This agrees with Section 7.4 on page 8, which selects each method's smallest executed budget whose two warm repeats reach Recall@10 ≥ 0.90.

The appendix curve titles still abbreviate their workload label as “selectivity” (pages 20–23). The explicit main-text definition makes their interpretation recoverable; I regard adding “mean” there as optional consistency polish, not a remaining material issue.

## 3. Base-fallback manual winners and caution-only prose — pass

**Table 6 on page 10** labels its last column “Best manual plan / executed route” and visibly marks both relevant rows:

- Genome at 6.292%: `256:lng (base fallback)`.
- VariousImg at 10.252%: `1024:lng (base fallback)`.

**Section 8.3 on page 9** also interprets the VariousImg result correctly: the one-overlay candidate's presence gate executes base search, and its advantage comes from rejecting the overlays. Section 7.3 on page 8 states why the one-overlay alternative always falls back. The reader no longer has to infer this critical behavior from the router definition.

The targeted prose cuts are also present. Section 5.3 on page 6 no longer detours into the optional lazy-heap complexity correction. Related Work on page 11 identifies the preprints without the redundant sentence announcing that they are distinguished from accepted publications. The digest/provenance paragraph has moved from the conclusion to Appendix A on page 13.

The retained assumptions and measurement facts serve interpretation: seed coverage conditions, predicate-safety invariants, repeated-measurement protocol, predicate composition, and the unmeasured query-quality equivalence of construction outputs are substantive information. I do not recommend deleting them merely because they also delimit the claims.

## Readability versus empirical strength

My final readability assessment is **4/5**: the problem, mechanism, execution example, and primary results now form a coherent manuscript. Method clarity remains **4/5**. Evidence adequacy remains **2/5**, and overall conference-submission readiness remains **3/5** because editorial improvements do not expand the experimental basis.

The unchanged empirical gaps are representative external comparisons under the same protocol, transfer of the primary prefix-base result beyond Amazon, quality checks for accelerated construction outputs, and stronger repeat evidence for small performance differences. The current predicate-mixture sweep also cannot establish a causal selectivity crossover, and a manual candidate that always falls back cannot establish one-overlay navigation quality.

These gaps belong in the scientific review, not in another round of defensive wording. The present PDF is sufficiently clear for that review to proceed. No broad rewrite is requested.
