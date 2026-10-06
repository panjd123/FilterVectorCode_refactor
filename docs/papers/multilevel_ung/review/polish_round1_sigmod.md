# Independent simulated SIGMOD review — frozen round 1

Reviewed artifact: `/tmp/mlung-review-round1/main.pdf` (21 pages), with the corresponding manuscript sections and generated TeX consulted to locate passages. I read the main paper and visually inspected the central method figure and several results pages; I did not audit every appendix number, read earlier reviews, inspect implementation code, or run experiments. This review treats the paper as an unfamiliar submission.

## Verdict and scores

There is a reviewable research idea here: replacing global set-minimal entries with a prefix frontier changes the connectivity contract, and certified blocks allow geometric navigation across exact-label groups. The worked example and the distinction between label coverage, predicate safety, and ANN recall make that idea understandable. I can conduct a useful human review of the design and identify its principal tradeoffs from this draft.

I would not recommend acceptance in its current form. The results presentation is unfinished, and the empirical support is primarily an internal Amazon configuration study. Honest disclosure of missing baseline and construction-quality evidence is valuable, but does not supply that evidence. Editorial revision can make this a coherent, useful manuscript for a substantive review; it cannot by itself make the current experimental case competitive for SIGMOD.

Scores use 1 = poor/insufficient and 5 = strong/ready.

| Dimension | Score | Reason |
|---|---:|---|
| Problem and story clarity | 4/5 | The entry/connectivity coupling and restrictive-versus-broad predicate tension are concrete. Secondary construction and policy threads dilute the focus. |
| Method clarity | 3/5 | Coverage and certification are well explained, but routing variants and bounded-queue semantics are not fully specified in the executable account. |
| Evidence adequacy | 2/5 | The internal factorial and retained failures are informative; current-protocol external comparison, generality of the main design, and accelerated-build quality are insufficient. |
| Overall submission readiness | 2/5 | Useful for a technical human review now; substantial editorial and empirical work remains before a conference submission. |

## What is convincing in the present draft

- Sections 2.2 and 3.1 explain a specific, nontrivial distinction: `{b}` is a subset of `{a,b}` but is not its canonical prefix, so the prefix forest needs an additional entry. This gives a concrete reason for the joint design rather than merely asserting that a trie is faster.
- Proposition 1 is explicitly a label-plane coverage result. Section 5 correctly refuses to turn that into an ANN recall guarantee after seed trimming and retagging. Proposition 2 states meaningful ownership and authorization assumptions.
- Section 7.4 distinguishes a matched-provider system comparison from a controlled topology ablation, identifies the modified LNG baseline, and keeps failed recall crossings visible. These distinctions should survive revision.
- The Amazon outcomes do not support a universally superior topology, and the abstract largely reports that limitation accurately. The VariousImg policy regression is visible rather than omitted.

## Five major editorial issues, in priority order

### 1. Rebuild the results section around findings and place evidence beside the relevant finding

**Location:** PDF pp. 8–13, Sections 7.6 and 8.2–8.7; `sections/evaluation.tex:123–188`; generated result subsections and Tables 7–15.

Section 8.3, “Automatic Configuration and Transfer,” consists of two sentences about the comparison contract and is immediately followed by Section 8.4, “Construction.” Section 8.5 then restarts the automatic-versus-manual topic. Section 8.7 contains only a statement about the profile binary. Section 8.4 discusses process-envelope timing and campaign case counts but barely interprets the construction data. The associated results float into Related Work, and Tables 14–15 appear after the conclusion on page 13. A reader encounters conclusions and related work before the numbers needed to assess them.

**Concrete fix:** Own all section structure in `evaluation.tex`; generated macros should provide tables or compact factual paragraphs, not insert competing subsection structures. Use four results subsections: (i) matched base and overlay choices, (ii) navigation and seed-work explanation, (iii) automatic policy transfer and routing failures, and (iv) construction cost and quality boundary. Give each a question, a quantitative answer, and a short explanation. Place or explicitly reference each primary table at that answer. Move long profile definitions and backend catalogs to an appendix. The existing data already support substantive prose: for example, VariousImg falls to 0.205× Plain under v1, while the separate v2 cohort reaches only 0.336× Plain; those are the policy study's dominant finding. No new experiment is needed to repair this narrative.

### 2. Make the search algorithm precise enough to distinguish the evaluated variants

**Location:** Sections 5.1–5.2, Algorithm 4 (PDF pp. 4–5); `sections/index_query.tex:178–231`; Section 6.3 and Algorithm 6 (p. 7).

The text says DRH-v1 tests whether an authorized level at least 2 exists, while DRH-v2 additionally tests mass. Algorithm 4 applies the mass comparison whenever routing is gated, without specifying how v1 disables it. Algorithm 6 returns the next-scale mass gate, again without connecting it to the named version. A reader cannot translate the labels in Tables 8 and 12 into an unambiguous algorithm configuration.

The central queue behavior is also hidden behind “offer,” “prioritize,” and “retain.” The initializer scores and marks all seeds seen before trimming. How block seeds are ranked against one another, how offers displace active states, and whether evicted or unexpanded states can become answers all affect the declared loss of coverage. These details are more consequential than the optional heap complexity discussion in Section 5.3.

**Concrete fix:** Add a three-row routing specification: ungated; presence gate (v1); presence plus mass gate (v2). Branch explicitly on those rows in Algorithm 4, or define the gate threshold used for v1. State deterministic seed ranking/tie handling and the queue's admission, eviction, and result-retention rules in a compact helper definition. Explain the consequence of marking subsequently discarded seeds seen. This is a request for the measured algorithm's specification, not a request to change its behavior.

### 3. Keep the contribution centered on the two coupled graph mechanisms and remove an unsupported deployment implication

**Location:** Abstract and Introduction; Sections 6.3, 8.1, 9.1, and Conclusion. In particular, the end of Section 8.1 says “while retaining an LNG fallback for the 10%–30% region” (`generated_topology_factorial.tex:51`).

The clearest novel claim is the entry/topology contract plus certified block navigation. GPU batching is described mainly as an adaptation of established distance/selection and graph-construction techniques; the hierarchy rule is an explicitly heuristic proxy. Giving these implementation choices comparable narrative space makes it harder to decide what technical advance the paper asks SIGMOD to recognize. The transfer study does not currently demonstrate that the heuristic reliably makes the system practical across datasets.

The quoted LNG-fallback wording is particularly misleading in context. The 14-configuration study identifies a hindsight winner across different bases. Section 6.3 says the policy does not choose the base. A Trie-base index therefore does not acquire an LNG-base fallback merely because the oracle prefers `0L[L]` at two workloads.

**Concrete fix:** State the two primary contributions together and present batching and the structural rule as implementation/evaluation choices unless stronger novelty and evidence are supplied. Replace the fallback sentence with: “The LNG base is best on the measured 10% and 30% workloads; the evaluated policy does not select between base topologies.” Summarize the static `2L[T|LT]` aggregate result separately from the per-workload oracle. In the conclusion, replace the broad “make the design practical to explore” formulation with the concrete demonstrated construction-cost and policy limitations. Preserve the distinction between predicate safety and approximate retrieval quality.

### 4. Make the empirical setup interpretable without an artifact manifest

**Location:** Sections 7.2–7.4 (PDF pp. 7–8), Table 2 (p. 9), and Section 6.3.

The paper provides vector counts and selectivities, but not enough account of where the datasets, embeddings, labels, and query predicates come from or how the selectivity workloads were formed. Names such as `query_minlen5_cov0.1k` are filesystem identifiers, not workload definitions. The distance function is declared “fixed” in Section 2.1 but is not concretely identified in the methodological account. The canonical label order is also an algorithmically important choice for this particular index and is not specified for the experiments. The five manual alternatives are not enumerated in the main paper's policy description.

These omissions matter to interpretation: Amazon has nearly one exact group per vector, and the 99% workload is mostly empty predicates. Without a brief provenance and generation description, I cannot tell which properties explain the claimed regimes or what “held out” establishes.

**Concrete fix:** Add a compact dataset/workload paragraph or table with source citation, embedding and distance/normalization choices, label provenance, query generation and label-cardinality information, the actual canonical order, and whether “held out” means unseen during policy design. Replace workload directory names with readable definitions and enumerate the finite manual candidate set. Put binary hashes, “deadline artifact,” supervisor timing, and campaign bookkeeping in the reproducibility appendix. If these metadata are unavailable, say so explicitly; do not infer them from names. This is initially a documentation requirement, separate from the new sensitivity studies below.

### 5. Use the figures and names to explain the difficult part of the method rather than repeat the easy part

**Location:** Figures 1–3; Table 1; Tables 7–8 and appendix curve legends. See `sections/index_query.tex:13–42` and `44–100`.

Figures 1 and 3 both explain the same four-group entry distinction. Figure 3 is useful and legible, but its singleton example does not show the new complexity: descendant-block exclusion, residual membership, independent overlays, retagging, or shared-queue competition. Those are the parts that later explain recall failures. Meanwhile, Table 7 switches to canonical forms such as `2L[T|LT]`, but the next table reverts to historical `2L-LT`, concealing the different base. A disclaimer that these are historical names makes readers perform a decoding task throughout the results. The overloaded `L` denotes an LNG letter, an overlay-count suffix, and search capacity elsewhere.

**Concrete fix:** Keep one compact entry example. Use the freed visual space for a small ownership/seeding diagram with a descendant block excluded from its parent's direct members and one query that retags an entry; show independent same-vector states and the absence of cross-level traversal. If retaining both entry figures, make one a genuine end-to-end query trace rather than a second relation sketch. Use canonical topology notation throughout tables and legends, and spell out the entry provider and routing policy when they differ. Enlarge Table 7 to use the available width, which would also improve its small text. Fix the visible `ML-UNGcombines` and `ML-UNGcouples` spacing in the Introduction and Conclusion.

## Additional evidence that requires experiments or new measurements

These are not editorial defects and should not be treated as repaired by softer wording.

1. **External competitiveness and baseline representativeness.** The modified LNG control is useful for internal attribution, but cannot establish competitiveness against the original UNG implementation or current filtered ANN systems. For a systems performance claim, add a common-protocol comparison to relevant external systems and an efficient filtered exact-scan reference where appropriate. The very large required search capacities and broad-predicate base slowdowns make that reference important. This need not expand into every cited system; choose representative, justified baselines.
2. **Generality of the primary contribution.** The complete base/frontier/overlay study is on Amazon. The other datasets evaluate a conservative LNG-base policy, so they do not test transfer of the paper's strongest prefix-base result. At minimum, evaluate representative matched prefix and LNG configurations on the other label distributions. Sensitivity to canonical label ordering is especially relevant because it changes frontier sizes and block certificates while leaving the semantic predicate unchanged.
3. **Construction quality.** Tables 10–11 measure builder time with potentially different adjacency and numerical behavior. Re-query each backend used in any quality-preserving construction-speedup claim. Until then, the existing wording correctly permits only time-to-build comparisons with an explicit quality gap. End-to-end composed times should also remain labeled as sums of independent stage summaries.
4. **Robustness and causal explanation.** Two warm repeats support a screening result, not a stable ranking of near-ties or a general confidence interval. More independent/interleaved repeats are needed before interpreting percent-level differences. For the entry/geometry story, add direct frontier/seed counts, block authorization/coverage, and retained-versus-dropped seeds at representative successful and failed points. Existing timing and work counters are useful, but do not isolate why the policy helps Reviews slightly and harms VariousImg severely. Label these mechanisms as hypotheses until the counters resolve them.

## Claim–evidence alignment

The abstract's 2.77–18.43× range is scoped to three Amazon workloads and the modified LNG system; that scope is appropriate. “Trie-base systems win seven of nine workloads” is a statement about a workload-wise configuration oracle, not a deployable automatic policy, and should say so immediately. The conditional safety claims are supportable as written at the mathematical level. The main conclusion should stay at the level of a demonstrated tradeoff and promising internal configurations; external superiority, robust automatic selection, and quality-preserving GPU acceleration remain unestablished.

The appropriate next milestone is a coherent paper that a reviewer can assess without decoding implementation history. That milestone is reachable by editorial work on the present evidence. Likely SIGMOD acceptance would still depend on the additional empirical support above.
