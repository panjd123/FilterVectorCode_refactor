# Independent simulated VLDB review — round 1

Reviewed only `/tmp/mlung-review-round1/main.pdf` as a standalone manuscript. I read the main paper and inspected the appendix curves/tables at the level needed to understand the evidence; I did not inspect project history, previous reviews, source implementation, or run experiments. Page references below are the PDF's printed page numbers.

## Verdict and ratings

The draft contains a coherent, reviewable research idea: trade minimal entry sets for a simpler prefix forest, then recover geometric navigation through predicate-certified blocks. The distinction between label-plane coverage, predicate safety, and approximate-search recall is unusually explicit and valuable. However, this is currently a promising internal research draft rather than a submission-ready VLDB paper. The results narrative is visibly unfinished, and the empirical evidence cannot yet establish competitiveness with current external systems or quality-preserving construction acceleration. My present submission verdict would be reject/major revision, not acceptance after copyediting alone.

| Dimension | Rating (1–5) | Reason |
|---|---:|---|
| Story clarity | 3 | The opening gives a recognizable problem and two complementary mechanisms; the paper then expands into construction and policy studies without a comparably integrated results narrative. |
| Method clarity | 4 | Definitions, pseudocode, ownership rules, and conditional guarantees are precise. A worked multilevel seeding example is still needed to connect these details. |
| Empirical support | 2 | The topology screen has useful controlled comparisons and honest failures, but the principal result is one development dataset with two warm repeats and an internal baseline. |
| Overall submission readiness | 2 | Significant writing repair and additional empirical validation are required. |
| Can a human productively review this draft? | 4 / Yes | A reviewer can evaluate the central mechanism, assumptions, and experimental design now. They should be told that the results organization and evidence package are incomplete. |

## Five prioritized major writing fixes

### 1. Turn the results section into an argument, and place each result beside the text that interprets it

**Exact passages:** §8.3, p.9 consists of “Manual alternatives are a frozen finite candidate set. Per-workload winners are hindsight comparisons, not executable query-independent policies.” §8.7, p.9 consists of “The separate detailed-profile binary is explanatory only and does not contribute primary QPS. Edge counts are measured here; disabled light-stat counters are never interpreted as zero.” §7.6, p.8 has a heading with no intervening prose before §8. Tables 7–8 appear on p.11, Tables 10–13 on p.12, and Tables 14–15 on p.13, after the conclusion has already begun/finished on p.12.

**Why this matters:** These are measurement qualifications rather than answers to Q2–Q5. The reader reaches related work before seeing most of the empirical evidence, and must reconstruct the claims from detached tables. Sections 8.3 and 8.5 also duplicate the automatic/manual comparison setup.

**Concrete fix:** Organize §8 into four results subsections: (i) matched entry/topology factorial, (ii) fixed-base depth and topology controls with profiles, (iii) structural policy transfer and gate failures, and (iv) construction cost and resources. Each should lead with a finding, cite its table, explain two representative numbers, and state one boundary. For example, the transfer subsection should say that DRH-v1 is near Plain on four workloads but reaches only 0.205× Plain on VariousImg; DRH-v2 raises that to 0.336× while losing the useful Reviews admission. The construction subsection should interpret the 9.45×/11.06× sidecar time ratios and the much smaller 1.21×/1.35× composed ratios, explicitly retaining the quality gap. Give §7.6 a short explanation of Tables 5–6 or merge it into construction setup. Use float placement/layout changes so that the principal results are encountered before discussion/conclusion.

**Classification:** Editorial and layout repair using existing evidence. No new experiments are needed to fix this problem.

### 2. Sharpen the novelty claim around one coupled design, and demote the less established extensions

**Exact passages:** The contribution list, p.2 ends with “An evaluated implementation. A complete topology factorial, query profiles, a structural configuration rule, and GPU-assisted construction expose the performance and cost of these design choices.” The abstract, p.1 adds “batched GPU tasks and a query-calibration-free hierarchy rule” immediately after the central guarantees. §9.3, p.12 says, “Our contribution is to batch the irregular local-graph and cross-edge tasks induced by a metadata hierarchy, retaining an explicit host-output boundary.”

**Why this matters:** The strongest conceptual contribution is the entry/topology contract plus block certification. Generic bitmap intersection, NN-Descent, GPU distance/top-k batching, and the structural heuristic are not comparably established as new research contributions by this manuscript. Listing them together dilutes the answer to “what is new beyond UNG and label-specialized indexes?” The related-work discussion answers much of this question, but arrives too late.

**Concrete fix:** Add a compact positioning paragraph at the end of the introduction: UNG uses global set-minimal entries with containment cover edges; this work changes both to prefix-frontier entries with terminal-prefix connectivity, then permits cross-group geometry inside independently certified regions. State that compared with clustering-based virtual indexes, these regions have fixed canonical-prefix membership and a sufficient root certificate. Present GPU batching and DRH as supporting implementation/configuration studies unless stronger novelty/evidence is supplied. Replace the third contribution with a concrete empirical contribution, e.g. a controlled analysis showing when cheaper entry discovery is outweighed by seeds/navigation and when coarse overlays reverse the result. Retain the statement that the baseline is a refactored UNG-style system, not the original release.

**Classification:** Primarily editorial. Establishing competitive novelty against external systems still requires experiments; rearranging related work cannot supply that evidence.

### 3. Show one complete multilevel query, including retagging and queue competition

**Exact passages:** §4.2, p.4: “Separate partition passes for T1 < T2 < ··· leave earlier levels unchanged; they do not imply nested direct-member sets or identical coverage at all levels.” §5.1, p.5: “Each group entry receives the finest authorized overlay owning that group, or level 0 if no such overlay exists.” Algorithm 4, lines 8–11: “discover group entries with e and assign seed levels”; “create authorized block seeds and group seeds”; “deduplicate by (i,ℓ)”; “prioritize block seeds; trim to Ls.”

**Why this matters:** These choices determine actual search behavior and the recall losses, but the only worked navigation figure shows one overlay and singleton groups. A reader can understand the safety proof while still being unsure how independently partitioned overlays cooperate when search never changes a state's level. “Finest authorized overlay” also needs an explicit minimum-level definition because the direct memberships are not nested.

**Concrete fix:** Extend the existing example with a small table of two overlay partitions and one query: list direct members and root certificates, show which blocks authorize, show each candidate seed as `(vector, level, source)`, show duplicate removal and a deliberately small capacity, then show one same-level expansion. Define the selected owner as the lowest numbered authorized overlay owning that group. Explicitly state that independently seeded searches compete in one queue; there is no interlevel traversal. Use this same example to demonstrate how a dropped or retagged frontier seed can remove the assumptions of Proposition 1. Figure 3 is already readable and should remain the simple intuition figure; the new example should explain execution rather than add more decorative arrows.

**Classification:** Editorial/method exposition, using the algorithm already stated. No new experiment is needed.

### 4. Describe the workload construction well enough that “selectivity regimes” have an interpretable meaning

**Exact passages:** §2.1, p.2: “The distance function is fixed across compared methods.” §7.3, p.8 enumerates the nine mean selectivities and says that held-out workloads' “exact workload directories and fingerprints are recorded in the artifact manifest.” Table 2, p.9 uses names such as `query_minlen5_cov0.1k`. §8.1, p.9 concludes that the results favor a topology “while retaining an LNG fallback for the 10%–30% region.”

**Why this matters:** The manuscript does not specify the actual distance metric/normalization, dataset and label provenance, or how vectors and predicates were sampled to create the nine workloads. A mean selectivity and a directory name do not explain a workload. Label cardinality, skew, group fragmentation, and geometric correlation could change along with selectivity. The known 99% mixture of empty and singleton predicates demonstrates that these batches differ structurally. It is therefore too easy to read the observed 10%–30% result as a general crossover rule.

**Concrete fix:** Add a concise workload-generation paragraph identifying metric, embedding normalization, source of labels, query-vector source, predicate generation/selection, and canonical label order. Replace filesystem-style workload names in the main table with readable descriptions and retain identifiers in the artifact. Include basic label/query statistics if already available. Phrase regime conclusions as “on these Amazon workloads near 10% and 30%” and state that selectivity alone has not been isolated as the causal variable. Specify the five manual candidate plans in prose or a compact table so that “frozen finite candidate set” is meaningful without opening an external manifest.

**Classification:** Existing setup facts are an editorial/reproducibility repair. Establishing a general selectivity rule, sensitivity to label ordering, or robustness to label/geometric distributions requires additional experiments; do not claim those from the current sweep.

### 5. Consolidate evidence qualifications into a clear scope statement, and avoid statistical presentation that exceeds the sample

**Exact passages:** §7.4, p.8 repeats the cold/two-warm protocol several times and calls it both “deadline evidence” and a “current deadline artifact.” §7.5 says bootstrap intervals “describe this exploratory sample and do not establish population-level uncertainty.” Table 11, p.12 nevertheless foregrounds “Exploratory 95% CI,” including intervals `[0.18, 0.18]`. §9.1, p.10 states that “A submission-strength evaluation still needs current-protocol external baselines, longer repeated timing, and query-quality checks for each accelerated construction profile.”

**Why this matters:** The limitations are honest, but their repetition makes the manuscript read like an audit log. Conversely, a numerical 95% CI with two stage repeats looks more authoritative than the accompanying caveats justify. The most consequential limitations should be visible once, clearly, and carried by accurate table labels.

**Concrete fix:** Define the screening protocol once in §7 and reuse the term consistently. Replace internal process vocabulary (“deadline,” “supervisor process-envelope,” “manuscript audit,” “authoritative experiments”) with the scientific measurement boundary. Remove the 95% CI column from the main construction table for this draft; report observed repeat values/ranges or leave dispersion in the appendix, explaining that composed totals sum independently measured stages. Add one compact evidence-scope paragraph distinguishing: measured internal system comparisons; heuristic transfer with a major failure; measured build time without matched-quality confirmation; and unmeasured external competitiveness. Keep the abstract's Amazon qualifier and the explicit measured Recall threshold.

**Classification:** Editorial/statistical presentation repair now. More repeats, current-protocol external baselines, and querying every accelerated index are substantive missing experiments, not polishing tasks.

## Evidence boundaries that remain after all writing fixes

The central factorial is useful: it separates controlled upper-topology substitutions from joint base/provider comparisons, keeps no-crossing cases visible, and reports that no static topology dominates. The proofs also avoid turning graph reachability into an ANN accuracy guarantee. These strengths should be preserved.

Nevertheless, the current manuscript cannot substantiate a broad claim of state-of-the-art filtered ANN performance. It needs current-protocol comparisons to relevant external systems, including a meaningful filter-first baseline for the broad-predicate cases, and repetition sufficient to distinguish close results. The dramatic broad-predicate gains are relative to a very weak internal base at those workloads; they show an internal repair, not external competitiveness. The automatic policy has little positive held-out benefit and one severe regression, so it currently supports a reproducible heuristic with known limits rather than successful automatic tuning. Construction timings remain useful engineering evidence, but an equal-quality acceleration claim requires rebuilding/re-querying the compared outputs. None of these gaps should be hidden by stronger prose.

Only review outputs were created: `/tmp/mlung-r1-vldb.md` and two temporary PDF-page renders for visual inspection. No manuscript files were changed.
