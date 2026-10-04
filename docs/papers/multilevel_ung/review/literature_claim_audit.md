# Literature and claim audit for the ML-UNG rewrite

Audit date: 2026-10-04. Scope: the four requested claims, the current `docs/papers/multilevel_ung/main.tex` and `references.bib`, and closely related recent work. This audit does not rerun experiments or change the manuscript. Primary papers, publisher DOI metadata, and the UNG authors' implementation were consulted. Downloaded source snapshots are in `literature_sources/` beside this file; extracted text line numbers below refer to those snapshots.

## Decisions for the rewrite

| Proposed claim | Verdict | Defensible replacement |
|---|---|---|
| Improvements to global graph search **only** work at medium/high selectivity | Unsupported as a universal claim | Low eligible fractions can disrupt filtered graph connectivity or require additional exploration. The severity depends on construction, query/vector correlation, degree, filter support, and the fallback policy. |
| Curator's temporary indexes and partition traversal are expensive for multi-label AND | Mechanisms exist; generic performance conclusion contradicted by Curator's own experiments | Curator introduces predicate preparation, temporary partition construction, centroid traversal, and buffer scanning. Measure those stages on the target workload before attributing a disadvantage to them. |
| FAVOR falls back to brute force and therefore fails when the eligible count is large | First clause supported conditionally; failure claim unsupported | FAVOR uses a filtered scan below an estimated 1% selectivity and exclusion-distance HNSW otherwise. Its scan branch has work proportional to the absolute qualifying set, so low selectivity alone does not imply low scan cost. |
| Trie has `n−1` edges versus LNG `O(n²)`, therefore lower ANN latency | Edge bound requires explicit vertex/root convention; latency implication unsupported | A terminal-prefix forest has `g−r` logical edges. A containment-cover graph can have Θ(g²) logical edges. These storage bounds do not determine visited vector edges, entry costs, Recall, or ANN latency. |

Use `s = |E(Q)|/N` throughout: **low selectivity means a small eligible fraction**, whereas “highly selective” conventionally means a restrictive predicate. Do not interchange these phrases. Report absolute eligible count as well as fraction. Multi-label AND is not synonymous with low selectivity: correlated/common labels can retain a large fraction, and even a small fraction can mean many vectors at large `N`.

## 1. ACORN and the scope of the graph-search claim

**Primary source.** Liana Patel, Peter Kraft, Carlos Guestrin, and Matei Zaharia. *ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data*. PACMMOD 2(3), 1–27, 2024; SIGMOD 2024. DOI: [10.1145/3654923](https://doi.org/10.1145/3654923). Public author manuscript: [arXiv 2403.04871v1](https://arxiv.org/html/2403.04871v1/).

Relevant passages and boundaries:

- **§3.2–3.2.1:** performance depends on selectivity, dataset size, and query correlation. Low correlation can make post-filtered graph search expensive even when one scalar selectivity value looks favorable. This directly argues against a universal selectivity-only classification.
- **§4:** ACORN searches the predicate subgraph; ACORN-γ expands candidate neighborhoods during construction, while ACORN-1 expands them during search. The two variants trade construction/storage for query work.
- **§5.2:** “One simple choice for γ is 1/s_min,” followed by a cost-based rule: search the index if estimated selectivity exceeds `1/γ`, otherwise pre-filter. The minimum supported graph-search fraction is configurable, not an invariant medium/high regime.
- **§7.3, Fig. 9:** reports advantages across selectivity percentiles. A percentile is not the eligible fraction itself; do not read “1st percentile” as `s=1%`.
- Snapshot anchors: `acorn.txt:160–188, 208–210, 297–300, 510–512`.

**Curator's narrower evidence.** Curator §1 and §6.4 show fixed ACORN configurations degrading at low eligible fractions on their workloads; their model argues that obtaining `m` qualifying neighbors may require roughly `m/s` candidates. This motivates a resource/performance tradeoff. It is not a lower-bound theorem that every graph adaptation fails below a universal threshold. ACORN-1 also avoids paying all neighborhood expansion in persistent index storage.

**Suggested manuscript prose:**

> Predicate-agnostic graph methods preserve a shared vector index and adapt neighborhood exploration to filtering. ACORN enlarges effective neighborhoods through construction-time or search-time expansion, while FAVOR biases traversal with a selectivity-aware exclusion distance. At small eligible fractions, maintaining useful connectivity can require more exploration or a fallback plan. The crossover depends on the index, data distribution, and query–vector correlation, which motivates evaluating both eligible fraction and absolute eligible-set size.

Do not describe graph adaptations as supporting “only” medium/high selectivity or imply that selectivity is sufficient to predict their performance.

## 2. Curator and multi-label AND

**Primary source.** Yicheng Jin, Yongji Wu, Wenjun Hu, Bruce M. Maggs, Jun Yang, Xiao Zhang, and Danyang Zhuo. *Curator: Efficient Vector Search with Low-Selectivity Filters*. PACMMOD 4(1), 1–27, 2026; SIGMOD 2026. DOI: [10.1145/3786635](https://doi.org/10.1145/3786635). Public manuscript: [arXiv 2601.01291v3](https://arxiv.org/html/2601.01291v3).

What the paper actually implements:

- **§3:** a shared hierarchical k-means tree with embedded label-specific partitions and buffers. Sparse qualifying regions can stop at coarser tree nodes.
- **§4.2, Algorithm 2 and Fig. 5:** arbitrary predicates produce a sorted list of qualifying vector IDs, supplied by an external relational system. Their layout encodes subtree membership. Temporary index construction recursively partitions ID ranges using binary search, reuses base-tree nodes/centroids, and stores range offsets plus pointers. It does **not** retrain a clustering index or perform a new vector-distance-based index build per query.
- **§4.2:** “Evaluating attribute predicates and composing bitmaps inside the database is far cheaper than vector distance computation, so we treat this sorted list as the algorithm input.” This is a consequential timing boundary: reproduce it explicitly or charge equivalent filtering preparation to every system in an end-to-end comparison.
- **§4.2, Pre-indexing Filters:** temporary structures may be cached or integrated as virtual labels. The paper treats that policy as pluggable.
- **§6.2, Fig. 9:** two-label AND and OR queries on YFCC-10M; AND averages approximately `0.01%` selectivity. “Curator without indexing achieves similar performance to the indexed version,” and both outperform ACORN on these tested complex predicates. Thus AND alone is not evidence that temporary index construction is a bottleneck.
- **§6.7, Fig. 15(c):** centroid traversal accounts for `26.9%` to `18.9%` of **distance computations** as data grows from 2M to 10M vectors. These are not percentages of end-to-end latency and not a multi-label-AND-specific measurement.
- **§6.4:** the dual-index integration uses selectivity estimates and fixed thresholds selected by offline profiling. This is a supported contrast to DRH's lack of query-performance calibration.
- Snapshot anchors: `curator.txt:210–258, 322–332, 342–349, 360–366`.

**Suggested manuscript prose:**

> Curator addresses sparse predicates with label-adaptive partitions embedded in a shared clustering tree. For complex predicates it builds a lightweight temporary partition structure from sorted qualifying IDs; its reported two-label AND experiments show little overhead relative to pre-indexed predicates. Our design instead reuses coarse blocks authorized by containment and searches their prebuilt proximity graphs. The two approaches place work in different query stages, motivating separate measurements of predicate preparation, entry discovery, authorization, and vector search.

**What would support a stronger local limitation:** complete phase timing on the actual multi-label workload, including qualifying-ID generation and any sorting/materialization; warmup/caching policy; centroid and point-distance counts; Recall-matched search; and evidence tying a measured loss to the alleged stage. Existing `EXPERIMENT_STATE.md:65` says a historical Curator timing anomaly was localized to `prepare_filter`, **not ANN search**. That is adapter/protocol evidence, not a general theorem about Curator's partitions.

## 3. FAVOR identified and checked

**Exact paper; not a guessed acronym.** Junjie Song, Yu Liu, Guoyu Hu, Zhongle Xie, Ming Yang, Beng Chin Ooi, and Ke Zhou. *FAVOR: Efficient Filter-Agnostic Vector ANNS Based on Selectivity-Aware Exclusion Distances*. PACMMOD 4(3), 1–25, 2026; SIGMOD 2026. DOI: [10.1145/3802060](https://doi.org/10.1145/3802060). Public manuscript: [arXiv 2605.07770v1](https://arxiv.org/html/2605.07770v1). Title/authors/volume/issue/pages/date are verified against Crossref's publisher deposit in `favor_crossref.json`.

- **§3.2:** uses an ordinary proximity graph, a selectivity-driven selector, an inline-filtering graph algorithm, and a pre-filtering brute-force algorithm.
- **§4.1:** “We determine the λ=1% based on empirical measurement.” When estimated `p < λ`, it chooses pre-filtering brute-force; otherwise it uses HNSW search. A broad predicate with many qualifying vectors is normally directed to the graph branch, so it is incorrect to imply FAVOR always scans large eligible sets.
- **§4.2:** estimates selectivity using sampling rather than an exact full-dataset count.
- **§5:** the graph branch uses exclusion distances to favor qualifying vectors while allowing nonqualifying vertices to preserve navigation. This is distinct from simply dropping every nonqualifying neighbor.
- **§6.2, Fig. 7:** at Recall@10=95% on SIFT1M and GIST1M range workloads, the scan/graph crossover lies between 1% and 3%; the authors choose 1% conservatively. Those observations justify their setting on tested data; they do not establish a universal threshold across `N`, dimension, metadata distribution, and hardware.
- Snapshot anchors: `favor.txt:388–413, 417–443, 767–777`.

**Supported analytical limitation:** if the selected scan branch processes `sN` qualifying vectors in dimension `d`, its vector-distance work is Θ(`sNd`) under a dense full-distance implementation, apart from predicate evaluation, top-k selection, and layout overhead. Therefore, even `s<1%` may leave many vectors on a very large dataset. This observation is already made for pre-filtering generally in ACORN §3.2 and Curator §1. It is a conditional cost argument, not evidence of a FAVOR failure or inaccurate results. Favor may also choose the graph branch due to its selectivity estimate; adapter route traces must be checked before assigning a local run to the scan branch.

**Suggested manuscript prose:**

> FAVOR estimates predicate selectivity and switches between pre-filtered scanning and an HNSW traversal guided by exclusion distances. Its published implementation uses an empirically chosen 1% switch. The scan branch remains sensitive to the absolute number of qualifying vectors, while its graph branch depends on filtered navigation quality. Our reusable containment-safe overlays address a different design choice: the navigation granularity available inside a label-containment graph. We do not infer superiority over FAVOR from its fallback mechanism.

## 4. UNG, terminal-prefix edges, and complexity

**Primary publication.** Yuzheng Cai, Jiayang Shi, Yizhuo Chen, and Weiguo Zheng. *Navigating Labels and Vectors: A Unified Approach to Filtered Approximate Nearest Neighbor Search*. PACMMOD 2(6), 1–27, **2024 publication; SIGMOD 2025 presentation**. DOI: [10.1145/3698822](https://doi.org/10.1145/3698822). The [official SIGMOD 2025 issue listing](https://2025.sigmod.org/toc-2-6.html) and the [first author's page](https://yz-cai.github.io/) verify the presentation context.

Direct ACM PDF retrieval returned HTTP 403. Consequently, this audit does not invent paper section numbers or claim to have inspected its full PDF. The publication's abstract and DOI metadata were checked, and the authors' [official code](https://github.com/YZ-Cai/Unified-Navigating-Graph) independently verifies the mechanism:

- [`codes/src/uni_nav_graph.cpp:90–134`](https://github.com/YZ-Cai/Unified-Navigating-Graph/blob/master/codes/src/uni_nav_graph.cpp#L90) gets Trie candidates, sorts by label-set size, and prunes candidates containing an already retained minimal superset.
- [`uni_nav_graph.cpp:227–239`](https://github.com/YZ-Cai/Unified-Navigating-Graph/blob/master/codes/src/uni_nav_graph.cpp#L227) builds each LNG adjacency list from minimal strict supersets with `avoid_self=true`.
- [`uni_nav_graph.cpp:469–474`](https://github.com/YZ-Cai/Unified-Navigating-Graph/blob/master/codes/src/uni_nav_graph.cpp#L469) gets minimal supersets for containment-query entry groups.
- [`codes/src/trie.cpp:64–111`](https://github.com/YZ-Cai/Unified-Navigating-Graph/blob/master/codes/src/trie.cpp#L64) is the candidate-discovery stage. Returning terminal group IDs and pruning under containment are distinct from materializing prefix-parent edges.

The existing manuscript correctly separates UNG's Trie-assisted lookup from LNG connectivity (`main.tex:176–194, 237–247`). Preserve that distinction.

### Correct edge statements and a tight example

Let `g` be the number of materialized group/block terminals in **one physical layer**, not the number of vectors and not the number of all prefix nodes. Connect each terminal to its first terminal descendants. Each non-root terminal has exactly one nearest terminal ancestor. If `r` terminals have no terminal ancestor, the resulting directed forest has exactly **`g−r` edges**, hence at most `g−1` for nonempty `g`. Equality requires one root terminal. An artificial root changes both the vertex and edge counts. An uncompressed Trie with `t` prefix nodes has `t−1` edges, but generally `t ≠ g`.

For the LNG containment-cover relation, Θ(`g²`) logical edges are possible. A self-contained construction is useful: take `k` singleton sets `A_i={a_i}` and `k` sets `B_j={a_1,…,a_k,b_j}`. Every `A_i` is a strict subset of every `B_j`, there is no observed set between them, and neither family has within-family containments. Thus the LNG on `g=2k` groups is directed complete bipartite with `k²=g²/4` cover edges. This is a structural worst case, not a claim that any measured dataset exhibits it. Label-set representation and containment-check costs also depend on label-set length.

This proof does **not** imply lower end-to-end ANN latency for the prefix forest:

1. The forest drops non-prefix containment edges and may require more entry terminals. For observed `{2}` and `{1,2}` under ascending labels, the LNG has `{2}→{1,2}` but the two terminals lie on different prefix branches. Query `{2}` must seed both branches in a prefix-only topology to preserve reachability.
2. Logical group edges differ from vector-level materialized edges; group masses, edge-realization budgets, intra-group graphs, and extra completion paths matter.
3. ANN only visits part of a stored graph. Fewer stored edges can increase visited nodes, entry costs, or search budget needed for a target Recall.
4. A graph-edge count says nothing by itself about geometry, path quality, Recall, queue work, or memory locality.
5. Current local evidence is already non-monotone: `generated_topology_factorial.tex:49–51` reports a Trie base losing in several regimes and a more consistent benefit from the coarse upper Trie. Cross-base comparisons also change their compatible entry providers.

**Suggested manuscript prose:**

> Nearest-terminal prefix connectivity bounds logical edges by `g−r` for a layer with `g` terminals and `r` roots, whereas the containment-cover relation can have Θ(`g²`) edges. This bounds topology storage but does not guarantee faster ANN search: prefix connectivity removes non-prefix containment routes and changes the required entry frontier. We therefore evaluate topology at matched Recall and separately report entry work, materialized vector edges, and visited search work.

Prefer “terminal-prefix forest” when proving the bound. “Trie frontier has n−1 edges” conflates the entry frontier with the entire materialized topology. The existing `main.tex:907–912` is appropriately cautious; a proof can strengthen it without adding a latency promise.

## 5. Historical local comparisons that constrain the narrative

These are **existing historical system results**, verified by reading the local saved tables; they are not reruns, not the current 14-configuration factorial, and not a replacement for a fresh unified comparison. The protocol uses Amazon x1, 1,000 queries, K=10, 100 threads and heterogeneous system adapters. Recall thresholds also differ by workload. Data and timing provenance should be revalidated before promoting these rows into the current main table.

`experiments/multilevel_special/EXPERIMENT_STATE.md:41,64–68,85` explicitly records external systems and negative results. The more concrete generated table is `experiments/multilevel_special/results_summary/paper_results.md:56–73`, with backing rows in `external_canonical_measured_points.csv` and source files `source/navix_favor_robust.csv` and `source/curator_robust.csv`.

| Mean eligible fraction | Declared Recall threshold | Tuned Multi-level batch median ms / Recall | FAVOR ms / Recall | Curator ms / Recall |
|---|---:|---:|---:|---:|
| 24.915% | 0.90 | 2986.040 / 0.9023 | **1754.680 / 0.9133** | **1838.554 / 0.9744** |
| 49.971% | 0.85 | 383.260 / 0.8518 | **200.929 / 0.8644** | 1590.117 / 0.9742 |
| 74.994% | 0.87 | 771.468 / 0.8729 | **265.841 / 0.8920** | 1942.862 / 0.9629 |

At 24.915%, NaviX is also faster than that multilayer configuration: 2413.740 ms at Recall 0.9153. The table includes lower-fraction results where the local UNG-derived methods outperform FAVOR and Curator, reinforcing a workload-dependent interpretation. These heterogeneous, non-identical actual-Recall comparisons must not be described as exact equal-Recall curves.

**Required narrative boundary:** claim measured gains over the stated UNG-style base in the current protocol, not blanket leadership over external filtered-ANN systems. Distinguish a literature mechanism, a conditional complexity concern, an adapter-specific profile, and a demonstrated current-system bottleneck. Neither low selectivity nor a fallback branch alone demonstrates that Curator or FAVOR “fails.”

## 6. Recent work that should inform novelty and coverage

The following are targeted additions, not an exhaustive bibliography.

| Work and verified source | Relevant mechanism / implication |
|---|---|
| **LSSG**, Ziqi Wang, Jingzhe Zhang, Shuo Shen, Wei Hu. *Fast Label-Filtering Approximate Nearest Neighbor Search via Progressive Label Set Stratification*. [arXiv:2609.15058v1](https://arxiv.org/html/2609.15058v1), posted 14 Sept 2026. | §3 builds multiple graph tiers by label-set-distance thresholds; equality, containment and overlap use tier-ordered in-filtering traversal. The bottom tier is label-agnostic, upper tiers tighten label consistency. A direct recent neighbor to any “multilevel label-aware graph” claim. Distinguish coarse block aggregation, exact block authorization and level-local expansion from label-distance edge tiers. Cite as a 2026 preprint: the HTML contains a placeholder DOI and future SIGMOD 2027 frontmatter, which do not verify acceptance. |
| **SIEVE**, Zhaoheng Li, Silu Huang, Wei Ding, Yongjoo Park, Jianjun Chen. *SIEVE: Effective Filtered Vector Search with Collection of Indexes*. PVLDB 18(11), 4723–4736, 2025. [DOI 10.14778/3749646.3749725](https://doi.org/10.14778/3749646.3749725); [full text](https://arxiv.org/html/2507.11907v2). | §3–4 construct and select a collection of useful subindexes using workload evidence, index budgets, and performance models. Necessary context for claiming automatic structure selection; DRH's distinction is its structural inputs and lack of query-performance calibration, not automatic index selection in general. |
| **Query-aware Routing**, Qianqian Xiong and Mengxuan Zhang. *Query-aware Routing for Filtered Approximate Nearest Neighbors Search*. [arXiv:2606.19898v1](https://arxiv.org/html/2606.19898v1), 2026. | §4 combines learned per-query Recall prediction with an offline table of method/parameter performance. Explicitly finds no method dominates. Treat as preprint unless final publication is independently verified. |
| **Compass**, Chunxiao Ye, Xiao Yan, Eric Lo. *Compass: General Filtered Search across Vector and Structured Data*. [arXiv:2510.27141v2](https://arxiv.org/html/2510.27141v2), 2025 preprint. | Coordinates candidate generation from existing vector and relational indexes through a shared queue, supporting conjunctions, disjunctions, and ranges. Prevents equating all general-filter methods with pure filtered graph traversal. Publication venue not established by the inspected version. |
| **Unified benchmark**, Jiayang Shi, Yuzheng Cai, Weiguo Zheng. *Filtered Approximate Nearest Neighbor Search: A Unified Benchmark and Systematic Experimental Study [Experiment, Analysis & Benchmark]*. [arXiv:2509.07789v1](https://arxiv.org/html/2509.07789v1), 2025. | Emphasizes coupled parameters, metadata implementation differences and workload effects; §5.6 studies selectivity. Its inspected HTML has placeholder PVLDB metadata, so do not copy its placeholder 2020 volume/DOI. |

The existing SeRF, iRangeGraph, UNIFY, dynamic-range, RangePQ, and Elastic Index Selection citations already support an ordered-range versus set-containment discussion. They do not establish that hierarchical indexing or automatic selection is new. A claim focused on reusable containment-authorized coarse graph blocks is more precise and easier to defend.

## 7. Bibliography changes

The existing ACORN, UNG, and Curator DOI/title/volume/issue/page fields match the publisher deposits. Preserve UNG's 2024 bibliographic year while describing its SIGMOD presentation as 2025. Crossref also confirms the currently stored publication years for UNIFY (December 2024, volume 18 issue 4) and Elastic Index Selection (December 2025, volume 19 issue 4); do not change these merely to match the later conference cycle. FAVOR is currently missing from `references.bib` and should be added under its exact title:

```bibtex
@article{song2026favor,
  author = {Song, Junjie and Liu, Yu and Hu, Guoyu and Xie, Zhongle and
            Yang, Ming and Ooi, Beng Chin and Zhou, Ke},
  title = {{FAVOR}: Efficient Filter-Agnostic Vector {ANNS} Based on
           Selectivity-Aware Exclusion Distances},
  journal = {Proceedings of the ACM on Management of Data},
  volume = {4},
  number = {3},
  pages = {1--25},
  year = {2026},
  doi = {10.1145/3802060}
}

@article{li2025sieve,
  author = {Li, Zhaoheng and Huang, Silu and Ding, Wei and Park, Yongjoo and Chen, Jianjun},
  title = {{SIEVE}: Effective Filtered Vector Search with Collection of Indexes},
  journal = {Proceedings of the VLDB Endowment},
  volume = {18},
  number = {11},
  pages = {4723--4736},
  year = {2025},
  doi = {10.14778/3749646.3749725}
}

@misc{wang2026lssg,
  author = {Wang, Ziqi and Zhang, Jingzhe and Shen, Shuo and Hu, Wei},
  title = {Fast Label-Filtering Approximate Nearest Neighbor Search via
           Progressive Label Set Stratification},
  year = {2026},
  eprint = {2609.15058},
  archivePrefix = {arXiv},
  primaryClass = {cs.DB},
  url = {https://arxiv.org/abs/2609.15058}
}

@misc{xiong2026routing,
  author = {Xiong, Qianqian and Zhang, Mengxuan},
  title = {Query-aware Routing for Filtered Approximate Nearest Neighbors Search},
  year = {2026},
  eprint = {2606.19898},
  archivePrefix = {arXiv},
  primaryClass = {cs.DB},
  url = {https://arxiv.org/abs/2606.19898}
}
```

No speedup from a cited paper should be copied as if it were a measurement in this manuscript. When using an outside result to explain a design tradeoff, retain its dataset, predicate regime, Recall target and timing boundary.
