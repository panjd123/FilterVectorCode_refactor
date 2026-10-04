Second-pass method and proof review
===================================

Scope: `docs/papers/multilevel_ung/sections/{frontier,index_query,construction,problem_motivation}.tex`, checked against the local source snapshot and relevant static experiment configurations. Review only; no experiments or source edits. Line references identify the text read for this pass and may shift during assembly. The already assigned fixes for LNG candidate notation, nonempty observed labels, and topology notation are deliberately excluded.

Verdict: the revised method is technically credible and substantially consistent with the implementation. Proposition 1 is correct; the forest edge count and distinction between prefix edges and LNG cover edges are correct. Predicate safety and level isolation hold for the declared root-authorized containment path. I found no new counterexample to that intended algorithm. A few formal and pseudocode details should be corrected before calling the section implementation-exact. These are local textual fixes, not evidence that the method or experiments must be replaced.

Remaining findings
------------------

1. **Medium — make the authorization invariant explicit in Proposition 2.**

   Paper: `index_query.tex:206–217`.

   The stated assumptions say group seeds are eligible, block seeds belong to authorized blocks, and overlay edges remain within a block or follow strict-superset transitions. Those assumptions alone do not ensure that an overlay-tagged group seed belongs to an authorized block. An eligible point can lie in a block whose other members are ineligible. The proof then uses “inside a certified block” without having explicitly established that all overlay states have certified owners.

   The surrounding seeding description supplies the missing condition, and the code enforces it. Group seeds receive an overlay tag only from an authorized direct owner (`UNG/codes/src/uni_nav_graph_search_backend.cpp:684–695`). Every scanned overlay edge checks the query authorization of its source owner and the matching level (`.../uni_nav_graph_search_backend.cpp:968–982`; `UNG/codes/include/ung_special_block_activation.h:102–115`). Source/direct-target ownership is validated in `UNG/codes/include/ung_special_blocks.h:124–182`.

   Suggested addition: “Every overlay-tagged seed belongs to a direct-member block authorized by Q, and an overlay edge is expanded only when its declared source owner is authorized and matches the state's level.” Equivalently state that Proposition 2 applies to Algorithm 3 with the previously specified seeding and permitted-edge rules. This is a formal completeness issue, not an observed implementation bug.

2. **Medium — show the seed-scoring and visited-state order in query pseudocode.**

   Paper: `index_query.tex:191–198`.

   The algorithm currently creates seeds and trims them, but never marks seeds seen. In production, `add_entry_point` marks each `(id, level)` seen immediately (`UNG/codes/src/uni_nav_graph_search_backend.cpp:623–632`), every collected seed is distance-scored (`:727–737`), and only then are block/group seed lists trimmed (`:739–759`). Thus a discarded seed was still scored and cannot later re-enter through an edge at that level. The current pseudocode can instead be interpreted as allowing such rediscovery and as scoring only retained seeds.

   Suggested replacement: “Deduplicate seeds by (id, level), mark them seen, and compute their distances; retain block seeds first and fill remaining capacity with group seeds.” The text already correctly describes block priority and retagging, so this is a small addition with important reproducibility implications.

3. **Low — include self when assigning terminal block roots.**

   Paper: `index_query.tex:93` says “nearest marked ancestor.”

   A terminal that is itself emitted as a block root belongs to that block. The implementation starts collection at the root and excludes a marked node only when `cur != root` (`UNG/codes/src/uni_nav_graph_special_blocks.cpp:1661–1675`). Write “nearest marked ancestor, including the terminal itself.” If ancestor is otherwise defined to include self, explicitly say so here because `frontier.tex` correctly uses proper ancestors elsewhere.

4. **Low — define whether query-cost entry time includes vector-seed setup.**

   Paper: `index_query.tex:238–247`.

   The array-queue bound is now safe for insertion attempts and removals. However, `T_entry` is not explicitly defined here. If it means only group-frontier discovery, the expression omits seed-level selection and initialization work: each frontier group scans ownership levels (`UNG/codes/src/uni_nav_graph_search_backend.cpp:675–707`), and seed lists are created, deduplicated, and selected (`:623–632,739–759`). Scoring is correctly covered by `D_q d`, provided `D_q` includes rejected seed distances.

   Easiest repair: define `T_entry` as discovery plus seed setup, excluding distances already charged to `D_q d`, or add a separate `T_seed` term. If the expression is intended as an expected bound, state that the initializer uses `std::nth_element` and hash-based deduplication, which have average/expected complexity qualifications.

5. **Low — remove an unsupported distance-cache implication.**

   Paper: `index_query.tex:253–254` mentions per-level visited state and “distance caches.”

   This level-local CPU path stores distances in candidate/seed records; it does not maintain a per-vector distance memoization cache. A vector can be scored again at a different level. `UNG/codes/include/search_cache.h:16–31,40–50` has visited structures and optional GPU distance buffers, but no such memoization cache; `UNG/codes/src/uni_nav_graph_search_backend.cpp:813–843` scores each newly visited level-state.

   Prefer “per-level visited state, seed buffers, and queue backing storage.” This preserves the correct workspace point without suggesting distance reuse that the algorithm does not perform.

6. **Low — describe the default small-block route as exact top-degree, not a complete graph.**

   Paper: `construction.tex:33–38` says “exact or bounded complete graphs.”

   Under the default `UNG_SPECIAL_INTRA_EXACT_TOPK=true`, the small route calls `build_exact_topk_graph_for_points(..., special_block_max_degree)` (`UNG/codes/src/uni_nav_graph_special_blocks.cpp:2697–2700,2803–2810`). This is an exact top-R neighbor graph and is not complete unless the block is sufficiently small. Bounded complete and complete graphs are alternative routes.

   Suggested wording: “small blocks with exact top-R graphs (or the configured bounded-complete alternative).” The 2048/8192 thresholds and 256 sampled-candidate setting are confirmed in `experiments/multilevel_special/config.authoritative_amazon_hierarchy_build_timing.json:33–35`; medium and large routes match source `:2821–2862`.

7. **Low — make directed reachability explicit in Proposition 1's lifting caveat.**

   Paper: `frontier.tex:35–40`.

   “A connected searchable local graph” could mean weak connectivity, which is insufficient for directed vector graphs. Prefer “each local graph is reachable from its seeded or incoming portal vertices, and required cross edges are reachable and materialized.” This clarifies an assumption rather than changing the valid group-level theorem.

Verified substantive repairs
----------------------------

- `frontier.tex:17–33`: exact `G-r` edge count and eligible-prefix-frontier coverage are correct. Construction emits the nearest-terminal-ancestor edge at `UNG/codes/src/ung_special_block_trie.cpp:277–300`.
- `frontier.tex:43–99`: measured algorithm now matches bitmap intersection followed by cached terminal-ancestor filtering, including empty-query root-frontier copying (`.../ung_special_block_trie.cpp:532–654,776–804`; production call at `.../uni_nav_graph_entry_provider.cpp:140–145`). The representation-aware cost and `A_Q <= c h` bound are appropriate.
- `frontier.tex:78–85`: optimized LNG is correctly described as cardinality buckets with a sorting fallback and unchanged pairwise containment pruning (`.../uni_nav_graph_label_graph.cpp:151–204`).
- `index_query.tex:98–103`: residual groups, strict emission threshold, independent layers, and non-nested direct membership match `.../uni_nav_graph_special_blocks.cpp:2237–2283`.
- `index_query.tex:113–123`: a prefix edge need not be an LNG cover edge; the reachability relation is the correct comparison.
- `index_query.tex:151–167`: highest-authorized-upper-level routing, finest authorized owner tagging, block-seed priority, and absence of duplicate base seeds agree with `UNG/codes/include/ung_special_block_activation.h:28–65` and search backend `:660–759`.
- `index_query.tex:219–223`: limiting safety to root-prefix authorization is supported by the authoritative campaign's `UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE=1` (`experiments/multilevel_special/generate_authoritative_campaign.py:102`). Legacy common-label authorization is appropriately excluded.
- `index_query.tex:225–236`: isolation and fallback are appropriately conditional and no longer imply coverage after retagging/pruning. Production dispatch is `UNG/codes/src/uni_nav_graph_search.cpp:39–60,104–115`.
- `index_query.tex:238–254`: array versus lazy-heap distinctions fix the previous asymptotic error. Array duplicate scans are at `.../ung_special_candidate_queue.cpp:111–142`; lazy slot retention at `:46–57,149–180`.
- `construction.tex:87–111`: abstract partition work is separated from implementation overhead; cubic generic LNG topology work and dense `O(m(N+G))` ownership maps are correctly included (`.../ung_group_topology.cpp:19–55,105–108`; `.../uni_nav_graph_special_blocks.cpp:2124–2129`).
- `construction.tex:123–166`: DRH recurrence, topology choice, and empirical-proxy qualification match `experiments/multilevel_special/derive_static_hierarchy.py:13–46`. Its general closed form also assumes the integer-overflow threshold guard does not trigger; this is immaterial for the presented datasets but can be stated beside the existing depth-cap qualification if desired.
- `construction.tex:113–119,168–177`: construction-quality evidence and static-update limitations are now honest and do not overclaim guarantees.
- `problem_motivation.tex:25–61`: the catalog example and drawn containment/prefix relations are internally correct. Its distinction between group relations and vector adjacency is useful and consistent with implementation.

Review limits: I checked method claims, source control flow, and the cited configuration constants. I did not independently re-audit numerical result tables, rendered layout, citations, or accelerated-index quality evidence in this pass. No experiments were launched. Only this review report was written.
