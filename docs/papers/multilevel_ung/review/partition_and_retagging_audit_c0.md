# Postorder partition structure and seed-coverage audit

This is a mathematical and source audit, not a vector-search benchmark. Manuscript and production code were not modified. Live code was read only from `/home/sunyahui/worktrees/FilterVectorCode_multilevel_special` through the supplied SSH route; that repository reported HEAD `ed4c48cab3ac56f6bc36e7ef895d6afd3a18c989`. The prohibited `/home/graphdb/FilterVectorCode_refactor` repository was not accessed. Source line references below refer to the live files inspected.

**Conclusions.** Block count is nonincreasing in the threshold. For integer masses and threshold, `B(T) <= floor(N/(T+1))`. Represented mass is neither globally nonincreasing nor globally nondecreasing, and independent direct-member partitions need not be nested even when block counts and total represented mass are equal. A stronger useful fact holds on count plateaus: equal block counts imply that increasing the threshold can only enlarge the represented point set. The supplied LNG-minimal-entry retagging counterexample is valid; complete prefix-frontier entries satisfy a complementary exhaustive-coverage theorem under explicit portal-reachability and no-pruning assumptions.

## 1. Model and correspondence to code

Let a finite rooted canonical trie have synthetic root `r`, with weight `w(r)=0`. Each nonempty observed group is a terminal with nonnegative integer point mass; nonterminal weights are zero. Positive terminal weights sum to `N`. Groups are indivisible for ownership, but it is convenient to regard their points as distinct atoms.

For threshold `T >= 0`, process nonroot nodes in postorder. At node `v`, form the pending point set `V_v(T)` from its own terminal points and the residual point sets returned by its children. Emit a block if `|V_v(T)| > T`; an emitted block owns `V_v(T)` and returns the empty residual. Otherwise return `U_v(T)=V_v(T)`. The synthetic root never emits and returns the union of its children's residuals. Define:

- `b_v(T)`: blocks emitted inside the subtree rooted at `v`;
- `U_v(T)`: residual points returned from that subtree;
- `B(T)=b_r(T)`;
- `C(T)`: represented point set, the disjoint union of all emitted direct-member sets;
- `M(T)=|C(T)|=N-|U_r(T)|`.

This matches `UNG/codes/src/uni_nav_graph_special_blocks.cpp:2237–2268`: independent uncovered array, child accumulation, `idx == 0 || uncovered <= threshold` skip, then reset to zero after emission. Member collection at `:1655–1683` includes the root itself, stops at another same-layer emitted root, and records direct terminal groups. It therefore assigns each terminal to its nearest marked ancestor **including itself**. Layers repeat independently at `:2273–2283`.

All mathematical statements assume exact nonnegative arithmetic, without machine-integer overflow. The production implementation uses finite-width point counters.

## 2. Count upper bound

**Theorem 1.** For integer `T >= 0`,

`B(T)(T+1) <= M(T) <= N`, hence `B(T) <= floor(N/(T+1))`.

**Proof.** The pending points collected at emission are exactly the block's eventual direct members: descendants already emitted have been removed by their reset. Distinct blocks have disjoint direct members. Every emitted block contains integer mass strictly greater than `T`, hence at least `T+1`. Sum over the emitted blocks. This also proves `B(T) <= G_+`, where `G_+` is the number of positive-mass terminal groups. QED.

The bound is tight over input families: use separate root-child terminal groups of mass `T+1`. It is a minimum-mass bound, **not** a maximum-size guarantee. One block may contain all `N` points.

For `T>0`, `N/T` is consequently a valid looser upper envelope for block count. It can be arbitrarily loose: `N` singleton root-child groups with `T=1` yield no blocks, despite `N/T=N`. The synthetic root cannot collect those residual branches into a block. There is no input-independent positive constant-factor lower bound relating actual `B(T)` to `N/T`.

An auxiliary residual bound is `|U_r(T)| <= degree(r) T`, since every nonroot child returns either zero or at most `T` points. This need not be useful when the root has many branches, and it does not imply total residual mass is at most `T`.

## 3. Monotonicity of block count, with a stronger invariant

**Theorem 2.** For any `T1<T2` and every subtree `v`:

1. `b_v(T1) >= b_v(T2)`;
2. if the counts are equal, `U_v(T1) superset-or-equal U_v(T2)` as point sets.

In particular, `B(T)` is nonincreasing.

**Proof by postorder induction.** Write `b_i(c)` and `U_i(c)` for child quantities at threshold `Ti`. Let

`d = sum_c [b_1(c)-b_2(c)] >= 0`,

and let `e_i` be the emission indicator of the current nonroot node. The block-count difference at this node is `d+e_1-e_2`.

- If `d>=1`, this difference is nonnegative, since each indicator is zero or one.
- If `d=0`, every child's count difference is zero. By induction, each low-threshold residual contains its high-threshold residual. Taking their unions with the same own-terminal points gives `V_1 superset-or-equal V_2`, so `|V_1|>=|V_2|`. It is therefore impossible for the low-threshold node not to emit while the high-threshold node emits: `|V_2|>T2>T1` would also force low-threshold emission. Thus the block-count difference cannot be negative.

For the residual-set claim when the final counts are equal, only two cases remain:

- `d=0` and `e_1=e_2`. If neither emits, `U_1=V_1` contains `U_2=V_2`; if both emit, both residuals are empty.
- `d=1`, `e_1=0`, `e_2=1`. The high-threshold residual is empty, so it is contained in the low-threshold residual.

At the synthetic root, both emission indicators are zero. Its block-count difference is `d`; if equal, childwise residual containment passes to their union. The leaf case is the same reasoning with empty child sums. QED.

**Corollary 2a.** If `B(T1)=B(T2)`, then `C(T1) subset-or-equal C(T2)` and `M(T1)<=M(T2)`.

This follows by taking complements in the fixed `N`-point universe. Coverage loss is possible only across a decrease in block count. This is stronger than a mass-only statement, but it still does not imply nested direct-member blocks.

The argument works for nonnegative real masses as well; only Theorem 1's integer `T+1` refinement uses integrality.

## 4. Counterexamples and verified supplied examples

Concatenations denote sorted label sets, and a group symbol denotes all points of that group. Unless specified otherwise, `a<b<c<d<e<f`.

### 4.1 Supplied nonnested example: verified

Weights: `a:1, ab:2, abc:3, ac:2`, total `N=8`.

| Threshold | Emitted roots and direct-member groups | B | M | Residual |
|---|---|---:|---:|---|
| 2 | `abc:{abc}` mass 3; `a:{a,ab,ac}` mass 5 | 2 | 8 | none |
| 4 | `ab:{ab,abc}` mass 5 | 1 | 5 | `a,ac`, mass 3 |

At threshold 2, `abc` emits, `ab` returns its own 2 points, and `a` sees `1+2+2=5`. At threshold 4, `abc` returns 3, `ab` emits `2+3=5`, and `a` returns only `1+2=3`.

The low-threshold `a` block loses `ab` to the higher-threshold `ab` block while its other direct groups become residual. This is not a contraction of the lower partition.

### 4.2 Supplied coverage increase: verified

Weights: `a:1, ab:3, ac:3`, total `N=7`.

| Threshold | Emitted roots and direct-member groups | B | M | Residual |
|---|---|---:|---:|---|
| 2 | `ab:{ab}` mass 3; `ac:{ac}` mass 3 | 2 | 6 | `a`, mass 1 |
| 6 | `a:{a,ab,ac}` mass 7 | 1 | 7 | none |

The same increase already occurs at threshold 4. Together with 4.1, this disproves both global directions of monotonicity for represented mass.

Represented sets can also be incomparable for a single threshold pair. Place example 4.1 under branch `a` and an independent renamed copy of this increase example under branch `d` (`d:1,de:3,df:3`), using thresholds 2 and 4. The higher threshold loses groups `a,ac` and gains group `d`. Thus neither represented set contains the other.

### 4.3 Stronger nonnested example: same count and full coverage

Weights: `a:2, ab:1, ac:1, abc:2`, total `N=6`.

| Threshold | Emitted direct-member blocks | B | M |
|---|---|---:|---:|
| 1 | `abc:{abc}` mass 2; `a:{a,ab,ac}` mass 4 | 2 | 6 |
| 2 | `ab:{ab,abc}` mass 3; `a:{a,ac}` mass 3 | 2 | 6 |

The old `a` block is split between two new blocks. Both partitions cover everything and have equal block count. Neither direct-member partition refines the other. Corollary 2a is satisfied because the represented sets are identical.

### 4.4 Query-authorized mass has no analogous plateau guarantee

Even when `B` is unchanged and global represented coverage increases, authorization can decrease. Use `a:1, ab:3` and query `Q={b}`. At threshold 2, block `ab` has mass 3 and is authorized. At threshold 3, the only block is `a` with mass 4; it is unauthorized. `B` stays 1, `M` rises from 3 to 4, but authorized direct mass falls from 3 to 0.

Authorization can also increase at constant count: in example 4.3 with `Q={b}`, authorized direct mass rises from 2 (`abc`) to 3 (`ab`). Thus block-count monotonicity does not establish monotonic query scale eligibility or routing decisions.

## 5. Bounded structural enumeration

Scratch script: `/tmp/mlung-partition-structure-enumerate.py`.

It enumerated all parent-before-child numbered rooted trees with 1–5 nonroot nodes and weights in `{0,1,2}`, excluding all-zero inputs, at every integer threshold from zero through total mass. This checked 31,134 weighted trees and 915,738 adjacent-threshold subtree comparisons. It verified Theorem 1, count monotonicity, the stronger equal-count residual-set containment invariant, and mass conservation. It also found example 4.3. Parent-before-child numbering is only a convenient enumeration representation, not a restriction needed by the proof. These finite checks corroborate the proofs; they do not replace them. No distance computation or ANN benchmark was performed.

## 6. What this says about DRH

Publication-useful formulation:

> Increasing the emission threshold cannot increase the number of blocks. Each block contains more than the threshold's point mass, giving `B(T)<=floor(N/(T+1))`. DRH's `N/T` term is therefore a simple structural upper envelope, although it may substantially overestimate the realized count. Independent threshold partitions can redistribute membership and leave different residual sets, so increasing the threshold does not imply nested blocks or monotone query-authorized coverage.

This strengthens “purely empirical proxy” in the current construction section without turning the proxy into a prediction of actual block count. It also limits the interpretation of `T+N/T`: the `N/T` part has a count-bound justification, but `T` is not an upper bound on block size or intra-block search work. Minimizing that expression at `sqrt(N)` remains a heuristic tradeoff, not an optimization theorem for the implemented search.

Actual `B(T)` could be used in a future query-calibration-free structural rule. Its monotonicity permits searching for the smallest threshold satisfying a desired block-count cap, subject to the discrete jumps of `B`. Such a rule should still inspect represented mass, block-size distribution, topology edges, and resource costs. Count alone does not predict predicate authorization, Recall, seed retention, or latency. No such alternative selector was implemented or benchmarked in this audit.

## 7. LNG-minimal entry retagging can destroy exhaustive coverage

**The supplied counterexample is valid.** Consider a query admitted to the overlay backend (for example an ungated one-overlay configuration), `Q={b}`, observed singleton-point groups `{b}`, `{b,c}`, `{a,b}`, and `T=1`.

Partition:

- Trie branch `b -> c`: `{b,c}` returns mass 1; node `b` accumulates 2 and emits one block rooted at `{b}`, with direct members `{b}` and `{b,c}`.
- Trie branch `a -> b`: mass 1 never exceeds the threshold, so `{a,b}` remains residual.
- The synthetic root cannot emit a catch-all block.

All three groups are eligible. The LNG-minimal entry provider returns only `{b}`. In the base LNG, `{b}` has cover edges to both `{b,c}` and `{a,b}`. But the group entry `{b}` is retagged to the authorized physical-level-1 block. That block's portal is also seeded at level 1. There is **no level-0 seed**, and the only overlay block has no other block to connect to. Thus the reachable scored point set is contained in `{b},{b,c}`, even with unlimited candidate capacity and perfectly connected local graphs. Placing the nearest eligible vector in `{a,b}` gives a Recall@1 failure that increased queue capacity cannot repair.

Live source verification:

- Authorized block seeds: `UNG/codes/src/uni_nav_graph_search_backend.cpp:660–674`.
- Group entry is assigned the first/finer authorized owner and that tag only: `:675–707`, particularly `:684–695`.
- Seed trimming occurs after those tags are assigned: `:739–759`.
- Overlay transition requires matching level and preserves it: `UNG/codes/include/ung_special_block_activation.h:102–115`.
- Overlay expansion ends with `continue` before the base adjacency branch: `UNG/codes/src/uni_nav_graph_search_backend.cpp:1127–1139`.

This is a coverage failure, not a predicate-safety failure: every reached point is still eligible. It is caused by the combination of LNG-minimal entries and replacement of their base seeds. It is not intrinsic to an LNG base: that same base with complete prefix-frontier entries receives `{a,b}` as a separate entry and avoids this particular gap.

Scope is important: a router that rejects the overlay executes base search, so the counterexample is about an **admitted** overlay query. It establishes a possible structural explanation for persistent no-crossing behavior, but does not establish that this mechanism caused any particular low-selectivity measurement. That empirical attribution would require inspecting the actual seed/ownership/reachability structure for those queries.

## 8. Positive theorem for complete prefix-frontier retagging

**Theorem 3 (exhaustive coverage under complete prefix seeding).** Assume:

1. Entries contain the complete prefix frontier `F(Q)` for the canonical order.
2. Every frontier entry is either retained as a level-0 seed or retagged only to an authorized direct-owner block at a selected layer.
3. Every authorized block at every selected layer has a valid independently retained portal seed; same-state deduplication is allowed, but no required seed is dropped.
4. From a block's portal, directed local edges reach every direct-member point of that block.
5. For an unretagged frontier group, the base materialization reaches every point in every terminal prefix-descendant group, including the requisite group-local and cross-group paths. Either a correctly materialized prefix forest or a complete LNG-cover topology can satisfy this assumption.
6. Search is exhaustive over the retained states: no capacity loss, early stopping, edge-scan caps, block-stall caps, or other pruning removes a required expansion.

Then retagging does not eliminate any eligible point from exhaustive reachability.

**Proof.** Every eligible terminal `S` has a first eligible terminal prefix ancestor `E` in `F(Q)`. If `E` remains at level 0, assumption 5 reaches all points of `S`, and level isolation preserves that base traversal.

Otherwise, let `B` be `E`'s authorized direct owner at selected layer `ell`. The root `R_B` is a canonical prefix of `E`, which is a prefix of `S`. Hence `S` lies inside `B`'s trie subtree. At that same independent layer, `S` belongs either to `B` itself or to its deepest marked descendant block `C`; it cannot be residual because it already has marked ancestor `B`. Every such `C` has root containing `R_B`, so `Q subset R_B subset-or-equal R_C`: `C` is authorized. The direct owner of `S` therefore has a retained independent portal seed. Assumption 4 reaches all points of `S` from that portal. Exhaustiveness completes the argument. QED.

The retagged part of the proof does not require overlay inter-block edges: independently seeding every authorized direct-member block is already sufficient under the strong local-reachability assumptions. It also does not require nested partitions across layers; the argument uses one selected layer for each retagged frontier entry. Multiple layers may duplicate states or increase work, which is irrelevant to this exhaustive coverage statement but highly relevant at finite capacity.

LNG-minimal entries do not provide the needed prefix-subtree decomposition. Their set-containment descendants may lie on unrelated prefix branches, as `{a,b}` does relative to `{b}` in section 7. Hence the proof cannot be substituted for a claim about LNG-minimal seeding.

Publication-useful distinction:

> Predicate certification guarantees that reached vectors satisfy the filter. Coverage additionally depends on the entry contract. Complete prefix-frontier entries admit a coverage-preserving interpretation of owner-level retagging when all authorized block portals and required graph paths are retained. LNG-minimal entries can depend on non-prefix base paths that retagging removes. Neither statement guarantees Recall or speed for the bounded approximate search.

## 9. Deliverables and limits

Written: this report and the bounded enumeration scratch script in `/tmp`. Read: the current local paper sections and the supplied remote repository's partition/seeding/level-transition source. No manuscript or production-code changes, no vector benchmarks, no child agents, and no access to the excluded repository.
