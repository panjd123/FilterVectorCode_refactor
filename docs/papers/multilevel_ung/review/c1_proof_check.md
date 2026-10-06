# C1 targeted proof check

Reviewed the newly revised `sections/index_query.tex`, `sections/structural_properties.tex`, and the `N/T` explanation in `sections/construction.tex` under `/tmp/filtervector-paper-18h-20261006/docs/papers/multilevel_ung`. This is a directed follow-up to the accepted structural audit, not a new cold evaluation. No manuscript or production files were changed; only this report was written.

**Verdict: the substantive additions pass.** The block-count induction, coverage counterexamples, three-group retagging example, and Proposition 4 agree with the previously verified algorithm. The text appropriately separates predicate safety, exhaustive structural coverage, and bounded-search behavior. One mathematical domain assumption should be explicit; the remaining suggestions below remove small ambiguities rather than repair a false core argument.

## Required domain clarification

**`index_query.tex:129–133`, echoed in `structural_properties.tex:27–28`: state that `T` is a nonnegative integer.**

The displayed inequalities `B(T)(T+1) <= M(T)` and `B(T) <= floor(N/(T+1))` require integer thresholds as well as integer point masses. The code uses integer thresholds, but the preceding mathematical definition and algorithm input currently say only “threshold T.” For real `T=1.5`, a single terminal of mass 2 emits a block, contradicting both the `T+1` mass bound and the displayed count bound. A minimal repair is: “For integer emission threshold T >= 0, ...”. Alternatively a real-threshold version would use `floor(T)+1`, but the integer version matches the implementation and is simpler.

Count monotonicity and the equal-count residual-set invariant themselves do not require integer thresholds. No change to their proof is needed.

## Small clarifications recommended

1. **Name the admitted-query scope of the three-group example.** At `index_query.tex:295–303` and/or the caption at `:348–352`, add “For an admitted, ungated one-overlay query ...”. The statement is already conditional on retagging, so its mathematical conclusion is correct. Making the route explicit prevents a reader from applying it to DRH-v1/v2, which reject a sole-overlay configuration without an authorized level at least 2 and then run the base search. This is a scope clarification, not a counterexample to the new claim.

2. **Include the entry's own group in Proposition 4's base reachability premise.** At `index_query.tex:360–362`, write “its own exact group and all terminal prefix descendants” or “terminal prefix descendants, including itself.” The proof at `:365–366` should similarly allow `E=S`. If “descendant” is interpreted strictly, an eligible frontier leaf with multiple points has no descendant terminals, so the stated premise would impose no reachability condition on its own remaining vectors. The intended inclusive interpretation makes the proposition sound.

3. **If the finite-budget trace is meant to follow uniquely from the listed data, specify the other neighbor distances or describe one offer.** `structural_properties.tex:76–79` correctly applies the queue ordering `(distance, vector ID, level)`: after offering `(y2,1)` at distance 0.5, the two best listed states are `(y2,1)` and `(y1,1)`, and `(y1,2)` is evicted. The text does not assign distances to every other neighbor that the same expansion could offer. “On offering y2 at distance 0.5, before any other improving offer, ...” makes it an exact intermediate trace; alternatively state that all other offered neighbors are no closer than y1. This does not affect the example's existence or its point about level-state competition.

## Verified proof details

**Partition induction (`structural_properties.tex:5–28`): correct.** By the child induction hypothesis, `d >= 0`. When `d=0`, every child count difference is zero, so the low-threshold pending set contains the high-threshold set. High-threshold emission without low-threshold emission is impossible. When final counts are equal, the two listed cases are exhaustive: `d=0,e1=e2`, or `d=1,e1=0,e2=1`. Both give residual-set inclusion. The synthetic root is correctly excluded from emission. The leaf base case is implicit in empty child sums and the same reasoning; it need not be expanded unless desired.

**Consequences (`index_query.tex:135–139`): correct with threshold direction understood.** Increasing the threshold cannot increase block count. Equal counts imply that represented coverage can only grow. When count drops, represented coverage may grow, shrink, or exchange points. Direct-member blocks need not be nested even when both layers cover every point. The text does not conflate this with monotonic authorization or search quality.

**Counterexamples (`structural_properties.tex:30–46`): all arithmetically correct.**

- `a:2,ab:1,ac:1,abc:2`: threshold 1 emits `abc:{abc}` of mass 2 and `a:{a,ab,ac}` of mass 4; threshold 2 emits `ab:{ab,abc}` of mass 3 and `a:{a,ac}` of mass 3. Both cover six points and have two blocks, while direct memberships cross.
- `a:1,ab:2,abc:3,ac:2`: thresholds 2 and 4 cover 8 and 5 points, with counts 2 and 1.
- `a:1,ab:3,ac:3`: thresholds 2 and 6 cover 6 and 7 points, with counts 2 and 1.
- `a:1,ab:3`, query `{b}`: thresholds 2 and 3 have one block each; authorized direct mass changes from 3 to 0 while global represented mass changes from 3 to 4.

**Three-group retagging figure (`index_query.tex:295–353`): consistent with the algorithm.** The singleton groups `{b}`, `{b,c}`, `{a,b}` with threshold 1 produce only the `{b}` block owning the first two. Base LNG has both outgoing edges from `{b}` shown in panel (a). The portal may be chosen as x0, as the caption expressly does. The group entry and portal then deduplicate to the same level-1 state. No level-0 seed remains in panel (b), and the crossed dashed base edge is unavailable to that level-1 state. Panel (c) retains residual `{a,b}` as a separate prefix-frontier base seed. All three vectors remain eligible; gray correctly denotes unreachability rather than ineligibility. Unlimited queue capacity cannot create an absent base seed or a cross-level transition.

**Proposition 4 (`index_query.tex:356–378`): sufficient under the stated inclusive reachability interpretation.** Each eligible terminal has a first eligible prefix-frontier ancestor. A retained base entry covers its relevant groups by hypothesis. For a retagged entry E, the authorized owner root prefixes E and therefore every prefix descendant S of E. In that same partition, S has a direct owner equal to B or a nested marked descendant, and cannot be residual. The owner's root contains the authorized root's labels, so its independently retained portal is authorized and reaches every direct-member point. This reasoning needs neither cross-layer nested membership nor overlay cross edges. It also works with either base topology if the hypothesized base paths exist. The explicit exhaustive/no-pruning condition prevents the theorem from being misread as a finite-budget Recall guarantee.

**Finite-budget partition and tie order (`structural_properties.tex:50–79`): correct.** At threshold 1 the three direct blocks have masses 2, 3, and 4; at threshold 4 the only block owns all nine points. The listed y0 group seed duplicates the level-1 a portal. The closest block states are `(y1,1)` and `(y1,2)`; level 1 breaks their exact tie first. Queue behavior agrees with `UNG/codes/include/ung_special_candidate_queue.h:24–31`, `UNG/codes/src/ung_special_candidate_queue.cpp:111–142`, and seed setup/selection at `UNG/codes/src/uni_nav_graph_search_backend.cpp:623–632,739–759` in the previously verified source snapshot. The small suggestion above concerns only unspecified additional offers.

**DRH explanation (`construction.tex:113–133`): appropriately bounded.** `N/T` is now correctly described as a loose count upper envelope, not an approximation theorem for actual count. The text expressly says that T is not a bound on block size or local search work and that query performance is empirical. The square-root minimizer is correctly attached to the stated surrogate, and no latency-optimality claim follows. Existing overflow/depth-cap qualifications remain explicit.

## Scope limits

This check read the TeX source and the relevant previously verified queue code; it did not render the figures or re-evaluate experimental tables. I found no reason to reopen the accepted mathematical investigation or run another enumeration. The final examples and theorems can remain after the small domain and wording clarifications above.
