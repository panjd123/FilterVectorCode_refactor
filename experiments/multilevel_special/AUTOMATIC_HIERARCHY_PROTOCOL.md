# Query-independent hierarchy selection protocol

## Scope

This protocol evaluates whether hierarchy depth, thresholds, and per-layer
topology can be chosen without query traces, Recall measurements, or latency
calibration. Entry discovery is fixed to `optimized_lng` for the structure
study. Entry-strategy effects are reported separately.

All performance claims use the post-refactor binary and its level-isolated
edge semantics. Results from the earlier two-level implementation are
historical context only.

## Predeclared candidate: DRH-v1

Degree-Ratio Hierarchy v1 uses only values already required to build the
index: point count `N`, per-layer maximum degree `R`, and per-block cross-edge
budget `C`.

1. Set `T1` to the nearest power of two to `sqrt(N)`. This is the minimizer of
   the symmetric static proxy `T + N/T`, representing within-block and
   between-block scale.
2. Set the scale ratio to `rho = max(2, round(R/C))` and
   `T[i+1] = rho * T[i]`.
3. Materialize a scale while `N/T[i] >= C`; the first scale below that bound
   is omitted. This chooses the layer count without a separate depth knob.
4. Use LNG topology when `N/T[i] > R`, otherwise Trie topology. The rule uses
   LNG while the estimated block population exceeds one graph neighborhood
   and switches to the sparse hierarchy near the top.

With `N=602453`, `R=64`, and `C=4`, DRH-v1 is fixed before examining the new
search results as `1024:lng,16384:trie`.

## Static comparator

The pre-existing query-free mass ladder is retained as a comparator, not as
new evidence. It uses `T1 = nextPowerOfTwo(R * Lbuild)`,
`rho = nextPowerOfTwo(max(2, R/C))`, and stops at the first empty static Trie
partition. For Amazon its thresholds are `8192,131072`; the current grid
measures every LNG/Trie assignment at those scales.

## Query-independent activation control

The Amazon screen also predeclares one routed control for DRH-v1 and one for
the mass-ladder comparator.  With
`UNG_SPECIAL_REQUIRE_UPPER_AUTHORIZATION=1`, a query uses the layered search
only when its labels authorize at least one upper-layer block under exact
root-label containment; otherwise it executes the unchanged zero-layer base
search.  This decision uses the current query predicate and persisted block
labels only.  It does not use selectivity estimates, latency, Recall, or a
trained selector.

The routed controls are reported separately from the 36-case orthogonal
factorial.  When routing selects the layered path, activation-level isolation
is unchanged: a candidate scans only edges owned by its activation level.
The no-router/routed pair therefore measures the value of avoiding an overlay
on queries for which no coarse block is structurally usable.

## Oracle and metrics

The manual oracle is the fastest explicitly measured grid point at the
smallest measured `Lsearch` whose every warm repeat reaches the declared
Recall threshold. No interpolation is allowed.

Report both:

- Per-workload oracle regret: `QPS_auto / QPS_best_for_that_workload`.
- Global-configuration regret: geometric-mean QPS of the automatic plan divided
  by the best single manual plan used unchanged across all workloads.

Also report layer count, thresholds, topology sequence, index size, build time,
median QPS, 95% bootstrap interval, nodes visited, edges scanned, distance
calculations, entry-group time, entry-point setup, authorization, and graph
search time.

## Dataset discipline

Amazon supplies the complete manual grid. Genome, Reviews, and VariousImg are
held out: DRH-v1 is applied from their static `N/R/C` values without changing
the rule or constants. A small declared neighborhood around each automatic
plan is built only to estimate held-out oracle regret; it cannot change the
automatic output.

Search performance and instrumented profiling are separate passes. Primary
QPS comes only from the light-statistics performance pass. Build comparisons
use complete process wall time and preserve internal phase timers as mechanism
evidence.
