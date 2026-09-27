# Query-independent hierarchy selection protocol

## Scope

This protocol evaluates whether hierarchy depth, thresholds, and per-layer
topology can be chosen without query traces, Recall measurements, or latency
calibration. Entry discovery is fixed to `optimized_lng` for the structure
study. Entry-strategy effects are reported separately.

All performance claims use the post-refactor binary and its level-isolated
edge semantics. Results from the earlier two-level implementation are
historical context only.

## Predeclared candidate: gated DRH-v1

Degree-Ratio Hierarchy v1 uses only values already required to build the
index: point count `N`, per-layer maximum degree `R`, and per-block cross-edge
budget `C`. Its activation gate uses only the current predicate and persisted
block labels; neither hierarchy construction nor activation uses query traces,
selectivity estimates, latency measurements, or Recall calibration.

1. Set `T1` to the nearest power of two to `sqrt(N)`. This is the minimizer of
   the symmetric static proxy `T + N/T`, representing within-block and
   between-block scale.
2. Set the scale ratio to `rho = max(2, round(R/C))` and
   `T[i+1] = rho * T[i]`.
3. Materialize a scale while `N/T[i] >= C`; the first scale below that bound
   is omitted. Equivalently, absent the implementation depth cap, the number
   of layers is
   `L = max(0, 1 + floor(log_rho(N / (C * T1))))`.
   This chooses the layer count without a separate depth knob.
4. Use LNG topology when `N/T[i] > R`, otherwise Trie topology. The rule uses
   LNG while the estimated block population exceeds one graph neighborhood
   and switches to the sparse hierarchy near the top.
5. Require exact upper-layer authorization before activating the hierarchy. If
   no upper block root contains the query predicate, execute the unchanged
   zero-layer search. This gate has no fitted threshold or learned parameter.

With `N=602453`, `R=64`, and `C=4`, the structural part of DRH-v1 is fixed
before examining the new search results as `1024:lng,16384:trie`; the exact
authorization gate was also predeclared before the authoritative screen.

`T + N/T` is a static structural cost proxy, not an analytical guarantee of
query latency or Recall optimality. The held-out oracle-regret experiment is
the empirical test of whether the query-independent rule transfers.

## Static comparator

The pre-existing query-free mass ladder is retained as a comparator, not as
new evidence. It uses `T1 = nextPowerOfTwo(R * Lbuild)`,
`rho = nextPowerOfTwo(max(2, R/C))`, and stops at the first empty static Trie
partition. For Amazon its thresholds are `8192,131072`; the current grid
measures every LNG/Trie assignment at those scales.

## Calibration-free activation gate

The Amazon screen predeclared both gated and ungated variants. The development
result shows that the ungated hierarchy fails to reach Recall 0.90 at
0.5%--30% on its measured grid, whereas the gate restores crossings at
0.5%--10% with measurable overhead. Therefore the held-out automatic method is
the gated variant; the ungated DRH remains an ablation. With
`UNG_SPECIAL_REQUIRE_UPPER_AUTHORIZATION=1`, a query uses the layered search
only when its labels authorize at least one upper-layer block under exact
root-label containment; otherwise it executes the unchanged zero-layer base
search.  This decision uses the current query predicate and persisted block
labels only.  It does not use selectivity estimates, latency, Recall, or a
trained selector.

For a fair automatic-versus-manual comparison, every held-out manual hierarchy
candidate uses the same gate. Manual tuning is therefore restricted to depth,
thresholds, and per-layer topology; it does not receive a different routing
capability. When the gate selects the layered path, activation-level isolation
is unchanged: a candidate scans only edges owned by its activation level.

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
held out: gated DRH-v1 is applied from their static `N/R/C` values without
changing the rule, constants, or gate. A small declared neighborhood around
each automatic plan is built only to estimate held-out oracle regret; it
cannot change the automatic output. Every candidate in that oracle grid uses
the same exact authorization gate, while ungated DRH is reported only as an
ablation.

Search performance and instrumented profiling are separate passes. Primary
QPS comes only from the light-statistics performance pass. Build comparisons
use complete process wall time and preserve internal phase timers as mechanism
evidence.
