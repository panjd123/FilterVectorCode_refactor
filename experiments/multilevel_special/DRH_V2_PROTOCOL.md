# DRH-v2 Next-Scale Mass Gate Protocol

Status: frozen before the first DRH-v2 measurement.

## Purpose

DRH-v1 derives hierarchy depth, thresholds, and layer topology from index-only
statistics, then uses an exact predicate-authorization gate. Held-out screening
showed that exact authorization is necessary but not sufficient: an authorized
upper overlay can still increase bounded-search work. DRH-v2 tests one
calibration-free refinement without changing the hierarchy itself.

## Frozen rule

Let the graph degree budget be `R`, the cross-edge budget be `C`, and
`rho = R / C`. DRH materializes thresholds

```text
T_1, T_2 = rho T_1, ..., T_L = rho T_{L-1}.
```

The next unmaterialized scale is

```text
T_next = rho T_L.
```

For a containment query `q`, let `A_h(q)` be the blocks at the highest
authorized upper layer `h` whose root-label set contains `q`. Let `n_b` be the
number of direct member points of block `b`. Direct members partition a layer,
so

```text
M_h(q) = sum_{b in A_h(q)} n_b
```

does not double-count descendants. DRH-v2 executes the multilevel backend iff

```text
A_h(q) is nonempty and M_h(q) >= T_next.
```

Otherwise it executes the zero-layer baseline backend. The decision occurs
before multilevel candidate initialization. It therefore does not retain the
middle-layer search cost on a fallback query.

For the frozen two-layer indexes (`R=64`, `C=4`, `rho=16`):

| Dataset | DRH hierarchy | `T_next` |
|---|---|---:|
| Amazon | `1024:LNG,16384:Trie` | 262144 |
| Genome | `256:LNG,4096:Trie` | 65536 |
| Reviews | `512:LNG,8192:Trie` | 131072 |
| VariousImg | `1024:LNG,16384:Trie` | 262144 |

## Information boundary

The rule may read only persisted block metadata and the current query labels.
It does not read a query-frequency distribution, sampled latency, Recall,
ground truth, selectivity labels, or a learned model. `T_next` is derived from
the same `R/C` recurrence that generated the materialized hierarchy; it is not
an additional tuned constant.

## Fixed ablation

Each dataset is tested with one newly compiled immutable search binary and
three methods:

1. zero-layer LNG with optimized-LNG entry discovery;
2. DRH-v1 with exact nonempty-upper authorization;
3. DRH-v2 with exact authorization plus the next-scale direct-mass gate.

Amazon uses the nine frozen workloads near 0.5%, 1%, 5%, 10%, 30%, 60%, 80%,
95%, and 99% selectivity. Genome, Reviews, and VariousImg retain the held-out
workloads selected before DRH-v2 was introduced. Every case uses one discarded
cold repeat and two warm repeats over the same frozen query file. A Recall
crossing is the minimum measured `Lsearch` at which every warm repeat reaches
Recall@10 >= 0.90. DRH-v2 receives the union of the already frozen plain and
DRH-v1 `Lsearch` grids so fallback is not penalized by a coarser grid.

Primary QPS uses the median warm full-batch time. Stage timings are collected
in the performance pass; expensive edge counters are reported only from a
separate profile pass. No result is discarded because it is slower, fails to
cross the Recall threshold, or contradicts the hypothesis.

## Interpretation boundary

`M_h(q) >= T_next` is a structural eligibility proxy, not a proof of speedup.
The ablation can establish whether it repairs observed low/mid-selectivity
regressions while retaining high-selectivity gains on these datasets. It cannot
establish a universal monotonicity theorem, and the present datasets do not
exercise a DRH depth other than two.
