# Independent simulated VLDB review — round 2

Reviewed the frozen `/tmp/mlung-review-round2/main.pdf`, with targeted reads of its TeX sources to locate layout and wording issues. I did not consult an author resolution ledger, audit implementation code, or run experiments. Page numbers below refer to the PDF. I compared the current manuscript with the defects identified in my own round-1 review.

## Verdict and ratings

The manuscript is ready for meaningful human review of the research argument. The prose now presents an identifiable contribution, an executable search description, and results that actually interpret the measurements. Conference submission readiness remains limited by the evidence package and a substantial layout problem. My conference verdict remains reject/major revision; this does not mean another broad rewrite is needed before colleagues can review it.

| Dimension | Round 2 (1–5) | Assessment |
|---|---:|---|
| Story clarity | 4 | The central entry/navigation tradeoff now leads the contribution list and results. |
| Method clarity | 4 | The new queue semantics and execution example remove the main conceptual ambiguity. |
| Empirical support | 2 | Clearer reporting makes the existing evidence easier to assess; it does not add independent validation. |
| Overall conference submission readiness | 2 | External competitiveness, construction-quality equivalence, and stronger measurement evidence remain unresolved. |
| Writing readiness for meaningful human review | 4 / Yes | A reviewer can now follow and challenge the complete argument without reconstructing it from disconnected fragments. |

## What is fixed

- **Results interpretation:** Fixed. §8.2 explains why discovery improves in both a winning and a losing base configuration. §8.3 states the VariousImg failure, the gate's partial recovery, and the lost Reviews benefit. §8.4 distinguishes the 11.06× sidecar time ratio from the 1.35× composed ratio and explains that more GPU work can increase time. Q1–Q5 now have recognizable answers.
- **Contribution positioning:** Substantially fixed. The abstract centers the prefix frontier and certified blocks, and the third contribution is now an analysis of the entry/navigation tradeoff. GPU construction and structural configuration are supporting studies. The related-work discussion gives a concrete distinction from UNG, Curator, and geometric graph hierarchies.
- **Two-overlay execution:** Fixed. On p.5, Table 2 and the accompanying example correctly connect thresholds, ownership, retagging, duplicate removal, and queue competition. For the stated singleton groups, thresholds 1 and 4 produce the listed partitions. The retained `(y1,1)` and `(y1,2)` states are legitimately distinct; inserting `(y2,1)` at distance 0.5 evicts the coarse state under the specified tie order. This is a useful worked execution, not just a conceptual picture. It need not be made more elaborate to illustrate every possible nonnested partition.
- **Measurement presentation:** Substantially fixed. The repeated deadline/audit vocabulary is largely removed, the two-warm protocol is stated compactly, and the construction confidence-interval column has been removed. Necessary facts about separate profiling, composed stage times, and unmeasured construction-quality equivalence remain.
- **Workload description:** Substantially improved. The manuscript now states L2 distance, ascending integer-label order, singleton replacement, the empty-predicate mixture, and the five manual plans. This disclosure exposes an important interpretation issue addressed below.
- **Float placement:** Only partly fixed. Results tables now precede the conclusion, but large float-only pages still separate prose from evidence and inflate the main paper.

## Three material further writing fixes

### 1. Repair the float layout without putting all results prose on one page and all evidence afterward

**Location:** PDF pp.9–10 and 12–14. Page 9 contains Tables 3–4 and a large blank region. Page 10 contains only Table 5 near the top, leaving most of the page empty. All four results subsections occupy p.11; their tables occupy pp.12–14. Table 14's construction evidence therefore appears three pages after its interpretation. The source has `\FloatBarrier` at `sections/evaluation.tex:127` and `main.tex:58`.

**Action:** Rework float placement and table widths so that methods tables share pages with methods prose, and each results block meets its principal table within the same spread. Prefer moving the detailed metric-definition table to the appendix or reducing it to the few definitions needed in the main text, retaining essential Recall/crossing and throughput definitions in §7. Allow the large tables to float without producing two mostly empty pages. Keep discussion after the results evidence, but do not enforce that through a wholesale flush of loosely packed tables. Inspect the resulting PDF, since source order alone does not guarantee readable placement.

**Why material:** The current 24-page PDF reaches discussion/conclusion on p.15 partly because of empty float pages. This is a real reading and presentation cost, not a cosmetic preference. It is the largest remaining obstacle to presenting the manuscript cleanly.

### 2. Make batch-mean selectivity the explicit experimental axis everywhere it is used to support the headline

**Exact passages:** The abstract says “three workloads below 6% selectivity.” §8.2 says “Blocks can recover throughput on broad predicates.” But §7.3, p.8 states that the 5% batch contains individual queries ranging from 0.0033% to 96.702%, and explains that several batches are made by replacing selected predicates with the frequent singleton `{1}`.

**Action:** Change the abstract phrase to “three workloads with mean selectivity below 6%.” Label the relevant result-table column “Mean sel.” and use “higher-mean-selectivity batches” or the actual mixture description where results currently imply uniformly broad predicates. Add a short positive characterization at the beginning of §8: “The sweep varies the mixture of selective predicates and a frequent singleton, producing the reported batch means.” Interpret the throughput change as a result for those mixtures. There is no need to append a defensive warning sentence.

**Why material:** A reader currently encountering the abstract or results first can mistake the experiment for homogeneous predicates swept across selectivity. The disclosed workload construction means the experiment supports a different, still useful claim. This is a necessary definition of the measured axis, not optional caveating.

### 3. Identify the manual candidate that executes base-only search by its actual behavior in the policy results

**Exact passages:** §7.3, p.8 says “the one-overlay alternative always falls back to the base under this gate.” Table 8, p.12 nevertheless presents `256:lng` and `1024:lng` as “Best manual plan.” §8.3 describes the comparison as transfer against five manual alternatives without bringing this execution behavior into its explanation of the winner.

**Action:** Mark the one-overlay entries in Table 8 as “base fallback (one-overlay plan)” or add a short `Executed route` column. In §8.3 state positively that the best manual alternative on VariousImg executes base search, so its win demonstrates the value of rejecting the overlay there. Explain the near-Plain Genome comparison in the same terms where applicable. Preserve the configured plan identifier in parentheses or the appendix for reproducibility.

**Why material:** The present table can be read as evidence that a tuned one-overlay hierarchy beat DRH through better graph navigation. Under the stated gate, it did not execute that hierarchy. The distinction changes the interpretation of the policy experiment and can be clarified entirely from the existing measurement facts.

## Applying the requested editorial rule

The revision is materially less defensive. I would not restore the old limitation-heavy discussion or add a general “submission-strength evaluation still needs...” paragraph to the paper. Proof assumptions, the batch-mean workload definition, queue behavior, fixed gates, separate timing boundaries, and whether construction-quality equivalence was measured are necessary to interpret the argument. They should remain concise scientific facts. Sentences that merely advertise the absence of some unrelated guarantee can be deleted. The three requested fixes above strengthen definitions and interpretation rather than adding caveat paragraphs.

## Conference evidence gaps, separate from writing readiness

The current primary comparison remains an internal refactored UNG-style baseline on one development dataset. Current-protocol external comparisons are needed to assess competitiveness. Two warm repeats do not support fine distinctions among nearly tied configurations. GPU construction times need matched-quality evaluation of the resulting graphs to support a quality-preserving acceleration claim. The held-out policy evidence is largely neutral on four workloads and strongly negative on VariousImg; it supports an analyzed heuristic with uneven transfer, rather than reliable automatic tuning. The newly explained predicate mixtures also mean that robustness across homogeneous selectivities and different label distributions remains an experimental question.

These are research-evidence gaps. They do not prevent colleagues from productively reviewing the present design and its reported behavior, and they should not be converted into an endless editorial rewrite cycle.

Only `/tmp/mlung-r2-vldb.md` and temporary PDF-page renders were created. No manuscript source was edited.
