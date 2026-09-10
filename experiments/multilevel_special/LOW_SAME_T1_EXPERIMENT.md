# Amazon x1 low-selectivity same-T1 experiment

## Question and conclusion

This experiment compares one Special Block layer at `T1=1000` with two layers
at the same fixed `T1=1000`, changing only `T2` across 4k, 10k, 25k, and
50k. A zero-layer plain UNG case is included as a reference. At the
predeclared conservative threshold `recall_min >= 0.90`, two layers do not
provide a consistent win over one layer at 0.499% or 0.903% selectivity, but
they decisively avoid the one-layer quality plateau at 9.907% selectivity.
Thus, two layers regress or are neutral at the two lowest selectivities, and do
not degenerate at 9.907%; there they are 2.46x--2.70x faster than one layer.

## Protocol

- Dataset: Amazon x1; three existing workloads with mean selectivity 0.499%,
  0.903%, and 9.907%.
- Quality: exact containment GT, `K=10`, 1000 queries. Selection is based on
  `recall_min >= 0.90` across all repeats, without interpolation.
- Timing: one immutable search binary, 100 threads, seven repeats per point;
  repeat 0 is cold and excluded, leaving six warm repeats for mean, median,
  standard deviation, and CV.
- ELS query-result reuse and CPU ELS warmup are disabled. Filter result
  validation and per-stage timing are enabled.
- Coarse guidance came from the existing low-selectivity scans, but all points
  in the table below were formally remeasured with the current binary.
- Resource isolation: `gpulock` is not installed on the host. Before each
  timing window, process inspection found no competing search/runner/profile
  process and `nvidia-smi` reported the L20 at 0% utilization, 3 MiB used, with
  no compute process. Other agents were explicitly asked not to overlap. This
  is coordinated best effort, not a claim of lock-enforced exclusivity.

## Equal-Recall results

All times are milliseconds per 1000-query batch. `speedup` uses warm means
against one-layer `T1=1000`; values below 1 mean regression.

| selectivity | method | L | recall min | warm mean | warm median | stddev | CV | speedup |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.499% | zero layer | 1375 | 0.9004 | 42.534 | 41.746 | 2.426 | 5.70% | 1.208x |
| 0.499% | one layer, T1=1k | 1750 | 0.9029 | 51.377 | 50.967 | 1.228 | 2.39% | 1.000x |
| 0.499% | two layers, T2=4k | 1750 | 0.9029 | 54.389 | 52.867 | 2.565 | 4.72% | 0.945x |
| 0.499% | two layers, T2=10k | 1750 | 0.9029 | 53.822 | 52.508 | 3.911 | 7.27% | 0.955x |
| 0.499% | two layers, T2=25k | 1750 | 0.9029 | 54.141 | 53.198 | 2.802 | 5.18% | 0.949x |
| 0.499% | two layers, T2=50k | 1750 | 0.9029 | 49.721 | 47.474 | 4.394 | 8.84% | 1.033x |
| 0.903% | zero layer | 1625 | 0.9057 | 58.918 | 57.743 | 2.847 | 4.83% | 2.661x |
| 0.903% | one layer, T1=1k | 4500 | 0.9043 | 156.780 | 154.999 | 7.877 | 5.02% | 1.000x |
| 0.903% | two layers, T2=4k | 4000 | 0.9017 | 161.437 | 157.593 | 9.256 | 5.73% | 0.971x |
| 0.903% | two layers, T2=10k | 4500 | 0.9043 | 158.774 | 156.116 | 7.510 | 4.73% | 0.987x |
| 0.903% | two layers, T2=25k | 4500 | 0.9043 | 164.571 | 168.684 | 9.412 | 5.72% | 0.953x |
| 0.903% | two layers, T2=50k | 4500 | 0.9043 | 157.961 | 157.411 | 3.306 | 2.09% | 0.993x |
| 9.907% | zero layer | 15000 | 0.9009 | 703.637 | 703.137 | 22.010 | 3.13% | 9.104x |
| 9.907% | one layer, T1=1k | 35000 | 0.9002 | 6406.040 | 6417.535 | 40.892 | 0.64% | 1.000x |
| 9.907% | two layers, T2=4k | 13500 | 0.9023 | 2604.133 | 2611.145 | 24.169 | 0.93% | 2.460x |
| 9.907% | two layers, T2=10k | 13500 | 0.9019 | 2381.910 | 2383.625 | 24.367 | 1.02% | 2.689x |
| 9.907% | two layers, T2=25k | 13500 | 0.9023 | 2374.420 | 2386.735 | 33.346 | 1.40% | 2.698x |
| 9.907% | two layers, T2=50k | 13500 | 0.9023 | 2376.902 | 2369.405 | 45.966 | 1.93% | 2.695x |

The zero-layer reference is fastest in all three workloads, so this experiment
does not establish Special Block superiority over plain UNG at low
selectivity. Its focused conclusion is about the incremental effect of adding
the second layer while fixing T1.

## Correctness and provenance

- `validate_selection_sweep.py`: `VALIDATION PASSED` for all 18
  method/workload cases and all declared L points.
- Every selected point has 70,000 checked result slots (7 repeats x 1000
  queries x K=10) and zero filter violations.
- Maximum absolute per-query stage closure error among selected points is
  `2.09548e-12 ms`; the zero-layer block-authorization stage is zero within the
  validator tolerance.
- The manifest records one search binary SHA-256 for every case. Exact query,
  GT, binary, and index-meta hashes are in `low_same_t1_hashes.csv`.

## Reproduction

From repository root:

```bash
python3 experiments/multilevel_special/run_selection_sweep.py \
  experiments/multilevel_special/config.amazon_x1_low_same_t1_formal.json
python3 experiments/multilevel_special/validate_selection_sweep.py \
  experiments/multilevel_special/config.amazon_x1_low_same_t1_formal.json
python3 experiments/multilevel_special/summarize_selection_sweep.py \
  experiments/multilevel_special/config.amazon_x1_low_same_t1_formal.json \
  --baseline layer1_t1_1000 --targets 0.90
python3 experiments/multilevel_special/summarize_low_same_t1.py \
  experiments/multilevel_special/config.amazon_x1_low_same_t1_formal.json \
  --output-dir experiments/multilevel_special/results_summary/low_same_t1
```

Raw outputs remain untracked under
`runs/low_same_t1_formal_amazon_x1_20260910`; only compact evidence is
committed.
