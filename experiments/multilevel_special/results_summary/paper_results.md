# Generated Multi-level Special Block Results

All rows are measured points. `quality_limit` means that no scanned point reached the declared Recall threshold.

## Internal comparison

| Selectivity | Recall target | Method | Configuration | Budget | Status | Recall | Batch median (ms) | Speedup vs plain |
|---:|---:|---|---|---:|---|---:|---:|---:|
| 0.499% | 0.9000 | Plain UNG | no special overlay | L1500 | pass | 0.9134 | 44.706 | 1.000x |
| 0.499% | 0.9000 | Single-level | T1=1k | L2000 | pass | 0.9133 | 48.049 | 0.930x |
| 0.499% | 0.9000 | Original Multi-level | T1=1k,T2=10k | L2000 | pass | 0.9133 | 48.224 | 0.927x |
| 0.499% | 0.9000 | Tuned Multi-level | T1=2k,T2=25k | L1500 | pass | 0.9101 | 42.829 | 1.044x |
| 0.903% | 0.9000 | Plain UNG | no special overlay | L2000 | pass | 0.9271 | 60.656 | 1.000x |
| 0.903% | 0.9000 | Single-level | T1=1k | L4500 | pass | 0.9043 | 152.857 | 0.397x |
| 0.903% | 0.9000 | Original Multi-level | T1=1k,T2=50k | L4500 | pass | 0.9043 | 141.684 | 0.428x |
| 0.903% | 0.9000 | Tuned Multi-level | T1=2k,T2=25k | L3000 | pass | 0.9169 | 108.401 | 0.560x |
| 9.907% | 0.9000 | Plain UNG | no special overlay | L15000 | pass | 0.9016 | 675.142 | 1.000x |
| 9.907% | 0.9000 | Single-level | T1=1k | L20000 | quality_limit | 0.8948 | 3455.295 | -- |
| 9.907% | 0.9000 | Original Multi-level | T1=1k,T2=10k | L15000 | pass | 0.9088 | 2669.605 | 0.253x |
| 9.907% | 0.9000 | Tuned Multi-level | T1=2k,T2=25k | L10000 | pass | 0.9009 | 1793.005 | 0.377x |
| 24.915% | 0.9000 | Plain UNG | no special overlay | L22000 | pass | 0.9026 | 1648.470 | 1.000x |
| 24.915% | 0.9000 | Single-level | T1=1k | L18000 | pass | 0.9037 | 5828.470 | 0.283x |
| 24.915% | 0.9000 | Original Multi-level | T1=1k,T2=25k | L14000 | pass | 0.9012 | 4520.320 | 0.365x |
| 24.915% | 0.9000 | Tuned Multi-level | T1=2k,T2=25k | L10000 | pass | 0.9023 | 2986.040 | 0.552x |
| 49.971% | 0.8500 | Plain UNG | no special overlay | L40000 | pass | 0.8561 | 5551.180 | 1.000x |
| 49.971% | 0.8500 | Single-level | T1=1k | L1800 | pass | 0.8523 | 698.601 | 7.946x |
| 49.971% | 0.8500 | Original Multi-level | T1=1k,T2=25k | L600 | pass | 0.8584 | 408.748 | 13.581x |
| 49.971% | 0.8500 | Tuned Multi-level | T1=2k,T2=25k | L500 | pass | 0.8518 | 383.260 | 14.484x |
| 74.994% | 0.8700 | Plain UNG | no special overlay | L110000 | pass | 0.8737 | 44264.200 | 1.000x |
| 74.994% | 0.8700 | Single-level | T1=1k | L4000 | pass | 0.8703 | 1983.815 | 22.313x |
| 74.994% | 0.8700 | Original Multi-level | T1=1k,T2=50k | L1200 | pass | 0.8725 | 854.962 | 51.773x |
| 74.994% | 0.8700 | Tuned Multi-level | T1=2k,T2=25k | L1000 | pass | 0.8729 | 771.468 | 57.377x |

## External system position

| Selectivity | Recall target | Method | Configuration | Budget | Status | Recall | Total median (ms) | Core median (ms) |
|---:|---:|---|---|---:|---|---:|---:|---:|
| 0.499% | 0.9000 | Plain UNG | no special overlay | L1500 | pass | 0.9134 | 44.706 | -- |
| 0.499% | 0.9000 | Tuned Multi-level | T1=2k,T2=25k | L1500 | pass | 0.9101 | 42.829 | -- |
| 0.499% | 0.9000 | FAVOR | official | L200 | pass | 0.9224 | 258.117 | -- |
| 0.499% | 0.9000 | NaviX | project_route | L10000 | quality_limit | 0.7649 | 2781.490 | -- |
| 0.499% | 0.9000 | Curator | official_v2_adapter | ef512 | pass | 0.9430 | 2624.251 | -- |
| 0.499% | 0.9000 | ACORN | ACORN-1 | ef4096 | quality_limit | 0.7391 | 1188.030 | 876.904 |
| 0.903% | 0.9000 | Plain UNG | no special overlay | L2000 | pass | 0.9271 | 60.656 | -- |
| 0.903% | 0.9000 | Tuned Multi-level | T1=2k,T2=25k | L3000 | pass | 0.9169 | 108.401 | -- |
| 0.903% | 0.9000 | FAVOR | official | L1000 | pass | 0.9094 | 403.783 | -- |
| 0.903% | 0.9000 | NaviX | project_route | L20000 | quality_limit | 0.7899 | 2861.120 | -- |
| 0.903% | 0.9000 | Curator | official_v2_adapter | ef512 | pass | 0.9582 | 2886.603 | -- |
| 0.903% | 0.9000 | ACORN | ACORN-1 | ef8192 | quality_limit | 0.7907 | 1790.202 | 1556.979 |
| 9.907% | 0.9000 | Plain UNG | no special overlay | L15000 | pass | 0.9016 | 675.142 | -- |
| 9.907% | 0.9000 | Tuned Multi-level | T1=2k,T2=25k | L10000 | pass | 0.9009 | 1793.005 | -- |
| 9.907% | 0.9000 | FAVOR | official | L10000 | pass | 0.9204 | 2641.120 | -- |
| 9.907% | 0.9000 | NaviX | project_route | L2000 | pass | 0.9036 | 2848.300 | -- |
| 9.907% | 0.9000 | Curator | official_v2_adapter | ef10240 | pass | 0.9816 | 1420.208 | -- |
| 9.907% | 0.9000 | ACORN | ACORN-1 | ef16384 | quality_limit | 0.8968 | 7873.439 | 7642.445 |
| 24.915% | 0.9000 | Plain UNG | no special overlay | L22000 | pass | 0.9026 | 1648.470 | -- |
| 24.915% | 0.9000 | Tuned Multi-level | T1=2k,T2=25k | L10000 | pass | 0.9023 | 2986.040 | -- |
| 24.915% | 0.9000 | FAVOR | official | L5000 | pass | 0.9133 | 1754.680 | -- |
| 24.915% | 0.9000 | NaviX | project route | L2000 | pass | 0.9153 | 2413.740 | -- |
| 24.915% | 0.9000 | Curator | official v2 adapter | ef10240 | pass | 0.9744 | 1838.554 | -- |
| 24.915% | 0.9000 | ACORN | ACORN-1 | ef16384 | pass | 0.9011 | 8207.948 | 7943.332 |
| 49.971% | 0.8500 | Plain UNG | no special overlay | L40000 | pass | 0.8561 | 5551.180 | -- |
| 49.971% | 0.8500 | Tuned Multi-level | T1=2k,T2=25k | L500 | pass | 0.8518 | 383.260 | -- |
| 49.971% | 0.8500 | FAVOR | official | L100 | pass | 0.8644 | 200.929 | -- |
| 49.971% | 0.8500 | NaviX | project route | L100 | pass | 0.8695 | 2612.825 | -- |
| 49.971% | 0.8500 | Curator | official v2 adapter | ef10240 | pass | 0.9742 | 1590.117 | -- |
| 49.971% | 0.8500 | ACORN | ACORN-1 | ef8192 | pass | 0.8531 | 3106.804 | 2864.419 |
| 74.994% | 0.8700 | Plain UNG | no special overlay | L110000 | pass | 0.8737 | 44264.200 | -- |
| 74.994% | 0.8700 | Tuned Multi-level | T1=2k,T2=25k | L1000 | pass | 0.8729 | 771.468 | -- |
| 74.994% | 0.8700 | FAVOR | official | L200 | pass | 0.8920 | 265.841 | -- |
| 74.994% | 0.8700 | NaviX | project route | L200 | pass | 0.9179 | 2065.955 | -- |
| 74.994% | 0.8700 | Curator | official v2 adapter | ef10240 | pass | 0.9629 | 1942.862 | -- |
| 74.994% | 0.8700 | ACORN | ACORN-gamma12 | ef2048 | pass | 0.8710 | 1148.216 | 843.998 |

## Build cost

| Method | T1 | T2 | Middle blocks | Upper blocks | Builder wall (s) | Edge stage (s) | Special edges | Sidecar bytes | Loaded bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single-level | 1000 |  | 210 | 0 | 72.927 | 53.472 | 40395111 | 539055889 |  |
| Original Multi-level | 1000 | 4000 | 210 | 46 | 94.425 | 73.003 | 72733732 | 799669854 |  |
| Original Multi-level | 1000 | 10000 | 210 | 22 | 89.051 | 68.690 | 71753752 | 791828031 |  |
| Original Multi-level | 1000 | 25000 | 210 | 8 | 92.024 | 70.157 | 69343169 | 772497914 | 864059967 |
| Original Multi-level | 1000 | 50000 | 210 | 5 | 94.396 | 70.898 | 66543978 | 749982894 |  |
| T1 sensitivity | 500 | 25000 | 442 | 8 | 102.373 |  |  |  | 918423543 |
| Tuned Multi-level | 2000 | 25000 | 111 | 8 | 58.760 |  |  |  | 812811383 |
