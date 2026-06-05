#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

out="$(make_out_dir nx256_exact)"
bin="$("$repo_root/scripts/benchmarks/build_nx256_exact_benchmark.sh" "$out/nx256_cross_edge_exact_benchmark")"

groups="${NUM_GROUPS:-100}"
nx="${NX:-256}"
dim="${DIM:-128}"
topk="${TOPK:-6}"
payloads="${PAYLOADS:-2}"
replays="${REPLAYS:-4}"

for nq in ${NQ_LIST:-1024 2560}; do
  log="$out/nx${nx}_nq${nq}.log"
  csv="$out/nx${nx}_nq${nq}.csv"
  echo "[RUN] groups=$groups nx=$nx nq=$nq dim=$dim topk=$topk -> $log"
  run_with_gpu_lock perf "$bin" "$groups" "$nx" "$nq" "$dim" "$topk" "$payloads" "$replays" > "$log" 2>&1
  rg -n "ERROR|CUDA error|cuBLAS error" "$log" || true
  rg -n "^method,|^nvidia_|^our_" "$log" | sed 's/^[0-9]*://' > "$csv"
done

python3 - "$out" <<'PY'
import csv, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for path in sorted(out.glob("nx*_nq*.csv")):
    with path.open() as f:
        rows.extend(csv.DictReader(f))
fields = ["method","groups","nx","nq_per_group","total_q","dim","topk","payloads","replays","h2d_ms","search_ms","e2e_reuse_ms","work_pairs","id_match_vs_sgemm","exact"]
with (out / "summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(out / "summary.csv")
PY

echo "[OK] wrote $out"
