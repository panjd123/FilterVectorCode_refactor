#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

out="$(make_out_dir tagore_nx256)"
groups="${NUM_GROUPS:-100}"
nx="${NX:-256}"
dim="${DIM:-128}"
topk="${TOPK:-6}"
degree="${DEGREE:-64}"

echo "[RUN] Tagore build groups=$groups nx=$nx dim=$dim -> $out"
run_with_gpu_lock perf python3 "$repo_root/scripts/benchmarks/generate_and_build_tagore_nx256.py" \
  --out "$out" --groups "$groups" --nx "$nx" --dim "$dim" > "$out/tagore_build.log" 2>&1

bin="$("$repo_root/scripts/benchmarks/build_tagore_query_benchmark.sh" "$out/tagore_index_gpu_query_bench")"

for nq in ${NQ_LIST:-1024 2560}; do
  cmd_file="$out/tagore_query_nq${nq}.cmd"
  log="$out/tagore_query_nq${nq}.log"
  csv="$out/tagore_query_nq${nq}.csv"
  python3 - "$out/tagore_nx256_manifest.csv" "$bin" "$dim" "$nq" "$topk" "$degree" > "$cmd_file" <<'PY'
import csv, shlex, sys
manifest, binary, dim, nq, topk, degree = sys.argv[1:]
cmd = [binary, dim, nq, topk, degree]
with open(manifest) as f:
    for row in csv.DictReader(f):
        cmd.extend([row["path"], row["index_path"], f'{float(row["build_ms"]):.6f}'])
print(" ".join(shlex.quote(x) for x in cmd))
PY
  echo "[RUN] Tagore query nq=$nq -> $log"
  run_with_gpu_lock perf bash -lc "$(cat "$cmd_file")" > "$log" 2>&1
  rg -n "ERROR|failed|CUDA error" "$log" || true
  rg -n "^groups,|^[0-9]+," "$log" | sed 's/^[0-9]*://' > "$csv"
done

python3 - "$out" <<'PY'
import csv, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for path in sorted(out.glob("tagore_query_nq*.csv")):
    with path.open() as f:
        rows.extend(csv.DictReader(f))
if rows:
    fields = list(rows[0].keys())
    with (out / "tagore_query_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(out / "tagore_query_summary.csv")
PY

echo "[OK] wrote $out"
