#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

dry_run="${DRY_RUN:-0}"
bin="${BENCH_BIN:-}"
if [[ -z "$bin" ]]; then
  if [[ "$dry_run" == "1" ]]; then
    bin="${BUILD_DIR:-$repo_root/build_query_entry_group_bench}/tools/query_entry_group_bench"
  else
    bin="$("$(dirname "$0")/build_query_entry_group_bench.sh" | tail -n 1)"
  fi
fi

out_dir="$(make_out_dir query_entry_group_sweep)"
base_label_file="${BASE_LABEL_FILE:-/home/graphdb/FilterVectorResultsRefactor/datasets/amazon/labels.txt}"
query_label_file="${QUERY_LABEL_FILE:-/home/graphdb/FilterVectorResultsRefactor/datasets/amazon/query_labels.txt}"
max_base="${MAX_BASE:-0}"
max_query="${MAX_QUERY:-2048}"
query_batch="${QUERY_BATCH:-2048}"
group_chunk="${GROUP_CHUNK:-65536}"
deltas="${FRONTIER_DELTAS:-1 2 3}"
caps="${FRONTIER_COVER_CAPS:-64 128 256 512 1024 2048 8192}"
compact_output_cap="${COMPACT_OUTPUT_CAP:-32768}"
repeats="${REPEATS:-5}"
warmup="${WARMUP:-1}"
threads="${THREADS:-128}"
gpu_id="${GPU_ID:-0}"
check="${CHECK:-1}"

summary="$out_dir/query_entry_group_sweep.csv"
raw_csv="$out_dir/raw_query_entry_group_sweep.csv"
log="$out_dir/run.log"
args=(
  --base-label-file "$base_label_file"
  --query-label-file "$query_label_file"
  --provider cover_frontier_sweep
  --max-query "$max_query"
  --query-batch "$query_batch"
  --group-chunk "$group_chunk"
  --frontier-deltas "$(printf '%s' "$deltas" | tr ' ' ',')"
  --frontier-cover-caps "$(printf '%s' "$caps" | tr ' ' ',')"
  --compact-output-cap "$compact_output_cap"
  --repeats "$repeats"
  --warmup "$warmup"
  --threads "$threads"
  --gpu "$gpu_id"
  --output-csv "$raw_csv"
)
if [[ "$max_base" != "0" ]]; then
  args+=(--max-base "$max_base")
fi
if [[ "$check" == "0" ]]; then
  args+=(--no-check)
fi

{
  echo "# date: $(date -Is)"
  echo "# bin: $bin"
  echo "# provider: cover_frontier_sweep"
  echo "# frontier_deltas: $deltas"
  echo "# frontier_cover_caps: $caps"
  echo "# compact_output_cap: $compact_output_cap"
  echo "# output_csv: $raw_csv"
} > "$log"

if [[ "$dry_run" == "1" ]]; then
  printf '[DRY_RUN]' >> "$log"
  printf ' %q' "$bin" "${args[@]}" >> "$log"
  printf '\n' >> "$log"
  {
    echo "case,provider,nq,ngroups,total_ms,kernel_ms,d2h_ms,prune_ms,avg_frontier_groups,avg_candidates,avg_min_groups,avg_output_groups,output_overflow_queries,output_truncated_groups,checksum"
    echo "dry_run,cover_frontier_sweep,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA"
  } > "$summary"
else
  run_with_gpu_lock perf "$bin" "${args[@]}" >> "$log" 2>&1
  python3 - "$raw_csv" "$summary" <<'PY'
import csv
import re
import sys
from pathlib import Path

raw = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = list(csv.DictReader(raw.open()))
fields = ["case", *rows[0].keys()]
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        provider = row["provider"]
        case = provider
        if provider.startswith("cpu_cover_frontier_ref_d"):
            m = re.match(r"cpu_cover_frontier_ref_d(\d+)_cap(\d+)", provider)
            if m:
                case = f"cpu_cover_frontier_d{m.group(1)}_cap{m.group(2)}"
        elif provider.startswith("gpu_cover_frontier_d"):
            case = provider
        elif provider.startswith("gpu_cover_frontier_compact_d"):
            case = provider
        row = {"case": case, **row}
        writer.writerow(row)
PY
fi

python3 - "$summary" "$out_dir/query_entry_group_sweep.md" <<'PY'
import csv
import math
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open()))

def f(row, key):
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return math.nan

cpu_scan = next((r for r in rows if r["case"] == "cpu_scan"), None)
cpu_exact = next((r for r in rows if r["case"] == "cpu_exact"), None)
cpu_cover_by_config = {}
for row in rows:
    case = row["case"]
    if case.startswith("cpu_cover_frontier_d") and "_cap" in case:
        tail = case.removeprefix("cpu_cover_frontier_d")
        delta, cap = tail.split("_cap", 1)
        cpu_cover_by_config[(delta, cap)] = row
scan_ms = f(cpu_scan, "total_ms") if cpu_scan else math.nan
exact_groups = f(cpu_exact, "avg_min_groups") if cpu_exact else math.nan

lines = [
    "# Query Entry Group Hyperparameter Sweep",
    "",
    "Quality here means fewer returned entry groups under the coverage-correct constraint. "
    "It is not end-to-end search quality; final graph/search quality must be checked by filtered-search recall.",
    "",
    "| case | total_ms | speedup_vs_cpu_scan | speedup_vs_cpu_cover_same_config | avg_frontier_groups | avg_output_groups | overflow_queries | truncated_groups | output_vs_exact_minimal | kernel_ms | d2h_ms | materialize_ms |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for row in rows:
    total = f(row, "total_ms")
    out_groups = f(row, "avg_min_groups")
    speed = scan_ms / total if scan_ms and total and not math.isnan(scan_ms) and not math.isnan(total) else math.nan
    ratio = out_groups / exact_groups if exact_groups and out_groups and not math.isnan(exact_groups) and not math.isnan(out_groups) else math.nan
    same_config_speed = math.nan
    case = row["case"]
    if "_d" in case and "_cap" in case:
        tail = case.split("_d", 1)[1]
        delta, cap = tail.split("_cap", 1)
        cap = cap.split("_outcap", 1)[0]
        cpu_cover = cpu_cover_by_config.get((delta, cap))
        cpu_cover_ms = f(cpu_cover, "total_ms") if cpu_cover else math.nan
        if cpu_cover_ms and total and not math.isnan(cpu_cover_ms) and not math.isnan(total):
            same_config_speed = cpu_cover_ms / total
    def fmt(x):
        return "NA" if x is None or math.isnan(x) else f"{x:.4g}"
    lines.append(
        f"| {row['case']} | {fmt(total)} | {fmt(speed)} | {fmt(same_config_speed)} | {fmt(f(row, 'avg_frontier_groups'))} | "
        f"{fmt(f(row, 'avg_output_groups'))} | {fmt(f(row, 'output_overflow_queries'))} | {fmt(f(row, 'output_truncated_groups'))} | "
        f"{fmt(ratio)} | {fmt(f(row, 'kernel_ms'))} | {fmt(f(row, 'd2h_ms'))} | {fmt(f(row, 'prune_ms'))} |"
    )

md_path.write_text("\n".join(lines) + "\n")
print(md_path)
PY

cat "$out_dir/query_entry_group_sweep.md"
echo "$out_dir"
