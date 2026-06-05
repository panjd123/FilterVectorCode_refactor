#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

root="$(make_out_dir celeba_singleton_cross_edge)"
app="${UNG_BUILD_APP:-$repo_root/build_ung_test_225902/apps/build_UNG_index}"

run_case() {
  local name="$1"
  local min_nx="$2"
  local max_nx="$3"
  local out="$root/$name"
  mkdir -p "$out/index_files" "$out/results" "$out/others"
  cat > "$out/others/build_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export UNG_CROSS_EDGE_BACKEND=1
export UNG_CROSS_EDGE_GPU_STRICT=1
export UNG_COVERAGE_IMPL=1
export UNG_COVERAGE_THREADS=32
export UNG_BENCH_MIN_NX=$min_nx
export UNG_BENCH_MAX_NX=$max_nx
export UNG_NAIVE_HEAVY_SGEMM=1
export UNG_NAIVE_HEAVY_NX=256
export UNG_NAIVE_HEAVY_NQ=256
"$app" \\
  --data_type float --dist_fn L2 --num_threads 32 \\
  --max_degree 32 --Lbuild 100 --alpha 1.2 --num_cross_edges 6 \\
  --base_bin_file /home/graphdb/aaaGPU/FilterVectorData/celeba/celeba_base.bin \\
  --base_label_file /home/graphdb/aaaGPU/FilterVectorData/celeba/celeba_base_labels.txt \\
  --base_label_info_file /home/graphdb/aaaGPU/FilterVectorData/celeba/celeba_base_labels_info.log \\
  --base_label_tree_roots /home/graphdb/aaaGPU/FilterVectorData/celeba/tree_roots.txt \\
  --index_path_prefix "$out/index_files/" \\
  --result_path_prefix "$out/results/" \\
  --scenario general --dataset celeba
EOF
  chmod +x "$out/others/build_command.sh"
  echo "[RUN] $name min_nx=$min_nx max_nx=$max_nx"
  /usr/bin/time -f "wall_sec=%e" -o "$out/others/wall.time" \
    "$out/others/build_command.sh" > "$out/others/ung_build.log" 2>&1
}

run_case nx1 1 1
run_case nx_ge2 2 1048576
run_case all 0 1048576

python3 - "$root" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
fields = ["case", "index_ms", "cross_edges_ms", "wall"]
with (root / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for case in ["nx1", "nx_ge2", "all"]:
        run_dir = root / case
        rows = dict(csv.reader((run_dir / "results/build_time.csv").open()))
        writer.writerow({
            "case": case,
            "index_ms": rows.get("index_time", ""),
            "cross_edges_ms": rows.get("build_cross_edges_time", ""),
            "wall": (run_dir / "others/wall.time").read_text().strip(),
        })
print(root / "summary.csv")
PY

echo "[OK] wrote $root"
