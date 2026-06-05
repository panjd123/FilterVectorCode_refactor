#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

mode="${1:-gpu}"
out="$(make_out_dir ung_cross_edge_${mode})"
app="${UNG_BUILD_APP:-$repo_root/build_ung_test_225902/apps/build_UNG_index}"

if [[ ! -x "$app" ]]; then
  echo "[ERROR] build_UNG_index not found or not executable: $app" >&2
  exit 2
fi

case "$mode" in
  cpu)
    export UNG_CROSS_EDGE_BACKEND=0
    ;;
  gpu)
    export UNG_CROSS_EDGE_BACKEND=1
    export UNG_CROSS_EDGE_GPU_STRICT="${UNG_CROSS_EDGE_GPU_STRICT:-1}"
    export UNG_NAIVE_HEAVY_SGEMM="${UNG_NAIVE_HEAVY_SGEMM:-1}"
    export UNG_NAIVE_HEAVY_NX="${UNG_NAIVE_HEAVY_NX:-256}"
    export UNG_NAIVE_HEAVY_NQ="${UNG_NAIVE_HEAVY_NQ:-256}"
    ;;
  fused)
    export UNG_CROSS_EDGE_BACKEND=1
    export UNG_CROSS_EDGE_GPU_STRICT="${UNG_CROSS_EDGE_GPU_STRICT:-1}"
    export UNG_SMALL_GROUP_FUSED="${UNG_SMALL_GROUP_FUSED:-1}"
    export UNG_MEDIUM_GROUP_FUSED="${UNG_MEDIUM_GROUP_FUSED:-1}"
    export UNG_BUCKET_GROUP_FUSED="${UNG_BUCKET_GROUP_FUSED:-1}"
    export UNG_LARGE_GROUP_FUSED="${UNG_LARGE_GROUP_FUSED:-1}"
    export UNG_LARGE_GROUP_FUSED_MODE="${UNG_LARGE_GROUP_FUSED_MODE:-2}"
    export UNG_NAIVE_HEAVY_SGEMM="${UNG_NAIVE_HEAVY_SGEMM:-1}"
    export UNG_NAIVE_HEAVY_NX="${UNG_NAIVE_HEAVY_NX:-256}"
    export UNG_NAIVE_HEAVY_NQ="${UNG_NAIVE_HEAVY_NQ:-256}"
    ;;
  *)
    echo "usage: $0 cpu|gpu|fused" >&2
    exit 1
    ;;
esac

export UNG_COVERAGE_IMPL="${UNG_COVERAGE_IMPL:-1}"
export UNG_COVERAGE_THREADS="${UNG_COVERAGE_THREADS:-32}"

mkdir -p "$out/index_files" "$out/results" "$out/others"
cat > "$out/others/build_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
$(env | rg '^UNG_' | sort | sed 's/^/export /')
"$app" \\
  --data_type float --dist_fn L2 --num_threads 32 \\
  --max_degree 32 --Lbuild 100 --alpha 1.2 --num_cross_edges 6 \\
  --base_bin_file /home/graphdb/aaaGPU/FilterVectorData/sift/sift_base.bin \\
  --base_label_file /home/graphdb/aaaGPU/FilterVectorData/sift/sift_base_30_labels_zipf_origstyle.txt \\
  --base_label_info_file /home/graphdb/aaaGPU/FilterVectorData/sift/sift_base_labels_info.log \\
  --base_label_tree_roots /home/graphdb/aaaGPU/FilterVectorData/sift/tree_roots.txt \\
  --index_path_prefix "$out/index_files/" \\
  --result_path_prefix "$out/results/" \\
  --scenario general --dataset sift30
EOF
chmod +x "$out/others/build_command.sh"

echo "[RUN] UNG cross-edge mode=$mode -> $out"
log_cmd "$out/others/ung_build.log" "$out/others/build_command.sh"
cp /tmp/ung_prof.log "$out/others/ung_prof.log" 2>/dev/null || true
rg -n "\\[PROF\\] cross_edges|large_group_fused|heavy_sgemm|failed|ERROR" "$out/others/ung_build.log" "$out/others/ung_prof.log" 2>/dev/null || true
cat "$out/results/build_time.csv"
echo "[OK] wrote $out"
