#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

out_bin="${1:-$repo_root/build/tagore_index_gpu_query_bench}"
mkdir -p "$(dirname "$out_bin")"
nvcc -O3 -std=c++17 -arch="${CUDA_ARCH:-sm_86}" \
  "$repo_root/tools/benchmarks/tagore_index_gpu_query_bench.cu" \
  -o "$out_bin"
echo "$out_bin"
