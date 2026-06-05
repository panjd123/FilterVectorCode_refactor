#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

build_dir="${BUILD_DIR:-$repo_root/build_query_entry_group_bench}"
cmake -S "$repo_root/UNG/codes" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$build_dir" --target query_entry_group_bench -j"${BUILD_JOBS:-16}"
echo "$build_dir/tools/query_entry_group_bench"
