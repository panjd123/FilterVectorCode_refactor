#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_ung.sh"

cmake -S "${SCRIPT_DIR}/UNG/codes" -B "${UNG_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${UNG_CUDA_ARCH:-86}" \
  -DGRAPHDB_ROOT="${GRAPHDB_ROOT}" \
  -DBOOST_ROOT="${GRAPHDB_ROOT}/boost-local" \
  -DBoost_DIR="${GRAPHDB_ROOT}/boost-local/lib/cmake/Boost-1.89.0" \
  -DZLIB_ROOT="${GRAPHDB_ROOT}/zlib" \
  -DBLA_VENDOR=OpenBLAS

cmake --build "${UNG_BUILD_DIR}" -j"${UNG_BUILD_JOBS:-16}" --target \
  build_UNG_index \
  search_UNG_index \
  fvecs_to_bin \
  compute_groundtruth \
  generate_mixed_queries

echo "UNG binaries are ready under ${UNG_BUILD_DIR}/apps and ${UNG_BUILD_DIR}/tools"
