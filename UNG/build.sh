#!/usr/bin/env bash
set -euo pipefail

mode="${1:-cuda}"
jobs="${JOBS:-$(nproc)}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
graphdb_root="$(cd "${root}/../.." && pwd)"
source_dir="${root}/codes"

case "${mode}" in
  cuda)
    build_dir="${root}/build"
    cuda=ON
    extras=ON
    ;;
  cpu)
    build_dir="${root}/build-cpu"
    cuda=OFF
    extras=OFF
    ;;
  *)
    echo "usage: $0 [cuda|cpu]" >&2
    exit 2
    ;;
esac

cmake -S "${source_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRAPHDB_ROOT="${graphdb_root}" \
  -DBOOST_ROOT="${graphdb_root}/boost_1_89_0" \
  -DBoost_NO_SYSTEM_PATHS=ON \
  -DUNG_BOOST_USE_CXX11_ABI=OFF \
  -DUNG_ENABLE_CUDA="${cuda}" \
  -DUNG_BUILD_APPS="${extras}" \
  -DUNG_BUILD_TOOLS="${extras}" \
  -DUNG_BUILD_TESTS="${extras}"

if [[ "${mode}" == "cuda" ]]; then
  cmake --build "${build_dir}" --parallel "${jobs}"
else
  cmake --build "${build_dir}" --parallel "${jobs}" --target ANNS
fi
