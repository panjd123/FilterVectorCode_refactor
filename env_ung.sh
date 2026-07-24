#!/usr/bin/env bash
# Source this file before running UNG tools: source ./env_ung.sh

UNG_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPHDB_ROOT="$(cd "${UNG_REPO_ROOT}/.." && pwd)"

export GRAPHDB_ROOT
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${GRAPHDB_ROOT}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${GRAPHDB_ROOT}/boost-local/lib:${GRAPHDB_ROOT}/openblas/lib:${GRAPHDB_ROOT}/zlib/lib:${UNG_REPO_ROOT}/UNG/codes/third_party/onnxruntime-linux-x64-1.16.3/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export UNG_BUILD_DIR="${UNG_BUILD_DIR:-${UNG_REPO_ROOT}/build_ung_rel}"
export UNG_DATA_ROOT="${UNG_DATA_ROOT:-${GRAPHDB_ROOT}/FilterVectorBenchData}"
export UNG_RESULTS_ROOT="${UNG_RESULTS_ROOT:-${GRAPHDB_ROOT}/FilterVectorResultsRefactor}"

echo "UNG environment ready"
echo "  repo:  ${UNG_REPO_ROOT}"
echo "  build: ${UNG_BUILD_DIR}"
