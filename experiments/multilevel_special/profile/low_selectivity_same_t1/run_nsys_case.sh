#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 <profile-prefix> <result-dir> <query-dir> <block-index> <lsearch> <single|multi>" >&2
  exit 2
fi

profile_prefix=$1
shift
export PROFILE_LIGHT_STATS=1
export PROFILE_NUM_REPEATS=${PROFILE_NUM_REPEATS:-7}

# CPU sampling and context switches identify a CPU-only search path. CUDA tracing
# remains enabled so the absence of CUDA APIs/kernels is directly testable.
exec nsys profile --force-overwrite=true --sample=cpu --cpuctxsw=process-tree \
  --trace=cuda,nvtx,osrt --stats=true --output "$profile_prefix" \
  "$(dirname "$0")/run_case.sh" "$@"
