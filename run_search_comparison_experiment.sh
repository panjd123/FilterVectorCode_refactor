#!/usr/bin/env bash

# ./run_search_comparison_experiment.sh experiments/search_comparison/config_genome.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/experiments/search_comparison/config_genome.json}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Error: search config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

# Search configurations may intentionally select a specialized build (for
# example the CUDA/multilevel build).  Resolve that directory before sourcing
# env_ung.sh so its generic build_ung_rel default cannot select or reconfigure
# an unrelated, possibly relocated, CMake cache.
CONFIG_BUILD_DIR="$(python3 - "${CONFIG_PATH}" "${SCRIPT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
repo_root = Path(sys.argv[2]).resolve()
config = json.loads(config_path.read_text())
raw_build_dir = config.get("build_dir")
if not raw_build_dir:
    raise SystemExit(f"missing required build_dir in {config_path}")
build_dir = Path(raw_build_dir).expanduser()
if not build_dir.is_absolute():
    build_dir = repo_root / build_dir
print(build_dir.resolve())
PY
)"

export UNG_BUILD_DIR="${CONFIG_BUILD_DIR}"
source "${SCRIPT_DIR}/env_ung.sh"

SEARCH_BINARY="${UNG_BUILD_DIR}/apps/search_UNG_index"
if [[ ! -x "${SEARCH_BINARY}" ]]; then
  echo "Error: search binary not found or not executable: ${SEARCH_BINARY}" >&2
  echo "Build the configuration's UNG tree first, then rerun this command." >&2
  exit 2
fi

echo "Using configured search binary: ${SEARCH_BINARY}"

# Useful for validating a configuration and its selected build without
# starting a potentially long search sweep.
if [[ "${UNG_SEARCH_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Search preflight passed; UNG_SEARCH_PREFLIGHT_ONLY=1, not running queries."
  exit 0
fi

python3 "${SCRIPT_DIR}/experiments/search_comparison/run_search_comparison.py" "${CONFIG_PATH}"
