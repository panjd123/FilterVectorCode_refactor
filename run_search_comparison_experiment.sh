#!/usr/bin/env bash

# ./run_search_comparison_experiment.sh experiments/search_comparison/config_genome.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/experiments/search_comparison/config_genome.json}"
"${SCRIPT_DIR}/build_ung_local.sh"

python3 "${SCRIPT_DIR}/experiments/search_comparison/run_search_comparison.py" "${CONFIG_PATH}"
