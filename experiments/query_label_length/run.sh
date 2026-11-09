#!/usr/bin/env bash
# Run all nine algorithms, or forward filters/validation flags to run.py.
# Examples: bash run.sh --dry-run
#           bash run.sh --datasets Amazon --methods trie_block ung
set -euo pipefail

CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${PYTHON_BIN:-python3}" "$CAMPAIGN_DIR/run.py" "$@"
