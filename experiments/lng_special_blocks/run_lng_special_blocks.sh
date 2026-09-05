#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/config.json}"
shift || true

cd "${REPO_ROOT}"
exec python3 "${SCRIPT_DIR}/run_lng_special_blocks.py" "${CONFIG_PATH}" "$@"
