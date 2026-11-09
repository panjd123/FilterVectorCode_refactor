#!/usr/bin/env bash
# Run both groups by default; arguments are forwarded to run.py.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${PYTHON_BIN:-python3}" "$CAMPAIGN_DIR/run.py" "$@"
