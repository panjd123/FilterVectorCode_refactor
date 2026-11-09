#!/usr/bin/env bash
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$CAMPAIGN_DIR/run.sh" --groups threads10 "$@"
