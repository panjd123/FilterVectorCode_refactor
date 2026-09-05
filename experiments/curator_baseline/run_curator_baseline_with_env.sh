#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
CONFIG_PATH="${1:-$SCRIPT_DIR/config.json}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv_curator/bin/python}"
CURATOR_REPO="${CURATOR_REPO:-$PROJECT_ROOT/thirdparty/curator-v2}"
OPENBLAS_LIB="${OPENBLAS_LIB:-/home/dev/graphdb/OpenBLAS/libopenblas.so}"
OPENBLAS_DIR="$(dirname "$OPENBLAS_LIB")"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "missing Curator Python environment: $PYTHON_BIN" >&2
  echo "run: $SCRIPT_DIR/setup_curator_env.sh" >&2
  exit 2
fi

export PYTHONPATH="$CURATOR_REPO:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [ -f "$OPENBLAS_LIB" ]; then
  export LD_LIBRARY_PATH="$OPENBLAS_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/run_curator_baseline.py" "$CONFIG_PATH"
