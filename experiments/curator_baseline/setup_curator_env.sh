#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
CURATOR_REPO="${1:-$PROJECT_ROOT/thirdparty/curator-v2}"
FAISS_DIR="$CURATOR_REPO/3rd_party/faiss"
JOBS="${JOBS:-${BUILD_JOBS:-$(nproc)}}"
DEFAULT_UV_PYTHON="/home/dev/graphdb/.local/share/uv/python/cpython-3.13.12-linux-x86_64-gnu/bin/python3"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv_curator}"
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/usr/bin/python3.10" ] && [ -x "/usr/bin/python3.10-config" ]; then
    BASE_PYTHON="/usr/bin/python3.10"
  elif command -v python3 >/dev/null 2>&1 && command -v python3-config >/dev/null 2>&1; then
    BASE_PYTHON="python3"
  elif [ -x "$DEFAULT_UV_PYTHON" ]; then
    BASE_PYTHON="$DEFAULT_UV_PYTHON"
  else
    BASE_PYTHON="python3"
  fi
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    "$BASE_PYTHON" -m venv "$VENV_DIR"
  fi
  PYTHON_BIN="$VENV_DIR/bin/python"
fi
OPENBLAS_LIB="${OPENBLAS_LIB:-/home/dev/graphdb/OpenBLAS/libopenblas.so}"
OPENBLAS_DIR="$(dirname "$OPENBLAS_LIB")"

if [ ! -f "$CURATOR_REPO/indexes/curator.py" ]; then
  echo "missing Curator-v2 checkout at $CURATOR_REPO" >&2
  echo "expected: $CURATOR_REPO/indexes/curator.py" >&2
  exit 2
fi

if [ ! -f "$FAISS_DIR/CMakeLists.txt" ]; then
  echo "missing Curator-v2 FAISS source at $FAISS_DIR" >&2
  exit 2
fi

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install "cmake>=3.23.1" numpy cython packaging setuptools wheel tqdm
PYTHON_BIN_DIR="$(dirname "$PYTHON_BIN")"
export PATH="$PYTHON_BIN_DIR:$PATH"
if [ -f "$OPENBLAS_LIB" ]; then
  export LD_LIBRARY_PATH="$OPENBLAS_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  BLAS_ARGS=(-DBLAS_LIBRARIES="$OPENBLAS_LIB")
else
  BLAS_ARGS=()
fi

cmake --version

cmake -S "$FAISS_DIR" -B "$FAISS_DIR/build"   -DFAISS_ENABLE_GPU=OFF   -DFAISS_ENABLE_PYTHON=ON   -DCMAKE_BUILD_TYPE=Release   -DFAISS_OPT_LEVEL=avx2   -DBUILD_TESTING=ON \
  -DPython_EXECUTABLE="$PYTHON_BIN" \
  -DSWIG_EXECUTABLE="/usr/bin/swig" \
  "${BLAS_ARGS[@]}"

cmake --build "$FAISS_DIR/build" -j"$JOBS" --target faiss_avx2 swigfaiss_avx2
(
  cd "$FAISS_DIR/build/faiss/python"
  "$PYTHON_BIN" setup.py install
)

PYTHONPATH="$CURATOR_REPO${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PYVERIFY'
import faiss
from indexes.curator import Curator
print("Curator-v2 FAISS import OK")
PYVERIFY
