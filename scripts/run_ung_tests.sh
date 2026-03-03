#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${REPO_ROOT}/build_ung_test_225902"
BIN="${BUILD_DIR}/apps/build_UNG_index"
RESULT_ROOT="/home/graphdb/FilterVectorResultsRefactor"
DATASET="celeba"
DATA_BIN="/home/graphdb/aaaGPU/FilterVectorData/celeba/celeba_base.bin"
DATA_LABEL="/home/graphdb/aaaGPU/FilterVectorData/celeba/celeba_base_labels.txt"
DUMMY_INFO="${REPO_ROOT}/.tmp/dummy_base_labels_info.log"
DUMMY_ROOT="${REPO_ROOT}/.tmp/dummy_tree_roots.txt"

NUM_THREADS=32
MAX_DEGREE=64
LBUILD=100
ALPHA=1.2
NUM_CROSS_EDGES=6

COVERAGE_IMPL=1
COVERAGE_THREADS=32

RUNS=2
LABEL="manual"
BEFORE_REF="HEAD^"
AFTER_REF="HEAD"
SKIP_BUILD=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_ung_tests.sh <mode> [options]

Modes:
  gpu        Run one GPU backend build test.
  cpu        Run one CPU backend build test.
  ab         A/B benchmark on git refs (before vs after), GPU backend.
  compare    Compare md5 and key log stats between two run directories.

Common options:
  --label <name>                 Output dir prefix label (default: manual)
  --result-root <path>           Results root (default: /home/graphdb/FilterVectorResultsRefactor)
  --build-dir <path>             Build dir containing build_UNG_index
  --skip-build                   Do not run cmake --build
  --coverage-impl <0|1>          UNG_COVERAGE_IMPL (default: 1)
  --coverage-threads <n>         UNG_COVERAGE_THREADS (default: 32)

Index options:
  --num-threads <n>              Runtime num threads (default: 32)
  --max-degree <n>               max_degree (default: 64)
  --lbuild <n>                   Lbuild (default: 100)
  --alpha <float>                alpha (default: 1.2)
  --num-cross-edges <n>          num_cross_edges (default: 6)
  --data-bin <path>              Base bin file path
  --data-label <path>            Base label txt path
  --dataset <name>               Dataset name (default: celeba)

AB mode options:
  --runs <n>                     Runs per variant (default: 2)
  --before-ref <git_ref>         Before ref (default: HEAD^)
  --after-ref <git_ref>          After ref (default: HEAD)

Compare mode options:
  --gpu-dir <path>               GPU run output directory
  --cpu-dir <path>               CPU run output directory

Examples:
  scripts/run_ung_tests.sh gpu --label quick_gpu
  scripts/run_ung_tests.sh cpu --label quick_cpu
  scripts/run_ung_tests.sh ab --label ab_check --runs 3 --before-ref HEAD~1 --after-ref HEAD
  scripts/run_ung_tests.sh compare --gpu-dir /path/to/gpu_run --cpu-dir /path/to/cpu_run
EOF
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: required command not found: $1" >&2
    exit 1
  }
}

ensure_dummy_files() {
  mkdir -p "${REPO_ROOT}/.tmp"
  : > "${DUMMY_INFO}"
  : > "${DUMMY_ROOT}"
}

build_target() {
  if [[ "${SKIP_BUILD}" -eq 1 ]]; then
    return
  fi
  cmake --build "${BUILD_DIR}" -j "${NUM_THREADS}" --target build_UNG_index >/dev/null
}

run_build_once() {
  local backend="$1"
  local out_prefix="$2"
  local ts out_dir
  ts="$(date +%Y%m%d_%H%M%S)"
  out_dir="${RESULT_ROOT}/${out_prefix}_${ts}"
  mkdir -p "${out_dir}/index_files" "${out_dir}/results" "${out_dir}/others"

  UNG_CROSS_EDGE_BACKEND="${backend}" \
  UNG_COVERAGE_IMPL="${COVERAGE_IMPL}" \
  UNG_COVERAGE_THREADS="${COVERAGE_THREADS}" \
  "${BIN}" \
    --dataset "${DATASET}" --data_type float --dist_fn L2 --num_threads "${NUM_THREADS}" \
    --max_degree "${MAX_DEGREE}" --Lbuild "${LBUILD}" --alpha "${ALPHA}" --num_cross_edges "${NUM_CROSS_EDGES}" \
    --base_bin_file "${DATA_BIN}" \
    --base_label_file "${DATA_LABEL}" \
    --base_label_info_file "${DUMMY_INFO}" \
    --base_label_tree_roots "${DUMMY_ROOT}" \
    --index_path_prefix "${out_dir}/index_files/" \
    --result_path_prefix "${out_dir}/results/" \
    --scenario general \
    > "${out_dir}/others/ung_build.log" 2>&1

  echo "${out_dir}"
}

extract_metrics_csv_line() {
  local log="$1"
  local lng desc cov cross idx
  lng="$(rg -o "Finished building LNG in ([0-9.]+) ms" -r '$1' "${log}" | tail -n1 || true)"
  desc="$(awk '/Calculating descendants info/{f=1;next} f&&/- Finish in/{print $4; exit}' "${log}" || true)"
  cov="$(awk '/Calculating coverage ratio/{f=1;next} f&&/- Finish in/{print $4; exit}' "${log}" || true)"
  cross="$(awk '/Building cross-group edges/{f=1;next} f&&/- Finish in/{print $4; exit}' "${log}" || true)"
  idx="$(rg -o "Index time: ([0-9]+)ms" -r '$1' "${log}" | tail -n1 || true)"
  echo "${lng:-NA},${desc:-NA},${cov:-NA},${cross:-NA},${idx:-NA}"
}

print_log_summary() {
  local log="$1"
  rg -n "Finished building LNG|Calculating descendants info|Average number of descendants|LNG 边 \\(Edges\\)|Calculating coverage ratio|coverage threads|coverage impl|Building cross-group edges|\\[cross_edges\\]|\\[GPU GEMM\\]|Finish in|Index time" "${log}" || true
}

compare_runs() {
  local gpu_dir="$1"
  local cpu_dir="$2"
  local gpu_log="${gpu_dir}/others/ung_build.log"
  local cpu_log="${cpu_dir}/others/ung_build.log"
  local f

  echo "== Log Structure =="
  echo "-- GPU: ${gpu_log}"
  rg -n "Average number of descendants|LNG 边 \\(Edges\\)|target_groups=|Index time" "${gpu_log}" || true
  echo "-- CPU: ${cpu_log}"
  rg -n "Average number of descendants|LNG 边 \\(Edges\\)|target_groups=|Index time" "${cpu_log}" || true

  echo
  echo "== MD5 Compare =="
  for f in lng_descendants_rb.bin covered_sets_rb.bin vector_attr_graph lng_descendants_num lng_coverage_ratio; do
    if [[ -f "${gpu_dir}/index_files/${f}" && -f "${cpu_dir}/index_files/${f}" ]]; then
      echo "== ${f}"
      md5sum "${gpu_dir}/index_files/${f}" "${cpu_dir}/index_files/${f}"
    else
      echo "Missing file for compare: ${f}"
    fi
  done
}

run_ab() {
  need_cmd git
  need_cmd python
  local git_status current_branch current_ref csv ts variant ref ref_short i out log mline

  git_status="$(git -C "${REPO_ROOT}" status --porcelain)"
  if [[ -n "${git_status}" ]]; then
    echo "Error: working tree is not clean. Commit/stash changes before ab mode." >&2
    exit 1
  fi

  current_branch="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)"
  current_ref="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"

  ts="$(date +%Y%m%d_%H%M%S)"
  csv="${RESULT_ROOT}/ab_${LABEL}_${ts}.csv"
  echo "variant,commit,run,log_path,lng_ms,desc_ms,cov_ms,cross_ms,index_ms" > "${csv}"

  restore_checkout() {
    if [[ "${current_branch}" == "HEAD" ]]; then
      git -C "${REPO_ROOT}" checkout "${current_ref}" >/dev/null 2>&1 || true
    else
      git -C "${REPO_ROOT}" checkout "${current_branch}" >/dev/null 2>&1 || true
    fi
  }
  trap restore_checkout EXIT

  for variant in before after; do
    if [[ "${variant}" == "before" ]]; then
      ref="${BEFORE_REF}"
    else
      ref="${AFTER_REF}"
    fi

    git -C "${REPO_ROOT}" checkout "${ref}" >/dev/null 2>&1
    ref_short="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
    build_target

    for i in $(seq 1 "${RUNS}"); do
      out="$(run_build_once 1 "${LABEL}_${variant}_${ref_short}_r${i}")"
      log="${out}/others/ung_build.log"
      mline="$(extract_metrics_csv_line "${log}")"
      echo "${variant},${ref_short},${i},${log},${mline}" >> "${csv}"
      echo "[${variant} r${i}] ${out}"
    done
  done

  python - <<PY
import csv,statistics
p="${csv}"
rows=list(csv.DictReader(open(p)))
metrics=['lng_ms','desc_ms','cov_ms','cross_ms','index_ms']
for v in ['before','after']:
    part=[r for r in rows if r['variant']==v]
    print(v)
    for m in metrics:
        vals=[float(r[m]) for r in part]
        print(f"  {m}: median={statistics.median(vals):.3f}, vals={vals}")
    print()
if {'before','after'}.issubset(set(r['variant'] for r in rows)):
    b=[r for r in rows if r['variant']=='before']
    a=[r for r in rows if r['variant']=='after']
    print("delta(after-before):")
    for m in metrics:
        mb=statistics.median([float(r[m]) for r in b])
        ma=statistics.median([float(r[m]) for r in a])
        pct=(ma/mb-1.0)*100.0
        print(f"  {m}: {ma-mb:.3f} ({pct:+.2f}%)")
print()
print("csv:", p)
PY
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

MODE="$1"
shift

GPU_DIR=""
CPU_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --result-root) RESULT_ROOT="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; BIN="${BUILD_DIR}/apps/build_UNG_index"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --data-bin) DATA_BIN="$2"; shift 2 ;;
    --data-label) DATA_LABEL="$2"; shift 2 ;;
    --num-threads) NUM_THREADS="$2"; shift 2 ;;
    --max-degree) MAX_DEGREE="$2"; shift 2 ;;
    --lbuild) LBUILD="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --num-cross-edges) NUM_CROSS_EDGES="$2"; shift 2 ;;
    --coverage-impl) COVERAGE_IMPL="$2"; shift 2 ;;
    --coverage-threads) COVERAGE_THREADS="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --before-ref) BEFORE_REF="$2"; shift 2 ;;
    --after-ref) AFTER_REF="$2"; shift 2 ;;
    --gpu-dir) GPU_DIR="$2"; shift 2 ;;
    --cpu-dir) CPU_DIR="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

need_cmd rg
ensure_dummy_files
mkdir -p "${RESULT_ROOT}"

case "${MODE}" in
  gpu)
    build_target
    out="$(run_build_once 1 "${LABEL}_gpu")"
    echo "output: ${out}"
    print_log_summary "${out}/others/ung_build.log"
    ;;
  cpu)
    build_target
    out="$(run_build_once 0 "${LABEL}_cpu")"
    echo "output: ${out}"
    print_log_summary "${out}/others/ung_build.log"
    ;;
  ab)
    run_ab
    ;;
  compare)
    if [[ -z "${GPU_DIR}" || -z "${CPU_DIR}" ]]; then
      echo "Error: compare mode requires --gpu-dir and --cpu-dir" >&2
      exit 1
    fi
    compare_runs "${GPU_DIR}" "${CPU_DIR}"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac
