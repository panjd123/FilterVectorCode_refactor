#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/experiments/cpu_special_blocks/config.json}"

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if command -v jq >/dev/null 2>&1; then
  JSON_READER="jq"
elif command -v python3 >/dev/null 2>&1; then
  JSON_READER="python3"
else
  echo "jq or python3 is required to read ${CONFIG_PATH}" >&2
  exit 1
fi

json_get_default() {
  local expr="$1"
  local fallback="$2"
  local value
  if [ "${JSON_READER}" = "jq" ]; then
    value="$(jq -r "${expr} // empty" "${CONFIG_PATH}")"
  else
    value="$(python3 -c 'import json,sys
with open(sys.argv[1]) as f:
    data=json.load(f)
cur=data
try:
    for part in sys.argv[2].strip(".").split("."):
        if part:
            cur=cur[part]
    print(cur)
except Exception:
    pass' "${CONFIG_PATH}" "${expr}")"
  fi
  if [ -z "${value}" ] || [ "${value}" = "null" ]; then
    printf '%s' "${fallback}"
  else
    printf '%s' "${value}"
  fi
}

json_array_items() {
  local expr="$1"
  if [ "${JSON_READER}" = "jq" ]; then
    jq -r "${expr}[]" "${CONFIG_PATH}"
  else
    python3 -c 'import json,sys
with open(sys.argv[1]) as f:
    data=json.load(f)
cur=data
for part in sys.argv[2].strip(".").split("."):
    if part:
        cur=cur[part]
for item in cur:
    print(item)' "${CONFIG_PATH}" "${expr}"
  fi
}

json_env_entries() {
  if [ "${JSON_READER}" = "jq" ]; then
    jq -r '.ung_env | to_entries[] | "\(.key)=\(.value)"' "${CONFIG_PATH}"
  else
    python3 -c 'import json,sys
with open(sys.argv[1]) as f:
    data=json.load(f)
for key, value in data.get("ung_env", {}).items():
    print(f"{key}={value}")' "${CONFIG_PATH}"
  fi
}

json_index_name_suffix() {
  python3 -c 'import json,re,sys
with open(sys.argv[1]) as f:
    data=json.load(f)

def nested(root, dotted):
    cur=root
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur=cur[part]
    return cur

def clean(value):
    value=str(value).strip()
    value=value.replace(".", "p")
    value=re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")
    return value or "empty"

parts=[]
for item in data.get("index_name_params", []):
    if not isinstance(item, dict):
        continue
    source=item.get("source", "env")
    key=item.get("key")
    if not key:
        continue
    label=item.get("label", key)
    if source == "build":
        value=nested(data.get("build", {}), key)
    elif source == "env":
        value=data.get("ung_env", {}).get(key)
    else:
        value=None
    if value is None:
        continue
    parts.append(f"{clean(label)}{clean(value)}")
print(("_" + "_".join(parts)) if parts else "")' "${CONFIG_PATH}"
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

normalize_dataset_name() {
  local dataset="$1"
  if [ "${dataset}" = "VariouImg" ]; then
    printf 'VariousImg'
  else
    printf '%s' "${dataset}"
  fi
}

RUN_NAME="$(json_get_default '.run_name' 'cpu_special_blocks')_$(date +%Y%m%d_%H%M%S)"
DATA_ROOT="$(json_get_default '.data_root' '/home/dev/graphdb/FilterVectorData')"
RESULT_ROOT="$(json_get_default '.result_root' '/home/dev/graphdb/FilterVectorResult')"
BUILD_DIR="$(json_get_default '.build_dir' "${SCRIPT_DIR}/build_ung_rel")"
BUILD_BIN="${BUILD_DIR}/apps/build_UNG_index"
OUTPUT_LAYOUT="$(json_get_default '.output_layout' 'run_root')"
INDEX_NAME_BASE="$(json_get_default '.index_name' 'UNG_special_blocks')"
INDEX_NAME="${INDEX_NAME_BASE}$(json_index_name_suffix)"
RUN_ROOT="${RESULT_ROOT}/${RUN_NAME}"

DATA_TYPE="$(json_get_default '.build.data_type' 'float')"
DIST_FN="$(json_get_default '.build.dist_fn' 'L2')"
SCENARIO="$(json_get_default '.build.scenario' 'general')"
NUM_THREADS="$(json_get_default '.build.num_threads' '32')"
MAX_DEGREE="$(json_get_default '.build.max_degree' '32')"
LBUILD="$(json_get_default '.build.Lbuild' '100')"
ALPHA="$(json_get_default '.build.alpha' '1.2')"
NUM_CROSS_EDGES="$(json_get_default '.build.num_cross_edges' '4')"


if [ ! -x "${BUILD_BIN}" ]; then
  log "build_UNG_index not found at ${BUILD_BIN}; building it now"
  cmake -S "${SCRIPT_DIR}/UNG/codes" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release || exit $?
  cmake --build "${BUILD_DIR}" -j"${BUILD_JOBS:-16}" --target build_UNG_index || exit $?
fi

export_env_from_config() {
  while IFS='=' read -r key value; do
    export "${key}=${value}"
  done < <(json_env_entries)
}

build_dataset() {
  local requested_dataset="$1"
  local dataset
  dataset="$(normalize_dataset_name "${requested_dataset}")"

  local data_dir="${DATA_ROOT}/${dataset}"
  local out_dir
  if [ "${OUTPUT_LAYOUT}" = "index_by_dataset" ]; then
    out_dir="${RESULT_ROOT}/${dataset}/index/${INDEX_NAME}"
  else
    out_dir="${RUN_ROOT}/${dataset}"
  fi
  local index_dir="${out_dir}/index_files"
  local results_dir="${out_dir}/results"
  local others_dir="${out_dir}/others"
  local placeholder="${others_dir}/placeholder.txt"
  local base_bin="${data_dir}/${dataset}_base.bin"
  local base_labels="${data_dir}/${dataset}_base_labels.txt"
  local label_info="${data_dir}/${dataset}_base_labels_info.log"
  local tree_roots="${data_dir}/tree_roots.txt"

  mkdir -p "${index_dir}" "${results_dir}" "${others_dir}"
  : > "${placeholder}"

  if [ ! -f "${base_bin}" ] || [ ! -f "${base_labels}" ]; then
    log "[${dataset}] missing base files under ${data_dir}"
    return 2
  fi
  [ -f "${label_info}" ] || label_info="${data_dir}/log"
  [ -f "${label_info}" ] || label_info="${placeholder}"
  [ -f "${tree_roots}" ] || tree_roots="${placeholder}"

  export_env_from_config

  log "[${dataset}] build start"
  local start_ts
  start_ts="$(date +%s)"
  local time_bin=""
  if command -v gtime >/dev/null 2>&1; then
    time_bin="$(command -v gtime)"
  elif [ -x /usr/bin/time ]; then
    time_bin="/usr/bin/time"
  elif [ -x /bin/time ]; then
    time_bin="/bin/time"
  fi

  if [ -n "${time_bin}" ]; then
    "${time_bin}" -f 'wall_seconds=%e\nmax_rss_kb=%M\nexit_code=%x' -o "${others_dir}/time.txt" \
      "${BUILD_BIN}" \
        --data_type "${DATA_TYPE}" \
        --dist_fn "${DIST_FN}" \
        --num_threads "${NUM_THREADS}" \
        --max_degree "${MAX_DEGREE}" \
        --Lbuild "${LBUILD}" \
        --alpha "${ALPHA}" \
        --num_cross_edges "${NUM_CROSS_EDGES}" \
        --base_bin_file "${base_bin}" \
        --base_label_file "${base_labels}" \
        --base_label_info_file "${label_info}" \
        --base_label_tree_roots "${tree_roots}" \
        --index_path_prefix "${index_dir}/" \
        --result_path_prefix "${results_dir}/" \
        --scenario "${SCENARIO}" \
        --dataset "${dataset}" \
      > "${others_dir}/build.log" 2>&1
  else
    log "[${dataset}] external time command not found; using shell wall-clock timing; max_rss_kb will be empty"
    local cmd_start_ts
    cmd_start_ts="$(date +%s)"
    "${BUILD_BIN}" \
      --data_type "${DATA_TYPE}" \
      --dist_fn "${DIST_FN}" \
      --num_threads "${NUM_THREADS}" \
      --max_degree "${MAX_DEGREE}" \
      --Lbuild "${LBUILD}" \
      --alpha "${ALPHA}" \
      --num_cross_edges "${NUM_CROSS_EDGES}" \
      --base_bin_file "${base_bin}" \
      --base_label_file "${base_labels}" \
      --base_label_info_file "${label_info}" \
      --base_label_tree_roots "${tree_roots}" \
      --index_path_prefix "${index_dir}/" \
      --result_path_prefix "${results_dir}/" \
      --scenario "${SCENARIO}" \
      --dataset "${dataset}" \
      > "${others_dir}/build.log" 2>&1
    local cmd_exit_code=$?
    local cmd_end_ts
    cmd_end_ts="$(date +%s)"
    printf 'wall_seconds=%s\nmax_rss_kb=\nexit_code=%s\n' "$((cmd_end_ts - cmd_start_ts))" "${cmd_exit_code}" > "${others_dir}/time.txt"
    return_code_for_build=${cmd_exit_code}
  fi
  local exit_code=${return_code_for_build:-$?}
  unset return_code_for_build
  local end_ts
  end_ts="$(date +%s)"
  local wall_seconds=$((end_ts - start_ts))

  if [ "${exit_code}" -eq 0 ]; then
    log "[${dataset}] build done in ${wall_seconds}s"
  else
    log "[${dataset}] build failed exit=${exit_code}; see ${others_dir}/build.log"
  fi
  return "${exit_code}"
}

log "config: ${CONFIG_PATH}"
log "run root: ${RUN_ROOT}"
log "output layout: ${OUTPUT_LAYOUT}"
if [ "${OUTPUT_LAYOUT}" = "index_by_dataset" ]; then
  log "index name: ${INDEX_NAME}"
fi

overall=0
while IFS= read -r dataset; do
  build_dataset "${dataset}" || overall=$?
done < <(json_array_items '.datasets')

exit "${overall}"
