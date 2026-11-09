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
    if isinstance(cur, bool):
        print("true" if cur else "false")
    else:
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
BLOCK_BUILD_BIN="${BUILD_DIR}/apps/build_special_block_index"
OUTPUT_LAYOUT="$(json_get_default '.output_layout' 'run_root')"
INDEX_NAME_BASE="$(json_get_default '.index_name' 'UNG_special_blocks')"
INDEX_NAME_BASE_EFFECTIVE="${INDEX_NAME_BASE}$(json_index_name_suffix)"
INDEX_NAME="${INDEX_NAME_BASE_EFFECTIVE}"
RUN_ROOT="${RESULT_ROOT}/${RUN_NAME}"
BUILD_UNG_INDEX="$(json_get_default '.build_ung' 'true')"
UNG_INDEX_PATH_PREFIX="$(json_get_default '.ung_index_path_prefix' '')"
REQUIRE_UNG_INDEX="$(json_get_default '.require_ung_index' 'false')"
REQUIRED_BASE_TOPOLOGY="$(json_get_default '.required_base_topology' '')"
REUSE_EXISTING_INDEX="$(json_get_default '.reuse_existing_index' 'true')"
CUDA_ENABLED="$(json_get_default '.cuda.enabled' 'false')"
CUDA_REQUIRED="$(json_get_default '.cuda.required' 'false')"
CUDA_COMPILER="$(json_get_default '.cuda.compiler' '/usr/local/cuda/bin/nvcc')"
CUDA_ARCHITECTURES="$(json_get_default '.cuda.architectures' '')"

DATA_TYPE="$(json_get_default '.build.data_type' 'float')"
DIST_FN="$(json_get_default '.build.dist_fn' 'L2')"
SCENARIO="$(json_get_default '.build.scenario' 'general')"
NUM_THREADS="$(json_get_default '.build.num_threads' '32')"
MAX_DEGREE="$(json_get_default '.build.max_degree' '32')"
LBUILD="$(json_get_default '.build.Lbuild' '100')"
ALPHA="$(json_get_default '.build.alpha' '1.2')"
NUM_CROSS_EDGES="$(json_get_default '.build.num_cross_edges' '4')"
BUILD_SPECIAL_BLOCKS="$(json_get_default '.ung_env.UNG_SPECIAL_BLOCKS' '0')"
SPECIAL_MIN_POINTS="$(json_get_default '.ung_env.UNG_SPECIAL_BLOCK_MIN_POINTS' '100')"
SPECIAL_MIN_POINTS_VALUES_RAW="$(json_get_default '.special_block_min_points_values' '')"
SPECIAL_MIN_POINTS_SWEEP=false
SPECIAL_MIN_POINTS_VALUES=("${SPECIAL_MIN_POINTS}")
if [ -n "${SPECIAL_MIN_POINTS_VALUES_RAW}" ]; then
  SPECIAL_MIN_POINTS_SWEEP=true
  mapfile -t SPECIAL_MIN_POINTS_VALUES < <(json_array_items '.special_block_min_points_values')
  if [ "${#SPECIAL_MIN_POINTS_VALUES[@]}" -eq 0 ]; then
    echo "special_block_min_points_values must contain at least one value" >&2
    exit 2
  fi
fi
for special_min_points_value in "${SPECIAL_MIN_POINTS_VALUES[@]}"; do
  if ! [[ "${special_min_points_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid special block minimum point count: ${special_min_points_value}" >&2
    exit 2
  fi
done
SPECIAL_MAX_DEGREE="$(json_get_default '.ung_env.UNG_SPECIAL_BLOCK_MAX_DEGREE' "${MAX_DEGREE}")"
if [ "${SPECIAL_MAX_DEGREE}" = "0" ]; then
  SPECIAL_MAX_DEGREE="${MAX_DEGREE}"
fi
SPECIAL_MAX_DEGREE_VALUES_RAW="$(json_get_default '.special_block_max_degree_values' '')"
SPECIAL_MAX_DEGREE_SWEEP=false
SPECIAL_MAX_DEGREE_VALUES=("${SPECIAL_MAX_DEGREE}")
if [ -n "${SPECIAL_MAX_DEGREE_VALUES_RAW}" ]; then
  SPECIAL_MAX_DEGREE_SWEEP=true
  mapfile -t SPECIAL_MAX_DEGREE_VALUES < <(json_array_items '.special_block_max_degree_values')
  if [ "${#SPECIAL_MAX_DEGREE_VALUES[@]}" -eq 0 ]; then
    echo "special_block_max_degree_values must contain at least one value" >&2
    exit 2
  fi
fi
for special_max_degree_value in "${SPECIAL_MAX_DEGREE_VALUES[@]}"; do
  if ! [[ "${special_max_degree_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid special block maximum degree: ${special_max_degree_value}" >&2
    exit 2
  fi
done
SPECIAL_NUM_CROSS_EDGES="$(json_get_default '.ung_env.UNG_SPECIAL_BLOCK_NUM_CROSS_EDGES' "${NUM_CROSS_EDGES}")"
if [ "${SPECIAL_NUM_CROSS_EDGES}" = "0" ]; then
  SPECIAL_NUM_CROSS_EDGES="${NUM_CROSS_EDGES}"
fi
SPECIAL_NUM_CROSS_EDGES_VALUES_RAW="$(json_get_default '.special_block_num_cross_edges_values' '')"
SPECIAL_NUM_CROSS_EDGES_SWEEP=false
SPECIAL_NUM_CROSS_EDGES_VALUES=("${SPECIAL_NUM_CROSS_EDGES}")
if [ -n "${SPECIAL_NUM_CROSS_EDGES_VALUES_RAW}" ]; then
  SPECIAL_NUM_CROSS_EDGES_SWEEP=true
  mapfile -t SPECIAL_NUM_CROSS_EDGES_VALUES < <(json_array_items '.special_block_num_cross_edges_values')
  if [ "${#SPECIAL_NUM_CROSS_EDGES_VALUES[@]}" -eq 0 ]; then
    echo "special_block_num_cross_edges_values must contain at least one value" >&2
    exit 2
  fi
fi
for special_num_cross_edges_value in "${SPECIAL_NUM_CROSS_EDGES_VALUES[@]}"; do
  if ! [[ "${special_num_cross_edges_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid special block cross-edge count: ${special_num_cross_edges_value}" >&2
    exit 2
  fi
done


compile_build_ung_index() {
  log "compiling required index builders before dataset builds"
  local cmake_args=(
    -S "${SCRIPT_DIR}/UNG/codes"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    -DUNG_BUILD_APPS=ON
    -DUNG_BUILD_TOOLS=OFF
    -DUNG_BUILD_TESTS=OFF
  )
  if [ "${CUDA_ENABLED}" = "true" ] || [ "${CUDA_ENABLED}" = "1" ]; then
    if [ ! -x "${CUDA_COMPILER}" ]; then
      log "CUDA compiler is not executable: ${CUDA_COMPILER}"
      exit 2
    fi
    cmake_args+=(
      -DUNG_ENABLE_CUDA=ON
      "-DCMAKE_CUDA_COMPILER=${CUDA_COMPILER}"
    )
    if [ -n "${CUDA_ARCHITECTURES}" ]; then
      cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}")
    fi
  else
    cmake_args+=(-DUNG_ENABLE_CUDA=OFF)
  fi
  cmake "${cmake_args[@]}" || exit $?
  if [ "${CUDA_REQUIRED}" = "true" ] || [ "${CUDA_REQUIRED}" = "1" ]; then
    if ! grep -q '^UNG_ENABLE_CUDA:BOOL=ON$' "${BUILD_DIR}/CMakeCache.txt"; then
      log "CUDA was required, but CMake did not keep UNG_ENABLE_CUDA=ON"
      exit 2
    fi
  fi
  if [ "${BUILD_UNG_INDEX}" = "true" ] || [ "${BUILD_UNG_INDEX}" = "1" ]; then
    cmake --build "${BUILD_DIR}" -j"${BUILD_JOBS:-16}" --target build_UNG_index build_special_block_index search_UNG_index || exit $?
  else
    cmake --build "${BUILD_DIR}" -j"${BUILD_JOBS:-16}" --target build_special_block_index || exit $?
  fi
}

BUILDERS_COMPILED=false

ensure_index_builders_compiled() {
  if [ "${BUILDERS_COMPILED}" != "true" ]; then
    compile_build_ung_index
    BUILDERS_COMPILED=true
  fi
}

meta_value() {
  local meta_file="$1"
  local key="$2"
  awk -F= -v wanted="${key}" '$1 == wanted { print $2; exit }' "${meta_file}"
}

existing_index_complete() {
  local index_dir="$1"
  local block_index_dir="$2"
  local checked=false
  local name

  if [ "${BUILD_UNG_INDEX}" = "true" ] || [ "${BUILD_UNG_INDEX}" = "1" ]; then
    checked=true
    for name in meta vecs.bin labels.txt group_id_to_label_set group_id_to_range; do
      [ -s "${index_dir}/${name}" ] || return 1
    done
    if [ -n "${REQUIRED_BASE_TOPOLOGY}" ] && \
       [ "$(meta_value "${index_dir}/meta" base_group_topology)" != "${REQUIRED_BASE_TOPOLOGY}" ]; then
      return 1
    fi
  fi

  if [ "${BUILD_SPECIAL_BLOCKS}" = "1" ]; then
    checked=true
    for name in meta special_edges.bin special_blocks.bin special_block_trie.bin; do
      [ -s "${block_index_dir}/${name}" ] || return 1
    done
    [ "$(meta_value "${block_index_dir}/meta" special_block_min_points)" = "${SPECIAL_MIN_POINTS}" ] || return 1
    [ "$(meta_value "${block_index_dir}/meta" special_block_max_degree)" = "${SPECIAL_MAX_DEGREE}" ] || return 1
    [ "$(meta_value "${block_index_dir}/meta" special_block_num_cross_edges)" = "${SPECIAL_NUM_CROSS_EDGES}" ] || return 1
  fi

  [ "${checked}" = "true" ]
}

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
  local block_index_dir="${out_dir}/block_index_files"
  local results_dir="${out_dir}/results"
  local block_results_dir="${out_dir}/block_results"
  local others_dir="${out_dir}/others"
  local placeholder="${others_dir}/placeholder.txt"
  local base_bin="${data_dir}/${dataset}_base.bin"
  local base_labels="${data_dir}/${dataset}_base_labels.txt"
  local label_info="${data_dir}/${dataset}_base_labels_info.log"
  local tree_roots="${data_dir}/tree_roots.txt"
  local block_source_index_dir="${index_dir}"
  if [ -n "${UNG_INDEX_PATH_PREFIX}" ]; then
    block_source_index_dir="${UNG_INDEX_PATH_PREFIX//\{dataset\}/${dataset}}"
    block_source_index_dir="${block_source_index_dir%/}"
  fi

  if { [ "${REUSE_EXISTING_INDEX}" = "true" ] || [ "${REUSE_EXISTING_INDEX}" = "1" ]; } && \
     existing_index_complete "${index_dir}" "${block_index_dir}"; then
    log "[${dataset}] complete index exists; skipping build: ${out_dir}"
    return 0
  fi

  mkdir -p "${index_dir}" "${block_index_dir}" "${results_dir}" "${block_results_dir}" "${others_dir}"
  : > "${placeholder}"

  if [ ! -f "${base_bin}" ] || [ ! -f "${base_labels}" ]; then
    log "[${dataset}] missing base files under ${data_dir}"
    return 2
  fi
  [ -f "${label_info}" ] || label_info="${data_dir}/log"
  [ -f "${label_info}" ] || label_info="${placeholder}"
  [ -f "${tree_roots}" ] || tree_roots="${placeholder}"

  ensure_index_builders_compiled
  export_env_from_config
  # A configured sweep overrides the scalar env value for this build only.
  export UNG_SPECIAL_BLOCK_MIN_POINTS="${SPECIAL_MIN_POINTS}"
  export UNG_SPECIAL_BLOCK_MAX_DEGREE="${SPECIAL_MAX_DEGREE}"
  unset UNG_SPECIAL_BLOCKS

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

  local exit_code=0
  if [ "${BUILD_UNG_INDEX}" = "true" ] || [ "${BUILD_UNG_INDEX}" = "1" ]; then
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
    exit_code=$?
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
      exit_code=${cmd_exit_code}
    fi
  else
    if [ -n "${UNG_INDEX_PATH_PREFIX}" ]; then
      log "[${dataset}] skipping UNG index build; block source index: ${block_source_index_dir}"
    else
      log "[${dataset}] skipping UNG index build; block builder will reuse index data or fall back to dataset files"
    fi
    printf 'wall_seconds=0\nmax_rss_kb=\nexit_code=0\n' > "${others_dir}/time.txt"
    printf 'UNG index build skipped (build_ung=false); block source index: %s\n' "${block_source_index_dir}" > "${others_dir}/build.log"
  fi

  if [ "${exit_code}" -eq 0 ] && [ "${BUILD_SPECIAL_BLOCKS}" = "1" ]; then
    if [ "${REQUIRE_UNG_INDEX}" = "true" ] || [ "${REQUIRE_UNG_INDEX}" = "1" ]; then
      local required_source_files=(
        meta vecs.bin labels.txt group_id_to_label_set group_id_to_range
      )
      local missing_source_files=()
      local source_name
      for source_name in "${required_source_files[@]}"; do
        if [ ! -s "${block_source_index_dir}/${source_name}" ]; then
          missing_source_files+=("${block_source_index_dir}/${source_name}")
        fi
      done
      if [ "${#missing_source_files[@]}" -ne 0 ]; then
        log "[${dataset}] required UNG source index is incomplete: ${missing_source_files[*]}"
        return 2
      fi
    fi
    if [ -n "${REQUIRED_BASE_TOPOLOGY}" ]; then
      local source_meta="${block_source_index_dir}/meta"
      if [ ! -s "${source_meta}" ]; then
        log "[${dataset}] cannot verify required base topology without ${source_meta}"
        return 2
      fi
      local actual_base_topology
      actual_base_topology="$(awk -F= '$1 == "base_group_topology" { print $2; exit }' "${source_meta}")"
      if [ "${actual_base_topology}" != "${REQUIRED_BASE_TOPOLOGY}" ]; then
        log "[${dataset}] source base topology mismatch: expected=${REQUIRED_BASE_TOPOLOGY} actual=${actual_base_topology:-missing}"
        return 2
      fi
    fi
    log "[${dataset}] independent Trie block build start"
    log "[${dataset}] using UNG source index ${block_source_index_dir}"
    printf '\n--- Independent Trie block index build ---\n' >> "${others_dir}/build.log"
    "${BLOCK_BUILD_BIN}" \
      --ung_index_path_prefix "${block_source_index_dir}/" \
      --base_bin_file "${base_bin}" \
      --base_label_file "${base_labels}" \
      --block_index_path_prefix "${block_index_dir}/" \
      --result_path_prefix "${block_results_dir}/" \
      --data_type "${DATA_TYPE}" \
      --dist_fn "${DIST_FN}" \
      --num_threads "${NUM_THREADS}" \
      --min_points "${SPECIAL_MIN_POINTS}" \
      --max_degree "${SPECIAL_MAX_DEGREE}" \
      --num_cross_edges "${SPECIAL_NUM_CROSS_EDGES}" \
      --Lbuild "${LBUILD}" \
      --alpha "${ALPHA}" \
      >> "${others_dir}/build.log" 2>&1
    exit_code=$?
  fi
  local end_ts
  end_ts="$(date +%s)"
  local wall_seconds=$((end_ts - start_ts))
  printf 'pipeline_wall_seconds=%s\n' "${wall_seconds}" >> "${others_dir}/time.txt"

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

overall=0
for special_min_points_value in "${SPECIAL_MIN_POINTS_VALUES[@]}"; do
  SPECIAL_MIN_POINTS="${special_min_points_value}"
  for special_max_degree_value in "${SPECIAL_MAX_DEGREE_VALUES[@]}"; do
    SPECIAL_MAX_DEGREE="${special_max_degree_value}"
    for special_num_cross_edges_value in "${SPECIAL_NUM_CROSS_EDGES_VALUES[@]}"; do
      SPECIAL_NUM_CROSS_EDGES="${special_num_cross_edges_value}"
      INDEX_NAME="${INDEX_NAME_BASE_EFFECTIVE}"
      if [ "${SPECIAL_MIN_POINTS_SWEEP}" = "true" ] || [ "${SPECIAL_MAX_DEGREE_SWEEP}" = "true" ] || [ "${SPECIAL_NUM_CROSS_EDGES_SWEEP}" = "true" ]; then
        INDEX_NAME="${INDEX_NAME}_${SPECIAL_MIN_POINTS}"
      fi
      if [ "${SPECIAL_MAX_DEGREE_SWEEP}" = "true" ] || [ "${SPECIAL_NUM_CROSS_EDGES_SWEEP}" = "true" ]; then
        INDEX_NAME="${INDEX_NAME}_M_${SPECIAL_MAX_DEGREE}"
      fi
      if [ "${SPECIAL_NUM_CROSS_EDGES_SWEEP}" = "true" ]; then
        INDEX_NAME="${INDEX_NAME}_cross_block_${SPECIAL_NUM_CROSS_EDGES}"
      fi
      if [ "${OUTPUT_LAYOUT}" = "index_by_dataset" ]; then
        log "index name: ${INDEX_NAME}"
      fi
      log "special block minimum points: ${SPECIAL_MIN_POINTS}"
      log "special block maximum degree: ${SPECIAL_MAX_DEGREE}"
      log "special block cross edges: ${SPECIAL_NUM_CROSS_EDGES}"
      while IFS= read -r dataset; do
        build_dataset "${dataset}" || overall=$?
      done < <(json_array_items '.datasets')
    done
  done
done

exit "${overall}"
