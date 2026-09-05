#!/usr/bin/env bash
set -euo pipefail

# QPS-Recall 绘图配置脚本。
# 只需要修改同目录下的 plot_qps_recall_config.json；本脚本会从 JSON 读取配置。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/plot_qps_recall_config.json}"

read_json() {
  local expr="$1"
  python3 - "$CONFIG_PATH" "$expr" <<'PYJSON'
import json
import sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
cur = cfg
for part in sys.argv[2].split('.'):
    cur = cur[part]
if isinstance(cur, bool):
    print('true' if cur else 'false')
elif cur is None:
    print('')
else:
    print(cur)
PYJSON
}

read_json_optional() {
  local expr="$1"
  python3 - "$CONFIG_PATH" "$expr" <<'PYJSON'
import json
import sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
cur = cfg
for part in sys.argv[2].split('.'):
    if not isinstance(cur, dict) or part not in cur:
        print('')
        raise SystemExit(0)
    cur = cur[part]
if isinstance(cur, bool):
    print('true' if cur else 'false')
elif cur is None:
    print('')
else:
    print(cur)
PYJSON
}

SCRIPT="$(read_json script)"
DATASET="$(read_json dataset)"
RESULTS_ROOT="$(read_json paths.results_root)"
DATA_ROOT="$(read_json paths.data_root)"
OUTPUT_DIR="$(read_json paths.output_dir)"
SORT_MODE="$(read_json draw_options.sort_mode)"
LEGEND_OUTSIDE="$(read_json draw_options.legend_outside)"
SKIP_MISSING="$(read_json draw_options.skip_missing)"
WRITE_LOG_SCALE="$(read_json draw_options.write_log_scale_svg)"
THRESHOLD_STEP="$(read_json draw_options.threshold_step)"
# Emit one pipe-separated record per query task.  `query_tasks` may contain
# strings or objects; the latter can override result_task/lsearch for one task.
while IFS='|' read -r QUERY_TASK RESULT_TASK LSEARCH_START LSEARCH_STEP LSEARCH_END; do
  [[ -n "$QUERY_TASK" ]] || continue
  RESULT_TASK="${RESULT_TASK//<query_task>/$QUERY_TASK}"
  RESULT_TASK="${RESULT_TASK//\{query_task\}/$QUERY_TASK}"
  if [[ -z "$RESULT_TASK" && -n "$LSEARCH_START" && -n "$LSEARCH_STEP" && -n "$LSEARCH_END" ]]; then
    RESULT_TASK="${QUERY_TASK}_${LSEARCH_START}_${LSEARCH_STEP}_${LSEARCH_END}"
  fi
  [[ -n "$RESULT_TASK" ]] || RESULT_TASK="$QUERY_TASK"

  EFFECTIVE_OUTPUT_DIR="$OUTPUT_DIR"
  EFFECTIVE_OUTPUT_DIR="${EFFECTIVE_OUTPUT_DIR//<query_task>/$QUERY_TASK}"
  EFFECTIVE_OUTPUT_DIR="${EFFECTIVE_OUTPUT_DIR//<result_task>/$RESULT_TASK}"
  if [[ -z "$EFFECTIVE_OUTPUT_DIR" ]]; then
    EFFECTIVE_OUTPUT_DIR="${RESULTS_ROOT}/plot/${RESULT_TASK}"
  fi

  cmd=(
    python3 "$SCRIPT"
    --results-root "$RESULTS_ROOT"
    --data-root "$DATA_ROOT"
    --dataset "$DATASET"
    --query-task "$QUERY_TASK"
    --result-task "$RESULT_TASK"
    --sort-mode "$SORT_MODE"
    --threshold-mode auto
    --threshold-step "$THRESHOLD_STEP"
  )

  if [[ -n "$OUTPUT_DIR" ]]; then
    cmd+=(--output-dir "$EFFECTIVE_OUTPUT_DIR")
  fi

  if [[ -n "$LSEARCH_START" && -n "$LSEARCH_STEP" && -n "$LSEARCH_END" ]]; then
    cmd+=(--lsearch-start "$LSEARCH_START")
    cmd+=(--lsearch-step "$LSEARCH_STEP")
    cmd+=(--lsearch-end "$LSEARCH_END")
  fi

  if [[ "$LEGEND_OUTSIDE" == "true" ]]; then
    cmd+=(--legend-outside)
  fi
  if [[ "$SKIP_MISSING" == "true" ]]; then
    cmd+=(--skip-missing)
  fi
  if [[ "$WRITE_LOG_SCALE" != "true" ]]; then
    cmd+=(--no-log)
  fi

  while IFS= read -r method_spec; do
    [[ -n "$method_spec" ]] || continue
    cmd+=(--method "$method_spec")
  done < <(
    python3 - "$CONFIG_PATH" "$QUERY_TASK" "$RESULT_TASK" "$LSEARCH_START" "$LSEARCH_STEP" "$LSEARCH_END" <<'PYJSON'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
query_task = sys.argv[2]
result_task = sys.argv[3]
task_lsearch = {"start": sys.argv[4], "step": sys.argv[5], "end": sys.argv[6]}
global_lsearch = cfg.get("lsearch") or {}
global_lsearch = {key: task_lsearch[key] or global_lsearch.get(key) for key in ("start", "step", "end")}
for method in cfg.get("methods", []):
    parts = [
        method["method_dir"],
        method["label"],
        method["color"],
        method["marker"],
    ]
    method_lsearch = method.get("lsearch")
    method_query_task = method.get("query_task")
    method_result_task = method.get("result_task")
    if method_result_task:
        method_result_task = (method_result_task
                              .replace("<query_task>", query_task)
                              .replace("{query_task}", query_task)
                              .replace("<result_task>", result_task)
                              .replace("{result_task}", result_task))
        parts.extend([method_query_task or query_task, method_result_task])
    elif method_query_task or method_lsearch:
        lsearch = method_lsearch or global_lsearch
        parts.extend(
            [
                method_query_task or query_task,
                str(lsearch["start"]),
                str(lsearch["step"]),
                str(lsearch["end"]),
            ]
        )
    print(":".join(parts))
PYJSON
  )

  while IFS= read -r method_dir; do
    [[ -n "$method_dir" ]] || continue
    cmd+=(--skip-first-lsearch-method "$method_dir")
  done < <(
    python3 - "$CONFIG_PATH" <<'PYJSON'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
for method in cfg.get("methods", []):
    if method.get("skip_first_lsearch"):
        print(method["method_dir"])
PYJSON
  )

  "${cmd[@]}"

  echo "Done. Query task: ${QUERY_TASK}"
  echo "Done. Result task: ${RESULT_TASK}"
  echo "Done. Output directory: ${EFFECTIVE_OUTPUT_DIR}"
done < <(
  python3 - "$CONFIG_PATH" <<'PYTASKS'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
tasks = cfg.get("query_tasks")
if tasks is None:
    if "query_task" not in cfg:
        raise SystemExit("config needs query_tasks or query_task")
    tasks = [{"query_task": cfg["query_task"], "result_task": cfg.get("result_task")}]
global_lsearch = cfg.get("lsearch") or {}
for task in tasks:
    if isinstance(task, str):
        task = {"query_task": task}
    if not isinstance(task, dict) or not task.get("query_task"):
        raise SystemExit("each query_tasks item must be a query-task string or object")
    lsearch = task.get("lsearch") or global_lsearch
    result_task = task.get("result_task")
    if result_task is None:
        result_task = cfg.get("result_task")
    values = [task.get("query_task"), result_task]
    values.extend(task.get(key, lsearch.get(key, "")) for key in ("start", "step", "end"))
    print("|".join("" if value is None else str(value) for value in values))
PYTASKS
)
