#!/usr/bin/env bash
# bash scripts/run_custom_query_tasks.sh  /home/dev/graphdb/FilterVectorCode_refactor/experiments/query_task/custom_query_tasks.example.json
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$repo_root/scripts/run_custom_query_tasks.py" "$@"
