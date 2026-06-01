#!/usr/bin/env bash
# ScienceWorld 实验入口（DeltaMem + 基线）
#
# 用法：
#   ./run.sh                            # DeltaMem, test split
#   ./run.sh --split dev                # DeltaMem, dev split
#   ./run.sh --memory baselines         # 全部基线, test split
#   ./run.sh --split dev --memory baselines   # 全部基线, dev split
#   ./run.sh --memory synapse           # 单个基线

set -euo pipefail

export DEEPSEEK_BASE_URL='https://api.deepseek.com'

model_name="deepseek-v4-flash"
split="test"
memory="prtree"
benchmark="sciworld"
script="run_sciworld.py"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)  split="$2";  shift 2 ;;
    --memory) memory="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p logs results trajectories storage

run_method() {
  local mem="$1"
  local need_store="${2:-true}"
  local extra_args=()
  if [[ "${need_store}" == "true" ]]; then
    extra_args+=(--memory-path "storage/${split}-${benchmark}-${model_name}-${mem}" --save-interval 10)
  fi
  DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}" \
  nohup python -u "${script}" \
    --mode        eval \
    --model       "${model_name}" \
    --split       "${split}" \
    --memory      "${mem}" \
    "${extra_args[@]}" \
    --traj-dir    "trajectories/${split}-${benchmark}-${model_name}-${mem}" \
    --results-csv "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-${mem}.log" 2>&1 &
  echo "Started ${mem} PID=$!"
}

if [[ "${memory}" == "baselines" ]]; then
  run_method "no-memory" false
  run_method "synapse"
  run_method "awm"
  run_method "reasoningbank"
else
  run_method "${memory}"
fi
