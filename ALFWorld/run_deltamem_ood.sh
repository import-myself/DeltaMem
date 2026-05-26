#!/usr/bin/env bash
# ALFWorld eval_out_of_distribution — DeltaMem PRTree
set -euo pipefail

export ALFWORLD_DATA="${ALFWORLD_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data/alfworld}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}"
export DEEPSEEK_BASE_URL='https://api.deepseek.com'

model_name="deepseek-v4-flash"
split="eval_out_of_distribution"
benchmark="alfworld"
memory="prtree"

mkdir -p logs results trajectories storage

nohup python -u example_dual_usage.py \
    --mode          eval \
    --model         "${model_name}" \
    --split         "${split}" \
    --memory        "${memory}" \
    --memory-path   "storage/${split}-${benchmark}-${model_name}-${memory}" \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-${memory}" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-${memory}.log" 2>&1 &
echo "Started ALFWorld out-of-distribution PID=$!"
