#!/usr/bin/env bash
# ALFWorld eval_in_distribution — DeltaMem PRTree
set -euo pipefail

export ALFWORLD_DATA='/hdd/REDACTED_USER/DeltaMem/ALFWorld/data/alfworld'
export DEEPSEEK_API_KEY='sk-0c763858ae194847ae13e1d85a554f9b'
export DEEPSEEK_BASE_URL='https://api.deepseek.com'

model_name="deepseek-v4-flash"
split="eval_in_distribution"
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
echo "Started ALFWorld in-distribution PID=$!"
