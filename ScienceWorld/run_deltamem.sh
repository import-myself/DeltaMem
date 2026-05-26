#!/usr/bin/env bash
# ScienceWorld 主实验：DeltaMem (PRTree 双树在线学习)
set -euo pipefail

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}"
export DEEPSEEK_BASE_URL='https://api.deepseek.com'

model_name="deepseek-v4-flash"
split="test"
benchmark="sciworld"
memory="prtree"

mkdir -p logs results trajectories storage

nohup python -u run_sciworld.py \
    --mode          eval \
    --model         "${model_name}" \
    --split         "${split}" \
    --memory        "${memory}" \
    --memory-path   "storage/${split}-${benchmark}-${model_name}-${memory}" \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-${memory}" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-${memory}.log" 2>&1 &
echo "Started DeltaMem(PRTree) PID=$!"
