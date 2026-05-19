#!/usr/bin/env bash
# Mind2web 主实验：DeltaMem (PRTree 双树在线学习)
set -euo pipefail

export DEEPSEEK_API_KEY='REDACTED_API_KEY'
export DEEPSEEK_BASE_URL='https://api.deepseek.com'

model_name="deepseek-v4-flash"
benchmark="test_task"
split="online"
memory="prtree"

mkdir -p logs results trajectories storage

nohup python -u run.py \
    --mode          eval \
    --model         "${model_name}" \
    --benchmark     "${benchmark}" \
    --memory        "${memory}" \
    --memory-path   "storage/${split}-${benchmark}-${model_name}-${memory}" \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-${memory}" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-${memory}.log" 2>&1 &
echo "Started DeltaMem(PRTree) PID=$!"
