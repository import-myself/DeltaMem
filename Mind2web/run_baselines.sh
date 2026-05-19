#!/usr/bin/env bash
# Mind2web 基线实验：no-memory / synapse / awm / reasoningbank
set -euo pipefail

export DEEPSEEK_API_KEY='REDACTED_API_KEY'
export DEEPSEEK_BASE_URL='https://api.deepseek.com'

model_name="deepseek-v4-flash"
benchmark="test_task"
split="online"

mkdir -p logs results trajectories storage

# ---- 1. no-memory ----
memory="no-memory"
nohup python -u run.py \
    --mode          eval \
    --model         "${model_name}" \
    --benchmark     "${benchmark}" \
    --memory        "${memory}" \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-${memory}" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-${memory}.log" 2>&1 &
echo "Started no-memory PID=$!"

# ---- 2. Synapse ----
memory="synapse"
nohup python -u run.py \
    --mode          eval \
    --model         "${model_name}" \
    --benchmark     "${benchmark}" \
    --memory        "${memory}" \
    --memory-path   "storage/${split}-${benchmark}-${model_name}-${memory}" \
    --save-interval 10 \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-${memory}" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-${memory}.log" 2>&1 &
echo "Started synapse PID=$!"

# ---- 3. AWM ----
memory="awm"
nohup python -u run.py \
    --mode          eval \
    --model         "${model_name}" \
    --benchmark     "${benchmark}" \
    --memory        "${memory}" \
    --memory-path   "storage/${split}-${benchmark}-${model_name}-${memory}" \
    --save-interval 10 \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-${memory}" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-${memory}.log" 2>&1 &
echo "Started awm PID=$!"

# ---- 4. ReasoningBank ----
memory="reasoningbank"
nohup python -u run.py \
    --mode          eval \
    --model         "${model_name}" \
    --benchmark     "${benchmark}" \
    --memory        "${memory}" \
    --memory-path   "storage/${split}-${benchmark}-${model_name}-${memory}" \
    --save-interval 10 \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-${memory}" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-${memory}.log" 2>&1 &
echo "Started reasoningbank PID=$!"
