#!/usr/bin/env bash
# ALFWorld OOD 基线实验：no-memory / synapse / awm / reasoningbank
# split: eval_out_of_distribution，各方法独立 API key
set -euo pipefail

export ALFWORLD_DATA="${ALFWORLD_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data/alfworld}"
export DEEPSEEK_BASE_URL='https://api.deepseek.com'

model_name="deepseek-v4-flash"
split="eval_out_of_distribution"
benchmark="alfworld"

mkdir -p logs results trajectories storage

# ---- 1. no-memory (key 1) ----
memory="no-memory"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}" \
nohup python -u example_dual_usage.py \
    --mode          eval \
    --model         "${model_name}" \
    --split         "${split}" \
    --memory        "${memory}" \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-${memory}" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-${memory}.log" 2>&1 &
echo "Started no-memory PID=$!"

# ---- 2. synapse (key 2) ----
memory="synapse"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}" \
nohup python -u example_dual_usage.py \
    --mode          eval \
    --model         "${model_name}" \
    --split         "${split}" \
    --memory        "${memory}" \
    --memory-path   "storage/${split}-${benchmark}-${model_name}-${memory}" \
    --save-interval 10 \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-${memory}" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-${memory}.log" 2>&1 &
echo "Started synapse PID=$!"

# ---- 3. awm (key 3) ----
memory="awm"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}" \
nohup python -u example_dual_usage.py \
    --mode          eval \
    --model         "${model_name}" \
    --split         "${split}" \
    --memory        "${memory}" \
    --memory-path   "storage/${split}-${benchmark}-${model_name}-${memory}" \
    --save-interval 10 \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-${memory}" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-${memory}.log" 2>&1 &
echo "Started awm PID=$!"

# ---- 4. reasoningbank (key 4) ----
memory="reasoningbank"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}" \
nohup python -u example_dual_usage.py \
    --mode          eval \
    --model         "${model_name}" \
    --split         "${split}" \
    --memory        "${memory}" \
    --memory-path   "storage/${split}-${benchmark}-${model_name}-${memory}" \
    --save-interval 10 \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-${memory}" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-${memory}.log" 2>&1 &
echo "Started reasoningbank PID=$!"
