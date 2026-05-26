#!/usr/bin/env bash
# ALFWorld 基线实验：synapse / awm / reasoningbank
# 冷启动（主实验），边跑边在线更新各自记忆库
# 各方法独立 API key，独立 memory-path，进程间零污染
set -euo pipefail

export ALFWORLD_DATA="${ALFWORLD_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data/alfworld}"
export DEEPSEEK_BASE_URL='https://api.deepseek.com'

model_name="deepseek-v4-flash"
split="eval_in_distribution"
benchmark="alfworld"

mkdir -p logs results trajectories storage

# ---- 1. Synapse  (key 1) ----
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

# ---- 2. AWM  (key 2) ----
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

# ---- 3. ReasoningBank  (key 3) ----
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
