#!/usr/bin/env bash
# ScienceWorld test split — 4 baselines，每种方法独立 API key
set -euo pipefail

export DEEPSEEK_BASE_URL='https://api.deepseek.com'

model_name="deepseek-v4-flash"
split="test"
benchmark="sciworld"

mkdir -p logs results trajectories storage

# ---- 1. no-memory ----
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}" \
nohup python -u run_sciworld.py \
    --mode          eval \
    --model         "${model_name}" \
    --split         "${split}" \
    --memory        no-memory \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-no-memory" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-no-memory.log" 2>&1 &
echo "Started no-memory      PID=$!"

# ---- 2. synapse ----
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}" \
nohup python -u run_sciworld.py \
    --mode          eval \
    --model         "${model_name}" \
    --split         "${split}" \
    --memory        synapse \
    --memory-path   "storage/${split}-${benchmark}-${model_name}-synapse" \
    --save-interval 10 \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-synapse" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-synapse.log" 2>&1 &
echo "Started synapse         PID=$!"

# ---- 3. awm ----
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}" \
nohup python -u run_sciworld.py \
    --mode          eval \
    --model         "${model_name}" \
    --split         "${split}" \
    --memory        awm \
    --memory-path   "storage/${split}-${benchmark}-${model_name}-awm" \
    --save-interval 10 \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-awm" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-awm.log" 2>&1 &
echo "Started awm             PID=$!"

# ---- 4. reasoningbank ----
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}" \
nohup python -u run_sciworld.py \
    --mode          eval \
    --model         "${model_name}" \
    --split         "${split}" \
    --memory        reasoningbank \
    --memory-path   "storage/${split}-${benchmark}-${model_name}-reasoningbank" \
    --save-interval 10 \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-reasoningbank" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-reasoningbank.log" 2>&1 &
echo "Started reasoningbank   PID=$!"
