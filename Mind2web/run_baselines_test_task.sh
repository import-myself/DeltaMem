#!/usr/bin/env bash
# Mind2Web test_task — 4 baselines，每种方法独立 API key
set -euo pipefail

export DEEPSEEK_BASE_URL='https://api.deepseek.com'

model_name="deepseek-v4-flash"
benchmark="test_task"
split="online"

mkdir -p logs results trajectories storage

# ---- 1. no-memory ----
DEEPSEEK_API_KEY='sk-5529a7a886ee4b9bb407b614c4ead012' \
nohup python -u run.py \
    --mode          eval \
    --model         "${model_name}" \
    --benchmark     "${benchmark}" \
    --memory        no-memory \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-no-memory" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-no-memory.log" 2>&1 &
echo "Started no-memory      PID=$!"

# ---- 2. synapse ----
DEEPSEEK_API_KEY='sk-b457f919725342e282ad5900ead23542' \
nohup python -u run.py \
    --mode          eval \
    --model         "${model_name}" \
    --benchmark     "${benchmark}" \
    --memory        synapse \
    --memory-path   "storage/${split}-${benchmark}-${model_name}-synapse" \
    --save-interval 10 \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-synapse" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-synapse.log" 2>&1 &
echo "Started synapse         PID=$!"

# ---- 3. awm ----
DEEPSEEK_API_KEY='sk-fda59b026c224dcb933036d18cea9a6a' \
nohup python -u run.py \
    --mode          eval \
    --model         "${model_name}" \
    --benchmark     "${benchmark}" \
    --memory        awm \
    --memory-path   "storage/${split}-${benchmark}-${model_name}-awm" \
    --save-interval 10 \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-awm" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-awm.log" 2>&1 &
echo "Started awm             PID=$!"

# ---- 4. reasoningbank ----
DEEPSEEK_API_KEY='sk-7cca8e16e772422796c73c5ef8bdc13f' \
nohup python -u run.py \
    --mode          eval \
    --model         "${model_name}" \
    --benchmark     "${benchmark}" \
    --memory        reasoningbank \
    --memory-path   "storage/${split}-${benchmark}-${model_name}-reasoningbank" \
    --save-interval 10 \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-reasoningbank" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-reasoningbank.log" 2>&1 &
echo "Started reasoningbank   PID=$!"
