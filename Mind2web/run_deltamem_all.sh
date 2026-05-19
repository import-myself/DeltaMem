#!/usr/bin/env bash
# Mind2web — DeltaMem PRTree，3个split各用独立API key
set -euo pipefail

export DEEPSEEK_BASE_URL='https://api.deepseek.com'

model_name="deepseek-v4-flash"
split="online"
memory="prtree"

mkdir -p logs results trajectories storage

# ---- test_task (key 5) ----
DEEPSEEK_API_KEY='sk-77cd89544f164fbf90e6660c11c0b244' \
nohup python -u run.py \
    --mode          eval \
    --model         "${model_name}" \
    --benchmark     test_task \
    --memory        "${memory}" \
    --memory-path   "storage/${split}-test_task-${model_name}-${memory}" \
    --traj-dir      "trajectories/${split}-test_task-${model_name}-${memory}" \
    --results-csv   "results/${split}-test_task-${model_name}.csv" \
    > "logs/${split}-test_task-${model_name}-${memory}.log" 2>&1 &
echo "Started test_task PID=$!"

# ---- test_website (key 6) ----
DEEPSEEK_API_KEY='sk-b8580c45f7d140608cad68690f3d9101' \
nohup python -u run.py \
    --mode          eval \
    --model         "${model_name}" \
    --benchmark     test_website \
    --memory        "${memory}" \
    --memory-path   "storage/${split}-test_website-${model_name}-${memory}" \
    --traj-dir      "trajectories/${split}-test_website-${model_name}-${memory}" \
    --results-csv   "results/${split}-test_website-${model_name}.csv" \
    > "logs/${split}-test_website-${model_name}-${memory}.log" 2>&1 &
echo "Started test_website PID=$!"

# ---- test_domain (key 7) ----
DEEPSEEK_API_KEY='sk-637985b50bfb4b4dbf7beeaed8e9fd37' \
nohup python -u run.py \
    --mode          eval \
    --model         "${model_name}" \
    --benchmark     test_domain \
    --memory        "${memory}" \
    --memory-path   "storage/${split}-test_domain-${model_name}-${memory}" \
    --traj-dir      "trajectories/${split}-test_domain-${model_name}-${memory}" \
    --results-csv   "results/${split}-test_domain-${model_name}.csv" \
    > "logs/${split}-test_domain-${model_name}-${memory}.log" 2>&1 &
echo "Started test_domain PID=$!"
