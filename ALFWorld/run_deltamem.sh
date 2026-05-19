#!/usr/bin/env bash
# ALFWorld 主实验：DeltaMem (PRTree 双树在线学习)
# 主实验 = 冷启动，online 在测试集上边跑边学习
set -euo pipefail

export ALFWORLD_DATA='/hdd/REDACTED_USER/DeltaMem/ALFWorld/data/alfworld'
export DEEPSEEK_API_KEY='REDACTED_API_KEY'
export DEEPSEEK_BASE_URL='https://api.deepseek.com'

model_name="deepseek-v4-flash"
split="eval_in_distribution"
benchmark="alfworld"
memory="prtree"

mkdir -p logs results trajectories storage

# 主实验：冷启动（不传 --resume），边评估边更新 PRTree
# 若中断后继续：加 --resume 参数，memory-path 会自动续读
nohup python -u example_dual_usage.py \
    --mode          eval \
    --model         "${model_name}" \
    --split         "${split}" \
    --memory        "${memory}" \
    --memory-path   "storage/${split}-${benchmark}-${model_name}-${memory}" \
    --traj-dir      "trajectories/${split}-${benchmark}-${model_name}-${memory}" \
    --results-csv   "results/${split}-${benchmark}-${model_name}.csv" \
    > "logs/${split}-${benchmark}-${model_name}-${memory}.log" 2>&1 &
echo "Started DeltaMem(PRTree) PID=$!"
