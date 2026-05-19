#!/usr/bin/env bash
# 实验 2.1：快慢双路路由消融（run_routing_ablation.py）
# 覆盖 3 个 benchmark，4 种路由模式: no_memory / skill_only / episodic_only / dual_routing
set -euo pipefail

export ALFWORLD_DATA='/hdd/REDACTED_USER/DeltaMem/ALFWorld/data/alfworld'
export DEEPSEEK_API_KEY='REDACTED_API_KEY'
export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRTREE_ROOT="$(dirname "${SCRIPT_DIR}")"

model_name="deepseek-v4-flash"
routing_modes="no_memory,skill_only,episodic_only,dual_routing"
icl_num=1
max_steps=30
output_csv="results/routing_ablation.csv"
traj_dir="trajectories/routing_ablation"

# 离线记忆路径（需要预先构建）
alfworld_memory="${PRTREE_ROOT}/ALFWorld/storage/prtree_dual_offline"
sciworld_memory="${PRTREE_ROOT}/ScienceWorld/storage/prtree_sciworld_offline"
mind2web_memory="${PRTREE_ROOT}/Mind2web/storage/prtree_mind2web_offline"

mkdir -p logs results "${traj_dir}"

# ---- ALFWorld（eval_in_distribution = 140 episodes） ----
nohup python -u run_routing_ablation.py \
    --benchmark              alfworld \
    --routing-modes          "${routing_modes}" \
    --model                  "${model_name}" \
    --icl-num                "${icl_num}" \
    --max-steps              "${max_steps}" \
    --alfworld-split         eval_in_distribution \
    --alfworld-load-memory   "${alfworld_memory}" \
    --output-csv             "${output_csv}" \
    --traj-dir               "${traj_dir}" \
    > "logs/routing_ablation-alfworld-${model_name}.log" 2>&1 &
echo "ALFWorld routing ablation started: PID=$!"

wait

# ---- ScienceWorld（dev split） ----
nohup python -u run_routing_ablation.py \
    --benchmark              sciworld \
    --routing-modes          "${routing_modes}" \
    --model                  "${model_name}" \
    --icl-num                "${icl_num}" \
    --max-steps              "${max_steps}" \
    --sciworld-split         dev \
    --sciworld-load-memory   "${sciworld_memory}" \
    --output-csv             "${output_csv}" \
    --traj-dir               "${traj_dir}" \
    > "logs/routing_ablation-sciworld-${model_name}.log" 2>&1 &
echo "ScienceWorld routing ablation started: PID=$!"

wait

# ---- Mind2Web（test_task split） ----
nohup python -u run_routing_ablation.py \
    --benchmark              mind2web \
    --routing-modes          "${routing_modes}" \
    --model                  "${model_name}" \
    --icl-num                "${icl_num}" \
    --mind2web-split         test_task \
    --mind2web-load-memory   "${mind2web_memory}" \
    --output-csv             "${output_csv}" \
    --traj-dir               "${traj_dir}" \
    > "logs/routing_ablation-mind2web-${model_name}.log" 2>&1 &
echo "Mind2Web routing ablation started: PID=$!"

wait

echo "============================================================"
echo "  路由消融全部完成 → ${output_csv}"
echo "  实时日志目录: logs/"
echo "============================================================"
