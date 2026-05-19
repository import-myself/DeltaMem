#!/usr/bin/env bash
# 实验 2.2：Train→Test 记忆迁移（run_train_to_test.py）
# phase=all → 先在 train split 积累记忆，再冻结评估 test split
set -euo pipefail

export ALFWORLD_DATA='/hdd/REDACTED_USER/DeltaMem/ALFWorld/data/alfworld'
export DEEPSEEK_API_KEY='REDACTED_API_KEY'
export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRTREE_ROOT="$(dirname "${SCRIPT_DIR}")"

model_name="deepseek-v4-flash"
icl_num=1
max_steps=30
output_csv="results/train_to_test.csv"
traj_dir="trajectories/train_to_test"

mkdir -p logs results "${traj_dir}"

# ===========================================================
# ALFWorld
# ===========================================================

for method in deltamem synapse awm reasoningbank no_memory; do
    mem_path="${PRTREE_ROOT}/ALFWorld/storage/t2t_${method}"
    log_file="logs/t2t-alfworld-${method}-${model_name}.log"
    echo "Starting ALFWorld / ${method} ..."
    nohup python -u run_train_to_test.py \
        --benchmark              alfworld \
        --method                 "${method}" \
        --phase                  all \
        --model                  "${model_name}" \
        --icl-num                "${icl_num}" \
        --max-steps              "${max_steps}" \
        --memory-path            "${mem_path}" \
        --alfworld-eval-split    eval_in_distribution \
        --output-csv             "${output_csv}" \
        --traj-dir               "${traj_dir}" \
        > "${log_file}" 2>&1
    echo "  done → ${log_file}"
done

# ===========================================================
# ScienceWorld
# ===========================================================

for method in deltamem awm reasoningbank no_memory; do
    mem_path="${PRTREE_ROOT}/ScienceWorld/storage/t2t_${method}"
    log_file="logs/t2t-sciworld-${method}-${model_name}.log"
    echo "Starting ScienceWorld / ${method} ..."
    nohup python -u run_train_to_test.py \
        --benchmark              sciworld \
        --method                 "${method}" \
        --phase                  all \
        --model                  "${model_name}" \
        --icl-num                "${icl_num}" \
        --max-steps              "${max_steps}" \
        --memory-path            "${mem_path}" \
        --sciworld-eval-split    dev \
        --output-csv             "${output_csv}" \
        --traj-dir               "${traj_dir}" \
        > "${log_file}" 2>&1
    echo "  done → ${log_file}"
done

# ===========================================================
# 选项 B：只跑 eval（已有离线记忆库，跳过 build）
# ===========================================================
# python run_train_to_test.py \
#     --benchmark   alfworld \
#     --method      deltamem \
#     --phase       eval \
#     --memory-path "${PRTREE_ROOT}/ALFWorld/storage/t2t_deltamem" \
#     --output-csv  "${output_csv}"

echo "============================================================"
echo "  Train→Test 迁移实验全部完成 → ${output_csv}"
echo "============================================================"
