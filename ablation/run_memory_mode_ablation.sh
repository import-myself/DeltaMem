export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}"
export DEEPSEEK_BASE_URL='https://api.deepseek.com'
export ALFWORLD_DATA="${ALFWORLD_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../ALFWorld/data/alfworld}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRTREE_ROOT="$(dirname "${SCRIPT_DIR}")"

model_name="deepseek-v4-flash"
memory_modes="task_only,env_only"
max_steps=30
icl_num=1
output_csv="results/memory_mode_ablation.csv"
traj_dir="trajectories/memory_mode_ablation"

mkdir -p logs results "${traj_dir}"

# ---- ALFWorld（全量 eval_in_distribution = 140） ----
nohup python -u run_memory_mode_ablation.py \
    --benchmark            alfworld \
    --memory-modes         "${memory_modes}" \
    --model                "${model_name}" \
    --icl-num              "${icl_num}" \
    --max-steps            "${max_steps}" \
    --output-csv           "${output_csv}" \
    --traj-dir             "${traj_dir}" \
    --alfworld-split       eval_in_distribution \
    --alfworld-load-memory "${PRTREE_ROOT}/ALFWorld/storage/prtree_dual_offline" \
    > "logs/memory_mode-alfworld-${model_name}.log" 2>&1
echo "ALFWorld memory_mode ablation done."

# ---- ScienceWorld（全量 test split） ----
nohup python -u run_memory_mode_ablation.py \
    --benchmark            sciworld \
    --memory-modes         "${memory_modes}" \
    --model                "${model_name}" \
    --icl-num              "${icl_num}" \
    --max-steps            "${max_steps}" \
    --output-csv           "${output_csv}" \
    --traj-dir             "${traj_dir}" \
    --sciworld-split       test \
    --sciworld-load-memory "${PRTREE_ROOT}/ScienceWorld/storage/prtree_sciworld_offline" \
    > "logs/memory_mode-sciworld-${model_name}.log" 2>&1
echo "ScienceWorld memory_mode ablation done."

# ---- Mind2Web（全量 test_task split） ----
nohup python -u run_memory_mode_ablation.py \
    --benchmark            mind2web \
    --memory-modes         "${memory_modes}" \
    --model                "${model_name}" \
    --icl-num              "${icl_num}" \
    --max-steps            "${max_steps}" \
    --output-csv           "${output_csv}" \
    --traj-dir             "${traj_dir}" \
    --mind2web-split       test_task \
    --mind2web-load-memory "${PRTREE_ROOT}/Mind2web/storage/prtree_mind2web_offline" \
    > "logs/memory_mode-mind2web-${model_name}.log" 2>&1
echo "Mind2Web memory_mode ablation done."

echo "============================================================"
echo "  记忆模式消融全部完成 → ${output_csv}"
echo "============================================================"
