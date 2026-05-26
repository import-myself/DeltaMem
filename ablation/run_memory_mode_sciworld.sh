export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}"
export DEEPSEEK_BASE_URL='https://api.deepseek.com'
model_name="deepseek-v4-flash"
split="test"
benchmark="sciworld"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRTREE_ROOT="$(dirname "${SCRIPT_DIR}")"

# 路径规则：results/memory_mode_ablation.csv
#           trajectories/memory_mode_ablation/{benchmark}/
#           logs/memory_mode-{benchmark}-{model}.log

output_csv="results/memory_mode_ablation.csv"
traj_dir="trajectories/memory_mode_ablation"

# ---- 记忆模式列表（逗号分隔） ----
memory_modes="task_only,env_only"

max_steps=30
icl_num=1
load_memory="${PRTREE_ROOT}/ScienceWorld/storage/prtree_sciworld_offline"

mkdir -p logs results "${traj_dir}"

nohup python -u run_memory_mode_ablation.py \
    --benchmark            sciworld \
    --memory-modes         "${memory_modes}" \
    --model                "${model_name}" \
    --icl-num              "${icl_num}" \
    --max-steps            "${max_steps}" \
    --output-csv           "${output_csv}" \
    --traj-dir             "${traj_dir}" \
    --sciworld-split       "${split}" \
    --sciworld-load-memory "${load_memory}" \
    > "logs/memory_mode-${benchmark}-${model_name}.log" 2>&1 &

echo "Started PID=$!"
