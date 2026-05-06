export ALFWORLD_DATA='/hdd/REDACTED_USER/PRTree/ALFWorld/data/alfworld'
export API_KEY='sk-uQ3Q4igYxnqjrEAcXfatMws18iO180Vn8dFRSYPcpmj3Zpc2'
start_port=8011
model_name="Qwen3-14B"
split="eval_in_distribution"
benchmark="alfworld"
export BASE_URL="http://localhost:$start_port/v1"

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
load_memory="${PRTREE_ROOT}/ALFWorld/storage/prtree_dual_offline"

mkdir -p logs results "${traj_dir}"

nohup python -u run_memory_mode_ablation.py \
    --benchmark            alfworld \
    --memory-modes         "${memory_modes}" \
    --model                "${model_name}" \
    --icl-num              "${icl_num}" \
    --max-steps            "${max_steps}" \
    --output-csv           "${output_csv}" \
    --traj-dir             "${traj_dir}" \
    --alfworld-split       "${split}" \
    --alfworld-load-memory "${load_memory}" \
    > "logs/memory_mode-${benchmark}-${model_name}.log" 2>&1 &

echo "Started PID=$!"
