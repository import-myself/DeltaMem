export API_KEY='sk-uQ3Q4igYxnqjrEAcXfatMws18iO180Vn8dFRSYPcpmj3Zpc2'
start_port=8011
model_name="Qwen3-14B"
split="test_task"
benchmark="mind2web"
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
load_memory="${PRTREE_ROOT}/Mind2web/storage/prtree_mind2web_offline"

mkdir -p logs results "${traj_dir}"

nohup python -u run_memory_mode_ablation.py \
    --benchmark            mind2web \
    --memory-modes         "${memory_modes}" \
    --model                "${model_name}" \
    --icl-num              "${icl_num}" \
    --max-steps            "${max_steps}" \
    --output-csv           "${output_csv}" \
    --traj-dir             "${traj_dir}" \
    --mind2web-split       "${split}" \
    --mind2web-load-memory "${load_memory}" \
    > "logs/memory_mode-${benchmark}-${model_name}.log" 2>&1 &

echo "Started PID=$!"
