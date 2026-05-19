export DEEPSEEK_API_KEY='REDACTED_API_KEY'
export DEEPSEEK_BASE_URL='https://api.deepseek.com'
model_name="deepseek-v4-flash"
split="eval_in_distribution"
benchmark="alfworld"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRTREE_ROOT="$(dirname "${SCRIPT_DIR}")"

# 路径规则：results/threshold_ablation.csv
#           trajectories/threshold_ablation/
#           logs/threshold-{benchmark}-{model}.log

output_csv="results/threshold_ablation.csv"
traj_dir="trajectories/threshold_ablation"

# ---- 阈值参数网格 ----
task_base_thresholds="0.70,0.75,0.80,0.85,0.90"
task_depth_steps="0.01,0.02,0.03,0.04,0.05"
env_base_thresholds="0.82,0.85,0.88,0.91,0.94"
env_depth_steps="0.01,0.02,0.03,0.04"

max_steps=30
icl_num=1
load_memory="${PRTREE_ROOT}/ALFWorld/storage/prtree_dual_offline"

mkdir -p logs results "${traj_dir}"

nohup python -u run_threshold_ablation.py \
    --mode                 both \
    --model                "${model_name}" \
    --split                "${split}" \
    --max-steps            "${max_steps}" \
    --icl-num              "${icl_num}" \
    --load-memory          "${load_memory}" \
    --output-csv           "${output_csv}" \
    --traj-dir             "${traj_dir}" \
    --task-base-thresholds "${task_base_thresholds}" \
    --task-depth-steps     "${task_depth_steps}" \
    --env-base-thresholds  "${env_base_thresholds}" \
    --env-depth-steps      "${env_depth_steps}" \
    > "logs/threshold-${benchmark}-${model_name}.log" 2>&1 &

echo "Started PID=$!"
