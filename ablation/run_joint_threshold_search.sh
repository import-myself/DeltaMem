export ALFWORLD_DATA='/data/REDACTED_USER/PRTree/ALFWorld/data/alfworld'
export API_KEY='sk-uQ3Q4igYxnqjrEAcXfatMws18iO180Vn8dFRSYPcpmj3Zpc2'
start_port=8010
model_name="Qwen3-14B"
export BASE_URL="http://localhost:$start_port/v1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRTREE_ROOT="$(dirname "${SCRIPT_DIR}")"

# =============================================================================
# ★ benchmark 选择：alfworld / sciworld / mind2web / all
# =============================================================================
benchmark="mind2web"   # 选择单一 benchmark 进行调试

# =============================================================================
# ★ 各 benchmark 的 split 和离线记忆路径（按需修改）
# =============================================================================

# ALFWorld
alfworld_split="eval_in_distribution"
alfworld_load_memory="${PRTREE_ROOT}/ALFWorld/storage/prtree_dual_offline_all"

# ScienceWorld
sciworld_split="dev"
sciworld_load_memory="${PRTREE_ROOT}/ScienceWorld/storage/prtree_sciworld_offline"

# Mind2Web
mind2web_split="test_task"
mind2web_load_memory="${PRTREE_ROOT}/Mind2web/storage/prtree_mind2web_offline"

# =============================================================================
# ★ 网格配置
#   默认 3×3×3×3 = 81 组
# =============================================================================

# ---- 粗粒度搜索（推荐先跑） ----
task_base_thresholds="0.75,0.80,0.85"
task_depth_steps="0.01,0.03,0.05"
env_base_thresholds="0.82,0.88,0.94"
env_depth_steps="0.01,0.02,0.04"

# ---- 细粒度搜索（在粗搜索最优点附近加密，按需解注释替换上面的配置）----
# task_base_thresholds="0.78,0.80,0.82,0.84"
# task_depth_steps="0.02,0.03,0.04"
# env_base_thresholds="0.86,0.88,0.90"
# env_depth_steps="0.015,0.020,0.025"

# =============================================================================
# ★ episode 数
#   max_episodes=0（或留空）→ 跑完整个 split，无需知道具体数量
#   max_episodes=N         → 最多跑 N 个 episode（用于快速调试）
# =============================================================================
max_episodes=0

# =============================================================================
# ★ 各 benchmark 的每局最大步数
#   ScienceWorld 的步数由内部 max_steps.json 按任务类型决定，alfworld_max_steps 对其无效
# =============================================================================
alfworld_max_steps=30

icl_num=1
top_k=5

# 路径规则：各 benchmark 输出到独立 CSV，如 results/joint_threshold_search_all_sciworld.csv
output_csv="results/joint_threshold_search_all"   # 脚本自动追加 _{benchmark}.csv
traj_dir="trajectories/joint_threshold_search_all"

mkdir -p logs results "${traj_dir}"

# 构造可选的 --max-episodes 参数（0 或空则不传，让脚本跑全量）
episodes_arg=""
if [ "${max_episodes}" -gt 0 ] 2>/dev/null; then
    episodes_arg="--max-episodes ${max_episodes}"
fi

nohup python -u run_joint_threshold_search.py \
    --mode                   grid \
    --benchmark              "${benchmark}" \
    --model                  "${model_name}" \
    ${episodes_arg} \
    --alfworld-max-steps     "${alfworld_max_steps}" \
    --icl-num                "${icl_num}" \
    --alfworld-split         "${alfworld_split}" \
    --alfworld-load-memory   "${alfworld_load_memory}" \
    --sciworld-split         "${sciworld_split}" \
    --sciworld-load-memory   "${sciworld_load_memory}" \
    --mind2web-split         "${mind2web_split}" \
    --mind2web-load-memory   "${mind2web_load_memory}" \
    --output-csv             "${output_csv}" \
    --traj-dir               "${traj_dir}" \
    --task-base-thresholds   "${task_base_thresholds}" \
    --task-depth-steps       "${task_depth_steps}" \
    --env-base-thresholds    "${env_base_thresholds}" \
    --env-depth-steps        "${env_depth_steps}" \
    --top-k                  "${top_k}" \
    > "logs/joint_threshold-${benchmark}-${model_name}_all.log" 2>&1 &

echo "Started PID=$!"
echo "实时日志: tail -f logs/joint_threshold-${benchmark}-${model_name}_all.log"
