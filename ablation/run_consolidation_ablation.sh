#!/usr/bin/env bash
# 实验 2.3：Skill 固化阈值消融（run_consolidation_ablation.py）
# 验证 CONSOLIDATION_THRESHOLD ∈ {1, 2, 3, 5, 8} 对 SkillCache 质量的影响
# 仅使用 ALFWorld eval_in_distribution
set -euo pipefail

export ALFWORLD_DATA="${ALFWORLD_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../ALFWorld/data/alfworld}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}"
export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRTREE_ROOT="$(dirname "${SCRIPT_DIR}")"

model_name="deepseek-v4-flash"
icl_num=1
max_steps=30
thresholds="1,2,3,5,8"
output_csv="results/consolidation_ablation.csv"
traj_dir="trajectories/consolidation_ablation"

# 预构建的离线 PRTree（SkillCache 会被自动清空并在运行时重新积累）
load_memory="${PRTREE_ROOT}/ALFWorld/storage/prtree_dual_offline"

mkdir -p logs results "${traj_dir}"

nohup python -u run_consolidation_ablation.py \
    --thresholds             "${thresholds}" \
    --load-memory            "${load_memory}" \
    --split                  eval_in_distribution \
    --model                  "${model_name}" \
    --icl-num                "${icl_num}" \
    --max-steps              "${max_steps}" \
    --output-csv             "${output_csv}" \
    --traj-dir               "${traj_dir}" \
    > "logs/consolidation_ablation-${model_name}.log" 2>&1 &

echo "Consolidation ablation started: PID=$!"
echo "实时日志: tail -f logs/consolidation_ablation-${model_name}.log"
echo "结果 CSV: ${output_csv}"
