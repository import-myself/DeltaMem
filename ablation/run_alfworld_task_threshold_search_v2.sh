#!/usr/bin/env bash
# ALFWorld task tree 阈值网格搜索 v2 — 收紧 skip 后验证
# task_base {0.70,0.72,0.75,0.78} x task_step {0.01,0.02} = 8 组
# 依据 v1 结果：b0.75 最优(SR=79.10%)，b0.80/0.85 碎片化
# 本轮：以 0.75 为中心，向下探 0.70/0.72，向上仅到 0.78；step 保留 0.01/0.02
set -euo pipefail

export ALFWORLD_DATA="${ALFWORLD_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../ALFWorld/data/alfworld}"
export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p results/ts_v2_shards logs

# 7 个 API key（循环使用）
KEYS=(
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
)

ENV_BASE="0.85"
ENV_STEP="0.03"

PIDS=()
KEY_IDX=0

for TB in 0.70 0.72 0.75 0.78; do
    for TS in 0.01 0.02; do
        KEY="${KEYS[$((KEY_IDX % ${#KEYS[@]}))]}"
        TAG="b${TB}_s${TS}"
        LOG="logs/alfworld_ts_v2_${TAG}.log"
        CSV="results/ts_v2_shards/shard_${TAG}"

        DEEPSEEK_API_KEY="${KEY}" nohup python -u run_joint_threshold_search.py \
            --benchmark            alfworld \
            --model                deepseek-v4-flash \
            --alfworld-split       eval_out_of_distribution \
            --task-base-thresholds "${TB}" \
            --task-depth-steps     "${TS}" \
            --env-base-thresholds  "${ENV_BASE}" \
            --env-depth-steps      "${ENV_STEP}" \
            --alfworld-max-steps   30 \
            --output-csv           "${CSV}" \
            > "${LOG}" 2>&1 &

        PID=$!
        PIDS+=($PID)
        echo "  [${TAG}] PID=${PID}  key=...${KEY: -6}  log=${LOG}"
        KEY_IDX=$((KEY_IDX + 1))
    done
done

echo ""
echo "启动了 ${#PIDS[@]} 个并发进程: ${PIDS[*]}"
echo "等待全部完成（预计 ~30 分钟）..."
wait "${PIDS[@]}"

# 合并所有 shard CSV
MERGED="results/alfworld_task_threshold_search_v2.csv"
HEADER_WRITTEN=0
> "${MERGED}"
for SHARD in results/ts_v2_shards/shard_*_alfworld.csv; do
    [ -f "${SHARD}" ] || continue
    if [ "${HEADER_WRITTEN}" -eq 0 ]; then
        cat "${SHARD}" >> "${MERGED}"
        HEADER_WRITTEN=1
    else
        tail -n +2 "${SHARD}" >> "${MERGED}"
    fi
done

echo ""
echo "合并完成 → ${MERGED}"
echo ""
echo "=== 最优参数组合（与 no-memory 79.85% 对比）==="
python3 -c "
import csv
rows = []
try:
    with open('${MERGED}') as f:
        for r in csv.DictReader(f):
            rows.append(r)
except FileNotFoundError:
    print('CSV not found'); exit()
rows.sort(key=lambda r: (-float(r['success_rate']), float(r['avg_steps'])))
print(f\"{'#':>3}  {'task_base':>10}  {'task_step':>10}  {'SR':>8}  {'Steps':>7}  {'TaskHit':>8}  {'AvgTaskDepth':>13}\")
print('-'*75)
for i, r in enumerate(rows, 1):
    marker = ' <-- BEST' if i == 1 else (' [>baseline]' if float(r['success_rate']) > 0.7985 else '')
    print(f\"{i:>3}  {float(r['task_base_threshold']):>10.2f}  {float(r['task_depth_step']):>10.2f}  \"
          f\"{float(r['success_rate'])*100:>7.2f}%  {float(r['avg_steps']):>7.2f}  \"
          f\"{float(r['task_hit_rate'])*100:>7.2f}%  {float(r['avg_task_retrieval_len_hit']):>13.3f}{marker}\")
"
