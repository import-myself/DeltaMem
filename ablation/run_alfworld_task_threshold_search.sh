#!/usr/bin/env bash
# ALFWorld task tree 阈值网格搜索 — 9 组并发，每组独立 CSV，完成后合并分析
# task_base {0.75,0.80,0.85} x task_step {0.01,0.02} = 6 组
# 全部并发，预计 ~30 分钟出结果
set -euo pipefail

export ALFWORLD_DATA="${ALFWORLD_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../ALFWorld/data/alfworld}"
export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p results/ts_shards logs

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

for TB in 0.75 0.80 0.85; do
    for TS in 0.01 0.02; do
        KEY="${KEYS[$((KEY_IDX % ${#KEYS[@]}))]}"
        TAG="b${TB}_s${TS}"
        LOG="logs/alfworld_ts_${TAG}.log"
        CSV="results/ts_shards/shard_${TAG}"   # 每组独立 CSV，避免并发写冲突

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

# 合并所有 shard CSV，去掉重复的 header
MERGED="results/alfworld_task_threshold_search_alfworld.csv"
HEADER_WRITTEN=0
> "${MERGED}"
for SHARD in results/ts_shards/shard_*_alfworld.csv; do
    [ -f "${SHARD}" ] || continue
    if [ "${HEADER_WRITTEN}" -eq 0 ]; then
        cat "${SHARD}" >> "${MERGED}"
        HEADER_WRITTEN=1
    else
        tail -n +2 "${SHARD}" >> "${MERGED}"   # 跳过 header 行
    fi
done

echo ""
echo "合并完成 → ${MERGED}"
echo ""
echo "=== 最优参数组合 ==="
python -c "
import csv, sys
rows = []
with open('${MERGED}') as f:
    for r in csv.DictReader(f):
        rows.append(r)
rows.sort(key=lambda r: (-float(r['success_rate']), float(r['avg_steps'])))
print(f\"{'#':>3}  {'task_base':>10}  {'task_step':>10}  {'SR':>8}  {'Steps':>7}  {'TaskHit':>8}  {'AvgTaskDepth':>13}\")
print('-'*75)
for i, r in enumerate(rows, 1):
    marker = ' <-- BEST' if i == 1 else ''
    print(f\"{i:>3}  {float(r['task_base_threshold']):>10.2f}  {float(r['task_depth_step']):>10.2f}  \"
          f\"{float(r['success_rate'])*100:>7.2f}%  {float(r['avg_steps']):>7.2f}  \"
          f\"{float(r['task_hit_rate'])*100:>7.2f}%  {float(r['avg_task_retrieval_len_hit']):>13.3f}{marker}\")
"
