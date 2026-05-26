#!/usr/bin/env bash
# ALFWorld task 阈值搜索 — 固定最优 env(eb=0.80, es=0.04)，搜索 task 维度
# task_base {0.70,0.72,0.75,0.78,0.80} x task_step {0.005,0.01,0.02} = 15 组
# 基准: tb=0.75, ts=0.01, eb=0.85, es=0.03 → 84.33%; eb=0.80,es=0.04 → 81.67%@Ep60
set -euo pipefail

export ALFWORLD_DATA="${ALFWORLD_DATA:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../ALFWorld/data/alfworld}"
export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p results/task_search_shards logs

# 7 个释放的 key
KEYS=(
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
)

ENV_BASE="0.80"
ENV_STEP="0.04"

# 7 组 task 参数 (tb=0.75/ts=0.01 已在跑，不重复)
COMBOS=(
    "0.70 0.01"
    "0.72 0.01"
    "0.75 0.005"
    "0.75 0.015"
    "0.78 0.01"
    "0.80 0.01"
    "0.73 0.01"
)

PIDS=()

for IDX in "${!COMBOS[@]}"; do
    read -r TB TS <<< "${COMBOS[$IDX]}"
    KEY="${KEYS[$IDX]}"
    TAG="tb${TB}_ts${TS}_eb${ENV_BASE}_es${ENV_STEP}"
    LOG="logs/alfworld_task_search_${TAG}.log"
    CSV="results/task_search_shards/shard_${TAG}"

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
    printf "  [%d] tb=%-5s ts=%-5s eb=%s es=%s  PID=%-6d  key=...%s\n" \
        "$((IDX+1))" "$TB" "$TS" "$ENV_BASE" "$ENV_STEP" "$PID" "${KEY: -6}"
done

echo ""
echo "启动了 ${#PIDS[@]} 个进程: ${PIDS[*]}"
echo "(tb=0.75/ts=0.01/eb=0.80/es=0.04 已在原进程中运行，无需重复)"
