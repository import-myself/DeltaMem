#!/usr/bin/env bash
# Mind2Web test_website 网格搜索 — 新 prompt 版本
set -euo pipefail

export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p results/mind2web_v2_shards logs

KEYS=(
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
)

# 7 组参数：围绕默认 tb=0.75/ts=0.03/eb=0.92/es=0.01 搜索
COMBOS=(
    "0.70 0.02 0.90 0.01"
    "0.70 0.03 0.92 0.01"
    "0.72 0.01 0.90 0.01"
    "0.72 0.02 0.90 0.01"
    "0.75 0.01 0.92 0.01"
    "0.75 0.02 0.94 0.01"
    "0.78 0.01 0.92 0.01"
)

PIDS=()

for IDX in "${!COMBOS[@]}"; do
    read -r TB TS EB ES <<< "${COMBOS[$IDX]}"
    KEY="${KEYS[$IDX]}"
    TAG="tb${TB}_ts${TS}_eb${EB}_es${ES}"
    LOG="logs/mind2web_v2_${TAG}.log"
    CSV="results/mind2web_v2_shards/shard_${TAG}"

    DEEPSEEK_API_KEY="${KEY}" nohup python -u run_joint_threshold_search.py \
        --benchmark            mind2web \
        --model                deepseek-v4-flash \
        --mind2web-split       test_website \
        --task-base-thresholds "${TB}" \
        --task-depth-steps     "${TS}" \
        --env-base-thresholds  "${EB}" \
        --env-depth-steps      "${ES}" \
        --output-csv           "${CSV}" \
        > "${LOG}" 2>&1 &

    PID=$!
    PIDS+=($PID)
    printf "  [%d] tb=%-4s ts=%-5s eb=%-4s es=%-5s  PID=%-6d  key=...%s\n" \
        "$((IDX+1))" "$TB" "$TS" "$EB" "$ES" "$PID" "${KEY: -6}"
done

echo ""
echo "启动了 ${#PIDS[@]} 个进程: ${PIDS[*]}"
