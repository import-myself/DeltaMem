#!/usr/bin/env bash
# Mind2web flat retrieval 验证实验 v1 — test_website split
#
# DFS 基线（默认阈值 tb=0.75/eb=0.92）:
#   Element Acc=0.3838  Action F1=0.4359  Step SR=0.3146  Task SR=1.69%
#
# flat retrieval 只有 base_threshold 有效，depth_step 传 0.0。
# 4 组：围绕默认阈值探索
#   F1: tb=0.75 / eb=0.92  对标 DFS 默认
#   F2: tb=0.75 / eb=0.90  放宽 env
#   F3: tb=0.70 / eb=0.92  放宽 task
#   F4: tb=0.70 / eb=0.90  双放宽
set -euo pipefail

export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p results/mind2web_flat_v1 logs

KEYS=(
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
)

COMBOS=(
    "0.75 0.92"   # F1: 对标 DFS 默认阈值
    "0.75 0.90"   # F2: 放宽 env
    "0.70 0.92"   # F3: 放宽 task
    "0.70 0.90"   # F4: 双放宽
)

LABELS=("F1_default" "F2_loose_env" "F3_loose_task" "F4_both_loose")

PIDS=()

for IDX in "${!COMBOS[@]}"; do
    read -r TB EB <<< "${COMBOS[$IDX]}"
    KEY="${KEYS[$IDX]}"
    LABEL="${LABELS[$IDX]}"
    TAG="flat_tb${TB}_eb${EB}"
    LOG="logs/mind2web_${TAG}.log"
    CSV="results/mind2web_flat_v1/shard_${TAG}"

    DEEPSEEK_API_KEY="${KEY}" nohup python -u run_joint_threshold_search.py \
        --benchmark            mind2web \
        --model                deepseek-v4-flash \
        --mind2web-split       test_website \
        --task-base-thresholds "${TB}" \
        --task-depth-steps     0.0 \
        --env-base-thresholds  "${EB}" \
        --env-depth-steps      0.0 \
        --output-csv           "${CSV}" \
        > "${LOG}" 2>&1 &

    PID=$!
    PIDS+=($PID)
    printf "  [%s] tb=%-5s eb=%-5s  PID=%-6d  key=...%s\n" \
        "$LABEL" "$TB" "$EB" "$PID" "${KEY: -6}"
done

echo ""
echo "启动了 ${#PIDS[@]} 个进程: ${PIDS[*]}"
echo "DFS 基线: tb=0.75/eb=0.92 → EleAcc=0.3838  F1=0.4359  StepSR=0.3146  TaskSR=1.69%"
echo "结果目录: results/mind2web_flat_v1/"
