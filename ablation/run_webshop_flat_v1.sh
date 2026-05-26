#!/usr/bin/env bash
# WebShop flat retrieval 验证实验 v1 — test split (200 sessions)
#
# flat retrieval 只有 base_threshold 有效，depth_step 传 0.0（不影响检索）。
# DFS 基线最优：tb=0.80,ts=0.01,eb=0.85 → SR=48.50%, AvgReward=0.5800
#
# 4 组 (task_base × env_base):
#   F1: 0.80 / 0.80  双宽松
#   F2: 0.80 / 0.85  对标 DFS 最优
#   F3: 0.85 / 0.85  默认 config
#   F4: 0.85 / 0.90  收紧 env
set -euo pipefail

export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p results/webshop_flat_v1 logs

KEYS=(
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
    "${DEEPSEEK_API_KEY}"
)

# 4 组: (task_base env_base) — depth_step 固定 0.0
COMBOS=(
    "0.80 0.80"   # F1: 双宽松
    "0.80 0.85"   # F2: 对标 DFS 最优基线
    "0.85 0.85"   # F3: 默认 config
    "0.85 0.90"   # F4: 收紧 env
)

LABELS=("F1_loose" "F2_dfs_baseline" "F3_default" "F4_tight_env")

PIDS=()

for IDX in "${!COMBOS[@]}"; do
    read -r TB EB <<< "${COMBOS[$IDX]}"
    KEY="${KEYS[$IDX]}"
    LABEL="${LABELS[$IDX]}"
    TAG="flat_tb${TB}_eb${EB}"
    LOG="logs/webshop_${TAG}.log"
    CSV="results/webshop_flat_v1/shard_${TAG}"

    DEEPSEEK_API_KEY="${KEY}" nohup python -u run_joint_threshold_search.py \
        --benchmark             webshop \
        --model                 deepseek-v4-flash \
        --webshop-sessions-file /tmp/ETO/eval_agent/data/webshop/test_indices.json \
        --webshop-max-steps     15 \
        --task-base-thresholds  "${TB}" \
        --task-depth-steps      0.0 \
        --env-base-thresholds   "${EB}" \
        --env-depth-steps       0.0 \
        --output-csv            "${CSV}" \
        > "${LOG}" 2>&1 &

    PID=$!
    PIDS+=($PID)
    printf "  [%s] tb=%-5s eb=%-5s  PID=%-6d  key=...%s\n" \
        "$LABEL" "$TB" "$EB" "$PID" "${KEY: -6}"
done

echo ""
echo "启动了 ${#PIDS[@]} 个进程: ${PIDS[*]}"
echo "DFS 基线: tb=0.80,ts=0.01,eb=0.85 → SR=48.50%, AvgReward=0.5800"
echo "结果目录: results/webshop_flat_v1/"
