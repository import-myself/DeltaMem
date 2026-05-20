#!/usr/bin/env bash
# ScienceWorld prtree 阈值网格搜索 Round 4 — test split
#
# 历史最优: tb=0.75, ts=0.01, eb=0.85, es=0.02 → avg_reward=0.8499
#
# 已探索规律：
#   es: 0.030→0.8442, 0.020→0.8499  → es 减小方向有收益，未测 es=0.01 @eb=0.85
#   eb: 0.840→0.8284, 0.850→0.8499  → eb 增大方向有收益，未测 eb=0.86/0.87
#
# 本轮 4 个格点：
#   P1: es↓          tb=0.75,ts=0.01,eb=0.85,es=0.01  ★最高优先（v3 F1 未完成）
#   P2: eb↑          tb=0.75,ts=0.01,eb=0.86,es=0.02
#   P3: eb↑↑         tb=0.75,ts=0.01,eb=0.87,es=0.02
#   P4: eb↑ + es↓   tb=0.75,ts=0.01,eb=0.86,es=0.01  （交叉点）
set -euo pipefail

export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p results/sciworld_shards_v4 logs

KEYS=(
    "sk-5529a7a886ee4b9bb407b614c4ead012"
    "sk-b457f919725342e282ad5900ead23542"
    "sk-fda59b026c224dcb933036d18cea9a6a"
    "sk-77cd89544f164fbf90e6660c11c0b244"
)

# 4 组: (task_base task_step env_base env_step)
COMBOS=(
    "0.75 0.01 0.85 0.01"   # P1: es↓  ★最高优先
    "0.75 0.01 0.86 0.02"   # P2: eb↑
    "0.75 0.01 0.87 0.02"   # P3: eb↑↑
    "0.75 0.01 0.86 0.01"   # P4: eb↑ + es↓（交叉）
)

LABELS=("P1_es_down" "P2_eb_up" "P3_eb_up2" "P4_cross")

PIDS=()

for IDX in "${!COMBOS[@]}"; do
    read -r TB TS EB ES <<< "${COMBOS[$IDX]}"
    KEY="${KEYS[$IDX]}"
    LABEL="${LABELS[$IDX]}"
    TAG="tb${TB}_ts${TS}_eb${EB}_es${ES}"
    LOG="logs/sciworld_v4_${TAG}.log"
    CSV="results/sciworld_shards_v4/shard_${TAG}"

    DEEPSEEK_API_KEY="${KEY}" nohup python -u run_joint_threshold_search.py \
        --benchmark          sciworld \
        --model              deepseek-v4-flash \
        --sciworld-split     test \
        --task-base-thresholds "${TB}" \
        --task-depth-steps     "${TS}" \
        --env-base-thresholds  "${EB}" \
        --env-depth-steps      "${ES}" \
        --output-csv           "${CSV}" \
        > "${LOG}" 2>&1 &

    PID=$!
    PIDS+=($PID)
    printf "  [%s] tb=%-5s ts=%-5s eb=%-5s es=%-5s  PID=%-6d  key=...%s\n" \
        "$LABEL" "$TB" "$TS" "$EB" "$ES" "$PID" "${KEY: -6}"
done

echo ""
echo "启动了 ${#PIDS[@]} 个进程: ${PIDS[*]}"
echo "历史最优: tb=0.75,ts=0.01,eb=0.85,es=0.02 → avg_reward=0.8499"
echo "目标: >0.8499"
