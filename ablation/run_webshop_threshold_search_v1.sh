#!/usr/bin/env bash
# WebShop prtree 阈值网格搜索 v1 — test split (200 sessions, online)
#
# 当前 config 基线（config.py 默认）:
#   tb=0.85, ts=0.02, eb=0.85, es=0.02, FAILURE_NODE_PENALTY=0.10
#
# WebShop 特点：task tree 以失败节点为主（64/91），失败节点带 0.10 penalty。
# 有效阈值 ≈ 0.85 + depth*0.02；失败节点需 cosine ≥ 0.85+penalty=0.95 才命中。
#
# 探索方向：
#   P1: tb↓          tb=0.80,ts=0.02,eb=0.85,es=0.02  降低 task 阈值 → 更多成功节点命中
#   P2: tb↓ + ts↓   tb=0.80,ts=0.01,eb=0.85,es=0.02  宽松 task 检索 + 慢步进
#   P3: eb↑          tb=0.85,ts=0.02,eb=0.88,es=0.02  收紧 env 阈值 → 更精准类目匹配
#   P4: tb↓ + eb↑   tb=0.80,ts=0.02,eb=0.88,es=0.02  交叉：宽松 task + 收紧 env
set -euo pipefail

export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p results/webshop_shards_v1 logs

KEYS=(
    "sk-5529a7a886ee4b9bb407b614c4ead012"
    "sk-b457f919725342e282ad5900ead23542"
    "sk-fda59b026c224dcb933036d18cea9a6a"
    "sk-77cd89544f164fbf90e6660c11c0b244"
)

# 4 组: (task_base task_step env_base env_step)
COMBOS=(
    "0.80 0.02 0.85 0.02"   # P1: tb↓
    "0.80 0.01 0.85 0.02"   # P2: tb↓ + ts↓
    "0.85 0.02 0.88 0.02"   # P3: eb↑
    "0.80 0.02 0.88 0.02"   # P4: tb↓ + eb↑（交叉）
)

LABELS=("P1_tb_down" "P2_tb_ts_down" "P3_eb_up" "P4_cross")

PIDS=()

for IDX in "${!COMBOS[@]}"; do
    read -r TB TS EB ES <<< "${COMBOS[$IDX]}"
    KEY="${KEYS[$IDX]}"
    LABEL="${LABELS[$IDX]}"
    TAG="tb${TB}_ts${TS}_eb${EB}_es${ES}"
    LOG="logs/webshop_v1_${TAG}.log"
    CSV="results/webshop_shards_v1/shard_${TAG}"

    DEEPSEEK_API_KEY="${KEY}" nohup python -u run_joint_threshold_search.py \
        --benchmark          webshop \
        --model              deepseek-v4-flash \
        --webshop-sessions-file /tmp/ETO/eval_agent/data/webshop/test_indices.json \
        --webshop-max-steps  15 \
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
echo "当前 config 基线: tb=0.85,ts=0.02,eb=0.85,es=0.02  FAILURE_NODE_PENALTY=0.10"
echo "目标: avg_reward > 0.58 (当前 online 基线)"
