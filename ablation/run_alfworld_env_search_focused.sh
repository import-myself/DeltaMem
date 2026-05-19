#!/usr/bin/env bash
# ALFWorld env 精细搜索 — 围绕最优点 eb=0.80/es=0.04 (81.67%) 加密
# 7 组新实验，复用 7 个释放的 API key
set -euo pipefail

export ALFWORLD_DATA='/hdd/REDACTED_USER/DeltaMem/ALFWorld/data/alfworld'
export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p results/env_search_shards logs

# 7 个释放的 key
KEYS=(
    "sk-5529a7a886ee4b9bb407b614c4ead012"
    "sk-b457f919725342e282ad5900ead23542"
    "sk-fda59b026c224dcb933036d18cea9a6a"
    "sk-77cd89544f164fbf90e6660c11c0b244"
    "sk-637985b50bfb4b4dbf7beeaed8e9fd37"
    "sk-9924eacd416741088d71d10bbd84c693"
    "sk-f7640e8114204d0380e6c264d898978a"
)

# 7 组：围绕 eb=0.80/es=0.04 邻域 + 验证低eb高step规律
COMBOS=(
    "0.75 0.01 0.78 0.03"
    "0.75 0.01 0.78 0.04"
    "0.75 0.01 0.78 0.05"
    "0.75 0.01 0.80 0.035"
    "0.75 0.01 0.80 0.05"
    "0.75 0.01 0.82 0.035"
    "0.75 0.01 0.82 0.04"
)

PIDS=()

for IDX in "${!COMBOS[@]}"; do
    read -r TB TS EB ES <<< "${COMBOS[$IDX]}"
    KEY="${KEYS[$IDX]}"
    TAG="tb${TB}_ts${TS}_eb${EB}_es${ES}"
    LOG="logs/alfworld_env_search_${TAG}.log"
    CSV="results/env_search_shards/shard_${TAG}"

    DEEPSEEK_API_KEY="${KEY}" nohup python -u run_joint_threshold_search.py \
        --benchmark            alfworld \
        --model                deepseek-v4-flash \
        --alfworld-split       eval_out_of_distribution \
        --task-base-thresholds "${TB}" \
        --task-depth-steps     "${TS}" \
        --env-base-thresholds  "${EB}" \
        --env-depth-steps      "${ES}" \
        --alfworld-max-steps   30 \
        --output-csv           "${CSV}" \
        > "${LOG}" 2>&1 &

    PID=$!
    PIDS+=($PID)
    printf "  [%d] tb=%-4s ts=%-4s eb=%-5s es=%-5s  PID=%-6d  key=...%s\n" \
        "$((IDX+1))" "$TB" "$TS" "$EB" "$ES" "$PID" "${KEY: -6}"
done

echo ""
echo "启动了 ${#PIDS[@]} 个新进程: ${PIDS[*]}"
