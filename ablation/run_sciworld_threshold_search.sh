#!/usr/bin/env bash
# ScienceWorld prtree 阈值网格搜索 — test split
# 基准: tb=0.85,ts=0.03,eb=0.82,es=0.04 → SR=59.24% (synapse=72.51%)
# 问题: 219 task nodes / 210 eps，基本没有节点共享，tb=0.85 过严
# Group A(8): 固定 env=0.82/0.04，扫描 task_base 0.70→0.85
# Group B(6): 固定 task=0.75/0.01，扫描 env 维度
set -euo pipefail

export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p results/sciworld_shards logs

KEYS=(
    "sk-5529a7a886ee4b9bb407b614c4ead012"
    "sk-b457f919725342e282ad5900ead23542"
    "sk-fda59b026c224dcb933036d18cea9a6a"
    "sk-7cca8e16e772422796c73c5ef8bdc13f"
    "sk-77cd89544f164fbf90e6660c11c0b244"
    "sk-b8580c45f7d140608cad68690f3d9101"
    "sk-637985b50bfb4b4dbf7beeaed8e9fd37"
    "sk-9924eacd416741088d71d10bbd84c693"
    "sk-00f63c0f59f2490f8b5ff17eea0c28ac"
    "sk-f7640e8114204d0380e6c264d898978a"
    "sk-d05f4122f07146b4b855a2a7f5bc8c71"
    "sk-26364e54c05848fe8f3dd4523077d9ea"
    "sk-a1347041cff94176a2bed4eddf35ad73"
    "sk-5c4c5a1ef40c4f9c901f120c2abb42e3"
)

# 14 组: (task_base task_step env_base env_step)
# Group A: env 固定 0.82/0.04，扫描 task
# Group B: task 固定 0.75/0.01，扫描 env
COMBOS=(
    "0.70 0.01 0.82 0.04"
    "0.72 0.01 0.82 0.04"
    "0.75 0.01 0.82 0.04"
    "0.75 0.02 0.82 0.04"
    "0.78 0.01 0.82 0.04"
    "0.80 0.01 0.82 0.04"
    "0.83 0.01 0.82 0.04"
    "0.85 0.01 0.82 0.04"
    "0.75 0.01 0.78 0.03"
    "0.75 0.01 0.80 0.03"
    "0.75 0.01 0.80 0.04"
    "0.75 0.01 0.82 0.02"
    "0.75 0.01 0.82 0.03"
    "0.75 0.01 0.85 0.03"
)

PIDS=()

for IDX in "${!COMBOS[@]}"; do
    read -r TB TS EB ES <<< "${COMBOS[$IDX]}"
    KEY="${KEYS[$IDX]}"
    TAG="tb${TB}_ts${TS}_eb${EB}_es${ES}"
    LOG="logs/sciworld_search_${TAG}.log"
    CSV="results/sciworld_shards/shard_${TAG}"

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
    GRP=$( [ "$IDX" -lt 8 ] && echo "A" || echo "B" )
    printf "  [%s%2d] tb=%-5s ts=%-5s eb=%-5s es=%-5s  PID=%-6d  key=...%s\n" \
        "$GRP" "$((IDX+1))" "$TB" "$TS" "$EB" "$ES" "$PID" "${KEY: -6}"
done

echo ""
echo "启动了 ${#PIDS[@]} 个进程: ${PIDS[*]}"
echo "基准: tb=0.85,ts=0.03,eb=0.82,es=0.04 → SR=59.24%  synapse=72.51%"
echo ""
echo "监控: watch -n60 'for log in logs/sciworld_search_*.log; do"
echo "  tag=\$(basename \"\$log\" .log | sed \"s/sciworld_search_//\")"
echo "  last=\$(grep -oP \"Ep [0-9]+/211: SR=[0-9.]+%\" \"\$log\" 2>/dev/null | tail -1)"
echo "  printf \"%-45s %s\n\" \"\$tag\" \"\$last\""
echo "done | sort -t= -k3 -rn'"
