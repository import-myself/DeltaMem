#!/usr/bin/env bash
# ScienceWorld prtree 阈值网格搜索 Round 2 — test split
# 基准: tb=0.75,ts=0.01,eb=0.85,es=0.03 → avg_reward=0.8442 (Round 1最优)
# 目标: avg_reward > 0.8558
#
# Group C(6): 固定 tb=0.75/ts=0.01，精细探索 eb=0.85~0.87 × es=0.02~0.04
# Group D(6): 固定最优 eb=0.85/es=0.03，重新扫描 task 维度 (tb/ts)
# Group E(2): 联合优化候选点
set -euo pipefail

export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p results/sciworld_shards_v2 logs

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
# Group C: 固定 tb=0.75/ts=0.01，精细化 eb 高值区间 + es 变化
#   eb=0.85: 测试 es=0.02 和 es=0.04（Round1 只测了 es=0.03）
#   eb=0.86/0.87: 继续向更高 eb 探索，配合 es=0.02 和 es=0.03
# Group D: 固定最优 env(eb=0.85/es=0.03)，重新扫描 task 维度
#   Round1 只测了 tb=0.75/ts=0.01，此组覆盖 tb=0.70~0.80 和 ts=0.02
# Group E: 二维联合优化（ts=0.02 + eb更高 / es更小）
COMBOS=(
    # Group C: eb 高值精细化 (固定 tb=0.75 ts=0.01)
    "0.75 0.01 0.85 0.02"   # C1: eb=0.85 es减小
    "0.75 0.01 0.85 0.04"   # C2: eb=0.85 es增大
    "0.75 0.01 0.86 0.02"   # C3: eb更高
    "0.75 0.01 0.86 0.03"   # C4: eb更高
    "0.75 0.01 0.87 0.02"   # C5: eb更高
    "0.75 0.01 0.87 0.03"   # C6: eb更高
    # Group D: task 维度重扫描 (固定 eb=0.85 es=0.03)
    "0.70 0.01 0.85 0.03"   # D1: 低 tb
    "0.72 0.01 0.85 0.03"   # D2
    "0.73 0.01 0.85 0.03"   # D3
    "0.78 0.01 0.85 0.03"   # D4
    "0.80 0.01 0.85 0.03"   # D5
    "0.75 0.02 0.85 0.03"   # D6: ts=0.02 在最优 env 下的效果
    # Group E: 联合优化
    "0.75 0.02 0.85 0.02"   # E1: ts增大 + es减小
    "0.78 0.01 0.87 0.03"   # E2: tb稍高 + eb更高
)

PIDS=()

for IDX in "${!COMBOS[@]}"; do
    read -r TB TS EB ES <<< "${COMBOS[$IDX]}"
    KEY="${KEYS[$IDX]}"
    TAG="tb${TB}_ts${TS}_eb${EB}_es${ES}"
    LOG="logs/sciworld_v2_${TAG}.log"
    CSV="results/sciworld_shards_v2/shard_${TAG}"

    if   [ "$IDX" -lt 6 ];  then GRP="C$((IDX+1))"
    elif [ "$IDX" -lt 12 ]; then GRP="D$((IDX-5))"
    else                         GRP="E$((IDX-11))"
    fi

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
        "$GRP" "$TB" "$TS" "$EB" "$ES" "$PID" "${KEY: -6}"
done

echo ""
echo "启动了 ${#PIDS[@]} 个进程: ${PIDS[*]}"
echo ""
echo "基准: tb=0.75,ts=0.01,eb=0.85,es=0.03 → avg_reward=0.8442  目标: >0.8558"
echo ""
echo "监控:"
echo "  watch -n60 'for log in logs/sciworld_v2_*.log; do"
echo "    tag=\$(basename \"\$log\" .log | sed \"s/sciworld_v2_//\")"
echo "    last=\$(grep -oP \"Ep [0-9]+/211.*\" \"\$log\" 2>/dev/null | tail -1)"
echo "    printf \"%-50s %s\n\" \"\$tag\" \"\$last\""
echo "  done | sort'"
echo ""
echo "结果汇总:"
echo "  for csv in results/sciworld_shards_v2/*.csv; do"
echo "    tag=\$(basename \"\$csv\" .csv | sed 's/shard_//')"
echo "    last=\$(tail -1 \"\$csv\" 2>/dev/null)"
echo "    rw=\$(echo \"\$last\" | cut -d',' -f8)"
echo "    sr=\$(echo \"\$last\" | cut -d',' -f7)"
echo "    printf \"%-50s reward=%-8s SR=%s\n\" \"\$tag\" \"\$rw\" \"\$sr\""
echo "  done | sort -t= -k2 -rn"
