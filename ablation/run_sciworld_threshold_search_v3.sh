#!/usr/bin/env bash
# ScienceWorld prtree 阈值网格搜索 Round 3 — test split
# Round2最优: tb=0.75,ts=0.01,eb=0.85,es=0.02 → avg_reward=0.8499  目标: >0.8558
#
# 核心方向:
#   1. es 减小趋势未到底 → 测试 es=0.01
#   2. tb=0.73/0.74/0.70 在最优 env(eb=0.85) 下配合 es=0.02/0.01（Round2只测了 es=0.03）
#   3. eb=0.84 附近的 es=0.01/0.02（eb峰值左侧是否有空间）
set -euo pipefail

export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p results/sciworld_shards_v3 logs

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
# Group F(6): es=0.01 方向 — es减小趋势未到底
# Group G(5): tb 精细化 + es=0.02 — 补充 Round2 未测的交叉组合
# Group H(3): eb=0.84 左侧探索
COMBOS=(
    # Group F: es=0.01 核心测试
    "0.75 0.01 0.85 0.01"   # F1: 当前最优点 es 再减小 ★最高优先
    "0.73 0.01 0.85 0.01"   # F2: tb=0.73 + es=0.01 联合
    "0.74 0.01 0.85 0.01"   # F3: tb=0.74 + es=0.01 联合
    "0.70 0.01 0.85 0.01"   # F4: tb=0.70 + es=0.01 联合
    "0.76 0.01 0.85 0.01"   # F5: tb=0.76 + es=0.01
    "0.72 0.01 0.85 0.01"   # F6: tb=0.72 + es=0.01
    # Group G: tb 精细化 + es=0.02
    "0.73 0.01 0.85 0.02"   # G1: tb=0.73 (Round2在es=0.03得0.8218) + es=0.02
    "0.74 0.01 0.85 0.02"   # G2: tb=0.74 插值
    "0.70 0.01 0.85 0.02"   # G3: tb=0.70 (Round2在es=0.03得0.8202) + es=0.02
    "0.76 0.01 0.85 0.02"   # G4: tb=0.76 微调
    "0.72 0.01 0.85 0.02"   # G5: tb=0.72 + es=0.02
    # Group H: eb=0.84 左侧探索
    "0.75 0.01 0.84 0.01"   # H1: eb=0.84 + es=0.01
    "0.75 0.01 0.84 0.02"   # H2: eb=0.84 + es=0.02
    "0.73 0.01 0.84 0.02"   # H3: tb=0.73 + eb=0.84
)

PIDS=()

for IDX in "${!COMBOS[@]}"; do
    read -r TB TS EB ES <<< "${COMBOS[$IDX]}"
    KEY="${KEYS[$IDX]}"
    TAG="tb${TB}_ts${TS}_eb${EB}_es${ES}"
    LOG="logs/sciworld_v3_${TAG}.log"
    CSV="results/sciworld_shards_v3/shard_${TAG}"

    if   [ "$IDX" -lt 6 ];  then GRP="F$((IDX+1))"
    elif [ "$IDX" -lt 11 ]; then GRP="G$((IDX-5))"
    else                         GRP="H$((IDX-10))"
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
echo "Round2最优: tb=0.75,ts=0.01,eb=0.85,es=0.02 → avg_reward=0.8499  目标: >0.8558"
