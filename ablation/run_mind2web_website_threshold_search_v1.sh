#!/usr/bin/env bash
# Mind2Web test_website prtree 阈值网格搜索 Round 1
# 基准: tb=0.75/ts=0.03/eb=0.92/es=0.01 → AF1=0.4359 (test_website, 177条)
#
# 策略:
#   Task 维度: ts=0.01 (借鉴 ALFWorld/SciWorld 经验，细步长更优),
#              tb ∈ {0.70, 0.75, 0.80}
#   Website/Env 维度: es=0.01 (固定), eb ∈ {0.88, 0.92, 0.95}
#   → 3×3 = 9 组联合搜索，覆盖核心参数空间
#
# 组别划分:
#   Group A (3): tb=0.75 (基准 task base), eb 三点扫描
#   Group B (3): tb=0.70 (低 task base, 提高召回)
#   Group C (3): tb=0.80 (高 task base, 提高精度)
set -euo pipefail

export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p results/mind2web_website_v1 logs

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
)

# 9 组: (task_base task_step env_base env_step)
# Group A: tb=0.75/ts=0.01, eb ∈ {0.88, 0.92, 0.95}
# Group B: tb=0.70/ts=0.01, eb ∈ {0.88, 0.92, 0.95}
# Group C: tb=0.80/ts=0.01, eb ∈ {0.88, 0.92, 0.95}
COMBOS=(
    # Group A: 基准 tb, 细步长 ts, eb 三点
    "0.75 0.01 0.88 0.01"   # A1: eb 下移探索
    "0.75 0.01 0.92 0.01"   # A2: tb=0.75+ts=0.01 改进版 (vs 基准 ts=0.03)
    "0.75 0.01 0.95 0.01"   # A3: eb 高精度
    # Group B: 低 task base, 提高任务树召回率
    "0.70 0.01 0.88 0.01"   # B1
    "0.70 0.01 0.92 0.01"   # B2
    "0.70 0.01 0.95 0.01"   # B3
    # Group C: 高 task base, 任务树更严格匹配
    "0.80 0.01 0.88 0.01"   # C1
    "0.80 0.01 0.92 0.01"   # C2
    "0.80 0.01 0.95 0.01"   # C3
)

NUM_KEYS=${#KEYS[@]}
PIDS=()

for IDX in "${!COMBOS[@]}"; do
    read -r TB TS EB ES <<< "${COMBOS[$IDX]}"
    KEY="${KEYS[$((IDX % NUM_KEYS))]}"
    TAG="tb${TB}_ts${TS}_eb${EB}_es${ES}"
    LOG="logs/mind2web_website_v1_${TAG}.log"
    CSV="results/mind2web_website_v1/shard_${TAG}"

    if   [ "$IDX" -lt 3 ]; then GRP="A$((IDX+1))"
    elif [ "$IDX" -lt 6 ]; then GRP="B$((IDX-2))"
    else                         GRP="C$((IDX-5))"
    fi

    DEEPSEEK_API_KEY="${KEY}" nohup python -u run_joint_threshold_search.py \
        --benchmark              mind2web \
        --model                  deepseek-v4-flash \
        --mind2web-split         test_website \
        --task-base-thresholds   "${TB}" \
        --task-depth-steps       "${TS}" \
        --env-base-thresholds    "${EB}" \
        --env-depth-steps        "${ES}" \
        --output-csv             "${CSV}" \
        > "${LOG}" 2>&1 &

    PID=$!
    PIDS+=("$PID")
    printf "  [%s] %s  tb=%.2f ts=%.2f eb=%.2f es=%.2f  PID=%d  key=...%s\n" \
        "$GRP" "$TAG" "$TB" "$TS" "$EB" "$ES" "$PID" "${KEY: -6}"
done

echo ""
echo "启动了 ${#PIDS[@]} 个进程: ${PIDS[*]}"
echo ""
echo "查看日志示例:"
echo "  tail -f logs/mind2web_website_v1_tb0.75_ts0.01_eb0.88_es0.01.log"
echo ""
echo "分析结果:"
echo "  python run_joint_threshold_search.py --mode analyze \\"
echo "    --benchmark mind2web \\"
echo "    --output-csv results/mind2web_website_v1/shard_tb0.75_ts0.01_eb0.88_es0.01"
