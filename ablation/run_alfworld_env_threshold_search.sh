#!/usr/bin/env bash
# ALFWorld env 阈值网格搜索 — 在 task 最优点(tb=0.75,ts=0.01)上探索 env 维度
# env_base {0.80,0.83,0.85,0.88,0.90} x env_step {0.01-0.04} + 1 组 task 探索 = 15 组
# 基准：tb=0.75,ts=0.01,eb=0.85,es=0.03 → SR=84.33% (已知，跳过)
set -euo pipefail

export ALFWORLD_DATA='/hdd/REDACTED_USER/DeltaMem/ALFWorld/data/alfworld'
export DEEPSEEK_BASE_URL='https://api.deepseek.com'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p results/env_search_shards logs

# 15 个 API key，一一对应 15 组实验
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
    "sk-bc219b04b617443bae3504eb6f299b10"
)

# 15 组参数: (task_base task_step env_base env_step)
COMBOS=(
    "0.75 0.01 0.80 0.01"
    "0.75 0.01 0.80 0.02"
    "0.75 0.01 0.80 0.03"
    "0.75 0.01 0.80 0.04"
    "0.75 0.01 0.83 0.02"
    "0.75 0.01 0.83 0.03"
    "0.75 0.01 0.83 0.04"
    "0.75 0.01 0.85 0.01"
    "0.75 0.01 0.85 0.02"
    "0.75 0.01 0.85 0.04"
    "0.75 0.01 0.88 0.02"
    "0.75 0.01 0.88 0.03"
    "0.75 0.01 0.90 0.02"
    "0.75 0.01 0.90 0.03"
    "0.72 0.01 0.85 0.03"
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
    printf "  [%2d] tb=%-4s ts=%-4s eb=%-4s es=%-4s  PID=%-6d  key=...%s\n" \
        "$((IDX+1))" "$TB" "$TS" "$EB" "$ES" "$PID" "${KEY: -6}"
done

echo ""
echo "启动了 ${#PIDS[@]} 个并发进程: ${PIDS[*]}"
echo "监控: watch -n30 'python3 -c \"
import csv,glob
rows=[]
for f in glob.glob(\\\"results/env_search_shards/*_alfworld.csv\\\"):
    [rows.append(r) for r in csv.DictReader(open(f))]
rows.sort(key=lambda r:(-float(r[\\\"success_rate\\\"]),float(r[\\\"avg_steps\\\"])))
print(f\\\"Done: {len(rows)}/15\\\")
[print(f\\\"{float(r[\\\"task_base_threshold\\\"]):>.2f} {float(r[\\\"task_depth_step\\\"]):>.3f} {float(r[\\\"env_base_threshold\\\"]):>.2f} {float(r[\\\"env_depth_step\\\"]):>.3f} SR={float(r[\\\"success_rate\\\"])*100:.2f}%\\\") for r in rows]
\"'"
echo ""
echo "等待全部完成..."
wait "${PIDS[@]}"

# 合并所有 shard CSV
MERGED="results/alfworld_env_threshold_search.csv"
HEADER_WRITTEN=0
> "${MERGED}"
for SHARD in results/env_search_shards/shard_*_alfworld.csv; do
    [ -f "${SHARD}" ] || continue
    if [ "${HEADER_WRITTEN}" -eq 0 ]; then
        cat "${SHARD}" >> "${MERGED}"
        HEADER_WRITTEN=1
    else
        tail -n +2 "${SHARD}" >> "${MERGED}"
    fi
done

echo ""
echo "合并完成 → ${MERGED}"
echo ""
echo "=== 最优参数组合（基准 tb=0.75,ts=0.01,eb=0.85,es=0.03 → 84.33%）==="
python3 << 'PYEOF'
import csv, glob

# 加载本轮结果 + 已知基准
rows = []
for f in glob.glob('results/alfworld_env_threshold_search.csv'):
    with open(f) as fp:
        for r in csv.DictReader(fp):
            rows.append(r)

# 加入已知基准
rows.append({
    'task_base_threshold': '0.75', 'task_depth_step': '0.01',
    'env_base_threshold': '0.85', 'env_depth_step': '0.03',
    'success_rate': '0.843284', 'avg_steps': '12.3582',
    'task_hit_rate': '0.992537', 'env_hit_rate': '0.992537',
})

rows.sort(key=lambda r: (-float(r['success_rate']), float(r.get('avg_steps', 99))))
print(f"{'#':>3}  {'tb':>6}  {'ts':>6}  {'eb':>6}  {'es':>6}  {'SR':>8}  {'Steps':>7}")
print('-' * 60)
for i, r in enumerate(rows, 1):
    marker = ' ← BEST' if i == 1 else (' ← baseline' if r['env_base_threshold']=='0.85' and r['env_depth_step']=='0.03' and r['task_base_threshold']=='0.75' else '')
    print(f"{i:>3}  {float(r['task_base_threshold']):>6.2f}  {float(r['task_depth_step']):>6.3f}  "
          f"{float(r['env_base_threshold']):>6.2f}  {float(r['env_depth_step']):>6.3f}  "
          f"{float(r['success_rate'])*100:>7.2f}%  {float(r.get('avg_steps', 0)):>7.2f}{marker}")
PYEOF
