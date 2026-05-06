export API_KEY='sk-uQ3Q4igYxnqjrEAcXfatMws18iO180Vn8dFRSYPcpmj3Zpc2'
start_port=8010
model_name="Qwen3-14B"
split="test"
benchmark="sciworld"
export BASE_URL="http://localhost:$start_port/v1"

# 路径规则：storage/{split}-{benchmark}-{model}-{memory}
#           trajectories/{split}-{benchmark}-{model}-{memory}

# ---- 1. PRTree Memory（默认，边跑边学） ----
# memory="prtree"
# nohup python -u run_sciworld.py \
#     --mode eval \
#     --model $model_name \
#     --split $split \
#     --memory $memory \
#     --save-memory storage/${split}-${benchmark}-${model_name}-${memory} \
#     --traj-dir trajectories/${split}-${benchmark}-${model_name}-${memory} \
#     > logs/${split}-${benchmark}-${model_name}-${memory}.log 2>&1 &

# ---- 2. Baseline：完全无 Memory（cold-start only） ----
# memory="no-memory"
# nohup python -u run_sciworld.py \
#     --mode eval \
#     --model $model_name \
#     --split $split \
#     --memory $memory \
#     --traj-dir trajectories/${split}-${benchmark}-${model_name}-${memory} \
#     > logs/${split}-${benchmark}-${model_name}-${memory}.log 2>&1 &

# ---- 3. Synapse Memory（在线检索 + 在线写回） ----
# 若已有离线建好的库：
#   cd ScienceWorld && python build_synapse_memory.py \
#       --traj-dir <轨迹目录> \
#       --memory-path storage/${split}-${benchmark}-${model_name}-synapse
memory="synapse"
nohup python -u run_sciworld.py \
    --mode eval \
    --model $model_name \
    --split $split \
    --memory $memory \
    --memory-file storage/${split}-${benchmark}-${model_name}-${memory} \
    --traj-dir trajectories/${split}-${benchmark}-${model_name}-${memory} \
    > logs/${split}-${benchmark}-${model_name}-${memory}.log 2>&1 &

# ---- 4. AWM (Autonomous Workflow Memory) ----
# memory="awm"
# nohup python -u run_sciworld.py \
#     --mode eval \
#     --model $model_name \
#     --split $split \
#     --memory $memory \
#     --memory-file storage/${split}-${benchmark}-${model_name}-${memory} \
#     --save-interval 10 \
#     --traj-dir trajectories/${split}-${benchmark}-${model_name}-${memory} \
#     > logs/${split}-${benchmark}-${model_name}-${memory}.log 2>&1 &

mkdir -p logs
echo "Started PID=$!"
