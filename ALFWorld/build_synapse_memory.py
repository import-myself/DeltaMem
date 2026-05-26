"""
build_synapse_memory.py  —  ALFWorld Synapse 记忆库离线构建脚本
=============================================================
用途
----
遍历 ALFWorld 训练集（需要预先完成 PRTree 在线评估并保存轨迹），
把每条成功轨迹的摘要/关键步骤构造成 exemplar，
并用任务描述作为 specifier 建 FAISS 向量索引。

运行方式
--------
cd /path/to/DeltaMem/ALFWorld
python build_synapse_memory.py \
    --traj-dir trajectories/offline_dual \
    --memory-path storage/synapse_memory \
    --max-exemplars 500

参数
----
--traj-dir      已保存轨迹的目录（每条 {idx}.json 对应一个 episode）
--memory-path   Synapse 记忆库保存目录
--max-exemplars 最多使用多少条轨迹（按成功优先）
--success-only  只使用成功轨迹（默认 True）
--embed-model   OpenAI embedding 模型（默认 text-embedding-ada-002）
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 确保 PRTree 根目录在 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_trajectories(traj_dir: str, success_only: bool, max_exemplars: int):
    """
    读取轨迹目录，返回 (specifiers, exemplars) 两个列表。

    trajectory JSON 格式（来自 agent_alfworld_dual.py）：
    [
      {"role": "user",      "content": "..."},   # system prompt / first obs
      {"role": "assistant", "content": "..."},   # action
      ...
      {                                           # summary dict（最后一条）
        "success": bool,
        "steps": int,
        "trajectory": [...],
        "task_node_id": "...",
        "env_node_id": "...",
        ...
      }
    ]
    """
    traj_path = Path(traj_dir)
    if not traj_path.exists():
        logger.error(f"Trajectory directory not found: {traj_dir}")
        sys.exit(1)

    files = sorted(traj_path.glob("*.json"))
    logger.info(f"Found {len(files)} trajectory files in {traj_dir}")

    specifiers = []
    exemplars = []

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Skip {fp}: {e}")
            continue

        if not data:
            continue

        summary = data[-1]
        if not isinstance(summary, dict):
            continue

        success = summary.get("success", False)
        if success_only and not success:
            continue

        # 从 summary.trajectory 提取任务描述（第一条一般是 "Observation: ..."）
        traj_list = summary.get("trajectory", [])
        # 第一条包含任务（格式 "Observation: You are in... Your task is to: ..."）
        task_obs = traj_list[0] if traj_list else ""
        task_instruction = task_obs.replace("Observation: ", "", 1).strip()

        if not task_instruction:
            # 从 system message 里提取
            for msg in data:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    task_instruction = msg.get("content", "")[:300]
                    break

        specifier = task_instruction[:400]  # 截断避免 embedding 过长

        # 构造 exemplar：messages list（去掉最后的 summary dict）
        msg_list = [m for m in data if isinstance(m, dict) and "role" in m]
        if not msg_list:
            continue

        exemplar = msg_list  # 直接用原始对话作为 few-shot exemplar

        specifiers.append(specifier)
        exemplars.append(exemplar)

        if len(specifiers) >= max_exemplars:
            break

    logger.info(
        f"Collected {len(specifiers)} exemplars "
        f"({'success only' if success_only else 'all'}) "
        f"from {len(files)} trajectories"
    )
    return specifiers, exemplars


def main():
    parser = argparse.ArgumentParser(description="Build Synapse memory for ALFWorld")
    parser.add_argument("--traj-dir", type=str, required=True,
                        help="Directory containing trajectory JSON files")
    parser.add_argument("--memory-path", type=str,
                        default="storage/synapse_memory",
                        help="Output directory for Synapse memory store")
    parser.add_argument("--max-exemplars", type=int, default=500,
                        help="Maximum number of exemplars to include")
    parser.add_argument("--success-only", action="store_true", default=True,
                        help="Only use successful trajectories as exemplars")
    parser.add_argument("--no-success-only", dest="success_only", action="store_false",
                        help="Include failed trajectories too")
    parser.add_argument("--embed-model", type=str, default="text-embedding-ada-002",
                        help="OpenAI embedding model")
    args = parser.parse_args()

    from memory.synapse.synapse_memory import SynapseMemoryStore  # noqa: E402

    specifiers, exemplars = load_trajectories(
        args.traj_dir, args.success_only, args.max_exemplars
    )

    if not specifiers:
        logger.error("No exemplars collected. Check --traj-dir and trajectory format.")
        sys.exit(1)

    logger.info(f"Building Synapse memory store at: {args.memory_path}")
    store = SynapseMemoryStore.build(
        memory_path=args.memory_path,
        specifiers=specifiers,
        exemplars=exemplars,
        embed_model=args.embed_model,
    )
    logger.info(f"✅ Done. {store}")


if __name__ == "__main__":
    main()
