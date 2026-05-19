"""
build_synapse_memory.py  —  ScienceWorld Synapse 记忆库离线构建脚本
=================================================================
用途
----
遍历 ScienceWorld 轨迹目录，把每条成功轨迹提取为 exemplar，
用 "{task_name} (var={variation_idx}): {task_description}" 作为 specifier 建库。

运行方式
--------
cd /hdd/REDACTED_USER/DeltaMem/ScienceWorld
python build_synapse_memory.py \
    --traj-dir dev/online_dual_memory_Qwen3-14B \
    --memory-path storage/synapse_memory \
    --max-exemplars 500

轨迹 JSON 格式（来自 agent_sciworld_dual.py）：
[
  {"role": "user",      "content": "...system prompt + task..."},
  {"role": "assistant", "content": "..."},
  ...
  {
    "success": bool,
    "reward": float,
    "steps": int,
    "task_name": str,
    "task_id": str,
    "variation_idx": int,
    "trajectory": [...],
    ...
  }
]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_trajectories(traj_dir: str, success_only: bool, max_exemplars: int):
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
        if not isinstance(summary, dict) or "task_name" not in summary:
            continue

        success = summary.get("success", False)
        if success_only and not success:
            continue

        task_name = summary.get("task_name", "unknown")
        variation_idx = summary.get("variation_idx", 0)
        reward = summary.get("reward", 0.0)

        # specifier：任务类型 + 变体 + 初始观察（前 400 chars）
        traj_list = summary.get("trajectory", [])
        first_obs = traj_list[0] if traj_list else ""
        specifier = (
            f"Task: {task_name} (variation={variation_idx})\n"
            f"Reward: {reward:.3f}\n"
            f"Initial observation: {first_obs[:300]}"
        )

        msg_list = [m for m in data if isinstance(m, dict) and "role" in m]
        if not msg_list:
            continue

        specifiers.append(specifier)
        exemplars.append(msg_list)

        if len(specifiers) >= max_exemplars:
            break

    logger.info(
        f"Collected {len(specifiers)} exemplars "
        f"({'success only' if success_only else 'all'}) "
        f"from {len(files)} trajectories"
    )
    return specifiers, exemplars


def main():
    parser = argparse.ArgumentParser(description="Build Synapse memory for ScienceWorld")
    parser.add_argument("--traj-dir", type=str, required=True)
    parser.add_argument("--memory-path", type=str, default="storage/synapse_memory")
    parser.add_argument("--max-exemplars", type=int, default=500)
    parser.add_argument("--success-only", action="store_true", default=True)
    parser.add_argument("--no-success-only", dest="success_only", action="store_false")
    parser.add_argument("--embed-model", type=str, default="text-embedding-ada-002")
    args = parser.parse_args()

    from memory.synapse.synapse_memory import SynapseMemoryStore

    specifiers, exemplars = load_trajectories(
        args.traj_dir, args.success_only, args.max_exemplars
    )
    if not specifiers:
        logger.error("No exemplars collected.")
        sys.exit(1)

    store = SynapseMemoryStore.build(
        memory_path=args.memory_path,
        specifiers=specifiers,
        exemplars=exemplars,
        embed_model=args.embed_model,
    )
    logger.info(f"✅ Done. {store}")


if __name__ == "__main__":
    main()
