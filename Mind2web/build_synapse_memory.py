"""
build_synapse_memory.py  —  Mind2Web Synapse 记忆库离线构建脚本
=============================================================
两种建库方式
------------
1. 从 Mind2Web 训练集直接构建（与 /data/REDACTED_USER/Synapse 完全一致的方式）
   python build_synapse_memory.py \
       --source train \
       --data-dir data \
       --memory-path storage/synapse_memory

2. 从已有 PRTree 轨迹构建（online learning 后的 exemplar 会更新当前任务分布）
   python build_synapse_memory.py \
       --source trajectories \
       --traj-dir test_task/online_dual_memory \
       --memory-path storage/synapse_memory

参数
----
--source        构建来源：train（训练集）或 trajectories（已有轨迹）
--data-dir      Mind2Web 数据目录（仅 source=train 时需要）
--traj-dir      轨迹目录（仅 source=trajectories 时需要）
--memory-path   Synapse 记忆库保存目录
--top-k         每个 sample 检索的 exemplar 数（仅 source=train 时有效，同 Synapse 原版）
--max-exemplars 最多使用多少条（source=trajectories 时有效）
--success-only  只使用成功轨迹（默认 True，source=trajectories 时有效）
--embed-model   OpenAI embedding 模型（默认 text-embedding-ada-002）
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))  # PRTree root → synapse_memory
sys.path.insert(0, str(Path(__file__).parent))         # Mind2web dir → mind2web_utils


# ---------------------------------------------------------------------------
# 来源 1：Mind2Web 训练集（对齐 Synapse 原版）
# ---------------------------------------------------------------------------

def build_from_train(data_dir: str, memory_path: str, top_k: int, embed_model: str):
    """
    完全对齐 /data/REDACTED_USER/Synapse/synapse/memory/mind2web/build_memory.py 的逻辑：
    - 用 (website/domain/subdomain/task) 作 specifier
    - 把整条轨迹的 (obs, action) 对构成 exemplar messages
    """
    from mind2web_utils import load_json_data, get_top_k_obs, get_target_obs_and_act, add_scores
    import pickle

    score_path = os.path.join(data_dir, "scores_all_data.pkl")
    samples = load_json_data(data_dir, "train")
    logger.info(f"Loaded {len(samples)} train samples from {data_dir}")

    if os.path.exists(score_path):
        logger.info(f"Attaching candidate scores from {score_path}")
        with open(score_path, "rb") as f:
            candidate_results = pickle.load(f)
        candidate_scores = candidate_results["scores"]
        candidate_ranks = candidate_results["ranks"]
        for sample in samples:
            for s in sample["actions"]:
                sample_id = f"{sample['annotation_id']}_{s['action_uid']}"
                for cands in [s["pos_candidates"], s["neg_candidates"]]:
                    for c in cands:
                        cid = c["backend_node_id"]
                        if sample_id in candidate_scores and cid in candidate_scores[sample_id]:
                            c["score"] = candidate_scores[sample_id][cid]
                            c["rank"] = candidate_ranks[sample_id][cid]
    else:
        logger.warning(f"scores_all_data.pkl not found at {score_path}, ranks may be missing.")

    specifiers = []
    exemplars = []

    for sample in samples:
        website = sample.get("website", "")
        domain = sample.get("domain", "")
        subdomain = sample.get("subdomain", "")
        goal = sample.get("confirmed_task", "")
        specifier = (
            f"Website: {website}\nDomain: {domain}\n"
            f"Subdomain: {subdomain}\nTask: {goal}"
        )

        prev_obs = []
        prev_actions = []
        for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
            _, target_act = get_target_obs_and_act(s)
            target_obs, _ = get_top_k_obs(s, top_k)
            if len(prev_obs) > 0:
                prev_obs.append("Observation: `" + target_obs + "`")
            else:
                query = f"Task: {goal}\nTrajectory:\n"
                prev_obs.append(query + "Observation: `" + target_obs + "`")
            prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")

        message = []
        for o, a in zip(prev_obs, prev_actions):
            message.append({"role": "user", "content": o})
            message.append({"role": "assistant", "content": a})

        specifiers.append(specifier)
        exemplars.append(message)

    logger.info(f"Collected {len(specifiers)} exemplars from training set")

    from memory.synapse.synapse_memory import SynapseMemoryStore
    store = SynapseMemoryStore.build(
        memory_path=memory_path,
        specifiers=specifiers,
        exemplars=exemplars,
        embed_model=embed_model,
    )
    logger.info(f"✅ Done. {store}")


# ---------------------------------------------------------------------------
# 来源 2：已有 PRTree 轨迹目录
# ---------------------------------------------------------------------------

def build_from_trajectories(traj_dir: str, memory_path: str, success_only: bool,
                             max_exemplars: int, embed_model: str):
    traj_path = Path(traj_dir)
    if not traj_path.exists():
        logger.error(f"Trajectory directory not found: {traj_dir}")
        sys.exit(1)

    files = sorted(traj_path.glob("*.json"))
    logger.info(f"Found {len(files)} trajectory files")

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

        # 反推 specifier：从 summary 或者第一条 user message 里提取
        traj_list = summary.get("trajectory", [])
        task_text = traj_list[0] if traj_list else ""
        if task_text.startswith("Task: "):
            task_text = task_text[6:]

        specifier = f"Task: {task_text[:400]}"

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

    from memory.synapse.synapse_memory import SynapseMemoryStore
    store = SynapseMemoryStore.build(
        memory_path=memory_path,
        specifiers=specifiers,
        exemplars=exemplars,
        embed_model=embed_model,
    )
    logger.info(f"✅ Done. {store}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Synapse memory for Mind2Web")
    parser.add_argument("--source", type=str, choices=["train", "trajectories"],
                        default="train", help="Data source for building memory")
    # train source
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Mind2Web data directory (for source=train)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Top-k elements per step (for source=train)")
    # trajectories source
    parser.add_argument("--traj-dir", type=str, default=None,
                        help="Trajectory directory (for source=trajectories)")
    parser.add_argument("--max-exemplars", type=int, default=500)
    parser.add_argument("--success-only", action="store_true", default=True)
    parser.add_argument("--no-success-only", dest="success_only", action="store_false")
    # common
    parser.add_argument("--memory-path", type=str, default="storage/synapse_memory")
    parser.add_argument("--embed-model", type=str, default="text-embedding-ada-002")
    args = parser.parse_args()

    if args.source == "train":
        build_from_train(args.data_dir, args.memory_path, args.top_k, args.embed_model)
    else:
        if not args.traj_dir:
            parser.error("--traj-dir is required for source=trajectories")
        build_from_trajectories(
            args.traj_dir, args.memory_path,
            args.success_only, args.max_exemplars, args.embed_model,
        )


if __name__ == "__main__":
    main()
