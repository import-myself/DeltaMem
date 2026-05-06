"""
ScienceWorld Dual-Tree Usage (PR-Tree v1.0)
双树 Agent 运行入口 — ScienceWorld 版本

模式说明：
- train (offline): 离线遍历训练集，建立双树记忆
- eval (online):   在线评估 + 边运行边学习
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

# PRTree root → synapse_memory.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_sciworld_dual import DualTreeSciWorldAgent
from common.llm_client import create_llm_client
from common.trajectory_logger import TrajectoryLogger
from utils import sciworld_monkey_patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================
# 数据加载工具
# =========================================================

def load_task_indices(split: str):
    """加载任务索引列表，返回 [(task_name, variation_idx), ...]"""
    split_file = {
        "train": "data/sciworld/train_indices.json",
        "dev":   "data/sciworld/dev_indices.json",
        "test":  "data/sciworld/test_indices.json",
    }
    if split not in split_file:
        raise ValueError(f"Unknown split: {split}. Choose from train/dev/test.")
    with open(split_file[split], "r") as f:
        task_idxs = json.load(f)
    logger.info(f"Loaded {len(task_idxs)} tasks for split='{split}'")
    return task_idxs


def load_sciworld_env():
    """初始化 ScienceWorldEnv 实例"""
    from scienceworld import ScienceWorldEnv
    sciworld_monkey_patch()
    env = ScienceWorldEnv()
    logger.info("ScienceWorldEnv initialized.")
    return env


# =========================================================
# Offline Training Mode
# =========================================================

def run_offline_training(args):
    logger.info("=" * 80)
    logger.info("OFFLINE TRAINING MODE: Building Dual Memory Trees (PR-Tree SciWorld v1.0)")
    logger.info("=" * 80)

    task_idxs = load_task_indices("train")
    env = load_sciworld_env()
    llm_client = create_llm_client(args.model)

    agent = DualTreeSciWorldAgent(
        agent_name="DualTreeOfflineBuilder",
        llm_client=llm_client,
        icl_num=args.icl_num,
        icl_data_path=args.icl_path,
    )

    if args.load_memory:
        task_fp = args.load_memory.replace(".json", "") + "_task.json"
        if Path(task_fp).exists():
            agent.load_memory(args.load_memory)
            stats = agent.get_memory_stats()
            logger.info(
                f"📥 Loaded memory. Task nodes: {stats['task_tree_nodes']}, "
                f"Env nodes: {stats['env_tree_nodes']}"
            )

    trajectory_logger = TrajectoryLogger(save_dir=args.traj_dir or "trajectories/offline_dual")
    episode_results = []
    n_tasks = min(args.max_episodes, len(task_idxs))

    for episode_idx in range(n_tasks):
        task_name, variation_idx = task_idxs[episode_idx]
        logger.info(f"\n--- Episode {episode_idx + 1}/{n_tasks}: {task_name} var={variation_idx} ---")

        messages = agent.run_episode(
            env=env,
            task_name=task_name,
            variation_idx=variation_idx,
        )
        result = messages[-1]
        episode_results.append(result["success"])

        trajectory_logger.log_episode(
            episode_idx=episode_idx,
            task_instruction=f"{task_name} (var={variation_idx})",
            task_type=task_name,
            result=result,
            split="train",
            mode="offline",
        )

        if (episode_idx + 1) % args.save_interval == 0:
            save_path = args.save_memory or "storage/prtree_sciworld_offline"
            agent.save_memory(save_path)
            stats = agent.get_memory_stats()
            success_rate = sum(episode_results) / len(episode_results)
            logger.info(
                f"\n📊 Checkpoint {episode_idx + 1}/{n_tasks}: "
                f"SR={success_rate:.2%}, "
                f"TaskNodes={stats['task_tree_nodes']}, "
                f"EnvNodes={stats['env_tree_nodes']}"
            )

    save_path = args.save_memory or "storage/prtree_sciworld_offline"
    agent.save_memory(save_path)
    trajectory_logger.save(mode="offline", split="train")

    stats = agent.get_memory_stats()
    logger.info("\n" + "=" * 80)
    logger.info("OFFLINE TRAINING COMPLETED")
    logger.info(f"Task Tree Nodes: {stats['task_tree_nodes']}")
    logger.info(f"Env Tree Nodes:  {stats['env_tree_nodes']}")
    logger.info(f"Total Nodes:     {stats['total_nodes']}")
    logger.info("=" * 80)


# =========================================================
# Online Evaluation Mode
# =========================================================

def run_online_evaluation(args):
    logger.info("=" * 80)
    logger.info("ONLINE EVALUATION MODE: Dual Tree Testing & Learning (PR-Tree SciWorld v1.0)")
    logger.info("=" * 80)

    task_idxs = load_task_indices(args.split)
    env = load_sciworld_env()
    llm_client = create_llm_client(args.model)

    # 轨迹输出目录（优先使用 --traj-dir 参数，否则默认 {split}/online_dual_memory_{model}/）
    traj_dir = args.traj_dir or os.path.join(args.split, f"online_dual_memory_{args.model.replace('/', '_')}")
    os.makedirs(traj_dir, exist_ok=True)

    agent = DualTreeSciWorldAgent(
        agent_name="DualTreeOnlineAgent",
        llm_client=llm_client,
        icl_num=args.icl_num,
        icl_data_path=args.icl_path,
    )

    # ---- Memory backend selection ----
    no_mem_flag   = (args.memory == "no-memory") or args.no_memory
    synapse_store = None
    awm_store     = None

    if no_mem_flag:
        logger.info("🚫 PRTree memory DISABLED (baseline mode)")
    elif args.memory == "synapse":
        from memory.synapse.synapse_memory import SynapseMemoryStore
        mem_path = args.memory_file or "storage/synapse_memory"
        synapse_store = SynapseMemoryStore(memory_path=mem_path)
        logger.info(f"🧠 Synapse memory loaded: {synapse_store}")
    elif args.memory == "file" and args.memory_file:
        logger.info(f"🔌 File memory mode: {args.memory_file}")
    elif args.memory == "awm":
        from memory.awm.awm_memory import AWMMemory
        awm_path  = args.memory_file or "storage/awm_memory"
        awm_store = AWMMemory(memory_path=awm_path, llm_client=llm_client, benchmark="sciworld")
        logger.info(f"🔧 AWM memory loaded: {awm_store}")

    if args.load_memory:
        task_fp = args.load_memory.replace(".json", "") + "_task.json"
        env_fp = args.load_memory.replace(".json", "") + "_env.json"
        if Path(task_fp).exists() or Path(env_fp).exists():
            agent.load_memory(args.load_memory)
            stats = agent.get_memory_stats()
            logger.info(
                f"📥 Loaded dual memory. Task: {stats['task_tree_nodes']} nodes, "
                f"Env: {stats['env_tree_nodes']} nodes"
            )
        else:
            logger.warning(f"⚠️  Memory files not found at {args.load_memory}")
            logger.info("🌱 Starting from scratch (cold-start)")

    results = []
    task_mem_hit = 0
    env_mem_hit = 0
    n_tasks = len(task_idxs)

    for episode_idx in range(n_tasks):
        task_name, variation_idx = task_idxs[episode_idx]
        logger.info(f"\n--- Episode {episode_idx + 1}/{n_tasks}: {task_name} var={variation_idx} ---")

        # --- Synapse 检索 ---
        ext_mem = None
        if synapse_store is not None:
            synapse_query = f"Task: {task_name} (variation={variation_idx})"
            ext_mem = synapse_store.retrieve_memory_str(synapse_query)

        # --- AWM workflow 注入 ---
        if awm_store is not None:
            wf_str = awm_store.get_workflow(task_name)
            if wf_str:
                ext_mem = wf_str


        messages = agent.run_episode(
            env=env,
            task_name=task_name,
            variation_idx=variation_idx,
            no_memory=no_mem_flag or (awm_store is not None),  # AWM 模式也禁止 PRTree 操作
            external_memory_str=ext_mem,
        )
        result = messages[-1]
        results.append(result)

        # --- Synapse 在线写入（仅成功轨迹）---
        if synapse_store is not None and not no_mem_flag and result.get('success', False):
            msg_list = [m for m in messages if isinstance(m, dict) and "role" in m]
            synapse_specifier = (
                f"Task: {task_name} (variation={variation_idx})\n"
                f"Success: {result.get('success', False)}\n"
                f"Reward: {result.get('reward', 0.0):.3f}"
            )
            synapse_store.add_exemplar(specifier=synapse_specifier, exemplar=msg_list)

        # --- AWM 诱导 ---
        if awm_store is not None and not no_mem_flag:
            awm_store.induce_and_update(
                task_type=task_name,
                task_description=f"{task_name} (variation={variation_idx})",
                trajectory=result.get('trajectory', []),
                success=result.get('success', False),
            )

        if result.get("task_memory_used", False):
            task_mem_hit += 1
        if result.get("env_memory_used", False):
            env_mem_hit += 1

        # 保存完整 messages（与 ALFWorld_New 保持一致）
        traj_filename = f"{episode_idx}_{task_name}_var{variation_idx}.json"
        traj_path = os.path.join(traj_dir, traj_filename)
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)

        # 定期打印进度
        if (episode_idx + 1) % 5 == 0:
            current_sr = sum(r["success"] for r in results) / len(results)
            current_avg_reward = sum(r["reward"] for r in results) / len(results)
            stats = agent.get_memory_stats()
            logger.info(
                f"📈 Ep {episode_idx + 1}: "
                f"SR={current_sr:.2%}, "
                f"AvgReward={current_avg_reward:.3f}, "
                f"TaskHit={task_mem_hit}/{len(results)}, "
                f"EnvHit={env_mem_hit}/{len(results)}, "
                f"TaskNodes={stats['task_tree_nodes']}, "
                f"EnvNodes={stats['env_tree_nodes']}"
            )

        # 定期保存记忆
        if not no_mem_flag and args.save_memory and (episode_idx + 1) % args.save_interval == 0:
            agent.save_memory(args.save_memory)
        if synapse_store is not None and (episode_idx + 1) % args.save_interval == 0:
            synapse_store.save()
        if awm_store is not None and (episode_idx + 1) % args.save_interval == 0:
            awm_store.save()

    # 最终保存
    if not no_mem_flag:
        save_path = args.save_memory or "storage/prtree_sciworld_online"
        agent.save_memory(save_path)
    if synapse_store is not None:
        synapse_store.save()
        logger.info(f"💾 Synapse store saved: {synapse_store}")
    if awm_store is not None:
        awm_store.save()
        logger.info(f"💾 AWM store saved: {awm_store}")

    # ========== 最终统计 ==========
    final_stats = agent.get_memory_stats()
    success_rate = sum(r["success"] for r in results) / len(results)
    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)

    with_any_mem = [r for r in results if r.get("memory_used", False)]
    without_mem = [r for r in results if not r.get("memory_used", False)]
    sr_with = (
        sum(r["success"] for r in with_any_mem) / len(with_any_mem)
        if with_any_mem else 0.0
    )
    sr_without = (
        sum(r["success"] for r in without_mem) / len(without_mem)
        if without_mem else 0.0
    )
    reward_with = (
        sum(r["reward"] for r in with_any_mem) / len(with_any_mem)
        if with_any_mem else 0.0
    )
    reward_without = (
        sum(r["reward"] for r in without_mem) / len(without_mem)
        if without_mem else 0.0
    )

    logger.info("\n" + "=" * 80)
    logger.info("DUAL PR-TREE v1.0 SCIENCEWORLD EVALUATION REPORT")
    logger.info("=" * 80)
    logger.info(f"Episodes: {len(results)} | Split: {args.split}")
    logger.info(f"Final Success Rate: {success_rate:.2%}")
    logger.info(f"Average Reward:     {avg_reward:.4f}")
    logger.info(f"Average Steps:      {avg_steps:.2f}")

    logger.info(f"\n�� Memory Impact Analysis:")
    logger.info(f"  Task Tree Hit Rate: {task_mem_hit / len(results):.2%}")
    logger.info(f"  Env Tree Hit Rate:  {env_mem_hit / len(results):.2%}")
    logger.info(f"  Any Memory Hit:     {len(with_any_mem) / len(results):.2%}")
    logger.info(f"  SR  (Memory Hit):   {sr_with:.2%}  (n={len(with_any_mem)})")
    logger.info(f"  SR  (Zero-shot):    {sr_without:.2%}  (n={len(without_mem)})")
    logger.info(f"  Rwd (Memory Hit):   {reward_with:.4f}")
    logger.info(f"  Rwd (Zero-shot):    {reward_without:.4f}")
    if with_any_mem and without_mem:
        logger.info(f"  SR Improvement:     {sr_with - sr_without:+.2%}")
        logger.info(f"  Rwd Improvement:    {reward_with - reward_without:+.4f}")

    logger.info(f"\n🌳 Dual Tree Topology (Final):")
    logger.info(f"  Task Tree Nodes: {final_stats['task_tree_nodes']}")
    logger.info(f"  Env Tree Nodes:  {final_stats['env_tree_nodes']}")
    logger.info(f"  Total Nodes:     {final_stats['total_nodes']}")
    logger.info(f"  Max Depth:       {final_stats['max_depth']}")

    # 按任务类型细分
    task_stats = {}
    for r in results:
        tt = r.get("task_name", "unknown")
        if tt not in task_stats:
            task_stats[tt] = {"total": 0, "succ": 0, "reward": 0.0}
        task_stats[tt]["total"] += 1
        if r["success"]:
            task_stats[tt]["succ"] += 1
        task_stats[tt]["reward"] += r["reward"]

    logger.info(f"\n📋 Task Breakdown:")
    for tt, s in sorted(task_stats.items()):
        avg_r = s["reward"] / s["total"]
        logger.info(
            f"  {tt:50s}: {s['succ']}/{s['total']} "
            f"({s['succ']/s['total']:.2%})  AvgRwd={avg_r:.3f}"
        )

    logger.info("=" * 80)


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dual PR-Tree v1.0 Memory Framework for ScienceWorld"
    )
    parser.add_argument(
        "--mode", type=str, choices=["train", "eval"], required=True,
        help="train: offline build memory; eval: online evaluate+learn"
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "dev", "test"], default="test",
        help="Data split to use"
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--icl-num", type=int, default=1, help="Number of ICL examples")
    parser.add_argument(
        "--icl-path", type=str,
        default="data/sciworld_icl.json",
        help="Path to ICL examples JSON file"
    )
    parser.add_argument("--max-episodes", type=int, default=1000,
                        help="Max episodes for offline training")
    parser.add_argument("--load-memory", type=str, help="Path prefix to load dual memory")
    parser.add_argument("--save-memory", type=str, help="Path prefix to save dual memory")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save memory every N episodes")
    parser.add_argument("--no-memory", action="store_true",
                        help="Disable PRTree memory (baseline mode): skip retrieval, reflection and tree update")
    parser.add_argument("--memory", type=str, choices=["no-memory","prtree","synapse","file","awm"], default="prtree",
                        help="Select memory backend: no-memory|prtree|synapse|file")
    parser.add_argument("--memory-file", type=str, default=None,
                        help="Optional path for memory backend (file path or synapse root)")
    parser.add_argument("--traj-dir", type=str, default=None,
                        help="Directory to save trajectories (overrides default path)")

    args = parser.parse_args()

    Path("storage").mkdir(exist_ok=True)
    Path("trajectories").mkdir(exist_ok=True)

    if args.mode == "train":
        run_offline_training(args)
    elif args.mode == "eval":
        run_online_evaluation(args)


if __name__ == "__main__":
    main()
