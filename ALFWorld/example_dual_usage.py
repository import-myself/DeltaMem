"""
ALFWorld Dual-Tree Usage (PR-Tree v4.0)
双树 Agent 使用入口

模式说明：
- train (offline): 离线建树
- eval (online):   在线评估 + 学习
"""

import os
import sys
import argparse
import logging
import yaml
import json
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))  # PRTree root for synapse_memory
from agent_alfworld_dual import DualTreeReflectiveAgent
from common.llm_client import create_llm_client
from common.trajectory_logger import TrajectoryLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_alfworld_env(split: str = "train"):
    """加载 ALFWorld 环境"""
    import alfworld
    import alfworld.agents.environment as environment

    if split == "train":
        n_tasks = 3553
    elif split == "eval_in_distribution":
        n_tasks = 140
    elif split == "eval_out_of_distribution":
        n_tasks = 134
    else:
        raise ValueError(f"Unknown split: {split}")

    if 'ALFWORLD_DATA' not in os.environ:
        logger.error("ALFWORLD_DATA environment variable is not set.")
        sys.exit(1)

    path = os.environ['ALFWORLD_DATA']
    with open(os.path.join(path, "base_config.yaml")) as f:
        config = yaml.safe_load(f)

    env = environment.get_environment(config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)

    logger.info(f"ALFWorld environment loaded: split={split}, n_tasks={n_tasks}")
    return env, n_tasks


def run_offline_training(args):
    """Offline 训练模式"""
    logger.info("=" * 80)
    logger.info("OFFLINE TRAINING MODE: Building Dual Memory Trees (PR-Tree v4.0)")
    logger.info("=" * 80)

    env, n_tasks = load_alfworld_env(split="train")
    llm_client = create_llm_client(args.model)

    agent = DualTreeReflectiveAgent(
        agent_name="DualTreeOfflineBuilder",
        llm_client=llm_client,
        icl_num=args.icl_num
    )

    if args.load_memory and Path(args.load_memory + "_task.json").exists():
        agent.load_memory(args.load_memory)
        stats = agent.get_memory_stats()
        logger.info(f"📥 Loaded existing memory. Task nodes: {stats['task_tree_nodes']}, Env nodes: {stats['env_tree_nodes']}")

    trajectory_logger = TrajectoryLogger(save_dir=args.traj_dir or "trajectories/offline_dual")
    episode_results = []

    for episode_idx in range(args.max_episodes):
        obs, info = env.reset()
        task_instruction = "\n".join(obs[0].split("\n\n")[1:])

        game_file = info.get("extra.gamefile", [""])[0]
        name = "/".join(game_file.split("/")[-3:-1])
        task_type = None
        if game_file:
            for prefix in ["pick_and_place", "pick_clean_then_place", "pick_heat_then_place",
                           "pick_cool_then_place", "look_at_obj", "pick_two_obj"]:
                if name.startswith(prefix):
                    task_type = prefix
                    break

        ext_mem = None
        if args.memory in ("file","synapse"):
            ext_mem = external_memory_dict.get(task_instruction, None) or external_memory_dict.get(task_instruction[:80], None)
        no_mem_flag = (args.memory == "no-memory")
        # --- AWM workflow 注入 ---
        if awm_store is not None:
            wf_str = awm_store.get_workflow(task_type or 'unknown')
            if wf_str:
                ext_mem = wf_str

        messages = agent.run_episode(
            task_instruction=task_instruction,
            env=env,
            task_type=task_type,
            max_steps=args.max_steps,
            no_memory=no_mem_flag,
            external_memory_str=ext_mem,
        )

        result = messages[-1]
        episode_results.append(result["success"])

        trajectory_logger.log_episode(
            episode_idx=episode_idx,
            task_instruction=task_instruction,
            task_type=task_type or "unknown",
            result={
                "success": result["success"],
                "steps": result["steps"],
                "task_reflection": result.get("task_reflection", {}),
                "env_reflection": result.get("env_reflection", {}),
                "trajectory": result["trajectory"]
            },
            split="train",
            mode="offline"
        )
        trajectory_logger.save(mode="offline", split="train")

        if (episode_idx + 1) % args.save_interval == 0:
            save_path = args.save_memory or "storage/prtree_dual_offline"
            agent.save_memory(save_path)
            stats = agent.get_memory_stats()
            success_rate = sum(episode_results) / len(episode_results)
            logger.info(f"\n📊 Checkpoint {episode_idx + 1}/{args.max_episodes}:")
            logger.info(f"   Task tree nodes: {stats['task_tree_nodes']}")
            logger.info(f"   Env tree nodes:  {stats['env_tree_nodes']}")
            logger.info(f"   Success rate: {success_rate:.2%}\n")

    save_path = args.save_memory or "storage/prtree_dual_offline"
    agent.save_memory(save_path)

    stats = agent.get_memory_stats()
    logger.info("\n" + "=" * 80)
    logger.info("OFFLINE TRAINING COMPLETED (Dual Tree)")
    logger.info(f"Task Tree Nodes: {stats['task_tree_nodes']}")
    logger.info(f"Env Tree Nodes:  {stats['env_tree_nodes']}")
    logger.info(f"Total Nodes:     {stats['total_nodes']}")
    logger.info("=" * 80)


def run_online_evaluation(args):
    """Online 评估模式"""
    logger.info("=" * 80)
    logger.info("ONLINE EVALUATION MODE: Dual Tree Testing & Learning (PR-Tree v4.0)")
    logger.info("=" * 80)

    env, n_tasks = load_alfworld_env(split=args.split)
    llm_client = create_llm_client(args.model)

    trajectory_dir = args.traj_dir or os.path.join(args.split, "online_dual_memory")
    os.makedirs(trajectory_dir, exist_ok=True)

    agent = DualTreeReflectiveAgent(
        agent_name="DualTreeOnlineAgent",
        llm_client=llm_client,
        icl_num=args.icl_num
    )

    # ---- Memory backend selection ----
    no_mem_flag = (args.memory == "no-memory")
    synapse_store = None
    awm_store = None

    if no_mem_flag:
        logger.info("🚫 PRTree memory DISABLED (baseline mode)")
    elif args.memory == "synapse":
        from memory.synapse.synapse_memory import SynapseMemoryStore
        mem_path = args.memory_file or "storage/synapse_memory"
        synapse_store = SynapseMemoryStore(memory_path=mem_path)
        logger.info(f"🧠 Synapse memory loaded: {synapse_store}")
    elif args.memory == "file":
        logger.info(f"🔌 File memory backend (not used in online eval, use synapse or prtree)")
    elif args.memory == "awm":
        from memory.awm.awm_memory import AWMMemory
        awm_path  = args.memory_file or "storage/awm_memory"
        awm_store = AWMMemory(memory_path=awm_path, llm_client=llm_client, benchmark="alfworld")
        logger.info(f"🔧 AWM memory loaded: {awm_store}")

    if args.load_memory:
        # 尝试加载双树格式
        task_fp = args.load_memory.replace(".json", "") + "_task.json"
        env_fp = args.load_memory.replace(".json", "") + "_env.json"
        if Path(task_fp).exists() or Path(env_fp).exists():
            agent.load_memory(args.load_memory)
            stats = agent.get_memory_stats()
            logger.info(f"📥 Loaded dual memory. Task: {stats['task_tree_nodes']} nodes, Env: {stats['env_tree_nodes']} nodes")
        else:
            logger.warning(f"⚠️  Memory files not found at {args.load_memory}")
            logger.info("🌱 Starting from scratch (cold-start)")

    results = []
    task_mem_hit = 0
    env_mem_hit = 0

    for episode_idx in range(n_tasks):
        obs, info = env.reset()
        task_instruction = "\n".join(obs[0].split("\n\n")[1:])

        game_file = info["extra.gamefile"][0]
        name = "/".join(game_file.split("/")[-3:-1])
        task_type = None
        if game_file:
            for prefix in ["pick_and_place", "pick_clean_then_place", "pick_heat_then_place",
                           "pick_cool_then_place", "look_at_obj", "pick_two_obj"]:
                if name.startswith(prefix):
                    task_type = prefix
                    break

        # --- Synapse memory retrieval ---
        ext_mem = None
        if synapse_store is not None:
            synapse_query = f"TaskType: {task_type or 'unknown'}\n{task_instruction[:350]}"
            ext_mem = synapse_store.retrieve_memory_str(synapse_query)

        # --- AWM workflow 注入 ---
        if awm_store is not None:
            wf_str = awm_store.get_workflow(task_type or 'unknown')
            if wf_str:
                ext_mem = wf_str

        messages = agent.run_episode(
            task_instruction=task_instruction,
            env=env,
            task_type=task_type,
            max_steps=args.max_steps,
            no_memory=no_mem_flag or (awm_store is not None),  # AWM 模式也禁止 PRTree 操作
            external_memory_str=ext_mem,
        )
        result = messages[-1]
        result["task_type"] = task_type
        results.append(result)

        # --- AWM 诱导：成功时抽象 workflow ---
        if awm_store is not None and not no_mem_flag:
            awm_store.induce_and_update(
                task_type=task_type or "unknown",
                task_description=task_instruction[:400],
                trajectory=result.get("trajectory", []),
                success=result.get("success", False),
            )




        # --- Synapse online update: 仅把成功轨迹写入 synapse store ---
        if synapse_store is not None and not no_mem_flag and result.get('success', False):
            msg_list = [m for m in messages if isinstance(m, dict) and "role" in m]
            # specifier 加入 task_type / success 让向量检索更准
            synapse_specifier = (
                f"TaskType: {task_type or 'unknown'}\n"
                f"Success: {result.get('success', False)}\n"
                f"{task_instruction[:350]}"
            )
            synapse_store.add_exemplar(
                specifier=synapse_specifier,
                exemplar=msg_list,
            )

        if result.get("task_memory_used", False):
            task_mem_hit += 1
        if result.get("env_memory_used", False):
            env_mem_hit += 1

        # 保存轨迹
        trajectory_path = os.path.join(trajectory_dir, f"{episode_idx}.json")
        with open(trajectory_path, "w") as f:
            json.dump(messages, f, indent=2)

        if (episode_idx + 1) % 5 == 0:
            current_sr = sum(r["success"] for r in results) / len(results)
            stats = agent.get_memory_stats()
            logger.info(
                f"📈 Ep {episode_idx + 1}: SR={current_sr:.2%}, "
                f"TaskHit={task_mem_hit}/{len(results)}, "
                f"EnvHit={env_mem_hit}/{len(results)}, "
                f"TaskNodes={stats['task_tree_nodes']}, "
                f"EnvNodes={stats['env_tree_nodes']}"
            )

        if args.save_memory and (episode_idx + 1) % args.save_interval == 0:
            agent.save_memory(args.save_memory)
        if synapse_store is not None and (episode_idx + 1) % args.save_interval == 0:
            synapse_store.save()
        if awm_store is not None and (episode_idx + 1) % args.save_interval == 0:
            awm_store.save()

    # 最终保存
    save_path = args.save_memory or "storage/prtree_dual_online"
    agent.save_memory(save_path)
    if synapse_store is not None:
        synapse_store.save()
        logger.info(f"💾 Synapse store saved: {synapse_store}")
    if awm_store is not None:
        awm_store.save()
        logger.info(f"💾 AWM store saved: {awm_store}")

    # === 最终统计 ===
    final_stats = agent.get_memory_stats()
    success_rate = sum(r["success"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)

    with_any_mem = [r for r in results if r.get("memory_used", False)]
    without_mem = [r for r in results if not r.get("memory_used", False)]

    sr_with = sum(r["success"] for r in with_any_mem) / len(with_any_mem) if with_any_mem else 0.0
    sr_without = sum(r["success"] for r in without_mem) / len(without_mem) if without_mem else 0.0

    logger.info("\n" + "=" * 80)
    logger.info("DUAL PR-TREE v4.0 EVALUATION REPORT")
    logger.info("=" * 80)
    logger.info(f"Episodes: {len(results)} | Split: {args.split}")
    logger.info(f"Final Success Rate: {success_rate:.2%}")
    logger.info(f"Average Steps: {avg_steps:.2f}")

    # 命中平均检索深度（只统计实际命中的 episode）
    task_hit_lengths = [r.get("task_retrieval_length", 0) for r in results if r.get("task_memory_used", False)]
    env_hit_lengths  = [r.get("env_retrieval_length", 0)  for r in results if r.get("env_memory_used", False)]
    avg_task_hit_len = sum(task_hit_lengths) / len(task_hit_lengths) if task_hit_lengths else 0.0
    avg_env_hit_len  = sum(env_hit_lengths)  / len(env_hit_lengths)  if env_hit_lengths  else 0.0
    # 全量平均检索深度（未命中计 0）
    avg_task_len_all = sum(r.get("task_retrieval_length", 0) for r in results) / len(results)
    avg_env_len_all  = sum(r.get("env_retrieval_length",  0) for r in results) / len(results)

    logger.info(f"\n🧠 Memory Impact Analysis:")
    logger.info(f"  Task Tree Hit Rate: {task_mem_hit / len(results):.2%}")
    logger.info(f"  Env Tree Hit Rate:  {env_mem_hit / len(results):.2%}")
    logger.info(f"  Any Memory Hit:     {len(with_any_mem) / len(results):.2%}")
    logger.info(f"  SR (Memory Hit):    {sr_with:.2%} (n={len(with_any_mem)})")
    logger.info(f"  SR (Zero-shot):     {sr_without:.2%} (n={len(without_mem)})")
    logger.info(f"\n📏 Retrieval Depth Analysis:")
    logger.info(f"  Avg Task Retrieval Depth (hit only): {avg_task_hit_len:.2f}  (n={len(task_hit_lengths)})")
    logger.info(f"  Avg Env  Retrieval Depth (hit only): {avg_env_hit_len:.2f}  (n={len(env_hit_lengths)})")
    logger.info(f"  Avg Task Retrieval Depth (all eps):  {avg_task_len_all:.2f}")
    logger.info(f"  Avg Env  Retrieval Depth (all eps):  {avg_env_len_all:.2f}")

    if with_any_mem and without_mem:
        logger.info(f"  Improvement:        {sr_with - sr_without:+.2%}")

    logger.info(f"\n🌳 Dual Tree Topology (Final):")
    logger.info(f"  Task Tree Nodes: {final_stats['task_tree_nodes']}")
    logger.info(f"  Env Tree Nodes:  {final_stats['env_tree_nodes']}")
    logger.info(f"  Total Nodes:     {final_stats['total_nodes']}")
    logger.info(f"  Max Depth:       {final_stats['max_depth']}")

    task_stats = {}
    for r in results:
        tt = r.get("task_type", "unknown")
        if tt not in task_stats:
            task_stats[tt] = {"total": 0, "succ": 0}
        task_stats[tt]["total"] += 1
        if r["success"]:
            task_stats[tt]["succ"] += 1

    logger.info(f"\n📋 Task Breakdown:")
    for tt, s in sorted(task_stats.items()):
        logger.info(f"  {tt:25s}: {s['succ']}/{s['total']} ({s['succ'] / s['total']:.2%})")

    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Dual PR-Tree v4.0 Memory Framework for ALFWorld"
    )

    parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True)
    parser.add_argument("--split", type=str,
                        choices=["eval_in_distribution", "eval_out_of_distribution"],
                        default="eval_in_distribution")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--icl-num", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--load-memory", type=str)
    parser.add_argument("--save-memory", type=str)
    parser.add_argument("--save-interval", type=int, default=10)
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
