"""
ScienceWorld Dual-Tree Usage (PR-Tree v1.0)
双树 Agent 运行入口 — ScienceWorld 版本

模式说明：
- train (offline): 离线遍历训练集，建立双树记忆
- eval (online):   在线评估 + 边运行边学习
"""

import os
import sys
import time
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
from common.memory_utils import check_memory_path
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
    """Offline 训练模式 = 在训练集上跑 online evaluation，直接复用同一套逻辑。"""
    args.split = "train"
    run_online_evaluation(args)


# =========================================================
# Online Evaluation Mode
# =========================================================

def _load_sciworld_results(traj_dir: str, task_idxs: list):
    """读取全部轨迹文件并汇总统计，任一文件缺失/损坏则返回 (None, 0, 0)。"""
    results, task_hit, env_hit = [], 0, 0
    for episode_idx, (task_name, variation_idx) in enumerate(task_idxs):
        path = os.path.join(traj_dir, f"{episode_idx}_{task_name}_var{variation_idx}.json")
        try:
            with open(path, encoding="utf-8") as f:
                msgs = json.load(f)
            r = msgs[-1]
            if not isinstance(r, dict) or "success" not in r:
                return None, 0, 0
            results.append(r)
            if r.get("task_memory_used"):
                task_hit += 1
            if r.get("env_memory_used"):
                env_hit += 1
        except Exception:
            return None, 0, 0
    return results, task_hit, env_hit


def run_online_evaluation(args):
    # 轨迹输出目录（优先使用 --traj-dir 参数，否则默认 {split}/online_dual_memory_{model}/）
    traj_dir = args.traj_dir or os.path.join(args.split, f"online_dual_memory_{args.model.replace('/', '_')}")
    os.makedirs(traj_dir, exist_ok=True)
    traj_dir_abs = os.path.abspath(traj_dir)
    results_csv = getattr(args, "results_csv", None)
    _key = {"traj_dir": traj_dir_abs}

    # Lock 1: 先查 CSV，已有结果直接跳过
    if results_csv:
        from common.result_logger import check_existing_result
        existing = check_existing_result(results_csv, _key)
        if existing:
            logger.info(f"⏭ [Lock1] 已有记录 (SR={existing.get('success_rate', '?')})，跳过本次运行。")
            return

    _mem_label = {
        "prtree": "DeltaMem (Dual PR-Tree)", "no-memory": "No Memory (Baseline)",
        "synapse": "Synapse", "awm": "AWM", "reasoningbank": "ReasoningBank", "file": "File",
    }.get(args.memory, args.memory)
    logger.info("=" * 80)
    logger.info(f"ONLINE EVALUATION MODE: {_mem_label}")
    logger.info("=" * 80)

    task_idxs = load_task_indices(args.split)
    n_tasks = len(task_idxs)

    # Lock 2a: 轨迹全部存在 → 直接从文件计算，不重跑
    n_done = sum(
        1 for i, (tn, vi) in enumerate(task_idxs)
        if os.path.exists(os.path.join(traj_dir, f"{i}_{tn}_var{vi}.json"))
    )
    if n_done == n_tasks:
        logger.info(f"📂 [Lock2a] 已有 {n_tasks} 条轨迹 → 直接从文件计算结果。")
        all_results, t_hit, e_hit = _load_sciworld_results(traj_dir, task_idxs)
        if all_results is not None and len(all_results) == n_tasks:
            n = len(all_results)
            sr = sum(r["success"] for r in all_results) / n
            avg_r = sum(r.get("reward", 0.0) for r in all_results) / n
            avg_s = sum(r["steps"] for r in all_results) / n
            logger.info(f"✅ [Lock2a] SR={sr:.2%}, AvgReward={avg_r:.4f}, AvgSteps={avg_s:.2f}")
            if results_csv:
                from common.result_logger import append_result
                append_result(results_csv, {
                    **_key,
                    "benchmark": "sciworld", "split": args.split, "model": args.model,
                    "memory": args.memory, "n_episodes": n,
                    "success_rate": round(sr, 6),
                    "avg_reward": round(avg_r, 6),
                    "avg_steps": round(avg_s, 4),
                    "memory_hit_rate": round(sum(1 for r in all_results if r.get("memory_used", False)) / n, 4),
                    "task_hit_rate": round(t_hit / n, 4),
                    "env_hit_rate": round(e_hit / n, 4),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                logger.info(f"📄 结果已写入 {results_csv}")
            return

    env = load_sciworld_env()
    llm_client = create_llm_client(args.model)

    agent = DualTreeSciWorldAgent(
        agent_name="DualTreeOnlineAgent",
        llm_client=llm_client,
        icl_num=args.icl_num,
        icl_data_path=args.icl_path,
    )

    # ---- Memory backend selection ----
    no_mem_flag   = (args.memory == "no-memory") or args.no_memory
    freeze        = getattr(args, "freeze", False)  # 冻结：加载已有库但禁止写入
    synapse_store = None
    awm_store     = None
    rb_store      = None

    if no_mem_flag:
        logger.info("🚫 PRTree memory DISABLED (baseline mode)")
    elif args.memory == "synapse":
        from memory.synapse.synapse_memory import SynapseMemoryStore
        mem_path = args.memory_path or args.memory_file or "storage/synapse_memory"
        if args.resume:
            check_memory_path(mem_path, "synapse")
        synapse_store = SynapseMemoryStore(memory_path=mem_path, load_existing=args.resume,
                                           allow_updates=not freeze)
        logger.info(f"🧠 Synapse memory loaded: {synapse_store}" + (" [FROZEN]" if freeze else ""))
    elif args.memory == "file" and (args.memory_path or args.memory_file):
        logger.info(f"🔌 File memory mode: {args.memory_path or args.memory_file}")
    elif args.memory == "awm":
        from memory.awm.awm_memory import AWMMemory
        awm_path  = args.memory_path or args.memory_file or "storage/awm_memory"
        if args.resume:
            check_memory_path(awm_path, "awm")
        awm_store = AWMMemory(memory_path=awm_path, llm_client=llm_client, benchmark="sciworld",
                              load_existing=args.resume, allow_updates=not freeze)
        logger.info(f"🔧 AWM memory loaded: {awm_store}" + (" [FROZEN]" if freeze else ""))
    elif args.memory == "reasoningbank":
        from memory.reasoningbank.reasoningbank_memory import ReasoningBankMemory
        rb_path = args.memory_path or args.memory_file or "storage/reasoningbank_memory"
        if args.resume:
            check_memory_path(rb_path, "reasoningbank")
        from config import EMBEDDING_MODEL_PATH as _EMB_PATH
        rb_store = ReasoningBankMemory(
            memory_path=rb_path,
            llm_client=llm_client,
            benchmark="sciworld",
            embed_model_path=_EMB_PATH,
            load_existing=args.resume,
            allow_updates=not freeze,
        )
        logger.info(f"📚 ReasoningBank memory loaded: {rb_store}" + (" [FROZEN]" if freeze else ""))

    # prtree: 仅在 --resume 时从 memory_path 加载已有记忆
    _prtree_path = args.memory_path or args.save_memory
    if args.memory == "prtree" and _prtree_path and not args.load_memory and args.resume:
        if check_memory_path(_prtree_path, "prtree"):
            task_fp = _prtree_path.replace(".json", "") + "_task.json"
            if Path(task_fp).exists():
                agent.load_memory(_prtree_path)
                stats = agent.get_memory_stats()
                logger.info(f"📥 Auto-loaded memory from {_prtree_path}. Task: {stats['task_tree_nodes']} nodes, Env: {stats['env_tree_nodes']} nodes")
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

    # 以 memory 的最大 episode_idx 为真正的 resume 点
    memory_last_idx = agent.dual_memory.get_last_committed_episode()
    if memory_last_idx >= 0:
        logger.info(f"🔖 Memory resume point: episode {memory_last_idx} (episodes 0-{memory_last_idx} already in memory)")

    results = []
    task_mem_hit = 0
    env_mem_hit = 0
    mem_hit = 0

    for episode_idx in range(n_tasks):
        task_name, variation_idx = task_idxs[episode_idx]
        logger.info(f"\n--- Episode {episode_idx + 1}/{n_tasks}: {task_name} var={variation_idx} ---")

        # --- 断点续跑：episode_idx <= memory_last_idx 才可安全跳过 ---
        if episode_idx <= memory_last_idx:
            traj_filename = f"{episode_idx}_{task_name}_var{variation_idx}.json"
            traj_path = os.path.join(traj_dir, traj_filename)
            if os.path.exists(traj_path):
                with open(traj_path) as f:
                    saved_messages = json.load(f)
                result = saved_messages[-1]
                if isinstance(result, dict) and "success" in result:
                    results.append(result)
                    if result.get("task_memory_used", False):
                        task_mem_hit += 1
                    if result.get("env_memory_used", False):
                        env_mem_hit += 1
                    if result.get("memory_used", False):
                        mem_hit += 1
                    logger.info(f"⏭  Episode {episode_idx} in memory, skipping (success={result['success']}).")
                    continue

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

        # --- ReasoningBank 检索注入 ---
        if rb_store is not None:
            rb_query = f"Task: {task_name} (variation={variation_idx})"
            rb_mem = rb_store.retrieve_memory_str(rb_query)
            if rb_mem:
                ext_mem = rb_mem

        messages = agent.run_episode(
            env=env,
            task_name=task_name,
            variation_idx=variation_idx,
            no_memory=no_mem_flag,
            no_prtree_update=freeze or (synapse_store is not None) or (awm_store is not None) or (rb_store is not None),
            external_memory_str=ext_mem,
            memory_type=args.memory,
            episode_idx=episode_idx,
        )
        result = messages[-1]
        results.append(result)

        # --- Synapse 在线写入（仅成功轨迹）---
        if synapse_store is not None and not no_mem_flag and result.get('success', False):
            msg_list = [m for m in messages if isinstance(m, dict) and "role" in m]
            # 只保留原始任务描述作为首条 user message，去除 instruction/ICL/memory header
            task_text = f"Task: {task_name} (variation={variation_idx})"
            msg_list = [{"role": "user", "content": task_text}] + msg_list[1:]
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

        # --- ReasoningBank 提炼并存储 ---
        if rb_store is not None and not no_mem_flag:
            rb_store.extract_and_store(
                query=f"{task_name} (variation={variation_idx})",
                trajectory=result.get('trajectory', []),
                success=result.get('success', False),
            )

        if result.get("task_memory_used", False):
            task_mem_hit += 1
        if result.get("env_memory_used", False):
            env_mem_hit += 1
        if result.get("memory_used", False):
            mem_hit += 1

        # 保存完整 messages（与 ALFWorld_New 保持一致）
        traj_filename = f"{episode_idx}_{task_name}_var{variation_idx}.json"
        traj_path = os.path.join(traj_dir, traj_filename)
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)

        # 定期打印进度
        if (episode_idx + 1) % 5 == 0:
            current_sr = sum(r["success"] for r in results) / len(results)
            current_avg_reward = sum(r["reward"] for r in results) / len(results)
            if args.memory == "prtree":
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
            else:
                logger.info(
                    f"📈 Ep {episode_idx + 1}: "
                    f"SR={current_sr:.2%}, "
                    f"AvgReward={current_avg_reward:.3f}, "
                    f"MemHit={mem_hit}/{len(results)}"
                )

        # 定期保存记忆
        _save = args.memory_path or args.save_memory
        if not freeze and args.memory == "prtree" and _save and (episode_idx + 1) % args.save_interval == 0:
            agent.save_memory(_save)
        if not freeze and synapse_store is not None and (episode_idx + 1) % args.save_interval == 0:
            synapse_store.save()
        if not freeze and awm_store is not None and (episode_idx + 1) % args.save_interval == 0:
            awm_store.save()
        if not freeze and rb_store is not None and (episode_idx + 1) % args.save_interval == 0:
            rb_store.save()

    # 最终保存（冻结模式下跳过）
    if not freeze and args.memory == "prtree":
        save_path = args.memory_path or args.save_memory or "storage/prtree_sciworld_online"
        agent.save_memory(save_path)
    if not freeze and synapse_store is not None:
        synapse_store.save()
        logger.info(f"💾 Synapse store saved: {synapse_store}")
    if not freeze and awm_store is not None:
        awm_store.save()
        logger.info(f"💾 AWM store saved: {awm_store}")
    if not freeze and rb_store is not None:
        rb_store.save()
        logger.info(f"💾 ReasoningBank store saved: {rb_store}")

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

    # 写入 CSV 结果（供后续 Lock1 去重使用）
    if results_csv:
        from common.result_logger import append_result
        n = len(results)
        append_result(results_csv, {
            **_key,
            "benchmark": "sciworld", "split": args.split, "model": args.model,
            "memory": args.memory, "n_episodes": n,
            "success_rate": round(success_rate, 6),
            "avg_reward": round(avg_reward, 6),
            "avg_steps": round(avg_steps, 4),
            "memory_hit_rate": round(mem_hit / n, 4),
            "task_hit_rate": round(task_mem_hit / n, 4),
            "env_hit_rate": round(env_mem_hit / n, 4),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        logger.info(f"📄 结果已写入 {results_csv}")


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
    parser.add_argument("--model", type=str, default="deepseek-v4-flash", help="LLM model name")
    parser.add_argument("--icl-num", type=int, default=1, help="Number of ICL examples")
    parser.add_argument(
        "--icl-path", type=str,
        default="data/sciworld_icl.json",
        help="Path to ICL examples JSON file"
    )
    parser.add_argument("--max-episodes", type=int, default=1000,
                        help="Max episodes for offline training")
    parser.add_argument("--memory-path", type=str, default=None,
                        help="Unified memory path for all methods (load+save for prtree; load+save for synapse/awm/rb)")
    parser.add_argument("--resume", action="store_true",
                        help="Load existing memory from --memory-path on startup (default: cold start)")
    parser.add_argument("--load-memory", type=str, help="prtree only: load from a different path (e.g. train→test)")
    parser.add_argument("--save-memory", type=str, help="Deprecated: use --memory-path instead")
    parser.add_argument("--memory-file", type=str, default=None, help="Deprecated: use --memory-path instead")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save memory every N episodes")
    parser.add_argument("--no-memory", action="store_true",
                        help="Disable PRTree memory (baseline mode): skip retrieval, reflection and tree update")
    parser.add_argument("--memory", type=str, choices=["no-memory","prtree","synapse","file","awm","reasoningbank"], default="prtree",
                        help="Select memory backend: no-memory|prtree|synapse|file|awm|reasoningbank")
    parser.add_argument("--traj-dir", type=str, default=None,
                        help="Directory to save trajectories (overrides default path)")
    parser.add_argument("--results-csv", type=str, default=None,
                        help="CSV file to record run results (used for Lock1 deduplication)")
    parser.add_argument("--freeze", action="store_true",
                        help="Load memory but disable all writes (frozen-memory eval)")

    args = parser.parse_args()

    Path("storage").mkdir(exist_ok=True)
    Path("trajectories").mkdir(exist_ok=True)

    if args.mode == "train":
        run_offline_training(args)
    elif args.mode == "eval":
        run_online_evaluation(args)


if __name__ == "__main__":
    main()
