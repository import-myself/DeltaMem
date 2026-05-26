"""
PRTree 记忆模式消融实验（3 benchmarks）
=========================================
实验类型：
  - task_only  : 只使用任务树记忆注入 Prompt（环境树仍写入，但不注入）
  - env_only   : 只使用环境树记忆注入 Prompt（任务树仍写入，但不注入）

支持的 benchmark：
  - alfworld   : ALFWorld (eval_in_distribution / eval_out_of_distribution)
  - sciworld   : ScienceWorld (dev / test)
  - mind2web   : Mind2Web (test_task / test_website / test_domain)

统计指标（统一 CSV 格式）：
  benchmark, memory_mode, split/benchmark_split,
  success_rate,
  avg_steps (ALFWorld / SciWorld) | avg_element_acc / avg_action_f1 / avg_step_sr (Mind2Web),
  task_hit_rate, env_hit_rate,
  avg_task_retrieval_len_all, avg_env_retrieval_len_all,
  avg_task_retrieval_len_hit, avg_env_retrieval_len_hit,
  task_tree_total_nodes, env_tree_total_nodes,
  task_tree_level_counts, env_tree_level_counts,
  n_episodes, timestamp

结果追加写入 CSV。

运行示例：
  # 只跑 ALFWorld 的两种记忆模式消融
  python run_memory_mode_ablation.py \\
      --benchmark alfworld \\
      --memory-modes task_only,env_only \\
      --model gpt-4o-mini \\
      --alfworld-split eval_in_distribution \\
      --alfworld-load-memory ../ALFWorld/storage/prtree_dual_offline \\
      --max-episodes 50 \\
      --output-csv results/memory_mode_ablation.csv

  # 跑全部 3 个 benchmark
  python run_memory_mode_ablation.py \\
      --benchmark all \\
      --memory-modes task_only,env_only \\
      --model gpt-4o-mini \\
      --alfworld-load-memory ../ALFWorld/storage/prtree_dual_offline \\
      --sciworld-load-memory ../ScienceWorld/storage/prtree_sciworld_offline \\
      --mind2web-load-memory ../Mind2web/storage/prtree_mind2web_offline \\
      --output-csv results/memory_mode_ablation.csv
"""

import os
import sys
import csv
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from typing import Dict, List, Any, Optional

# ---- 路径设置 ----
_THIS_DIR    = Path(__file__).parent.resolve()
_PRTREE_ROOT = _THIS_DIR.parent
_ALFWORLD    = _PRTREE_ROOT / "ALFWorld"
_SCIWORLD    = _PRTREE_ROOT / "ScienceWorld"
_MIND2WEB    = _PRTREE_ROOT / "Mind2web"

sys.path.insert(0, str(_PRTREE_ROOT))
# benchmark-specific dirs 在各函数中按需插入，避免命名冲突

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =====================================================================
# 通用工具
# =====================================================================

def get_tree_level_stats(tree) -> Dict[str, int]:
    level_counts: Dict[int, int] = {}
    queue = deque([(tree.root, 0)])
    visited = {tree.root.node_id}
    while queue:
        node, depth = queue.popleft()
        level_counts[depth] = level_counts.get(depth, 0) + 1
        for child in node.children:
            if child.node_id not in visited:
                visited.add(child.node_id)
                queue.append((child, depth + 1))
    return {str(d): cnt for d, cnt in sorted(level_counts.items())}


def _base_row(benchmark: str, memory_mode: str, split: str, n: int,
               results: List[Dict], agent) -> Dict[str, Any]:
    """构建通用统计字段（所有 benchmark 共用）"""
    task_hits = [r for r in results if r.get("task_memory_used", False)]
    env_hits  = [r for r in results if r.get("env_memory_used",  False)]
    task_hit_lens = [r.get("task_retrieval_length", 0) for r in task_hits]
    env_hit_lens  = [r.get("env_retrieval_length",  0) for r in env_hits]

    task_level = get_tree_level_stats(agent.dual_memory.task_tree)
    env_level  = get_tree_level_stats(agent.dual_memory.env_tree)
    mem_stats  = agent.get_memory_stats()

    return {
        "benchmark":                  benchmark,
        "memory_mode":                memory_mode,
        "split":                      split,
        "n_episodes":                 n,
        "task_hit_rate":              round(len(task_hits) / n, 6),
        "env_hit_rate":               round(len(env_hits)  / n, 6),
        "avg_task_retrieval_len_all": round(sum(r.get("task_retrieval_length", 0) for r in results) / n, 4),
        "avg_env_retrieval_len_all":  round(sum(r.get("env_retrieval_length",  0) for r in results) / n, 4),
        "avg_task_retrieval_len_hit": round(sum(task_hit_lens) / len(task_hit_lens), 4) if task_hit_lens else 0.0,
        "avg_env_retrieval_len_hit":  round(sum(env_hit_lens)  / len(env_hit_lens),  4) if env_hit_lens  else 0.0,
        "task_tree_total_nodes":      mem_stats["task_tree_nodes"],
        "env_tree_total_nodes":       mem_stats["env_tree_nodes"],
        "task_tree_level_counts":     json.dumps(task_level),
        "env_tree_level_counts":      json.dumps(env_level),
        "timestamp":                  time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# =====================================================================
# ALFWorld
# =====================================================================

def _load_alfworld_env(split: str):
    import yaml
    import alfworld
    import alfworld.agents.environment as environment

    split_sizes = {
        "train": 3553,
        "eval_in_distribution": 140,
        "eval_out_of_distribution": 134,
    }
    if split not in split_sizes:
        raise ValueError(f"Unknown ALFWorld split: {split}")
    n = split_sizes[split]

    if "ALFWORLD_DATA" not in os.environ:
        raise EnvironmentError("ALFWORLD_DATA env var not set")
    path = os.environ["ALFWORLD_DATA"]
    with open(os.path.join(path, "base_config.yaml")) as f:
        config = yaml.safe_load(f)

    env = environment.get_environment(config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    return env, n


def _get_alfworld_task_type(game_file: str) -> Optional[str]:
    name = "/".join(game_file.split("/")[-3:-1])
    for prefix in ["pick_and_place", "pick_clean_then_place", "pick_heat_then_place",
                   "pick_cool_then_place", "look_at_obj", "pick_two_obj"]:
        if name.startswith(prefix):
            return prefix
    return None


def run_alfworld_experiment(args, memory_mode: str) -> Dict[str, Any]:
    """运行 ALFWorld 单组 memory_mode 实验"""
    # 延迟导入，避免污染其他 benchmark 的 sys.path
    if str(_ALFWORLD) not in sys.path:
        sys.path.insert(0, str(_ALFWORLD))
    from agent_alfworld_dual import DualTreeReflectiveAgent
    from common.llm_client import create_llm_client

    split = args.alfworld_split
    env, n_tasks = _load_alfworld_env(split)
    n_episodes = min(args.max_episodes or n_tasks, n_tasks)

    llm_client = create_llm_client(args.model)
    agent = DualTreeReflectiveAgent(
        agent_name=f"AblationAgent_{memory_mode}",
        llm_client=llm_client,
        icl_num=args.icl_num,
        icl_data_path=str(_ALFWORLD / "data" / "alfworld_icl.json"),
    )
    if args.alfworld_load_memory:
        agent.load_memory(args.alfworld_load_memory)
        stats = agent.get_memory_stats()
        logger.info(f"[ALFWorld] Loaded: task={stats['task_tree_nodes']}, env={stats['env_tree_nodes']}")

    exp_id  = f"alfworld__{memory_mode}__{split}"
    traj_dir = os.path.join(args.traj_dir or "trajectories/memory_mode", exp_id)
    os.makedirs(traj_dir, exist_ok=True)

    results = []
    for ep_idx in range(n_episodes):
        obs, info = env.reset()
        task_instruction = "\n".join(obs[0].split("\n\n")[1:])
        task_type = _get_alfworld_task_type(info["extra.gamefile"][0])

        messages = agent.run_episode(
            task_instruction=task_instruction,
            env=env,
            task_type=task_type,
            max_steps=args.max_steps,
            memory_mode=memory_mode,
        )
        result = messages[-1]
        results.append(result)
        with open(os.path.join(traj_dir, f"{ep_idx}.json"), "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        if (ep_idx + 1) % 10 == 0:
            logger.info(f"  [ALFWorld/{memory_mode}] Ep {ep_idx+1}/{n_episodes}: "
                        f"SR={sum(r['success'] for r in results)/len(results):.2%}")

    if args.save_tree_prefix:
        agent.save_memory(f"{args.save_tree_prefix}_{exp_id}")

    n = len(results)
    row = _base_row("alfworld", memory_mode, split, n, results, agent)
    row["success_rate"]  = round(sum(r["success"] for r in results) / n, 6)
    row["avg_steps"]     = round(sum(r["steps"]   for r in results) / n, 4)
    row["avg_element_acc"]    = "N/A"
    row["avg_action_f1"]      = "N/A"
    row["avg_step_success_rate"] = "N/A"
    return row


# =====================================================================
# ScienceWorld
# =====================================================================

def run_sciworld_experiment(args, memory_mode: str) -> Dict[str, Any]:
    """运行 SciWorld 单组 memory_mode 实验"""
    if str(_SCIWORLD) not in sys.path:
        sys.path.insert(0, str(_SCIWORLD))
    from agent_sciworld_dual import DualTreeSciWorldAgent
    from common.llm_client import create_llm_client
    from utils import sciworld_monkey_patch

    split = args.sciworld_split
    # 加载任务索引
    split_file = {
        "train": str(_SCIWORLD / "data/sciworld/train_indices.json"),
        "dev":   str(_SCIWORLD / "data/sciworld/dev_indices.json"),
        "test":  str(_SCIWORLD / "data/sciworld/test_indices.json"),
    }
    if split not in split_file:
        raise ValueError(f"Unknown SciWorld split: {split}")
    with open(split_file[split]) as f:
        task_idxs = json.load(f)

    sciworld_monkey_patch()
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv()

    n_episodes = min(args.max_episodes or len(task_idxs), len(task_idxs))

    llm_client = create_llm_client(args.model)
    icl_path          = str(_SCIWORLD / "data/sciworld_icl.json")
    max_steps_path    = str(_SCIWORLD / "data/sciworld/max_steps.json")
    taskname2id_path  = str(_SCIWORLD / "data/sciworld/taskname2id.json")
    agent = DualTreeSciWorldAgent(
        agent_name=f"SciAblation_{memory_mode}",
        llm_client=llm_client,
        icl_num=args.icl_num,
        icl_data_path=icl_path,
        max_steps_path=max_steps_path,
        taskname2id_path=taskname2id_path,
    )
    if args.sciworld_load_memory:
        agent.load_memory(args.sciworld_load_memory)
        stats = agent.get_memory_stats()
        logger.info(f"[SciWorld] Loaded: task={stats['task_tree_nodes']}, env={stats['env_tree_nodes']}")

    exp_id   = f"sciworld__{memory_mode}__{split}"
    traj_dir = os.path.join(args.traj_dir or "trajectories/memory_mode", exp_id)
    os.makedirs(traj_dir, exist_ok=True)

    results = []
    for ep_idx, (task_name, variation_idx) in enumerate(task_idxs[:n_episodes]):
        messages = agent.run_episode(
            env=env,
            task_name=task_name,
            variation_idx=variation_idx,
            memory_mode=memory_mode if memory_mode != "none" else "both",
            no_memory=(memory_mode == "none"),
            episode_idx=ep_idx,
        )
        result = messages[-1]
        results.append(result)
        fname = f"{ep_idx}_{task_name}_var{variation_idx}.json"
        with open(os.path.join(traj_dir, fname), "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        if (ep_idx + 1) % 10 == 0:
            logger.info(f"  [SciWorld/{memory_mode}] Ep {ep_idx+1}/{n_episodes}: "
                        f"SR={sum(r['success'] for r in results)/len(results):.2%}")

    if args.save_tree_prefix:
        agent.save_memory(f"{args.save_tree_prefix}_{exp_id}")

    n = len(results)
    row = _base_row("sciworld", memory_mode, split, n, results, agent)
    row["success_rate"]  = round(sum(r["success"]  for r in results) / n, 6)
    row["avg_steps"]     = round(sum(r.get("steps", r.get("reward", 0)) for r in results) / n, 4)
    row["avg_element_acc"]       = "N/A"
    row["avg_action_f1"]         = "N/A"
    row["avg_step_success_rate"] = "N/A"
    return row


# =====================================================================
# Mind2Web
# =====================================================================

def run_mind2web_experiment(args, memory_mode: str) -> Dict[str, Any]:
    """运行 Mind2Web 单组 memory_mode 实验"""
    if str(_MIND2WEB) not in sys.path:
        sys.path.insert(0, str(_MIND2WEB))
    from agent_mind2web_dual import DualTreeMind2WebAgent
    from common.llm_client import create_llm_client
    from mind2web_utils import add_scores, load_json_data, calculate_metrics

    benchmark_split = args.mind2web_split  # test_task / test_website / test_domain
    data_dir   = str(_MIND2WEB / "data")
    score_path = os.path.join(data_dir, "scores_all_data.pkl")

    samples = load_json_data(data_dir, benchmark_split)
    if os.path.exists(score_path):
        samples = add_scores(samples, score_path=score_path)
    if args.max_episodes:
        samples = samples[:args.max_episodes]

    n_tasks = len(samples)
    logger.info(f"[Mind2Web] {n_tasks} samples, split={benchmark_split}")

    llm_client    = create_llm_client(args.model)
    exemplar_path = os.path.join(data_dir, "example", "exemplars.json")
    agent = DualTreeMind2WebAgent(
        agent_name=f"M2WAblation_{memory_mode}",
        llm_client=llm_client,
        exemplar_path=exemplar_path,
    )
    if args.mind2web_load_memory:
        agent.load_memory(args.mind2web_load_memory)
        stats = agent.get_memory_stats()
        logger.info(f"[Mind2Web] Loaded: task={stats['task_tree_nodes']}, env={stats['env_tree_nodes']}")

    exp_id   = f"mind2web__{memory_mode}__{benchmark_split}"
    traj_dir = os.path.join(args.traj_dir or "trajectories/memory_mode", exp_id)
    os.makedirs(traj_dir, exist_ok=True)

    results = []
    for ep_idx, sample in enumerate(samples):
        try:
            result = agent.run_episode(
                sample=sample,
                model_name=args.model,
                memory_mode=memory_mode,
            )
        except Exception as e:
            logger.error(f"Episode {ep_idx} failed: {e}")
            result = {
                "success": False, "element_acc": [], "action_f1": [], "step_success": [],
                "memory_used": False, "task_memory_used": False, "env_memory_used": False,
                "task_retrieval_length": 0, "env_retrieval_length": 0,
                "task_reflection": {}, "env_reflection": {},
                "task_node_id": "", "env_node_id": "",
                "conversation": [], "trajectory": [f"ERROR: {e}"],
            }
        results.append(result)
        with open(os.path.join(traj_dir, f"{ep_idx}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if (ep_idx + 1) % 10 == 0:
            logger.info(f"  [Mind2Web/{memory_mode}] Ep {ep_idx+1}/{n_tasks}: "
                        f"SR={sum(r['success'] for r in results)/len(results):.2%}")

    if args.save_tree_prefix:
        agent.save_memory(f"{args.save_tree_prefix}_{exp_id}")

    n = len(results)
    metrics = calculate_metrics(results)
    row = _base_row("mind2web", memory_mode, benchmark_split, n, results, agent)
    row["success_rate"]          = round(sum(r["success"] for r in results) / n, 6)
    row["avg_steps"]             = "N/A"
    row["avg_element_acc"]       = round(metrics.get("element_acc", 0.0), 4)
    row["avg_action_f1"]         = round(metrics.get("action_f1", 0.0),   4)
    row["avg_step_success_rate"] = round(metrics.get("step_sr", 0.0),     4)
    return row


# =====================================================================
# WebShop
# =====================================================================

_WEBSHOP = _PRTREE_ROOT / "WebShop"
_WEBSHOP_SESSIONS_DEFAULT = "/tmp/ETO/eval_agent/data/webshop/test_indices.json"


def run_webshop_experiment(args, memory_mode: str) -> Dict[str, Any]:
    """运行 WebShop 单组 memory_mode 实验"""
    if str(_WEBSHOP) not in sys.path:
        sys.path.insert(0, str(_WEBSHOP))

    import gym
    from web_agent_site.envs import WebAgentTextEnv  # noqa: triggers registration
    from agent_webshop_dual import DualTreeWebShopAgent
    from common.llm_client import create_llm_client

    sessions_file = getattr(args, "webshop_sessions_file", None) or _WEBSHOP_SESSIONS_DEFAULT
    with open(sessions_file) as f:
        sessions = json.load(f)
    if args.max_episodes:
        sessions = sessions[: args.max_episodes]
    n_episodes = len(sessions)

    env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=1000)
    logger.info(f"[WebShop] Env ready. {n_episodes} sessions, memory_mode={memory_mode}")

    llm_client = create_llm_client(args.model)
    agent = DualTreeWebShopAgent(
        agent_name=f"WebShopAblation_{memory_mode}",
        llm_client=llm_client,
    )
    if getattr(args, "webshop_load_memory", None):
        agent.load_memory(args.webshop_load_memory)
        stats = agent.get_memory_stats()
        logger.info(f"[WebShop] Loaded: task={stats['task_tree_nodes']}, env={stats['env_tree_nodes']}")

    max_steps = getattr(args, "webshop_max_steps", None) or 15
    exp_id   = f"webshop__{memory_mode}"
    traj_dir = os.path.join(args.traj_dir or "trajectories/memory_mode", exp_id)
    os.makedirs(traj_dir, exist_ok=True)

    results = []
    for ep_idx, session_id in enumerate(sessions):
        try:
            result, messages = agent.run_episode(
                env=env,
                session_id=session_id,
                max_steps=max_steps,
                memory_mode=memory_mode,
                episode_idx=ep_idx,
            )
        except Exception as e:
            logger.error(f"Episode {ep_idx} (session={session_id}) failed: {e}")
            result = {
                "success": False, "reward": 0.0, "steps": max_steps,
                "task_memory_used": False, "env_memory_used": False,
                "task_retrieval_length": 0, "env_retrieval_length": 0,
            }
            messages = []
        results.append(result)
        with open(os.path.join(traj_dir, f"{ep_idx}.json"), "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        if (ep_idx + 1) % 20 == 0:
            logger.info(f"  [WebShop/{memory_mode}] Ep {ep_idx+1}/{n_episodes}: "
                        f"SR={sum(r['success'] for r in results)/len(results):.2%} "
                        f"AvgRew={sum(r['reward'] for r in results)/len(results):.4f}")

    if args.save_tree_prefix:
        agent.save_memory(f"{args.save_tree_prefix}_{exp_id}")

    n = len(results)
    row = _base_row("webshop", memory_mode, "test", n, results, agent)
    row["success_rate"]          = round(sum(r["success"] for r in results) / n, 6)
    row["avg_steps"]             = round(sum(r["steps"]  for r in results) / n, 4)
    row["avg_reward"]            = round(sum(r["reward"] for r in results) / n, 6)
    row["avg_element_acc"]       = "N/A"
    row["avg_action_f1"]         = "N/A"
    row["avg_step_success_rate"] = "N/A"
    return row


# =====================================================================
# CSV 追加写入
# =====================================================================

CSV_FIELDNAMES = [
    "benchmark", "memory_mode", "split", "n_episodes",
    "success_rate", "avg_steps",
    "avg_element_acc", "avg_action_f1", "avg_step_success_rate",
    "task_hit_rate", "env_hit_rate",
    "avg_task_retrieval_len_all", "avg_env_retrieval_len_all",
    "avg_task_retrieval_len_hit", "avg_env_retrieval_len_hit",
    "task_tree_total_nodes", "env_tree_total_nodes",
    "task_tree_level_counts", "env_tree_level_counts",
    "timestamp",
]


def append_to_csv(filepath: str, row: Dict[str, Any]) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(filepath).exists() and Path(filepath).stat().st_size > 0
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"✅ Appended to {filepath}")


# =====================================================================
# 主流程
# =====================================================================

BENCHMARK_RUNNERS = {
    "alfworld":  run_alfworld_experiment,
    "sciworld":  run_sciworld_experiment,
    "mind2web":  run_mind2web_experiment,
    "webshop":   run_webshop_experiment,
}


def main():
    p = argparse.ArgumentParser(
        description="PRTree 记忆模式消融实验（task_only / env_only），支持 3 benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 实验控制
    p.add_argument("--benchmark", type=str,
                   choices=["alfworld", "sciworld", "mind2web", "webshop", "all"], default="all",
                   help="要测试的 benchmark，all 表示全部")
    p.add_argument("--memory-modes", type=str, default="task_only,env_only",
                   help="逗号分隔的记忆模式列表，可选 task_only / env_only / both")
    p.add_argument("--model",    type=str, default="gpt-4o-mini")
    p.add_argument("--icl-num",  type=int, default=1)
    p.add_argument("--max-episodes", type=int, default=None,
                   help="每组实验最大 episode 数，默认使用 split 全量")
    p.add_argument("--max-steps",    type=int, default=30)

    # 输出
    p.add_argument("--output-csv",      type=str, default="results/memory_mode_ablation.csv")
    p.add_argument("--traj-dir",        type=str, default=None)
    p.add_argument("--save-tree-prefix", type=str, default=None)

    # ALFWorld 专属
    p.add_argument("--alfworld-split",
                   choices=["eval_in_distribution", "eval_out_of_distribution"],
                   default="eval_in_distribution")
    p.add_argument("--alfworld-load-memory", type=str, default=None)

    # ScienceWorld 专属
    p.add_argument("--sciworld-split", choices=["dev", "test"], default="test")
    p.add_argument("--sciworld-load-memory", type=str, default=None)

    # Mind2Web 专属
    p.add_argument("--mind2web-split",
                   choices=["test_task", "test_website", "test_domain"],
                   default="test_task")
    p.add_argument("--mind2web-load-memory", type=str, default=None)

    # WebShop 专属
    p.add_argument("--webshop-sessions-file", dest="webshop_sessions_file", type=str,
                   default=_WEBSHOP_SESSIONS_DEFAULT)
    p.add_argument("--webshop-load-memory", dest="webshop_load_memory", type=str, default=None)
    p.add_argument("--webshop-max-steps",   dest="webshop_max_steps",   type=int, default=15)

    args = p.parse_args()

    modes = [m.strip() for m in args.memory_modes.split(",") if m.strip()]
    benchmarks = (
        list(BENCHMARK_RUNNERS.keys())
        if args.benchmark == "all"
        else [args.benchmark]
    )

    logger.info("=" * 70)
    logger.info("PRTree 记忆模式消融实验")
    logger.info(f"  Benchmarks   : {benchmarks}")
    logger.info(f"  Memory modes : {modes}")
    logger.info(f"  Model        : {args.model}")
    logger.info(f"  Max episodes : {args.max_episodes or 'full split'}")
    logger.info(f"  Output CSV   : {args.output_csv}")
    logger.info("=" * 70)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    total = len(benchmarks) * len(modes)
    done  = 0

    for bm in benchmarks:
        runner = BENCHMARK_RUNNERS[bm]
        for mode in modes:
            done += 1
            logger.info(f"\n[{done}/{total}] benchmark={bm}, memory_mode={mode}")
            try:
                row = runner(args, memory_mode=mode)
                append_to_csv(args.output_csv, row)
                logger.info(f"  SR={row['success_rate']}")
            except Exception as e:
                logger.error(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()

    logger.info(f"\n🎉 All done → {args.output_csv}")


if __name__ == "__main__":
    main()
