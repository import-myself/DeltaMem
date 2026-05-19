"""
ALFWorld PRTree 阈值参数消融实验
====================================
功能：固定一棵树的参数（默认值），扫描另一棵树的 base_threshold × depth_step 网格
支持：--mode task / env / both

统计指标：
  成功率、平均步数、task/env 命中率、平均检索深度（全量/命中）、树形状（每层节点数）

结果追加写入 CSV。

注意：本脚本放于 PRTree/ablation/，通过 sys.path 引入 ALFWorld 和公共模块。

运行示例：
  cd /hdd/REDACTED_USER/DeltaMem/ablation
  python run_threshold_ablation.py \\
      --mode task \\
      --model gpt-4o-mini \\
      --split eval_in_distribution \\
      --load-memory ../ALFWorld/storage/prtree_dual_offline \\
      --max-episodes 50 \\
      --output-csv results/threshold_ablation.csv
"""

import os
import sys
import csv
import json
import time
import logging
import argparse
from pathlib import Path
from collections import deque
from typing import Dict, List, Any, Optional

# ---- 路径设置 ----
_THIS_DIR    = Path(__file__).parent.resolve()         # .../DeltaMem/ablation/
_PRTREE_ROOT = _THIS_DIR.parent                         # .../DeltaMem/
_ALFWORLD    = _PRTREE_ROOT / "ALFWorld"

sys.path.insert(0, str(_PRTREE_ROOT))   # common, memory
sys.path.insert(0, str(_ALFWORLD))      # agent_alfworld_dual, config, dual_prompt

import yaml

from agent_alfworld_dual import DualTreeReflectiveAgent
from common.llm_client import create_llm_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =====================================================================
# 当前默认参数（固定端使用这些值）
# =====================================================================
DEFAULT_TASK_BASE_THRESHOLD = 0.80
DEFAULT_TASK_DEPTH_STEP     = 0.03
DEFAULT_TASK_MAX_THRESHOLD  = 0.95

DEFAULT_ENV_BASE_THRESHOLD  = 0.88
DEFAULT_ENV_DEPTH_STEP      = 0.02
DEFAULT_ENV_MAX_THRESHOLD   = 0.99

# =====================================================================
# 工具函数
# =====================================================================

def load_alfworld_env(split: str):
    import alfworld
    import alfworld.agents.environment as environment

    split_sizes = {
        "train": 3553,
        "eval_in_distribution": 140,
        "eval_out_of_distribution": 134,
    }
    if split not in split_sizes:
        raise ValueError(f"Unknown split: {split}")
    n_tasks = split_sizes[split]

    if "ALFWORLD_DATA" not in os.environ:
        logger.error("ALFWORLD_DATA environment variable is not set.")
        sys.exit(1)

    path = os.environ["ALFWORLD_DATA"]
    with open(os.path.join(path, "base_config.yaml")) as f:
        config = yaml.safe_load(f)

    env = environment.get_environment(config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    return env, n_tasks


def get_tree_level_stats(tree) -> Dict[str, int]:
    """BFS 遍历，返回每层节点数 {"0": n, "1": n, ...}"""
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


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def get_task_type(game_file: str) -> Optional[str]:
    name = "/".join(game_file.split("/")[-3:-1])
    for prefix in [
        "pick_and_place", "pick_clean_then_place", "pick_heat_then_place",
        "pick_cool_then_place", "look_at_obj", "pick_two_obj",
    ]:
        if name.startswith(prefix):
            return prefix
    return None


# =====================================================================
# 单组参数实验
# =====================================================================

def run_single_experiment(
    args,
    task_base: float, task_step: float,
    env_base: float,  env_step: float,
    exp_type: str, exp_id: str,
) -> Dict[str, Any]:
    env, n_tasks = load_alfworld_env(args.split)
    n_episodes = min(args.max_episodes, n_tasks)

    llm_client = create_llm_client(args.model)
    agent = DualTreeReflectiveAgent(
        agent_name="ThresholdAblationAgent",
        llm_client=llm_client,
        icl_num=args.icl_num,
        icl_data_path=str(_ALFWORLD / "data" / "alfworld_icl.json"),
    )

    if args.load_memory:
        task_fp = args.load_memory.replace(".json", "") + "_task.json"
        env_fp  = args.load_memory.replace(".json", "") + "_env.json"
        if Path(task_fp).exists() or Path(env_fp).exists():
            agent.load_memory(args.load_memory)
            stats = agent.get_memory_stats()
            logger.info(f"📥 Loaded: task={stats['task_tree_nodes']}, env={stats['env_tree_nodes']}")

    # 覆盖阈值参数
    agent.dual_memory.task_tree.base_threshold = task_base
    agent.dual_memory.task_tree.depth_step     = task_step
    agent.dual_memory.task_tree.max_threshold  = DEFAULT_TASK_MAX_THRESHOLD
    agent.dual_memory.env_tree.base_threshold  = env_base
    agent.dual_memory.env_tree.depth_step      = env_step
    agent.dual_memory.env_tree.max_threshold   = DEFAULT_ENV_MAX_THRESHOLD

    traj_dir = os.path.join(args.traj_dir or "trajectories/threshold_ablation", exp_id)
    os.makedirs(traj_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for ep_idx in range(n_episodes):
        obs, info = env.reset()
        task_instruction = "\n".join(obs[0].split("\n\n")[1:])
        task_type = get_task_type(info["extra.gamefile"][0])

        messages = agent.run_episode(
            task_instruction=task_instruction,
            env=env,
            task_type=task_type,
            max_steps=args.max_steps,
        )
        result = messages[-1]
        result["task_type"] = task_type
        results.append(result)

        with open(os.path.join(traj_dir, f"{ep_idx}.json"), "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)

        if (ep_idx + 1) % 10 == 0:
            logger.info(f"  Ep {ep_idx + 1}/{n_episodes}: SR={sum(r['success'] for r in results)/len(results):.2%}")

    if args.save_tree_prefix:
        agent.save_memory(f"{args.save_tree_prefix}_{exp_id}")

    n = len(results)
    task_hits = [r for r in results if r.get("task_memory_used", False)]
    env_hits  = [r for r in results if r.get("env_memory_used",  False)]

    task_hit_lens = [r.get("task_retrieval_length", 0) for r in task_hits]
    env_hit_lens  = [r.get("env_retrieval_length",  0) for r in env_hits]

    task_level = get_tree_level_stats(agent.dual_memory.task_tree)
    env_level  = get_tree_level_stats(agent.dual_memory.env_tree)
    mem_stats  = agent.get_memory_stats()

    row = {
        "experiment_type":            exp_type,
        "exp_id":                     exp_id,
        "task_base_threshold":        task_base,
        "task_depth_step":            task_step,
        "env_base_threshold":         env_base,
        "env_depth_step":             env_step,
        "success_rate":               round(sum(r["success"] for r in results) / n, 6),
        "avg_steps":                  round(sum(r["steps"] for r in results) / n, 4),
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
        "n_episodes":                 n,
        "split":                      args.split,
        "timestamp":                  time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    logger.info(f"📊 [{exp_id}] SR={row['success_rate']:.2%}  "
                f"TaskHit={row['task_hit_rate']:.2%} depth_hit={row['avg_task_retrieval_len_hit']:.2f}  "
                f"EnvHit={row['env_hit_rate']:.2%} depth_hit={row['avg_env_retrieval_len_hit']:.2f}")
    return row


# =====================================================================
# CSV 追加写入
# =====================================================================

CSV_FIELDNAMES = [
    "experiment_type", "exp_id",
    "task_base_threshold", "task_depth_step", "env_base_threshold", "env_depth_step",
    "success_rate", "avg_steps", "task_hit_rate", "env_hit_rate",
    "avg_task_retrieval_len_all", "avg_env_retrieval_len_all",
    "avg_task_retrieval_len_hit", "avg_env_retrieval_len_hit",
    "task_tree_total_nodes", "env_tree_total_nodes",
    "task_tree_level_counts", "env_tree_level_counts",
    "n_episodes", "split", "timestamp",
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
# 消融主流程
# =====================================================================

def run_task_tree_ablation(args) -> None:
    bases = parse_float_list(args.task_base_thresholds)
    steps = parse_float_list(args.task_depth_steps)
    total, done = len(bases) * len(steps), 0
    for base in bases:
        for step in steps:
            done += 1
            exp_id = f"task_ablation__tb{base:.3f}_ts{step:.3f}__eb{DEFAULT_ENV_BASE_THRESHOLD:.3f}_es{DEFAULT_ENV_DEPTH_STEP:.3f}"
            logger.info(f"\n[{done}/{total}] {exp_id}")
            row = run_single_experiment(args, task_base=base, task_step=step,
                                        env_base=DEFAULT_ENV_BASE_THRESHOLD,
                                        env_step=DEFAULT_ENV_DEPTH_STEP,
                                        exp_type="task_ablation", exp_id=exp_id)
            append_to_csv(args.output_csv, row)


def run_env_tree_ablation(args) -> None:
    bases = parse_float_list(args.env_base_thresholds)
    steps = parse_float_list(args.env_depth_steps)
    total, done = len(bases) * len(steps), 0
    for base in bases:
        for step in steps:
            done += 1
            exp_id = f"env_ablation__tb{DEFAULT_TASK_BASE_THRESHOLD:.3f}_ts{DEFAULT_TASK_DEPTH_STEP:.3f}__eb{base:.3f}_es{step:.3f}"
            logger.info(f"\n[{done}/{total}] {exp_id}")
            row = run_single_experiment(args, task_base=DEFAULT_TASK_BASE_THRESHOLD,
                                        task_step=DEFAULT_TASK_DEPTH_STEP,
                                        env_base=base, env_step=step,
                                        exp_type="env_ablation", exp_id=exp_id)
            append_to_csv(args.output_csv, row)


# =====================================================================
# CLI
# =====================================================================

def main():
    p = argparse.ArgumentParser(description="PRTree 阈值参数消融 (ALFWorld)")
    p.add_argument("--mode",   type=str, choices=["task", "env", "both"], default="both")
    p.add_argument("--split",  type=str, choices=["eval_in_distribution", "eval_out_of_distribution"],
                   default="eval_in_distribution")
    p.add_argument("--model",  type=str, default="gpt-4o-mini")
    p.add_argument("--icl-num", type=int, default=1)
    p.add_argument("--max-episodes", type=int, default=140)
    p.add_argument("--max-steps",    type=int, default=30)
    p.add_argument("--load-memory",  type=str, default=None)
    p.add_argument("--save-tree-prefix", type=str, default=None)
    p.add_argument("--output-csv",   type=str, default="results/threshold_ablation.csv")
    p.add_argument("--traj-dir",     type=str, default=None)
    p.add_argument("--task-base-thresholds", type=str, default="0.70,0.75,0.80,0.85,0.90")
    p.add_argument("--task-depth-steps",     type=str, default="0.01,0.02,0.03,0.04,0.05")
    p.add_argument("--env-base-thresholds",  type=str, default="0.82,0.85,0.88,0.91,0.94")
    p.add_argument("--env-depth-steps",      type=str, default="0.01,0.02,0.03,0.04")
    args = p.parse_args()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    if args.mode in ("task", "both"):
        run_task_tree_ablation(args)
    if args.mode in ("env", "both"):
        run_env_tree_ablation(args)

    logger.info(f"\n🎉 Done → {args.output_csv}")


if __name__ == "__main__":
    main()
