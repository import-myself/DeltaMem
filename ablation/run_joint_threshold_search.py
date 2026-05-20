"""
PRTree 联合阈值网格搜索（支持 ALFWorld / ScienceWorld / Mind2Web / WebShop）
====================================================================
对 task 树和 env 树的 (base_threshold, depth_step) 进行四维联合网格搜索，
找出使成功率最优的参数组合。

搜索空间（笛卡尔积）：
  task_base × task_step × env_base × env_step

运行模式：
  --mode grid     : 运行全部参数组合（默认）
  --mode analyze  : 仅读取已有 CSV 输出最优组合，不运行实验

各 benchmark 结果写入独立 CSV：
  {output_csv_prefix}_alfworld.csv
  {output_csv_prefix}_sciworld.csv
  {output_csv_prefix}_mind2web.csv
  {output_csv_prefix}_webshop.csv

已跑过的 exp_id 自动跳过（断点续跑）。

示例：
  cd /hdd/REDACTED_USER/DeltaMem/ablation

  # 单个 benchmark
  python run_joint_threshold_search.py \\
      --benchmark sciworld \\
      --model Qwen3-14B \\
      --sciworld-split test \\
      --sciworld-load-memory ../ScienceWorld/storage/prtree_sciworld_offline \\
      --max-episodes 50 \\
      --output-csv results/joint_threshold_search

  # WebShop
  python run_joint_threshold_search.py \\
      --benchmark webshop \\
      --model deepseek-v4-flash \\
      --webshop-sessions-file /tmp/ETO/eval_agent/data/webshop/test_indices.json \\
      --output-csv results/joint_threshold_search

  # 全部四个 benchmark
  python run_joint_threshold_search.py \\
      --benchmark all \\
      --model Qwen3-14B \\
      --alfworld-load-memory ../ALFWorld/storage/prtree_dual_offline \\
      --sciworld-load-memory ../ScienceWorld/storage/prtree_sciworld_offline \\
      --mind2web-load-memory ../Mind2web/storage/prtree_mind2web_offline \\
      --output-csv results/joint_threshold_search
"""

import os
import sys
import csv
import json
import time
import logging
import argparse
from itertools import product
from pathlib import Path
from collections import deque
from typing import Any, Dict, List, Optional, Set

# ── 路径设置 ──────────────────────────────────────────────────────────────────
_THIS_DIR    = Path(__file__).parent.resolve()
_PRTREE_ROOT = _THIS_DIR.parent
_ALFWORLD    = _PRTREE_ROOT / "ALFWorld"
_SCIWORLD    = _PRTREE_ROOT / "ScienceWorld"
_MIND2WEB    = _PRTREE_ROOT / "Mind2web"
_WEBSHOP     = _PRTREE_ROOT / "WebShop"

sys.path.insert(0, str(_PRTREE_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── 树的最大阈值上限 ───────────────────────────────────────────────────────────
TASK_MAX_THRESHOLD = 0.95
ENV_MAX_THRESHOLD  = 0.99

# ── 各 benchmark 专属 CSV 字段 ────────────────────────────────────────────────
_COMMON_FIELDS = [
    "exp_id", "benchmark",
    "task_base_threshold", "task_depth_step",
    "env_base_threshold",  "env_depth_step",
    "task_hit_rate", "env_hit_rate",
    "avg_task_retrieval_len_all", "avg_env_retrieval_len_all",
    "avg_task_retrieval_len_hit", "avg_env_retrieval_len_hit",
    "task_tree_total_nodes", "env_tree_total_nodes",
    "task_tree_level_counts", "env_tree_level_counts",
    "n_episodes", "split", "timestamp",
]

CSV_FIELDNAMES: Dict[str, List[str]] = {
    "alfworld": [
        "exp_id", "benchmark",
        "task_base_threshold", "task_depth_step",
        "env_base_threshold",  "env_depth_step",
        "success_rate", "avg_steps",
        "task_hit_rate", "env_hit_rate",
        "avg_task_retrieval_len_all", "avg_env_retrieval_len_all",
        "avg_task_retrieval_len_hit", "avg_env_retrieval_len_hit",
        "task_tree_total_nodes", "env_tree_total_nodes",
        "task_tree_level_counts", "env_tree_level_counts",
        "n_episodes", "split", "timestamp",
    ],
    "sciworld": [
        "exp_id", "benchmark",
        "task_base_threshold", "task_depth_step",
        "env_base_threshold",  "env_depth_step",
        "success_rate", "avg_reward", "avg_steps",
        "task_hit_rate", "env_hit_rate",
        "avg_task_retrieval_len_all", "avg_env_retrieval_len_all",
        "avg_task_retrieval_len_hit", "avg_env_retrieval_len_hit",
        "task_tree_total_nodes", "env_tree_total_nodes",
        "task_tree_level_counts", "env_tree_level_counts",
        "n_episodes", "split", "timestamp",
    ],
    "mind2web": [
        "exp_id", "benchmark",
        "task_base_threshold", "task_depth_step",
        "env_base_threshold",  "env_depth_step",
        "success_rate",
        "avg_element_acc", "avg_action_f1", "avg_step_success_rate",
        "task_hit_rate", "env_hit_rate",
        "avg_task_retrieval_len_all", "avg_env_retrieval_len_all",
        "avg_task_retrieval_len_hit", "avg_env_retrieval_len_hit",
        "task_tree_total_nodes", "env_tree_total_nodes",
        "task_tree_level_counts", "env_tree_level_counts",
        "n_episodes", "split", "timestamp",
    ],
    "webshop": [
        "exp_id", "benchmark",
        "task_base_threshold", "task_depth_step",
        "env_base_threshold",  "env_depth_step",
        "success_rate", "avg_reward", "avg_steps",
        "task_hit_rate", "env_hit_rate",
        "avg_task_retrieval_len_all", "avg_env_retrieval_len_all",
        "avg_task_retrieval_len_hit", "avg_env_retrieval_len_hit",
        "task_tree_total_nodes", "env_tree_total_nodes",
        "task_tree_level_counts", "env_tree_level_counts",
        "n_episodes", "split", "timestamp",
    ],
}


# =============================================================================
# 通用工具函数
# =============================================================================

def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def make_exp_id(benchmark: str, tb: float, ts: float, eb: float, es: float) -> str:
    return f"joint__tb{tb:.3f}_ts{ts:.3f}__eb{eb:.3f}_es{es:.3f}"


def get_csv_path(output_csv: str, benchmark: str) -> str:
    """将 results/foo.csv → results/foo_alfworld.csv"""
    p = Path(output_csv)
    return str(p.parent / f"{p.stem}_{benchmark}{p.suffix}")


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


def _build_base_stats(results: List[Dict], agent, benchmark: str, split: str,
                      task_base: float, task_step: float,
                      env_base: float, env_step: float,
                      exp_id: str) -> Dict[str, Any]:
    """构建所有 benchmark 共用的统计字段。"""
    n = len(results)
    task_hits  = [r for r in results if r.get("task_memory_used", False)]
    env_hits   = [r for r in results if r.get("env_memory_used",  False)]
    task_h_len = [r.get("task_retrieval_length", 0) for r in task_hits]
    env_h_len  = [r.get("env_retrieval_length",  0) for r in env_hits]

    task_level = get_tree_level_stats(agent.dual_memory.task_tree)
    env_level  = get_tree_level_stats(agent.dual_memory.env_tree)
    mem_stats  = agent.get_memory_stats()

    return {
        "exp_id":                     exp_id,
        "benchmark":                  benchmark,
        "task_base_threshold":        task_base,
        "task_depth_step":            task_step,
        "env_base_threshold":         env_base,
        "env_depth_step":             env_step,
        "task_hit_rate":              round(len(task_hits) / n, 6),
        "env_hit_rate":               round(len(env_hits)  / n, 6),
        "avg_task_retrieval_len_all": round(sum(r.get("task_retrieval_length", 0) for r in results) / n, 4),
        "avg_env_retrieval_len_all":  round(sum(r.get("env_retrieval_length",  0) for r in results) / n, 4),
        "avg_task_retrieval_len_hit": round(sum(task_h_len) / len(task_h_len), 4) if task_h_len else 0.0,
        "avg_env_retrieval_len_hit":  round(sum(env_h_len)  / len(env_h_len),  4) if env_h_len  else 0.0,
        "task_tree_total_nodes":      mem_stats["task_tree_nodes"],
        "env_tree_total_nodes":       mem_stats["env_tree_nodes"],
        "task_tree_level_counts":     json.dumps(task_level),
        "env_tree_level_counts":      json.dumps(env_level),
        "n_episodes":                 n,
        "split":                      split,
        "timestamp":                  time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _set_tree_thresholds(agent, task_base: float, task_step: float,
                          env_base: float, env_step: float) -> None:
    agent.dual_memory.task_tree.base_threshold = task_base
    agent.dual_memory.task_tree.depth_step     = task_step
    agent.dual_memory.task_tree.max_threshold  = TASK_MAX_THRESHOLD
    agent.dual_memory.env_tree.base_threshold  = env_base
    agent.dual_memory.env_tree.depth_step      = env_step
    agent.dual_memory.env_tree.max_threshold   = ENV_MAX_THRESHOLD


def load_finished_ids(csv_path: str) -> Set[str]:
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return set()
    finished: Set[str] = set()
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("exp_id"):
                finished.add(row["exp_id"])
    return finished


def append_to_csv(filepath: str, row: Dict[str, Any], benchmark: str) -> None:
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not p.exists() or p.stat().st_size == 0
    fieldnames = CSV_FIELDNAMES[benchmark]
    with open(p, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


# =============================================================================
# ALFWorld 单次实验
# =============================================================================

def _load_alfworld_env(split: str):
    import yaml
    import alfworld
    import alfworld.agents.environment as environment

    split_sizes = {
        "train":                    3553,
        "eval_in_distribution":     140,
        "eval_out_of_distribution": 134,
    }
    if split not in split_sizes:
        raise ValueError(f"Unknown ALFWorld split: {split}")

    if "ALFWORLD_DATA" not in os.environ:
        logger.error("ALFWORLD_DATA environment variable is not set.")
        sys.exit(1)

    data_path = os.environ["ALFWORLD_DATA"]
    with open(os.path.join(data_path, "base_config.yaml")) as f:
        config = yaml.safe_load(f)

    env = environment.get_environment(config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    return env, split_sizes[split]


def _get_alfworld_task_type(game_file: str) -> Optional[str]:
    name = "/".join(game_file.split("/")[-3:-1])
    for prefix in [
        "pick_and_place", "pick_clean_then_place", "pick_heat_then_place",
        "pick_cool_then_place", "look_at_obj", "pick_two_obj",
    ]:
        if name.startswith(prefix):
            return prefix
    return None


def run_alfworld_experiment(
    args,
    task_base: float, task_step: float,
    env_base:  float, env_step:  float,
    exp_id: str,
) -> Dict[str, Any]:
    if str(_ALFWORLD) not in sys.path:
        sys.path.insert(0, str(_ALFWORLD))
    from agent_alfworld_dual import DualTreeReflectiveAgent
    from common.llm_client import create_llm_client

    split = args.alfworld_split
    env, n_tasks = _load_alfworld_env(split)
    n_episodes = n_tasks if not args.max_episodes else min(args.max_episodes, n_tasks)

    llm_client = create_llm_client(args.model)
    agent = DualTreeReflectiveAgent(
        agent_name="JointGridSearchAgent",
        llm_client=llm_client,
        icl_num=args.icl_num,
        icl_data_path=str(_ALFWORLD / "data" / "alfworld_icl.json"),
    )
    if args.alfworld_load_memory:
        agent.load_memory(args.alfworld_load_memory)
        stats = agent.get_memory_stats()
        logger.info(f"  [ALFWorld] 📥 Loaded: task={stats['task_tree_nodes']} env={stats['env_tree_nodes']}")

    _set_tree_thresholds(agent, task_base, task_step, env_base, env_step)

    results: List[Dict[str, Any]] = []
    for ep_idx in range(n_episodes):
        obs, info = env.reset()
        task_instruction = "\n".join(obs[0].split("\n\n")[1:])
        task_type = _get_alfworld_task_type(info["extra.gamefile"][0])

        messages = agent.run_episode(
            task_instruction=task_instruction,
            env=env,
            task_type=task_type,
            max_steps=args.alfworld_max_steps,
        )
        result = messages[-1]
        result["task_type"] = task_type
        results.append(result)

        if (ep_idx + 1) % 10 == 0:
            sr = sum(r["success"] for r in results) / len(results)
            logger.info(f"    [ALFWorld] Ep {ep_idx + 1}/{n_episodes}: SR={sr:.2%}")

    n = len(results)
    row = _build_base_stats(results, agent, "alfworld", split,
                            task_base, task_step, env_base, env_step, exp_id)
    row["success_rate"] = round(sum(r["success"] for r in results) / n, 6)
    row["avg_steps"]    = round(sum(r["steps"]   for r in results) / n, 4)

    logger.info(
        f"  ✅ [ALFWorld] SR={row['success_rate']:.2%}  Steps={row['avg_steps']:.1f}  "
        f"TaskHit={row['task_hit_rate']:.2%}  EnvHit={row['env_hit_rate']:.2%}"
    )
    return row


# =============================================================================
# ScienceWorld 单次实验
# =============================================================================

def run_sciworld_experiment(
    args,
    task_base: float, task_step: float,
    env_base:  float, env_step:  float,
    exp_id: str,
) -> Dict[str, Any]:
    if str(_SCIWORLD) not in sys.path:
        sys.path.insert(0, str(_SCIWORLD))
    from agent_sciworld_dual import DualTreeSciWorldAgent
    from common.llm_client import create_llm_client
    from utils import sciworld_monkey_patch

    split = args.sciworld_split
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

    # ScienceWorld max_steps 由 agent 内部 max_steps.json 按任务类型决定，不从外部传入
    n_episodes = len(task_idxs) if not args.max_episodes else min(args.max_episodes, len(task_idxs))

    llm_client = create_llm_client(args.model)
    agent = DualTreeSciWorldAgent(
        agent_name="JointGridSearchAgent",
        llm_client=llm_client,
        icl_num=args.icl_num,
        icl_data_path=str(_SCIWORLD / "data/sciworld_icl.json"),
        max_steps_path=str(_SCIWORLD / "data/sciworld/max_steps.json"),
        taskname2id_path=str(_SCIWORLD / "data/sciworld/taskname2id.json"),
    )
    if args.sciworld_load_memory:
        agent.load_memory(args.sciworld_load_memory)
        stats = agent.get_memory_stats()
        logger.info(f"  [SciWorld] 📥 Loaded: task={stats['task_tree_nodes']} env={stats['env_tree_nodes']}")

    _set_tree_thresholds(agent, task_base, task_step, env_base, env_step)

    results: List[Dict[str, Any]] = []
    for ep_idx, (task_name, variation_idx) in enumerate(task_idxs[:n_episodes]):
        messages = agent.run_episode(
            env=env,
            task_name=task_name,
            variation_idx=variation_idx,
        )
        result = messages[-1]
        results.append(result)

        if (ep_idx + 1) % 10 == 0:
            sr  = sum(r["success"] for r in results) / len(results)
            rwd = sum(r["reward"]  for r in results) / len(results)
            logger.info(f"    [SciWorld] Ep {ep_idx + 1}/{n_episodes}: SR={sr:.2%}  AvgReward={rwd:.4f}")

    n = len(results)
    row = _build_base_stats(results, agent, "sciworld", split,
                            task_base, task_step, env_base, env_step, exp_id)
    row["success_rate"] = round(sum(r["success"]         for r in results) / n, 6)
    row["avg_reward"]   = round(sum(r["reward"]          for r in results) / n, 4)
    row["avg_steps"]    = round(sum(r.get("steps", 0)    for r in results) / n, 4)

    logger.info(
        f"  ✅ [SciWorld] SR={row['success_rate']:.2%}  "
        f"AvgReward={row['avg_reward']:.4f}  Steps={row['avg_steps']:.1f}  "
        f"TaskHit={row['task_hit_rate']:.2%}  EnvHit={row['env_hit_rate']:.2%}"
    )
    return row


# =============================================================================
# Mind2Web 单次实验
# =============================================================================

def run_mind2web_experiment(
    args,
    task_base: float, task_step: float,
    env_base:  float, env_step:  float,
    exp_id: str,
) -> Dict[str, Any]:
    if str(_MIND2WEB) not in sys.path:
        sys.path.insert(0, str(_MIND2WEB))
    from agent_mind2web_dual import DualTreeMind2WebAgent
    from common.llm_client import create_llm_client
    from mind2web_utils import add_scores, load_json_data, calculate_metrics

    benchmark_split = args.mind2web_split
    data_dir   = str(_MIND2WEB / "data")
    score_path = os.path.join(data_dir, "scores_all_data.pkl")

    samples = load_json_data(data_dir, benchmark_split)
    if os.path.exists(score_path):
        samples = add_scores(samples, score_path=score_path)
    if args.max_episodes:
        samples = samples[: args.max_episodes]

    n_tasks = len(samples)  # max_episodes=None/0 时跑完整个 split
    logger.info(f"  [Mind2Web] {n_tasks} samples, split={benchmark_split}")

    llm_client    = create_llm_client(args.model)
    exemplar_path = os.path.join(data_dir, "example", "exemplars.json")
    agent = DualTreeMind2WebAgent(
        agent_name="JointGridSearchAgent",
        llm_client=llm_client,
        exemplar_path=exemplar_path,
    )
    if args.mind2web_load_memory:
        agent.load_memory(args.mind2web_load_memory)
        stats = agent.get_memory_stats()
        logger.info(f"  [Mind2Web] 📥 Loaded: task={stats['task_tree_nodes']} env={stats['env_tree_nodes']}")

    _set_tree_thresholds(agent, task_base, task_step, env_base, env_step)

    results: List[Dict[str, Any]] = []
    for ep_idx, sample in enumerate(samples):
        try:
            result = agent.run_episode(
                sample=sample,
                model_name=args.model,
            )
        except Exception as e:
            logger.error(f"    [Mind2Web] Episode {ep_idx} failed: {e}")
            result = {
                "success": False, "element_acc": [], "action_f1": [], "step_success": [],
                "task_memory_used": False, "env_memory_used": False,
                "task_retrieval_length": 0, "env_retrieval_length": 0,
            }
        results.append(result)

        if (ep_idx + 1) % 10 == 0:
            sr = sum(r["success"] for r in results) / len(results)
            logger.info(f"    [Mind2Web] Ep {ep_idx + 1}/{n_tasks}: SR={sr:.2%}")

    n = len(results)
    metrics = calculate_metrics(results)
    row = _build_base_stats(results, agent, "mind2web", benchmark_split,
                            task_base, task_step, env_base, env_step, exp_id)
    row["success_rate"]          = round(sum(r["success"] for r in results) / n, 6)
    row["avg_element_acc"]       = round(metrics.get("element_acc", 0.0), 4)
    row["avg_action_f1"]         = round(metrics.get("action_f1",   0.0), 4)
    row["avg_step_success_rate"] = round(metrics.get("step_success_rate", 0.0), 4)

    logger.info(
        f"  ✅ [Mind2Web] SR={row['success_rate']:.2%}  "
        f"ElemAcc={row['avg_element_acc']:.4f}  F1={row['avg_action_f1']:.4f}  "
        f"TaskHit={row['task_hit_rate']:.2%}  EnvHit={row['env_hit_rate']:.2%}"
    )
    return row


# =============================================================================
# WebShop 单次实验
# =============================================================================

def run_webshop_experiment(
    args,
    task_base: float, task_step: float,
    env_base:  float, env_step:  float,
    exp_id: str,
) -> Dict[str, Any]:
    if str(_WEBSHOP) not in sys.path:
        sys.path.insert(0, str(_WEBSHOP))

    import gym
    from web_agent_site.envs import WebAgentTextEnv  # noqa: F401
    from agent_webshop_dual import DualTreeWebShopAgent
    from common.llm_client import create_llm_client

    sessions_file = getattr(args, "webshop_sessions_file",
                            "/tmp/ETO/eval_agent/data/webshop/test_indices.json")
    with open(sessions_file) as f:
        all_sessions = json.load(f)

    max_steps = getattr(args, "webshop_max_steps", 15)
    n_episodes = len(all_sessions) if not args.max_episodes else min(args.max_episodes, len(all_sessions))
    sessions = all_sessions[:n_episodes]

    logger.info("Bootstrapping WebShop env (loading 1000 products) ...")
    env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=1000)
    logger.info("Env ready.")

    llm_client = create_llm_client(args.model)
    agent = DualTreeWebShopAgent(
        agent_name="JointGridSearchAgent",
        llm_client=llm_client,
    )

    load_memory_path = getattr(args, "webshop_load_memory", None)
    if load_memory_path:
        agent.load_memory(load_memory_path)
        stats = agent.get_memory_stats()
        logger.info(f"  [WebShop] 📥 Loaded: task={stats['task_tree_nodes']} env={stats['env_tree_nodes']}")

    _set_tree_thresholds(agent, task_base, task_step, env_base, env_step)

    results: List[Dict[str, Any]] = []
    for ep_idx, session_id in enumerate(sessions):
        try:
            result_dict, _ = agent.run_episode(
                env=env,
                session_id=session_id,
                max_steps=max_steps,
                episode_idx=ep_idx,
            )
        except Exception as e:
            logger.error(f"    [WebShop] Episode {ep_idx} (session={session_id}) failed: {e}")
            result_dict = {
                "reward": 0.0, "steps": max_steps, "success": False,
                "task_memory_used": False, "env_memory_used": False,
                "task_retrieval_length": 0, "env_retrieval_length": 0,
            }
        results.append(result_dict)

        if (ep_idx + 1) % 10 == 0:
            sr  = sum(r["success"] for r in results) / len(results)
            rwd = sum(r["reward"]  for r in results) / len(results)
            logger.info(f"    [WebShop] Ep {ep_idx + 1}/{n_episodes}: SR={sr:.2%}  AvgReward={rwd:.4f}")

    n = len(results)
    row = _build_base_stats(results, agent, "webshop", "test",
                            task_base, task_step, env_base, env_step, exp_id)
    row["success_rate"] = round(sum(r["success"]       for r in results) / n, 6)
    row["avg_reward"]   = round(sum(r["reward"]        for r in results) / n, 4)
    row["avg_steps"]    = round(sum(r.get("steps", 0)  for r in results) / n, 4)

    logger.info(
        f"  ✅ [WebShop] SR={row['success_rate']:.2%}  "
        f"AvgReward={row['avg_reward']:.4f}  Steps={row['avg_steps']:.1f}  "
        f"TaskHit={row['task_hit_rate']:.2%}  EnvHit={row['env_hit_rate']:.2%}"
    )
    return row


# ── benchmark → runner 映射 ──────────────────────────────────────────────────
BENCHMARK_RUNNERS = {
    "alfworld":  run_alfworld_experiment,
    "sciworld":  run_sciworld_experiment,
    "mind2web":  run_mind2web_experiment,
    "webshop":   run_webshop_experiment,
}


# =============================================================================
# 网格搜索主流程
# =============================================================================

def run_grid_search(args) -> None:
    task_bases = parse_float_list(args.task_base_thresholds)
    task_steps = parse_float_list(args.task_depth_steps)
    env_bases  = parse_float_list(args.env_base_thresholds)
    env_steps  = parse_float_list(args.env_depth_steps)

    combos = list(product(task_bases, task_steps, env_bases, env_steps))
    total  = len(combos)

    benchmarks = (
        list(BENCHMARK_RUNNERS.keys())
        if args.benchmark == "all"
        else [args.benchmark]
    )

    logger.info(f"\n{'='*64}")
    logger.info(f"  PRTree 联合阈值网格搜索")
    logger.info(f"  Benchmarks  : {benchmarks}")
    logger.info(f"  task_base   : {task_bases}")
    logger.info(f"  task_step   : {task_steps}")
    logger.info(f"  env_base    : {env_bases}")
    logger.info(f"  env_step    : {env_steps}")
    logger.info(f"  总组合数/bm  : {total}  ×  {len(benchmarks)} = {total * len(benchmarks)}")
    logger.info(f"  每组 episodes: {args.max_episodes}")
    for bm in benchmarks:
        logger.info(f"  {bm} CSV     : {get_csv_path(args.output_csv, bm)}")
    logger.info(f"{'='*64}\n")

    for bm in benchmarks:
        runner   = BENCHMARK_RUNNERS[bm]
        csv_path = get_csv_path(args.output_csv, bm)
        finished = load_finished_ids(csv_path)
        if finished:
            logger.info(f"  [{bm}] ⏩ 检测到 {len(finished)} 组已完成，自动跳过\n")

        for idx, (tb, ts, eb, es) in enumerate(combos, 1):
            exp_id = make_exp_id(bm, tb, ts, eb, es)
            if exp_id in finished:
                logger.info(f"[{bm}][{idx:>3}/{total}] 跳过（已完成）: {exp_id}")
                continue

            logger.info(f"\n[{bm}][{idx:>3}/{total}] task(base={tb}, step={ts})  env(base={eb}, step={es})")
            row = runner(args, tb, ts, eb, es, exp_id)
            append_to_csv(csv_path, row, bm)

    logger.info(f"\n{'='*64}")
    logger.info(f"  网格搜索完成")
    for bm in benchmarks:
        logger.info(f"  → {get_csv_path(args.output_csv, bm)}")
    logger.info(f"{'='*64}\n")


# =============================================================================
# 结果分析：输出最优组合
# =============================================================================

def analyze_results(csv_path: str, benchmark: str, top_k: int = 5) -> None:
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        logger.warning(f"结果文件不存在或为空: {csv_path}")
        return

    is_mind2web = (benchmark == "mind2web")
    is_sciworld = (benchmark in ("sciworld", "webshop"))

    rows: List[Dict[str, Any]] = []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                entry: Dict[str, Any] = {
                    "exp_id":        row["exp_id"],
                    "task_base":     float(row["task_base_threshold"]),
                    "task_step":     float(row["task_depth_step"]),
                    "env_base":      float(row["env_base_threshold"]),
                    "env_step":      float(row["env_depth_step"]),
                    "success_rate":  float(row["success_rate"]),
                    "task_hit_rate": float(row["task_hit_rate"]),
                    "env_hit_rate":  float(row["env_hit_rate"]),
                    "n_episodes":    int(row["n_episodes"]),
                    "split":         row.get("split", ""),
                }
                if is_mind2web:
                    entry["avg_element_acc"]       = float(row.get("avg_element_acc", 0))
                    entry["avg_action_f1"]         = float(row.get("avg_action_f1",   0))
                    entry["avg_step_success_rate"] = float(row.get("avg_step_success_rate", 0))
                elif is_sciworld:
                    entry["avg_reward"] = float(row.get("avg_reward", 0))
                    entry["avg_steps"]  = float(row.get("avg_steps",  0))
                else:
                    entry["avg_steps"] = float(row.get("avg_steps", 0))
                rows.append(entry)
            except (KeyError, ValueError):
                continue

    if not rows:
        logger.error(f"[{benchmark}] CSV 中无有效数据行。")
        return

    # 排序：按主指标降序；次键按 benchmark 类型区分
    if is_mind2web:
        rows.sort(key=lambda r: (-r["success_rate"], -r["avg_element_acc"], -r["avg_action_f1"]))
    elif is_sciworld:
        # ScienceWorld 主指标为 avg_reward（partial credit），次键为 success_rate
        rows.sort(key=lambda r: (-r["avg_reward"], -r["success_rate"]))
    else:
        rows.sort(key=lambda r: (-r["success_rate"], r["avg_steps"]))

    best_k = rows[:min(top_k, len(rows))]
    bm_upper = benchmark.upper()

    if is_mind2web:
        sep   = "─" * 112
        title = f"  [{bm_upper}] 联合阈值网格搜索 — Top-{top_k} 最优参数组合  (共 {len(rows)} 组)  "
        print(f"\n{'='*112}")
        print(f"{title:^112}")
        print(f"{'='*112}")
        print(f"{'#':>3}  {'task_base':>10}  {'task_step':>10}  {'env_base':>9}  {'env_step':>9}  "
              f"{'SR':>7}  {'ElemAcc':>8}  {'ActF1':>7}  {'StepSR':>7}  "
              f"{'TaskHit':>8}  {'EnvHit':>8}  {'Episodes':>9}")
        print(sep)
        for rank, r in enumerate(best_k, 1):
            marker = " ◀ BEST" if rank == 1 else ""
            print(
                f"{rank:>3}  {r['task_base']:>10.3f}  {r['task_step']:>10.3f}  "
                f"{r['env_base']:>9.3f}  {r['env_step']:>9.3f}  "
                f"{r['success_rate']:>7.2%}  {r['avg_element_acc']:>8.4f}  "
                f"{r['avg_action_f1']:>7.4f}  {r['avg_step_success_rate']:>7.4f}  "
                f"{r['task_hit_rate']:>8.2%}  {r['env_hit_rate']:>8.2%}  "
                f"{r['n_episodes']:>9}{marker}"
            )
        print(sep)
        best = best_k[0]
        print(f"\n★ [{bm_upper}] 最优参数组合：")
        print(f"   Task 树 → base_threshold = {best['task_base']:.3f}   depth_step = {best['task_step']:.3f}")
        print(f"   Env  树 → base_threshold = {best['env_base']:.3f}   depth_step = {best['env_step']:.3f}")
        print(f"   成功率      = {best['success_rate']:.2%}")
        print(f"   Element Acc = {best['avg_element_acc']:.4f}   Action F1 = {best['avg_action_f1']:.4f}"
              f"   Step SR = {best['avg_step_success_rate']:.4f}")
        print(f"   （Task 命中率 {best['task_hit_rate']:.2%} / Env 命中率 {best['env_hit_rate']:.2%}）")
        print(f"\n  exp_id: {best['exp_id']}")
        print(f"{'='*112}\n")
    elif is_sciworld:
        sep   = "─" * 108
        title = f"  [{bm_upper}] 联合阈值网格搜索 — Top-{top_k} 最优参数组合  (共 {len(rows)} 组)  "
        print(f"\n{'='*108}")
        print(f"{title:^108}")
        print(f"{'='*108}")
        print(f"{'#':>3}  {'task_base':>10}  {'task_step':>10}  {'env_base':>9}  {'env_step':>9}  "
              f"{'AvgReward':>10}  {'SR':>7}  {'Steps':>7}  "
              f"{'TaskHit':>8}  {'EnvHit':>8}  {'Episodes':>9}")
        print(sep)
        for rank, r in enumerate(best_k, 1):
            marker = " ◀ BEST" if rank == 1 else ""
            print(
                f"{rank:>3}  {r['task_base']:>10.3f}  {r['task_step']:>10.3f}  "
                f"{r['env_base']:>9.3f}  {r['env_step']:>9.3f}  "
                f"{r['avg_reward']:>10.4f}  {r['success_rate']:>7.2%}  {r['avg_steps']:>7.2f}  "
                f"{r['task_hit_rate']:>8.2%}  {r['env_hit_rate']:>8.2%}  "
                f"{r['n_episodes']:>9}{marker}"
            )
        print(sep)
        best = best_k[0]
        print(f"\n★ [{bm_upper}] 最优参数组合：")
        print(f"   Task 树 → base_threshold = {best['task_base']:.3f}   depth_step = {best['task_step']:.3f}")
        print(f"   Env  树 → base_threshold = {best['env_base']:.3f}   depth_step = {best['env_step']:.3f}")
        print(f"   平均奖励 = {best['avg_reward']:.4f}   成功率 = {best['success_rate']:.2%}   平均步数 = {best['avg_steps']:.2f}")
        print(f"   （Task 命中率 {best['task_hit_rate']:.2%} / Env 命中率 {best['env_hit_rate']:.2%}）")
        print(f"\n  exp_id: {best['exp_id']}")
        print(f"{'='*108}\n")
    else:
        sep   = "─" * 100
        title = f"  [{bm_upper}] 联合阈值网格搜索 — Top-{top_k} 最优参数组合  (共 {len(rows)} 组)  "
        print(f"\n{'='*100}")
        print(f"{title:^100}")
        print(f"{'='*100}")
        print(f"{'#':>3}  {'task_base':>10}  {'task_step':>10}  {'env_base':>9}  {'env_step':>9}  "
              f"{'SR':>7}  {'Steps':>7}  {'TaskHit':>8}  {'EnvHit':>8}  {'Episodes':>9}")
        print(sep)
        for rank, r in enumerate(best_k, 1):
            marker = " ◀ BEST" if rank == 1 else ""
            print(
                f"{rank:>3}  {r['task_base']:>10.3f}  {r['task_step']:>10.3f}  "
                f"{r['env_base']:>9.3f}  {r['env_step']:>9.3f}  "
                f"{r['success_rate']:>7.2%}  {r['avg_steps']:>7.2f}  "
                f"{r['task_hit_rate']:>8.2%}  {r['env_hit_rate']:>8.2%}  "
                f"{r['n_episodes']:>9}{marker}"
            )
        print(sep)
        best = best_k[0]
        print(f"\n★ [{bm_upper}] 最优参数组合：")
        print(f"   Task 树 → base_threshold = {best['task_base']:.3f}   depth_step = {best['task_step']:.3f}")
        print(f"   Env  树 → base_threshold = {best['env_base']:.3f}   depth_step = {best['env_step']:.3f}")
        print(f"   成功率   = {best['success_rate']:.2%}   平均步数 = {best['avg_steps']:.2f}")
        print(f"   （Task 命中率 {best['task_hit_rate']:.2%} / Env 命中率 {best['env_hit_rate']:.2%}）")
        print(f"\n  exp_id: {best['exp_id']}")
        print(f"{'='*100}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="PRTree 联合阈值网格搜索（ALFWorld / ScienceWorld / Mind2Web）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--mode", choices=["grid", "analyze"], default="grid",
                   help="grid: 运行全部实验\nanalyze: 仅分析已有 CSV")

    # benchmark 选择
    p.add_argument("--benchmark", type=str,
                   choices=["alfworld", "sciworld", "mind2web", "webshop", "all"],
                   default="alfworld",
                   help="要测试的 benchmark，all 表示全部四个")

    # 通用实验参数
    p.add_argument("--model",        default="gpt-4o-mini")
    p.add_argument("--icl-num",      type=int, default=1)
    p.add_argument("--max-episodes", type=int, default=None,
                   help="每组实验最大 episode 数；不传或传 0 则跑完整个 split")
    p.add_argument("--traj-dir",     type=str, default="trajectories/joint_threshold_search")
    p.add_argument("--output-csv",   type=str, default="results/joint_threshold_search",
                   help="CSV 路径前缀（不含扩展名也可），实际输出为 {prefix}_alfworld.csv 等")

    # 各 benchmark 独立 max_steps
    # ScienceWorld 的步数由内部 max_steps.json 按任务类型决定，此参数对其无效
    p.add_argument("--alfworld-max-steps", type=int, default=30,
                   help="ALFWorld 每局最大步数（默认 30）")
    p.add_argument("--mind2web-max-steps", type=int, default=10,
                   help="Mind2Web 每局最大步数（默认 10，若 agent 支持）")

    # 网格定义（默认 3×3×3×3 = 81 组）
    p.add_argument("--task-base-thresholds", default="0.75,0.80,0.85")
    p.add_argument("--task-depth-steps",     default="0.01,0.03,0.05")
    p.add_argument("--env-base-thresholds",  default="0.82,0.88,0.94")
    p.add_argument("--env-depth-steps",      default="0.01,0.02,0.04")

    # ALFWorld 专属
    p.add_argument("--alfworld-split",
                   choices=["eval_in_distribution", "eval_out_of_distribution"],
                   default="eval_in_distribution")
    p.add_argument("--alfworld-load-memory", type=str, default=None)

    # ScienceWorld 专属
    p.add_argument("--sciworld-split", choices=["train", "dev", "test"], default="test")
    p.add_argument("--sciworld-load-memory", type=str, default=None)

    # Mind2Web 专属
    p.add_argument("--mind2web-split",
                   choices=["test_task", "test_website", "test_domain"],
                   default="test_task")
    p.add_argument("--mind2web-load-memory", type=str, default=None)

    # WebShop 专属
    p.add_argument("--webshop-sessions-file", type=str,
                   default="/tmp/ETO/eval_agent/data/webshop/test_indices.json",
                   help="JSON list of session ids (默认 ETO test_indices)")
    p.add_argument("--webshop-max-steps", type=int, default=15,
                   help="WebShop 每局最大步数（默认 15）")
    p.add_argument("--webshop-load-memory", type=str, default=None)

    # 分析参数
    p.add_argument("--top-k", type=int, default=5)

    args = p.parse_args()

    # 确保 output_csv 以 .csv 结尾，方便 get_csv_path 切分
    if not args.output_csv.endswith(".csv"):
        args.output_csv += ".csv"

    benchmarks = (
        list(BENCHMARK_RUNNERS.keys())
        if args.benchmark == "all"
        else [args.benchmark]
    )

    if args.mode == "analyze":
        for bm in benchmarks:
            analyze_results(get_csv_path(args.output_csv, bm), bm, args.top_k)
        return

    # grid 模式
    run_grid_search(args)
    for bm in benchmarks:
        analyze_results(get_csv_path(args.output_csv, bm), bm, args.top_k)


if __name__ == "__main__":
    main()
