"""
快慢双路路由消融实验（3 benchmarks）
=====================================
实验类型：
  - no_memory     : 不使用任何记忆（baseline）
  - skill_only    : 只用 SkillCache 快路；未命中时退化为无记忆
  - episodic_only : 只用慢路 DFS 残差检索；屏蔽 SkillCache
  - dual_routing  : SkillCache 优先，未命中回退慢路（完整系统）

支持的 benchmark：
  - alfworld   : ALFWorld (eval_in_distribution / eval_out_of_distribution)
  - sciworld   : ScienceWorld (dev / test)
  - mind2web   : Mind2Web (test_task / test_website / test_domain)

统计指标（统一 CSV 格式）：
  benchmark, routing_mode, split,
  success_rate, avg_steps（ALFWorld/SciWorld）| avg_element_acc/avg_action_f1/avg_step_sr（Mind2Web）,
  skill_hit_rate, skill_cache_size,
  task_hit_rate, env_hit_rate,
  avg_task_retrieval_len_all, avg_env_retrieval_len_all,
  avg_task_retrieval_len_hit, avg_env_retrieval_len_hit,
  avg_prompt_tokens,
  task_tree_total_nodes, env_tree_total_nodes,
  n_episodes, timestamp

运行示例：
  # ALFWorld 全部 4 种路由模式
  python run_routing_ablation.py \\
      --benchmark alfworld \\
      --routing-modes no_memory,skill_only,episodic_only,dual_routing \\
      --model deepseek-v4-flash \\
      --alfworld-split eval_in_distribution \\
      --alfworld-load-memory ../ALFWorld/storage/prtree_dual_offline \\
      --output-csv results/routing_ablation.csv

  # 全部 3 个 benchmark
  python run_routing_ablation.py \\
      --benchmark all \\
      --alfworld-load-memory ../ALFWorld/storage/prtree_dual_offline \\
      --sciworld-load-memory ../ScienceWorld/storage/prtree_sciworld_offline \\
      --mind2web-load-memory ../Mind2web/storage/prtree_mind2web_offline \\
      --output-csv results/routing_ablation.csv
"""

import os
import sys
import csv
import json
import time
import logging
import argparse
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from collections import deque
from typing import Dict, List, Any, Optional

_THIS_DIR    = Path(__file__).parent.resolve()
_PRTREE_ROOT = _THIS_DIR.parent
_ALFWORLD    = _PRTREE_ROOT / "ALFWorld"
_SCIWORLD    = _PRTREE_ROOT / "ScienceWorld"
_MIND2WEB    = _PRTREE_ROOT / "Mind2web"

sys.path.insert(0, str(_PRTREE_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =====================================================================
# 路由模式上下文管理器
# =====================================================================

@contextmanager
def routing_mode_context(agent, routing_mode: str):
    """
    通过 monkey-patch agent 的 SkillCache 行为来实现路由模式切换，
    结束后自动恢复原始行为。

    - no_memory     : 不改变 SkillCache，run_episode 时传 no_memory=True
    - skill_only    : 不改变 SkillCache（让快路正常工作）
    - episodic_only : patch check_match 永远返回 None（禁用快路）
    - dual_routing  : 不做任何 patch（完整系统）
    """
    skill_cache = agent.dual_memory.skill_cache
    original_check = skill_cache.check_match

    if routing_mode == "episodic_only":
        skill_cache.check_match = lambda state: None
        logger.info("[routing] episodic_only: SkillCache patched → always None")

    try:
        yield
    finally:
        if routing_mode == "episodic_only":
            skill_cache.check_match = original_check
            logger.info("[routing] episodic_only: SkillCache restored")


def _run_episode_with_routing(agent, routing_mode: str, run_kwargs: dict) -> dict:
    """
    统一 episode 执行入口，根据 routing_mode 决定调用参数与 SkillCache 行为。

    对于 skill_only：
      - 先手动查 SkillCache；命中则用 external_memory_str 注入，
        同时 patch check_match 为 None 防止 run_episode 内部二次触发
      - 未命中则用 no_memory=True
    """
    skill_cache = agent.dual_memory.skill_cache
    original_check = skill_cache.check_match

    if routing_mode == "no_memory":
        run_kwargs["no_memory"] = True
        result = agent.run_episode(**run_kwargs)
        result["skill_hit"] = False

    elif routing_mode == "skill_only":
        # 手动检查 SkillCache
        task_goal    = run_kwargs.get("task_instruction", "")
        env_desc     = run_kwargs.get("task_instruction", "")  # alfworld 合并在一起，拆分逻辑在 agent 内部
        skill_state  = {"task": task_goal, "env": env_desc}
        matched      = skill_cache.check_match(skill_state)

        if matched:
            # 快路命中：用 external_memory_str 传入，禁用内部再次触发
            skill_cache.check_match = lambda _: None
            run_kwargs["external_memory_str"] = (
                "# Skill-Based Memory (Fast Path)\n\n"
                f"**Activation Condition**: {matched.activation_condition}\n\n"
                f"**Execution Procedure**:\n{matched.execution_procedure}\n\n"
                f"**Termination Condition**: {matched.termination_condition}"
            )
            run_kwargs["no_prtree_update"] = True
            try:
                result = agent.run_episode(**run_kwargs)
            finally:
                skill_cache.check_match = original_check
            result["skill_hit"] = True
        else:
            # 快路未命中：退化为无记忆
            run_kwargs["no_memory"] = True
            result = agent.run_episode(**run_kwargs)
            result["skill_hit"] = False

    elif routing_mode == "episodic_only":
        # 禁用快路
        skill_cache.check_match = lambda _: None
        try:
            result = agent.run_episode(**run_kwargs)
        finally:
            skill_cache.check_match = original_check
        result["skill_hit"] = False

    else:  # dual_routing
        result = agent.run_episode(**run_kwargs)
        # 判断快路是否命中：task_retrieval_length==0 且 task_memory_used==True 通常表示快路
        # 更准确：在 result 中检查是否包含 'fast_path_used'（agent 可能不返回此字段）
        result["skill_hit"] = result.get("fast_path_used", False)

    return result


# =====================================================================
# 通用统计工具（复用 memory_mode_ablation 模式）
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


def _base_row(benchmark: str, routing_mode: str, split: str, n: int,
               results: List[Dict], agent) -> Dict[str, Any]:
    task_hits     = [r for r in results if r.get("task_memory_used", False)]
    env_hits      = [r for r in results if r.get("env_memory_used",  False)]
    skill_hits    = [r for r in results if r.get("skill_hit", False)]
    task_hit_lens = [r.get("task_retrieval_length", 0) for r in task_hits]
    env_hit_lens  = [r.get("env_retrieval_length",  0) for r in env_hits]

    mem_stats   = agent.get_memory_stats()
    task_level  = get_tree_level_stats(agent.dual_memory.task_tree)
    env_level   = get_tree_level_stats(agent.dual_memory.env_tree)
    skill_cache = agent.dual_memory.skill_cache

    return {
        "benchmark":                   benchmark,
        "routing_mode":                routing_mode,
        "split":                       split,
        "n_episodes":                  n,
        "skill_hit_rate":              round(len(skill_hits) / n, 6),
        "skill_cache_size":            len(skill_cache),
        "task_hit_rate":               round(len(task_hits) / n, 6),
        "env_hit_rate":                round(len(env_hits)  / n, 6),
        "avg_task_retrieval_len_all":  round(sum(r.get("task_retrieval_length", 0) for r in results) / n, 4),
        "avg_env_retrieval_len_all":   round(sum(r.get("env_retrieval_length",  0) for r in results) / n, 4),
        "avg_task_retrieval_len_hit":  round(sum(task_hit_lens) / len(task_hit_lens), 4) if task_hit_lens else 0.0,
        "avg_env_retrieval_len_hit":   round(sum(env_hit_lens)  / len(env_hit_lens),  4) if env_hit_lens  else 0.0,
        "avg_prompt_tokens":           round(sum(r.get("prompt_tokens", 0) for r in results) / n, 2),
        "task_tree_total_nodes":       mem_stats["task_tree_nodes"],
        "env_tree_total_nodes":        mem_stats["env_tree_nodes"],
        "task_tree_level_counts":      json.dumps(task_level),
        "env_tree_level_counts":       json.dumps(env_level),
        "timestamp":                   time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# =====================================================================
# ALFWorld
# =====================================================================

def _load_alfworld_env(split: str):
    import yaml
    import alfworld.agents.environment as environment

    split_sizes = {
        "train": 3553,
        "eval_in_distribution": 140,
        "eval_out_of_distribution": 134,
    }
    if split not in split_sizes:
        raise ValueError(f"Unknown ALFWorld split: {split}")
    n = split_sizes[split]

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


def run_alfworld_routing(args, routing_mode: str) -> Dict[str, Any]:
    if str(_ALFWORLD) not in sys.path:
        sys.path.insert(0, str(_ALFWORLD))
    from agent_alfworld_dual import DualTreeReflectiveAgent
    from common.llm_client import create_llm_client

    split = args.alfworld_split
    env, n_tasks = _load_alfworld_env(split)
    n_episodes = min(args.max_episodes or n_tasks, n_tasks)

    llm_client = create_llm_client(args.model)
    agent = DualTreeReflectiveAgent(
        agent_name=f"RoutingAblation_{routing_mode}",
        llm_client=llm_client,
        icl_num=args.icl_num,
        icl_data_path=str(_ALFWORLD / "data" / "alfworld_icl.json"),
    )
    if args.alfworld_load_memory:
        agent.load_memory(args.alfworld_load_memory)
        stats = agent.get_memory_stats()
        logger.info(f"[ALFWorld] Loaded: task={stats['task_tree_nodes']}, env={stats['env_tree_nodes']}, "
                    f"skill_cache={len(agent.dual_memory.skill_cache)}")

    exp_id   = f"alfworld__{routing_mode}__{split}"
    traj_dir = os.path.join(args.traj_dir or "trajectories/routing_ablation", exp_id)
    os.makedirs(traj_dir, exist_ok=True)

    results = []
    for ep_idx in range(n_episodes):
        obs, info = env.reset()
        task_instruction = "\n".join(obs[0].split("\n\n")[1:])
        task_type = _get_alfworld_task_type(info["extra.gamefile"][0])

        run_kwargs = dict(
            task_instruction=task_instruction,
            env=env,
            task_type=task_type,
            max_steps=args.max_steps,
            episode_idx=ep_idx,
        )
        result = _run_episode_with_routing(agent, routing_mode, run_kwargs)
        results.append(result)

        with open(os.path.join(traj_dir, f"{ep_idx}.json"), "w", encoding="utf-8") as f:
            json.dump(result if isinstance(result, dict) else {}, f, indent=2, ensure_ascii=False)

        if (ep_idx + 1) % 10 == 0:
            sr = sum(r.get("success", False) for r in results) / len(results)
            sh = sum(r.get("skill_hit", False) for r in results) / len(results)
            logger.info(f"  [ALFWorld/{routing_mode}] Ep {ep_idx+1}/{n_episodes}: SR={sr:.2%}, SkillHit={sh:.2%}")

    n = len(results)
    row = _base_row("alfworld", routing_mode, split, n, results, agent)
    row["success_rate"]          = round(sum(r.get("success", False) for r in results) / n, 6)
    row["avg_steps"]             = round(sum(r.get("steps", 0) for r in results) / n, 4)
    row["avg_element_acc"]       = "N/A"
    row["avg_action_f1"]         = "N/A"
    row["avg_step_success_rate"] = "N/A"
    return row


# =====================================================================
# ScienceWorld
# =====================================================================

def run_sciworld_routing(args, routing_mode: str) -> Dict[str, Any]:
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
    with open(split_file[split]) as f:
        task_idxs = json.load(f)

    sciworld_monkey_patch()
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv()

    n_episodes = min(args.max_episodes or len(task_idxs), len(task_idxs))

    llm_client = create_llm_client(args.model)
    agent = DualTreeSciWorldAgent(
        agent_name=f"RoutingAblation_{routing_mode}",
        llm_client=llm_client,
        icl_num=args.icl_num,
        icl_data_path=str(_SCIWORLD / "data/sciworld_icl.json"),
    )
    if args.sciworld_load_memory:
        agent.load_memory(args.sciworld_load_memory)
        stats = agent.get_memory_stats()
        logger.info(f"[SciWorld] Loaded: task={stats['task_tree_nodes']}, env={stats['env_tree_nodes']}, "
                    f"skill_cache={len(agent.dual_memory.skill_cache)}")

    exp_id   = f"sciworld__{routing_mode}__{split}"
    traj_dir = os.path.join(args.traj_dir or "trajectories/routing_ablation", exp_id)
    os.makedirs(traj_dir, exist_ok=True)

    results = []
    for ep_idx, (task_name, variation_idx) in enumerate(task_idxs[:n_episodes]):
        run_kwargs = dict(
            env=env,
            task_name=task_name,
            variation_idx=variation_idx,
            episode_idx=ep_idx,
        )
        result = _run_episode_with_routing(agent, routing_mode, run_kwargs)
        results.append(result)

        fname = f"{ep_idx}_{task_name}_var{variation_idx}.json"
        with open(os.path.join(traj_dir, fname), "w", encoding="utf-8") as f:
            json.dump(result if isinstance(result, dict) else {}, f, indent=2, ensure_ascii=False)

        if (ep_idx + 1) % 10 == 0:
            sr = sum(r.get("success", False) for r in results) / len(results)
            sh = sum(r.get("skill_hit", False) for r in results) / len(results)
            logger.info(f"  [SciWorld/{routing_mode}] Ep {ep_idx+1}/{n_episodes}: SR={sr:.2%}, SkillHit={sh:.2%}")

    n = len(results)
    row = _base_row("sciworld", routing_mode, split, n, results, agent)
    row["success_rate"]          = round(sum(r.get("success", False) for r in results) / n, 6)
    row["avg_steps"]             = round(sum(r.get("steps", r.get("reward", 0)) for r in results) / n, 4)
    row["avg_element_acc"]       = "N/A"
    row["avg_action_f1"]         = "N/A"
    row["avg_step_success_rate"] = "N/A"
    return row


# =====================================================================
# Mind2Web
# =====================================================================

def run_mind2web_routing(args, routing_mode: str) -> Dict[str, Any]:
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
        samples = samples[:args.max_episodes]

    n_tasks = len(samples)
    llm_client = create_llm_client(args.model)
    agent = DualTreeMind2WebAgent(
        agent_name=f"RoutingAblation_{routing_mode}",
        llm_client=llm_client,
        exemplar_path=os.path.join(data_dir, "example", "exemplars.json"),
    )
    if args.mind2web_load_memory:
        agent.load_memory(args.mind2web_load_memory)
        stats = agent.get_memory_stats()
        logger.info(f"[Mind2Web] Loaded: task={stats['task_tree_nodes']}, env={stats['env_tree_nodes']}, "
                    f"skill_cache={len(agent.dual_memory.skill_cache)}")

    exp_id   = f"mind2web__{routing_mode}__{benchmark_split}"
    traj_dir = os.path.join(args.traj_dir or "trajectories/routing_ablation", exp_id)
    os.makedirs(traj_dir, exist_ok=True)

    results = []
    for ep_idx, sample in enumerate(samples):
        try:
            run_kwargs = dict(sample=sample, model_name=args.model, episode_idx=ep_idx)
            result = _run_episode_with_routing(agent, routing_mode, run_kwargs)
        except Exception as e:
            logger.error(f"Episode {ep_idx} failed: {e}")
            result = {
                "success": False, "element_acc": [], "action_f1": [], "step_success": [],
                "memory_used": False, "task_memory_used": False, "env_memory_used": False,
                "task_retrieval_length": 0, "env_retrieval_length": 0,
                "skill_hit": False,
            }
        results.append(result)
        with open(os.path.join(traj_dir, f"{ep_idx}.json"), "w", encoding="utf-8") as f:
            json.dump(result if isinstance(result, dict) else {}, f, indent=2, ensure_ascii=False)

        if (ep_idx + 1) % 10 == 0:
            sr = sum(r.get("success", False) for r in results) / len(results)
            sh = sum(r.get("skill_hit", False) for r in results) / len(results)
            logger.info(f"  [Mind2Web/{routing_mode}] Ep {ep_idx+1}/{n_tasks}: SR={sr:.2%}, SkillHit={sh:.2%}")

    n = len(results)
    metrics = calculate_metrics(results)
    row = _base_row("mind2web", routing_mode, benchmark_split, n, results, agent)
    row["success_rate"]          = round(sum(r.get("success", False) for r in results) / n, 6)
    row["avg_steps"]             = "N/A"
    row["avg_element_acc"]       = round(metrics.get("element_acc", 0.0), 4)
    row["avg_action_f1"]         = round(metrics.get("action_f1", 0.0),   4)
    row["avg_step_success_rate"] = round(metrics.get("step_sr", 0.0),     4)
    return row


# =====================================================================
# CSV 追加写入
# =====================================================================

CSV_FIELDNAMES = [
    "benchmark", "routing_mode", "split", "n_episodes",
    "success_rate", "avg_steps",
    "avg_element_acc", "avg_action_f1", "avg_step_success_rate",
    "skill_hit_rate", "skill_cache_size",
    "task_hit_rate", "env_hit_rate",
    "avg_task_retrieval_len_all", "avg_env_retrieval_len_all",
    "avg_task_retrieval_len_hit", "avg_env_retrieval_len_hit",
    "avg_prompt_tokens",
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
    logger.info(f"Appended to {filepath}")


# =====================================================================
# 主流程
# =====================================================================

BENCHMARK_RUNNERS = {
    "alfworld": run_alfworld_routing,
    "sciworld": run_sciworld_routing,
    "mind2web": run_mind2web_routing,
}

VALID_ROUTING_MODES = {"no_memory", "skill_only", "episodic_only", "dual_routing"}


def main():
    p = argparse.ArgumentParser(
        description="快慢双路路由消融实验，支持 3 benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--benchmark", choices=["alfworld", "sciworld", "mind2web", "all"],
                   default="alfworld")
    p.add_argument("--routing-modes", type=str,
                   default="no_memory,skill_only,episodic_only,dual_routing",
                   help="逗号分隔，可选: no_memory,skill_only,episodic_only,dual_routing")
    p.add_argument("--model",        type=str, default="deepseek-v4-flash")
    p.add_argument("--icl-num",      type=int, default=1)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--max-steps",    type=int, default=30)

    p.add_argument("--output-csv",   type=str, default="results/routing_ablation.csv")
    p.add_argument("--traj-dir",     type=str, default=None)

    p.add_argument("--alfworld-split",
                   choices=["eval_in_distribution", "eval_out_of_distribution"],
                   default="eval_in_distribution")
    p.add_argument("--alfworld-load-memory", type=str, default=None)

    p.add_argument("--sciworld-split", choices=["dev", "test"], default="dev")
    p.add_argument("--sciworld-load-memory", type=str, default=None)

    p.add_argument("--mind2web-split",
                   choices=["test_task", "test_website", "test_domain"],
                   default="test_task")
    p.add_argument("--mind2web-load-memory", type=str, default=None)

    args = p.parse_args()

    modes = [m.strip() for m in args.routing_modes.split(",") if m.strip()]
    invalid = set(modes) - VALID_ROUTING_MODES
    if invalid:
        p.error(f"Invalid routing modes: {invalid}. Valid: {VALID_ROUTING_MODES}")

    benchmarks = (
        list(BENCHMARK_RUNNERS.keys()) if args.benchmark == "all" else [args.benchmark]
    )

    logger.info("=" * 70)
    logger.info("快慢双路路由消融实验")
    logger.info(f"  Benchmarks    : {benchmarks}")
    logger.info(f"  Routing modes : {modes}")
    logger.info(f"  Model         : {args.model}")
    logger.info(f"  Max episodes  : {args.max_episodes or 'full split'}")
    logger.info(f"  Output CSV    : {args.output_csv}")
    logger.info("=" * 70)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    total = len(benchmarks) * len(modes)
    done  = 0

    for bm in benchmarks:
        runner = BENCHMARK_RUNNERS[bm]
        for mode in modes:
            done += 1
            logger.info(f"\n[{done}/{total}] benchmark={bm}, routing_mode={mode}")
            try:
                row = runner(args, routing_mode=mode)
                append_to_csv(args.output_csv, row)
                logger.info(f"  SR={row['success_rate']}, SkillHitRate={row['skill_hit_rate']}, "
                            f"SkillCacheSize={row['skill_cache_size']}")
            except Exception as e:
                logger.error(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()

    logger.info(f"\nAll done → {args.output_csv}")


if __name__ == "__main__":
    main()
