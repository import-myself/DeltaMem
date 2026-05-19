"""
Skill 固化阈值消融实验 (实验 2.3)
=====================================
验证 CONSOLIDATION_THRESHOLD 对 SkillCache 质量与双路路由效果的影响。

参数网格: threshold ∈ {1, 2, 3, 5, 8}

实验设置:
  - Benchmark  : ALFWorld eval_in_distribution（140 episodes）
  - 起始记忆   : 相同的离线 PRTree（已有树节点，SkillCache 清空）
  - 顺序执行评估集，触发固化后记录 SkillCache 状态

统计指标:
  threshold, success_rate, avg_steps,
  skill_cache_size,                  # 最终固化出的 Skill 数
  skill_avg_reuse_count,             # 平均每个 Skill 被命中次数
  skill_hit_rate,                    # 快路命中率
  task_hit_rate, env_hit_rate,
  avg_prompt_tokens,
  task_tree_total_nodes, env_tree_total_nodes,
  n_episodes, timestamp

运行示例:
  cd /hdd/REDACTED_USER/DeltaMem/ablation
  python run_consolidation_ablation.py \\
      --load-memory ../ALFWorld/storage/prtree_dual_offline \\
      --thresholds 1,2,3,5,8 \\
      --model deepseek-v4-flash \\
      --output-csv results/consolidation_ablation.csv
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

_THIS_DIR    = Path(__file__).parent.resolve()
_PRTREE_ROOT = _THIS_DIR.parent
_ALFWORLD    = _PRTREE_ROOT / "ALFWorld"

sys.path.insert(0, str(_PRTREE_ROOT))
sys.path.insert(0, str(_ALFWORLD))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =====================================================================
# 运行时 patch CONSOLIDATION_THRESHOLD
# =====================================================================

def _patch_consolidation_threshold(new_threshold: int) -> None:
    """
    在运行时将 CONSOLIDATION_THRESHOLD 替换为指定值。
    需要同时 patch consolidation 模块和 dual_tree_manager 中已导入的引用。
    """
    import memory.prtree.consolidation as cons_mod
    import memory.prtree.dual_tree_manager as dtm_mod

    cons_mod.CONSOLIDATION_THRESHOLD = new_threshold
    dtm_mod.CONSOLIDATION_THRESHOLD = new_threshold
    logger.info(f"[ConsolidationAblation] CONSOLIDATION_THRESHOLD patched → {new_threshold}")


# =====================================================================
# ALFWorld helpers
# =====================================================================

def _load_alfworld_env(split: str):
    import yaml
    import alfworld.agents.environment as environment

    split_sizes = {
        "train": 3553,
        "eval_in_distribution":     140,
        "eval_out_of_distribution": 134,
    }
    path = os.environ["ALFWORLD_DATA"]
    with open(os.path.join(path, "base_config.yaml")) as f:
        config = yaml.safe_load(f)
    env = environment.get_environment(config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    return env, split_sizes[split]


def _get_alfworld_task_type(game_file: str) -> Optional[str]:
    name = "/".join(game_file.split("/")[-3:-1])
    for prefix in ["pick_and_place", "pick_clean_then_place", "pick_heat_then_place",
                   "pick_cool_then_place", "look_at_obj", "pick_two_obj"]:
        if name.startswith(prefix):
            return prefix
    return None


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


# =====================================================================
# 单次实验运行
# =====================================================================

def run_one_threshold(args, threshold: int) -> Dict[str, Any]:
    _patch_consolidation_threshold(threshold)

    from agent_alfworld_dual import DualTreeReflectiveAgent
    from common.llm_client import create_llm_client

    split = args.split
    env, n_tasks = _load_alfworld_env(split)
    n_episodes = min(args.max_episodes or n_tasks, n_tasks)

    llm = create_llm_client(args.model)
    agent = DualTreeReflectiveAgent(
        agent_name=f"ConsolidationAblation_K{threshold}",
        llm_client=llm,
        icl_num=args.icl_num,
        icl_data_path=str(_ALFWORLD / "data" / "alfworld_icl.json"),
    )

    # 加载已有 PRTree，清空 SkillCache（确保从零开始固化）
    if args.load_memory:
        agent.load_memory(args.load_memory)
        agent.dual_memory.skill_cache.patches.clear()
        agent.dual_memory.skill_cache._embeddings.clear()
        # 重置所有节点的固化标记，使其可被新阈值重新固化
        for node in agent.dual_memory.task_tree.node_index.values():
            if "GLOBAL_ROOT_PLACEHOLDER" not in node.payload.get("scenario_description", ""):
                node.meta["is_consolidated"] = False
        logger.info(f"[K={threshold}] Loaded PRTree, SkillCache cleared, consolidation flags reset.")
        stats = agent.get_memory_stats()
        logger.info(f"  task_nodes={stats['task_tree_nodes']}, env_nodes={stats['env_tree_nodes']}")

    exp_id   = f"alfworld__consolidation_K{threshold}__{split}"
    traj_dir = Path(args.traj_dir or "trajectories/consolidation_ablation") / exp_id
    traj_dir.mkdir(parents=True, exist_ok=True)

    results       = []
    skill_reuses  = {}   # skill patch source_node_id → hit_count（近似用 skill_cache 大小快照）

    for ep_idx in range(n_episodes):
        obs, info = env.reset()
        task_instruction = "\n".join(obs[0].split("\n\n")[1:])
        task_type = _get_alfworld_task_type(info["extra.gamefile"][0])

        result = agent.run_episode(
            task_instruction=task_instruction,
            env=env,
            task_type=task_type,
            max_steps=args.max_steps,
            episode_idx=ep_idx,
        )
        results.append(result)

        if result.get("fast_path_used", False) or result.get("skill_hit", False):
            # 记录快路命中，用于计算 avg_reuse_count（粗略统计）
            node_id = result.get("task_node_id", "unknown")
            skill_reuses[node_id] = skill_reuses.get(node_id, 0) + 1

        with open(traj_dir / f"{ep_idx}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if (ep_idx + 1) % 10 == 0:
            sr       = sum(r.get("success", False) for r in results) / len(results)
            sh       = sum(r.get("fast_path_used", False) or r.get("skill_hit", False)
                          for r in results) / len(results)
            sc_size  = len(agent.dual_memory.skill_cache)
            logger.info(f"  [K={threshold}] Ep {ep_idx+1}/{n_episodes}: "
                        f"SR={sr:.2%}, SkillHit={sh:.2%}, SkillCacheSize={sc_size}")

    n           = len(results)
    skill_cache = agent.dual_memory.skill_cache
    mem_stats   = agent.get_memory_stats()

    skill_hits_n = sum(
        r.get("fast_path_used", False) or r.get("skill_hit", False) for r in results
    )
    skill_reuse_counts = list(skill_reuses.values())
    avg_reuse = round(sum(skill_reuse_counts) / len(skill_reuse_counts), 4) if skill_reuse_counts else 0.0

    task_hits = [r for r in results if r.get("task_memory_used", False)]
    env_hits  = [r for r in results if r.get("env_memory_used",  False)]

    row = {
        "threshold":               threshold,
        "split":                   split,
        "n_episodes":              n,
        "success_rate":            round(sum(r.get("success", False) for r in results) / n, 6),
        "avg_steps":               round(sum(r.get("steps", 0) for r in results) / n, 4),
        "skill_cache_size":        len(skill_cache),
        "skill_hit_rate":          round(skill_hits_n / n, 6),
        "skill_avg_reuse_count":   avg_reuse,
        "task_hit_rate":           round(len(task_hits) / n, 6),
        "env_hit_rate":            round(len(env_hits)  / n, 6),
        "avg_prompt_tokens":       round(sum(r.get("prompt_tokens", 0) for r in results) / n, 2),
        "task_tree_total_nodes":   mem_stats.get("task_tree_nodes", 0),
        "env_tree_total_nodes":    mem_stats.get("env_tree_nodes", 0),
        "task_tree_level_counts":  json.dumps(get_tree_level_stats(agent.dual_memory.task_tree)),
        "timestamp":               time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return row


# =====================================================================
# CSV
# =====================================================================

CSV_FIELDNAMES = [
    "threshold", "split", "n_episodes",
    "success_rate", "avg_steps",
    "skill_cache_size", "skill_hit_rate", "skill_avg_reuse_count",
    "task_hit_rate", "env_hit_rate",
    "avg_prompt_tokens",
    "task_tree_total_nodes", "env_tree_total_nodes",
    "task_tree_level_counts",
    "timestamp",
]


def append_to_csv(filepath: str, row: Dict[str, Any]) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    exists = Path(filepath).exists() and Path(filepath).stat().st_size > 0
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"Appended to {filepath}")


# =====================================================================
# main
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description="Skill 固化阈值消融实验（实验 2.3）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--thresholds",    type=str, default="1,2,3,5,8",
                   help="逗号分隔的 CONSOLIDATION_THRESHOLD 值列表")
    p.add_argument("--load-memory",   type=str, default=None,
                   help="预构建的离线 PRTree 路径（SkillCache 会被清空）")
    p.add_argument("--split",
                   choices=["eval_in_distribution", "eval_out_of_distribution"],
                   default="eval_in_distribution")
    p.add_argument("--model",         type=str, default="deepseek-v4-flash")
    p.add_argument("--icl-num",       type=int, default=1)
    p.add_argument("--max-episodes",  type=int, default=None)
    p.add_argument("--max-steps",     type=int, default=30)
    p.add_argument("--output-csv",    type=str, default="results/consolidation_ablation.csv")
    p.add_argument("--traj-dir",      type=str, default=None)

    args = p.parse_args()

    try:
        thresholds = [int(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    except ValueError:
        p.error("--thresholds 必须为逗号分隔的整数列表，如 '1,2,3,5,8'")

    logger.info("=" * 70)
    logger.info("Skill 固化阈值消融实验 (实验 2.3)")
    logger.info(f"  Thresholds  : {thresholds}")
    logger.info(f"  Split       : {args.split}")
    logger.info(f"  Model       : {args.model}")
    logger.info(f"  Load memory : {args.load_memory}")
    logger.info(f"  Output CSV  : {args.output_csv}")
    logger.info("=" * 70)

    for i, k in enumerate(thresholds):
        logger.info(f"\n[{i+1}/{len(thresholds)}] threshold={k}")
        try:
            row = run_one_threshold(args, k)
            append_to_csv(args.output_csv, row)
            logger.info(f"  SR={row['success_rate']}, SkillCacheSize={row['skill_cache_size']}, "
                        f"SkillHitRate={row['skill_hit_rate']}, AvgReuse={row['skill_avg_reuse_count']}")
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\nAll done → {args.output_csv}")


if __name__ == "__main__":
    main()
