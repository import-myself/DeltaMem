"""
DeltaMem 参数敏感性扫描 — ALFWorld eval_in_distribution / SciWorld dev

扫描模式：
  grid_sweep : tb × eb 联合网格，固定 K（用 --k 指定，0=∞）
  single     : 单点运行（调试用）

网格范围（最优点附近 ±2~3 格）：
  ALFWorld : tb ∈ {0.70,0.75,0.80,0.85} × eb ∈ {0.80,0.85,0.88,0.91}   → 16 组
  SciWorld : tb ∈ {0.80,0.82,0.84,0.86} × eb ∈ {0.82,0.84,0.86,0.88}   → 16 组

最优参数（中心点）：
  ALFWorld in-dist : tb=0.75, eb=0.85, K=3
  SciWorld dev     : tb=0.84, eb=0.86, K=3

用法：
  python run_param_sweep.py --benchmark alfworld --sweep-type grid_sweep --k 3
  python run_param_sweep.py --benchmark sciworld --sweep-type grid_sweep --k 0
  python run_param_sweep.py --benchmark sciworld --sweep-type single --tb 0.84 --eb 0.86 --k 3
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── 路径 ─────────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent
_ABLATION = _THIS_DIR.parent
_ROOT     = _ABLATION.parent
_SCIWORLD = _ROOT / "ScienceWorld"
_ALFWORLD = _ROOT / "ALFWorld"

# ── 各 Benchmark 最优参数 ────────────────────────────────────────────────────
OPT = {
    "sciworld": {"tb": 0.84, "eb": 0.86, "k": 3},
    "alfworld": {"tb": 0.75, "eb": 0.85, "k": 3},
}

# ── 网格搜索范围（tb×eb 联合，最优点附近 ±2~3 格）─────────────────────────────
GRID_VALUES = {
    "alfworld": {
        "tb": [0.70, 0.75, 0.80, 0.85],
        "eb": [0.80, 0.85, 0.88, 0.91],
    },
    "sciworld": {
        "tb": [0.80, 0.82, 0.84, 0.86],
        "eb": [0.82, 0.84, 0.86, 0.88],
    },
}

CSV_FIELDS = [
    "benchmark", "sweep_type", "tb", "eb", "k",
    "n_episodes", "avg_reward", "sr", "avg_steps",
    "task_tree_total_nodes", "env_tree_total_nodes",
    "task_tree_max_depth",   "env_tree_max_depth",
    "task_tree_level_counts", "env_tree_level_counts",
    "consolidation_count",
    "task_hit_rate", "env_hit_rate",
    "timestamp",
]


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def _get_tree_depth(level_counts: Dict) -> int:
    if not level_counts:
        return 0
    return max(int(k) for k in level_counts.keys())


def _build_level_counts(tree) -> Dict:
    counts: Dict[str, int] = {}
    for node in tree.node_index.values():
        depth = len(node.get_path_to_root()) - 1
        counts[str(depth)] = counts.get(str(depth), 0) + 1
    return counts


def _already_done(path: str, benchmark: str, tb: float, eb: float, k: int) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                if (row.get("benchmark") == benchmark
                        and abs(float(row.get("tb", -1)) - tb) < 1e-9
                        and abs(float(row.get("eb", -1)) - eb) < 1e-9
                        and int(row.get("k", -999)) == k):
                    return True
            except (ValueError, TypeError):
                continue
    return False


def _append_csv(path: str, row: Dict) -> None:
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _make_summary_row(
    benchmark: str, sweep_type: str, tb: float, eb: float, k: int,
    results: List[Dict], agent, task_hits: List[bool], env_hits: List[bool],
) -> Dict:
    import datetime
    n = len(results)
    avg_reward = sum(r.get("reward", 0.0) for r in results) / n
    sr_val     = sum(bool(r.get("success", False)) for r in results) / n
    avg_steps  = sum(r.get("steps", 0) for r in results) / n

    mem_stats = agent.get_memory_stats()
    task_lc   = _build_level_counts(agent.dual_memory.task_tree)
    env_lc    = _build_level_counts(agent.dual_memory.env_tree)

    return {
        "benchmark":              benchmark,
        "sweep_type":             sweep_type,
        "tb":                     tb,
        "eb":                     eb,
        "k":                      k,
        "n_episodes":             n,
        "avg_reward":             round(avg_reward, 6),
        "sr":                     round(sr_val, 6),
        "avg_steps":              round(avg_steps, 4),
        "task_tree_total_nodes":  mem_stats.get("task_tree_nodes", 0),
        "env_tree_total_nodes":   mem_stats.get("env_tree_nodes", 0),
        "task_tree_max_depth":    _get_tree_depth(task_lc),
        "env_tree_max_depth":     _get_tree_depth(env_lc),
        "task_tree_level_counts": json.dumps(task_lc),
        "env_tree_level_counts":  json.dumps(env_lc),
        "consolidation_count":    agent.dual_memory.consolidation_count,
        "task_hit_rate":          round(sum(task_hits) / n, 6),
        "env_hit_rate":           round(sum(env_hits)  / n, 6),
        "timestamp":              datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── SciWorld runner ───────────────────────────────────────────────────────────
def run_sciworld_one(
    sweep_type: str, tb: float, eb: float, k: int,
    split: str, icl_num: int, model: str,
    traj_dir: str, output_csv: str, max_episodes: int = None,
) -> Dict:
    if str(_SCIWORLD) not in sys.path:
        sys.path.insert(0, str(_SCIWORLD))
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from agent_sciworld_dual import DualTreeSciWorldAgent
    from common.llm_client import create_llm_client
    from utils import sciworld_monkey_patch

    split_file = {
        "dev":  str(_SCIWORLD / "data/sciworld/dev_indices.json"),
        "test": str(_SCIWORLD / "data/sciworld/test_indices.json"),
    }
    with open(split_file[split]) as f:
        task_idxs = json.load(f)

    sciworld_monkey_patch()
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv()

    n = min(max_episodes or len(task_idxs), len(task_idxs))
    k_val = k if k > 0 else 10_000

    agent = DualTreeSciWorldAgent(
        agent_name=f"Sweep_{sweep_type}_tb{tb}_eb{eb}_k{k}",
        llm_client=create_llm_client(model),
        icl_num=icl_num,
        icl_data_path=str(_SCIWORLD / "data/sciworld_icl.json"),
        max_steps_path=str(_SCIWORLD / "data/sciworld/max_steps.json"),
        taskname2id_path=str(_SCIWORLD / "data/sciworld/taskname2id.json"),
        task_base_threshold=tb,
        env_base_threshold=eb,
        consolidation_threshold=k_val,
    )

    ep_dir = os.path.join(traj_dir, f"sci__{sweep_type}__tb{tb}_eb{eb}_k{k}")
    os.makedirs(ep_dir, exist_ok=True)

    results, task_hits, env_hits = [], [], []
    for ep_idx, (task_name, variation_idx) in enumerate(task_idxs[:n]):
        messages = agent.run_episode(env=env, task_name=task_name,
                                     variation_idx=variation_idx, episode_idx=ep_idx)
        result = messages[-1]
        results.append(result)
        task_hits.append(bool(result.get("task_memory_used", False)))
        env_hits.append(bool(result.get("env_memory_used", False)))
        with open(os.path.join(ep_dir, f"{ep_idx}_{task_name}_var{variation_idx}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        if (ep_idx + 1) % 20 == 0 or ep_idx == n - 1:
            avg_r = sum(r["reward"] for r in results) / len(results)
            logger.info(f"  [sci/{sweep_type}] tb={tb} eb={eb} k={k}  "
                        f"Ep {ep_idx+1}/{n}: AvgRew={avg_r:.4f}")

    row = _make_summary_row("sciworld", sweep_type, tb, eb, k,
                            results, agent, task_hits, env_hits)
    _append_csv(output_csv, row)
    logger.info(f"  ✅ sci tb={tb} eb={eb} k={k} "
                f"AvgRew={row['avg_reward']:.4f} SR={row['sr']:.2%} "
                f"task_nodes={row['task_tree_total_nodes']} consol={row['consolidation_count']}")
    return row


# ── ALFWorld runner ───────────────────────────────────────────────────────────
def _load_alfworld_env(split: str):
    import yaml
    import alfworld.agents.environment as environment

    split_sizes = {
        "eval_in_distribution":     140,
        "eval_out_of_distribution": 134,
        "train":                    3553,
    }
    if split not in split_sizes:
        raise ValueError(f"Unknown ALFWorld split: {split}")

    alfworld_data = os.environ.get("ALFWORLD_DATA",
                                   str(_ALFWORLD / "data/alfworld"))
    with open(os.path.join(alfworld_data, "base_config.yaml")) as f:
        config = yaml.safe_load(f)

    env = environment.get_environment(config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    return env, split_sizes[split]


def _get_alfworld_task_type(game_file: str) -> str:
    name = "/".join(game_file.split("/")[-3:-1])
    for prefix in ["pick_and_place", "pick_clean_then_place", "pick_heat_then_place",
                   "pick_cool_then_place", "look_at_obj", "pick_two_obj"]:
        if name.startswith(prefix):
            return prefix
    return "unknown"


def run_alfworld_one(
    sweep_type: str, tb: float, eb: float, k: int,
    split: str, icl_num: int, model: str,
    traj_dir: str, output_csv: str, max_episodes: int = None,
) -> Dict:
    if str(_ALFWORLD) not in sys.path:
        sys.path.insert(0, str(_ALFWORLD))
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from agent_alfworld_dual import DualTreeReflectiveAgent
    from common.llm_client import create_llm_client

    env, n_tasks = _load_alfworld_env(split)
    n     = min(max_episodes or n_tasks, n_tasks)
    k_val = k if k > 0 else 10_000

    agent = DualTreeReflectiveAgent(
        agent_name=f"Sweep_{sweep_type}_tb{tb}_eb{eb}_k{k}",
        llm_client=create_llm_client(model),
        icl_num=icl_num,
        icl_data_path=str(_ALFWORLD / "data" / "alfworld_icl.json"),
        task_base_threshold=tb,
        env_base_threshold=eb,
        consolidation_threshold=k_val,
    )

    ep_dir = os.path.join(traj_dir, f"alf__{sweep_type}__tb{tb}_eb{eb}_k{k}")
    os.makedirs(ep_dir, exist_ok=True)

    results, task_hits, env_hits = [], [], []
    for ep_idx in range(n):
        obs, info = env.reset()
        task_instruction = "\n".join(obs[0].split("\n\n")[1:])
        task_type = _get_alfworld_task_type(info["extra.gamefile"][0])

        messages = agent.run_episode(
            task_instruction=task_instruction,
            env=env,
            task_type=task_type,
            max_steps=30,
            episode_idx=ep_idx,
        )
        result = messages[-1]
        results.append(result)
        task_hits.append(bool(result.get("task_memory_used", False)))
        env_hits.append(bool(result.get("env_memory_used", False)))
        with open(os.path.join(ep_dir, f"ep_{ep_idx:04d}.json"), "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        if (ep_idx + 1) % 10 == 0 or ep_idx == n - 1:
            sr = sum(bool(r.get("success", False)) for r in results) / len(results)
            logger.info(f"  [alf/{sweep_type}] tb={tb} eb={eb} k={k}  "
                        f"Ep {ep_idx+1}/{n}: SR={sr:.2%}")

    row = _make_summary_row("alfworld", sweep_type, tb, eb, k,
                            results, agent, task_hits, env_hits)
    _append_csv(output_csv, row)
    logger.info(f"  ✅ alf tb={tb} eb={eb} k={k} "
                f"SR={row['sr']:.2%} "
                f"task_nodes={row['task_tree_total_nodes']} consol={row['consolidation_count']}")
    return row


# ── 构建任务列表 ──────────────────────────────────────────────────────────────
def _build_jobs(benchmark: str, sweep_type: str, k_fixed: int = None) -> List[Tuple]:
    opt = OPT[benchmark]
    if sweep_type == "grid_sweep":
        grid = GRID_VALUES[benchmark]
        k = k_fixed if k_fixed is not None else opt["k"]
        return [
            ("grid_sweep", round(tb, 4), round(eb, 4), k)
            for tb in grid["tb"]
            for eb in grid["eb"]
        ]
    if sweep_type == "single":
        return []  # handled separately in main
    raise ValueError(f"Unknown sweep_type: {sweep_type}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark",    choices=["sciworld", "alfworld"], required=True)
    parser.add_argument("--sweep-type",   choices=["grid_sweep", "single"], required=True)
    parser.add_argument("--tb",           type=float, default=None)
    parser.add_argument("--eb",           type=float, default=None)
    parser.add_argument("--k",            type=int,   default=None,
                        help="K value for grid_sweep (0=∞ disabled)")
    parser.add_argument("--split",        default=None)
    parser.add_argument("--icl-num",      type=int, default=1)
    parser.add_argument("--model",        default="deepseek-v4-flash")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--traj-dir",     default=str(_THIS_DIR / "trajectories"))
    parser.add_argument("--output-csv",   default=str(_THIS_DIR / "results" / "param_sweep.csv"))
    args = parser.parse_args()

    if args.split is None:
        args.split = "dev" if args.benchmark == "sciworld" else "eval_in_distribution"

    opt = OPT[args.benchmark]
    if args.tb is None: args.tb = opt["tb"]
    if args.eb is None: args.eb = opt["eb"]
    if args.k  is None: args.k  = opt["k"]

    os.makedirs(args.traj_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    if args.sweep_type == "single":
        jobs = [("single", args.tb, args.eb, args.k)]
    else:
        jobs = _build_jobs(args.benchmark, args.sweep_type, k_fixed=args.k)

    logger.info(f"Benchmark={args.benchmark}  Sweep={args.sweep_type}  "
                f"K={args.k}  Jobs={len(jobs)}")
    for j in jobs:
        logger.info(f"  tb={j[1]}  eb={j[2]}  k={j[3]}")

    runner = run_sciworld_one if args.benchmark == "sciworld" else run_alfworld_one

    for sweep_type, tb, eb, k in jobs:
        if _already_done(args.output_csv, args.benchmark, tb, eb, k):
            logger.info(f"  ⏩ SKIP (already done): tb={tb} eb={eb} k={k}")
            continue
        runner(
            sweep_type=sweep_type, tb=tb, eb=eb, k=k,
            split=args.split, icl_num=args.icl_num, model=args.model,
            traj_dir=args.traj_dir, output_csv=args.output_csv,
            max_episodes=args.max_episodes,
        )

    logger.info(f"All done. Results → {args.output_csv}")


if __name__ == "__main__":
    main()
