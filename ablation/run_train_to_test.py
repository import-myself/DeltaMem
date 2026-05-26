"""
Train→Test 记忆迁移实验 (实验 2.2)
=====================================
实验流程:
  Phase 1 (build) — 在 train split 上运行 agent，积累记忆，保存
  Phase 2 (eval)  — 加载冻结记忆（--freeze），在 test split 评估

支持方法:
  deltamem / synapse / awm / reasoningbank / no_memory

支持 Benchmark:
  alfworld  : train → eval_in_distribution
  sciworld  : train → dev/test

实现说明:
  build 阶段直接调用各 benchmark 的 run_online_evaluation(args.split="train")。
  eval  阶段调用时额外传 --resume --freeze --load-memory，复用所有记忆逻辑。
  所有方法（包括 awm/reasoningbank）均自动支持，无需在本文件重复实现。

运行示例:
  # ALFWorld DeltaMem 完整流程
  python run_train_to_test.py \\
      --benchmark alfworld --method deltamem --phase all \\
      --memory-path ../ALFWorld/storage/prtree_t2t \\
      --output-csv results/train_to_test.csv

  # 只跑 eval（已有离线记忆库）
  python run_train_to_test.py \\
      --benchmark alfworld --method deltamem --phase eval \\
      --memory-path ../ALFWorld/storage/prtree_t2t \\
      --output-csv results/train_to_test.csv
"""

import os
import sys
import csv
import time
import logging
import argparse
import types
from pathlib import Path
from typing import Dict, Any, List, Optional

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
# 工具：把 argparse.Namespace 风格的 dict 转为 SimpleNamespace
# =====================================================================

def _make_args(**kwargs) -> types.SimpleNamespace:
    """生成模拟 argparse.Namespace 的对象，供 benchmark runner 调用。"""
    return types.SimpleNamespace(**kwargs)


def _get_disk_size_mb(path: Optional[str]) -> float:
    if not path:
        return 0.0
    p = Path(path)
    if not p.exists():
        return 0.0
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / 1024 / 1024, 4)


# =====================================================================
# CSV 输出
# =====================================================================

CSV_FIELDNAMES = [
    "benchmark", "method", "phase", "split", "n_episodes",
    "success_rate", "avg_steps",
    "memory_disk_mb",
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
# ALFWorld
# =====================================================================

def _run_alfworld(args_main, method: str, phase: str) -> None:
    if str(_ALFWORLD) not in sys.path:
        sys.path.insert(0, str(_ALFWORLD))

    # 动态导入 ALFWorld runner
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "alf_runner", str(_ALFWORLD / "example_dual_usage.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mem_path   = args_main.memory_path
    traj_base  = args_main.traj_dir or "trajectories/train_to_test"
    eval_split = args_main.alfworld_eval_split

    # deltamem 在 runner 里注册为 "prtree"
    _MEM_ALIAS = {"deltamem": "prtree", "no_memory": "no-memory"}
    _mem_backend = _MEM_ALIAS.get(method, method)

    def _make_alf_args(split: str, freeze: bool, resume: bool) -> types.SimpleNamespace:
        return _make_args(
            split=split,
            model=args_main.model,
            icl_num=args_main.icl_num,
            max_steps=args_main.max_steps,
            memory=_mem_backend,
            memory_path=mem_path if method not in ("no_memory",) else None,
            memory_file=None,
            save_memory=None,
            load_memory=None,
            resume=resume,
            freeze=freeze,
            save_interval=50,
            traj_dir=os.path.join(traj_base, f"alfworld__{method}__{('build' if not freeze else 'eval')}__{split}"),
            results_csv=None,
            no_memory=(method == "no_memory"),
        )

    output_csv = args_main.output_csv

    if phase in ("build", "all"):
        logger.info(f"[ALFWorld/{method}] Phase: BUILD on train")
        run_args = _make_alf_args(split="train", freeze=False, resume=False)
        mod.run_online_evaluation(run_args)
        row = {
            "benchmark": "alfworld", "method": method, "phase": "build",
            "split": "train", "n_episodes": args_main.max_episodes or 3553,
            "success_rate": None, "avg_steps": None,
            "memory_disk_mb": _get_disk_size_mb(mem_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_to_csv(output_csv, row)

    if phase in ("eval", "all"):
        freeze_eval = not getattr(args_main, "no_freeze_eval", False)
        label = "frozen" if freeze_eval else "unfrozen"
        logger.info(f"[ALFWorld/{method}] Phase: EVAL on {eval_split} ({label})")
        run_args = _make_alf_args(split=eval_split, freeze=freeze_eval, resume=True)
        mod.run_online_evaluation(run_args)
        row = {
            "benchmark": "alfworld", "method": method, "phase": "eval",
            "split": eval_split, "n_episodes": args_main.max_episodes or 140,
            "success_rate": None, "avg_steps": None,
            "memory_disk_mb": _get_disk_size_mb(mem_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_to_csv(output_csv, row)


# =====================================================================
# ScienceWorld
# =====================================================================

def _run_sciworld(args_main, method: str, phase: str) -> None:
    if str(_SCIWORLD) not in sys.path:
        sys.path.insert(0, str(_SCIWORLD))

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sci_runner", str(_SCIWORLD / "run_sciworld.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mem_path   = args_main.memory_path
    traj_base  = args_main.traj_dir or "trajectories/train_to_test"
    eval_split = args_main.sciworld_eval_split

    # deltamem 在 runner 里注册为 "prtree"
    _MEM_ALIAS = {"deltamem": "prtree", "no_memory": "no-memory"}
    _mem_backend = _MEM_ALIAS.get(method, method)

    def _make_sci_args(split: str, freeze: bool, resume: bool) -> types.SimpleNamespace:
        return _make_args(
            split=split,
            model=args_main.model,
            icl_num=args_main.icl_num,
            icl_path=str(_SCIWORLD / "data/sciworld_icl.json"),
            max_episodes=args_main.max_episodes or 10000,
            memory=_mem_backend,
            memory_path=mem_path if method not in ("no_memory",) else None,
            memory_file=None,
            save_memory=None,
            load_memory=None,
            resume=resume,
            freeze=freeze,
            save_interval=50,
            no_memory=(method == "no_memory"),
            traj_dir=os.path.join(traj_base, f"sciworld__{method}__{('build' if not freeze else 'eval')}__{split}"),
            results_csv=None,
        )

    output_csv = args_main.output_csv

    if phase in ("build", "all"):
        logger.info(f"[SciWorld/{method}] Phase: BUILD on train")
        run_args = _make_sci_args(split="train", freeze=False, resume=False)
        mod.run_online_evaluation(run_args)
        row = {
            "benchmark": "sciworld", "method": method, "phase": "build",
            "split": "train", "n_episodes": args_main.max_episodes or 10000,
            "success_rate": None, "avg_steps": None,
            "memory_disk_mb": _get_disk_size_mb(mem_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_to_csv(output_csv, row)

    if phase in ("eval", "all"):
        freeze_eval = not getattr(args_main, "no_freeze_eval", False)
        label = "frozen" if freeze_eval else "unfrozen"
        logger.info(f"[SciWorld/{method}] Phase: EVAL on {eval_split} ({label})")
        run_args = _make_sci_args(split=eval_split, freeze=freeze_eval, resume=True)
        mod.run_online_evaluation(run_args)
        row = {
            "benchmark": "sciworld", "method": method, "phase": "eval",
            "split": eval_split, "n_episodes": args_main.max_episodes or 10000,
            "success_rate": None, "avg_steps": None,
            "memory_disk_mb": _get_disk_size_mb(mem_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_to_csv(output_csv, row)


# =====================================================================
# dispatch
# =====================================================================

VALID_METHODS    = {"deltamem", "synapse", "awm", "reasoningbank", "no_memory"}
VALID_BENCHMARKS = {"alfworld", "sciworld", "all"}
VALID_PHASES     = {"build", "eval", "all"}

_BENCHMARK_RUNNERS = {
    "alfworld": _run_alfworld,
    "sciworld": _run_sciworld,
}


# =====================================================================
# main
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description="Train→Test 记忆迁移实验（实验 2.2）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--benchmark", choices=list(VALID_BENCHMARKS), default="alfworld")
    p.add_argument("--method",    type=str, default="deltamem",
                   help="逗号分隔，可选: deltamem,synapse,awm,reasoningbank,no_memory")
    p.add_argument("--phase",     choices=list(VALID_PHASES), default="all",
                   help="build=只跑 train 积累记忆；eval=只跑 test 评估；all=先 build 后 eval")
    p.add_argument("--model",        type=str, default="deepseek-v4-flash")
    p.add_argument("--icl-num",      type=int, default=1)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--max-steps",    type=int, default=30)
    p.add_argument("--output-csv",   type=str, default="results/train_to_test.csv")
    p.add_argument("--traj-dir",     type=str, default=None)

    p.add_argument("--memory-path", type=str, default=None,
                   help="所有方法共用的记忆存储路径（build 写入，eval 读出冻结）")

    p.add_argument("--no-freeze-eval", action="store_true",
                   help="eval 阶段不冻结记忆库（允许写入），默认冻结")
    p.add_argument("--alfworld-eval-split",
                   choices=["eval_in_distribution", "eval_out_of_distribution"],
                   default="eval_in_distribution")
    p.add_argument("--sciworld-eval-split", choices=["dev", "test"], default="dev")

    args = p.parse_args()

    methods = [m.strip() for m in args.method.split(",") if m.strip()]
    invalid = set(methods) - VALID_METHODS
    if invalid:
        p.error(f"Invalid methods: {invalid}")

    benchmarks = (["alfworld", "sciworld"] if args.benchmark == "all"
                  else [args.benchmark])

    logger.info("=" * 70)
    logger.info("Train→Test 记忆迁移实验 (实验 2.2)")
    logger.info(f"  Benchmarks : {benchmarks}")
    logger.info(f"  Methods    : {methods}")
    logger.info(f"  Phase      : {args.phase}")
    logger.info(f"  Model      : {args.model}")
    logger.info(f"  Memory path: {args.memory_path}")
    logger.info(f"  Output CSV : {args.output_csv}")
    logger.info("=" * 70)

    for bm in benchmarks:
        runner = _BENCHMARK_RUNNERS.get(bm)
        if runner is None:
            logger.warning(f"Benchmark '{bm}' not yet implemented, skipping.")
            continue
        for method in methods:
            logger.info(f"\n>>> benchmark={bm}, method={method}")
            try:
                runner(args, method, args.phase)
            except Exception as e:
                logger.error(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()

    logger.info(f"\nAll done → {args.output_csv}")


if __name__ == "__main__":
    main()
