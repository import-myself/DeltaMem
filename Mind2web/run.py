"""
PRTree Mind2Web - 主运行脚本 (v4.0 Dual Tree)
对标 ALFWorld_New/example_dual_usage.py

结果保存格式（每个任务独立 JSON，与 ALFWorld_New/eval_in_distribution 一致）:
[
  {"role": "user",      "content": "...system + task prompt..."},
  {"role": "assistant", "content": "...step 1 response..."},
  {"role": "user",      "content": "...step 2 obs..."},
  ...
  {
    "success": true/false,
    "element_acc": [...],
    "action_f1": [...],
    "step_success": [...],
    "memory_used": bool,
    "task_memory_used": bool,
    "env_memory_used": bool,
    "task_reflection": {...},
    "env_reflection": {...},
    "task_node_id": "...",
    "env_node_id": "...",
    "trajectory": [...]
  }
]
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any

# PRTree root → synapse_memory.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_mind2web_dual import DualTreeMind2WebAgent
from common.llm_client import create_llm_client
from common.memory_utils import check_memory_path
from mind2web_utils import add_scores, load_json_data, get_all_combinations, calculate_metrics
from config import STORAGE_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ================================================================
# 结果保存（与 ALFWorld_New 完全一致的格式）
# ================================================================

def save_episode_result(
    episode_idx: int,
    messages: List[Dict],
    result: Dict[str, Any],
    save_dir: str,
):
    """
    将完整 messages（含最后的 summary dict）写入 {idx}.json
    与 ALFWorld_New/eval_in_distribution/online_dual_memory/{idx}.json 格式一致
    """
    # 展开 conversation 为 message list
    episode_messages = []
    for turn in result.get("conversation", []):
        if isinstance(turn, dict) and "input" in turn and "output" in turn:
            # 取最后一轮的 input messages 逐条加入（去重处理：只加新增的）
            if not episode_messages:
                episode_messages.extend(turn["input"])
            else:
                # 追加本轮新增的 user message（最后一条）
                episode_messages.append(turn["input"][-1])
            episode_messages.append({"role": "assistant", "content": turn["output"]})
        # 跳过 {"pred_act": ..., "target_act": ...} 这类纯对比记录

    # 最后附上 summary dict（与 ALFWorld_New 格式一致）
    summary = {
        "success":          result["success"],
        "element_acc":      result["element_acc"],
        "action_f1":        result["action_f1"],
        "step_success":     result["step_success"],
        "memory_used":      result["memory_used"],
        "task_memory_used": result["task_memory_used"],
        "env_memory_used":  result["env_memory_used"],
        "task_reflection":  result.get("task_reflection", {}),
        "env_reflection":   result.get("env_reflection", {}),
        "trajectory":       result.get("trajectory", []),
        "task_node_id":     result.get("task_node_id", ""),
        "env_node_id":      result.get("env_node_id", ""),
    }
    episode_messages.append(summary)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{episode_idx}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(episode_messages, f, indent=2, ensure_ascii=False)


# ================================================================
# 数据过滤
# ================================================================

def filter_samples(
    samples: List[Dict],
    domain:    Optional[str] = None,
    subdomain: Optional[str] = None,
    website:   Optional[str] = None,
) -> List[Dict]:
    if domain:
        samples = [s for s in samples if s.get("domain", "") == domain]
    if subdomain:
        samples = [s for s in samples if s.get("subdomain", "") == subdomain]
    if website:
        samples = [s for s in samples if s.get("website", "") == website]
    return samples


# ================================================================
# 训练/评估入口
# ================================================================

def run_offline_training(args):
    """Offline 训练模式 = 在 train benchmark 上跑 online evaluation，直接复用同一套逻辑。"""
    args.benchmark = "train"
    run_evaluation(args)


def _load_mind2web_results(traj_dir: str, n_tasks: int):
    """读取全部轨迹文件并汇总统计，任一文件缺失/损坏则返回 (None, 0, 0)。"""
    results, task_hit, env_hit = [], 0, 0
    for i in range(n_tasks):
        path = os.path.join(traj_dir, f"{i}.json")
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


def run_evaluation(args):
    # 轨迹保存目录（优先使用 --traj-dir 参数，否则默认 {benchmark}/online_dual_memory/）
    trajectory_dir = args.traj_dir or os.path.join(
        str(Path(__file__).parent),
        args.benchmark,
        "online_dual_memory",
    )
    os.makedirs(trajectory_dir, exist_ok=True)
    traj_dir_abs = os.path.abspath(trajectory_dir)
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
    }.get(getattr(args, "memory", "prtree"), getattr(args, "memory", "prtree"))
    logger.info("=" * 80)
    logger.info(f"ONLINE EVALUATION MODE: {_mem_label}")
    logger.info("=" * 80)

    # ---------- 加载数据 ----------
    data_dir   = str(Path(__file__).parent / "data")
    score_path = os.path.join(data_dir, "scores_all_data.pkl")
    logger.info(f"📂 加载数据集: {args.benchmark}")
    samples = load_json_data(data_dir, args.benchmark)

    if os.path.exists(score_path):
        logger.info(f"📊 附加候选元素分数: {score_path}")
        samples = add_scores(samples, score_path=score_path)
    else:
        logger.warning(f"⚠️  未找到分数文件: {score_path}")

    # 过滤
    samples = filter_samples(samples, args.domain, args.subdomain, args.website)
    if not samples:
        logger.error("❌ 过滤后样本为空，检查 --domain/--subdomain/--website 参数")
        sys.exit(1)

    if args.max_episodes:
        samples = samples[: args.max_episodes]
        logger.info(f"⚡ 限制样本数: {args.max_episodes}")

    n_tasks = len(samples)
    logger.info(f"▶ 共 {n_tasks} 条样本")

    # Lock 2a: 轨迹全部存在 → 直接从文件计算，不重跑
    n_done = sum(1 for i in range(n_tasks) if os.path.exists(os.path.join(trajectory_dir, f"{i}.json")))
    if n_done == n_tasks:
        logger.info(f"📂 [Lock2a] 已有 {n_tasks} 条轨迹 → 直接从文件计算结果。")
        all_results, t_hit, e_hit = _load_mind2web_results(trajectory_dir, n_tasks)
        if all_results is not None and len(all_results) == n_tasks:
            metrics = calculate_metrics(all_results)
            n = len(all_results)
            sr = metrics.get("task_success_rate", 0)
            logger.info(f"✅ [Lock2a] SR={sr:.2%}, ElemAcc={metrics.get('element_acc',0):.4f}, ActionF1={metrics.get('action_f1',0):.4f}")
            if results_csv:
                from common.result_logger import append_result
                append_result(results_csv, {
                    **_key,
                    "benchmark": f"mind2web_{args.benchmark}", "model": args.model,
                    "memory": getattr(args, "memory", "prtree"), "n_episodes": n,
                    "success_rate": round(sr, 6),
                    "element_acc": round(metrics.get("element_acc", 0), 6),
                    "action_f1": round(metrics.get("action_f1", 0), 6),
                    "step_success_rate": round(metrics.get("step_success_rate", 0), 6),
                    "memory_hit_rate": round(sum(1 for r in all_results if r.get("memory_used", False)) / n, 4),
                    "task_hit_rate": round(t_hit / n, 4),
                    "env_hit_rate": round(e_hit / n, 4),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                logger.info(f"📄 结果已写入 {results_csv}")
            return

    logger.info(f"💾 轨迹保存目录: {trajectory_dir}")

    # ---------- 初始化 Agent ----------
    llm_client    = create_llm_client(args.model)
    exemplar_path = os.path.join(data_dir, "example", "exemplars.json")

    agent = DualTreeMind2WebAgent(
        agent_name="PRTree-Mind2Web",
        llm_client=llm_client,
        exemplar_path=exemplar_path,
        top_k_elements=args.top_k_elements,
        previous_top_k_elements=args.previous_top_k_elements,
    )

    # ---- Memory backend selection ----
    no_mem_flag   = getattr(args, 'memory', None) == 'no-memory' or args.no_memory
    freeze        = getattr(args, "freeze", False)  # 冻结：加载已有库但禁止写入
    synapse_store = None
    awm_store     = None
    rb_store      = None

    _resume = getattr(args, 'resume', False)
    if no_mem_flag:
        logger.info("🚫 PRTree memory DISABLED (baseline mode)")
    elif getattr(args, 'memory', None) == 'synapse':
        from memory.synapse.synapse_memory import SynapseMemoryStore
        mem_path = getattr(args, 'memory_path', None) or getattr(args, 'memory_file', None) or "storage/synapse_memory"
        if _resume:
            check_memory_path(mem_path, "synapse")
        synapse_store = SynapseMemoryStore(memory_path=mem_path, load_existing=_resume,
                                           allow_updates=not freeze)
        logger.info(f"🧠 Synapse memory loaded: {synapse_store}" + (" [FROZEN]" if freeze else ""))
    elif getattr(args, 'memory', None) == 'awm':
        from memory.awm.awm_memory import AWMMemory
        awm_path  = getattr(args, 'memory_path', None) or getattr(args, 'memory_file', None) or "storage/awm_memory"
        if _resume:
            check_memory_path(awm_path, "awm")
        awm_store = AWMMemory(memory_path=awm_path, llm_client=llm_client, benchmark="mind2web",
                              load_existing=_resume, allow_updates=not freeze)
        logger.info(f"🔧 AWM memory loaded: {awm_store}" + (" [FROZEN]" if freeze else ""))
    elif getattr(args, 'memory', None) == 'reasoningbank':
        from memory.reasoningbank.reasoningbank_memory import ReasoningBankMemory
        rb_path = getattr(args, 'memory_path', None) or getattr(args, 'memory_file', None) or "storage/reasoningbank_memory"
        if _resume:
            check_memory_path(rb_path, "reasoningbank")
        from config import EMBEDDING_MODEL_PATH as _EMB_PATH
        rb_store = ReasoningBankMemory(
            memory_path=rb_path,
            llm_client=llm_client,
            benchmark="mind2web",
            embed_model_path=_EMB_PATH,
            load_existing=_resume,
            allow_updates=not freeze,
        )
        logger.info(f"📚 ReasoningBank memory loaded: {rb_store}" + (" [FROZEN]" if freeze else ""))

    # prtree: 仅在 --resume 时从 memory_path 加载已有记忆
    _prtree_path = getattr(args, 'memory_path', None) or getattr(args, 'save_memory', None)
    if getattr(args, 'memory', None) == 'prtree' and _prtree_path and not args.load_memory and _resume:
        if check_memory_path(_prtree_path, "prtree"):
            task_fp = _prtree_path.replace(".json", "") + "_task.json"
            if Path(task_fp).exists():
                agent.load_memory(_prtree_path)
                stats = agent.get_memory_stats()
                logger.info(f"📥 Auto-loaded memory from {_prtree_path}. Task: {stats['task_tree_nodes']} nodes, Env: {stats['env_tree_nodes']} nodes")
    if args.load_memory:
        task_fp = args.load_memory.replace(".json", "") + "_task.json"
        env_fp  = args.load_memory.replace(".json", "") + "_env.json"
        if Path(task_fp).exists() or Path(env_fp).exists():
            agent.load_memory(args.load_memory)
            stats = agent.get_memory_stats()
            logger.info(
                f"📥 Loaded memory. Task: {stats['task_tree_nodes']} nodes, "
                f"Env: {stats['env_tree_nodes']} nodes"
            )
        else:
            logger.warning(f"⚠️  记忆文件未找到: {args.load_memory}，从零开始")

    # 以 memory 的最大 episode_idx 为真正的 resume 点
    memory_last_idx = agent.dual_memory.get_last_committed_episode()
    if memory_last_idx >= 0:
        logger.info(f"🔖 Memory resume point: episode {memory_last_idx} (episodes 0-{memory_last_idx} already in memory)")

    # ---------- 主循环 ----------
    results       = []
    task_mem_hit  = 0
    env_mem_hit   = 0
    mem_hit       = 0

    for episode_idx, sample in enumerate(samples):
        task_desc = sample.get("confirmed_task", "")[:80]
        website   = sample.get("website", "unknown")
        logger.info(f"\n[{episode_idx + 1}/{n_tasks}] website={website} | task={task_desc}...")

        # --- 断点续跑：episode_idx <= memory_last_idx 才可安全跳过 ---
        if episode_idx <= memory_last_idx:
            saved_traj_path = os.path.join(trajectory_dir, f"{episode_idx}.json")
            if os.path.exists(saved_traj_path):
                with open(saved_traj_path, encoding="utf-8") as f:
                    saved_messages = json.load(f)
                saved_result = saved_messages[-1] if saved_messages else {}
                if isinstance(saved_result, dict) and "success" in saved_result:
                    results.append(saved_result)
                    if saved_result.get("task_memory_used", False):
                        task_mem_hit += 1
                    if saved_result.get("env_memory_used", False):
                        env_mem_hit += 1
                    if saved_result.get("memory_used", False):
                        mem_hit += 1
                    logger.info(f"⏭  Episode {episode_idx} in memory, skipping (success={saved_result['success']}).")
                    continue

        # --- Synapse 检索 ---
        ext_mem = None
        if synapse_store is not None:
            confirmed_task = sample.get('confirmed_task', '')
            website        = sample.get('website', '')
            synapse_query  = f"Website: {website}\nTask: {confirmed_task[:300]}"
            ext_mem = synapse_store.retrieve_memory_str(synapse_query)
        # --- AWM workflow 注入（按 website 分类）---
        if awm_store is not None:
            _website = sample.get('website', 'unknown')
            wf_str = awm_store.get_workflow(_website)
            if wf_str:
                ext_mem = wf_str

        # --- ReasoningBank 检索注入 ---
        if rb_store is not None:
            _website = sample.get('website', 'unknown')
            _task    = sample.get('confirmed_task', '')
            rb_query = f"Website: {_website}\nTask: {_task[:300]}"
            rb_mem   = rb_store.retrieve_memory_str(rb_query)
            if rb_mem:
                ext_mem = rb_mem

        try:
            result = agent.run_episode(sample, args.model,
                                       no_memory=no_mem_flag,
                                       no_prtree_update=freeze or (synapse_store is not None) or (awm_store is not None) or (rb_store is not None),
                                       external_memory_str=ext_mem,
                                       memory_type=args.memory,
                                       episode_idx=episode_idx)
        except KeyboardInterrupt:
            logger.info("\n⚠️  用户中断，保存当前进度...")
            break
        except Exception as e:
            logger.error(f"❌ Episode {episode_idx} failed: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "success": False,
                "element_acc": [], "action_f1": [], "step_success": [],
                "memory_used": False, "task_memory_used": False, "env_memory_used": False,
                "task_reflection": {}, "env_reflection": {},
                "task_node_id": "", "env_node_id": "",
                "conversation": [], "trajectory": [f"ERROR: {e}"],
            }

        results.append(result)

        # --- Synapse 在线写入（仅成功轨迹）---
        if synapse_store is not None and not no_mem_flag and result.get('success', False):
            confirmed_task = sample.get('confirmed_task', '')
            website        = sample.get('website', '')
            synapse_specifier = (
                f"Website: {website}\n"
                f"Task: {confirmed_task[:300]}\n"
                f"Success: {result.get('success', False)}"
            )
            # 从 conversation 重建 msg_list，首条替换为纯任务文本，去除 system prompt/memory header
            msg_list = []
            for turn in result.get('conversation', []):
                if isinstance(turn, dict) and 'input' in turn and 'output' in turn:
                    if not msg_list:
                        msg_list.extend(turn['input'])
                    else:
                        msg_list.append(turn['input'][-1])
                    msg_list.append({'role': 'assistant', 'content': turn['output']})
            if msg_list:
                task_text = f"Website: {website}\nTask: {confirmed_task}"
                msg_list = [{"role": "user", "content": task_text}] + msg_list[1:]
                synapse_store.add_exemplar(specifier=synapse_specifier, exemplar=msg_list)

        # --- AWM 诱导（按 website 分类）---
        if awm_store is not None and not no_mem_flag:
            _website  = sample.get('website', 'unknown')
            _task     = sample.get('confirmed_task', '')[:400]
            # 构建轨迹文本
            _traj = []
            for turn in result.get('conversation', []):
                if isinstance(turn, dict) and 'input' in turn and 'output' in turn:
                    _traj.append(turn['input'][-1].get('content','')[:200])
                    _traj.append('Action: ' + turn['output'][:200])
            awm_store.induce_and_update(
                task_type=_website,
                task_description=_task,
                trajectory=_traj,
                success=result.get('success', False),
            )

        # --- ReasoningBank 提炼并存储 ---
        if rb_store is not None and not no_mem_flag:
            _website  = sample.get('website', 'unknown')
            _task     = sample.get('confirmed_task', '')[:400]
            _traj     = []
            for turn in result.get('conversation', []):
                if isinstance(turn, dict) and 'input' in turn and 'output' in turn:
                    _traj.append(turn['input'][-1].get('content', '')[:200])
                    _traj.append('Action: ' + turn['output'][:200])
            rb_store.extract_and_store(
                query=f"Website: {_website}\nTask: {_task}",
                trajectory=_traj,
                success=result.get('success', False),
            )

        if result.get("task_memory_used", False): task_mem_hit += 1
        if result.get("env_memory_used",  False): env_mem_hit  += 1
        if result.get("memory_used",      False): mem_hit      += 1

        # 保存本 episode 完整 messages（与 ALFWorld_New 格式一致）
        save_episode_result(episode_idx, [], result, trajectory_dir)

        # 定期汇报
        if (episode_idx + 1) % 5 == 0:
            current_sr = sum(r["success"] for r in results) / len(results)
            if getattr(args, "memory", "prtree") == "prtree":
                stats = agent.get_memory_stats()
                logger.info(
                    f"📈 Ep {episode_idx + 1}: SR={current_sr:.2%}, "
                    f"TaskHit={task_mem_hit}/{len(results)}, "
                    f"EnvHit={env_mem_hit}/{len(results)}, "
                    f"TaskNodes={stats['task_tree_nodes']}, "
                    f"EnvNodes={stats['env_tree_nodes']}"
                )
            else:
                logger.info(
                    f"📈 Ep {episode_idx + 1}: SR={current_sr:.2%}, "
                    f"MemHit={mem_hit}/{len(results)}"
                )

        # 定期存档记忆
        _save = getattr(args, 'memory_path', None) or getattr(args, 'save_memory', None)
        if not freeze and getattr(args, "memory", "prtree") == "prtree" and _save and (episode_idx + 1) % args.save_interval == 0:
            agent.save_memory(_save)
        if not freeze and synapse_store is not None and (episode_idx + 1) % args.save_interval == 0:
            synapse_store.save()
        if not freeze and awm_store is not None and (episode_idx + 1) % args.save_interval == 0:
            awm_store.save()
        if not freeze and rb_store is not None and (episode_idx + 1) % args.save_interval == 0:
            rb_store.save()

    # ---------- 最终保存记忆（冻结模式下跳过）----------
    if not freeze and getattr(args, "memory", "prtree") == "prtree":
        save_path = getattr(args, 'memory_path', None) or getattr(args, 'save_memory', None) or STORAGE_PATH
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

    # ---------- 最终统计（对标 ALFWorld_New）----------
    final_stats = agent.get_memory_stats()
    metrics     = calculate_metrics(results)
    n = len(results)

    with_any_mem  = [r for r in results if r.get("memory_used",  False)]
    without_mem   = [r for r in results if not r.get("memory_used", False)]
    sr_with    = sum(r["success"] for r in with_any_mem) / len(with_any_mem)  if with_any_mem  else 0.0
    sr_without = sum(r["success"] for r in without_mem)  / len(without_mem)   if without_mem   else 0.0

    logger.info("\n" + "=" * 80)
    logger.info("DUAL PR-TREE v4.0 EVALUATION REPORT (Mind2Web)")
    logger.info("=" * 80)
    logger.info(f"Episodes:         {n}  |  Benchmark: {args.benchmark}")
    logger.info(f"Task Success Rate:{metrics.get('task_success_rate', 0):.2%}")
    logger.info(f"Element Acc:      {metrics.get('element_acc', 0):.4f}")
    logger.info(f"Action F1:        {metrics.get('action_f1',  0):.4f}")
    logger.info(f"Step Success Rate:{metrics.get('step_success_rate', 0):.4f}")
    logger.info(f"\n🧠 Memory Impact Analysis:")
    logger.info(f"  Task Tree Hit Rate: {task_mem_hit / n:.2%}")
    logger.info(f"  Env  Tree Hit Rate: {env_mem_hit  / n:.2%}")
    logger.info(f"  Any Memory Hit:     {len(with_any_mem) / n:.2%}")
    logger.info(f"  SR (Memory Hit):    {sr_with:.2%}  (n={len(with_any_mem)})")
    logger.info(f"  SR (Zero-shot):     {sr_without:.2%} (n={len(without_mem)})")
    if with_any_mem and without_mem:
        logger.info(f"  Improvement:        {sr_with - sr_without:+.2%}")
    logger.info(f"\n🌳 Dual Tree Topology (Final):")
    logger.info(f"  Task Tree Nodes: {final_stats['task_tree_nodes']}")
    logger.info(f"  Env  Tree Nodes: {final_stats['env_tree_nodes']}")
    logger.info(f"  Total Nodes:     {final_stats['total_nodes']}")
    logger.info(f"  Max Depth:       {final_stats['max_depth']}")
    logger.info("=" * 80)

    # 写入 CSV 结果（供后续 Lock1 去重使用）
    if results_csv:
        from common.result_logger import append_result
        append_result(results_csv, {
            **_key,
            "benchmark": f"mind2web_{args.benchmark}", "model": args.model,
            "memory": getattr(args, "memory", "prtree"), "n_episodes": n,
            "success_rate": round(metrics.get("task_success_rate", 0), 6),
            "element_acc": round(metrics.get("element_acc", 0), 6),
            "action_f1": round(metrics.get("action_f1", 0), 6),
            "step_success_rate": round(metrics.get("step_success_rate", 0), 6),
            "memory_hit_rate": round(mem_hit / n, 4),
            "task_hit_rate": round(task_mem_hit / n, 4),
            "env_hit_rate": round(env_mem_hit / n, 4),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        logger.info(f"📄 结果已写入 {results_csv}")


# ================================================================
# 参数解析
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="PRTree Dual-Tree Mind2Web Agent")
    p.add_argument("--mode",       type=str, choices=["train", "eval"], default="eval")
    p.add_argument("--model",      type=str, default="deepseek-v4-flash")
    p.add_argument("--benchmark",  type=str, default="test_task",
                   choices=["test_task", "test_website", "test_domain", "train"])
    # 过滤
    p.add_argument("--domain",    type=str, default=None)
    p.add_argument("--subdomain", type=str, default=None)
    p.add_argument("--website",   type=str, default=None)
    # top-k（与 AWM 保持一致）
    p.add_argument("--top-k-elements",          dest="top_k_elements",
                   type=int, default=5,
                   help="当前步骤候选元素数量（AWM 默认 5）")
    p.add_argument("--previous-top-k-elements", dest="previous_top_k_elements",
                   type=int, default=3,
                   help="历史步骤候选元素数量（AWM 默认 3）")
    # 记忆
    p.add_argument("--memory-path",   dest="memory_path",   type=str, default=None,
                   help="Unified memory path for all methods (load+save for prtree; load+save for synapse/awm/rb)")
    p.add_argument("--resume",        dest="resume",        action="store_true",
                   help="Load existing memory from --memory-path on startup (default: cold start)")
    p.add_argument("--load-memory",   dest="load_memory",   type=str, default=None,
                   help="prtree only: load from a different path (e.g. train→test)")
    p.add_argument("--save-memory",   dest="save_memory",   type=str, default=None,
                   help="Deprecated: use --memory-path instead")
    p.add_argument("--memory-file",   dest="memory_file",   type=str, default=None,
                   help="Deprecated: use --memory-path instead")
    p.add_argument("--save-interval", dest="save_interval", type=int, default=10)
    # 调试
    p.add_argument("--max-episodes",  dest="max_episodes",  type=int, default=None)
    p.add_argument("--no-memory", action="store_true",
                   help="Disable PRTree memory (baseline mode)")
    p.add_argument("--memory", type=str,
                   choices=["no-memory", "prtree", "synapse", "file", "awm", "reasoningbank"],
                   default="prtree",
                   help="Memory backend: no-memory|prtree|synapse|file|awm|reasoningbank")
    p.add_argument("--traj-dir", dest="traj_dir", type=str, default=None,
                   help="Directory to save trajectories (overrides default path)")
    p.add_argument("--results-csv", dest="results_csv", type=str, default=None,
                   help="CSV file to record run results (used for Lock1 deduplication)")
    p.add_argument("--freeze", action="store_true",
                   help="Load memory but disable all writes (frozen-memory eval)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path("storage").mkdir(exist_ok=True)
    if args.mode == "train":
        run_offline_training(args)
    else:
        run_evaluation(args)
