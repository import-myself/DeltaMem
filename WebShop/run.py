"""
WebShop Memory Runner (升级自 baseline_no_memory.py)

支持以下 memory backend（通过 --memory dispatch）：
  - no-memory      : 纯 baseline（默认）
  - synapse        : SynapseMemoryStore (FAISS/difflib 检索 + exemplar 注入)
  - awm            : AWMMemory (按统一 task_type='webshop' 维护 workflow)
  - reasoningbank  : ReasoningBankMemory (向量检索 + LLM 提炼)
  - prtree         : 占位，本脚本暂不实现（后续接入双树）

整体交互逻辑参考 ALFWorld/example_dual_usage.py：
  · 轨迹文件 `{traj_dir}/{idx}.json`，最后一项为 summary dict
  · CSV Lock1 去重 + Lock2a 全量轨迹直接复算
  · 记忆按 --save-interval 增量保存，--freeze 模式只读
  · 端到端续跑：已有 summary 的 episode 直接跳过

Usage:
  DEEPSEEK_API_KEY=... python -u run.py --memory no-memory --n 10
  DEEPSEEK_API_KEY=... python -u run.py --memory synapse --memory-path storage/synapse_memory --n 50
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# WebShop on path
sys.path.insert(0, str(Path(__file__).parent))
# DeltaMem common
sys.path.insert(0, str(Path(__file__).parent.parent))

import gym
from web_agent_site.envs import WebAgentTextEnv  # noqa: F401  triggers registration

from common.llm_client import create_llm_client
from common.memory_utils import check_memory_path
from dual_prompt import (
    MEMORY_HEADERS,
    PROMPT_WITH_ICL_TEMPLATE,
    PROMPT_WITH_ICL_TEMPLATE_DUAL_MEMORY,
    webshop_instruction,
)
from agent_webshop_dual import ICL_EXAMPLE as _PRTREE_ICL_EXAMPLE, _format_icl_block

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # gym/spacy/pyserini 会先 basicConfig 一次，需强制覆盖
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts (sourced from ETO)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are web shopping.
I will give you instructions about what to do.
You have to follow the instructions.
Every round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.
You can use search action if search is available.
You can click one of the buttons in clickables.
An action should be of the following structure:
search[keywords]
click[value]
If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions.
Remember that your keywords in search should be carefully designed.
Your response should use the following format:

Thought: I think ...
Action: click[something]"""


# One-shot ICL example (from ETO webshop_icl.json)
ICL_EXAMPLE = [
    {
        "role": "user",
        "content": "Observation:\nWebShop [SEP] Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Search",
    },
    {
        "role": "assistant",
        "content": "Thought: I think I should use the search bar to look for the product I need.\nAction: search[l'eau d'issey 6.76 fl oz bottle price < 100.00]",
    },
    {
        "role": "user",
        "content": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B000VOHH8I [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] $64.98",
    },
    {
        "role": "assistant",
        "content": "Thought: I think I should click on the product I need, which is B000VOHH8I.\nAction: click[B000VOHH8I]",
    },
    {
        "role": "user",
        "content": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now",
    },
    {
        "role": "assistant",
        "content": "Thought: I think I should click on the '6.76 fl oz (pack of 1)' option to select the size I need.\nAction: click[6.76 fl oz (pack of 1)]",
    },
    {
        "role": "user",
        "content": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now",
    },
    {
        "role": "assistant",
        "content": "Thought: I think I should click on the 'Buy Now' button to purchase the product.\nAction: click[Buy Now]",
    },
]


# WebShop 没有官方 task_type 分类，统一用 "webshop" 作 AWM workflow 的分组 key
WEBSHOP_TASK_TYPE = "webshop"

# 记忆段统一标题：注入到 system prompt 末尾
MEMORY_HEADER = "\n\n[Relevant memory from past episodes]\n"


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

ACTION_RE = re.compile(r"Action:\s*(.+?)(?:\n|$)", re.DOTALL)
INSTRUCTION_RE = re.compile(r"Instruction:\s*\[SEP\]\s*(.+?)(?:\s*\[SEP\]|$)")


def parse_action(llm_output: str) -> Optional[str]:
    m = ACTION_RE.search(llm_output)
    if not m:
        return None
    action = m.group(1).strip()
    if not (action.startswith("search[") or action.startswith("click[")):
        return None
    if not action.endswith("]"):
        return None
    return action


def extract_instruction(obs: str) -> str:
    """从初始 observation 中切出 instruction 文本（去除 [SEP] 等装饰）。"""
    m = INSTRUCTION_RE.search(obs or "")
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------

def run_episode_icl(
    env,
    llm,
    session_id: int,
    max_steps: int = 15,
    external_memory_str: Optional[str] = None,
    memory_type: str = "synapse",
    verbose: bool = True,
) -> Dict[str, Any]:
    """与 prtree 相同的单条 user message 格式：instruction + ICL text + memory + task 全合并。
    用于 synapse_i，使 prompt 结构与 prtree 保持一致。
    """
    env.reset(session=str(session_id))
    obs = env.observation
    instruction = extract_instruction(obs)

    icl_text = _format_icl_block(_PRTREE_ICL_EXAMPLE)
    task_block = f"Observation:\n{obs}"

    if external_memory_str:
        header = MEMORY_HEADERS.get(memory_type, MEMORY_HEADERS["synapse"])
        first_content = PROMPT_WITH_ICL_TEMPLATE_DUAL_MEMORY.format(
            instruction=webshop_instruction,
            examples=icl_text,
            memory_header=header,
            memory_context=external_memory_str.strip(),
            task=task_block,
        )
    else:
        first_content = PROMPT_WITH_ICL_TEMPLATE.format(
            instruction=webshop_instruction,
            examples=icl_text,
            task=task_block,
        )

    messages: List[Dict[str, str]] = [{"role": "user", "content": first_content}]

    final_reward = 0.0
    done = False
    step = 0
    invalid_in_a_row = 0
    trajectory: List[str] = []

    if verbose:
        logger.info(f"  Task: {instruction[:160]}")

    while step < max_steps and not done:
        try:
            resp = llm.chat(messages, temperature=0.0, max_tokens=300)
        except Exception as e:
            logger.warning(f"  [step {step}] LLM error: {e}")
            break

        action = parse_action(resp)
        messages.append({"role": "assistant", "content": resp})

        if action is None:
            invalid_in_a_row += 1
            if invalid_in_a_row >= 3:
                break
            messages.append({"role": "user", "content": "Observation: Invalid format. Use 'Action: search[...] or click[...]'."})
            step += 1
            continue
        invalid_in_a_row = 0

        trajectory.append(f"Action: {action}")
        try:
            new_obs, reward, done, _ = env.step(action)
        except AssertionError:
            new_obs, reward, done = "Invalid action!", 0.0, False

        trajectory.append(f"Observation: {new_obs[:200]}")
        messages.append({"role": "user", "content": f"Observation:\n{new_obs}"})
        final_reward = reward
        step += 1
        if verbose:
            logger.info(f"  [step {step}] {action!r}  reward={reward:.3f}  done={done}")
        if done:
            break

    success = (final_reward == 1.0)
    return {
        "session_id": session_id, "instruction": instruction,
        "reward": final_reward, "steps": step, "done": done, "success": success,
        "memory_used": bool(external_memory_str),
        "task_memory_used": False, "env_memory_used": False,
        "trajectory": trajectory,
    }, messages


def run_episode(
    env,
    llm,
    session_id: int,
    max_steps: int = 15,
    external_memory_str: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """单个 episode 主循环，返回 result dict + messages list。"""
    env.reset(session=str(session_id))
    obs = env.observation
    instruction = extract_instruction(obs)

    # 将 memory 注入 system prompt（保持 ICL 不变）
    system_content = SYSTEM_PROMPT
    if external_memory_str:
        system_content = SYSTEM_PROMPT + MEMORY_HEADER + external_memory_str.strip()

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
    messages.extend(list(ICL_EXAMPLE))
    messages.append({"role": "user", "content": f"Observation:\n{obs}"})

    final_reward = 0.0
    done = False
    step = 0
    invalid_in_a_row = 0
    trajectory: List[str] = []  # 用于 awm/reasoningbank 提炼

    if verbose:
        logger.info(f"  Task: {instruction[:160]}")

    while step < max_steps and not done:
        try:
            resp = llm.chat(messages, temperature=0.0, max_tokens=300)
        except Exception as e:
            logger.warning(f"  [step {step}] LLM error: {e}")
            break

        action = parse_action(resp)
        if action is None:
            invalid_in_a_row += 1
            if verbose:
                logger.info(f"  [step {step}] INVALID llm output: {resp[:120]!r}")
            messages.append({"role": "assistant", "content": resp})
            if invalid_in_a_row >= 3:
                if verbose:
                    logger.info("  ... 3 invalid in a row, abort")
                break
            messages.append({
                "role": "user",
                "content": "Observation: Invalid format. The input must contain 'Action: '",
            })
            step += 1
            continue
        invalid_in_a_row = 0

        try:
            obs, reward, done, info = env.step(action)
        except AssertionError:
            obs = "Invalid action!"
            reward = 0.0
            done = False

        # 用于 memory 提炼的紧凑文本（截断防爆 token）
        obs_compact = re.sub(r"\s+", " ", str(obs))[:200]
        trajectory.append(f"Obs: {obs_compact}")
        trajectory.append(f"Action: {action}")

        if verbose:
            short_obs = re.sub(r"\s+", " ", str(obs))[:200]
            logger.info(
                f"  [step {step}] action={action!r} reward={reward:.3f} done={done} obs={short_obs!r}"
            )

        messages.append({"role": "assistant", "content": resp})
        messages.append({"role": "user", "content": f"Observation:\n{obs}"})
        final_reward = reward
        step += 1

    success = (final_reward == 1.0)
    result: Dict[str, Any] = {
        "session_id":       session_id,
        "instruction":      instruction,
        "reward":           final_reward,
        "steps":            step,
        "done":             done,
        "success":          success,
        "memory_used":      bool(external_memory_str),
        "task_memory_used": False,  # 兼容 ALFWorld 字段（WebShop 单树场景）
        "env_memory_used":  False,
        "trajectory":       trajectory,
    }
    return result, messages


# ---------------------------------------------------------------------------
# Trajectory persistence (与 ALFWorld 一致：messages + summary dict)
# ---------------------------------------------------------------------------

def save_episode_trajectory(
    episode_idx: int,
    messages: List[Dict[str, str]],
    result: Dict[str, Any],
    save_dir: str,
):
    """messages 末尾 append summary dict，整体写入 {idx}.json。"""
    os.makedirs(save_dir, exist_ok=True)
    payload = list(messages) + [result]
    out_path = os.path.join(save_dir, f"{episode_idx}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_webshop_results(traj_dir: str, n_tasks: int):
    """读取全部轨迹，任一缺失/损坏返回 (None, 0)。"""
    results, mem_hit = [], 0
    for i in range(n_tasks):
        path = os.path.join(traj_dir, f"{i}.json")
        try:
            with open(path, encoding="utf-8") as f:
                msgs = json.load(f)
            r = msgs[-1]
            if not isinstance(r, dict) or "success" not in r:
                return None, 0
            results.append(r)
            if r.get("memory_used", False):
                mem_hit += 1
        except Exception:
            return None, 0
    return results, mem_hit


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(args):
    # 轨迹保存目录
    trajectory_dir = args.traj_dir or os.path.join(
        str(Path(__file__).parent),
        "trajectories",
        f"{args.memory}",
    )
    os.makedirs(trajectory_dir, exist_ok=True)
    traj_dir_abs = os.path.abspath(trajectory_dir)
    results_csv = args.results_csv
    _key = {"traj_dir": traj_dir_abs}

    # Lock 1: 先查 CSV，已有记录直接跳过
    if results_csv:
        from common.result_logger import check_existing_result
        existing = check_existing_result(results_csv, _key)
        if existing:
            logger.info(
                f"⏭ [Lock1] 已有记录 (SR={existing.get('success_rate', '?')})，跳过本次运行。"
            )
            return

    _mem_label = {
        "no-memory":     "No Memory (Baseline)",
        "synapse":       "Synapse",
        "awm":           "AWM",
        "reasoningbank": "ReasoningBank",
        "prtree":        "DeltaMem (Dual PR-Tree)",
    }.get(args.memory, args.memory)
    logger.info("=" * 80)
    logger.info(f"WEBSHOP EVALUATION: {_mem_label}")
    logger.info("=" * 80)

    # ---------- 加载 sessions ----------
    sessions = json.load(open(args.sessions_file))
    if args.max_episodes:
        sessions = sessions[: args.max_episodes]
    sessions = sessions[args.start : args.start + args.n] if args.n > 0 else sessions[args.start:]
    n_tasks = len(sessions)
    logger.info(f"▶ 共 {n_tasks} 个 session：start={args.start} n={args.n}")

    # Lock 2a: 轨迹全部存在 → 直接从文件计算
    n_done = sum(1 for i in range(n_tasks) if os.path.exists(os.path.join(trajectory_dir, f"{i}.json")))
    if n_done == n_tasks and n_tasks > 0:
        logger.info(f"📂 [Lock2a] 已有 {n_tasks} 条轨迹 → 直接从文件计算结果。")
        all_results, m_hit = _load_webshop_results(trajectory_dir, n_tasks)
        if all_results is not None and len(all_results) == n_tasks:
            n = len(all_results)
            sr = sum(r["success"] for r in all_results) / n
            avg_r = sum(r["reward"] for r in all_results) / n
            avg_s = sum(r["steps"] for r in all_results) / n
            t_hit = sum(1 for r in all_results if r.get("task_memory_used", False))
            e_hit = sum(1 for r in all_results if r.get("env_memory_used",  False))
            logger.info(f"✅ [Lock2a] SR={sr:.2%}, AvgReward={avg_r:.3f}, AvgSteps={avg_s:.2f}")
            if results_csv:
                from common.result_logger import append_result
                append_result(results_csv, {
                    **_key,
                    "benchmark": "webshop", "model": args.model,
                    "memory": args.memory, "n_episodes": n,
                    "success_rate":   round(sr, 6),
                    "avg_reward":     round(avg_r, 6),
                    "avg_steps":      round(avg_s, 4),
                    "memory_hit_rate": round(m_hit / n, 4),
                    "task_hit_rate":   round(t_hit / n, 4),
                    "env_hit_rate":    round(e_hit / n, 4),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                logger.info(f"📄 结果已写入 {results_csv}")
            return

    # ---------- 初始化 env / llm ----------
    logger.info("Bootstrapping WebShop env (loading 1000 products) ...")
    env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=1000)
    logger.info("Env ready.")

    llm_client = create_llm_client(model_name=args.model)

    # ---------- Memory backend dispatch ----------
    no_mem_flag   = (args.memory == "no-memory")
    freeze        = args.freeze
    synapse_store = None
    awm_store     = None
    rb_store      = None

    # PRTree agent（仅 --memory prtree 时实例化）
    prtree_agent = None
    if args.memory == "prtree":
        from agent_webshop_dual import DualTreeWebShopAgent
        prtree_agent = DualTreeWebShopAgent(
            agent_name="PRTree-WebShop",
            llm_client=llm_client,
        )
        # 加载已有记忆（--resume 或 --load-memory）
        _prtree_path = args.memory_path or args.load_memory
        if _prtree_path and (args.resume or args.load_memory):
            if check_memory_path(_prtree_path, "prtree"):
                task_fp = _prtree_path.replace(".json", "") + "_task.json"
                if Path(task_fp).exists():
                    prtree_agent.load_memory(_prtree_path)
                    stats = prtree_agent.get_memory_stats()
                    logger.info(
                        f"📥 PRTree memory loaded: TaskNodes={stats.get('task_tree_nodes', 0)}, "
                        f"EnvNodes={stats.get('env_tree_nodes', 0)}"
                    )
        logger.info(f"🌳 PRTree agent ready" + (" [FROZEN]" if freeze else ""))
    elif no_mem_flag:
        logger.info("🚫 Memory DISABLED (baseline mode)")
    elif args.memory in ("synapse", "synapse_i"):
        from memory.synapse.synapse_memory import SynapseMemoryStore
        mem_path = args.memory_path or args.memory_file or f"storage/{args.memory}_memory"
        if args.resume:
            check_memory_path(mem_path, "synapse")
        synapse_store = SynapseMemoryStore(
            memory_path=mem_path,
            load_existing=args.resume,
            allow_updates=not freeze,
        )
        logger.info(f"🧠 Synapse memory loaded ({args.memory}): {synapse_store}" + (" [FROZEN]" if freeze else ""))
    elif args.memory == "awm":
        from memory.awm.awm_memory import AWMMemory
        awm_path = args.memory_path or args.memory_file or "storage/awm_memory"
        if args.resume:
            check_memory_path(awm_path, "awm")
        awm_store = AWMMemory(
            memory_path=awm_path,
            llm_client=llm_client,
            benchmark="webshop",
            load_existing=args.resume,
            allow_updates=not freeze,
        )
        logger.info(f"🔧 AWM memory loaded: {awm_store}" + (" [FROZEN]" if freeze else ""))
    elif args.memory == "reasoningbank":
        from memory.reasoningbank.reasoningbank_memory import ReasoningBankMemory
        rb_path = args.memory_path or args.memory_file or "storage/reasoningbank_memory"
        if args.resume:
            check_memory_path(rb_path, "reasoningbank")
        # WebShop 无本地 config.py，沿用 Mind2web/ALFWorld 的 e5-base-v2 默认路径
        emb_path = args.embed_model_path or os.environ.get("EMBEDDING_MODEL_PATH") or str(Path(__file__).resolve().parent.parent / "embedding" / "e5-base-v2")
        rb_store = ReasoningBankMemory(
            memory_path=rb_path,
            llm_client=llm_client,
            benchmark="webshop",
            embed_model_path=emb_path,
            load_existing=args.resume,
            allow_updates=not freeze,
        )
        logger.info(f"📚 ReasoningBank memory loaded: {rb_store}" + (" [FROZEN]" if freeze else ""))

    # ---------- 主循环 ----------
    results: List[Dict[str, Any]] = []
    mem_hit = 0
    t0 = time.time()

    for episode_idx, session_id in enumerate(sessions):
        logger.info(f"\n[{episode_idx + 1}/{n_tasks}] session={session_id} ({args.memory})")

        # 断点续跑：已有完整轨迹 → 直接读
        traj_path = os.path.join(trajectory_dir, f"{episode_idx}.json")
        if os.path.exists(traj_path):
            try:
                with open(traj_path, encoding="utf-8") as f:
                    saved = json.load(f)
                saved_result = saved[-1] if saved else {}
                if isinstance(saved_result, dict) and "success" in saved_result:
                    results.append(saved_result)
                    if saved_result.get("memory_used", False):
                        mem_hit += 1
                    logger.info(
                        f"⏭  Episode {episode_idx} resumed from disk (success={saved_result['success']})."
                    )
                    continue
            except Exception:
                pass  # 文件损坏则重跑

        # --- 检索注入：先用 env.reset 得到 instruction，再走 memory ---
        env.reset(session=str(session_id))
        instruction = extract_instruction(env.observation)

        ext_mem: Optional[str] = None
        if synapse_store is not None:
            ext_mem = synapse_store.retrieve_memory_str(query=instruction[:400])
        if awm_store is not None:
            wf_str = awm_store.get_workflow(WEBSHOP_TASK_TYPE)
            if wf_str:
                ext_mem = wf_str
        if rb_store is not None:
            rb_mem = rb_store.retrieve_memory_str(query=instruction[:400])
            if rb_mem:
                ext_mem = rb_mem

        # --- 跑 episode：prtree agent 自带检索/反思/写入；其他 backend 走通用 run_episode ---
        try:
            if prtree_agent is not None:
                result, messages = prtree_agent.run_episode(
                    env=env,
                    session_id=session_id,
                    max_steps=args.max_steps,
                    no_memory=False,
                    no_prtree_update=freeze,
                    external_memory_str=None,
                    memory_type="prtree",
                    episode_idx=episode_idx,
                )
            else:
                # no-memory / synapse / synapse_i / awm / reasoningbank
                # 统一使用单条 user message 格式（与 prtree 一致）
                mem_type = args.memory if args.memory != "no-memory" else "synapse"
                result, messages = run_episode_icl(
                    env=env,
                    llm=llm_client,
                    session_id=session_id,
                    max_steps=args.max_steps,
                    external_memory_str=ext_mem,
                    memory_type=mem_type,
                    verbose=True,
                )
        except KeyboardInterrupt:
            logger.info("\n⚠️  用户中断，保存当前进度...")
            break
        except Exception as e:
            logger.error(f"❌ Episode {episode_idx} failed: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "session_id": session_id, "instruction": instruction,
                "reward": 0.0, "steps": 0, "done": False, "success": False,
                "memory_used": bool(ext_mem),
                "task_memory_used": False, "env_memory_used": False,
                "trajectory": [f"ERROR: {e}"],
            }
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        results.append(result)
        if result.get("memory_used", False):
            mem_hit += 1

        # 保存轨迹（ALFWorld 风格：messages + summary dict 同文件）
        save_episode_trajectory(episode_idx, messages, result, trajectory_dir)

        # ---- 在线写入 memory（成功轨迹）----
        if synapse_store is not None and not no_mem_flag and result.get("success", False):
            # 重建纯净 message list：第一条仅保留 instruction，去掉 system prompt / memory header
            msg_list = [{"role": "user", "content": instruction}] + messages[1 + len(ICL_EXAMPLE) + 1 :]
            synapse_specifier = (
                f"WebShop\n"
                f"Task: {instruction[:300]}\n"
                f"Success: True"
            )
            synapse_store.add_exemplar(specifier=synapse_specifier, exemplar=msg_list)

        if awm_store is not None and not no_mem_flag:
            awm_store.induce_and_update(
                task_type=WEBSHOP_TASK_TYPE,
                task_description=instruction[:400],
                trajectory=result.get("trajectory", []),
                success=result.get("success", False),
            )

        if rb_store is not None and not no_mem_flag:
            rb_store.extract_and_store(
                query=f"WebShop\nTask: {instruction[:400]}",
                trajectory=result.get("trajectory", []),
                success=result.get("success", False),
            )

        # 定期汇报
        if (episode_idx + 1) % 5 == 0:
            current_sr = sum(r["success"] for r in results) / len(results)
            logger.info(
                f"📈 Ep {episode_idx + 1}: SR={current_sr:.2%}, MemHit={mem_hit}/{len(results)}"
            )

        # 定期持久化
        if not freeze and (episode_idx + 1) % args.save_interval == 0:
            if synapse_store is not None: synapse_store.save()
            if awm_store     is not None: awm_store.save()
            if rb_store      is not None: rb_store.save()
            if prtree_agent  is not None and args.memory_path:
                prtree_agent.save_memory(args.memory_path)

    # ---------- 最终持久化（freeze 模式跳过）----------
    if not freeze:
        if synapse_store is not None:
            synapse_store.save()
            logger.info(f"💾 Synapse store saved: {synapse_store}")
        if awm_store is not None:
            awm_store.save()
            logger.info(f"💾 AWM store saved: {awm_store}")
        if rb_store is not None:
            rb_store.save()
            logger.info(f"💾 ReasoningBank store saved: {rb_store}")
        if prtree_agent is not None:
            save_path = args.memory_path or "storage/prtree_webshop"
            prtree_agent.save_memory(save_path)

    # ---------- 最终统计 ----------
    elapsed = time.time() - t0
    n = len(results) if results else 1
    sr      = sum(r["success"] for r in results) / n if results else 0.0
    avg_r   = sum(r["reward"]  for r in results) / n if results else 0.0
    avg_s   = sum(r["steps"]   for r in results) / n if results else 0.0

    with_mem  = [r for r in results if r.get("memory_used", False)]
    without_m = [r for r in results if not r.get("memory_used", False)]
    sr_with    = sum(r["success"] for r in with_mem)  / len(with_mem)  if with_mem  else 0.0
    sr_without = sum(r["success"] for r in without_m) / len(without_m) if without_m else 0.0

    logger.info("\n" + "=" * 80)
    logger.info(f"WEBSHOP EVALUATION REPORT ({_mem_label})")
    logger.info("=" * 80)
    logger.info(f"Episodes:     {len(results)}  |  Time: {elapsed:.1f}s")
    logger.info(f"Success Rate: {sr:.2%}")
    logger.info(f"Avg Reward:   {avg_r:.4f}")
    logger.info(f"Avg Steps:    {avg_s:.2f}")
    logger.info(f"\n🧠 Memory Impact Analysis:")
    logger.info(f"  Memory Hit Rate:  {mem_hit / n:.2%}")
    logger.info(f"  SR (Memory Hit):  {sr_with:.2%}  (n={len(with_mem)})")
    logger.info(f"  SR (Zero-shot):   {sr_without:.2%}  (n={len(without_m)})")
    if with_mem and without_m:
        logger.info(f"  Improvement:      {sr_with - sr_without:+.2%}")
    if prtree_agent is not None:
        stats = prtree_agent.get_memory_stats()
        task_hit = sum(1 for r in results if r.get("task_memory_used", False))
        env_hit  = sum(1 for r in results if r.get("env_memory_used",  False))
        logger.info(f"\n🌳 PRTree Topology (Final):")
        logger.info(f"  Task Tree Nodes: {stats.get('task_tree_nodes', 0)}")
        logger.info(f"  Env  Tree Nodes: {stats.get('env_tree_nodes',  0)}")
        logger.info(f"  Total Nodes:     {stats.get('total_nodes',     0)}")
        logger.info(f"  Max Depth:       {stats.get('max_depth',       0)}")
        logger.info(f"  Task Hit Rate:   {task_hit / n:.2%}")
        logger.info(f"  Env  Hit Rate:   {env_hit  / n:.2%}")
    logger.info("=" * 80)

    # CSV 写入
    if results_csv and results:
        from common.result_logger import append_result
        t_hit = sum(1 for r in results if r.get("task_memory_used", False))
        e_hit = sum(1 for r in results if r.get("env_memory_used",  False))
        append_result(results_csv, {
            **_key,
            "benchmark": "webshop", "model": args.model,
            "memory": args.memory, "n_episodes": len(results),
            "success_rate":   round(sr, 6),
            "avg_reward":     round(avg_r, 6),
            "avg_steps":      round(avg_s, 4),
            "memory_hit_rate": round(mem_hit / n, 4),
            "task_hit_rate":   round(t_hit / n, 4),
            "env_hit_rate":    round(e_hit / n, 4),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        logger.info(f"📄 结果已写入 {results_csv}")

    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="WebShop memory-aware runner")
    p.add_argument("--mode",   type=str, choices=["eval"], default="eval",
                   help="目前仅支持 eval；train 模式留作占位")
    p.add_argument("--model",  type=str, default="deepseek-v4-flash")
    p.add_argument("--n",      type=int, default=10, help="number of sessions to run (-1 = all)")
    p.add_argument("--start",  type=int, default=0,  help="start index into sessions list")
    p.add_argument("--max-episodes", dest="max_episodes", type=int, default=None,
                   help="先按全量裁剪到 max_episodes，再按 --start/--n 进一步切片")
    p.add_argument("--max-steps", dest="max_steps", type=int, default=15)
    p.add_argument("--sessions-file", dest="sessions_file", type=str,
                   default="/tmp/ETO/eval_agent/data/webshop/test_indices.json",
                   help="JSON list of session ids; default = ETO test_indices")
    # 记忆 backend
    p.add_argument("--memory", type=str,
                   choices=["no-memory", "synapse", "synapse_i", "awm", "reasoningbank", "prtree"],
                   default="no-memory",
                   help="Memory backend (synapse_i = synapse with prtree-style single-message prompt)")
    p.add_argument("--memory-path", dest="memory_path", type=str, default=None,
                   help="统一的 memory 路径（load+save）")
    p.add_argument("--memory-file", dest="memory_file", type=str, default=None,
                   help="Deprecated: 用 --memory-path")
    p.add_argument("--load-memory", dest="load_memory", type=str, default=None,
                   help="prtree only: 从指定路径加载已有 PRTree 记忆（train→test 跨路径）")
    p.add_argument("--resume", action="store_true",
                   help="从 --memory-path 加载已有记忆（默认冷启动）")
    p.add_argument("--freeze", action="store_true",
                   help="加载记忆但禁止写入（frozen-memory eval）")
    p.add_argument("--save-interval", dest="save_interval", type=int, default=10)
    p.add_argument("--embed-model-path", dest="embed_model_path", type=str, default=None,
                   help="ReasoningBank 用的本地 embedding 模型路径")
    # 轨迹与结果
    p.add_argument("--traj-dir",    dest="traj_dir",    type=str, default=None,
                   help="轨迹保存目录（覆盖默认路径）")
    p.add_argument("--results-csv", dest="results_csv", type=str, default=None,
                   help="结果 CSV 路径（用于 Lock1 去重）")
    return p.parse_args()


def main():
    args = parse_args()
    Path("storage").mkdir(exist_ok=True)
    Path("trajectories").mkdir(exist_ok=True)
    run_evaluation(args)


if __name__ == "__main__":
    main()
