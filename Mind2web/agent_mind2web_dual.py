"""
Dual PR-Tree Agent for Mind2Web (v4.0)
基于双树 (任务树 + 网站树) 的 Mind2Web 反思型 Agent

核心改动 (相对 ALFWorld_New):
- env_description = 网站名称 (e.g. "united.com")
- task_goal       = 任务自然语言描述 (e.g. "Book a flight from NYC to LA")
- 评估指标: element_acc, action_f1, step_success (Mind2Web 标准)
- 数据格式: 使用 Mind2Web 的 pos_candidates / neg_candidates / cleaned_html
"""

import re
import json
import logging
import numpy as np

_ELEMENT_ID_RE = re.compile(r'\[(\d+)\]')
_REFLECTION_RETRY_MSG = (
    "\n\nCRITICAL: Your previous response contained element IDs like [1234]. "
    "These are ABSOLUTELY PROHIBITED — they are meaningless in other episodes. "
    "Rewrite the entire JSON WITHOUT any [number] references. "
    "Describe elements ONLY by aria-label, placeholder, visible text, or structural position."
)
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from memory.prtree.dual_tree_manager import DualTreeMemory
from memory.prtree.dual_memory_reader import DualMemoryReader
from memory.prtree.dual_memory_writer import DualMemoryWriter
from memory.prtree.memory_node import MemoryNode, ResultStatus
from common.retriever import VectorRetriever
from common.llm_client import LLMClient

from dual_prompt import (
    mind2web_instruction,
    TaskTree_Prompt_Map, EnvTree_Prompt_Map,
    PROMPT_WITH_ICL_TEMPLATE, PROMPT_WITH_ICL_TEMPLATE_DUAL_MEMORY,
    MEMORY_HEADERS,
    get_task_prompt_key, get_env_prompt_key,
)
from mind2web_utils import (
    get_top_k_obs,
    get_target_obs_and_act,
    num_tokens_from_messages,
    parse_act_str,
    calculate_f1,
    construct_act_str,
    add_scores,
    extract_from_response,
    MAX_TOKENS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DualTreeMind2WebAgent:
    """
    基于双树 PR-Tree v4.0 的 Mind2Web Agent

    Workflow:
    1. Split:    website_name → env_description, confirmed_task → task_goal
    2. Retrieve: 分别从任务树和网站树检索记忆
    3. Act:      基于融合记忆与 Mind2Web 环境逐步交互
    4. Reflect:  分别生成任务反思和网站反思
    5. Evolve:   分别向两棵树写入经验
    """

    def __init__(
        self,
        agent_name: str,
        llm_client: LLMClient,
        exemplar_path: str = "data/example/exemplars.json",
        icl_num: int = 1,
        top_k_elements: int = 5,
        previous_top_k_elements: int = 3,
    ):
        self.agent_name = agent_name
        self.llm_client = llm_client
        self.icl_num = icl_num
        self.top_k_elements = top_k_elements
        self.previous_top_k_elements = previous_top_k_elements

        # --- 初始化双树记忆系统 ---
        self.retriever = VectorRetriever()
        self.dual_memory = DualTreeMemory(self.retriever)
        self.reader = DualMemoryReader(self.dual_memory)
        self.writer = DualMemoryWriter(self.dual_memory)

        # 加载 exemplars (ICL few-shot)
        self.all_exemplars = self._load_exemplars(exemplar_path)

        logger.info(f"✅ {agent_name} initialized with Dual PR-Tree v4.0 (Mind2Web)")

    # =================================================================
    # 数据加载
    # =================================================================

    def _load_exemplars(self, path: str) -> List:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load exemplars: {e}")
            return []

    def get_example(self, domain: str, subdomain: str, website: str) -> str:
        """筛选 few-shot 例子，优先匹配 website > subdomain > domain"""
        examples = self.all_exemplars
        if not examples:
            return ""

        if any(website in cex[0].get("specifier", "") for cex in examples):
            examples = [
                cex for cex in examples
                if all(tag in cex[0].get("specifier", "") for tag in [domain, subdomain, website])
            ]
        elif any(subdomain in cex[0].get("specifier", "") for cex in examples):
            examples = [
                cex for cex in examples
                if all(tag in cex[0].get("specifier", "") for tag in [domain, subdomain])
            ]

        if not examples:
            examples = self.all_exemplars

        import random
        chosen = random.sample(examples, min(self.icl_num, len(examples)))

        example_text = ""
        for i, step in enumerate(chosen[0]):
            content = step.get("content", "")
            if i == 0:
                example_text += content + "\n"
            elif i % 2 == 0:
                example_text += content + "\n\n"
            else:
                example_text += content + "\n"
        return example_text

    # =================================================================
    # Prompt 构建
    # =================================================================

    def _build_system_prompt(
        self,
        task_desc: str,
        domain: str,
        subdomain: str,
        website: str,
        memory_context: Optional[str],
        memory_type: str = "prtree",
    ) -> List[Dict[str, str]]:
        """构建 system message，融合双树记忆"""
        examples = self.get_example(domain, subdomain, website)

        if memory_context:
            memory_header = MEMORY_HEADERS.get(memory_type, MEMORY_HEADERS["prtree"])
            content = PROMPT_WITH_ICL_TEMPLATE_DUAL_MEMORY.format(
                instruction=mind2web_instruction,
                examples=examples,
                memory_header=memory_header,
                memory_context=memory_context,
                task=f"Task: {task_desc}",
            )
        else:
            content = PROMPT_WITH_ICL_TEMPLATE.format(
                instruction=mind2web_instruction,
                examples=examples,
                task=f"Task: {task_desc}",
            )
        return [{"role": "system", "content": content}]

    # =================================================================
    # 核心 Episode 运行
    # =================================================================

    def run_episode(
        self,
        sample: Dict[str, Any],
        model_name: str = "deepseek-v4-flash",
        no_memory: bool = False,
        no_prtree_update: bool = False,
        external_memory_str: Optional[str] = None,
        memory_mode: str = "both",
        memory_type: str = "prtree",
        episode_idx: int = -1,
    ) -> Dict[str, Any]:
        """
        运行一个完整 Mind2Web Episode

        Args:
            sample: Mind2Web 数据样本，包含 confirmed_task, website, actions, action_reprs 等
            model_name: 用于 token 计数
            no_memory: 若为 True，跳过所有 PRTree 记忆操作（检索/反思/写入），作为 baseline 运行
            external_memory_str: 若不为 None，直接使用该字符串作为 memory_context 注入 prompt，
                                  跳过 PRTree 内部检索（但仍执行反思与写入）。
                                  可用于接入任意外部 Memory 系统（flat list、RAG 等）。
                                  优先级：no_memory > external_memory_str > PRTree 自动检索

        Returns:
            包含 element_acc, action_f1, step_success, success 等指标的结果字典
        """
        task_desc = sample["confirmed_task"]
        website    = sample.get("website", "unknown")
        domain     = sample.get("domain", "")
        subdomain  = sample.get("subdomain", "")

        # Composite env key: domain::subdomain::website
        # subdomain captures task category (e.g., flight/hotel/rental), giving finer-grained
        # env tree nodes vs. domain::website alone. Falls back gracefully when fields missing.
        env_key = "::".join(filter(None, [domain, subdomain, website])) if (domain or subdomain) else website

        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Task: {task_desc[:80]}...")
        logger.info(f"🌐 Website: {website} | Domain: {domain}/{subdomain} | EnvKey: {env_key}")

        # =================================================================
        # Phase 1: Dual Retrieval (DFS 检索)
        # 任务树 key: task_desc
        # 网站树 key: website (e.g. "united.com")
        # =================================================================
        if no_memory:
            # Baseline / AWM 模式：跳过 PRTree 检索
            task_path = []
            env_path = []
            task_memory_used = False
            env_memory_used = False
            if external_memory_str is not None:
                # AWM 等外部记忆注入：使用外部 memory 字符串，但禁止 PRTree 操作
                memory_context = external_memory_str
                memory_used = True
                logger.info(f"🔌 External memory injected in no-prtree mode ({len(external_memory_str)} chars).")
            else:
                memory_context = None
                memory_used = False
                logger.info("🚫 Memory disabled (baseline): skipping retrieval.")
        elif external_memory_str is not None:
            # 外部 Memory 注入模式：跳过 PRTree 检索，直接使用外部提供的 memory 字符串
            task_path = []
            env_path = []
            memory_context = external_memory_str
            task_memory_used = False
            env_memory_used = False
            memory_used = True
            logger.info(f"🔌 External memory injected ({len(external_memory_str)} chars), skipping PRTree retrieval.")
        else:
            # 获取两棵树的检索路径（无论 memory_mode 如何，均检索两棵树，路径用于写入）
            dual_paths = self.reader.get_dual_paths(task_desc, env_key)
            task_path  = dual_paths["task_path"]
            env_path   = dual_paths["env_path"]

            # 根据 memory_mode 决定注入 Prompt 的记忆内容
            if memory_mode == "task_only":
                task_text = self.reader.render_task_memory(task_path)
                memory_context = (
                    "# Task Strategy Memory\n"
                    "The following are experiences from similar tasks that may help guide your workflow and strategy:\n\n"
                    f"{task_text}"
                ) if task_text else None
                task_memory_used = not self.reader._is_empty_path(task_path)
                env_memory_used  = False
            elif memory_mode == "env_only":
                env_text = self.reader.render_env_memory(env_path)
                memory_context = (
                    "# Environment Knowledge Memory\n"
                    "The following are experiences from similar environments:\n\n"
                    f"{env_text}"
                ) if env_text else None
                task_memory_used = False
                env_memory_used  = not self.reader._is_empty_path(env_path)
            else:
                # "both"：双树融合（默认行为）
                memory_context   = self.reader.get_dual_narrative_context(
                    task_desc, env_key, task_path=task_path, env_path=env_path
                )
                task_memory_used = not self.reader._is_empty_path(task_path)
                env_memory_used  = not self.reader._is_empty_path(env_path)

            memory_used = task_memory_used or env_memory_used

            if task_memory_used:
                logger.info(f"💡 Task Tree Hit. Depth={len(task_path)}. "
                            f"Last: {task_path[-1].payload['scenario_description'][:50]}...")
            else:
                logger.info("🆕 Task Tree: No match (Zero-shot).")
            if env_memory_used:
                logger.info(f"🌐 Website Tree Hit. Depth={len(env_path)}. "
                            f"Last: {env_path[-1].payload['scenario_description'][:50]}...")
            else:
                logger.info("🆕 Website Tree: No match (Zero-shot).")

        # =================================================================
        # Phase 2: Execution (Mind2Web 步骤执行)
        # =================================================================
        sys_messages = self._build_system_prompt(task_desc, domain, subdomain, website, memory_context, memory_type)

        element_acc_list = []
        action_f1_list   = []
        step_success_list = []
        conversation      = []
        episode_length    = len(sample["action_reprs"])

        prev_obs     = []
        prev_actions = []
        trajectory   = [f"Task: {task_desc}"]

        for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
            _, target_act = get_target_obs_and_act(s)

            pos_candidates = [
                c for c in s["pos_candidates"] if c["rank"] < self.top_k_elements
            ]

            # --- Ground truth element not in cleaned html ---
            if len(pos_candidates) == 0:
                element_acc_list.append(0)
                action_f1_list.append(0)
                step_success_list.append(0)
                target_obs, _ = get_top_k_obs(s, self.previous_top_k_elements)
                prev_obs.append("Observation: `" + target_obs + "`")
                prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")
                conversation.append("The ground truth element is not in cleaned html")
                trajectory.append(f"[SKIP] Ground truth not in cleaned html | target: {target_act}")
                continue

            # --- 构建 query messages ---
            query = []
            for o, a in zip(prev_obs, prev_actions):
                if len(query) == 0:
                    query.append({
                        "role": "user",
                        "content": f"Task: {task_desc}\nTrajectory:\n{o}",
                    })
                else:
                    query.append({"role": "user", "content": o})
                query.append({"role": "assistant", "content": a})

            obs, _ = get_top_k_obs(s, self.top_k_elements, use_raw=False)
            obs_content = "Observation: `" + obs + "`"
            if len(query) == 0:
                query.append({
                    "role": "user",
                    "content": f"Task: {task_desc}\nTrajectory:\n{obs_content}",
                })
            else:
                query.append({"role": "user", "content": obs_content})

            target_obs, _ = get_top_k_obs(s, self.previous_top_k_elements)
            prev_obs.append("Observation: `" + target_obs + "`")
            prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")

            full_messages = sys_messages + query

            # --- Token check ---
            total_tokens = num_tokens_from_messages(full_messages, model_name)
            if total_tokens > MAX_TOKENS.get(model_name, 128000):
                logger.warning(f"Token limit exceeded ({total_tokens}), skipping step.")
                element_acc_list.append(0)
                action_f1_list.append(0)
                step_success_list.append(0)
                conversation.append({
                    "input": full_messages,
                    "output": f"FAILED DUE TO CONTEXT LIMIT: {total_tokens}",
                })
                trajectory.append(f"[SKIP] Token limit {total_tokens}")
                continue

            # --- LLM 推理 ---
            try:
                llm_response = self.llm_client.chat(full_messages)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                element_acc_list.append(0)
                action_f1_list.append(0)
                step_success_list.append(0)
                conversation.append({"input": full_messages, "output": f"LLM ERROR: {e}"})
                trajectory.append(f"[ERROR] {e}")
                continue

            conversation.append({"input": full_messages, "output": llm_response})

            # --- 解析动作 & 计算指标 ---
            pred_act = extract_from_response(llm_response, backtick="`").strip()
            pred_op, pred_id, pred_val = parse_act_str(pred_act)
            target_op, _, target_val   = parse_act_str(target_act)

            pos_ids = [c["backend_node_id"] for c in s["pos_candidates"]][:1]
            elem_acc = 1 if pred_id in pos_ids else 0
            act_f1   = calculate_f1(
                construct_act_str(pred_op, pred_val),
                construct_act_str(target_op, target_val),
            )
            step_ok  = 1 if pred_act == target_act else 0
            element_acc_list.append(elem_acc)
            action_f1_list.append(act_f1)
            step_success_list.append(step_ok)
            conversation.append({"pred_act": pred_act, "target_act": target_act})
            # trajectory 附上每步三项指标，让反思 LLM 知道哪里出了问题
            trajectory.append(
                f"Step {len(step_success_list)}: "
                f"pred=`{pred_act}` | target=`{target_act}` | "
                f"elem_acc={elem_acc} action_f1={act_f1:.2f} step_success={step_ok}"
            )

        # --- Episode 成功判断 ---
        success = int(np.sum(step_success_list[-episode_length:]) == episode_length)

        logger.info(
            f"{'✅ SUCCESS' if success else '❌ FAILED'} | "
            f"elem_acc={np.mean(element_acc_list):.2f} "
            f"act_f1={np.mean(action_f1_list):.2f} "
            f"step_sr={np.mean(step_success_list):.2f}"
        )

        # =================================================================
        # Phase 3: Dual Reflection (双树反思)
        # Phase 4: Dual Evolution (双树写入)
        # =================================================================
        if no_memory or no_prtree_update:
            if no_prtree_update:
                logger.info("🔌 External memory mode: skipping PRTree reflection & update.")
            else:
                logger.info("🚫 Memory disabled (no-memory baseline): skipping reflection & tree update.")
            return {
                "success": bool(success),
                "element_acc": element_acc_list,
                "action_f1": action_f1_list,
                "step_success": step_success_list,
                "memory_used": memory_used,
                "task_memory_used": task_memory_used,
                "env_memory_used": env_memory_used,
                "task_retrieval_length": 0,
                "env_retrieval_length": 0,
                "task_reflection": {},
                "env_reflection": {},
                "task_node_id": None,
                "env_node_id": None,
                "conversation": conversation,
                "trajectory": trajectory,
            }

        task_reflection, env_reflection = self._generate_dual_reflection(
            task_goal=task_desc,
            env_description=env_key,
            success=bool(success),
            steps=len(step_success_list),
            trajectory=trajectory,
            task_path=task_path,
            env_path=env_path,
            element_acc_list=element_acc_list,
            action_f1_list=action_f1_list,
            step_success_list=step_success_list,
        )

        status = ResultStatus.SUCCESS if success else ResultStatus.FAILURE
        task_skip = task_reflection.get("skip", False)
        env_skip = env_reflection.get("skip", False)

        task_node = None
        env_node = None

        if not task_skip:
            task_node = self.writer.write_task_experience(
                scenario_description=task_desc,
                skill=task_reflection,
                result_status=status,
                retrieved_path=task_path,
                episode_idx=episode_idx,
            )
        else:
            logger.info("⏭ Task: existing skill covers this episode, skipping node creation.")

        if not env_skip:
            env_node = self.writer.write_env_experience(
                scenario_description=env_key,
                skill=env_reflection,
                result_status=ResultStatus.SUCCESS,  # env 知识始终以 SUCCESS 存储
                retrieved_path=env_path,
                episode_idx=episode_idx,
            )
        else:
            logger.info("⏭ Env: no new env knowledge, skipping node creation.")

        # 更新成功计数并触发固化检查
        if success:
            deepest_retrieved = next(
                (n for n in reversed(task_path)
                 if "GLOBAL_ROOT_PLACEHOLDER" not in n.payload.get("scenario_description", "")),
                None
            )
            if deepest_retrieved is not None:
                deepest_retrieved.meta["success_count"] = deepest_retrieved.meta.get("success_count", 0) + 1
                self.dual_memory.trigger_consolidation_check(deepest_retrieved, self.llm_client, tree_type="task")

            deepest_env_retrieved = next(
                (n for n in reversed(env_path)
                 if "GLOBAL_ROOT_PLACEHOLDER" not in n.payload.get("scenario_description", "")),
                None
            )
            if deepest_env_retrieved is not None:
                deepest_env_retrieved.meta["success_count"] = deepest_env_retrieved.meta.get("success_count", 0) + 1
                self.dual_memory.trigger_consolidation_check(deepest_env_retrieved, self.llm_client, tree_type="env")

        return {
            "success": bool(success),
            "element_acc": element_acc_list,
            "action_f1": action_f1_list,
            "step_success": step_success_list,
            "memory_used": memory_used,
            "task_memory_used": task_memory_used,
            "env_memory_used": env_memory_used,
            "task_retrieval_length": len(task_path) - 1 if task_memory_used else 0,
            "env_retrieval_length": len(env_path) - 1 if env_memory_used else 0,
            "task_reflection": task_reflection,
            "env_reflection": env_reflection,
            "task_skip": task_skip,
            "env_skip": env_skip,
            "task_node_id": task_node.node_id if task_node else None,
            "env_node_id": env_node.node_id if env_node else None,
            "conversation": conversation,
            "trajectory": trajectory,
        }

    # =================================================================
    # Phase 3: 双树反思生成
    # =================================================================

    def _generate_dual_reflection(
        self,
        task_goal: str,
        env_description: str,
        success: bool,
        steps: int,
        trajectory: List[str],
        task_path: List[MemoryNode],
        env_path: List[MemoryNode],
        element_acc_list: List[float] = None,
        action_f1_list: List[float] = None,
        step_success_list: List[int] = None,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        # 构建带指标摘要的 trajectory 字符串
        # Mind2Web 成功条件：ALL steps 的 element_acc=1 AND action_f1=1 AND step_success=1
        if element_acc_list and action_f1_list and step_success_list:
            avg_elem  = np.mean(element_acc_list)
            avg_f1    = np.mean(action_f1_list)
            avg_step  = np.mean(step_success_list)
            metrics_summary = (
                f"[Mind2Web Metrics] "
                f"element_acc={avg_elem:.3f} (correct element selected per step) | "
                f"action_f1={avg_f1:.3f} (op+value token F1 per step) | "
                f"step_success={avg_step:.3f} (exact full-step match per step)\n"
                f"[Success Condition] ALL steps must satisfy: element correct AND action_f1=1 AND step_success=1\n"
                f"[Overall] {'SUCCESS' if success else 'FAILURE'} "
                f"(failed steps: {sum(1 for s in step_success_list if s == 0)}/{len(step_success_list)})"
            )
        else:
            metrics_summary = f"[Overall] {'SUCCESS' if success else 'FAILURE'}"
        traj_str = metrics_summary + "\n\nStep-by-step Trajectory:\n" + "\n".join(trajectory)

        # --- 步骤级失败分析 (改进2) ---
        # 从 trajectory 中提取 element_acc=0 的步骤，供反思 LLM 做元素辨识分析
        failed_steps_analysis = self._build_failed_steps_analysis(
            trajectory, element_acc_list, action_f1_list, step_success_list, success
        )

        # --- Task Tree Reflection ---
        task_is_root = self.reader._is_empty_path(task_path)
        task_compare_memory = (
            "" if task_is_root
            else self.reader.render_task_path_for_reflection(task_path)
        )
        task_key      = get_task_prompt_key(task_is_root, success)
        task_template = TaskTree_Prompt_Map[task_key]
        if task_is_root:
            task_prompt = task_template.format(
                env_description=env_description,
                task_description=task_goal,
                steps=steps,
                trajectory=traj_str,
                failed_steps_analysis=failed_steps_analysis,
            )
        else:
            task_prompt = task_template.format(
                env_description=env_description,
                task_description=task_goal,
                retrieved_task_memory=task_compare_memory,
                steps=steps,
                trajectory=traj_str,
                failed_steps_analysis=failed_steps_analysis,
            )
        task_resp = self.llm_client.chat([{"role": "user", "content": task_prompt}], temperature=0.0, max_tokens=8192)
        if _ELEMENT_ID_RE.search(task_resp):
            logger.warning("Task reflection contains element IDs — retrying with correction prompt.")
            task_resp = self.llm_client.chat(
                [{"role": "user", "content": task_prompt},
                 {"role": "assistant", "content": task_resp},
                 {"role": "user", "content": _REFLECTION_RETRY_MSG}],
                temperature=0.0,
            )
        task_reflection = self._parse_json_response(task_resp)

        # --- Website (Env) Tree Reflection ---
        env_is_root = self.reader._is_empty_path(env_path)
        env_compare_memory = (
            "" if env_is_root
            else self.reader.render_env_path_for_reflection(env_path)
        )
        env_key      = get_env_prompt_key(env_is_root, success)
        env_template = EnvTree_Prompt_Map[env_key]
        if env_is_root:
            env_prompt = env_template.format(
                env_description=env_description,
                task_description=task_goal,
                steps=steps,
                trajectory=traj_str,
                failed_steps_analysis=failed_steps_analysis,
            )
        else:
            env_prompt = env_template.format(
                env_description=env_description,
                task_description=task_goal,
                retrieved_env_memory=env_compare_memory,
                steps=steps,
                trajectory=traj_str,
                failed_steps_analysis=failed_steps_analysis,
            )
        env_resp = self.llm_client.chat([{"role": "user", "content": env_prompt}], temperature=0.0, max_tokens=8192)
        if _ELEMENT_ID_RE.search(env_resp):
            logger.warning("Env reflection contains element IDs — retrying with correction prompt.")
            env_resp = self.llm_client.chat(
                [{"role": "user", "content": env_prompt},
                 {"role": "assistant", "content": env_resp},
                 {"role": "user", "content": _REFLECTION_RETRY_MSG}],
                temperature=0.0,
            )
        env_reflection = self._parse_json_response(env_resp)

        return task_reflection, env_reflection

    # =================================================================
    # 工具方法
    # =================================================================

    def _build_failed_steps_analysis(
        self,
        trajectory: List[str],
        element_acc_list: Optional[List[float]],
        action_f1_list: Optional[List[float]],
        step_success_list: Optional[List[int]],
        success: bool,
    ) -> str:
        """
        从逐步指标中提取 element_acc=0 的步骤，构建供反思 LLM 使用的失败分析摘要。
        成功 episode 返回简短的全通过说明。
        """
        if success or not element_acc_list:
            return "All steps passed — no element selection failures to analyze."

        failed_lines = []
        traj_steps = [t for t in trajectory if t.startswith("Step ")]
        for i, (e_acc, f1, s_ok) in enumerate(zip(element_acc_list, action_f1_list or [], step_success_list or [])):
            if e_acc == 0 or s_ok == 0:
                traj_line = traj_steps[i] if i < len(traj_steps) else f"Step {i+1}: (no trajectory info)"
                if e_acc == 0:
                    # 元素选错：主要影响 element_acc 和 step_success
                    failure_type = f"WRONG ELEMENT (element_acc=0, action_f1={f1:.2f}, step_success={int(s_ok)})"
                    hint = "Analyze: what semantic/structural signals distinguish the CORRECT element from the selected one?"
                else:
                    # 元素对但 op/value 错：主要影响 step_success，拉低 action_f1
                    failure_type = f"WRONG OP/VALUE (element_acc=1, action_f1={f1:.2f}, step_success={int(s_ok)})"
                    hint = "Analyze: element was correct but action type (CLICK/TYPE/SELECT) or value string was wrong — what is the correct format/value?"
                failed_lines.append(
                    f"- {traj_line}\n"
                    f"  → Failure type: {failure_type}\n"
                    f"  → {hint}"
                )

        if not failed_lines:
            return "No element selection failures detected."
        header = f"{len(failed_lines)} step(s) failed element selection:\n"
        return header + "\n".join(failed_lines)

    @staticmethod
    def _escape_json_strings(text: str) -> str:
        """将 JSON 字符串值中的字面换行/制表符转义，同时跳过已转义字符。"""
        result = []
        in_string = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == '\\' and in_string:
                # 已转义序列，原样保留两个字符
                result.append(ch)
                i += 1
                if i < len(text):
                    result.append(text[i])
            elif ch == '"':
                result.append(ch)
                in_string = not in_string
            elif in_string and ch == '\n':
                result.append('\\n')
            elif in_string and ch == '\r':
                result.append('\\r')
            elif in_string and ch == '\t':
                result.append('\\t')
            else:
                result.append(ch)
            i += 1
        return ''.join(result)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析 JSON，降级时始终返回 Skill 格式"""
        raw = response.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        # 尝试1：直接解析
        try:
            return json.loads(raw, strict=False)
        except json.JSONDecodeError:
            pass

        # 尝试2：转义字符串内的字面换行符后再解析
        try:
            return json.loads(self._escape_json_strings(raw), strict=False)
        except json.JSONDecodeError:
            pass

        # 尝试3：去除非打印控制字符后解析
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
        try:
            return json.loads(self._escape_json_strings(cleaned), strict=False)
        except json.JSONDecodeError:
            pass

        # 尝试4：正则提取三字段（容忍字段值中的转义引号）
        _field = r'"((?:[^"\\]|\\.)*)"'
        ac = re.search(r'"activation_condition"\s*:\s*' + _field, raw, re.DOTALL)
        ep = re.search(r'"execution_procedure"\s*:\s*' + _field, raw, re.DOTALL)
        tc = re.search(r'"termination_condition"\s*:\s*' + _field, raw, re.DOTALL)
        if ac and ep:
            return {
                "activation_condition": ac.group(1).replace("\\n", " ").replace("\n", " ").strip(),
                "execution_procedure":  ep.group(1).replace("\\n", "\n").strip(),
                "termination_condition": tc.group(1).replace("\\n", " ").strip() if tc else "",
            }

        logger.warning(f"JSON parse fallback. Raw[:200]={raw[:200]}")
        summary = raw[:120].replace('"', "'").replace("\n", " ").strip()
        return {
            "activation_condition": summary,
            "execution_procedure": raw.replace('"', "'").replace("\n", " ").strip(),
            "termination_condition": "",
        }

    # --- 持久化 ---
    def load_memory(self, filepath: str):
        self.dual_memory.load_tree(filepath)
        logger.info(f"📥 Memory loaded from {filepath}")

    def save_memory(self, filepath: str):
        self.dual_memory.save_trees(
            task_filepath=filepath.replace(".json", "") + "_task.json",
            env_filepath=filepath.replace(".json", "") + "_env.json",
        )
        logger.info(f"💾 Memory saved to {filepath}")

    def get_memory_stats(self) -> Dict[str, Any]:
        return self.dual_memory.stats
