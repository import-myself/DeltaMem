"""
Reflective Agent with Dual PR-Tree Architecture (v4.0)
基于双树 (任务树 + 环境树) 的反思型 Agent

核心改动:
- 将 ALFWorld 的 task_instruction 拆分为 env_description 和 task_goal
- 分别在两棵树上做检索和写入
- 反思时分别生成 任务反思 和 环境反思
- Prompt 注入时融合两棵树的记忆
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import traceback

from memory.prtree.dual_tree_manager import DualTreeMemory
from memory.prtree.dual_memory_reader import DualMemoryReader
from memory.prtree.dual_memory_writer import DualMemoryWriter
from memory.prtree.memory_node import MemoryNode, ResultStatus
from common.retriever import VectorRetriever
from common.llm_client import LLMClient

from dual_prompt import (
    alfworld_instruction,
    TaskTree_Prompt_Map, EnvTree_Prompt_Map,
    PROMPT_WITH_ICL_TEMPLATE, PROMPT_WITH_ICL_TEMPLATE_DUAL_MEMORY,
    get_task_prompt_key, get_env_prompt_key
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ') + 2:]
    return ob


def split_task_instruction(task_instruction: str) -> Tuple[str, str]:
    """
    将 ALFWorld 的 task_instruction 拆分为环境描述和任务目标
    
    ALFWorld 格式:
      "You are in the middle of a room. Looking quickly around you, you see a cabinet 4, ...
       Your task is to: put some spraybottle on toilet."
    
    Returns:
        (env_description, task_goal)
    """
    # 尝试按 "Your task is to:" 分割
    separator = "Your task is to:"
    if separator in task_instruction:
        parts = task_instruction.split(separator, 1)
        env_description = parts[0].strip()
        task_goal = separator + parts[1].strip()
        return env_description, task_goal
    
    # 备选: 按 "\nYour task is to:" 分割
    separator2 = "\nYour task is to:"
    if separator2 in task_instruction:
        parts = task_instruction.split(separator2, 1)
        env_description = parts[0].strip()
        task_goal = "Your task is to:" + parts[1].strip()
        return env_description, task_goal
    
    # 如果找不到分隔符，整体作为任务描述，环境为空
    logger.warning("Could not split task instruction. Using full text as task goal.")
    return "", task_instruction


class DualTreeReflectiveAgent:
    """
    基于双树 PR-Tree v4.0 的反思型 Agent
    
    Workflow:
    1. Split: 将 task_instruction 拆分为 env_description + task_goal
    2. Retrieve: 分别从任务树和环境树检索
    3. Act: 基于融合记忆与环境交互
    4. Reflect: 分别生成任务反思和环境反思
    5. Evolve: 分别向两棵树写入经验
    """

    def __init__(
        self,
        agent_name: str,
        llm_client: LLMClient,
        icl_num: int = 1,
        icl_data_path: str = "data/alfworld_icl.json"
    ):
        self.agent_name = agent_name
        self.llm_client = llm_client
        self.icl_num = icl_num

        # --- 初始化双树记忆系统 ---
        self.retriever = VectorRetriever()
        self.dual_memory = DualTreeMemory(self.retriever)
        self.reader = DualMemoryReader(self.dual_memory)
        self.writer = DualMemoryWriter(self.dual_memory)

        # 加载 ICL 数据
        self.icl_data = self._load_icl_data(icl_data_path)

        logger.info(f"✅ {agent_name} initialized with Dual PR-Tree v4.0 Architecture")

    def _load_icl_data(self, filepath: str) -> Dict:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load ICL data: {e}")
            return {}

    def build_icl_messages(self, task_type: str) -> List[Dict[str, str]]:
        icl_messages = []
        if task_type in self.icl_data and self.icl_num > 0:
            examples = self.icl_data[task_type][:self.icl_num]
            for example_dialogue in examples:
                icl_messages.extend(example_dialogue)
        return icl_messages

    def run_episode(
        self,
        task_instruction: str,
        env: Any,
        task_type: str = "unknown",
        max_steps: int = 30,
        no_memory: bool = False,
        external_memory_str: Optional[str] = None,
        memory_mode: str = "both",
    ) -> Dict[str, Any]:
        """
        运行一个完整 Episode (双树版本)

        Args:
            no_memory: 若为 True，跳过所有 PRTree 记忆操作（检索/反思/写入），作为 baseline 运行
            external_memory_str: 若不为 None，直接使用该字符串作为 memory_context 注入 prompt，
                                  跳过 PRTree 内部检索（但仍执行反思与写入）。
                                  可用于接入任意外部 Memory 系统（flat list、RAG 等）。
                                  优先级：no_memory > external_memory_str > PRTree 自动检索
        """
        if self.llm_client is None:
            raise ValueError("LLM client not initialized")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"🎯 Starting episode: {task_type}")
        logger.info(f"📝 Task: {task_instruction[:100]}...")

        # =================================================================
        # Phase 0: Split task_instruction → env_description + task_goal
        # =================================================================
        env_description, task_goal = split_task_instruction(task_instruction)
        logger.info(f"🌍 Env: {env_description[:80]}...")
        logger.info(f"🎯 Goal: {task_goal}")

        # =================================================================
        # FAST PATH: O(1) Skill Cache 匹配
        # =================================================================
        fast_path_used = False
        if not no_memory and external_memory_str is None:
            skill_state = {"task": task_goal, "env": env_description}
            matched_skill = self.dual_memory.skill_cache.check_match(skill_state)
            if matched_skill:
                fast_path_used = True
                task_path = []
                env_path = []
                task_memory_used = True
                env_memory_used = False
                memory_used = True
                memory_context_str = (
                    "# Skill-Based Memory (Fast Path)\n\n"
                    f"**Activation Condition**: {matched_skill.activation_condition}\n\n"
                    f"**Execution Procedure**:\n{matched_skill.execution_procedure}\n\n"
                    f"**Termination Condition**: {matched_skill.termination_condition}"
                )
                logger.info(f"⚡ FAST PATH: Skill patch hit (node {matched_skill.source_node_id[:8]})")

        # =================================================================
        # Phase 1: Dual Retrieval (慢路径 DFS 检索)
        # =================================================================

        if fast_path_used:
            pass  # fast path already set above
        elif no_memory:
            # Baseline / AWM 模式：跳过 PRTree 检索
            task_path = []
            env_path = []
            task_memory_used = False
            env_memory_used = False
            if external_memory_str is not None:
                # AWM 等外部记忆注入：使用外部 memory 字符串，但禁止 PRTree 操作
                memory_context_str = external_memory_str
                memory_used = True
                logger.info(f"🔌 External memory injected in no-prtree mode ({len(external_memory_str)} chars).")
            else:
                memory_context_str = None
                memory_used = False
                logger.info("🚫 Memory disabled (baseline): skipping retrieval.")
        elif external_memory_str is not None:
            # 外部 Memory 注入模式：跳过 PRTree 检索，直接使用外部提供的 memory 字符串
            task_path = []
            env_path = []
            memory_context_str = external_memory_str
            task_memory_used = False
            env_memory_used = False
            memory_used = True
            logger.info(f"🔌 External memory injected ({len(external_memory_str)} chars), skipping PRTree retrieval.")
        else:
            # 获取两棵树的检索路径（无论 memory_mode 如何，均检索两棵树，路径用于写入）
            dual_paths = self.reader.get_dual_paths(task_goal, env_description)
            task_path = dual_paths["task_path"]
            env_path = dual_paths["env_path"]

            # 根据 memory_mode 决定注入 Prompt 的记忆内容
            if memory_mode == "task_only":
                # 仅注入任务树记忆，忽略环境树
                task_text = self.reader.render_task_memory(task_path)
                memory_context_str = (
                    "# Task Strategy Memory\n"
                    "The following are experiences from similar tasks that may help guide your workflow and strategy:\n\n"
                    f"{task_text}"
                ) if task_text else None
                task_memory_used = not self.reader._is_empty_path(task_path)
                env_memory_used  = False
            elif memory_mode == "env_only":
                # 仅注入环境树记忆，忽略任务树
                env_text = self.reader.render_env_memory(env_path)
                memory_context_str = (
                    "# Environment Knowledge Memory\n"
                    "The following are experiences from similar environments that may help with navigation and object interaction:\n\n"
                    f"{env_text}"
                ) if env_text else None
                task_memory_used = False
                env_memory_used  = not self.reader._is_empty_path(env_path)
            else:
                # "both"：双树融合（默认行为）
                memory_context_str = self.reader.get_dual_narrative_context(task_goal, env_description)
                task_memory_used = not self.reader._is_empty_path(task_path)
                env_memory_used  = not self.reader._is_empty_path(env_path)

            memory_used = task_memory_used or env_memory_used

            if task_memory_used:
                logger.info(f"💡 Task Tree Hit. Depth: {len(task_path)}. Last: {task_path[-1].payload['scenario_description'][:50]}...")
            else:
                logger.info("🆕 Task Tree: No match (Zero-shot).")

            if env_memory_used:
                logger.info(f"🌍 Env Tree Hit. Depth: {len(env_path)}. Last: {env_path[-1].payload['scenario_description'][:50]}...")
            else:
                logger.info("🆕 Env Tree: No match (Zero-shot).")

        # =================================================================
        # Phase 2: Execution (任务执行)
        # =================================================================

        messages = self._build_prompt(task_instruction, task_type, memory_context_str)

        observation = task_instruction
        success = False
        steps = 0
        trajectory = [f"Observation: {observation}"]

        for step in range(max_steps):
            if step == 0:
                logger.info(f"Init Obs: {observation[:100]}...")
            else:
                logger.info(f"Step {step} Obs: {observation[:100]}...")
                messages.append({"role": "user", "content": f"Observation: {observation}"})

            if step > 0:
                trajectory.append(f"Observation: {observation}")

            try:
                llm_response = self.llm_client.chat(messages)
                action = self._parse_action(llm_response)

                obs, reward, done, info = env.step([action])
                observation, reward, done = process_ob(obs[0]), reward[0], done[0]

                logger.info(llm_response)
            except Exception as e:
                logger.error(f"Env Step Error: {e}")
                observation = "Observation: Error Input. Your input must contains 'Action: ' and a valid action. Please check the environment action space for valid actions."
                done = False
                reward = 0

            trajectory.append(llm_response)
            messages.append({"role": "assistant", "content": llm_response})

            steps += 1
            if done:
                success = reward > 0
                break

        logger.info(f"{'✅ SUCCESS' if success else '❌ FAILED'} in {steps} steps")

        # =================================================================
        # Phase 3: Dual Reflection (双树反思)
        # Phase 4: Dual Evolution (双树写入)
        # =================================================================

        if no_memory:
            # Baseline 模式：跳过反思和写入
            logger.info("🚫 Memory disabled (baseline): skipping reflection & tree update.")
            messages.append({
                'success': success,
                'steps': steps,
                'memory_used': False,
                'task_memory_used': False,
                'env_memory_used': False,
                'task_retrieval_length': 0,
                'env_retrieval_length': 0,
                'task_reflection': {},
                'env_reflection': {},
                'trajectory': trajectory,
                'task_node_id': None,
                'env_node_id': None
            })
        else:
            task_reflection, env_reflection = self._generate_dual_reflection(
                task_goal=task_goal,
                env_description=env_description,
                success=success,
                steps=steps,
                trajectory=trajectory,
                task_path=task_path,
                env_path=env_path
            )

            status = ResultStatus.SUCCESS if success else ResultStatus.FAILURE

            new_nodes = self.writer.write_dual_experience(
                task_description=task_goal,
                env_description=env_description,
                task_reflection=task_reflection,
                env_reflection=env_reflection,
                status=status,
                task_retrieved_path=task_path,
                env_retrieved_path=env_path
            )

            # 更新成功计数并触发固化检查
            if success:
                task_anchor = new_nodes["task_node"]
                task_anchor.meta["success_count"] = task_anchor.meta.get("success_count", 0) + 1
                self.dual_memory.trigger_consolidation_check(task_anchor, self.llm_client)

            messages.append({
                'success': success,
                'steps': steps,
                'memory_used': memory_used,
                'task_memory_used': task_memory_used,
                'env_memory_used': env_memory_used,
                'task_retrieval_length': len(task_path) - 1 if task_memory_used else 0,
                'env_retrieval_length': len(env_path) - 1 if env_memory_used else 0,
                'task_reflection': task_reflection,
                'env_reflection': env_reflection,
                'trajectory': trajectory,
                'task_node_id': new_nodes["task_node"].node_id,
                'env_node_id': new_nodes["env_node"].node_id
            })

        return messages

    def _build_prompt(
        self,
        task_instruction: str,
        task_type: str,
        memory_context_str: Optional[str]
    ) -> List[Dict[str, str]]:
        """构建 Prompt (支持双树记忆融合)"""
        icl_text = ""
        icl_messages = self.build_icl_messages(task_type)
        if icl_messages:
            for i, msg in enumerate(icl_messages):
                if i == 0:
                    icl_text += f"{msg['content']}\n"
                elif i % 2 == 0:
                    icl_text += f"{msg['content']}\n\n"
                else:
                    icl_text += f"{msg['content']}\n"

        if memory_context_str:
            full_content = PROMPT_WITH_ICL_TEMPLATE_DUAL_MEMORY.format(
                instruction=alfworld_instruction,
                examples=icl_text,
                memory_context=memory_context_str,
                task=task_instruction
            )
        else:
            full_content = PROMPT_WITH_ICL_TEMPLATE.format(
                instruction=alfworld_instruction,
                examples=icl_text,
                task=task_instruction
            )

        return [{"role": "user", "content": full_content}]

    def _generate_dual_reflection(
        self,
        task_goal: str,
        env_description: str,
        success: bool,
        steps: int,
        trajectory: List[str],
        task_path: List[MemoryNode],
        env_path: List[MemoryNode]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        生成双树结构化反思
        
        Returns:
            (task_reflection, env_reflection)
            每个都是 {"memory_description": "...", "content_body": "..."}
        """
        traj_str = "\n".join(trajectory)

        # =====================================================
        # Task Tree Reflection
        # =====================================================
        task_is_root = self.reader._is_empty_path(task_path)

        task_compare_memory = ""
        if not task_is_root:
            task_compare_memory = self.reader.render_task_path_for_reflection(task_path)

        task_prompt_key = get_task_prompt_key(task_is_root, success)
        task_template = TaskTree_Prompt_Map[task_prompt_key]

        if task_is_root:
            task_prompt = task_template.format(
                env_description=env_description,
                task_description=task_goal,
                steps=steps,
                trajectory=traj_str
            )
        else:
            task_prompt = task_template.format(
                env_description=env_description,
                task_description=task_goal,
                retrieved_task_memory=task_compare_memory,
                steps=steps,
                trajectory=traj_str
            )

        task_response = self.llm_client.chat(
            [{"role": "user", "content": task_prompt}],
            temperature=0.0
        )
        task_reflection = self._parse_json_response(task_response)

        # =====================================================
        # Env Tree Reflection
        # =====================================================
        env_is_root = self.reader._is_empty_path(env_path)

        env_compare_memory = ""
        if not env_is_root:
            env_compare_memory = self.reader.render_env_path_for_reflection(env_path)

        env_prompt_key = get_env_prompt_key(env_is_root, success)
        env_template = EnvTree_Prompt_Map[env_prompt_key]

        if env_is_root:
            env_prompt = env_template.format(
                env_description=env_description,
                task_description=task_goal,
                steps=steps,
                trajectory=traj_str
            )
        else:
            env_prompt = env_template.format(
                env_description=env_description,
                task_description=task_goal,
                retrieved_env_memory=env_compare_memory,
                steps=steps,
                trajectory=traj_str
            )

        env_response = self.llm_client.chat(
            [{"role": "user", "content": env_prompt}],
            temperature=0.0
        )
        env_reflection = self._parse_json_response(env_response)

        return task_reflection, env_reflection

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """\u9c81\u68d2\u5730\u89e3\u6790 JSON\uff0c\u591a\u5c42\u964d\u7ea7\u7b56\u7565"""
        raw = response.strip()

        # Step 1: \u63d0\u53d6\u4ee3\u7801\u5757\u5185\u5bb9
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        # Step 2: \u76f4\u63a5\u89e3\u6790\uff08strict=False \u5141\u8bb8\u63a7\u5236\u5b57\u7b26\uff09
        try:
            return json.loads(raw, strict=False)
        except json.JSONDecodeError:
            pass

        # Step 3: \u7528 re \u6e05\u7406\u88f8\u63a7\u5236\u5b57\u7b26\u540e\u518d\u89e3\u6790
        import re as _re
        cleaned = _re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
        try:
            return json.loads(cleaned, strict=False)
        except json.JSONDecodeError:
            pass

        # Step 4: \u5c1d\u8bd5\u63d0\u53d6\u65b0 Skill \u683c\u5f0f\u5b57\u6bb5
        ac = _re.search(r'"activation_condition"\s*:\s*"(.*?)"(?=\s*[,}])', raw, _re.DOTALL)
        ep = _re.search(r'"execution_procedure"\s*:\s*"(.*?)"(?=\s*[,}])', raw, _re.DOTALL)
        tc = _re.search(r'"termination_condition"\s*:\s*"(.*?)"(?=\s*[,}])', raw, _re.DOTALL)
        if ac and ep:
            return {
                "activation_condition": ac.group(1).replace("\n", " ").strip(),
                "execution_procedure": ep.group(1).replace("\n", " ").strip(),
                "termination_condition": tc.group(1).replace("\n", " ").strip() if tc else "",
            }

        # Step 5: \u5c1d\u8bd5\u63d0\u53d6\u65e7\u683c\u5f0f\u5b57\u6bb5
        md = _re.search(
            r'"memory_description"\s*:\s*"(.*?)"(?=\s*,\s*"content_body"|\s*})',
            raw, _re.DOTALL
        )
        cb = _re.search(r'"content_body"\s*:\s*"(.*?)"(?=\s*})', raw, _re.DOTALL)
        if md and cb:
            return {
                "memory_description": md.group(1).replace("\n", " ").strip(),
                "content_body": cb.group(1).replace("\n", " ").strip(),
            }

        # Step 6: \u6700\u7ec8\u964d\u7ea7
        logger.warning(f"JSON parse fallback. Raw[:200]={raw[:200]}")
        summary = raw[:120].replace('"', "'").replace("\n", " ").strip()
        return {
            "activation_condition": summary,
            "execution_procedure": raw.replace('"', "'").replace("\n", " ").strip(),
            "termination_condition": "",
        }
    def _parse_action(self, llm_output: str) -> str:
        """解析 LLM 输出中的动作"""
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action:\s?(.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
        put_action = re.findall(r"put\s+(.*)\s+[io]n\s+(.*)", action)
        if put_action:
            action = f"put {put_action[0][0]} in/on {put_action[0][1]}"
        try:
            assert action is not None
            assert 'task complete' not in action.lower()
            return action
        except:
            action = re.findall(pattern, llm_output)[1]
            put_action = re.findall(r"put\s+(.*)\s+[io]n\s+(.*)", action)
            if put_action:
                action = f"put {put_action[0][0]} in/on {put_action[0][1]}"
            assert action is not None
            return action

    # --- 辅助方法: 持久化与统计 ---
    def load_memory(self, filepath: str):
        self.dual_memory.load_tree(filepath)
        logger.info(f"📥 Memory loaded from {filepath}")

    def save_memory(self, filepath: str):
        self.dual_memory.save_tree(filepath)
        logger.info(f"💾 Memory saved to {filepath}")

    def get_memory_stats(self) -> Dict[str, Any]:
        return self.dual_memory.stats
