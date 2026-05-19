"""
Reflective Agent with Dual PR-Tree Architecture for ScienceWorld (v1.0)
基于双树 (任务树 + 环境树) 的反思型 Agent — ScienceWorld 版本

ScienceWorld 与 ALFWorld 的主要区别:
1. 环境交互: 使用 scienceworld.ScienceWorldEnv，调用 env.step(action_str)
2. 成功判定: reward==1.0 才算完全成功；done=True 时任务结束但不代表成功（可能仅部分步骤正确）
3. 任务描述: 从 env.reset() 的 info['taskDesc'] 获取任务描述
4. env_description: 使用初始观察作为环境描述
5. 错误处理: "No known action matches" 表示无效动作
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from memory.prtree.dual_tree_manager import DualTreeMemory
from memory.prtree.dual_memory_reader import DualMemoryReader
from memory.prtree.dual_memory_writer import DualMemoryWriter
from memory.prtree.memory_node import MemoryNode, ResultStatus
from common.retriever import VectorRetriever
from common.llm_client import LLMClient

from dual_prompt import (
    scienceworld_instruction,
    TaskTree_Prompt_Map, EnvTree_Prompt_Map,
    PROMPT_WITH_ICL_TEMPLATE, PROMPT_WITH_ICL_TEMPLATE_DUAL_MEMORY,
    MEMORY_HEADERS,
    get_task_prompt_key, get_env_prompt_key
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DualTreeSciWorldAgent:
    """
    基于双树 PR-Tree 的反思型 Agent — ScienceWorld 版本

    Workflow:
    1. Reset: 获取 taskDesc 和初始观察，分别作为 task_description 和 env_description
    2. Retrieve: 分别从任务树和环境树检索相关记忆
    3. Act: 基于融合记忆与环境交互
    4. Reflect: 分别生成任务反思和环境反思
    5. Evolve: 分别向两棵树写入经验
    """

    def __init__(
        self,
        agent_name: str,
        llm_client: LLMClient,
        icl_num: int = 1,
        icl_data_path: str = "data/sciworld_icl.json",
        max_steps_path: str = "data/sciworld/max_steps.json",
        taskname2id_path: str = "data/sciworld/taskname2id.json",
    ):
        self.agent_name = agent_name
        self.llm_client = llm_client
        self.icl_num = icl_num

        # 初始化双树记忆系统
        self.retriever = VectorRetriever()
        self.dual_memory = DualTreeMemory(self.retriever)
        self.reader = DualMemoryReader(self.dual_memory)
        self.writer = DualMemoryWriter(self.dual_memory)

        # 加载 ICL 数据
        self.icl_data = self._load_icl_data(icl_data_path)

        # 加载每个任务类型的最大步数（不同任务步数差异很大，10~120）
        self.max_steps_dict = self._load_max_steps(max_steps_path)

        # 加载任务名称到 ID 的映射
        self.taskname2id = self._load_taskname2id(taskname2id_path)

        # 连续无效动作上限
        self.max_error_steps = 10

        logger.info(f"✅ {agent_name} initialized with Dual PR-Tree Architecture (ScienceWorld)")

    def _load_max_steps(self, filepath: str) -> Dict[str, int]:
        """加载每个任务类型对应的最大步数（来自 max_steps.json）"""
        try:
            with open(filepath, "r") as f:
                d = json.load(f)
            logger.info(f"Loaded max_steps for {len(d)} task types from {filepath}")
            return d
        except Exception as e:
            logger.warning(f"Failed to load max_steps from {filepath}: {e}. Using default=50.")
            return {}

    def _load_taskname2id(self, filepath: str) -> Dict[str, int]:
        """加载任务名称到 ID 的映射（来自 taskname2id.json）"""
        try:
            with open(filepath, "r") as f:
                d = json.load(f)
            logger.info(f"Loaded taskname2id for {len(d)} task types from {filepath}")
            return d
        except Exception as e:
            logger.warning(f"Failed to load taskname2id from {filepath}: {e}")
            return {}

    def _load_icl_data(self, filepath: str) -> List:
        """加载 ICL 示例数据（ScienceWorld 格式：list of conversation list）"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load ICL data from {filepath}: {e}")
            return []

    def build_icl_examples_text(self) -> str:
        """将 ICL 示例转换为纯文本格式"""
        if not self.icl_data or self.icl_num == 0:
            return ""
        examples = self.icl_data[: self.icl_num]
        icl_text = ""
        # for i, dialogue in enumerate(examples):
        #     if self.icl_num > 1:
        #         icl_text += f"Example task {i + 1}:\n"
        #     for j, msg in enumerate(dialogue):
        #         icl_text += msg["content"] + "\n"
        #     icl_text += "\n"
        for i, msg in enumerate(examples[0]):
            if i == 0:
                icl_text += f"{msg['content']}\n"
            elif i % 2 == 0:
                icl_text += f"{msg['content']}\n\n"
            else:
                icl_text += f"{msg['content']}\n"
        return icl_text

    def run_episode(
        self,
        env: Any,
        task_name: str,
        variation_idx: int,
        no_memory: bool = False,
        no_prtree_update: bool = False,
        external_memory_str: Optional[str] = None,
        memory_mode: str = "both",
        memory_type: str = "prtree",
        episode_idx: int = -1,
    ) -> Dict[str, Any]:
        """
        运行一个完整 Episode

        Args:
            env: ScienceWorldEnv 实例（已调用过 env.load）
            task_name: 子任务名称（如 "task-1-boil"）
            variation_idx: 变体索引
            no_memory: 若为 True，跳过所有 PRTree 记忆操作（检索/反思/写入），作为 baseline 运行
            external_memory_str: 若不为 None，直接使用该字符串作为 memory_context 注入 prompt，
                                  跳过 PRTree 内部检索（但仍执行反思与写入）。
                                  可用于接入任意外部 Memory 系统（flat list、RAG 等）。
                                  优先级：no_memory > external_memory_str > PRTree 自动检索

        Returns:
            包含 success/reward/steps 等信息的结果字典
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"🔬 Episode: {task_name} (var={variation_idx})")

        # ================================================================
        # Phase 0: Reset 环境，获取任务描述和初始观察
        # ================================================================
        env.load(task_name, variation_idx, simplificationStr="easy", generateGoldPath=False)
        obs, info = env.reset()
        task_description = info.get("taskDesc", "").strip()
        env_description = obs.strip()

        logger.info(f"🎯 Task: {task_description}")
        logger.info(f"🌍 Init Obs: {env_description[:120]}...")

        # ================================================================
        # Phase 1: Dual Retrieval (DFS 检索)
        # ================================================================
        if no_memory:
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
            dual_paths = self.reader.get_dual_paths(task_description, env_description)
            task_path = dual_paths["task_path"]
            env_path = dual_paths["env_path"]

            # 根据 memory_mode 决定注入 Prompt 的记忆内容
            if memory_mode == "task_only":
                task_text = self.reader.render_task_memory(task_path)
                memory_context_str = (
                    "# Task Strategy Memory\n"
                    "The following are experiences from similar tasks that may help guide your workflow and strategy:\n\n"
                    f"{task_text}"
                ) if task_text else None
                task_memory_used = not self.reader._is_empty_path(task_path)
                env_memory_used  = False
            elif memory_mode == "env_only":
                env_text = self.reader.render_env_memory(env_path)
                memory_context_str = (
                    "# Environment Knowledge Memory\n"
                    "The following are experiences from similar environments:\n\n"
                    f"{env_text}"
                ) if env_text else None
                task_memory_used = False
                env_memory_used  = not self.reader._is_empty_path(env_path)
            else:
                # "both"：双树融合（默认行为）
                memory_context_str = self.reader.get_dual_narrative_context(
                    task_description, env_description, task_path=task_path, env_path=env_path
                )
                task_memory_used = not self.reader._is_empty_path(task_path)
                env_memory_used  = not self.reader._is_empty_path(env_path)

            memory_used = task_memory_used or env_memory_used

            if task_memory_used:
                logger.info(f"💡 Task Tree Hit. Depth: {len(task_path)}.")
            else:
                logger.info("🆕 Task Tree: No match (Zero-shot).")
            if env_memory_used:
                logger.info(f"🌍 Env Tree Hit. Depth: {len(env_path)}.")
            else:
                logger.info("🆕 Env Tree: No match (Zero-shot).")

        # ================================================================
        # Phase 2: Execution
        # ================================================================
        messages = self._build_prompt(task_description, env_description, memory_context_str, memory_type)

        observation = task_description  # 第一轮 observation 即任务描述
        success = False
        final_reward = 0.0
        steps = 0
        error_steps = 0
        max_steps = self.max_steps_dict.get(task_name, 50)
        max_error_steps = self.max_error_steps
        trajectory = [f"Observation: {observation}"]

        logger.info(f"⏱ Max steps for '{task_name}': {max_steps}")

        for step in range(max_steps):
            if step > 0:
                messages.append({"role": "user", "content": f"Observation: {observation}"})
                trajectory.append(f"Observation: {observation}")

            # LLM 生成动作
            try:
                llm_response = self.llm_client.chat(messages)
                action = self._parse_action(llm_response)
            except Exception as e:
                logger.error(f"LLM/Parse Error at step {step}: {e}")
                llm_response = "Action: look around"
                action = "look around"

            logger.info(f"Step {step}: {llm_response[:150]}")
            trajectory.append(llm_response)
            messages.append({"role": "assistant", "content": llm_response})

            # 执行动作
            try:
                obs_raw, _, done, info = env.step(action)
                observation = obs_raw.strip()
                reward = info.get('raw_score', 0.0)

                if "No known action matches that input" in observation:
                    error_steps += 1
                    if error_steps >= max_error_steps:
                        logger.warning("Too many invalid actions, terminating episode.")
                        break
                else:
                    error_steps = 0

                if reward is not None and float(reward) > final_reward:
                    final_reward = float(reward)

            except Exception as e:
                logger.error(f"Env Step Error: {e}")
                observation = "Invalid action!"
                done = False

            steps += 1
            # done=True 时任务必须结束（无论成功与否）
            # ScienceWorld 中 success 由最终 reward==1.0 决定，而非 done
            if done:
                logger.info(f"🏁 Episode ended (done=True) at step {steps}, score={final_reward:.3f}")
                break

        # reward==1.0 才算完全成功，否则即使 done=True 也视为失败（部分步骤正确）
        success = (final_reward >= 1.0)
        if success:
            logger.info(f"✅ Success (score={final_reward:.3f}) in {steps} steps")
        else:
            logger.info(f"❌ Failed (score={final_reward:.3f}) after {steps} steps")

        # ================================================================
        # Phase 3: Dual Reflection
        # Phase 4: Dual Evolution
        # ================================================================

        # 构建 task_id（与 MPO-main 保持一致）
        task_id_num = self.taskname2id.get(task_name, -1)
        task_id = f"{task_id_num}_{variation_idx}"

        if no_memory or no_prtree_update:
            if no_prtree_update:
                logger.info("🔌 External memory mode: skipping PRTree reflection & update.")
            else:
                logger.info("🚫 Memory disabled (no-memory baseline): skipping reflection & tree update.")
            messages.append({
                "success": success,
                "reward": final_reward,
                "steps": steps,
                "task_name": task_name,
                "task_id": task_id,
                "variation_idx": variation_idx,
                "memory_used": memory_used,
                "task_memory_used": task_memory_used,
                "env_memory_used": env_memory_used,
                "task_retrieval_length": 0,
                "env_retrieval_length": 0,
                "task_reflection": {},
                "env_reflection": {},
                "trajectory": trajectory,
                "task_node_id": None,
                "env_node_id": None,
            })
        else:
            task_reflection, env_reflection = self._generate_dual_reflection(
                task_description=task_description,
                env_description=env_description,
                success=success,
                reward=final_reward,
                steps=steps,
                trajectory=trajectory,
                task_path=task_path,
                env_path=env_path,
            )

            task_status = ResultStatus.SUCCESS if success else ResultStatus.FAILURE
            task_skip = task_reflection.get("skip", False)
            env_skip = env_reflection.get("skip", False)

            task_node = None
            env_node = None

            if not task_skip:
                task_node = self.writer.write_task_experience(
                    scenario_description=task_description,
                    skill=task_reflection,
                    result_status=task_status,
                    retrieved_path=task_path,
                    episode_idx=episode_idx,
                )
            else:
                logger.info("⏭ Task: existing skill covers this episode, skipping node creation.")

            if not env_skip:
                env_node = self.writer.write_env_experience(
                    scenario_description=env_description,
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

            messages.append({
                "success": success,
                "reward": final_reward,
                "steps": steps,
                "task_name": task_name,
                "task_id": task_id,
                "variation_idx": variation_idx,
                "memory_used": memory_used,
                "task_memory_used": task_memory_used,
                "env_memory_used": env_memory_used,
                "task_retrieval_length": len(task_path) - 1 if task_memory_used else 0,
                "env_retrieval_length": len(env_path) - 1 if env_memory_used else 0,
                "task_reflection": task_reflection,
                "env_reflection": env_reflection,
                "trajectory": trajectory,
                "task_node_id": task_node.node_id if task_node else None,
                "env_node_id": env_node.node_id if env_node else None,
                "task_skip": task_skip,
                "env_skip": env_skip,
            })

        return messages

    def _build_prompt(
        self,
        task_description: str,
        env_description: str,
        memory_context_str: Optional[str],
        memory_type: str = "prtree",
    ) -> List[Dict[str, str]]:
        """构建初始 Prompt（支持双树记忆融合）"""
        icl_text = self.build_icl_examples_text()

        if memory_context_str:
            memory_header = MEMORY_HEADERS.get(memory_type, MEMORY_HEADERS["prtree"])
            full_content = PROMPT_WITH_ICL_TEMPLATE_DUAL_MEMORY.format(
                instruction=scienceworld_instruction,
                examples=icl_text,
                memory_header=memory_header,
                memory_context=memory_context_str,
                task=task_description,
            )
        else:
            full_content = PROMPT_WITH_ICL_TEMPLATE.format(
                instruction=scienceworld_instruction,
                examples=icl_text,
                task=task_description,
            )

        return [{"role": "user", "content": full_content}]

    def _generate_dual_reflection(
        self,
        task_description: str,
        env_description: str,
        success: bool,
        reward: float,
        steps: int,
        trajectory: List[str],
        task_path: List[MemoryNode],
        env_path: List[MemoryNode],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """生成双树结构化反思（reward 反映任务完成程度）"""
        traj_str = "\n".join(trajectory)

        # ----- Task Tree Reflection -----
        task_is_root = self.reader._is_empty_path(task_path)
        task_compare_memory = (
            "" if task_is_root
            else self.reader.render_task_path_for_reflection(task_path)
        )
        task_prompt_key = get_task_prompt_key(task_is_root, success)
        task_template = TaskTree_Prompt_Map[task_prompt_key]

        if task_is_root:
            task_prompt = task_template.format(
                env_description=env_description,
                task_description=task_description,
                reward=reward,
                steps=steps,
                trajectory=traj_str,
            )
        else:
            task_prompt = task_template.format(
                env_description=env_description,
                task_description=task_description,
                retrieved_task_memory=task_compare_memory,
                reward=reward,
                steps=steps,
                trajectory=traj_str,
            )

        task_response = self.llm_client.chat(
            [{"role": "user", "content": task_prompt}], temperature=0.0, max_tokens=8192
        )
        task_reflection = self._parse_json_response(task_response)

        # ----- Env Tree Reflection -----
        env_is_root = self.reader._is_empty_path(env_path)
        env_compare_memory = (
            "" if env_is_root
            else self.reader.render_env_path_for_reflection(env_path)
        )
        env_prompt_key = get_env_prompt_key(env_is_root, success)
        env_template = EnvTree_Prompt_Map[env_prompt_key]

        if env_is_root:
            env_prompt = env_template.format(
                env_description=env_description,
                task_description=task_description,
                reward=reward,
                steps=steps,
                trajectory=traj_str,
            )
        else:
            env_prompt = env_template.format(
                env_description=env_description,
                task_description=task_description,
                retrieved_env_memory=env_compare_memory,
                reward=reward,
                steps=steps,
                trajectory=traj_str,
            )

        env_response = self.llm_client.chat(
            [{"role": "user", "content": env_prompt}], temperature=0.0, max_tokens=8192
        )
        env_reflection = self._parse_json_response(env_response)

        return task_reflection, env_reflection

    @staticmethod
    def _escape_json_strings(text: str) -> str:
        result = []
        in_string = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == '\\' and in_string:
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

        try:
            return json.loads(raw, strict=False)
        except json.JSONDecodeError:
            pass

        try:
            return json.loads(self._escape_json_strings(raw), strict=False)
        except json.JSONDecodeError:
            pass

        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
        try:
            return json.loads(self._escape_json_strings(cleaned), strict=False)
        except json.JSONDecodeError:
            pass

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

    def _parse_action(self, llm_output: str) -> str:
        """解析 LLM 输出中的动作"""
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action:\s?(.*)")
        matches = re.findall(pattern, llm_output)
        if not matches:
            return "look around"

        action = matches[0].strip()
        # LLM 有时先输出 "task complete" 再给真实动作，取后一个
        if 'task complete' in action.lower() and len(matches) > 1:
            action = matches[1].strip()

        return action if action else "look around"

    # --- 持久化与统计 ---
    def load_memory(self, filepath: str):
        self.dual_memory.load_tree(filepath)
        logger.info(f"📥 Memory loaded from {filepath}")

    def save_memory(self, filepath: str):
        self.dual_memory.save_tree(filepath)
        logger.info(f"💾 Memory saved to {filepath}")

    def get_memory_stats(self) -> Dict[str, Any]:
        return self.dual_memory.stats
