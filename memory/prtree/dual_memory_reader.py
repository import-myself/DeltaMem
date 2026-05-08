"""
Dual Memory Reader (v7.0 - Skill Format)
双树记忆读取器：支持 Skill 格式渲染，向后兼容旧格式
"""

import logging
from typing import Optional, List, Dict, Any
from .memory_node import MemoryNode, NodeType, ResultStatus
from .dual_tree_manager import DualTreeMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DualMemoryReader:
    def __init__(self, dual_memory: DualTreeMemory):
        self.dual_memory = dual_memory

    def _is_empty_path(self, path: List[MemoryNode]) -> bool:
        if not path:
            return True
        if len(path) == 1:
            desc = path[0].payload.get("scenario_description", "")
            if "GLOBAL_ROOT_PLACEHOLDER" in desc:
                return True
        return False

    def _render_node_skill(self, node: MemoryNode, item_idx: int, label: str) -> str:
        """渲染单个节点：优先使用 Skill 字段，回退到旧格式"""
        payload = node.payload
        is_success = node.meta.get("result_status") == ResultStatus.SUCCESS
        status_label = "✅ SUCCESS" if is_success else "⚠️ FAILURE (learn what to AVOID)"

        activation = payload.get("activation_condition")
        execution = payload.get("execution_procedure")
        termination = payload.get("termination_condition")

        text = f"## {label} {item_idx} [{status_label}]\n"

        if activation and execution:
            # 新 Skill 格式
            text += f"- **Activation Condition**: {activation}\n"
            text += f"- **Execution Procedure**:\n{execution}\n"
            if termination:
                text += f"- **Termination Condition**: {termination}\n"
        else:
            # 旧格式回退
            text += f"- Summary: {payload.get('memory_description', '')}\n"
            text += f"- Guidance:\n{payload.get('content_body', '')}\n"

        return text + "\n"

    # =====================================================================
    # 任务树渲染
    # =====================================================================

    def render_task_memory(self, task_path: List[MemoryNode]) -> Optional[str]:
        if self._is_empty_path(task_path):
            return None
        text = ""
        item_idx = 0
        for node in task_path:
            if "GLOBAL_ROOT_PLACEHOLDER" in node.payload.get("scenario_description", ""):
                continue
            item_idx += 1
            text += self._render_node_skill(node, item_idx, "Task Skill")
        return text if text.strip() else None

    # =====================================================================
    # 环境树渲染
    # =====================================================================

    def render_env_memory(self, env_path: List[MemoryNode]) -> Optional[str]:
        if self._is_empty_path(env_path):
            return None
        text = ""
        item_idx = 0
        for node in env_path:
            if "GLOBAL_ROOT_PLACEHOLDER" in node.payload.get("scenario_description", ""):
                continue
            item_idx += 1
            text += self._render_node_skill(node, item_idx, "Environment Skill")
        return text if text.strip() else None

    # =====================================================================
    # 融合接口
    # =====================================================================

    def get_dual_narrative_context(
        self, task_description: str, env_description: str
    ) -> Optional[str]:
        task_path = self.dual_memory.retrieve_task_path(task_description)
        env_path = self.dual_memory.retrieve_env_path(env_description)

        task_text = self.render_task_memory(task_path)
        env_text = self.render_env_memory(env_path)

        if task_text is None and env_text is None:
            logger.info("No relevant memory found in either tree.")
            return None

        sections = []
        if task_text:
            sections.append(
                "# Task Strategy Memory\n"
                "The following are skill-based experiences from similar task types.\n"
                "⚠️ Any element IDs mentioned are from past episodes and DO NOT apply here.\n\n"
                f"{task_text}"
            )
        if env_text:
            sections.append(
                "# Website Knowledge Memory\n"
                "The following are skill-based experiences from the same or related website/environment.\n"
                "⚠️ Any element IDs mentioned are from past episodes and DO NOT apply here.\n\n"
                f"{env_text}"
            )
        return "\n---\n\n".join(sections)

    def get_dual_paths(
        self, task_description: str, env_description: str
    ) -> Dict[str, List[MemoryNode]]:
        return self.dual_memory.retrieve_dual_paths(task_description, env_description)

    # =====================================================================
    # 路径渲染（用于反思 Prompt 对比）
    # =====================================================================

    def _render_path_for_reflection(self, path: List[MemoryNode]) -> str:
        if self._is_empty_path(path):
            return ""
        text = ""
        item_idx = 0
        for node in path:
            if "GLOBAL_ROOT_PLACEHOLDER" in node.payload.get("scenario_description", ""):
                continue
            item_idx += 1
            payload = node.payload
            status = "SUCCESS" if node.meta["result_status"] == ResultStatus.SUCCESS else "FAILURE"
            activation = payload.get("activation_condition")
            execution = payload.get("execution_procedure")

            text += f"[Existing Skill {item_idx}] (Status: {status})\n"
            if activation and execution:
                text += f"  Activation: {activation}\n"
                text += f"  Execution: {execution}\n"
                if payload.get("termination_condition"):
                    text += f"  Termination: {payload['termination_condition']}\n"
            else:
                text += f"  Description: {payload.get('memory_description', '')}\n"
                text += f"  Content: {payload.get('content_body', '')}\n"
            text += "\n"
        return text

    def render_task_path_for_reflection(self, task_path: List[MemoryNode]) -> str:
        return self._render_path_for_reflection(task_path)

    def render_env_path_for_reflection(self, env_path: List[MemoryNode]) -> str:
        return self._render_path_for_reflection(env_path)
