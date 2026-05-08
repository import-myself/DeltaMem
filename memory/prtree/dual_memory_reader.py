"""
Dual Memory Reader (v7.0 - Skill Format Only)
双树记忆读取器：仅渲染 Skill 格式字段
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
        if len(path) == 1 and "GLOBAL_ROOT_PLACEHOLDER" in path[0].payload.get("scenario_description", ""):
            return True
        return False

    def _is_placeholder(self, node: MemoryNode) -> bool:
        return "GLOBAL_ROOT_PLACEHOLDER" in node.payload.get("scenario_description", "")

    def _render_node(self, node: MemoryNode, idx: int, label: str) -> str:
        payload = node.payload
        is_success = node.meta.get("result_status") == ResultStatus.SUCCESS
        status_label = "✅ SUCCESS" if is_success else "⚠️ FAILURE (learn what to AVOID)"

        text = f"## {label} {idx} [{status_label}]\n"
        text += f"- **Activation Condition**: {payload['activation_condition']}\n"
        text += f"- **Execution Procedure**:\n{payload['execution_procedure']}\n"
        if payload.get("termination_condition"):
            text += f"- **Termination Condition**: {payload['termination_condition']}\n"
        return text + "\n"

    # =====================================================================
    # 任务树渲染
    # =====================================================================

    def render_task_memory(self, task_path: List[MemoryNode]) -> Optional[str]:
        if self._is_empty_path(task_path):
            return None
        text = ""
        idx = 0
        for node in task_path:
            if self._is_placeholder(node):
                continue
            idx += 1
            text += self._render_node(node, idx, "Task Skill")
        return text if text.strip() else None

    # =====================================================================
    # 环境树渲染
    # =====================================================================

    def render_env_memory(self, env_path: List[MemoryNode]) -> Optional[str]:
        if self._is_empty_path(env_path):
            return None
        text = ""
        idx = 0
        for node in env_path:
            if self._is_placeholder(node):
                continue
            idx += 1
            text += self._render_node(node, idx, "Environment Skill")
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
                "Skill-based experiences from similar task types. "
                "Any element IDs are from past episodes — DO NOT use them.\n\n"
                f"{task_text}"
            )
        if env_text:
            sections.append(
                "# Environment/Website Knowledge Memory\n"
                "Skill-based experiences from the same or related environment/website. "
                "Any element IDs are from past episodes — DO NOT use them.\n\n"
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
        idx = 0
        for node in path:
            if self._is_placeholder(node):
                continue
            idx += 1
            payload = node.payload
            status = "SUCCESS" if node.meta["result_status"] == ResultStatus.SUCCESS else "FAILURE"
            text += f"[Existing Skill {idx}] (Status: {status})\n"
            text += f"  Activation: {payload['activation_condition']}\n"
            text += f"  Execution: {payload['execution_procedure']}\n"
            if payload.get("termination_condition"):
                text += f"  Termination: {payload['termination_condition']}\n"
            text += "\n"
        return text

    def render_task_path_for_reflection(self, task_path: List[MemoryNode]) -> str:
        return self._render_path_for_reflection(task_path)

    def render_env_path_for_reflection(self, env_path: List[MemoryNode]) -> str:
        return self._render_path_for_reflection(env_path)
