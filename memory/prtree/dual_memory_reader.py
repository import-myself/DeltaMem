"""
Dual Memory Reader (v6.0)
双树记忆读取器：分别从任务树和环境树读取经验并融合为统一的 Prompt Context

v6.0 设计:
- 给 Agent 的渲染: 只展示 memory_description + content_body (scenario_description 只做检索索引)
- 给反思 LLM 的渲染: 展示 memory_description + content_body，供残差对比
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

    # =====================================================================
    # 任务树渲染 (给 Agent 看的 Prompt)
    # =====================================================================

    def render_task_memory(self, task_path: List[MemoryNode]) -> Optional[str]:
        """只展示 memory_description + content_body，标注成功/失败状态"""
        if self._is_empty_path(task_path):
            return None

        text = ""
        item_idx = 0
        for node in task_path:
            if "GLOBAL_ROOT_PLACEHOLDER" in node.payload.get("scenario_description", ""):
                continue
            item_idx += 1
            payload = node.payload
            is_success = node.meta.get("result_status") == ResultStatus.SUCCESS
            status_label = "✅ SUCCESS" if is_success else "⚠️ FAILURE (learn what to AVOID)"
            text += f"## Task Experience {item_idx} [{status_label}]\n"
            text += f"- Summary: {payload['memory_description']}\n"
            text += f"- Guidance:\n{payload.get('content_body', '')}\n"
            text += "\n"

        return text if text.strip() else None

    # =====================================================================
    # 环境树渲染 (给 Agent 看的 Prompt)
    # =====================================================================

    def render_env_memory(self, env_path: List[MemoryNode]) -> Optional[str]:
        """只展示 memory_description + content_body，标注成功/失败状态"""
        if self._is_empty_path(env_path):
            return None

        text = ""
        item_idx = 0
        for node in env_path:
            if "GLOBAL_ROOT_PLACEHOLDER" in node.payload.get("scenario_description", ""):
                continue
            item_idx += 1
            payload = node.payload
            is_success = node.meta.get("result_status") == ResultStatus.SUCCESS
            status_label = "✅ SUCCESS" if is_success else "⚠️ FAILURE (learn what to AVOID)"
            text += f"## Environment Experience {item_idx} [{status_label}]\n"
            text += f"- Summary: {payload['memory_description']}\n"
            text += f"- Guidance:\n{payload.get('content_body', '')}\n"
            text += "\n"

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
                "The following are experiences from similar task types that may guide your workflow and element identification strategy.\n"
                "⚠️ Any element IDs mentioned are from past episodes and DO NOT apply here. Use semantic descriptions to locate elements.\n\n"
                f"{task_text}"
            )
        if env_text:
            sections.append(
                "# Website Knowledge Memory\n"
                "The following are experiences from the same or related website that describe UI component layout and interaction rules.\n"
                "⚠️ Any element IDs mentioned are from past episodes and DO NOT apply here. Use component type, position, and label to locate elements.\n\n"
                f"{env_text}"
            )
        return "\n---\n\n".join(sections)

    def get_dual_paths(
        self, task_description: str, env_description: str
    ) -> Dict[str, List[MemoryNode]]:
        return self.dual_memory.retrieve_dual_paths(task_description, env_description)

    # =====================================================================
    # 整条路径渲染 (用于反思 Prompt，供残差对比)
    #
    # 这里展示每个前序节点的 memory_description + content_body
    # 让反思 LLM 明确知道已有经验说了什么，从而生成差异化的增量
    # =====================================================================

    def render_task_path_for_reflection(self, task_path: List[MemoryNode]) -> str:
        if self._is_empty_path(task_path):
            return ""

        text = ""
        item_idx = 0
        for node in task_path:
            if "GLOBAL_ROOT_PLACEHOLDER" in node.payload.get("scenario_description", ""):
                continue
            item_idx += 1
            payload = node.payload
            status = "SUCCESS" if node.meta["result_status"] == ResultStatus.SUCCESS else "FAILURE"

            text += f"[Existing Memory {item_idx}] (Status: {status})\n"
            text += f"  Description: {payload['memory_description']}\n"
            text += f"  Content: {payload.get('content_body', '')}\n\n"

        return text

    def render_env_path_for_reflection(self, env_path: List[MemoryNode]) -> str:
        if self._is_empty_path(env_path):
            return ""

        text = ""
        item_idx = 0
        for node in env_path:
            if "GLOBAL_ROOT_PLACEHOLDER" in node.payload.get("scenario_description", ""):
                continue
            item_idx += 1
            payload = node.payload
            status = "SUCCESS" if node.meta["result_status"] == ResultStatus.SUCCESS else "FAILURE"

            text += f"[Existing Memory {item_idx}] (Status: {status})\n"
            text += f"  Description: {payload['memory_description']}\n"
            text += f"  Content: {payload.get('content_body', '')}\n\n"

        return text
