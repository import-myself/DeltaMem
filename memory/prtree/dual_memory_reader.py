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

    def _render_task_node(self, node: MemoryNode, delta_idx: int, is_base: bool) -> str:
        payload = node.payload
        is_success = node.meta.get("result_status") == ResultStatus.SUCCESS

        if is_success:
            if is_base:
                text = f"### [Base Skill] [✅ SUCCESS]\n"
                text += f"- **Trigger**: {payload['activation_condition']}\n"
                if payload.get("trajectory"):
                    # 新格式：trajectory（具体轨迹）+ execution_procedure（hint提示）
                    text += f"\n**Reference Trajectory** (read and imitate the reasoning pattern):\n"
                    text += payload["trajectory"].replace("\\n", "\n") + "\n"
                    if payload.get("execution_procedure"):
                        text += f"\n**Key Tip**: {payload['execution_procedure']}\n"
                else:
                    # 旧格式兼容
                    text += f"- **Procedure**:\n{payload['execution_procedure']}\n"
                if payload.get("termination_condition"):
                    text += f"- **Done when**: {payload['termination_condition']}\n"
            else:
                text = f"### [Skill Patch {delta_idx}] [✅ SUCCESS]\n"
                text += f"- **When to apply**: {payload['activation_condition']}\n"
                text += f"- **Revision / Additional notes**:\n{payload['execution_procedure']}\n"
                if payload.get("termination_condition"):
                    text += f"- **Done when**: {payload['termination_condition']}\n"
        else:
            title = f"[Base Failure Record] [⛔ FAILURE]" if is_base else f"[Failure Record {delta_idx}] [⛔ FAILURE]"
            text = f"### {title}\n"
            text += f"> ⛔ DO NOT follow this as a procedure. This records what went WRONG.\n"
            text += f"- **When this failure occurred**: {payload['activation_condition']}\n"
            text += f"- **What failed / What was not tried**:\n{payload['execution_procedure']}\n"

        return text + "\n"

    def _render_env_node(self, node: MemoryNode, delta_idx: int, is_base: bool) -> str:
        payload = node.payload
        title = "Base Environment Knowledge" if is_base else f"Environment Knowledge Update {delta_idx}"
        text = f"### {title}\n"
        text += f"- **Applicable scenario**: {payload['activation_condition']}\n"
        text += f"- **Layout & operation rules**:\n{payload['execution_procedure']}\n"
        if payload.get("termination_condition"):
            text += f"- **Navigation complete when**: {payload['termination_condition']}\n"
        return text + "\n"

    # =====================================================================
    # 任务树渲染
    # =====================================================================

    def render_task_memory(self, task_path: List[MemoryNode]) -> Optional[str]:
        if self._is_empty_path(task_path):
            return None
        nodes = [n for n in task_path if not self._is_placeholder(n)]
        if not nodes:
            return None
        text = ""
        delta_idx = 0
        for i, node in enumerate(nodes):
            is_base = (i == 0)
            if not is_base:
                delta_idx += 1
            text += self._render_task_node(node, delta_idx, is_base)
        return text if text.strip() else None

    # =====================================================================
    # 环境树渲染
    # =====================================================================

    def render_env_memory(self, env_path: List[MemoryNode]) -> Optional[str]:
        if self._is_empty_path(env_path):
            return None
        nodes = [n for n in env_path if not self._is_placeholder(n)]
        if not nodes:
            return None
        text = ""
        delta_idx = 0
        for i, node in enumerate(nodes):
            is_base = (i == 0)
            if not is_base:
                delta_idx += 1
            text += self._render_env_node(node, delta_idx, is_base)
        return text if text.strip() else None

    # =====================================================================
    # 融合接口
    # =====================================================================

    def get_dual_narrative_context(
        self, task_description: str, env_description: str,
        task_path: Optional[List[MemoryNode]] = None,
        env_path: Optional[List[MemoryNode]] = None,
    ) -> Optional[str]:
        if task_path is None:
            task_path = self.dual_memory.retrieve_task_path(task_description)
        if env_path is None:
            env_path = self.dual_memory.retrieve_env_path(env_description)

        task_text = self.render_task_memory(task_path)
        env_text = self.render_env_memory(env_path)

        if task_text is None and env_text is None:
            logger.info("No relevant memory found in either tree.")
            return None

        sections = []
        if task_text:
            sections.append(
                "# Task Skill Memory (procedural)\n"
                "Start with the **Base Skill** as your foundation procedure. "
                "If any **Skill Delta** below has a trigger condition that matches your current situation, "
                "apply it on top of the base — otherwise execute the base directly. "
                "Specific values (item names, locations, quantities) are from past episodes — adapt them to the current environment.\n\n"
                f"{task_text}"
            )
        if env_text:
            sections.append(
                "# Environment Knowledge Memory (declarative)\n"
                "This is NOT a procedure to execute — it is background knowledge about the environment. "
                "Use it to decide WHERE to search for objects and HOW to operate receptacles/appliances. "
                "Start with the **Base Environment Knowledge**, then apply any **Environment Knowledge Updates** "
                "whose scenario matches your current environment. "
                "Specific values (item names, locations, states) are from past episodes — verify with observation if in doubt.\n\n"
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

    def _render_path_for_reflection(self, path: List[MemoryNode], is_task_tree: bool = True) -> str:
        if self._is_empty_path(path):
            return ""
        text = ""
        idx = 0
        for node in path:
            if self._is_placeholder(node):
                continue
            idx += 1
            payload = node.payload
            is_success = node.meta["result_status"] == ResultStatus.SUCCESS
            if is_task_tree:
                if is_success:
                    label = "Base Skill" if idx == 1 else f"Skill Patch {idx - 1}"
                    text += f"[{label}] (✅ SUCCESS)\n"
                    text += f"  Activation: {payload['activation_condition']}\n"
                    if payload.get("trajectory"):
                        traj_preview = payload["trajectory"].replace("\\n", "\n")
                        text += f"  Reference Trajectory:\n{traj_preview[:800]}\n"
                        if len(payload["trajectory"]) > 800:
                            text += "  ...[truncated]\n"
                    text += f"  Key Tip: {payload['execution_procedure']}\n"
                    if payload.get("termination_condition"):
                        text += f"  Termination: {payload['termination_condition']}\n"
                else:
                    label = "Base Failure Record" if idx == 1 else f"Failure Record {idx - 1}"
                    text += f"[{label}] (⛔ FAILURE — what went wrong, not a solution)\n"
                    text += f"  When this failure occurred: {payload['activation_condition']}\n"
                    text += f"  What failed / What was not tried: {payload['execution_procedure']}\n"
            else:
                label = "Base Environment Knowledge" if idx == 1 else f"Environment Knowledge Update {idx - 1}"
                text += f"[{label}]\n"
                text += f"  Applicable scenario: {payload['activation_condition']}\n"
                text += f"  Layout & operation rules: {payload['execution_procedure']}\n"
                if payload.get("termination_condition"):
                    text += f"  Navigation complete when: {payload['termination_condition']}\n"
            text += "\n"
        return text

    def render_task_path_for_reflection(self, task_path: List[MemoryNode]) -> str:
        return self._render_path_for_reflection(task_path, is_task_tree=True)

    def render_env_path_for_reflection(self, env_path: List[MemoryNode]) -> str:
        return self._render_path_for_reflection(env_path, is_task_tree=False)
