"""
Dual Memory Writer (v5.0 - Skill Format)
双树记忆写入器：支持新 Skill 格式 (activation_condition / execution_procedure / termination_condition)
同时保持对旧格式 (memory_description / content_body) 的向后兼容
"""

import re
import logging
from typing import Optional, Dict, Any, List, Union

from .memory_node import MemoryNode, NodeType, ResultStatus
from .dual_tree_manager import DualTreeMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ELEMENT_ID_RE = re.compile(r'\[(\d+)\]')


def _sanitize_content(content: str) -> str:
    if not content:
        return content
    cleaned = _ELEMENT_ID_RE.sub('[<element_id>]', content)
    if cleaned != content:
        logger.warning("content 包含 Element ID（已自动移除）。")
    return cleaned


def _normalize_reflection(reflection: Dict[str, Any]) -> Dict[str, str]:
    """
    将新 Skill 格式或旧格式的 reflection dict 统一规范化。

    新格式: {"activation_condition": ..., "execution_procedure": ..., "termination_condition": ...}
    旧格式: {"memory_description": ..., "content_body": ...}

    Returns: dict with all 5 keys guaranteed
    """
    has_new = "activation_condition" in reflection and "execution_procedure" in reflection

    if has_new:
        activation = reflection.get("activation_condition", "")
        execution = reflection.get("execution_procedure", "")
        termination = reflection.get("termination_condition", "")
        # 兼容渲染管道：memory_description = activation_condition 摘要
        memory_desc = reflection.get("memory_description") or activation
        content_body = reflection.get("content_body") or execution
    else:
        # 旧格式降级
        memory_desc = reflection.get("memory_description", "")
        content_body = reflection.get("content_body", "")
        activation = memory_desc
        execution = content_body
        termination = reflection.get("termination_condition", "")

    return {
        "memory_description": memory_desc,
        "content_body": content_body,
        "activation_condition": activation,
        "execution_procedure": execution,
        "termination_condition": termination,
    }


class DualMemoryWriter:
    def __init__(self, dual_memory: DualTreeMemory):
        self.dual_memory = dual_memory

    def _determine_anchor(self, tree, retrieved_path: Optional[List[MemoryNode]]) -> MemoryNode:
        if not retrieved_path:
            return tree.root
        last_node = retrieved_path[-1]
        if last_node.node_id in tree.node_index:
            return last_node
        logger.warning(f"Anchor node {last_node.node_id[:8]} not found. Fallback to Root.")
        return tree.root

    def _is_root_path(self, path: Optional[List[MemoryNode]], tree_name: str) -> bool:
        if not path:
            return True
        if len(path) == 1:
            desc = path[0].payload.get("scenario_description", "")
            if "GLOBAL_ROOT_PLACEHOLDER" in desc:
                return True
        return False

    # =====================================================================
    # 内部写入（共用逻辑）
    # =====================================================================

    def _write_to_tree(
        self,
        tree,
        add_root_fn,
        add_experience_fn,
        scenario_description: str,
        reflection: Dict[str, Any],
        status: Union[str, ResultStatus],
        retrieved_path: Optional[List[MemoryNode]],
        tree_label: str,
    ) -> MemoryNode:
        norm = _normalize_reflection(reflection)
        anchor_node = self._determine_anchor(tree, retrieved_path)

        if self._is_root_path(retrieved_path, tree_label):
            logger.info(f"[{tree_label}] Writing new ROOT SCHEMA (Zero-shot).")
            node = add_root_fn(
                scenario_description=scenario_description,
                memory_description=norm["memory_description"],
                content_body=_sanitize_content(norm["content_body"]),
                result_status=status
            )
        else:
            logger.info(f"[{tree_label}] Writing RESIDUAL under anchor: {anchor_node.node_id[:8]}")
            node = add_experience_fn(
                anchor_node=anchor_node,
                scenario_description=scenario_description,
                memory_description=norm["memory_description"],
                content_body=_sanitize_content(norm["content_body"]),
                result_status=status,
                force_sibling=False
            )

        # 写入 Skill 字段
        node.payload["activation_condition"] = norm["activation_condition"]
        node.payload["execution_procedure"] = _sanitize_content(norm["execution_procedure"])
        node.payload["termination_condition"] = norm["termination_condition"]

        # 更新 hit_count on retrieved path nodes
        if retrieved_path:
            for path_node in retrieved_path:
                if "GLOBAL_ROOT_PLACEHOLDER" not in path_node.payload.get("scenario_description", ""):
                    path_node.meta["hit_count"] = path_node.meta.get("hit_count", 0) + 1

        return node

    # =====================================================================
    # 任务树写入
    # =====================================================================

    def write_task_experience(
        self,
        scenario_description: str,
        reflection: Dict[str, Any],
        status: Union[str, ResultStatus],
        retrieved_path: Optional[List[MemoryNode]] = None
    ) -> MemoryNode:
        return self._write_to_tree(
            tree=self.dual_memory.task_tree,
            add_root_fn=self.dual_memory.add_task_root_schema,
            add_experience_fn=self.dual_memory.add_task_experience,
            scenario_description=scenario_description,
            reflection=reflection,
            status=status,
            retrieved_path=retrieved_path,
            tree_label="TaskTree",
        )

    # =====================================================================
    # 环境树写入
    # =====================================================================

    def write_env_experience(
        self,
        scenario_description: str,
        reflection: Dict[str, Any],
        status: Union[str, ResultStatus],
        retrieved_path: Optional[List[MemoryNode]] = None
    ) -> MemoryNode:
        return self._write_to_tree(
            tree=self.dual_memory.env_tree,
            add_root_fn=self.dual_memory.add_env_root_schema,
            add_experience_fn=self.dual_memory.add_env_experience,
            scenario_description=scenario_description,
            reflection=reflection,
            status=status,
            retrieved_path=retrieved_path,
            tree_label="EnvTree",
        )

    # =====================================================================
    # 统一写入接口（向后兼容）
    # =====================================================================

    def write_dual_experience(
        self,
        task_description: str,
        env_description: str,
        task_reflection: Dict[str, Any],
        env_reflection: Dict[str, Any],
        status: Union[str, ResultStatus],
        task_retrieved_path: Optional[List[MemoryNode]] = None,
        env_retrieved_path: Optional[List[MemoryNode]] = None
    ) -> Dict[str, MemoryNode]:
        task_node = self.write_task_experience(
            scenario_description=task_description,
            reflection=task_reflection,
            status=status,
            retrieved_path=task_retrieved_path,
        )
        env_node = self.write_env_experience(
            scenario_description=env_description,
            reflection=env_reflection,
            status=status,
            retrieved_path=env_retrieved_path,
        )
        return {"task_node": task_node, "env_node": env_node}
