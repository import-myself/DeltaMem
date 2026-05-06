"""
Dual Memory Writer (v4.0)
双树记忆写入器：分别向任务树和环境树写入经验

设计:
- 任务树写入: Workflow Schema / 策略残差 / 任务级别的反思
- 环境树写入: 环境布局认知 / 物品位置经验 / 环境交互教训
- 两棵树独立写入，各自维护自己的 Anchor 路径
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
    """移除 [数字] 格式的 Element ID，这些 ID 是 episode 独有的，跨 episode 无效"""
    if not content:
        return content
    cleaned = _ELEMENT_ID_RE.sub('[<element_id>]', content)
    if cleaned != content:
        logger.warning(
            "content_body 包含 Element ID（已自动移除）。"
            "请检查反思 Prompt 是否有效禁止了 ID 输出。"
        )
    return cleaned


class DualMemoryWriter:
    """
    双树记忆写入器
    
    职责:
    1. 接收 Agent 执行后的原始数据 (拆分为 task 和 env 两组)
    2. 分别确定两棵树的挂载点 (Anchor Node)
    3. 分别调用 DualTreeMemory 完成写入
    """

    def __init__(self, dual_memory: DualTreeMemory):
        self.dual_memory = dual_memory

    def _determine_anchor(self, tree, retrieved_path: Optional[List[MemoryNode]]) -> MemoryNode:
        """解析路径，找到最后一个有效的节点作为 Anchor"""
        if not retrieved_path:
            return tree.root

        last_node = retrieved_path[-1]
        if last_node.node_id in tree.node_index:
            return last_node
        else:
            logger.warning(f"Anchor node {last_node.node_id[:8]} not found in tree. Fallback to Root.")
            return tree.root

    def _is_root_path(self, path: Optional[List[MemoryNode]], tree_name: str) -> bool:
        """判断路径是否只命中了占位根（即 zero-shot）"""
        if not path:
            return True
        if len(path) == 1:
            desc = path[0].payload.get("scenario_description", "")
            if "GLOBAL_ROOT_PLACEHOLDER" in desc:
                return True
        return False

    # =====================================================================
    # 任务树写入
    # =====================================================================

    def write_task_experience(
        self,
        scenario_description: str,
        memory_description: str,
        content_body: str,
        status: Union[str, ResultStatus],
        retrieved_path: Optional[List[MemoryNode]] = None
    ) -> MemoryNode:
        """
        写入任务经验到任务树
        
        Args:
            scenario_description: 任务目标描述 (用于向量索引)
            memory_description: 经验摘要 (一句话)
            content_body: Workflow / 策略内容
            status: SUCCESS / FAILURE
            retrieved_path: 任务树检索到的路径
        """
        tree = self.dual_memory.task_tree
        anchor_node = self._determine_anchor(tree, retrieved_path)

        # Case A: 全新 Root (Zero-shot on task tree)
        if self._is_root_path(retrieved_path, "task"):
            logger.info("[TaskTree] Writing new ROOT SCHEMA (Zero-shot task experience).")
            return self.dual_memory.add_task_root_schema(
                scenario_description=scenario_description,
                memory_description=memory_description,
                content_body=content_body,
                result_status=status
            )

        # Case B: 残差更新
        logger.info(f"[TaskTree] Writing RESIDUAL under anchor: {anchor_node.node_id[:8]}")
        return self.dual_memory.add_task_experience(
            anchor_node=anchor_node,
            scenario_description=scenario_description,
            memory_description=memory_description,
            content_body=content_body,
            result_status=status,
            force_sibling=False
        )

    # =====================================================================
    # 环境树写入
    # =====================================================================

    def write_env_experience(
        self,
        scenario_description: str,
        memory_description: str,
        content_body: str,
        status: Union[str, ResultStatus],
        retrieved_path: Optional[List[MemoryNode]] = None
    ) -> MemoryNode:
        """
        写入环境经验到环境树
        
        Args:
            scenario_description: 环境描述 (用于向量索引)
            memory_description: 经验摘要 (一句话)
            content_body: 环境反思 / 物品位置经验
            status: SUCCESS / FAILURE
            retrieved_path: 环境树检索到的路径
        """
        tree = self.dual_memory.env_tree
        anchor_node = self._determine_anchor(tree, retrieved_path)

        # Case A: 全新 Root
        if self._is_root_path(retrieved_path, "env"):
            logger.info("[EnvTree] Writing new ROOT SCHEMA (Zero-shot env experience).")
            return self.dual_memory.add_env_root_schema(
                scenario_description=scenario_description,
                memory_description=memory_description,
                content_body=content_body,
                result_status=status
            )

        # Case B: 残差更新
        logger.info(f"[EnvTree] Writing RESIDUAL under anchor: {anchor_node.node_id[:8]}")
        return self.dual_memory.add_env_experience(
            anchor_node=anchor_node,
            scenario_description=scenario_description,
            memory_description=memory_description,
            content_body=content_body,
            result_status=status,
            force_sibling=False
        )

    # =====================================================================
    # 统一写入接口 (Agent 调用)
    # =====================================================================

    def write_dual_experience(
        self,
        task_description: str,
        env_description: str,
        task_reflection: Dict[str, str],
        env_reflection: Dict[str, str],
        status: Union[str, ResultStatus],
        task_retrieved_path: Optional[List[MemoryNode]] = None,
        env_retrieved_path: Optional[List[MemoryNode]] = None
    ) -> Dict[str, MemoryNode]:
        """
        一次性写入两棵树

        Args:
            task_description: 任务目标 (e.g. "put some spraybottle on toilet")
            env_description:  环境描述 (e.g. "You are in the middle of a room...")
            task_reflection:  任务反思 {"memory_description": "...", "content_body": "..."}
            env_reflection:   环境反思 {"memory_description": "...", "content_body": "..."}
            status: SUCCESS / FAILURE
            task_retrieved_path: 任务树的检索路径
            env_retrieved_path:  环境树的检索路径

        Returns:
            {"task_node": MemoryNode, "env_node": MemoryNode}
        """
        task_node = self.write_task_experience(
            scenario_description=task_description,
            memory_description=task_reflection["memory_description"],
            content_body=_sanitize_content(task_reflection["content_body"]),
            status=status,
            retrieved_path=task_retrieved_path
        )

        env_node = self.write_env_experience(
            scenario_description=env_description,
            memory_description=env_reflection["memory_description"],
            content_body=_sanitize_content(env_reflection["content_body"]),
            status=status,
            retrieved_path=env_retrieved_path
        )

        return {"task_node": task_node, "env_node": env_node}
