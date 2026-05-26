"""
Dual Memory Writer
双树记忆写入器：根据是否有已检索路径决定写 ROOT 还是 RESIDUAL，
并维护 hit_count / episode_idx 元数据。
"""

import logging
from typing import Optional, Dict, Any, List, Union

from .memory_node import MemoryNode, ResultStatus
from .dual_tree_manager import DualTreeMemory

logger = logging.getLogger(__name__)


class DualMemoryWriter:
    def __init__(self, dual_memory: DualTreeMemory):
        self.dual_memory = dual_memory

    def _is_root_path(self, path: Optional[List[MemoryNode]]) -> bool:
        if not path:
            return True
        if len(path) == 1 and "GLOBAL_ROOT_PLACEHOLDER" in path[0].payload.get("scenario_description", ""):
            return True
        return False

    def _valid_anchor(self, tree, path: Optional[List[MemoryNode]]) -> MemoryNode:
        if not path:
            return tree.root
        last = path[-1]
        if last.node_id in tree.node_index:
            return last
        logger.warning(f"Anchor {last.node_id[:8]} not in tree index, falling back to root.")
        return tree.root

    def _write(
        self,
        tree,
        add_root_fn,
        add_residual_fn,
        scenario_description: str,
        skill: Dict[str, Any],
        result_status: Union[str, ResultStatus],
        retrieved_path: Optional[List[MemoryNode]],
        tree_label: str,
        episode_idx: int = -1,
    ) -> MemoryNode:
        if self._is_root_path(retrieved_path):
            logger.info(f"[{tree_label}] Writing ROOT skill (cold start).")
            node = add_root_fn(scenario_description=scenario_description, skill=skill, result_status=result_status)
        else:
            anchor = self._valid_anchor(tree, retrieved_path)
            logger.info(f"[{tree_label}] Writing RESIDUAL under {anchor.node_id[:8]}.")
            node = add_residual_fn(
                anchor_node=anchor,
                scenario_description=scenario_description,
                skill=skill,
                result_status=result_status,
            )

        node.meta["episode_idx"] = episode_idx

        if retrieved_path:
            for n in retrieved_path:
                if "GLOBAL_ROOT_PLACEHOLDER" not in n.payload.get("scenario_description", ""):
                    n.meta["hit_count"] = n.meta.get("hit_count", 0) + 1

        return node

    def write_task_experience(
        self,
        scenario_description: str,
        skill: Dict[str, Any],
        result_status: Union[str, ResultStatus],
        retrieved_path: Optional[List[MemoryNode]] = None,
        episode_idx: int = -1,
    ) -> MemoryNode:
        return self._write(
            tree=self.dual_memory.task_tree,
            add_root_fn=self.dual_memory.add_task_root_skill,
            add_residual_fn=self.dual_memory.add_task_residual_skill,
            scenario_description=scenario_description,
            skill=skill,
            result_status=result_status,
            retrieved_path=retrieved_path,
            tree_label="TaskTree",
            episode_idx=episode_idx,
        )

    def write_env_experience(
        self,
        scenario_description: str,
        skill: Dict[str, Any],
        result_status: Union[str, ResultStatus],
        retrieved_path: Optional[List[MemoryNode]] = None,
        episode_idx: int = -1,
    ) -> MemoryNode:
        return self._write(
            tree=self.dual_memory.env_tree,
            add_root_fn=self.dual_memory.add_env_root_skill,
            add_residual_fn=self.dual_memory.add_env_residual_skill,
            scenario_description=scenario_description,
            skill=skill,
            result_status=result_status,
            retrieved_path=retrieved_path,
            tree_label="EnvTree",
            episode_idx=episode_idx,
        )

    def write_dual_experience(
        self,
        task_description: str,
        env_description: str,
        task_skill: Dict[str, Any],
        env_skill: Dict[str, Any],
        result_status: Union[str, ResultStatus],
        task_retrieved_path: Optional[List[MemoryNode]] = None,
        env_retrieved_path: Optional[List[MemoryNode]] = None,
        episode_idx: int = -1,
        env_result_status: Optional[Union[str, ResultStatus]] = None,
    ) -> Dict[str, MemoryNode]:
        task_node = self.write_task_experience(
            scenario_description=task_description,
            skill=task_skill,
            result_status=result_status,
            retrieved_path=task_retrieved_path,
            episode_idx=episode_idx,
        )
        env_node = self.write_env_experience(
            scenario_description=env_description,
            skill=env_skill,
            result_status=env_result_status if env_result_status is not None else result_status,
            retrieved_path=env_retrieved_path,
            episode_idx=episode_idx,
        )
        return {"task_node": task_node, "env_node": env_node}
