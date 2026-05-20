"""
Dual PR-Tree Memory Manager
双树管理器：将经验分别组织为「任务树」和「环境树」

TaskTree: 以任务目标为索引，存储程序性 Skill（怎么做）
EnvTree:  以环境描述为索引，存储声明性知识（环境布局、物品位置、交互规则）
"""

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

from config import (
    MAX_DEPTH,
    JSON_INDENT, JSON_ENSURE_ASCII,
    TASK_TREE_BASE_THRESHOLD, TASK_TREE_DEPTH_STEP, TASK_TREE_MAX_THRESHOLD,
    TASK_TREE_EMBED_WITH_ACTIVATION,
    ENV_TREE_BASE_THRESHOLD, ENV_TREE_DEPTH_STEP, ENV_TREE_MAX_THRESHOLD,
    CONSOLIDATION_THRESHOLD, FAILURE_NODE_PENALTY,
)
from .memory_node import MemoryNode, NodeType, ResultStatus
from .consolidation import SkillCompiler, KnowledgeCompiler
from common.retriever import VectorRetriever

logger = logging.getLogger(__name__)


class PRTreeMemory:
    """单棵 PR-Tree — DualTreeMemory 的内部组件"""

    def __init__(
        self,
        retriever: VectorRetriever,
        tree_name: str = "default",
        base_threshold: float = 0.88,
        depth_step: float = 0.02,
        max_threshold: float = 0.99,
        embed_with_activation: bool = False,
    ):
        self.retriever = retriever
        self.tree_name = tree_name
        self.base_threshold = base_threshold
        self.depth_step = depth_step
        self.max_threshold = max_threshold
        self.embed_with_activation = embed_with_activation

        self.root = MemoryNode(
            node_type=NodeType.ROOT,
            result_status=ResultStatus.SUCCESS,
            embedding=np.zeros(self.retriever.embedding_dim),
            scenario_description=f"GLOBAL_ROOT_PLACEHOLDER_{tree_name.upper()}",
        )
        self.node_index: Dict[str, MemoryNode] = {self.root.node_id: self.root}
        self.stats = {"total_nodes": 1, "max_depth": 0}

    def get_last_episode_idx(self) -> int:
        max_idx = -1
        for node in self.node_index.values():
            if "GLOBAL_ROOT_PLACEHOLDER" in node.payload.get("scenario_description", ""):
                continue
            max_idx = max(max_idx, node.meta.get("episode_idx", -1))
        return max_idx

    def _get_dynamic_threshold(self, depth: int) -> float:
        return min(self.base_threshold + depth * self.depth_step, self.max_threshold)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def retrieve_context_path(self, query_text: str) -> List[MemoryNode]:
        query_emb = self.retriever.encode(query_text)
        current = self.root
        path = [current]

        while len(path) - 1 < MAX_DEPTH and current.children:
            threshold = self._get_dynamic_threshold(len(path) - 1)
            match = self.retriever.retrieve_best_match(
                query_emb, current.children,
                threshold=threshold,
                failure_penalty=FAILURE_NODE_PENALTY,
            )
            if match:
                current = match[0]
                path.append(current)
            else:
                break

        return path

    def retrieve_context_path_flat(self, query_text: str) -> List[MemoryNode]:
        """全局平铺检索：跨越层级在整棵树中找最相似节点，返回其完整祖先路径。
        避免层级阈值导致深层好节点因父节点相似度不足而被截断。
        """
        query_emb = self.retriever.encode(query_text)

        # BFS 收集全树所有非根节点
        all_nodes: List[MemoryNode] = []
        queue = deque(self.root.children)
        while queue:
            node = queue.popleft()
            all_nodes.append(node)
            queue.extend(node.children)

        if not all_nodes:
            return [self.root]

        # 全局 top-1，使用 base_threshold（不按层级递增），保留 failure_penalty
        match = self.retriever.retrieve_best_match(
            query_emb, all_nodes,
            threshold=self.base_threshold,
            failure_penalty=FAILURE_NODE_PENALTY,
        )

        if match is None:
            return [self.root]

        # 沿 parent 链回溯到根，得到完整路径 [root → ... → best_node]
        return match[0].get_path_to_root()

    # ------------------------------------------------------------------
    # 写入（直接接受 skill dict）
    # ------------------------------------------------------------------

    def _new_node(
        self,
        parent: MemoryNode,
        scenario_description: str,
        skill: Dict[str, Any],
        result_status: Union[str, ResultStatus],
        node_type: NodeType = NodeType.RESIDUAL,
    ) -> MemoryNode:
        if self.embed_with_activation:
            embed_text = skill.get("activation_condition") or scenario_description
        else:
            embed_text = scenario_description
        embedding = self.retriever.encode(embed_text)
        node = MemoryNode(
            node_type=node_type,
            result_status=result_status,
            embedding=embedding,
            scenario_description=scenario_description,
            parent=parent,
        )
        node.payload["activation_condition"] = skill.get("activation_condition", "")
        node.payload["trajectory"] = skill.get("trajectory")
        node.payload["execution_procedure"] = skill.get("execution_procedure", "")
        node.payload["termination_condition"] = skill.get("termination_condition", "")
        self.node_index[node.node_id] = node
        self.stats["total_nodes"] += 1
        self.stats["max_depth"] = max(self.stats["max_depth"], node.depth)
        return node

    def add_root_skill(
        self,
        scenario_description: str,
        skill: Dict[str, Any],
        result_status: Union[str, ResultStatus] = ResultStatus.SUCCESS,
    ) -> MemoryNode:
        node = self._new_node(self.root, scenario_description, skill, result_status, NodeType.ROOT)
        return node

    def add_residual_skill(
        self,
        anchor_node: MemoryNode,
        scenario_description: str,
        skill: Dict[str, Any],
        result_status: Union[str, ResultStatus],
        force_sibling: bool = False,
    ) -> MemoryNode:
        parent = anchor_node
        if (anchor_node.depth >= MAX_DEPTH or force_sibling) and anchor_node.parent:
            parent = anchor_node.parent
        return self._new_node(parent, scenario_description, skill, result_status, NodeType.RESIDUAL)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save_tree(self, filepath: str) -> None:
        nodes_list, queue, visited = [], [self.root], {self.root.node_id}
        while queue:
            curr = queue.pop(0)
            nodes_list.append(curr.to_dict())
            for child in curr.children:
                if child.node_id not in visited:
                    visited.add(child.node_id)
                    queue.append(child)

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(
                {"meta": {"tree_name": self.tree_name, "timestamp": int(time.time()), "stats": self.stats},
                 "nodes": nodes_list},
                f, indent=JSON_INDENT, ensure_ascii=JSON_ENSURE_ASCII,
            )
        logger.info(f"[{self.tree_name}] Saved {len(nodes_list)} nodes → {filepath}")

    def load_tree(self, filepath: str) -> None:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.stats = data.get("meta", {}).get("stats", self.stats)
            temp = {n["node_id"]: MemoryNode.from_dict(n) for n in data.get("nodes", [])}
            root_found = None
            for n_data in data.get("nodes", []):
                node = temp[n_data["node_id"]]
                pid = n_data.get("parent_id")
                if pid and pid in temp:
                    temp[pid].add_child(node)
                if pid is None:
                    root_found = node
            if root_found:
                self.root = root_found
                self.node_index = temp
                logger.info(f"[{self.tree_name}] Loaded {len(temp)} nodes from {filepath}")
            else:
                logger.warning(f"[{self.tree_name}] No root found in {filepath}")
        except FileNotFoundError:
            logger.warning(f"[{self.tree_name}] File not found: {filepath}")
        except Exception as e:
            logger.error(f"[{self.tree_name}] Load failed: {e}")


class DualTreeMemory:
    """
    双树记忆管理器（对外统一接口）

    task_tree: 任务目标索引 → 程序性 Skill（怎么做）
    env_tree:  环境描述索引 → 声明性知识（在哪找/怎么操作）
    """

    def __init__(self, retriever: VectorRetriever):
        self.retriever = retriever

        self.task_tree = PRTreeMemory(
            retriever, tree_name="task",
            base_threshold=TASK_TREE_BASE_THRESHOLD,
            depth_step=TASK_TREE_DEPTH_STEP,
            max_threshold=TASK_TREE_MAX_THRESHOLD,
            embed_with_activation=TASK_TREE_EMBED_WITH_ACTIVATION,
        )
        self.env_tree = PRTreeMemory(
            retriever, tree_name="env",
            base_threshold=ENV_TREE_BASE_THRESHOLD,
            depth_step=ENV_TREE_DEPTH_STEP,
            max_threshold=ENV_TREE_MAX_THRESHOLD,
            embed_with_activation=False,
        )
        self.stats = self._compute_stats()
        self._skill_compiler = SkillCompiler()
        self._knowledge_compiler = KnowledgeCompiler()

    def _compute_stats(self) -> Dict[str, int]:
        return {
            "task_tree_nodes": self.task_tree.stats["total_nodes"],
            "env_tree_nodes": self.env_tree.stats["total_nodes"],
            "total_nodes": self.task_tree.stats["total_nodes"] + self.env_tree.stats["total_nodes"],
            "max_depth": max(self.task_tree.stats["max_depth"], self.env_tree.stats["max_depth"]),
        }

    def _sync_stats(self) -> None:
        self.stats = self._compute_stats()

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def retrieve_task_path(self, task_description: str) -> List[MemoryNode]:
        return self.task_tree.retrieve_context_path(task_description)

    def retrieve_env_path(self, env_description: str) -> List[MemoryNode]:
        return self.env_tree.retrieve_context_path(env_description)

    def retrieve_dual_paths(self, task_description: str, env_description: str) -> Dict[str, List[MemoryNode]]:
        return {
            "task_path": self.retrieve_task_path(task_description),
            "env_path": self.retrieve_env_path(env_description),
        }

    def retrieve_task_path_flat(self, task_description: str) -> List[MemoryNode]:
        return self.task_tree.retrieve_context_path_flat(task_description)

    def retrieve_env_path_flat(self, env_description: str) -> List[MemoryNode]:
        return self.env_tree.retrieve_context_path_flat(env_description)

    def retrieve_dual_paths_flat(self, task_description: str, env_description: str) -> Dict[str, List[MemoryNode]]:
        return {
            "task_path": self.retrieve_task_path_flat(task_description),
            "env_path": self.retrieve_env_path_flat(env_description),
        }

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def add_task_root_skill(self, scenario_description: str, skill: Dict, result_status=ResultStatus.SUCCESS) -> MemoryNode:
        node = self.task_tree.add_root_skill(scenario_description, skill, result_status)
        self._sync_stats()
        return node

    def add_task_residual_skill(self, anchor_node: MemoryNode, scenario_description: str, skill: Dict, result_status, force_sibling=False) -> MemoryNode:
        node = self.task_tree.add_residual_skill(anchor_node, scenario_description, skill, result_status, force_sibling)
        self._sync_stats()
        return node

    def add_env_root_skill(self, scenario_description: str, skill: Dict, result_status=ResultStatus.SUCCESS) -> MemoryNode:
        node = self.env_tree.add_root_skill(scenario_description, skill, result_status)
        self._sync_stats()
        return node

    def add_env_residual_skill(self, anchor_node: MemoryNode, scenario_description: str, skill: Dict, result_status, force_sibling=False) -> MemoryNode:
        node = self.env_tree.add_residual_skill(anchor_node, scenario_description, skill, result_status, force_sibling)
        self._sync_stats()
        return node

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save_trees(self, task_filepath: str, env_filepath: str) -> None:
        self.task_tree.save_tree(task_filepath)
        self.env_tree.save_tree(env_filepath)

    def load_trees(self, task_filepath: str, env_filepath: str) -> None:
        self.task_tree.load_tree(task_filepath)
        self.env_tree.load_tree(env_filepath)
        self._sync_stats()

    def save_tree(self, filepath: str) -> None:
        """兼容接口：filepath 去掉 .json 后缀作为前缀"""
        base = filepath.replace(".json", "")
        self.save_trees(f"{base}_task.json", f"{base}_env.json")

    def load_tree(self, filepath: str) -> None:
        """兼容接口"""
        base = filepath.replace(".json", "")
        self.load_trees(f"{base}_task.json", f"{base}_env.json")

    def get_last_committed_episode(self) -> int:
        return max(
            self.task_tree.get_last_episode_idx(),
            self.env_tree.get_last_episode_idx(),
        )

    # ------------------------------------------------------------------
    # 记忆固化
    # ------------------------------------------------------------------

    def trigger_consolidation_check(self, node: MemoryNode, llm_client=None, tree_type: str = "task") -> None:
        if node.meta.get("is_consolidated", False):
            return
        if node.meta.get("success_count", 0) < CONSOLIDATION_THRESHOLD:
            return

        logger.info(f"[Consolidation/{tree_type}] Node {node.node_id[:8]} hit success_count={node.meta['success_count']} — compiling...")

        path = node.get_path_to_root()
        chain_texts = []
        for n in path:
            if "GLOBAL_ROOT_PLACEHOLDER" in n.payload.get("scenario_description", ""):
                continue
            chain_texts.append(
                f"[{n.meta.get('node_type', 'RESIDUAL')}]\n"
                f"Activation: {n.payload.get('activation_condition', '')}\n"
                f"Execution: {n.payload.get('execution_procedure', '')}\n"
                f"Termination: {n.payload.get('termination_condition', '')}"
            )

        if not chain_texts:
            return

        if tree_type == "env":
            compiled = self._knowledge_compiler.compile_to_knowledge(chain_texts, llm_client)
            if not compiled:
                return
            new_root = self.add_env_root_skill(
                scenario_description=node.payload.get("scenario_description", "consolidated"),
                skill=compiled,
                result_status=ResultStatus.SUCCESS,
            )
        else:
            compiled = self._skill_compiler.compile_to_skill(chain_texts, llm_client)
            if not compiled:
                return
            new_root = self.add_task_root_skill(
                scenario_description=node.payload.get("scenario_description", "consolidated"),
                skill=compiled,
                result_status=ResultStatus.SUCCESS,
            )

        new_root.meta["is_consolidated_root"] = True
        node.meta["is_consolidated"] = True
        logger.info(f"[Consolidation/{tree_type}] ✅ New ROOT {new_root.node_id[:8]} written from chain ending at {node.node_id[:8]}")
