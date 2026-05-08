"""
Dual PR-Tree Memory Manager (v4.0)
双树管理器：将经验分别组织为「任务树」和「环境树」

设计理念:
- ALFWorld 的每个任务包含两个维度的信息：
  1. 任务维度 (Task): "put some spraybottle on toilet" → 偏向 Workflow / 策略
  2. 环境维度 (Environment): "You are in the middle of a room. Looking quickly around you, you see a cabinet 4..." → 偏向环境布局认知 / 物品位置经验
- 原始单树将两者混合，导致检索不精确。
- 双树将两个维度分别组织，检索时分别匹配，最终融合为一个完整的记忆上下文。

架构:
  TaskTree:   以任务目标为索引，存储 Workflow Schema / 策略残差
  EnvTree:    以环境描述为索引，存储 环境反思 / 物品位置经验 / 环境交互教训
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from config import (
    STORAGE_PATH, MAX_DEPTH,
    TREE_STRUCTURE_FILE, JSON_INDENT, JSON_ENSURE_ASCII,
    TASK_TREE_BASE_THRESHOLD, TASK_TREE_DEPTH_STEP, TASK_TREE_MAX_THRESHOLD,
    ENV_TREE_BASE_THRESHOLD, ENV_TREE_DEPTH_STEP, ENV_TREE_MAX_THRESHOLD
)
from .memory_node import MemoryNode, NodeType, ResultStatus
from .skill_patch import SkillCache, SkillCompiler, CONSOLIDATION_THRESHOLD
from common.retriever import VectorRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PRTreeMemory:
    """
    单棵 PR-Tree (与原始 tree_manager 逻辑一致)
    保留为内部组件，供 DualTreeMemory 使用
    """

    def __init__(
        self,
        retriever: VectorRetriever,
        tree_name: str = "default",
        base_threshold: float = 0.88,
        depth_step: float = 0.02,
        max_threshold: float = 0.99
    ):
        self.retriever = retriever
        self.tree_name = tree_name

        # 阈值参数 (每棵树可独立配置)
        self.base_threshold = base_threshold
        self.depth_step = depth_step
        self.max_threshold = max_threshold

        # 初始化虚拟根
        self.root = MemoryNode(
            node_type=NodeType.ROOT,
            result_status=ResultStatus.SUCCESS,
            embedding=np.zeros(self.retriever.embedding_dim),
            scenario_description=f"GLOBAL_ROOT_PLACEHOLDER_{tree_name.upper()}",
            memory_description=f"The absolute root of the {tree_name} tree.",
            content_body=f"Base schema for all {tree_name} entries."
        )

        self.node_index: Dict[str, MemoryNode] = {self.root.node_id: self.root}
        self.stats = {"total_nodes": 1, "max_depth": 0}

    # --- 动态阈值 (使用实例级参数) ---
    def _get_dynamic_threshold(self, current_depth: int) -> float:
        """
        根据当前深度计算相似度阈值
        阈值参数由构造时传入，任务树和环境树各自独立
        
        TaskTree (宽松): base=0.80, step=0.02 → Depth 0: 0.80, 1: 0.82, 2: 0.84 ...
        EnvTree  (严格): base=0.88, step=0.02 → Depth 0: 0.88, 1: 0.90, 2: 0.92 ...
        """
        threshold = self.base_threshold + (current_depth * self.depth_step)
        return min(threshold, self.max_threshold)

    # --- DFS 检索 ---
    def retrieve_context_path(self, query_text: str) -> List[MemoryNode]:
        query_emb = self.retriever.encode(query_text)
        current_node = self.root
        path = [current_node]

        while True:
            current_depth = len(path) - 1
            if current_depth >= MAX_DEPTH:
                break

            children = current_node.children
            if not children:
                break

            threshold = self._get_dynamic_threshold(current_depth)
            best_child_tuple = self.retriever.retrieve_best_match(
                query_emb, children, threshold=threshold
            )

            if best_child_tuple:
                best_child, score = best_child_tuple
                path.append(best_child)
                current_node = best_child
            else:
                break

        return path

    # --- 写入 ---
    def add_experience_node(
        self,
        anchor_node: MemoryNode,
        scenario_description: str,
        memory_description: str,
        content_body: str,
        result_status: Union[str, ResultStatus],
        force_sibling: bool = False
    ) -> MemoryNode:
        parent_node = anchor_node

        if (anchor_node.depth >= MAX_DEPTH or force_sibling) and anchor_node.parent:
            parent_node = anchor_node.parent

        embedding = self.retriever.encode(scenario_description)

        new_node = MemoryNode(
            node_type=NodeType.RESIDUAL,
            result_status=result_status,
            embedding=embedding,
            scenario_description=scenario_description,
            memory_description=memory_description,
            content_body=content_body,
            parent=parent_node
        )

        self.node_index[new_node.node_id] = new_node
        self.stats["total_nodes"] += 1
        self.stats["max_depth"] = max(self.stats["max_depth"], new_node.depth)
        return new_node

    def add_root_schema(
        self,
        scenario_description: str,
        memory_description: str,
        content_body: str,
        result_status: str = ResultStatus.SUCCESS
    ) -> MemoryNode:
        node = self.add_experience_node(
            anchor_node=self.root,
            scenario_description=scenario_description,
            memory_description=memory_description,
            content_body=content_body,
            result_status=result_status
        )
        node.meta["node_type"] = NodeType.ROOT.value
        return node

    # --- 持久化 ---
    def save_tree(self, filepath: str) -> None:
        nodes_list = []
        queue = [self.root]
        visited = {self.root.node_id}

        while queue:
            curr = queue.pop(0)
            nodes_list.append(curr.to_dict())
            for child in curr.children:
                if child.node_id not in visited:
                    visited.add(child.node_id)
                    queue.append(child)

        data = {
            "meta": {
                "tree_name": self.tree_name,
                "timestamp": int(time.time()),
                "stats": self.stats
            },
            "nodes": nodes_list
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=JSON_INDENT, ensure_ascii=JSON_ENSURE_ASCII)
        logger.info(f"[{self.tree_name}] Tree saved to {filepath} ({len(nodes_list)} nodes)")

    def load_tree(self, filepath: str) -> None:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            nodes_data = data.get("nodes", [])
            self.stats = data.get("meta", {}).get("stats", self.stats)

            temp_index = {}
            for n_data in nodes_data:
                node = MemoryNode.from_dict(n_data)
                temp_index[node.node_id] = node

            root_found = None
            for n_data in nodes_data:
                curr_node = temp_index[n_data["node_id"]]
                parent_id = n_data.get("parent_id")

                if parent_id and parent_id in temp_index:
                    parent_node = temp_index[parent_id]
                    parent_node.add_child(curr_node)

                if parent_id is None:
                    root_found = curr_node

            if root_found:
                self.root = root_found
                self.node_index = temp_index
                logger.info(f"[{self.tree_name}] Tree loaded. Total nodes: {len(self.node_index)}")
            else:
                logger.warning(f"[{self.tree_name}] No root found after loading.")

        except FileNotFoundError:
            logger.warning(f"[{self.tree_name}] Tree file not found: {filepath}")
        except Exception as e:
            logger.error(f"[{self.tree_name}] Failed to load tree: {e}")


class DualTreeMemory:
    """
    双树记忆管理器
    
    管理两棵独立的 PR-Tree:
    - task_tree: 以任务目标为索引 (e.g. "put some spraybottle on toilet")
      存储内容: Workflow Schema, 策略残差, 任务完成步骤总结
    - env_tree:  以环境描述为索引 (e.g. "You are in the middle of a room. Looking quickly around you...")
      存储内容: 环境反思, 物品位置经验, 环境交互教训

    对外接口与原 PRTreeMemory 保持兼容。
    """

    def __init__(self, retriever: VectorRetriever):
        self.retriever = retriever

        # 创建两棵子树 (各自使用独立的阈值策略)
        # 任务树: 宽松阈值，增加召回率 (任务描述短、语义差异大)
        self.task_tree = PRTreeMemory(
            retriever, tree_name="task",
            base_threshold=TASK_TREE_BASE_THRESHOLD,
            depth_step=TASK_TREE_DEPTH_STEP,
            max_threshold=TASK_TREE_MAX_THRESHOLD
        )
        # 环境树: 严格阈值，保证精确匹配 (环境描述长、相似环境 embedding 天然接近)
        self.env_tree = PRTreeMemory(
            retriever, tree_name="env",
            base_threshold=ENV_TREE_BASE_THRESHOLD,
            depth_step=ENV_TREE_DEPTH_STEP,
            max_threshold=ENV_TREE_MAX_THRESHOLD
        )

        # 合并统计
        self.stats = {
            "total_nodes": 2,
            "max_depth": 0,
            "task_tree_nodes": 1,
            "env_tree_nodes": 1,
        }

        # Skill 补丁缓存（快思考路径），传入 retriever 以使用 embedding 匹配
        self.skill_cache = SkillCache(retriever=self.retriever)
        self._skill_compiler = SkillCompiler()

        # 尝试加载
        self._load_trees_if_exist()

    def _load_trees_if_exist(self):
        task_file = Path(STORAGE_PATH) / "task_tree.json"
        env_file = Path(STORAGE_PATH) / "env_tree.json"
        if task_file.exists():
            self.task_tree.load_tree(str(task_file))
        if env_file.exists():
            self.env_tree.load_tree(str(env_file))
        self._sync_stats()

    def _sync_stats(self):
        self.stats["task_tree_nodes"] = self.task_tree.stats["total_nodes"]
        self.stats["env_tree_nodes"] = self.env_tree.stats["total_nodes"]
        self.stats["total_nodes"] = (
            self.task_tree.stats["total_nodes"] + self.env_tree.stats["total_nodes"]
        )
        self.stats["max_depth"] = max(
            self.task_tree.stats["max_depth"],
            self.env_tree.stats["max_depth"]
        )

    # =====================================================================
    # 检索接口
    # =====================================================================

    def retrieve_task_path(self, task_description: str) -> List[MemoryNode]:
        """在任务树中检索"""
        return self.task_tree.retrieve_context_path(task_description)

    def retrieve_env_path(self, env_description: str) -> List[MemoryNode]:
        """在环境树中检索"""
        return self.env_tree.retrieve_context_path(env_description)

    def retrieve_dual_paths(
        self, task_description: str, env_description: str
    ) -> Dict[str, List[MemoryNode]]:
        """同时检索两棵树，返回两条路径"""
        task_path = self.retrieve_task_path(task_description)
        env_path = self.retrieve_env_path(env_description)
        return {
            "task_path": task_path,
            "env_path": env_path
        }

    # =====================================================================
    # 写入接口
    # =====================================================================

    def add_task_experience(
        self,
        anchor_node: MemoryNode,
        scenario_description: str,
        memory_description: str,
        content_body: str,
        result_status: Union[str, ResultStatus],
        force_sibling: bool = False
    ) -> MemoryNode:
        """向任务树写入经验"""
        node = self.task_tree.add_experience_node(
            anchor_node=anchor_node,
            scenario_description=scenario_description,
            memory_description=memory_description,
            content_body=content_body,
            result_status=result_status,
            force_sibling=force_sibling
        )
        self._sync_stats()
        return node

    def add_env_experience(
        self,
        anchor_node: MemoryNode,
        scenario_description: str,
        memory_description: str,
        content_body: str,
        result_status: Union[str, ResultStatus],
        force_sibling: bool = False
    ) -> MemoryNode:
        """向环境树写入经验"""
        node = self.env_tree.add_experience_node(
            anchor_node=anchor_node,
            scenario_description=scenario_description,
            memory_description=memory_description,
            content_body=content_body,
            result_status=result_status,
            force_sibling=force_sibling
        )
        self._sync_stats()
        return node

    def add_task_root_schema(
        self,
        scenario_description: str,
        memory_description: str,
        content_body: str,
        result_status: str = ResultStatus.SUCCESS
    ) -> MemoryNode:
        node = self.task_tree.add_root_schema(
            scenario_description=scenario_description,
            memory_description=memory_description,
            content_body=content_body,
            result_status=result_status
        )
        self._sync_stats()
        return node

    def add_env_root_schema(
        self,
        scenario_description: str,
        memory_description: str,
        content_body: str,
        result_status: str = ResultStatus.SUCCESS
    ) -> MemoryNode:
        node = self.env_tree.add_root_schema(
            scenario_description=scenario_description,
            memory_description=memory_description,
            content_body=content_body,
            result_status=result_status
        )
        self._sync_stats()
        return node

    # =====================================================================
    # 持久化
    # =====================================================================

    def save_trees(self, task_filepath: Optional[str] = None, env_filepath: Optional[str] = None):
        task_fp = task_filepath or str(Path(STORAGE_PATH) / "task_tree.json")
        env_fp = env_filepath or str(Path(STORAGE_PATH) / "env_tree.json")
        self.task_tree.save_tree(task_fp)
        self.env_tree.save_tree(env_fp)

    def load_trees(self, task_filepath: str, env_filepath: str):
        self.task_tree.load_tree(task_filepath)
        self.env_tree.load_tree(env_filepath)
        self._sync_stats()

    # 兼容旧接口
    def save_tree(self, filepath: Optional[str] = None):
        """兼容旧接口：保存两棵树（filepath 用作前缀）"""
        if filepath:
            base = filepath.replace(".json", "")
            self.save_trees(f"{base}_task.json", f"{base}_env.json")
        else:
            self.save_trees()

    def load_tree(self, filepath: str):
        """兼容旧接口：加载两棵树"""
        base = filepath.replace(".json", "")
        task_fp = f"{base}_task.json"
        env_fp = f"{base}_env.json"
        if Path(task_fp).exists():
            self.task_tree.load_tree(task_fp)
        if Path(env_fp).exists():
            self.env_tree.load_tree(env_fp)
        self._sync_stats()

    # =====================================================================
    # 记忆固化：热点延迟编译触发器
    # =====================================================================

    def trigger_consolidation_check(self, node: MemoryNode, llm_client=None) -> None:
        """
        在每次任务执行成功并更新节点 success_count 后调用。
        当 success_count >= CONSOLIDATION_THRESHOLD 且未固化时，
        异步编译该节点的完整路径链为 ProceduralSkillPatch。
        """
        if node.meta.get("is_consolidated", False):
            return
        if node.meta.get("success_count", 0) < CONSOLIDATION_THRESHOLD:
            return

        logger.info(
            f"[Consolidation] Node {node.node_id[:8]} reached success_count="
            f"{node.meta['success_count']} — compiling Skill Patch..."
        )

        # 提取从 Root 到当前节点的完整文本链条
        path = node.get_path_to_root()
        chain_texts = []
        for path_node in path:
            if "GLOBAL_ROOT_PLACEHOLDER" in path_node.payload.get("scenario_description", ""):
                continue
            activation = path_node.payload.get("activation_condition") or path_node.payload.get("memory_description", "")
            execution = path_node.payload.get("execution_procedure") or path_node.payload.get("content_body", "")
            termination = path_node.payload.get("termination_condition", "")
            chain_texts.append(
                f"[Node {path_node.node_id[:8]}] ({path_node.meta.get('node_type', 'RESIDUAL')})\n"
                f"Activation: {activation}\n"
                f"Execution: {execution}\n"
                f"Termination: {termination}"
            )

        if not chain_texts:
            return

        patch = self._skill_compiler.compile(
            node_chain_texts=chain_texts,
            skill_cache=self.skill_cache,
            source_node_id=node.node_id,
            llm_client=llm_client,
        )

        if patch:
            node.meta["is_consolidated"] = True
            logger.info(f"[Consolidation] ✅ Patch compiled for node {node.node_id[:8]}")
