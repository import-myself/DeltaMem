"""
PR-Tree Memory Node
核心数据结构：记忆树节点
"""

import uuid
import json
import time
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import numpy as np


class NodeType(str, Enum):
    ROOT = "ROOT"
    RESIDUAL = "RESIDUAL"


class ResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class MemoryNode:
    """
    PR-Tree 节点

    payload 核心字段（Skill 格式）:
      activation_condition  — 触发条件 / 检索锚
      execution_procedure   — 程序步骤（任务树）或声明性事实（环境树）
      termination_condition — 完成判定（失败节点为空字符串）
    """

    def __init__(
        self,
        node_type: Union[str, NodeType],
        result_status: Union[str, ResultStatus],
        embedding: np.ndarray,
        scenario_description: str,
        node_id: Optional[str] = None,
        parent: Optional['MemoryNode'] = None,
    ):
        self.node_id = node_id or str(uuid.uuid4())

        self.parent: Optional[MemoryNode] = None
        self.children: List[MemoryNode] = []

        if parent:
            parent.add_child(self)

        self.meta = {
            "node_type": NodeType(node_type).value,
            "result_status": ResultStatus(result_status).value,
            "created_at": int(time.time()),
            "usage_count": 0,
            "hit_count": 0,
            "success_count": 0,
            "is_consolidated": False,
        }

        self.index = {"embedding": embedding}

        self.payload = {
            "scenario_description": scenario_description,
            # Skill 字段（由 DualMemoryWriter 填充）
            "activation_condition": None,
            "execution_procedure": None,
            "termination_condition": None,
        }

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def node_type(self) -> str:
        return self.meta["node_type"]

    @property
    def status(self) -> str:
        return self.meta["result_status"]

    @property
    def embedding(self) -> np.ndarray:
        return self.index["embedding"]

    @property
    def depth(self) -> int:
        d, curr = 0, self
        while curr.parent:
            d += 1
            curr = curr.parent
        return d

    # ------------------------------------------------------------------
    # 树操作
    # ------------------------------------------------------------------

    def add_child(self, child_node: 'MemoryNode') -> None:
        child_node.parent = self
        if child_node not in self.children:
            self.children.append(child_node)

    def increment_usage(self) -> None:
        self.meta["usage_count"] += 1

    def get_path_to_root(self) -> List['MemoryNode']:
        path, current = [], self
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        emb = self.index["embedding"]
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()
        return {
            "node_id": self.node_id,
            "parent_id": self.parent.node_id if self.parent else None,
            "children_ids": [c.node_id for c in self.children],
            "meta": self.meta,
            "index": {"embedding": emb},
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        meta = data.get("meta", {})
        payload = data.get("payload", {})
        emb = np.array(data.get("index", {}).get("embedding", []))

        node = cls(
            node_id=data.get("node_id"),
            node_type=meta.get("node_type", NodeType.RESIDUAL.value),
            result_status=meta.get("result_status", ResultStatus.SUCCESS.value),
            embedding=emb,
            scenario_description=payload.get("scenario_description", ""),
        )

        node.meta.update({
            "created_at": meta.get("created_at", int(time.time())),
            "usage_count": meta.get("usage_count", 0),
            "hit_count": meta.get("hit_count", 0),
            "success_count": meta.get("success_count", 0),
            "is_consolidated": meta.get("is_consolidated", False),
            "episode_idx": meta.get("episode_idx", -1),
        })

        # Skill 字段（兼容旧格式：memory_description/content_body）
        node.payload["activation_condition"] = (
            payload.get("activation_condition")
            or payload.get("memory_description")
        )
        node.payload["execution_procedure"] = (
            payload.get("execution_procedure")
            or payload.get("content_body")
        )
        node.payload["termination_condition"] = payload.get("termination_condition")

        return node

    def __repr__(self) -> str:
        desc = self.payload.get("scenario_description", "")[:40]
        return f"<MemoryNode type={self.node_type} status={self.status} desc='{desc}'>"
