"""
PR-Tree Memory Node (v3.0 Unified Definition)
核心数据结构：定义记忆树的节点
"""

import uuid
import json
import time
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import numpy as np

# 假设 config 中有这些常量配置
# from config import JSON_INDENT, JSON_ENSURE_ASCII

class NodeType(str, Enum):
    ROOT = "ROOT"
    RESIDUAL = "RESIDUAL"

class ResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"

class MemoryNode:
    """
    记忆节点类 (PR-Tree Node)
    
    结构严格遵循 v3.0 JSON Schema:
    - meta: 元数据 (类型, 状态, 时间, 热度)
    - index: 向量索引 (Embedding)
    - payload: 核心载荷 (场景描述, 经验/教训内容)
    """
    
    def __init__(
        self,
        node_type: Union[str, NodeType],
        result_status: Union[str, ResultStatus],
        embedding: np.ndarray,
        scenario_description: str,
        memory_description: str,
        content_body: str,
        node_id: Optional[str] = None,
        parent: Optional['MemoryNode'] = None
    ):
        # 1. 基础 ID 管理
        self.node_id = node_id or str(uuid.uuid4())
        
        # 2. 树拓扑关系 (运行时对象引用)
        self.parent: Optional[MemoryNode] = None
        self.children: List[MemoryNode] = []
        
        # 如果初始化时传入了 parent，立即挂载
        if parent:
            parent.add_child(self)

        # 3. 构建符合 Schema 的数据区
        
        # --- A. 元数据 (Metadata) ---
        self.meta = {
            "node_type": NodeType(node_type).value,         # 确保是字符串枚举值
            "result_status": ResultStatus(result_status).value,
            "created_at": int(time.time()),                 # 使用时间戳
            "usage_count": 0,
            "hit_count": 0,       # 作为成功检索路径一部分时加 1
            "success_count": 0,   # 利用该节点经验执行任务最终成功时加 1
            "is_consolidated": False  # 被编译为 Skill 补丁后设为 True
        }

        # --- B. 索引 (Index) ---
        self.index = {
            "embedding": embedding  # 注意: 序列化时需转换为 list
        }

        # --- C. 核心载荷 (Unified Payload) ---
        self.payload = {
            "scenario_description": scenario_description,
            "memory_description": memory_description,
            "content_body": content_body,
            # Skill 字段 (可选, 新写入的节点会填充这些字段)
            "activation_condition": None,
            "execution_procedure": None,
            "termination_condition": None,
        }

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
        """动态计算深度 (Root = 0)"""
        d = 0
        curr = self
        while curr.parent:
            d += 1
            curr = curr.parent
        return d

    def add_child(self, child_node: 'MemoryNode') -> None:
        """添加子节点并维护双向引用"""
        child_node.parent = self
        if child_node not in self.children:
            self.children.append(child_node)
    
    def increment_usage(self) -> None:
        """增加使用热度"""
        self.meta["usage_count"] += 1

    def get_path_to_root(self) -> List['MemoryNode']:
        """获取从 Root 到当前节点的完整路径 (用于构建 Context)"""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]  # Reverse: Root -> ... -> Current
    
    def to_dict(self) -> Dict[str, Any]:
        """
        序列化为符合定义的 JSON 字典
        """
        # 处理 numpy 数组转 list
        embedding_data = self.index["embedding"]
        if isinstance(embedding_data, np.ndarray):
            embedding_data = embedding_data.tolist()

        return {
            "node_id": self.node_id,
            "parent_id": self.parent.node_id if self.parent else None,
            "children_ids": [child.node_id for child in self.children],
            
            "meta": self.meta,
            
            "index": {
                "embedding": embedding_data
            },
            
            "payload": self.payload
        }
    
    def to_json_file(self, filepath: str) -> None:
        """保存单节点到文件"""
        # 简单处理，实际生产中可能不需要单节点存文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        """
        反序列化：从字典恢复节点对象
        注意：此方法无法自动恢复 parent/children 的对象引用，
        需要在树加载器(TreeLoader)中处理拓扑连接。
        """
        # 提取各个部分的字段
        meta = data.get("meta", {})
        idx = data.get("index", {})
        payload = data.get("payload", {})
        
        # 恢复 numpy 数组
        emb = np.array(idx.get("embedding", []))
        
        # 实例化
        node = cls(
            node_id=data.get("node_id"),
            node_type=meta.get("node_type", NodeType.RESIDUAL.value),
            result_status=meta.get("result_status", ResultStatus.SUCCESS.value),
            embedding=emb,
            scenario_description=payload.get("scenario_description", ""),
            memory_description=payload.get("memory_description", ""),
            content_body=payload.get("content_body", "")
        )
        
        # 恢复其他元数据
        node.meta["created_at"] = meta.get("created_at", int(time.time()))
        node.meta["usage_count"] = meta.get("usage_count", 0)
        node.meta["hit_count"] = meta.get("hit_count", 0)
        node.meta["success_count"] = meta.get("success_count", 0)
        node.meta["is_consolidated"] = meta.get("is_consolidated", False)

        # 恢复 Skill 字段 (向后兼容旧节点)
        node.payload["activation_condition"] = payload.get("activation_condition")
        node.payload["execution_procedure"] = payload.get("execution_procedure")
        node.payload["termination_condition"] = payload.get("termination_condition")

        return node

    def __repr__(self) -> str:
        """打印时的友好显示"""
        short_desc = self.payload['scenario_description'][:30] + "..." 
        return (f"<MemoryNode type={self.node_type} "
                f"status={self.status} "
                f"desc='{short_desc}'>")

# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 创建 Root Node (Workflow Schema)
    root = MemoryNode(
        node_type=NodeType.ROOT,
        result_status=ResultStatus.SUCCESS,
        embedding=np.zeros(384), # 模拟向量
        scenario_description="General Python Data Plotting",
        memory_description="Standard data plotting workflow",
        content_body="1. Load Data\n2. Plot Data\n3. Save"
    )

    # 2. 创建 Residual Node (Failure Pivot)
    child = MemoryNode(
        node_type=NodeType.RESIDUAL,
        result_status=ResultStatus.SUCCESS,
        embedding=np.zeros(384),
        scenario_description="Data is JSON string, not CSV file",
        memory_description="Handling JSON data formats",
        content_body="Use json.loads() instead of read_csv()",
        parent=root # 自动挂载
    )
    
    print(json.dumps(child.to_dict(), indent=2))