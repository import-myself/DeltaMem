"""
PR-Tree Configuration File (ScienceWorld)
配置文件：定义所有全局参数和路径
"""

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent        # ScienceWorld/
_ROOT = _HERE.parent                           # project root

# ==================== 路径配置 ====================
# 本地 Embedding 模型路径 (使用 sentence-transformers)
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", str(_ROOT / "embedding" / "e5-base-v2"))

# 记忆存储路径
STORAGE_PATH = "./storage"

# ==================== 树结构参数 ====================
# 树的最大深度限制 (强制约束)
MAX_DEPTH = 5

# 相似度阈值 (Cosine Similarity)
SIMILARITY_THRESHOLD = 0.85
SIMILARITY_THRESHOLD_ROOT = 0.85
SIMILARITY_THRESHOLD_RESIDUAL = 0.85

# ==================== 双树差异化阈值配置 ====================
# 任务树: 任务目标描述较短且语义集中
TASK_TREE_BASE_THRESHOLD = 0.84   # 网格搜索最优 (dev sr=0.6907, reward=0.8498)
TASK_TREE_DEPTH_STEP = 0.0
TASK_TREE_MAX_THRESHOLD = 0.95
TASK_TREE_EMBED_WITH_ACTIVATION = False

# 环境树: ScienceWorld 场景描述较长，需要较高阈值精确匹配
ENV_TREE_BASE_THRESHOLD = 0.86   # 网格搜索最优 (dev sr=0.6907, reward=0.8498)
ENV_TREE_DEPTH_STEP = 0.0
ENV_TREE_MAX_THRESHOLD = 0.99

# ==================== 检索参数 ====================
TOP_K_CANDIDATES = 5
MAX_SIBLING_CHECK = 10

# ==================== 持久化参数 ====================
JSON_INDENT = 2
JSON_ENSURE_ASCII = False

TREE_STRUCTURE_FILE = "prtree_structure.json"
TREE_METADATA_FILE = "prtree_metadata.json"

# ==================== 记忆固化参数 ====================
CONSOLIDATION_THRESHOLD = 3

# FAILURE 节点检索惩罚
FAILURE_NODE_PENALTY = 0.05

# ==================== 日志参数 ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==================== 节点状态枚举 ====================
class NodeStatus:
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PARTIAL = "PARTIAL"

# ==================== 调试模式 ====================
DEBUG_MODE = True
VERBOSE_RETRIEVAL = True
