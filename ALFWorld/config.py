"""
PR-Tree Configuration File
配置文件：定义所有全局参数和路径
"""

import os
from pathlib import Path

# ==================== 路径配置 ====================
# ALFWorld 数据路径
ALFWORLD_DATA_PATH = '/data/REDACTED_USER/PRTree/ALFWorld/data/alfworld'

# 本地 Embedding 模型路径 (使用 sentence-transformers)
# 推荐模型: all-MiniLM-L6-v2, all-mpnet-base-v2
EMBEDDING_MODEL_PATH = "/hdd/REDACTED_USER/PRTree/embedding/e5-base-v2"

# 记忆存储路径
STORAGE_PATH = "./memory_storage"

# ==================== 树结构参数 ====================
# 树的最大深度限制 (强制约束)
MAX_DEPTH = 5

# 相似度阈值 (Cosine Similarity)
# 低于此值的检索结果将被视为不相关
SIMILARITY_THRESHOLD = 0.85
SIMILARITY_THRESHOLD_ROOT = 0.85  # 根节点检索阈值
SIMILARITY_THRESHOLD_RESIDUAL = 0.85  # 残差检索阈值

# ==================== 双树差异化阈值配置 ====================
# 任务树: 任务目标描述较短且语义集中，不同任务之间的 embedding 差异较大，
#          因此用较宽松的阈值以增加召回率
TASK_TREE_BASE_THRESHOLD = 0.80   # 任务树基础阈值 (Root children 层)
TASK_TREE_DEPTH_STEP = 0.03       # 每深一层增加的严格度
TASK_TREE_MAX_THRESHOLD = 0.95    # 任务树阈值上限

# 环境树: 环境描述较长且包含大量物品列表，相似环境之间 embedding 天然就很接近，
#          因此需要较高的阈值以保证精确匹配
ENV_TREE_BASE_THRESHOLD = 0.88    # 环境树基础阈值 (Root children 层)
ENV_TREE_DEPTH_STEP = 0.02        # 每深一层增加的严格度
ENV_TREE_MAX_THRESHOLD = 0.99     # 环境树阈值上限

# ==================== 检索参数 ====================
# 检索时返回的候选节点数量
TOP_K_CANDIDATES = 5

# 反思增强时的兄弟节点检查范围
MAX_SIBLING_CHECK = 10

# ==================== 持久化参数 ====================
# JSON 文件保存格式
JSON_INDENT = 2
JSON_ENSURE_ASCII = False

# 树结构文件名
TREE_STRUCTURE_FILE = "prtree_structure.json"
TREE_METADATA_FILE = "prtree_metadata.json"

# ==================== 日志参数 ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==================== 自动创建存储目录 ====================
Path(STORAGE_PATH).mkdir(parents=True, exist_ok=True)

# ==================== 节点状态枚举 ====================
class NodeStatus:
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PARTIAL = "PARTIAL"
    
# ==================== 调试模式 ====================
DEBUG_MODE = True
VERBOSE_RETRIEVAL = True

