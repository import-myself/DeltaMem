"""
PR-Tree WebShop Configuration

参考 Mind2web/config.py 与 ALFWorld/config.py 的双树阈值差异化策略：
- Task Tree:  instruction 自由文本变化大，用较宽松阈值（高召回）
- Env  Tree:  env_description 是产品类目短语，相似类目间天然相近，提高基础阈值（避免误命中）
"""

# ==================== 路径配置 ====================
EMBEDDING_MODEL_PATH = "/hdd/REDACTED_USER/DeltaMem/embedding/e5-base-v2"
STORAGE_PATH = "./storage"
DATA_DIR = "./data"

# ==================== 树结构参数 ====================
MAX_DEPTH = 5
JSON_INDENT = 2
JSON_ENSURE_ASCII = False
TREE_STRUCTURE_FILE = "prtree_structure.json"
TREE_METADATA_FILE = "prtree_metadata.json"

# ==================== 双树阈值（网格搜索最优参数）====================
# ==================== 双树阈值（网格搜索最优参数，按 avg_reward 排序）====================
# 网格搜索结果（200 ep, test split）：
#   flat tb=0.85 / eb=0.85  → reward=0.5889  SR=51.5%  ← 最优，采用
#   flat tb=0.80 / eb=0.80  → reward=0.5876  SR=53.0%
#   DFS  tb=0.80 / eb=0.85  → reward=0.5800  SR=48.5%
# flat retrieval (depth_step=0.0) 优于 DFS，最优参数：tb=0.85 / eb=0.85
TASK_TREE_BASE_THRESHOLD = 0.85
TASK_TREE_DEPTH_STEP     = 0.0   # flat retrieval，depth_step 无效
TASK_TREE_MAX_THRESHOLD  = 0.97
TASK_TREE_EMBED_WITH_ACTIVATION = False

ENV_TREE_BASE_THRESHOLD = 0.85
ENV_TREE_DEPTH_STEP     = 0.0   # flat retrieval，depth_step 无效
ENV_TREE_MAX_THRESHOLD  = 0.97

# ==================== 检索参数 ====================
SIMILARITY_THRESHOLD = 0.85
SIMILARITY_THRESHOLD_ROOT     = 0.85
SIMILARITY_THRESHOLD_RESIDUAL = 0.85
TOP_K_CANDIDATES   = 5
MAX_SIBLING_CHECK  = 10

# ==================== 记忆固化 ====================
CONSOLIDATION_THRESHOLD = 3
FAILURE_NODE_PENALTY    = 0.10

# ==================== 日志 ====================
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEBUG_MODE        = False
VERBOSE_RETRIEVAL = False
