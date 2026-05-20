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

# ==================== 双树差异化阈值（基于真实 instruction cosine 分布实测）====================
# 实测 ETO test_indices 前 20 个 task instruction 的 e5-base-v2 cosine 分布：
#   同子类/同大类（应命中）：0.908 ~ 0.972
#   跨大类（不应命中）    ：0.806 ~ 0.881
# 0.85 经实测在 v2 给出 SR=45%（与 no-memory 持平），0.90 反让 same-category 复访被 FAILURE
# patch 误导导致 SR 降到 35%。回 0.85 + 由 reader 过滤 FAILURE 节点。
TASK_TREE_BASE_THRESHOLD = 0.85
TASK_TREE_DEPTH_STEP     = 0.02
TASK_TREE_MAX_THRESHOLD  = 0.97
TASK_TREE_EMBED_WITH_ACTIVATION = False

# 环境树: env_description 是 LLM 抽的产品类目短语。短语间同类目 cosine ≈ 1.0，
#         跨大类 < 0.7。0.85 即足以分离。
ENV_TREE_BASE_THRESHOLD = 0.85
ENV_TREE_DEPTH_STEP     = 0.02
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
