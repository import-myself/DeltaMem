"""
PR-Tree Mind2Web Configuration (v4.0)
Mind2Web 专用配置文件
"""

from pathlib import Path

# ==================== 路径配置 ====================
EMBEDDING_MODEL_PATH = "/data/REDACTED_USER/PRTree/embedding/e5-base-v2"
STORAGE_PATH = "./memory_storage"
DATA_DIR = "./data"
EXEMPLAR_PATH = "./data/example/exemplars.json"
SCORE_PATH = "./data/scores_all_data.pkl"

# ==================== 树结构参数 ====================
MAX_DEPTH = 5
JSON_INDENT = 2
JSON_ENSURE_ASCII = False
TREE_STRUCTURE_FILE = "prtree_structure.json"
TREE_METADATA_FILE = "prtree_metadata.json"

# ==================== 双树差异化阈值配置 ====================
# 任务树: 任务自然语言描述差异较大，使用宽松阈值
TASK_TREE_BASE_THRESHOLD = 0.75
TASK_TREE_DEPTH_STEP = 0.03
TASK_TREE_MAX_THRESHOLD = 0.95

# 网站树 (对应 ALFWorld 的环境树):
# env_key = domain::subdomain::website，相似度天然偏高，提高基础阈值防止跨网站误命中
# 消融实验显示 eb=0.92~0.94 时 hit_rate 更精准，避免将不相关网站经验噪声注入
WEBSITE_TREE_BASE_THRESHOLD = 0.92
WEBSITE_TREE_DEPTH_STEP = 0.01
WEBSITE_TREE_MAX_THRESHOLD = 0.99

# 别名，供 dual_tree_manager.py 使用 (读取 ENV_TREE_* 名称)
ENV_TREE_BASE_THRESHOLD = WEBSITE_TREE_BASE_THRESHOLD
ENV_TREE_DEPTH_STEP = WEBSITE_TREE_DEPTH_STEP
ENV_TREE_MAX_THRESHOLD = WEBSITE_TREE_MAX_THRESHOLD

# ==================== 检索参数 (retriever.py 依赖) ====================
SIMILARITY_THRESHOLD = 0.85
SIMILARITY_THRESHOLD_ROOT = 0.85
SIMILARITY_THRESHOLD_RESIDUAL = 0.85
TOP_K_CANDIDATES = 5
MAX_SIBLING_CHECK = 10

# ==================== 日志参数 ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==================== 调试模式 ====================
DEBUG_MODE = True
VERBOSE_RETRIEVAL = True

# ==================== 自动创建存储目录 ====================
Path(STORAGE_PATH).mkdir(parents=True, exist_ok=True)
