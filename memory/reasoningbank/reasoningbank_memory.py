"""
reasoningbank_memory.py  —  ReasoningBank 记忆后端
====================================================
从成功与失败轨迹中双向提炼可泛化推理策略，实现闭环记忆。

架构：
  MemoryStore     → JSON 持久化 + numpy 余弦相似度检索
  MemoryExtractor → LLM 提炼 memory_items（success / failure 分支）
  MemoryRetriever → 向量化查询 + top-k 语义检索
  ReasoningBankMemory → 三者整合的对外接口，与 AWMMemory 同级

存储路径（按 benchmark 隔离）：
  {memory_path}/{benchmark}/memory_pool.json
  {memory_path}/{benchmark}/embeddings.npy
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt 模板（按 benchmark 适配）
# ---------------------------------------------------------------------------

_SUCCESS_SYSTEM: Dict[str, str] = {
    "alfworld": (
        "You are an expert in household task planning. "
        "You will be given a user query and a trajectory where an agent successfully completed the task."
    ),
    "sciworld": (
        "You are an expert in science experiment reasoning. "
        "You will be given a user query and a trajectory where an agent successfully completed the experiment task."
    ),
    "mind2web": (
        "You are an expert in web navigation. "
        "You will be given a user query and a trajectory where an agent successfully accomplished the task."
    ),
}

_FAILURE_SYSTEM: Dict[str, str] = {
    "alfworld": (
        "You are an expert in household task planning. "
        "You will be given a user query and a trajectory where an agent attempted but failed to complete the task."
    ),
    "sciworld": (
        "You are an expert in science experiment reasoning. "
        "You will be given a user query and a trajectory where an agent attempted but failed the experiment task."
    ),
    "mind2web": (
        "You are an expert in web navigation. "
        "You will be given a user query and a trajectory where an agent attempted to resolve the task but failed."
    ),
}

_EXTRACTOR_BODY = """
## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's {status} trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
- You must first {think_instruction}, and then summarize the insights.
- You can extract at most 3 memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific object names, locations, queries, or string contents, but rather focus on generalizable insights.

## Output Format
Respond with a JSON array of objects, each with exactly three keys: "title", "description", "content".
Example:
[
  {{
    "title": "Short strategy title",
    "description": "One-sentence summary of this memory item",
    "content": "1-3 sentences describing the concrete actionable insight or preventive strategy."
  }}
]
Output ONLY the JSON array, no other text.

## User Query
{query}

## Trajectory ({status})
{trajectory}
"""

_INJECTION_PREFIX = (
    "Below are some memory items that I accumulated from past interactions "
    "that may be helpful to solve the task. You can use them when you feel they are relevant. "
    "In each step, please first explicitly discuss if you want to use each memory item or not, "
    "and then take action.\n\n"
)

_DEFAULT_SYSTEM = "alfworld"


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    """JSON + numpy 向量存储。"""

    POOL_FILE = "memory_pool.json"
    EMB_FILE = "embeddings.npy"

    def __init__(self, store_dir: Path, load_existing: bool = True):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._pool: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None
        self._dirty = False
        self._lock = threading.Lock()
        if load_existing:
            self._load()

    def _load(self):
        pool_file = self.store_dir / self.POOL_FILE
        emb_file = self.store_dir / self.EMB_FILE
        if pool_file.exists():
            with open(pool_file, "r", encoding="utf-8") as f:
                self._pool = json.load(f)
            logger.info(f"[RB] Loaded {len(self._pool)} entries ← {pool_file}")
        if emb_file.exists():
            self._embeddings = np.load(str(emb_file))
            logger.info(f"[RB] Loaded embeddings shape={self._embeddings.shape}")

    def append(self, entry: Dict[str, Any], embedding: Optional[np.ndarray] = None):
        with self._lock:
            self._pool.append(entry)
            if embedding is not None:
                vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
                if self._embeddings is None:
                    self._embeddings = vec
                else:
                    self._embeddings = np.concatenate([self._embeddings, vec], axis=0)
            self._dirty = True

    def save(self):
        with self._lock:
            if not self._dirty:
                return
            with open(self.store_dir / self.POOL_FILE, "w", encoding="utf-8") as f:
                json.dump(self._pool, f, ensure_ascii=False, indent=2)
            if self._embeddings is not None:
                np.save(str(self.store_dir / self.EMB_FILE), self._embeddings)
            self._dirty = False
            logger.info(f"[RB] Saved {len(self._pool)} entries → {self.store_dir}")

    @property
    def pool(self) -> List[Dict[str, Any]]:
        return self._pool

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        return self._embeddings

    def __len__(self) -> int:
        return len(self._pool)


# ---------------------------------------------------------------------------
# MemoryExtractor
# ---------------------------------------------------------------------------

class MemoryExtractor:
    """调用 LLM 从轨迹中提炼 memory_items。"""

    def __init__(self, llm_client, benchmark: str = "alfworld"):
        self.llm_client = llm_client
        self.benchmark = benchmark.lower()

    def extract(
        self,
        query: str,
        trajectory: List[str],
        success: bool,
    ) -> List[Dict[str, str]]:
        status = "successful" if success else "failed"
        think_instruction = (
            "think why the trajectory is successful"
            if success
            else "reflect and think why the trajectory failed, and summarize lessons to prevent failure in the future"
        )
        system_map = _SUCCESS_SYSTEM if success else _FAILURE_SYSTEM
        system_content = system_map.get(self.benchmark, system_map.get(_DEFAULT_SYSTEM, ""))

        traj_text = "\n".join(trajectory)
        user_content = _EXTRACTOR_BODY.format(
            status=status,
            think_instruction=think_instruction,
            query=query,
            trajectory=traj_text,
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        try:
            raw = self.llm_client.chat(messages, temperature=0.0, max_tokens=2048)
            items = self._parse_json(raw)
            # 校验格式
            valid = [
                it for it in items
                if isinstance(it, dict) and "title" in it and "content" in it
            ]
            logger.info(f"[RB] Extracted {len(valid)} memory_items (success={success})")
            return valid
        except Exception as e:
            logger.warning(f"[RB] Extraction failed: {e}")
            return []

    @staticmethod
    def _parse_json(raw: str) -> List[Dict]:
        raw = raw.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return [result]
        except Exception:
            pass
        return []


# ---------------------------------------------------------------------------
# MemoryRetriever
# ---------------------------------------------------------------------------

class MemoryRetriever:
    """基于余弦相似度的向量检索。"""

    def __init__(self, embed_fn):
        self.embed_fn = embed_fn  # Callable[[str], List[float]]

    def embed(self, text: str) -> np.ndarray:
        vec = np.array(self.embed_fn(text), dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def retrieve(
        self,
        query: str,
        store: MemoryStore,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        if len(store) == 0 or store.embeddings is None:
            return []

        q_vec = self.embed(query)
        scores = store.embeddings @ q_vec  # cosine similarity (already L2-normed)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [store.pool[i] for i in top_indices]


# ---------------------------------------------------------------------------
# 嵌入函数（用 LLMClient 所在环境的 sentence-transformers 或 openai）
# ---------------------------------------------------------------------------

def _build_embed_fn(embed_model_path: Optional[str] = None):
    """
    优先尝试 sentence-transformers（本地模型），
    若路径为空或加载失败则尝试 openai/text-embedding-ada-002，
    最后降级为基于词频的简单哈希向量。
    """
    # 1. sentence-transformers 本地模型
    if embed_model_path:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(embed_model_path)
            logger.info(f"[RB] Using SentenceTransformer: {embed_model_path}")
            def _st_embed(text: str) -> List[float]:
                return model.encode(text, normalize_embeddings=True).tolist()
            return _st_embed
        except Exception as e:
            logger.warning(f"[RB] SentenceTransformer load failed ({e}), trying openai…")

    # 2. openai text-embedding-ada-002
    try:
        import openai as _openai
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        client = _openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else _openai.OpenAI(api_key=api_key)
        logger.info("[RB] Using openai text-embedding-ada-002 for embeddings")

        def _oai_embed(text: str) -> List[float]:
            resp = client.embeddings.create(model="text-embedding-ada-002", input=text[:8000])
            return resp.data[0].embedding

        return _oai_embed
    except Exception as e:
        logger.warning(f"[RB] OpenAI embedding failed ({e}), using hash fallback.")

    # 3. 哈希向量降级（不依赖任何外部库）
    import hashlib

    DIM = 256

    def _hash_embed(text: str) -> List[float]:
        vec = np.zeros(DIM, dtype=np.float32)
        words = text.lower().split()
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % DIM] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    return _hash_embed


# ---------------------------------------------------------------------------
# ReasoningBankMemory（对外主接口）
# ---------------------------------------------------------------------------

class ReasoningBankMemory:
    """
    ReasoningBank 记忆后端。

    用法（与 AWMMemory 接口对称）：

        rb = ReasoningBankMemory(
            memory_path="storage/rb_memory",
            llm_client=llm_client,
            benchmark="alfworld",
            embed_model_path="/path/to/e5-base-v2",
        )

        # episode 前：检索并获取注入字符串
        memory_str = rb.retrieve_memory_str(query=task_instruction)

        # episode 后：提取并存储
        rb.extract_and_store(
            query=task_instruction,
            trajectory=trajectory_list,
            success=True,
        )

        # 定期或结束时持久化
        rb.save()
    """

    def __init__(
        self,
        memory_path: str,
        llm_client,
        benchmark: str = "alfworld",
        embed_model_path: Optional[str] = None,
        top_k: int = 1,
        max_memory_chars: int = 3000,
        load_existing: bool = True,
        allow_updates: bool = True,
    ):
        self.benchmark = benchmark.lower()
        self.top_k = top_k
        self.max_memory_chars = max_memory_chars
        self.allow_updates = allow_updates

        base = Path(memory_path) / self.benchmark
        self.store = MemoryStore(base, load_existing=load_existing)
        self.extractor = MemoryExtractor(llm_client, benchmark=self.benchmark)

        embed_fn = _build_embed_fn(embed_model_path)
        self.retriever = MemoryRetriever(embed_fn)
        self._embed_fn = embed_fn

        logger.info(
            f"[RB] ReasoningBankMemory initialized: benchmark={benchmark}, "
            f"path={base}, n={len(self.store)}"
        )

    # ------------------------------------------------------------------
    # 检索接口
    # ------------------------------------------------------------------

    def retrieve_memory_str(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Optional[str]:
        """返回可直接注入 system prompt 的记忆字符串，库空时返回 None。"""
        k = top_k or self.top_k
        entries = self.retriever.retrieve(query, self.store, top_k=k)
        if not entries:
            return None
        return self._format_entries(entries)

    def _format_entries(self, entries: List[Dict[str, Any]]) -> str:
        # 只返回记忆条目本身，不加前缀说明（外层 benchmark template 已提供上下文标题）
        parts = []
        idx = 1
        for entry in entries:
            status = entry.get("status", "")
            status_tag = "✅ SUCCESS" if status == "Success" else "⚠️ FAILURE" if status == "Failure" else ""
            for item in entry.get("memory_items", []):
                title = item.get("title", "")
                content = item.get("content", "")
                tag = f" [{status_tag}]" if status_tag else ""
                parts.append(f"[Memory {idx}]{tag} **{title}**: {content}")
                idx += 1
        if not parts:
            return ""
        result = "\n".join(parts)
        if len(result) > self.max_memory_chars:
            result = result[: self.max_memory_chars] + "\n...[truncated]"
        return result

    # ------------------------------------------------------------------
    # 存储接口
    # ------------------------------------------------------------------

    def extract_and_store(
        self,
        query: str,
        trajectory: List[str],
        success: bool,
    ) -> None:
        """episode 结束后调用：LLM 提炼 + 存储 + 更新向量索引。"""
        if not self.allow_updates:
            return
        items = self.extractor.extract(query, trajectory, success)
        if not items:
            return

        entry: Dict[str, Any] = {
            "query": query,
            "trajectory": "\n".join(trajectory[:30]),  # 只存前 30 行防 JSON 爆炸
            "status": "Success" if success else "Failure",
            "memory_items": items,
        }

        # 以 query + items 拼接文本作为向量化的语义锚点
        anchor_text = query + " " + " ".join(
            it.get("title", "") + " " + it.get("content", "")
            for it in items
        )
        try:
            embedding = self._embed_fn(anchor_text[:1000])
            emb_arr = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(emb_arr)
            if norm > 0:
                emb_arr = emb_arr / norm
        except Exception as e:
            logger.warning(f"[RB] Embedding failed ({e}), storing without vector.")
            emb_arr = None

        self.store.append(entry, emb_arr)
        logger.info(
            f"[RB] Stored entry (success={success}, items={len(items)}, total={len(self.store)})"
        )

    def save(self) -> None:
        self.store.save()

    def __repr__(self) -> str:
        return (
            f"ReasoningBankMemory(benchmark={self.benchmark}, "
            f"path={self.store.store_dir}, n={len(self.store)})"
        )

    def __len__(self) -> int:
        return len(self.store)
