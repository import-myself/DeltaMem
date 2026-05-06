"""
synapse_memory.py  —  PRTree 通用 Synapse 记忆后端 (v2.0)
=========================================================
支持三个 benchmark（ALFWorld / ScienceWorld / Mind2web）共用。

核心功能
--------
1. 离线建库   SynapseMemoryStore.build(memory_path, specifiers, exemplars)
   - 用 OpenAI text-embedding-ada-002 将 specifiers 向量化存入 FAISS
   - 将 exemplars 保存为 exemplars.json / specifiers.json

2. 在线检索   store.retrieve_memory_str(query, top_k) -> str | None
   - FAISS 语义检索（无 FAISS 时降级为 difflib 文本相似度）
   - 返回 top-k exemplar 拼接好的 memory_str，可直接作为 external_memory_str 注入 agent

3. 在线写入   store.add_exemplar(specifier, exemplar)
   - 运行中将新轨迹写入记忆（同步更新 FAISS）
   - store.save() 持久化到磁盘

冷启动支持
----------
- 若 memory_path 目录为空，自动以空库初始化，不报错
- 第一条 add_exemplar 后即可检索（FAISS 或 difflib）

软依赖
------
- langchain-community + faiss-cpu（推荐安装）
  pip install langchain-community faiss-cpu openai
- 若未安装，自动降级为 difflib 文本相似度检索（无需向量数据库）
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 软依赖检测
# ---------------------------------------------------------------------------
_EMBEDDINGS_CLS = None
_FAISS = None

for _emb_path in (
    "langchain_community.embeddings.OpenAIEmbeddings",
    "langchain.embeddings.openai.OpenAIEmbeddings",
):
    try:
        mod, cls = _emb_path.rsplit(".", 1)
        import importlib
        _EMBEDDINGS_CLS = getattr(importlib.import_module(mod), cls)
        break
    except Exception:
        pass

for _faiss_path in (
    "langchain_community.vectorstores.FAISS",
    "langchain.vectorstores.FAISS",
):
    try:
        mod, cls = _faiss_path.rsplit(".", 1)
        import importlib
        _FAISS = getattr(importlib.import_module(mod), cls)
        break
    except Exception:
        pass

_HAS_FAISS = (_EMBEDDINGS_CLS is not None) and (_FAISS is not None)
if not _HAS_FAISS:
    logger.warning(
        "⚠️  langchain/faiss-cpu 未检测到，SynapseMemoryStore 将使用 difflib 文本相似度检索。\n"
        "   建议安装: pip install langchain-community faiss-cpu openai"
    )


# ---------------------------------------------------------------------------
# SynapseMemoryStore
# ---------------------------------------------------------------------------

class SynapseMemoryStore:
    """
    三个 benchmark 通用的 Synapse 记忆后端。

    Parameters
    ----------
    memory_path : str
        存储目录。包含 exemplars.json / specifiers.json 和（可选）FAISS 索引文件。
    top_k : int
        检索时默认返回的 exemplar 数量。
    embed_model : str
        OpenAI embedding 模型名（仅 _HAS_FAISS=True 时使用）。
    allow_updates : bool
        是否允许在线写入（默认 True）。
    """

    EXEMPLAR_FILE  = "exemplars.json"
    SPECIFIER_FILE = "specifiers.json"

    def __init__(
        self,
        memory_path: str,
        top_k: int = 1,
        embed_model: str = "text-embedding-ada-002",
        allow_updates: bool = True,
    ):
        self.memory_path   = Path(memory_path)
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.top_k         = top_k
        self.embed_model   = embed_model
        self.allow_updates = allow_updates
        self._lock         = threading.Lock()
        self._dirty        = False

        self._exemplars:  List        = []
        self._specifiers: List[str]   = []
        self._faiss_store             = None
        self._embedding               = None

        self._load()

    # ------------------------------------------------------------------
    # 离线建库（类方法）
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        memory_path: str,
        specifiers: List[str],
        exemplars: List,
        embed_model: str = "text-embedding-ada-002",
        top_k: int = 1,
    ) -> "SynapseMemoryStore":
        """
        从头构建 Synapse 记忆库（离线使用）。

        Parameters
        ----------
        memory_path : str
            输出目录（自动创建）。
        specifiers : List[str]
            每条 exemplar 对应的检索键文本。
        exemplars : List
            每条 exemplar 内容（message list / dict / str）。
        embed_model : str
            OpenAI embedding 模型。
        """
        assert len(specifiers) == len(exemplars), (
            f"specifiers ({len(specifiers)}) 与 exemplars ({len(exemplars)}) 数量不一致"
        )

        p = Path(memory_path)
        p.mkdir(parents=True, exist_ok=True)

        # 持久化 exemplars / specifiers
        with open(p / cls.EXEMPLAR_FILE,  "w", encoding="utf-8") as f:
            json.dump(exemplars,  f, ensure_ascii=False, indent=2)
        with open(p / cls.SPECIFIER_FILE, "w", encoding="utf-8") as f:
            json.dump(specifiers, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 已保存 {len(exemplars)} 条 exemplar → {p / cls.EXEMPLAR_FILE}")

        # 构建 FAISS 向量索引
        if _HAS_FAISS:
            logger.info(f"🔢 构建 FAISS 索引（{len(specifiers)} 条）…")
            embedding  = _EMBEDDINGS_CLS(model=embed_model)
            metadatas  = [{"idx": i} for i in range(len(specifiers))]
            store      = _FAISS.from_texts(
                texts=specifiers,
                embedding=embedding,
                metadatas=metadatas,
            )
            store.save_local(str(p))
            logger.info(f"✅ FAISS 索引已保存 → {p}")
        else:
            logger.warning("⚠️  FAISS 不可用，跳过向量索引构建（检索将使用 difflib）。")

        return cls(memory_path=memory_path, embed_model=embed_model, top_k=top_k)

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def _load(self):
        exemplar_file  = self.memory_path / self.EXEMPLAR_FILE
        specifier_file = self.memory_path / self.SPECIFIER_FILE

        if exemplar_file.exists():
            with open(exemplar_file, "r", encoding="utf-8") as f:
                self._exemplars = json.load(f)
            logger.info(f"📥 已加载 {len(self._exemplars)} 条 exemplar ← {exemplar_file}")
        else:
            self._exemplars = []
            logger.info(f"ℹ️  {exemplar_file} 不存在，以空库初始化（冷启动）")

        if specifier_file.exists():
            with open(specifier_file, "r", encoding="utf-8") as f:
                self._specifiers = json.load(f)
        else:
            self._specifiers = []

        # 加载 FAISS（可选）
        faiss_index = self.memory_path / "index.faiss"
        if _HAS_FAISS and faiss_index.exists():
            try:
                self._embedding   = _EMBEDDINGS_CLS(model=self.embed_model)
                self._faiss_store = _FAISS.load_local(
                    str(self.memory_path),
                    self._embedding,
                    allow_dangerous_deserialization=True,
                )
                logger.info(f"✅ FAISS 索引已加载 ← {self.memory_path}")
            except Exception as e:
                logger.warning(f"⚠️  FAISS 加载失败 ({e})，降级到 difflib 检索。")
                self._faiss_store = None
                self._embedding   = None
        else:
            self._faiss_store = None
            self._embedding   = None
            if _HAS_FAISS and self._exemplars:
                logger.info("ℹ️  无 FAISS 索引文件，当前使用 difflib 检索；"
                            "可运行 build_synapse_memory.py 建立向量索引以提升检索质量。")

    # ------------------------------------------------------------------
    # 在线检索
    # ------------------------------------------------------------------

    def retrieve_memory_str(self, query: str, top_k: Optional[int] = None) -> Optional[str]:
        """
        给定 query，返回 top-k exemplar 拼接后的记忆字符串。
        若库为空返回 None。
        """
        if not self._exemplars:
            return None

        k       = top_k or self.top_k
        indices = self._retrieve_indices(query, k)
        picked  = [self._exemplars[i] for i in indices if 0 <= i < len(self._exemplars)]
        if not picked:
            return None
        return self._format_exemplars(picked)

    def _retrieve_indices(self, query: str, k: int) -> List[int]:
        """返回 top-k exemplar 的下标列表。"""
        # ---- FAISS 向量检索 ----
        if self._faiss_store is not None:
            try:
                docs_scores = self._faiss_store.similarity_search_with_score(query, k=k)
                return [doc.metadata.get("idx", 0) for doc, _ in docs_scores]
            except Exception as e:
                logger.warning(f"FAISS 检索失败 ({e})，降级到 difflib。")

        # ---- difflib 文本相似度（降级） ----
        import difflib
        specs = self._specifiers if self._specifiers else [
            self._exemplar_to_text(e) for e in self._exemplars
        ]
        # get_close_matches cutoff=0 确保总能返回结果
        matches = difflib.get_close_matches(query, specs, n=k, cutoff=0.0)
        indices = []
        for m in matches:
            try:
                indices.append(specs.index(m))
            except ValueError:
                pass
        # 若 difflib 仍返回空（极少见），取前 k 条
        if not indices:
            indices = list(range(min(k, len(self._exemplars))))
        return indices

    # ------------------------------------------------------------------
    # 在线写入
    # ------------------------------------------------------------------

    def add_exemplar(
        self,
        specifier: str,
        exemplar,
        save_immediately: bool = False,
    ) -> None:
        """
        在线写入新的 exemplar（运行时调用）。

        Parameters
        ----------
        specifier : str
            该 exemplar 的检索键文本（任务描述 / 网站+任务 等）。
        exemplar : any
            exemplar 内容（message list / dict / str）。
        save_immediately : bool
            是否立即持久化（会写磁盘，建议每 N 条才调用一次 save() 以节省 I/O）。
        """
        if not self.allow_updates:
            return

        with self._lock:
            new_idx = len(self._exemplars)
            self._exemplars.append(exemplar)
            self._specifiers.append(specifier)

            # 同步更新 FAISS
            if self._faiss_store is not None:
                try:
                    self._faiss_store.add_texts(
                        texts=[specifier],
                        metadatas=[{"idx": new_idx}],
                    )
                except Exception as e:
                    logger.warning(f"FAISS add_texts 失败 ({e})。")
            elif _HAS_FAISS and self._embedding is not None and len(self._exemplars) == 1:
                # 第一次写入时尝试用 FAISS 初始化（冷启动场景）
                try:
                    self._faiss_store = _FAISS.from_texts(
                        texts=[specifier],
                        embedding=self._embedding,
                        metadatas=[{"idx": new_idx}],
                    )
                except Exception as e:
                    logger.warning(f"FAISS 冷启动初始化失败 ({e})。")

            self._dirty = True

        if save_immediately:
            self.save()

    def save(self) -> None:
        """持久化 exemplars / specifiers / FAISS 到 memory_path。"""
        with self._lock:
            if not self._dirty:
                return
            with open(self.memory_path / self.EXEMPLAR_FILE,  "w", encoding="utf-8") as f:
                json.dump(self._exemplars,  f, ensure_ascii=False, indent=2)
            with open(self.memory_path / self.SPECIFIER_FILE, "w", encoding="utf-8") as f:
                json.dump(self._specifiers, f, ensure_ascii=False, indent=2)
            if self._faiss_store is not None:
                self._faiss_store.save_local(str(self.memory_path))
            self._dirty = False
            logger.info(
                f"💾 SynapseMemoryStore 已保存：{len(self._exemplars)} 条 → {self.memory_path}"
            )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _format_exemplars(self, exemplars: List) -> str:
        """将 exemplar 列表格式化为可注入 prompt 的字符串。

        格式与 build_icl_messages + _build_prompt 保持一致：
          - 每条 exemplar 是 message list，按 role/content 拼接
          - i=0 (user首条): content + '\n'
          - i 奇数 (assistant): content + '\n'
          - i 偶数且非0 (user后续): content + '\n\n'
        """
        parts = []
        for ex in exemplars:
            if isinstance(ex, list):
                seg = ""
                for i, m in enumerate(ex):
                    if isinstance(m, dict):
                        content = m.get("content", "")
                    else:
                        content = str(m)
                    if i == 0:
                        seg += f"{content}\n"
                    elif i % 2 == 0:
                        seg += f"{content}\n\n"
                    else:
                        seg += f"{content}\n"
                parts.append(seg.rstrip())
            elif isinstance(ex, dict):
                parts.append(json.dumps(ex, ensure_ascii=False))
            else:
                parts.append(str(ex))
        return "\n\n".join(parts)

    def _exemplar_to_text(self, ex) -> str:
        """将 exemplar 转为检索文本（无 specifier 时的 fallback）。"""
        if isinstance(ex, list):
            return " ".join(
                m.get("content", "") if isinstance(m, dict) else str(m) for m in ex
            )
        if isinstance(ex, dict):
            return json.dumps(ex, ensure_ascii=False)
        return str(ex)

    # ------------------------------------------------------------------
    # 特殊方法
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._exemplars)

    def __repr__(self) -> str:
        return (
            f"SynapseMemoryStore("
            f"path={self.memory_path}, "
            f"n={len(self._exemplars)}, "
            f"faiss={'✓' if self._faiss_store else '✗'})"
        )
