"""
Procedural Skill Patch (v2.0 - Embedding Matching)
程序化技能补丁：ProceduralSkillPatch / SkillCache / SkillCompiler

v2.0: SkillCache 改用 VectorRetriever embedding 余弦相似度做召回，
      关键词重叠保留为无 retriever 时的降级路径。
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from common.retriever import VectorRetriever

logger = logging.getLogger(__name__)

CONSOLIDATION_THRESHOLD = 3
SKILL_CACHE_MATCH_THRESHOLD = 0.82  # 高于 TaskTree base_threshold(0.80)，保证精准触发


@dataclass
class ProceduralSkillPatch:
    source_node_id: str
    activation_condition: str   # 技能触发条件 (I_delta)
    execution_procedure: str    # 可执行动作/代码序列 (pi_delta)
    termination_condition: str  # 技能终止/交还控制权的条件 (beta_delta)


class SkillCache:
    """
    O(1) 技能补丁缓存：快思考路径。

    召回策略：
      主路径 — embedding 余弦相似度（需传入 VectorRetriever）
        query = task_description + env_description
        对所有 patch 的 activation_condition embedding 做批量余弦相似度
        返回 score >= SKILL_CACHE_MATCH_THRESHOLD 的最高分 patch

      降级路径 — 关键词重叠率（retriever 不可用时）
        score = |words(activation_condition) ∩ words(query)| / |words(activation_condition)|
        阈值 0.30
    """

    def __init__(
        self,
        retriever: Optional["VectorRetriever"] = None,
        match_threshold: float = SKILL_CACHE_MATCH_THRESHOLD,
    ):
        self.patches: List[ProceduralSkillPatch] = []
        self._embeddings: List[Optional[np.ndarray]] = []  # 与 patches 一一对应
        self.retriever = retriever
        self.match_threshold = match_threshold

    def add_patch(self, patch: ProceduralSkillPatch) -> None:
        """添加 patch，同时预计算 activation_condition 的 embedding"""
        self.patches.append(patch)
        emb = None
        if self.retriever is not None:
            try:
                emb = self.retriever.encode(patch.activation_condition)
            except Exception as e:
                logger.warning(f"[SkillCache] Failed to encode patch embedding: {e}")
        self._embeddings.append(emb)
        logger.info(
            f"[SkillCache] Added patch from node {patch.source_node_id[:8]}. "
            f"Total={len(self.patches)} embedding={'yes' if emb is not None else 'no'}"
        )

    def check_match(self, state: Dict[str, Any]) -> Optional[ProceduralSkillPatch]:
        """检索最匹配的 Skill Patch"""
        if not self.patches:
            return None

        task_desc = str(state.get("task", ""))
        env_desc = str(state.get("env", ""))
        query = (task_desc + " " + env_desc).strip()
        if not query:
            return None

        # ── 主路径：embedding 余弦相似度 ──
        valid = [(i, e) for i, e in enumerate(self._embeddings) if e is not None]
        if self.retriever is not None and valid:
            return self._match_by_embedding(query, valid)

        # ── 降级路径：关键词重叠率 ──
        return self._match_by_keywords(query)

    def _match_by_embedding(
        self, query: str, valid: List[tuple]
    ) -> Optional[ProceduralSkillPatch]:
        try:
            query_emb = self.retriever.encode(query)
            indices = [i for i, _ in valid]
            emb_matrix = np.stack([e for _, e in valid])          # (N, dim)
            scores = self.retriever._batch_cosine_similarity(query_emb, emb_matrix)

            best_j = int(np.argmax(scores))
            best_score = float(scores[best_j])
            best_idx = indices[best_j]

            if best_score >= self.match_threshold:
                patch = self.patches[best_idx]
                logger.info(
                    f"[SkillCache] Embedding hit (cos={best_score:.3f}) "
                    f"→ node {patch.source_node_id[:8]}"
                )
                return patch

            logger.debug(f"[SkillCache] Best embedding score {best_score:.3f} < {self.match_threshold}, no hit.")
            return None
        except Exception as e:
            logger.warning(f"[SkillCache] Embedding match failed ({e}), falling back to keywords.")
            return self._match_by_keywords(query)

    def _match_by_keywords(self, query: str) -> Optional[ProceduralSkillPatch]:
        query_words = set(query.lower().split())
        best_score, best_patch = 0.0, None
        for patch in self.patches:
            cond_words = set(patch.activation_condition.lower().split())
            if not cond_words:
                continue
            score = len(cond_words & query_words) / len(cond_words)
            if score > best_score:
                best_score, best_patch = score, patch
        if best_score >= 0.30:
            logger.info(f"[SkillCache] Keyword hit (overlap={best_score:.2f}) → node {best_patch.source_node_id[:8]}")
            return best_patch
        return None

    def to_list(self) -> List[Dict]:
        return [
            {
                "source_node_id": p.source_node_id,
                "activation_condition": p.activation_condition,
                "execution_procedure": p.execution_procedure,
                "termination_condition": p.termination_condition,
            }
            for p in self.patches
        ]

    def from_list(self, data: List[Dict]) -> None:
        """从序列化列表恢复 patch，重新计算 embedding"""
        self.patches = []
        self._embeddings = []
        for d in data:
            patch = ProceduralSkillPatch(
                source_node_id=d["source_node_id"],
                activation_condition=d["activation_condition"],
                execution_procedure=d["execution_procedure"],
                termination_condition=d["termination_condition"],
            )
            self.add_patch(patch)

    def __len__(self) -> int:
        return len(self.patches)


_COMPILE_PROMPT = """You are a high-level Skill Compiler. Consolidate the following experience residual chain into a single, self-contained Procedural Skill Patch.

Experience Chain (Base Skill + Skill Deltas):
{experience_chain}

**Core Constraint: Self-Contained**
The compiled patch CANNOT depend on undeclared external context or implicit state:
- `activation_condition`: ALL environment prerequisites required for independent operation
- `execution_procedure`: Self-contained steps (no undefined variables; UI actions must specify exact element descriptions)
- `termination_condition`: Explicit exit / handoff condition

Discard all historical reflection reasoning. Output ONLY the final executable JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}"""


class SkillCompiler:
    """将情景残差链条异步编译为自包含的 ProceduralSkillPatch"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def compile(
        self,
        node_chain_texts: List[str],
        skill_cache: SkillCache,
        source_node_id: str,
        llm_client=None,
    ) -> Optional[ProceduralSkillPatch]:
        client = llm_client or self.llm_client
        if not client:
            logger.warning("[SkillCompiler] No LLM client available, skipping compilation.")
            return None

        chain_text = "\n\n---\n\n".join(node_chain_texts)
        prompt = _COMPILE_PROMPT.format(experience_chain=chain_text)

        try:
            response = client.chat([{"role": "user", "content": prompt}], temperature=0.0)
            data = _parse_skill_json(response)
            if not data or not all(
                k in data for k in ("activation_condition", "execution_procedure", "termination_condition")
            ):
                logger.warning(f"[SkillCompiler] Invalid compiled patch. Raw: {response[:200]}")
                return None

            patch = ProceduralSkillPatch(
                source_node_id=source_node_id,
                activation_condition=data["activation_condition"],
                execution_procedure=data["execution_procedure"],
                termination_condition=data["termination_condition"],
            )
            # add_patch 内部会自动计算 embedding
            skill_cache.add_patch(patch)
            logger.info(f"[SkillCompiler] Compiled patch for node {source_node_id[:8]}")
            return patch

        except Exception as e:
            logger.error(f"[SkillCompiler] Compilation error: {e}")
            return None


def _parse_skill_json(response: str) -> Optional[Dict]:
    raw = response.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(raw, strict=False)
    except Exception:
        pass
    return None
