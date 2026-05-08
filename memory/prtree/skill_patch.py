"""
Procedural Skill Patch (v1.0)
程序化技能补丁：ProceduralSkillPatch / SkillCache / SkillCompiler
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

CONSOLIDATION_THRESHOLD = 3


@dataclass
class ProceduralSkillPatch:
    source_node_id: str
    activation_condition: str   # 技能触发条件 (I_delta)
    execution_procedure: str    # 可执行动作/代码序列 (pi_delta)
    termination_condition: str  # 技能终止/交还控制权的条件 (beta_delta)


class SkillCache:
    """O(1) 技能补丁缓存：快思考路径"""

    def __init__(self):
        self.patches: List[ProceduralSkillPatch] = []

    def add_patch(self, patch: ProceduralSkillPatch) -> None:
        self.patches.append(patch)
        logger.info(f"[SkillCache] Added patch from node {patch.source_node_id[:8]}. Total: {len(self.patches)}")

    def check_match(self, state: Dict[str, Any]) -> Optional[ProceduralSkillPatch]:
        """判断 state 是否命中缓存中任意 patch 的 activation_condition（轻量关键词重叠）"""
        task_desc = str(state.get("task", "")).lower()
        env_desc = str(state.get("env", "")).lower()
        query = (task_desc + " " + env_desc).strip()
        if not query or not self.patches:
            return None

        query_words = set(query.split())
        best_patch = None
        best_score = 0.0
        for patch in self.patches:
            cond_words = set(patch.activation_condition.lower().split())
            if not cond_words:
                continue
            overlap = len(cond_words & query_words)
            score = overlap / max(len(cond_words), 1)
            if score > best_score:
                best_score = score
                best_patch = patch

        # 30% 词重叠作为命中阈值
        if best_score >= 0.30:
            logger.info(f"[SkillCache] Match (score={best_score:.2f}) → node {best_patch.source_node_id[:8]}")
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
        self.patches = [
            ProceduralSkillPatch(
                source_node_id=d["source_node_id"],
                activation_condition=d["activation_condition"],
                execution_procedure=d["execution_procedure"],
                termination_condition=d["termination_condition"],
            )
            for d in data
        ]

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
            if not data or not all(k in data for k in ("activation_condition", "execution_procedure", "termination_condition")):
                logger.warning(f"[SkillCompiler] Failed to parse compiled patch. Raw: {response[:200]}")
                return None

            patch = ProceduralSkillPatch(
                source_node_id=source_node_id,
                activation_condition=data["activation_condition"],
                execution_procedure=data["execution_procedure"],
                termination_condition=data["termination_condition"],
            )
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
