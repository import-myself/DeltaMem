"""
Skill / Knowledge Consolidation
将成功命中的经验链条编译为单一固化节点，写回对应树作为新 ROOT 节点。
- SkillCompiler:     TaskTree 用，输出程序性 Skill
- KnowledgeCompiler: EnvTree 用，输出声明性 Knowledge
"""

import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

_SKILL_COMPILE_PROMPT = """You are a Skill Compiler. Consolidate the following experience chain (Base Skill + Skill Deltas, root → deepest match) into a single self-contained Base Skill.

Experience Chain:
{experience_chain}

Output a single JSON that is fully self-contained — no references to "above" or "the base skill":
{{
    "activation_condition": "Applicable when [task_type_tag] ...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}"""

_KNOWLEDGE_COMPILE_PROMPT = """You are a Knowledge Compiler. Consolidate the following environment knowledge chain (Base Knowledge + Knowledge Updates, root → deepest match) into a single self-contained Base Environment Knowledge entry.

Knowledge Chain:
{experience_chain}

Output a single JSON that is fully self-contained — no references to "above" or "the base knowledge":
{{
    "activation_condition": "Applicable in [environment_type] environments where ...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}"""


class SkillCompiler:
    """将任务树经验链条编译为自包含的 Base Skill（返回 dict，调用方写回 task_tree）"""

    def compile_to_skill(
        self,
        node_chain_texts: List[str],
        llm_client,
    ) -> Optional[Dict[str, Any]]:
        if not llm_client:
            logger.warning("[SkillCompiler] No LLM client, skipping.")
            return None

        chain_text = "\n\n---\n\n".join(node_chain_texts)
        prompt = _SKILL_COMPILE_PROMPT.format(experience_chain=chain_text)

        try:
            response = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
            data = _parse_json(response)
            if data and all(k in data for k in ("activation_condition", "execution_procedure", "termination_condition")):
                return data
            logger.warning(f"[SkillCompiler] Invalid response: {response[:200]}")
            return None
        except Exception as e:
            logger.error(f"[SkillCompiler] Error: {e}")
            return None


class KnowledgeCompiler:
    """将环境树经验链条编译为自包含的 Base Environment Knowledge（返回 dict，调用方写回 env_tree）"""

    def compile_to_knowledge(
        self,
        node_chain_texts: List[str],
        llm_client,
    ) -> Optional[Dict[str, Any]]:
        if not llm_client:
            logger.warning("[KnowledgeCompiler] No LLM client, skipping.")
            return None

        chain_text = "\n\n---\n\n".join(node_chain_texts)
        prompt = _KNOWLEDGE_COMPILE_PROMPT.format(experience_chain=chain_text)

        try:
            response = llm_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
            data = _parse_json(response)
            if data and all(k in data for k in ("activation_condition", "execution_procedure", "termination_condition")):
                return data
            logger.warning(f"[KnowledgeCompiler] Invalid response: {response[:200]}")
            return None
        except Exception as e:
            logger.error(f"[KnowledgeCompiler] Error: {e}")
            return None


def _parse_json(response: str) -> Optional[Dict]:
    raw = response.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(raw, strict=False)
    except Exception:
        return None
