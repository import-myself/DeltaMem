"""
Dual PR-Tree Agent for WebShop

参考 ALFWorld_New/agent_alfworld_dual.py，适配 WebShop 的文本式 env：
- env.reset(session=session_id) → env.observation
- env.step("search[...]" | "click[...]") → (obs, reward, done, info)
- success 判据: reward == 1.0
- 双树拆分:
    * env_description = 产品类目短语（去掉"i need a/i'm looking for ..."等填充后取主词）
    * task_description = 完整 instruction（含价格/规格约束）
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

# 注意：必须先 import memory.prtree 让其 __init__ 完成，再 import common.retriever，
# 否则会触发 common.retriever ↔ memory.prtree.dual_tree_manager 的循环 import。
from memory.prtree.dual_tree_manager import DualTreeMemory
from memory.prtree.dual_memory_reader import DualMemoryReader
from memory.prtree.dual_memory_writer import DualMemoryWriter
from memory.prtree.memory_node import MemoryNode, ResultStatus
from common.retriever import VectorRetriever
from common.llm_client import LLMClient

from dual_prompt import (
    EnvTree_Prompt_Map,
    MEMORY_HEADERS,
    PROMPT_WITH_ICL_TEMPLATE,
    PROMPT_WITH_ICL_TEMPLATE_DUAL_MEMORY,
    TaskTree_Prompt_Map,
    get_env_prompt_key,
    get_task_prompt_key,
    webshop_instruction,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 文本工具
# ---------------------------------------------------------------------------

_INSTRUCTION_RE = re.compile(r"Instruction:\s*\[SEP\]\s*(.+?)(?:\s*\[SEP\]|$)")
_ACTION_RE      = re.compile(r"Action:\s*(.+?)(?:\n|$)", re.DOTALL)

# Product category 抽取：LLM 1-shot 把任意 instruction 映射为 ≤4 词的产品类目短语
# 例：
#   "Find me machine wash men's t-shirts with long sleeve ..." → "men's t-shirts"
#   "i need a long lasting 6.76 fl oz bottle of l'eau d'issey ..." → "perfume"
#   "heavy duty iphone accessories for wireless charging..."     → "iphone accessories"
# 失败时回退到启发式（仅作兜底，不影响主路径）。
_CATEGORY_LLM_PROMPT = """You are a product category extractor for WebShop instructions.
Given a user's shopping instruction, output ONLY a short noun phrase (1-4 words, lowercase) that names the PRODUCT CATEGORY being bought.

Rules:
- Output the product TYPE, not attributes. e.g. "men's t-shirt", "perfume", "phone case", "running shoes", "coffee table".
- NEVER output washing/material/fit/size descriptors as the category (e.g. NEVER output "machine wash", "slim fit", "wash cold", "long sleeve" as standalone category).
- If the instruction names a sub-category (e.g. "tuxedo shirts" vs "dress shirts"), prefer the higher-level type ("dress shirts" → "dress shirts"; "tuxedo shirts" → "dress shirts" if both are men's formal shirts is too broad — keep the noun the user used: "tuxedo shirts").
- Drop brand names, sizes, colors, prices.
- Output the phrase ONLY. No quotes, no explanation, no JSON.

Examples:
Instruction: Find me machine wash men's t-shirts with long sleeve with color: black, and size: xx-large, and price lower than 50.00 dollars
men's t-shirts

Instruction: i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars
perfume

Instruction: heavy duty iphone accessories for wireless charging with color: electroplate black, and price lower than 50.00 dollars
iphone accessories

Instruction: i'm looking for a coffee table with steel frame under 200 dollars
coffee table

Instruction: hand wash women's sweaters with long sleeve and v-neck
women's sweaters

Now extract for:
Instruction: {instruction}
"""

# 启发式兜底：去掉常见开头填充 + 已知属性词（仅用于 LLM 不可用时）
_FILLER_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"i\s+(?:would\s+like|need|want|require|am\s+looking\s+for|am\s+searching\s+for)|"
    r"i'?d\s+like|"
    r"i'?m\s+(?:looking\s+for|searching\s+for|after)|"
    r"please\s+(?:find|get|buy)|"
    r"find\s+me|give\s+me|"
    r"looking\s+for|buy\s+me"
    r")\s+(?:a|an|the|some|one|two|three)?\s*",
    re.IGNORECASE,
)
# 已知属性/状语词，从结果短语首部去掉（如 "machine wash men's t-shirts" → "men's t-shirts"）
_ATTRIBUTE_PREFIX_RE = re.compile(
    r"^(?:machine\s+wash|hand\s+wash|wash\s+cold|dry\s+clean|"
    r"slim\s+fit|loose\s+fit|classic\s+fit|"
    r"heavy\s+duty|long\s+lasting|fast\s+drying|"
    r"long\s+sleeve|short\s+sleeve|sleeveless|"
    r"butt\s+lifting|knee\s+high|open\s+toe|"
    r"high\s+waist|v[-\s]?neck|crew\s+neck|"
    r"organic|natural|fresh|premium)\s+",
    re.IGNORECASE,
)


def extract_instruction(obs: str) -> str:
    """从初始 observation 中切出 instruction 文本。"""
    m = _INSTRUCTION_RE.search(obs or "")
    return m.group(1).strip() if m else (obs or "").strip()[:300]


def _fallback_extract_category(instruction: str, max_len: int = 60) -> str:
    """启发式兜底：仅在 LLM 抽取失败时使用。"""
    if not instruction:
        return "webshop product"
    text = _FILLER_PREFIX_RE.sub("", instruction).strip()
    # 反复剥离已知属性词前缀
    prev = None
    while prev != text:
        prev = text
        text = _ATTRIBUTE_PREFIX_RE.sub("", text).strip()
    # 在第一个逗号/with/under 处切断
    for stop in [",", " under ", " below ", " with price ", " priced "]:
        i = text.lower().find(stop)
        if i > 0:
            text = text[:i]
            break
    text = text.strip(",.;:!? ").lower()
    return (text or instruction.lower())[:max_len]


def extract_product_category(
    instruction: str,
    llm_client=None,
    max_len: int = 60,
) -> str:
    """
    用 LLM 把 instruction 映射为产品类目短语作为 env_description。
    LLM 调用失败或不传 llm_client 时回退到启发式兜底。

    设计：每个 episode 仅调用 1 次（很短的 prompt + 输出 ≤4 词，成本可忽略）。
    """
    if not instruction:
        return "webshop product"

    if llm_client is not None:
        try:
            resp = llm_client.chat(
                [{"role": "user", "content": _CATEGORY_LLM_PROMPT.format(instruction=instruction[:400])}],
                temperature=0.0,
                max_tokens=24,
            )
            # 取第一行非空文本，去掉引号/JSON 标记
            line = resp.strip().splitlines()[0] if resp.strip() else ""
            line = line.strip().strip('"').strip("'").strip("`").lower()
            # 去掉可能的 "category:" 前缀
            for prefix in ("category:", "product category:", "answer:"):
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
            # 简单校验：长度合理 + 不是空 + 不含明显的属性词单独占位
            if line and 2 <= len(line) <= max_len and " " in line or (line and len(line) <= 30):
                return line[:max_len]
        except Exception as e:
            logger.warning(f"[category] LLM extract failed ({e}); fallback to heuristic.")

    return _fallback_extract_category(instruction, max_len)


def parse_action(llm_output: str) -> Optional[str]:
    m = _ACTION_RE.search(llm_output)
    if not m:
        return None
    action = m.group(1).strip()
    if not (action.startswith("search[") or action.startswith("click[")):
        return None
    if not action.endswith("]"):
        return None
    return action


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

# 与 ETO baseline 一致的 1-shot ICL 示例（穿插在 user 角色之间）
ICL_EXAMPLE: List[Dict[str, str]] = [
    {
        "role": "user",
        "content": "Observation:\nWebShop [SEP] Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Search",
    },
    {
        "role": "assistant",
        "content": "Thought: I think I should use the search bar to look for the product I need.\nAction: search[l'eau d'issey 6.76 fl oz bottle price < 100.00]",
    },
    {
        "role": "user",
        "content": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B000VOHH8I [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] $64.98",
    },
    {
        "role": "assistant",
        "content": "Thought: I think I should click on the product I need, which is B000VOHH8I.\nAction: click[B000VOHH8I]",
    },
    {
        "role": "user",
        "content": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now",
    },
    {
        "role": "assistant",
        "content": "Thought: I think I should click on the '6.76 fl oz (pack of 1)' option to select the size I need.\nAction: click[6.76 fl oz (pack of 1)]",
    },
    {
        "role": "user",
        "content": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now",
    },
    {
        "role": "assistant",
        "content": "Thought: I think I should click on the 'Buy Now' button to purchase the product.\nAction: click[Buy Now]",
    },
]


def _format_icl_block(icl_messages: List[Dict[str, str]]) -> str:
    """与 ALFWorld build_prompt 一致：把 ICL 消息序列化为单段文本。"""
    if not icl_messages:
        return ""
    lines = []
    for i, msg in enumerate(icl_messages):
        if i == 0:
            lines.append(msg["content"])
        elif i % 2 == 0:
            lines.append("")
            lines.append(msg["content"])
        else:
            lines.append(msg["content"])
    return "\n".join(lines) + "\n"


class DualTreeWebShopAgent:
    """
    基于 PR-Tree v4.0 的 WebShop 双树 Agent。

    Workflow:
      1. Split:     instruction → (env_description=产品类目, task_description=完整instruction)
      2. Retrieve:  分别在 Task / Env 双树上做 DFS 检索
      3. Act:       基于融合记忆与 WebShop env 多步交互（search / click）
      4. Reflect:   分别生成 Task / Env 反思
      5. Evolve:    根/残差节点写入 + consolidation 触发
    """

    def __init__(
        self,
        agent_name: str,
        llm_client: LLMClient,
    ):
        self.agent_name = agent_name
        self.llm_client = llm_client

        self.retriever   = VectorRetriever()
        self.dual_memory = DualTreeMemory(self.retriever)
        self.reader      = DualMemoryReader(self.dual_memory)
        self.writer      = DualMemoryWriter(self.dual_memory)

        logger.info(f"✅ {agent_name} initialized with Dual PR-Tree (WebShop)")

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run_episode(
        self,
        env: Any,
        session_id: int,
        max_steps: int = 15,
        no_memory: bool = False,
        no_prtree_update: bool = False,
        external_memory_str: Optional[str] = None,
        memory_mode: str = "both",
        memory_type: str = "prtree",
        episode_idx: int = -1,
    ) -> Dict[str, Any]:
        """跑一个 episode；返回 result dict + 完整 messages。"""

        # ---------- Phase 0: reset + split ----------
        env.reset(session=str(session_id))
        init_obs = env.observation
        instruction = extract_instruction(init_obs)
        # 用 LLM 抽产品类目（1 次小调用，回退到启发式兜底）
        env_description = extract_product_category(instruction, llm_client=self.llm_client)
        task_description = instruction

        logger.info(f"\n{'=' * 60}")
        logger.info(f"🛒 session={session_id}")
        logger.info(f"🎯 Task: {task_description[:160]}")
        logger.info(f"🏷  Category: {env_description}")

        # ---------- Phase 1: Dual retrieval ----------
        if no_memory:
            task_path, env_path = [], []
            task_mem_used = env_mem_used = False
            memory_context_str = external_memory_str if external_memory_str else None
            mem_used = bool(memory_context_str)
            if memory_context_str:
                logger.info(f"🔌 External memory injected in no-prtree mode ({len(memory_context_str)} chars).")
            else:
                logger.info("🚫 Memory disabled (baseline): skipping retrieval.")
        elif external_memory_str is not None:
            task_path, env_path = [], []
            memory_context_str = external_memory_str
            task_mem_used = env_mem_used = False
            mem_used = True
            logger.info(f"🔌 External memory injected ({len(external_memory_str)} chars), skipping PRTree retrieval.")
        else:
            paths = self.dual_memory.retrieve_dual_paths_flat(task_description, env_description)
            task_path = paths["task_path"]
            env_path  = paths["env_path"]

            if memory_mode == "task_only":
                t = self.reader.render_task_memory(task_path)
                memory_context_str = (
                    "# Task Strategy Memory\n"
                    "The following are experiences from similar WebShop tasks that may help guide your search/click strategy:\n\n"
                    f"{t}"
                ) if t else None
                task_mem_used = not self.reader._is_empty_path(task_path)
                env_mem_used  = False
            elif memory_mode == "env_only":
                e = self.reader.render_env_memory(env_path)
                memory_context_str = (
                    "# Site UI Knowledge Memory\n"
                    "The following are WebShop UI patterns observed for this product category:\n\n"
                    f"{e}"
                ) if e else None
                task_mem_used = False
                env_mem_used  = not self.reader._is_empty_path(env_path)
            else:
                memory_context_str = self.reader.get_dual_narrative_context(
                    task_description, env_description, task_path=task_path, env_path=env_path
                )
                task_mem_used = not self.reader._is_empty_path(task_path)
                env_mem_used  = not self.reader._is_empty_path(env_path)

            mem_used = task_mem_used or env_mem_used
            logger.info(
                f"💡 TaskHit={task_mem_used} (depth={len(task_path)}), "
                f"EnvHit={env_mem_used} (depth={len(env_path)})"
            )

        # ---------- Phase 2: Execution ----------
        messages = self._build_prompt(
            task_text=task_description,
            initial_obs=init_obs,
            memory_context_str=memory_context_str,
            memory_type=memory_type,
        )

        trajectory: List[str] = [f"Observation: {self._compact(init_obs)}"]
        obs = init_obs
        success = False
        done = False
        steps = 0
        final_reward = 0.0
        invalid_in_a_row = 0

        for step in range(max_steps):
            try:
                llm_response = self.llm_client.chat(messages, temperature=0.0, max_tokens=300)
            except Exception as e:
                logger.error(f"  [step {step}] LLM error: {e}")
                break

            action = parse_action(llm_response)
            messages.append({"role": "assistant", "content": llm_response})

            if action is None:
                invalid_in_a_row += 1
                trajectory.append(f"LLMRaw: {llm_response[:200]}")
                if invalid_in_a_row >= 3:
                    logger.info("  ... 3 invalid in a row, abort")
                    break
                err = "Observation: Invalid format. The input must contain 'Action: '"
                messages.append({"role": "user", "content": err})
                trajectory.append(err)
                steps += 1
                continue
            invalid_in_a_row = 0

            trajectory.append(f"Action: {action}")
            try:
                obs, reward, done, info = env.step(action)
            except AssertionError:
                obs = "Invalid action!"
                reward = 0.0
                done = False

            trajectory.append(f"Observation: {self._compact(obs)}")
            messages.append({"role": "user", "content": f"Observation:\n{obs}"})

            final_reward = reward
            steps += 1
            short_obs = self._compact(obs)
            logger.info(f"  [step {step}] action={action!r} reward={reward:.3f} done={done} obs={short_obs!r}")
            if done:
                break

        success = (final_reward == 1.0)
        logger.info(f"{'✅ SUCCESS' if success else '❌ FAIL'} reward={final_reward:.3f} steps={steps}")

        # ---------- Phase 3 + 4: Reflect + Evolve ----------
        result_dict: Dict[str, Any] = {
            "session_id":      session_id,
            "instruction":     instruction,
            "env_description": env_description,
            "reward":          final_reward,
            "steps":           steps,
            "done":            done,
            "success":         success,
            "memory_used":     mem_used,
            "task_memory_used": task_mem_used,
            "env_memory_used":  env_mem_used,
            "task_retrieval_length": len(task_path) - 1 if task_mem_used else 0,
            "env_retrieval_length":  len(env_path)  - 1 if env_mem_used  else 0,
            "trajectory":      trajectory,
        }

        if no_memory or no_prtree_update:
            if no_prtree_update:
                logger.info("🔌 External memory mode: skipping PRTree reflection & update.")
            result_dict.update({
                "task_reflection": {}, "env_reflection": {},
                "task_node_id": None,  "env_node_id":  None,
            })
        else:
            task_reflection, env_reflection = self._generate_dual_reflection(
                task_description=task_description,
                env_description=env_description,
                success=success,
                steps=steps,
                max_steps=max_steps,
                trajectory=trajectory,
                task_path=task_path,
                env_path=env_path,
            )

            task_status = ResultStatus.SUCCESS if success else ResultStatus.FAILURE
            task_skip = task_reflection.get("skip", False)
            env_skip  = env_reflection.get("skip",  False)

            task_node = None
            env_node  = None

            if not task_skip:
                task_node = self.writer.write_task_experience(
                    scenario_description=task_description,
                    skill=task_reflection,
                    result_status=task_status,
                    retrieved_path=task_path,
                    episode_idx=episode_idx,
                )
            else:
                logger.info("⏭ Task: existing skill covers this episode, skipping node creation.")

            if not env_skip:
                env_node = self.writer.write_env_experience(
                    scenario_description=env_description,
                    skill=env_reflection,
                    result_status=ResultStatus.SUCCESS,  # env 知识统一以 SUCCESS 存储
                    retrieved_path=env_path,
                    episode_idx=episode_idx,
                )
            else:
                logger.info("⏭ Env: no new site knowledge, skipping node creation.")

            # consolidation：成功时累加最深检索节点的 success_count
            if success:
                deepest_task = next(
                    (n for n in reversed(task_path)
                     if "GLOBAL_ROOT_PLACEHOLDER" not in n.payload.get("scenario_description", "")),
                    None,
                )
                if deepest_task is not None:
                    deepest_task.meta["success_count"] = deepest_task.meta.get("success_count", 0) + 1
                    self.dual_memory.trigger_consolidation_check(deepest_task, self.llm_client, tree_type="task")

                deepest_env = next(
                    (n for n in reversed(env_path)
                     if "GLOBAL_ROOT_PLACEHOLDER" not in n.payload.get("scenario_description", "")),
                    None,
                )
                if deepest_env is not None:
                    deepest_env.meta["success_count"] = deepest_env.meta.get("success_count", 0) + 1
                    self.dual_memory.trigger_consolidation_check(deepest_env, self.llm_client, tree_type="env")

            result_dict.update({
                "task_reflection": task_reflection,
                "env_reflection":  env_reflection,
                "task_node_id":    task_node.node_id if task_node else None,
                "env_node_id":     env_node.node_id  if env_node  else None,
                "task_skip":       task_skip,
                "env_skip":        env_skip,
            })

        return result_dict, messages

    # ------------------------------------------------------------------
    # Prompt 构造
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        task_text: str,
        initial_obs: str,
        memory_context_str: Optional[str],
        memory_type: str = "prtree",
    ) -> List[Dict[str, str]]:
        """与 ALFWorld 一致：单条 user 消息，把 instruction + ICL + memory + task 全塞进去。

        WebShop 后续的 obs/action 通过 append 形式继续追加。
        """
        icl_text = _format_icl_block(ICL_EXAMPLE)

        # task 字段：用初始 observation 作为本回合开局（与 ICL 例子一致）
        task_block = f"Observation:\n{initial_obs}"

        if memory_context_str:
            header = MEMORY_HEADERS.get(memory_type, MEMORY_HEADERS["prtree"])
            full_content = PROMPT_WITH_ICL_TEMPLATE_DUAL_MEMORY.format(
                instruction=webshop_instruction,
                examples=icl_text,
                memory_header=header,
                memory_context=memory_context_str,
                task=task_block,
            )
        else:
            full_content = PROMPT_WITH_ICL_TEMPLATE.format(
                instruction=webshop_instruction,
                examples=icl_text,
                task=task_block,
            )
        return [{"role": "user", "content": full_content}]

    # ------------------------------------------------------------------
    # 反思
    # ------------------------------------------------------------------

    def _generate_dual_reflection(
        self,
        task_description: str,
        env_description: str,
        success: bool,
        steps: int,
        max_steps: int,
        trajectory: List[str],
        task_path: List[MemoryNode],
        env_path: List[MemoryNode],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        traj_str = "\n".join(trajectory)

        # ---- Task tree reflection ----
        task_is_root = self.reader._is_empty_path(task_path)
        task_compare = self.reader.render_task_path_for_reflection(task_path) if not task_is_root else ""
        task_template = TaskTree_Prompt_Map[get_task_prompt_key(task_is_root, success)]

        if task_is_root:
            task_prompt = task_template.format(
                env_description=env_description,
                task_description=task_description,
                steps=steps,
                max_steps=max_steps,
                trajectory=traj_str,
            )
        else:
            task_prompt = task_template.format(
                env_description=env_description,
                task_description=task_description,
                retrieved_task_memory=task_compare,
                steps=steps,
                max_steps=max_steps,
                trajectory=traj_str,
            )

        task_resp = self.llm_client.chat(
            [{"role": "user", "content": task_prompt}],
            temperature=0.0, max_tokens=4096,
        )
        task_reflection = self._parse_json_response(task_resp)
        # 根节点成功时直接存储原始轨迹，无需 LLM 重新生成
        if task_is_root and success and isinstance(task_reflection, dict):
            task_reflection["trajectory"] = traj_str

        # ---- Env tree reflection ----
        env_is_root = self.reader._is_empty_path(env_path)
        env_compare = self.reader.render_env_path_for_reflection(env_path) if not env_is_root else ""
        env_template = EnvTree_Prompt_Map[get_env_prompt_key(env_is_root, success)]
        result_str = "SUCCESS" if success else "FAILURE"

        if env_is_root:
            env_prompt = env_template.format(
                env_description=env_description,
                task_description=task_description,
                result=result_str,
                steps=steps,
                max_steps=max_steps,
                trajectory=traj_str,
            )
        else:
            env_prompt = env_template.format(
                env_description=env_description,
                task_description=task_description,
                result=result_str,
                retrieved_env_memory=env_compare,
                steps=steps,
                max_steps=max_steps,
                trajectory=traj_str,
            )

        env_resp = self.llm_client.chat(
            [{"role": "user", "content": env_prompt}],
            temperature=0.0, max_tokens=4096,
        )
        env_reflection = self._parse_json_response(env_resp)

        return task_reflection, env_reflection

    # ------------------------------------------------------------------
    # JSON 健壮解析（与 ALFWorld 实现一致）
    # ------------------------------------------------------------------

    @staticmethod
    def _escape_json_strings(text: str) -> str:
        out, in_str, i = [], False, 0
        while i < len(text):
            ch = text[i]
            if ch == "\\" and in_str:
                out.append(ch)
                i += 1
                if i < len(text):
                    out.append(text[i])
            elif ch == '"':
                out.append(ch)
                in_str = not in_str
            elif in_str and ch == "\n":
                out.append("\\n")
            elif in_str and ch == "\r":
                out.append("\\r")
            elif in_str and ch == "\t":
                out.append("\\t")
            else:
                out.append(ch)
            i += 1
        return "".join(out)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        raw = response.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(raw, strict=False)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(self._escape_json_strings(raw), strict=False)
        except json.JSONDecodeError:
            pass

        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
        try:
            return json.loads(self._escape_json_strings(cleaned), strict=False)
        except json.JSONDecodeError:
            pass

        field = r'"((?:[^"\\]|\\.)*)"'
        ac = re.search(r'"activation_condition"\s*:\s*' + field, raw, re.DOTALL)
        ep = re.search(r'"execution_procedure"\s*:\s*'  + field, raw, re.DOTALL)
        tc = re.search(r'"termination_condition"\s*:\s*' + field, raw, re.DOTALL)
        if ac and ep:
            return {
                "activation_condition":  ac.group(1).replace("\\n", " ").replace("\n", " ").strip(),
                "execution_procedure":   ep.group(1).replace("\\n", "\n").strip(),
                "termination_condition": tc.group(1).replace("\\n", " ").strip() if tc else "",
            }

        logger.warning(f"JSON parse fallback. Raw[:200]={raw[:200]}")
        summary = raw[:120].replace('"', "'").replace("\n", " ").strip()
        return {
            "activation_condition":  summary,
            "execution_procedure":   raw.replace('"', "'").replace("\n", " ").strip(),
            "termination_condition": "",
        }

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _compact(text: str, n: int = 220) -> str:
        return re.sub(r"\s+", " ", str(text))[:n]

    # ---- 持久化 ----
    def load_memory(self, filepath: str) -> None:
        self.dual_memory.load_tree(filepath)
        logger.info(f"📥 PRTree memory loaded from {filepath}")

    def save_memory(self, filepath: str) -> None:
        self.dual_memory.save_tree(filepath)
        logger.info(f"💾 PRTree memory saved to {filepath}")

    def get_memory_stats(self) -> Dict[str, Any]:
        return self.dual_memory.stats
