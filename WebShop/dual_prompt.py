"""
Dual PR-Tree Prompt Templates for WebShop

设计原则（与 ALFWorld 一致）:
- Root: 冷启动，从轨迹提取完整自包含 skill/knowledge
- Node: 增量，只提取与已有记忆不同的最小 delta
- Failure: 记录陷阱（搜索关键词失策 / 选错产品 / 漏选 size 等），不是可执行步骤
- 不用 Rules 列表，直接在字段描述里说清要求

WebShop 双树划分:
- Task Tree (任务):   instruction 描述「要买什么 + 价格/规格约束」
- Env  Tree (网站): 抽取的产品类目（perfume / phone case / shampoo / ...）。
                      用于跨任务沉淀「在 WebShop 上买此类商品的通用 UI 操作模式」。
"""

# =================================================================
# WebShop Instruction（沿用 ETO baseline 的 system prompt）
# =================================================================

webshop_instruction = """You are web shopping.
I will give you instructions about what to do.
You have to follow the instructions.
Every round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.
You can use search action if search is available.
You can click one of the buttons in clickables.
An action should be of the following structure:
search[keywords]
click[value]
If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions.
Remember that your keywords in search should be carefully designed.
Your response should use the following format:

Thought: I think ...
Action: click[something]"""


# =================================================================
# Prompt 模板（无记忆 / 有记忆）
# =================================================================

PROMPT_WITH_ICL_TEMPLATE = """{instruction}
---
Here is an example for a complete web-shopping trajectory.

{examples}
---

Now, it's your turn and here is the task.
{task}
"""

PROMPT_WITH_ICL_TEMPLATE_DUAL_MEMORY = """{instruction}
---
Here is an example for a complete web-shopping trajectory.

{examples}
---

{memory_header}

{memory_context}

Now, it's your turn and here is the task.
{task}
"""

MEMORY_HEADERS: dict = {
    "prtree": (
        "## How to use the memory below — READ CAREFULLY\n"
        "The memory contains three kinds of entries:\n"
        "  • **[Base Skill]** — a SUCCESSFUL past shopping episode with two parts: (1) a **Reference Trajectory** showing the exact Observation→Thought→Action sequence, and (2) a **Key Tip** summarising the core strategy. **This Base Skill ALWAYS applies to your current task.** Treat the Reference Trajectory as a second worked example to imitate (how search keywords were composed, how the candidate was picked, the option-selection order, when Buy Now was pressed). The Key Tip distils the transferable lesson.\n"
        "  • **[Skill Patch N]** — a short supplementary patch adding extra advice ON TOP of the Base. Apply a patch only if its 'When to apply' condition matches your current task; otherwise ignore that patch (but still follow the Base).\n"
        "  • **[Failure Record / ⛔]** — a warning about a trap. NEVER execute its procedure; use it only to AVOID the described trap.\n\n"
        "### How to actually use it (mandatory)\n"
        "  STEP A — Before your first Action, your **Thought MUST contain one sentence stating how you adapt the Base Skill's pattern to your current task.** E.g.: \"Following the Base Skill, I'll compose a search query combining product type + most discriminating attribute + price ceiling.\"\n"
        "  STEP B — Write `Action: search[...]` using YOUR task's product/attributes/price, NOT the Base's literal values.\n"
        "  STEP C — On the item page, follow the SAME option-selection order shown in the Base's Reference Trajectory, but click YOUR task's values.\n"
        "  STEP D — If any Patch's 'When to apply' matches your task, weave its advice into Steps B–C.\n\n"
        "Anti-pattern: writing a Thought that only lists your task's attributes without referencing the Base Skill means you ignored the memory and will likely fail."
    ),
    "synapse": (
        "The following past web-shopping trajectories are retrieved from memory as few-shot examples. "
        "Use them as reference if the task pattern is similar to the current one:"
    ),
    "awm": (
        "The following workflow procedure was distilled from past WebShop sessions. "
        "Follow it as a step-by-step guide if the task type matches:"
    ),
    "reasoningbank": (
        "The following memory items are distilled lessons from past WebShop sessions. "
        "[✅ SUCCESS] entries describe strategies that worked; "
        "[⚠️ FAILURE] entries highlight mistakes to avoid. "
        "Apply the relevant insights to guide your actions:"
    ),
}


# =================================================================
# Task Tree Reflection Prompts
# =================================================================

TaskTree_Prompt_Map: dict = {}

TaskTree_Prompt_Map['root_success'] = """You are writing a Key Tip for a successful WebShop trajectory. The trajectory itself is already saved separately — you only need to write the abstract guidance.

Product Category: {env_description}
Task: {task_description}
Result: SUCCESS ({steps}/{max_steps} steps)

=== TRAJECTORY ===
{trajectory}
=== END ===

Write a Key Tip (1–2 sentences, ≤ 40 words) capturing the most transferable lesson from this trajectory:
- What search/selection strategy made this succeed?
- What generalizable decision pattern should future agents remember?
- Do NOT repeat task-specific values (exact product names, prices, colors).

Output ONLY the JSON:
{{
    "activation_condition": "Short phrase describing what product category / task type this skill applies to. ≤ 30 words. Specific to category but NOT to exact constraints.",
    "execution_procedure": "Key Tip: ...",
    "termination_condition": "After Action: click[Buy Now] succeeds with full reward."
}}
"""

TaskTree_Prompt_Map['root_failure'] = """You are a Failure Recorder. This is a FAILURE RECORD — NOT a skill to execute. Future agents will see a ⛔ warning with this.

Product Category: {env_description}
Task: {task_description}
Result: FAILURE ({steps}/{max_steps} steps)

{trajectory}

- `activation_condition`: In plain natural language, describe what WebShop task situation this applies to and what wrong assumption caused the failure (e.g. too-broad search keywords, picking the first hit without checking constraints, missing size/option selection before Buy Now).
- `execution_procedure`:
  [FAILED]: Actions tried and how the WebShop env responded (e.g. "search returned 0 results", "Buy Now pressed without selecting size, got reward 0.5").
  [UNEXPLORED]: Plausible alternative search keywords / candidates / option choices never attempted.
- `termination_condition`: Leave empty string.

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "[FAILED]: ...\\n[UNEXPLORED]: ...",
    "termination_condition": ""
}}
"""

TaskTree_Prompt_Map['node_success'] = """You are writing a **Skill Patch** that supplements an existing Base Skill — NOT another standalone trajectory or procedure.

=== EXISTING SKILL CHAIN (Base trajectory + Key Tip + previous patches) ===
{retrieved_task_memory}
=== END ===

Product Category (this episode): {env_description}
Task (this episode): {task_description}
Result: SUCCESS ({steps}/{max_steps} steps)

{trajectory}

### Your job
Compare this new trajectory against the Base trajectory shown above. Decide whether it reveals a **new tactic, special case, or caveat** not already covered. Examples of valid patches:
- A new search-keyword trick for this product sub-category (e.g. "for perfumes, include the fluid-ounce number verbatim").
- A different option-selection order required by this sub-category.
- A constraint prioritization rule when results are sparse.
- A deviation from the base trajectory that was necessary here, with an explanation.

### Rules for writing the patch
- **DO NOT** reproduce the base trajectory. Assume the reader already has it.
- Write ONLY what is *different / additional / clarifying* relative to the base trajectory + key tip.
- ≤80 words total in execution_procedure.
- `activation_condition`: name the SPECIFIC trigger condition for this patch.
- `execution_procedure`: start with "In addition to the base:" or "Replace base step X with:".
- `termination_condition`: leave empty unless the patch changes the stop condition.

### Skip rule
Output `{{"skip": true}}` ONLY if this trajectory follows the base trajectory exactly, with no new tactics or deviations worth noting.

Output ONLY one of these two JSON formats:
{{"skip": true}}
OR
{{
    "activation_condition": "When [specific sub-category / constraint combo]...",
    "execution_procedure": "In addition to the base: ...",
    "termination_condition": ""
}}
"""

TaskTree_Prompt_Map['node_failure'] = """You are recording a **Failure Patch** — a warning that supplements the existing Base Skill (trajectory + tip), not a standalone failure log.

=== EXISTING SKILL CHAIN (Base trajectory + Key Tip + previous patches/warnings) ===
{retrieved_task_memory}
=== END ===

Product Category: {env_description}
Task: {task_description}
Result: FAILURE ({steps}/{max_steps} steps)

{trajectory}

### Your job
Identify the SPECIFIC trap this trajectory hit, then write a SHORT warning that a future agent can use to avoid it. Examples of valid traps:
- "For sparse-result sub-categories (e.g. tuxedo shirts), refining the search by removing fabric attributes recovers results."
- "Pressing Buy Now without clicking the 'size' option even when no size is listed yields partial reward only."
- "Searching with all stylistic adjectives ('butt lifting', 'high waist', 'tummy control') simultaneously returns 0 results; drop the adjectives, keep the product noun."

### Rules
- DO NOT restate base procedure. Just the trap.
- `activation_condition`: name the specific sub-category + symptom ("when shopping for {env_description} AND results page shows 0-3 items after first search").
- `execution_procedure` (≤80 words): start with `[TRAP]:` followed by what tempted the wrong action, then `[FIX]:` followed by the corrective tactic that should have been tried.
- `termination_condition`: leave empty.

### Skip rule
Output `{{"skip": true}}` ONLY if an existing failure record in the chain ALREADY warns about the exact same trap (same sub-category + same trap symptom).

Output ONLY one of these two JSON formats:
{{"skip": true}}
OR
{{
    "activation_condition": "When [specific sub-category + trap symptom]...",
    "execution_procedure": "[TRAP]: ...\\n[FIX]: ...",
    "termination_condition": ""
}}
"""


# =================================================================
# Env Tree Reflection Prompts (WebShop 中环境 = 产品类目 UI 知识)
# =================================================================

EnvTree_Prompt_Map: dict = {}

EnvTree_Prompt_Map['root'] = """You are a WebShop Site Knowledge Extractor. Extract declarative FACTS about WebShop's UI behaviour for THIS PRODUCT CATEGORY, written so they remain useful when a future agent shops in the SAME category but for a different specific item.

Product Category: {env_description}
Task this episode: {task_description}
Outcome: {result} ({steps}/{max_steps} steps)

{trajectory}

Output self-contained Base Site Knowledge — FACTS, NOT step-by-step procedure:
- `activation_condition`: "Applicable when shopping for {env_description} on WebShop." Add any structural traits observed (e.g. "items in this category typically expose size and color as required options before Buy Now").
- `execution_procedure`: Three groups of FACTS (use short bullets):
  (A) Option dimensions usually exposed for this category (size? color? scent? pack-size? quantity?). State which are REQUIRED to select before Buy Now to get full reward.
  (B) Search-keyword patterns that worked here — describe as patterns ("combining the product type with the price ceiling and the most discriminating attribute"), not the exact strings from this trajectory.
  (C) Pitfalls observed (e.g. "Buy Now without selecting size → partial reward only"; "results page shows up to N items, pagination via 'Next >'").
- `termination_condition`: When the agent has used these facts to inform a Buy-Now-correct selection.

Output ONLY the JSON:
{{
    "activation_condition": "Applicable when shopping for {env_description} on WebShop ...",
    "execution_procedure": "(A) ... (B) ... (C) ...",
    "termination_condition": "..."
}}
"""

EnvTree_Prompt_Map['node'] = """You are writing a **Site Knowledge Patch** — a small note that supplements existing site knowledge, NOT a restatement of it.

=== EXISTING SITE KNOWLEDGE (Base + previous updates) ===
{retrieved_env_memory}
=== END ===

Product Category (this episode): {env_description}
Task this episode: {task_description}
Outcome: {result} ({steps}/{max_steps} steps)

{trajectory}

### Your job
Note a NEW site fact this trajectory revealed that the base/existing knowledge doesn't already state. Examples:
- "For {env_description}, the option panel shows 'fit type' (men/women/youth) ALONGSIDE size — both must be selected."
- "Quantity option for {env_description} defaults to 1; do not need to click unless task says 'pack of N'."
- "Search results for {env_description} are capped at 50; pagination via 'Next >'."

### Rules
- DO NOT restate base knowledge. Just the new fact.
- 1-3 short bullets, ≤60 words total in execution_procedure.
- `activation_condition`: name the SPECIFIC sub-category trigger.
- `execution_procedure`: just the facts ("the panel shows X required option"), not a procedure.

### Skip rule
Output `{{"skip": true}}` ONLY if every UI fact observed here is already in the existing knowledge above.

Output ONLY one of these two JSON formats:
{{"skip": true}}
OR
{{
    "activation_condition": "When shopping for {env_description} on WebShop ...",
    "execution_procedure": "...",
    "termination_condition": ""
}}
"""


# =================================================================
# Helpers
# =================================================================

def get_task_prompt_key(is_root: bool, is_success: bool) -> str:
    if is_root:
        return "root_success" if is_success else "root_failure"
    return "node_success" if is_success else "node_failure"


def get_env_prompt_key(is_root: bool, is_success: bool = True) -> str:
    return "root" if is_root else "node"
