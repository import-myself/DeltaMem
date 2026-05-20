# """
# Dual PR-Tree Prompt Templates for Mind2Web (v8.0 - Compact)

# v8.0: 全面精简反思 prompt，强制 bullet 限制，消除冗长结构化模板
# v6.1: 移除 .format() 预处理，常量块直接内联
# """

# # =================================================================
# # Mind2Web 基础 Instruction（所有方法共享，不可修改）
# # =================================================================

# mind2web_instruction = """You are a large language model trained to navigate the web. \
# Output the next action and wait for the next observation. Here is the action space:
# 1. `CLICK [id]`: Click on an HTML element with its id.
# 2. `TYPE [id] [value]`: Type a string into the element with the id.
# 3. `SELECT [id] [value]`: Select a value for an HTML element by its id.

# You should choose from two actions: "Thought" or "Action".
# - If you choose "Thought": first think about the current condition and plan, then output your action.
#   Format: "Thought: your thoughts.\\nAction: your next action"
# - If you choose "Action": directly output the action.
#   Format: "Action: your next action"

# Wrap your final action in backticks. Example: Action: `CLICK [1234]`"""

# # =================================================================
# # Prompt 模板 (无记忆 / 有双树记忆)
# # =================================================================

# PROMPT_WITH_ICL_TEMPLATE = """{instruction}

# ---
# Here is an example for a complete task trajectory:

# {examples}
# ---

# Now, it's your turn and here is the task.
# {task}"""

# PROMPT_WITH_ICL_TEMPLATE_DUAL_MEMORY = """{instruction}

# ---
# Here is an example for a complete task trajectory:

# {examples}
# ---

# {memory_header}

# {memory_context}

# Now, it's your turn and here is the task.
# {task}"""

# MEMORY_HEADERS: dict = {
#     "prtree": (
#         "Below is brief reference knowledge from past episodes on similar tasks/websites.\n"
#         "• Task Memory — action-type rules and interaction patterns for this task type\n"
#         "• Website Memory — UI behavior patterns for this website type\n"
#         "Use as hints, not rigid procedures. Element IDs from memory are stale — match by visible text/aria-label."
#     ),
#     "synapse": (
#         "The following past task trajectories are retrieved from memory as few-shot examples.\n"
#         "⚠️ IMPORTANT: Element IDs (numbers in brackets like [1234]) in these examples are "
#         "from DIFFERENT past episodes and DO NOT exist on the current page — ignore them completely. "
#         "Locate elements by semantic attributes: aria-label, placeholder, visible text, role, "
#         "or structural position."
#     ),
#     "awm": (
#         "The following workflow procedure was distilled from past similar web navigation tasks. "
#         "Follow it as a semantic step-by-step guide if the task type matches.\n"
#         "⚠️ Do not use any element IDs from this workflow — "
#         "locate all elements by semantic attributes on the current page."
#     ),
#     "reasoningbank": (
#         "The following memory items are distilled lessons from past web navigation interactions. "
#         "[✅ SUCCESS] entries describe navigation strategies that worked; "
#         "[⚠️ FAILURE] entries highlight mistakes to avoid.\n"
#         "⚠️ Do not use any element IDs from past experiences — they are invalid on the current page. "
#         "Always locate elements by semantic attributes: aria-label, placeholder, visible text, or role."
#     ),
# }


# # =================================================================
# # 任务树反思 Prompt (Task Tree) — Compact 格式
# # =================================================================

# TaskTree_Prompt_Map = {}

# TaskTree_Prompt_Map['root_success'] = """You are a Skill Extractor. Extract a COMPACT, generalizable Base Skill from this trajectory.

# ⚠️ No element IDs — describe elements by aria-label, placeholder, visible text, or role.

# Website: {env_description}
# Task: {task_description}
# Result: SUCCESS (Steps: {steps})

# {trajectory}

# **Step-by-Step Causal Analysis:**
# {step_causal_analysis}

# Use the causal analysis above to understand WHY each step succeeded or failed, then distill actionable task-level rules and insights.

# Output a self-contained Base Skill as JSON:
# - `activation_condition`: Task TYPE and website context. Do NOT mention specific values.
# - `execution_procedure`: 3-5 bullet points MAX. Focus on TASK-LEVEL knowledge (not website UI facts — those belong in Website Memory):
#   • Action-type decisions: when to CLICK vs TYPE vs SELECT, and what about the TASK CONTEXT determines this (e.g., "search fields in booking tasks need TYPE, not CLICK")
#   • Interaction sequences: task-specific ordering of steps (e.g., "select departure before return date", "apply filters before sorting")
#   • Element identification strategy: how to locate the correct element for each task step based on semantic role in the task flow (e.g., "the submit button is at the end of the form, identified by role or text like 'Search'/'Book'")
#   • Execution insights: non-obvious knowledge about how to execute this task type correctly (e.g., "must apply filters before sorting to get correct results", "price comparison requires scrolling to load all options", "multi-step forms must be filled in order because later fields depend on earlier selections")
#   Do NOT include specific values — describe CATEGORIES. Do NOT describe website UI component behaviors (that belongs in Website Memory).
# - `termination_condition`: When this task type is complete.

# Output ONLY the JSON:
# {{
#     "activation_condition": "...",
#     "execution_procedure": "...",
#     "termination_condition": "..."
# }}"""

# TaskTree_Prompt_Map['root_failure'] = """You are a Skill Extractor from failed trajectories. Even failed episodes contain useful knowledge — extract what works and what to avoid.

# ⚠️ No element IDs — describe elements by aria-label, placeholder, visible text, or role.

# Website: {env_description}
# Task: {task_description}
# Result: FAILURE (Steps: {steps})

# {trajectory}

# **Step-by-Step Causal Analysis:**
# {step_causal_analysis}

# Use the causal analysis to identify: (1) correct steps — extract valid task-level rules, (2) failed steps — understand what task-level decision went wrong.

# Output a failure record as JSON:
# - `activation_condition`: Task type + what went wrong at a CATEGORY level (not specific values).
# - `execution_procedure`: 3-5 bullets MAX. Focus on TASK-LEVEL knowledge (not website UI facts):
#   [WORKED]: Task-level rules from correct steps — which action-type decisions were right, how the correct element was identified in the task flow. Skip ONLY if every step was wrong on all dimensions.
#   [FAILED]: What task-level decision went wrong — wrong action-type choice, wrong step ordering, wrong element targeted in the task flow. Describe the decision error, not the website UI behavior.
#   [UNEXPLORED]: Alternative task strategies not tried.
# - `termination_condition`: Empty string.

# Output ONLY the JSON:
# {{
#     "activation_condition": "...",
#     "execution_procedure": "[WORKED]: ...\\n[FAILED]: ...\\n[UNEXPLORED]: ...",
#     "termination_condition": ""
# }}"""

# TaskTree_Prompt_Map['node_success'] = """You are a Skill Delta Extractor. Extract ONLY the minimal new patch not covered by existing skills.

# ⚠️ No element IDs — describe elements by aria-label, placeholder, visible text, or role.

# === EXISTING SKILLS (DO NOT REPEAT) ===
# {retrieved_task_memory}
# === END ===

# Website: {env_description}
# Task: {task_description}
# Result: SUCCESS (Steps: {steps})

# {trajectory}

# **Step-by-Step Causal Analysis:**
# {step_causal_analysis}

# Use the causal analysis to understand what task-level rules this trajectory reveals. Then check against existing skills.

# Output {{"skip": true}} ONLY if you can satisfy ALL of the following, with direct evidence:
# 1. Identify ONE existing skill whose execution_procedure explicitly covers every task-level interaction pattern in this trajectory — quote the exact phrase from that skill for each pattern.
# 2. No step reveals a new task-level decision rule (action-type choice, step ordering, element identification strategy) not described by that quoted text.
# If you cannot quote matching text for even ONE pattern, you MUST write a delta.

# If writing a delta: 1-3 bullets MAX of new task-level rules only. Do NOT repeat website UI facts from Website Memory.
# - `activation_condition`: The NEW pattern trigger (must differ from existing).
# - `execution_procedure`: New task-level rules: action-type decisions, step ordering, element identification strategies, or task-level pitfalls.
# - `termination_condition`: When done.

# Output ONLY one of these two JSON formats:
# {{"skip": true}}
# OR
# {{
#     "activation_condition": "...",
#     "execution_procedure": "...",
#     "termination_condition": "..."
# }}"""

# TaskTree_Prompt_Map['node_failure'] = """You are a Skill Extractor from failed trajectories. Record the gap in existing skills that caused this failure, and preserve what worked.

# ⚠️ No element IDs — describe elements by aria-label, placeholder, visible text, or role.

# === EXISTING SKILLS ===
# {retrieved_task_memory}
# === END ===

# Website: {env_description}
# Task: {task_description}
# Result: FAILURE (Steps: {steps})

# {trajectory}

# **Step-by-Step Causal Analysis:**
# {step_causal_analysis}

# Use the causal analysis to identify what existing skills missed. Even in failure, correct steps reveal valid task-level knowledge.

# Output {{"skip": true}} ONLY if an existing record describes the EXACT SAME failure: you must quote the specific failed interaction and the exact element type from that record. If the failure involves a different task-level decision error, you MUST write a new record.

# Otherwise, output 3-5 bullets MAX:
# - `activation_condition`: The new failure situation (at category level, not specific values).
# - `execution_procedure`:
#   [WORKED]: Task-level rules from correct steps — which action-type decisions were right, how the correct element was identified. Skip ONLY if every step was wrong on all dimensions.
#   [FAILED]: What task-level decision went wrong, and execution insights — e.g., "should have scrolled to load more results before selecting", "filter must be applied before sort for correct ordering", "multi-step form requires completing section A before section B becomes interactive".
#   [UNEXPLORED]: Alternative task strategies not attempted.
# - `termination_condition`: Empty string.

# Output ONLY one of these two JSON formats:
# {{"skip": true}}
# OR
# {{
#     "activation_condition": "...",
#     "execution_procedure": "[WORKED]: ...\\n[FAILED]: ...\\n[UNEXPLORED]: ...",
#     "termination_condition": ""
# }}"""


# # =================================================================
# # 网站树反思 Prompt (Website/Env Tree) — Compact 格式
# # 合并 success/failure 为统一模板（参考 ALFWorld）
# # =================================================================

# EnvTree_Prompt_Map = {}

# EnvTree_Prompt_Map['root'] = """You are a Website Knowledge Extractor. Extract BRIEF website-level UI facts — regardless of task outcome.

# ⚠️ No element IDs — describe UI components by type, label, position, or behavior.

# Website: {env_description}
# Task: {task_description}
# Outcome: {result} (Steps: {steps})

# {trajectory}

# Output website UI knowledge as JSON — FACTS and INSIGHTS about the website, not task procedures:
# - `activation_condition`: "Applicable on [website_type] websites where ..." — key UI characteristics.
# - `execution_procedure`: 3-5 bullets MAX. Include BOTH:
#   **UI Facts** (what the website has):
#   • Component types: autocomplete, calendar picker, modal dialogs, dynamic loading, custom dropdowns
#   • Value format requirements: date format, city vs airport code, required input format
#   **UI Insights** (non-obvious website behaviors):
#   • Deceptive elements: components that look like one type but behave as another (e.g., styled div functioning as SELECT dropdown, text field that looks like a button)
#   • Hidden dependencies: UI elements that only become interactive after a prior interaction (e.g., return date picker disabled until departure date is set)
#   • Dynamic behavior: content that loads on scroll, options that change based on prior selections, forms that reset on certain interactions
#   Do NOT describe task steps or action-type decisions (that belongs in Task Memory).
# - `termination_condition`: When these UI facts are applied.

# Output ONLY the JSON:
# {{
#     "activation_condition": "Applicable on ... websites where ...",
#     "execution_procedure": "...",
#     "termination_condition": "..."
# }}"""

# EnvTree_Prompt_Map['node'] = """You are a Website Knowledge Extractor. Extract ONLY new website UI facts not covered by existing knowledge.

# ⚠️ No element IDs — describe UI components by type, label, position, or behavior.

# === EXISTING WEBSITE KNOWLEDGE ===
# {retrieved_env_memory}
# === END ===

# Website: {env_description}
# Task: {task_description}
# Outcome: {result} (Steps: {steps})

# {trajectory}

# Output {{"skip": true}} ONLY when EVERY UI component behavior AND EVERY interaction pattern (autocomplete, calendar picker, modal, dynamic loading, etc.) observed in this trajectory is already explicitly stated in the existing knowledge above. If even ONE component behavior or interaction quirk is not explicitly covered, you MUST write an update.

# If writing: 1-2 bullets MAX of genuinely new website behavior only.
# - `activation_condition`: "Applicable on [website_type] websites where ..." — the new UI characteristic.
# - `execution_procedure`: New website behavior facts only, as bullet points.
# - `termination_condition`: When this adaptation is complete.

# Output ONLY one of these two JSON formats:
# {{"skip": true}}
# OR
# {{
#     "activation_condition": "Applicable on ... websites where ...",
#     "execution_procedure": "...",
#     "termination_condition": "..."
# }}"""


# # =================================================================
# # Helper 函数
# # =================================================================

# def get_task_prompt_key(is_root: bool, is_success: bool) -> str:
#     if is_root:
#         return "root_success" if is_success else "root_failure"
#     return "node_success" if is_success else "node_failure"


# def get_env_prompt_key(is_root: bool, is_success: bool = True) -> str:
#     return "root" if is_root else "node"


"""
Dual PR-Tree Prompt Templates for Mind2Web (v8.0 - Compact & Natural Language Optimized)
"""

# =================================================================
# Mind2Web 基础 Instruction（所有方法共享）
# =================================================================

mind2web_instruction = """You are a large language model trained to navigate the web. \
Output the next action and wait for the next observation. Here is the action space:
1. `CLICK [id]`: Click on an HTML element with its id.
2. `TYPE [id] [value]`: Type a string into the element with the id.
3. `SELECT [id] [value]`: Select a value for an HTML element by its id.

You should choose from two actions: "Thought" or "Action".
- If you choose "Thought": first think about the current condition and plan, then output your action.
  Format: "Thought: your thoughts.\\nAction: your next action"
- If you choose "Action": directly output the action.
  Format: "Action: your next action"

Wrap your final action in backticks. Example: Action: `CLICK [1234]`"""

# =================================================================
# Prompt 模板 (无记忆 / 有双树记忆)
# =================================================================

PROMPT_WITH_ICL_TEMPLATE = """{instruction}

---
Here is an example for a complete task trajectory:

{examples}
---

Now, it's your turn and here is the task.
{task}"""

PROMPT_WITH_ICL_TEMPLATE_DUAL_MEMORY = """{instruction}

---
Here is an example for a complete task trajectory:

{examples}
---

{memory_header}

{memory_context}

Now, it's your turn and here is the task.
{task}"""

MEMORY_HEADERS: dict = {
    "prtree": (
        "Below are decision-pitfall hints distilled from past episodes — each is a short caution about "
        "where the agent previously chose the WRONG element among ambiguous candidates on similar tasks/websites.\n"
        "• Task Memory — task-type-specific pitfalls: which element to PICK among look-alike candidates\n"
        "• Website Memory — declarative UI behavior facts for this website type\n"
        "How to use: at every step, check whether the current Observation matches any pitfall's "
        "'Trigger' situation; if it does, follow the pitfall's recommendation. "
        "Element IDs in past memory are stale — match candidates by aria-label, placeholder, visible text, or role."
    ),
    "synapse": (
        "The following past task trajectories are retrieved from memory as few-shot examples.\n"
        "⚠️ IMPORTANT: Element IDs in these examples are from DIFFERENT past episodes and DO NOT exist on the current page. "
        "Locate elements by semantic attributes: aria-label, placeholder, visible text, role, or structural position."
    ),
    "awm": (
        "The following workflow procedure was distilled from past similar web navigation tasks. "
        "Follow it as a semantic step-by-step guide if the task type matches.\n"
        "⚠️ Do not use any element IDs from this workflow — locate all elements by semantic attributes."
    ),
    "reasoningbank": (
        "The following memory items are distilled lessons from past web navigation interactions. "
        "[✅ SUCCESS] entries describe navigation strategies that worked; "
        "[⚠️ FAILURE] entries highlight mistakes to avoid.\n"
        "⚠️ Do not use any element IDs from past experiences. Always locate elements by semantic attributes."
    ),
}

# =================================================================
# 任务树反思 Prompt (Task Tree)
# =================================================================

TaskTree_Prompt_Map = {}

TaskTree_Prompt_Map['root_success'] = """You are a Decision Pitfall Extractor.

Examine this successful trajectory and capture the SINGLE most consequential decision the agent
made that future agents could easily get wrong on similar tasks. Express it as a short
DECISION PITFALL — NOT a step-by-step procedure. The pitfall must describe an ambiguous
situation and tell future agents which element to PICK and which look-alike decoy to AVOID.

⚠️ No element IDs — refer to elements only by aria-label, placeholder, visible text, or role.
⚠️ ONE pitfall ONLY. Do NOT list multiple steps. Do NOT describe a procedure.

**Rule of Thumb:**
❌ Bad (procedure): "First open the menu, then click Collections, then create a new entry."
✅ Good (pitfall):  "When both a main-menu button and a sidebar 'Collections' link are visible,
                     click the main-menu button — the sidebar link only views existing collections."

Website: {env_description}
Task: {task_description}
Result: SUCCESS (Steps: {steps})

{trajectory}

**Step-by-Step Causal Analysis:**
{step_causal_analysis}

Identify the decision-ambiguity moment that the agent navigated correctly. Distil it into a pitfall
generalisable to similar tasks/websites.

Output ONLY valid JSON. Do NOT wrap in Markdown (no ```json). Start directly with {{.
{{
    "activation_condition": "Situation: which UI elements appear together causing the ambiguity (refer by aria-label / placeholder / visible text / role). Example: 'A Round-Trip / One-way toggle AND a Find-flights button are both visible on the flight search form.' Do NOT mention specific values like city names or dates.",
    "execution_procedure": "ONE sentence (≤ 60 words). Pattern: 'Click <right element by semantic descriptor> not <wrong element by semantic descriptor>, because <reason>.' Do NOT write a multi-step procedure. Do NOT include element IDs or specific values.",
    "termination_condition": ""
}}"""

TaskTree_Prompt_Map['root_failure'] = """You are a Decision Pitfall Extractor (from a failed trajectory).

Identify the SINGLE most consequential WRONG decision the agent made, and turn it into a pitfall
future agents can use to avoid repeating it on similar tasks. Express it as ONE concise
decision-pitfall — NOT a multi-step procedure or a long lesson narrative.

⚠️ No element IDs — refer to elements only by aria-label, placeholder, visible text, or role.
⚠️ ONE pitfall ONLY. The pitfall must name the AMBIGUOUS situation, the WRONG element the
   agent picked, and the RIGHT element it should have picked.

**Rule of Thumb:**
❌ Bad (narrative):  "The agent first did X correctly, then made a reasoning error and chose Y; next time it should consider Z."
✅ Good (pitfall):   "When both a 'Review' icon link (top nav) and an inline 'Write a review' link are present, click the top-nav 'Review' icon — the inline link leads to the wrong flow."

Website: {env_description}
Task: {task_description}
Result: FAILURE (Steps: {steps})

{trajectory}

**Step-by-Step Causal Analysis:**
{step_causal_analysis}

From the causal analysis, find the step where the agent picked the wrong element among look-alikes,
and turn that into a pitfall.

Output ONLY valid JSON. Do NOT wrap in Markdown (no
```json). Start directly with {{.
{{
    "activation_condition": "Situation: which UI elements appear together causing this ambiguity (semantic descriptors only, no element IDs, no specific values).",
    "execution_procedure": "ONE sentence (≤ 60 words). Pattern: 'Click <right element by semantic descriptor> not <wrong element by semantic descriptor>, because <reason>.' Do NOT write a multi-step procedure or a long lesson story.",
    "termination_condition": ""
}}"""

TaskTree_Prompt_Map['node_success'] = """You are a Decision Pitfall Delta Extractor.

Existing pitfalls already capture some decision-ambiguities. From this successful trajectory,
extract ONE NEW decision pitfall that the existing pitfalls do NOT already warn about.
If every observed ambiguity is already covered, output skip.

⚠️ No element IDs — refer to elements by aria-label, placeholder, visible text, or role.
⚠️ ONE pitfall ONLY, never a procedure or multi-step plan.

Website: {env_description}
Task: {task_description}
Result: SUCCESS (Steps: {steps})

{trajectory}

**Step-by-Step Causal Analysis:**
{step_causal_analysis}

=== EXISTING PITFALLS (DO NOT REPEAT) ===
{retrieved_task_memory}
=== END ===

Identify any decision-ambiguity in this trajectory NOT explicitly covered by an existing pitfall above.
Output ONLY valid JSON. Do NOT wrap in Markdown (no ```json). Start directly with {{.

If every decision the agent made is already covered by an existing pitfall:
{{
    "reasoning": "State which existing pitfall covers each decision and why no new one is needed.",
    "skip": true
}}

If extracting a NEW pitfall:
{{
    "reasoning": "Name the NEW decision-ambiguity that no existing pitfall addresses.",
    "activation_condition": "Situation: which UI elements appear together causing this NEW ambiguity (semantic descriptors only).",
    "execution_procedure": "ONE sentence (≤ 60 words). Pattern: 'Click <right element by semantic descriptor> not <wrong element by semantic descriptor>, because <reason>.' Not a procedure.",
    "termination_condition": ""
}}"""

TaskTree_Prompt_Map['node_failure'] = """You are a Decision Pitfall Delta Extractor (from a failed trajectory).

This trajectory failed even though pitfalls were available. Record the NEW pitfall — the
decision-ambiguity no existing pitfall warned about. If an existing pitfall already covered
this exact decision, output skip (the gap is in retrieval, not knowledge).

⚠️ No element IDs — refer to elements by aria-label, placeholder, visible text, or role.
⚠️ ONE pitfall ONLY. Do NOT write a procedure or multi-paragraph narrative.

Website: {env_description}
Task: {task_description}
Result: FAILURE (Steps: {steps})

{trajectory}

**Step-by-Step Causal Analysis:**
{step_causal_analysis}

=== EXISTING PITFALLS ===
{retrieved_task_memory}
=== END ===

From the causal analysis find the step where the wrong element was chosen, then check whether
any existing pitfall above already names that exact ambiguity.

Output ONLY valid JSON. Do NOT wrap in Markdown (no
```json). Start directly with {{.

If an existing pitfall already names this exact decision-ambiguity:
{{
    "reasoning": "Quote which existing pitfall covers this situation.",
    "skip": true
}}

If a NEW pitfall is required:
{{
    "reasoning": "Name the new decision-ambiguity that no existing pitfall addresses.",
    "activation_condition": "Situation: which UI elements together caused this failure (semantic descriptors only, no element IDs).",
    "execution_procedure": "ONE sentence (≤ 60 words). Pattern: 'Click <right element by semantic descriptor> not <wrong element by semantic descriptor>, because <reason>.' Not a procedure or narrative.",
    "termination_condition": ""
}}"""

# =================================================================
# 网站树反思 Prompt (Website/Env Tree)
# =================================================================

EnvTree_Prompt_Map = {}

EnvTree_Prompt_Map['root'] = """You are a Website Knowledge Extractor. Extract website-level UI facts and quirks.

⚠️ No element IDs — describe UI components by type, label, position, or behavior.
**Rule of Thumb:**
❌ WRONG (Task-level): "First click search, then select the cheapest flight."
✅ RIGHT (Env-level): "The search results page uses infinite scroll; cheapest options may not appear until scrolled."

Website: {env_description}
Task: {task_description}
Outcome: {result} (Steps: {steps})

{trajectory}

Focus on FACTS and INSIGHTS about the website's UI (not task procedures).
Output ONLY valid JSON. Do NOT wrap the JSON in Markdown formatting (no ```json). Start directly with {{.
{{
    "activation_condition": "Applicable on [website_type] websites where ... (key UI characteristics)",
    "execution_procedure": "A clear natural language description of the website's UI behaviors and quirks. Describe how specific components (like modals, dropdowns, dynamic loading) behave, any hidden dependencies (e.g., fields that unlock only after prior selections), and unusual element structures. Focus on how the website reacts, rather than task steps.",
    "termination_condition": "When these UI facts are applied."
}}"""

EnvTree_Prompt_Map['node'] = """You are a Website Knowledge Extractor. Extract ONLY new website UI facts not covered by existing knowledge.

⚠️ No element IDs — describe UI components by type, label, position, or behavior.

Website: {env_description}
Task: {task_description}
Outcome: {result} (Steps: {steps})

{trajectory}

=== EXISTING WEBSITE KNOWLEDGE ===
{retrieved_env_memory}
=== END ===

Check if every UI behavior and quirk observed is already explicitly stated in EXISTING WEBSITE KNOWLEDGE.
Output ONLY valid JSON. Do NOT wrap the JSON in Markdown formatting (no 
```json). Start directly with {{.

If ALL UI characteristics are already fully covered:
{{
    "reasoning": "Explain which existing knowledge covers these UI facts...",
    "skip": true
}}

If extracting genuinely NEW website behaviors:
{{
    "reasoning": "Identify what new UI behavior or quirk was observed...",
    "activation_condition": "Applicable on [website_type] websites where ... (the new UI characteristic)",
    "execution_procedure": "A natural language description of the new website UI behaviors or hidden dependencies not covered before.",
    "termination_condition": "When this adaptation is complete."
}}"""

# =================================================================
# Helper 函数
# =================================================================

def get_task_prompt_key(is_root: bool, is_success: bool) -> str:
    if is_root:
        return "root_success" if is_success else "root_failure"
    return "node_success" if is_success else "node_failure"

def get_env_prompt_key(is_root: bool, is_success: bool = True) -> str:
    # 按照设定，Env树合并了 success/failure 的逻辑，仅通过 outcome 区分
    return "root" if is_root else "node"