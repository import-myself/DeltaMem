"""
Dual PR-Tree Prompt Templates for Mind2Web (v5.0)

双树设计:
- TaskTree:    以任务类型为索引 (e.g. "flight_booking", "product_search_filter")
               存储通用的任务类型策略、语义化元素识别启发式、常见误操作模式
- WebsiteTree: 以 domain::website 为索引 (e.g. "travel::united", "shopping::kohls")
               存储网站特有 UI 组件布局、交互规则、网站操作陷阱

v5.0 核心变化:
- 明确禁止在记忆中存储 backend_node_id (形如 [1234] 的数字 ID)
  ——这类 ID 是 episode 独有的，跨任务完全无效，存入记忆会误导 Agent
- content_body 改为结构化格式：任务类别 → 元素语义识别策略 → 操作陷阱
- Env Tree 以组件类型为粒度，要求描述交互方式而非抽象评论
"""

# =================================================================
# Mind2Web 基础 Instruction
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

The following relevant experiences may help you complete the task.
⚠️ IMPORTANT RULES FOR USING MEMORY:
1. Any element IDs (numbers in brackets like [1234]) are from DIFFERENT past episodes — they DO NOT exist on the current page. IGNORE them completely.
2. Experiences labeled [✅ SUCCESS] describe what WORKS — follow their semantic strategies.
3. Experiences labeled [⚠️ FAILURE] describe what WENT WRONG — use them to AVOID the same mistakes.
4. Always locate elements by semantic attributes: aria-label, placeholder, visible text, role, or structural position.

{memory_context}

Now, it's your turn and here is the task.
{task}"""

# =================================================================
# 任务树反思 Prompt (Task Tree)
# 索引键: 任务类别 (e.g. "flight_booking", "product_search_filter")
# 存储内容: 语义化元素识别策略 / 操作步骤模板 / 常见误操作
#
# ⚠️ CRITICAL RULE FOR ALL TASK TREE PROMPTS:
#   NEVER include element IDs ([number]) in content_body.
#   Backend node IDs are unique per episode and meaningless in other tasks.
#   Always describe elements by: role, type, aria-label, placeholder,
#   visible text content, or structural position (e.g. "top header", "sidebar").
# =================================================================

_TASK_ID_PROHIBITION = """\
⚠️ ABSOLUTE PROHIBITION: Do NOT include any element IDs (numbers in brackets like [2058], [19433]).
These IDs are unique to this specific episode's HTML and are completely useless — even harmful — for other tasks.
Describe elements ONLY by their semantic attributes:
  • HTML role/type: "an input with type='text'", "a button with role='button'"
  • ARIA attributes: "element with aria-label='Departure city'", "aria-expanded='true'"
  • Visible text: "a button labeled 'Search Flights'", "link text 'Sign In'"
  • Placeholder: "input with placeholder='Enter city name'"
  • Structural position: "the search bar in the top header", "leftmost filter panel"
  • Interaction behavior: "a dropdown that opens on CLICK", "an autocomplete that shows suggestions after typing 3+ characters"\
"""

_TASK_CONTENT_FORMAT = """\
Required format for "content_body":

Task Category: [Exactly one of: flight_booking | hotel_booking | rental_booking |
  product_search_filter | form_fill_submit | event_booking |
  account_settings | content_search | navigation | other]

Element Identification Strategies:
- For [step description, e.g. "origin city / autocomplete input"]:
    Identify: [semantic description — role, aria-label, placeholder, visible text, position]
    Interact: [exact action type and sequence, e.g. "TYPE to trigger autocomplete, then CLICK matching suggestion — do NOT press Enter before selecting"]
    Value format: [what value to pass — e.g. "city name only, not airport code", "exact visible option text for SELECT"]
- For [next step type]: ...

Element Disambiguation Rules (critical for candidate selection):
- Component: [e.g. "origin city autocomplete"]
    Correct element signals: [aria-label/placeholder/text that marks the RIGHT element, e.g. "aria-label contains 'departure'/'from'/'origin'"]
    Confusable elements to avoid: [what nearby elements look similar but are wrong, e.g. "destination input has aria-label 'to'/'arrival'"]
- Component: [another component that caused confusion in this task]
    Correct element signals: [...]
    Confusable elements to avoid: [...]

Action Sequence Outline:
1. [What to do — describe element semantically, NOT by ID]
2. ...
(List the key steps in order; omit trivial ones)

Key Pitfalls to Avoid:
- [Specific mistake → correct approach, using semantic descriptions only]\
"""

TaskTree_Prompt_Map = {}

TaskTree_Prompt_Map['root_success'] = """You have successfully completed a web navigation task.
Your job: Extract a **generalizable task strategy** for future agents handling similar web tasks.

{id_prohibition}

**Focus on TASK STRATEGY — transferable across different websites:**
- What category of web task is this?
- What UI component types appear in this task and how to identify them semantically?
- What is the correct interaction sequence for each component type?
- What decision points or common pitfalls exist?

**Full Scenario:**
Website: {{env_description}}
Task Goal: {{task_description}}
Result: SUCCESS (Steps: {{steps}})

Trajectory:
{{trajectory}}

**Output Requirements:**
1. "content_body" must be fully self-contained and understandable without seeing the trajectory.
2. Describe the task category explicitly.
3. Use ONLY semantic element descriptions — no element IDs.
4. Be concrete and actionable.

{content_format}

**Output (JSON only, no extra text):**
{{{{
    "memory_description": "One sentence: task category + key workflow insight (no element IDs).",
    "content_body": "<follow the format above exactly>"
}}}}""".format(id_prohibition=_TASK_ID_PROHIBITION, content_format=_TASK_CONTENT_FORMAT)

TaskTree_Prompt_Map['root_failure'] = """You attempted a web navigation task but FAILED.
Your job: Generate a **corrective task strategy** so future agents avoid the same mistake.

{id_prohibition}

**Focus on TASK STRATEGY — what went wrong and how to fix it:**
- What category of web task is this?
- At which step did the workflow fail, and why?
- What is the correct element identification approach and interaction sequence?

**Full Scenario:**
Website: {{env_description}}
Task Goal: {{task_description}}
Result: FAILURE (Steps: {{steps}})

Trajectory:
{{trajectory}}

**Step-Level Failure Analysis (element selection errors):**
{{failed_steps_analysis}}

**Output Requirements:**
1. "content_body" must be fully self-contained.
2. Describe the task category explicitly.
3. Use ONLY semantic element descriptions — no element IDs.
4. Be concrete — for each failed step, describe what the correct element looks like (aria-label, placeholder, position) to distinguish it from confusable candidates.

{content_format}

**Output (JSON only, no extra text):**
{{{{
    "memory_description": "One sentence: task category + what went wrong + how to fix it (no element IDs).",
    "content_body": "<follow the format above exactly>"
}}}}""".format(id_prohibition=_TASK_ID_PROHIBITION, content_format=_TASK_CONTENT_FORMAT)

TaskTree_Prompt_Map['node_success'] = """You successfully completed a web task. Existing task strategy memories are stored.
Your job: Identify what **NEW strategic insight** this experience adds that is NOT already covered.

{id_prohibition}

=== EXISTING TASK MEMORIES (already stored — DO NOT REPEAT) ===
{{retrieved_task_memory}}
=== END ===

**Current Experience:**
Website: {{env_description}}
Task Goal: {{task_description}}
Result: SUCCESS (Steps: {{steps}})

Trajectory:
{{trajectory}}

**Residual Generation Instructions:**
1. READ existing memories carefully. List what element types and strategies they already cover.
2. ANALYZE this trajectory. Find genuinely NEW knowledge:
   - A different UI component type not previously described
   - A more precise semantic identification strategy for an existing component type
   - An edge case or recovery pattern not covered
   - A more efficient interaction sequence variant
3. Output ONLY new incremental knowledge. DO NOT repeat existing memories.

**Output Requirements:**
1. "content_body" must be self-contained.
2. Mention the task category.
3. Use ONLY semantic element descriptions — no element IDs.
4. The new knowledge must genuinely differ from all existing memories.

{content_format}

**Output (JSON only, no extra text):**
{{{{
    "memory_description": "One sentence about the NEW insight only. Must differ from all existing (no element IDs).",
    "content_body": "<follow the format above exactly, focusing only on the new insight>"
}}}}""".format(id_prohibition=_TASK_ID_PROHIBITION, content_format=_TASK_CONTENT_FORMAT)

TaskTree_Prompt_Map['node_failure'] = """You attempted a web task but FAILED despite existing task strategy memories.
Your job: Identify the **specific gap** in existing strategies that caused the failure.

{id_prohibition}

=== EXISTING TASK MEMORIES (already stored — DO NOT REPEAT) ===
{{retrieved_task_memory}}
=== END ===

**Current Experience:**
Website: {{env_description}}
Task Goal: {{task_description}}
Result: FAILURE (Steps: {{steps}})

Trajectory:
{{trajectory}}

**Step-Level Failure Analysis (element selection errors):**
{{failed_steps_analysis}}

**Residual Generation Instructions:**
1. READ existing memories. What do they recommend?
2. ANALYZE the failure using the step-level analysis above. At which step did things go wrong?
   - Was the element identification description insufficient (wrong candidate selected)?
   - Was the interaction sequence wrong for this component type?
   - Was there an edge case the existing strategies missed?
3. For each failed step: describe what distinguishes the CORRECT element from confusable candidates.
4. Identify the SPECIFIC gap. Output ONLY gap-filling correction.

**Output Requirements:**
1. "content_body" must be self-contained.
2. Mention the task category.
3. Use ONLY semantic element descriptions — no element IDs.

{content_format}

**Output (JSON only, no extra text):**
{{{{
    "memory_description": "One sentence: specific gap + correction. Must differ from existing (no element IDs).",
    "content_body": "<follow the format above exactly, focusing only on the gap and fix>"
}}}}""".format(id_prohibition=_TASK_ID_PROHIBITION, content_format=_TASK_CONTENT_FORMAT)


# =================================================================
# 网站树反思 Prompt (Website/Env Tree)
# 索引键: domain::website (e.g. "travel::united", "shopping::kohls")
# 存储内容: 网站特有 UI 组件位置、交互规则、操作陷阱
#
# ⚠️ CRITICAL RULE FOR ALL ENV TREE PROMPTS:
#   NEVER include element IDs in content_body.
#   Describe UI components by their visual/structural properties.
# =================================================================

_ENV_ID_PROHIBITION = """\
⚠️ ABSOLUTE PROHIBITION: Do NOT include any element IDs (numbers in brackets like [9123], [8494]).
Describe UI components ONLY by their observable properties:
  • Visual/structural position: "top navigation bar", "left sidebar filter panel", "modal dialog"
  • Component type: "text input", "dropdown", "calendar date picker", "toggle button", "checkbox"
  • Visible labels/text: "labeled 'Departure'", "button text 'Search'", "placeholder 'MM/DD/YYYY'"
  • ARIA attributes: "aria-label='Select date'", "role='combobox'"
  • Interaction trigger: "opens on CLICK", "requires hover to reveal", "appears after typing"\
"""

_ENV_CONTENT_FORMAT = """\
Required format for "content_body":

Website Type: [e.g. travel-booking | e-commerce | entertainment | news | finance | rental | other]
Website: [name as provided in env_description]
Task Category on This Website: [e.g. flight_booking, product_search_filter]

Key UI Components and Interaction Rules:
- Search / Text Input:
    Location: [where on page, e.g. "top header bar", "center of homepage"]
    Identification: [aria-label, placeholder, or visible label]
    Interaction: [e.g. "TYPE query, then press Enter" or "TYPE to trigger autocomplete, then CLICK suggestion"]
    Value format: [e.g. "type city name only, not airport code", "partial text triggers dropdown"]

- Date Picker (if present):
    Identification: [e.g. "input with aria-label containing 'date'", "calendar icon button"]
    Interaction: [e.g. "CLICK to open calendar overlay, then CLICK target date — do NOT TYPE date directly"]
    Value format: [e.g. "calendar date cells use MM/DD/YYYY format visible on cell"]

- Dropdown / SELECT (if present):
    Identification: [e.g. "labeled 'Sort by'", "role='listbox'"]
    Interaction: [e.g. "use SELECT action with EXACT visible option text"]
    Value format: [e.g. "option values are visible display strings like 'Price: Low to High', not internal codes"]

- Autocomplete Input (if present):
    Identification: [e.g. "city input with placeholder 'Enter city'"]
    Interaction: [e.g. "TYPE 3+ characters, wait for suggestion list, then CLICK the matching suggestion — do NOT submit before selecting"]
    Value format: [e.g. "type just the city name; full suggestion text is selected by CLICK"]

- Submit / Confirm Button (if present):
    Identification: [e.g. "button labeled 'Search Flights'", "button in the form footer"]
    Interaction: [e.g. "CLICK after all fields are filled"]

Element Disambiguation Rules (for candidate selection):
- Component: [e.g. "departure city input"]
    Correct element signals: [specific aria-label, placeholder, or structural position that identifies the right element among candidates]
    Confusable with: [what other candidate elements look similar, and how to tell them apart]
- Component: [another component if applicable]
    Correct element signals: [...]
    Confusable with: [...]

Known Website-Specific Traps:
- [Specific pitfall: what action fails → what the correct action is]
- [Another trap, described concisely]\
"""

EnvTree_Prompt_Map = {}

EnvTree_Prompt_Map['root_success'] = """You completed a web task on a specific website.
Your job: Extract **website-specific UI knowledge** — practical patterns for operating on this website.

{id_prohibition}

**Full Scenario:**
Website: {{env_description}}
Task Goal: {{task_description}}
Result: SUCCESS (Steps: {{steps}})

Trajectory:
{{trajectory}}

**Output Requirements:**
1. "content_body" must be fully self-contained and understandable without seeing the trajectory.
2. Describe the website type and name explicitly.
3. Cover key UI component types encountered: how to identify them semantically and how to interact correctly.
4. Include known pitfalls observed during this trajectory.
5. Use ONLY semantic/structural descriptions — no element IDs.

{content_format}

**Output (JSON only, no extra text):**
{{{{
    "memory_description": "One sentence: website type + most important operational insight (no element IDs).",
    "content_body": "<follow the format above exactly>"
}}}}""".format(id_prohibition=_ENV_ID_PROHIBITION, content_format=_ENV_CONTENT_FORMAT)

EnvTree_Prompt_Map['root_failure'] = """You attempted a web task on a specific website but FAILED.
Your job: Extract **website-specific warnings** — what UI factors caused the failure and how to avoid them.

{id_prohibition}

**Full Scenario:**
Website: {{env_description}}
Task Goal: {{task_description}}
Result: FAILURE (Steps: {{steps}})

Trajectory:
{{trajectory}}

**Step-Level Failure Analysis (element selection errors):**
{{failed_steps_analysis}}

**Output Requirements:**
1. "content_body" must be fully self-contained.
2. Describe the website type and name explicitly.
3. For each failed step, describe: which UI component was involved, what the correct element looks like (aria-label, placeholder, structural position), and what confusable elements exist on this website.
4. Use ONLY semantic/structural descriptions — no element IDs.

{content_format}

**Output (JSON only, no extra text):**
{{{{
    "memory_description": "One sentence: website type + key pitfall discovered (no element IDs).",
    "content_body": "<follow the format above exactly>"
}}}}""".format(id_prohibition=_ENV_ID_PROHIBITION, content_format=_ENV_CONTENT_FORMAT)

EnvTree_Prompt_Map['node_success'] = """You completed a task on a website. Existing website knowledge memories are stored.
Your job: Identify what **NEW website-specific UI knowledge** this experience adds that is NOT already covered.

{id_prohibition}

=== EXISTING WEBSITE MEMORIES (already stored — DO NOT REPEAT) ===
{{retrieved_env_memory}}
=== END ===

**Current Experience:**
Website: {{env_description}}
Task Goal: {{task_description}}
Result: SUCCESS (Steps: {{steps}})

Trajectory:
{{trajectory}}

**Residual Generation Instructions:**
1. READ existing memories carefully. List which component types and interaction rules they already cover.
2. ANALYZE this trajectory. Find genuinely NEW website knowledge:
   - A new component type not previously described
   - A more precise identification or interaction rule for an existing component
   - A new navigation shortcut or UI state transition
   - A new pitfall discovered
3. Output ONLY new incremental knowledge. DO NOT repeat existing memories.

**Output Requirements:**
1. "content_body" must be self-contained.
2. Describe the website explicitly.
3. Use ONLY semantic/structural descriptions — no element IDs.
4. The new knowledge must genuinely differ from all existing memories.

{content_format}

**Output (JSON only, no extra text):**
{{{{
    "memory_description": "One sentence about the NEW insight only. Must differ from all existing (no element IDs).",
    "content_body": "<follow the format above exactly, focusing only on the new knowledge>"
}}}}""".format(id_prohibition=_ENV_ID_PROHIBITION, content_format=_ENV_CONTENT_FORMAT)

EnvTree_Prompt_Map['node_failure'] = """You attempted a task on a website but FAILED despite existing website knowledge.
Your job: Identify what **website knowledge gap** caused the failure.

{id_prohibition}

=== EXISTING WEBSITE MEMORIES (already stored — DO NOT REPEAT) ===
{{retrieved_env_memory}}
=== END ===

**Current Experience:**
Website: {{env_description}}
Task Goal: {{task_description}}
Result: FAILURE (Steps: {{steps}})

Trajectory:
{{trajectory}}

**Step-Level Failure Analysis (element selection errors):**
{{failed_steps_analysis}}

**Residual Generation Instructions:**
1. READ existing memories. What do they know about this website's UI components?
2. ANALYZE the failure using the step-level analysis above. Was it caused by:
   - Wrong identification of a UI component type (selected wrong candidate)?
   - Incorrect interaction sequence for a component?
   - A website-specific quirk not previously recorded?
3. For each failed step, describe the correct vs confusable elements on THIS website.
4. Identify the SPECIFIC gap. Output ONLY gap-filling knowledge.

**Output Requirements:**
1. "content_body" must be self-contained.
2. Describe the website explicitly.
3. Use ONLY semantic/structural descriptions — no element IDs.

{content_format}

**Output (JSON only, no extra text):**
{{{{
    "memory_description": "One sentence: specific website gap. Must differ from existing (no element IDs).",
    "content_body": "<follow the format above exactly, focusing only on the gap and correction>"
}}}}""".format(id_prohibition=_ENV_ID_PROHIBITION, content_format=_ENV_CONTENT_FORMAT)


# =================================================================
# Helper 函数
# =================================================================

def get_task_prompt_key(is_root: bool, is_success: bool) -> str:
    if is_root:
        return "root_success" if is_success else "root_failure"
    return "node_success" if is_success else "node_failure"


def get_env_prompt_key(is_root: bool, is_success: bool) -> str:
    if is_root:
        return "root_success" if is_success else "root_failure"
    return "node_success" if is_success else "node_failure"
