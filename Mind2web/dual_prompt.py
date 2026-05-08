"""
Dual PR-Tree Prompt Templates for Mind2Web (v6.0 - Skill Format)

v6.0 核心改进:
- 反思 Prompt 重构为面向 Skill 的格式：activation_condition / execution_procedure / termination_condition
- Root 节点 → Base Skill（基础任务技能提取）
- Residual 节点 → Skill Delta（技能修正残差）
- 保留 element ID 禁止规则
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
# 公共约束块
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

_EXECUTION_FORMAT = """\
Required format for `execution_procedure`:

Task Category: [Exactly one of: flight_booking | hotel_booking | rental_booking |
  product_search_filter | form_fill_submit | event_booking |
  account_settings | content_search | navigation | other]

Element Identification Strategies:
- For [step description, e.g. "origin city / autocomplete input"]:
    Identify: [semantic description — role, aria-label, placeholder, visible text, position]
    Interact: [exact action type and sequence]
    Value format: [what value to pass]
- For [next step type]: ...

Element Disambiguation Rules:
- Component: [e.g. "origin city autocomplete"]
    Correct element signals: [aria-label/placeholder/text that marks the RIGHT element]
    Confusable elements to avoid: [what nearby elements look similar but are wrong]

Action Sequence Outline:
1. [What to do — describe element semantically, NOT by ID]
2. ...

Key Pitfalls to Avoid:
- [Specific mistake → correct approach, using semantic descriptions only]\
"""


# =================================================================
# 任务树反思 Prompt (Task Tree) — Skill 格式
# =================================================================

TaskTree_Prompt_Map = {}

TaskTree_Prompt_Map['root_success'] = """You are a Skill Extractor. Based on this successful web navigation trajectory, extract a **Base Skill** for this task type.

{id_prohibition}

**Full Scenario:**
Website: {{env_description}}
Task Goal: {{task_description}}
Result: SUCCESS (Steps: {{steps}})

Trajectory:
{{trajectory}}

**Output Requirements:**
Your output will be placed in a global skill cache and triggered DIRECTLY with NO access to this trajectory.

Output a **self-contained Base Skill** as JSON:
- `activation_condition`: The task TYPE + website context that triggers this skill (e.g., "for flight booking tasks on travel websites where user must search for one-way flights"). Describe using semantic task features, NOT element IDs.
- `execution_procedure`: Complete self-contained procedure. Follow this format:
{execution_format}
- `termination_condition`: When to consider this web task skill complete (e.g., "search results page loaded with results, or confirmation page displayed").

Output ONLY the JSON (no element IDs anywhere):
{{{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}}}""".format(id_prohibition=_TASK_ID_PROHIBITION, execution_format=_EXECUTION_FORMAT)

TaskTree_Prompt_Map['root_failure'] = """You are a Skill Extractor. Based on this FAILED web navigation trajectory, extract a **corrective Base Skill**.

{id_prohibition}

**Full Scenario:**
Website: {{env_description}}
Task Goal: {{task_description}}
Result: FAILURE (Steps: {{steps}})

Trajectory:
{{trajectory}}

**Step-Level Failure Analysis:**
{{failed_steps_analysis}}

**Output Requirements:**
Output a **self-contained corrective Base Skill** as JSON:
- `activation_condition`: Task type + what went wrong (e.g., "for flight booking tasks where agent selects wrong input for origin city").
- `execution_procedure`: Corrected procedure with explicit error-avoidance rules. No element IDs.
{execution_format}
- `termination_condition`: When this corrective skill is complete.

Output ONLY the JSON (no element IDs anywhere):
{{{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}}}""".format(id_prohibition=_TASK_ID_PROHIBITION, execution_format=_EXECUTION_FORMAT)

TaskTree_Prompt_Map['node_success'] = """You are a Skill Delta Extractor. Extract the **Task Skill Delta** — what NEW strategic knowledge this trajectory adds beyond existing memories.

{id_prohibition}

=== EXISTING TASK SKILL MEMORIES (already stored — DO NOT REPEAT) ===
{{retrieved_task_memory}}
=== END ===

**Current Experience:**
Website: {{env_description}}
Task Goal: {{task_description}}
Result: SUCCESS (Steps: {{steps}})

Trajectory:
{{trajectory}}

**Output Requirements:**
1. READ existing memories. List what element types and strategies they cover.
2. FIND genuinely NEW knowledge: a different UI component type, more precise identification, edge case, or efficiency improvement.
3. Output ONLY the new Skill Delta as JSON. Must genuinely differ from existing conditions.

- `activation_condition`: The SPECIFIC NEW trigger — new UI pattern, edge case, or condition NOT covered by existing skills. No element IDs.
- `execution_procedure`: The NEW incremental steps/rules only. Self-contained, no references to existing memories.
{execution_format}
- `termination_condition`: When this delta's modification is complete.

Output ONLY the JSON (no element IDs anywhere):
{{{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}}}""".format(id_prohibition=_TASK_ID_PROHIBITION, execution_format=_EXECUTION_FORMAT)

TaskTree_Prompt_Map['node_failure'] = """You are a Skill Delta Extractor. Identify the **gap in existing task skills** that caused this failure.

{id_prohibition}

=== EXISTING TASK SKILL MEMORIES (already stored — DO NOT REPEAT) ===
{{retrieved_task_memory}}
=== END ===

**Current Experience:**
Website: {{env_description}}
Task Goal: {{task_description}}
Result: FAILURE (Steps: {{steps}})

Trajectory:
{{trajectory}}

**Step-Level Failure Analysis:**
{{failed_steps_analysis}}

**Output Requirements:**
1. READ existing skills. What element types and strategies do they cover?
2. IDENTIFY the specific gap: wrong element identification, incorrect interaction, or uncovered edge case.
3. Output ONLY the corrective Skill Delta as JSON.

- `activation_condition`: The specific new situation the existing skills failed to handle.
- `execution_procedure`: The corrective rules for the gap. Self-contained, semantic descriptions only.
{execution_format}
- `termination_condition`: When this corrective delta is complete.

Output ONLY the JSON (no element IDs anywhere):
{{{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}}}""".format(id_prohibition=_TASK_ID_PROHIBITION, execution_format=_EXECUTION_FORMAT)


# =================================================================
# 网站树反思 Prompt (Website/Env Tree) — Skill 格式
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

_ENV_EXECUTION_FORMAT = """\
Required format for `execution_procedure`:

Website Type: [e.g. travel-booking | e-commerce | entertainment | news | finance | rental | other]
Website: [name as provided]
Task Category: [e.g. flight_booking, product_search_filter]

Key UI Components and Interaction Rules:
- Search / Text Input:
    Location: [where on page]
    Identification: [aria-label, placeholder, or visible label]
    Interaction: [TYPE/CLICK sequence]
    Value format: [what to pass]
- Date Picker (if present): [identification, interaction, value format]
- Dropdown / SELECT (if present): [identification, interaction, value format]
- Autocomplete Input (if present): [identification, interaction, value format]
- Submit Button (if present): [identification, interaction]

Element Disambiguation Rules:
- Component: [e.g. "departure city input"]
    Correct element signals: [aria-label, placeholder, or structural position]
    Confusable with: [similar elements to avoid]

Known Website-Specific Traps:
- [Pitfall: what fails → correct action]\
"""

EnvTree_Prompt_Map = {}

EnvTree_Prompt_Map['root_success'] = """You are a Skill Extractor for Website Knowledge. Extract a **Base Website Skill** from this successful web navigation.

{id_prohibition}

**Full Scenario:**
Website: {{env_description}}
Task Goal: {{task_description}}
Result: SUCCESS (Steps: {{steps}})

Trajectory:
{{trajectory}}

**Output Requirements:**
Your output will be triggered DIRECTLY in similar websites with NO access to this trajectory.

Output a **self-contained Base Website Skill** as JSON:
- `activation_condition`: Website type + key UI features that make this skill applicable (e.g., "on travel-booking websites like united.com with autocomplete city inputs and calendar date pickers").
- `execution_procedure`: Complete website-specific UI knowledge.
{execution_format}
- `termination_condition`: When website-specific navigation is complete (e.g., "all required form fields filled and search/submit button clicked").

Output ONLY the JSON (no element IDs):
{{{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}}}""".format(id_prohibition=_ENV_ID_PROHIBITION, execution_format=_ENV_EXECUTION_FORMAT)

EnvTree_Prompt_Map['root_failure'] = """You are a Skill Extractor for Website Knowledge. Extract a **corrective Base Website Skill** from this failed navigation.

{id_prohibition}

**Full Scenario:**
Website: {{env_description}}
Task Goal: {{task_description}}
Result: FAILURE (Steps: {{steps}})

Trajectory:
{{trajectory}}

**Step-Level Failure Analysis:**
{{failed_steps_analysis}}

**Output Requirements:**
Output a **self-contained corrective Base Website Skill** as JSON:
- `activation_condition`: Website type + the specific trap condition that caused failure.
- `execution_procedure`: Corrective website knowledge with pitfall descriptions and correct interaction rules. No element IDs.
{execution_format}
- `termination_condition`: When website-specific pitfalls have been addressed.

Output ONLY the JSON (no element IDs):
{{{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}}}""".format(id_prohibition=_ENV_ID_PROHIBITION, execution_format=_ENV_EXECUTION_FORMAT)

EnvTree_Prompt_Map['node_success'] = """You are a Skill Delta Extractor for Website Knowledge. Extract the **Website Skill Delta** — new UI knowledge NOT covered by existing memories.

{id_prohibition}

=== EXISTING WEBSITE SKILL MEMORIES (already stored — DO NOT REPEAT) ===
{{retrieved_env_memory}}
=== END ===

**Current Experience:**
Website: {{env_description}}
Task Goal: {{task_description}}
Result: SUCCESS (Steps: {{steps}})

Trajectory:
{{trajectory}}

**Output Requirements:**
1. READ existing memories. Which component types and interaction rules do they cover?
2. FIND genuinely NEW website knowledge: new component type, more precise rule, new pitfall.
3. Output ONLY the new Website Skill Delta as JSON.

- `activation_condition`: The specific new website condition / UI pattern that activates this delta.
- `execution_procedure`: NEW website knowledge only. Self-contained.
{execution_format}
- `termination_condition`: When this website adaptation is complete.

Output ONLY the JSON (no element IDs):
{{{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}}}""".format(id_prohibition=_ENV_ID_PROHIBITION, execution_format=_ENV_EXECUTION_FORMAT)

EnvTree_Prompt_Map['node_failure'] = """You are a Skill Delta Extractor for Website Knowledge. Identify the **website knowledge gap** that caused this failure.

{id_prohibition}

=== EXISTING WEBSITE SKILL MEMORIES (already stored — DO NOT REPEAT) ===
{{retrieved_env_memory}}
=== END ===

**Current Experience:**
Website: {{env_description}}
Task Goal: {{task_description}}
Result: FAILURE (Steps: {{steps}})

Trajectory:
{{trajectory}}

**Step-Level Failure Analysis:**
{{failed_steps_analysis}}

**Output Requirements:**
1. READ existing memories. What do they know about this website's UI?
2. IDENTIFY the specific gap: wrong component identification, incorrect interaction, or uncovered website quirk.
3. Output ONLY the corrective Website Skill Delta as JSON.

- `activation_condition`: The specific new website situation existing memories failed to handle.
- `execution_procedure`: Corrective website rule. Self-contained, no element IDs.
{execution_format}
- `termination_condition`: When this website correction is complete.

Output ONLY the JSON (no element IDs):
{{{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}}}""".format(id_prohibition=_ENV_ID_PROHIBITION, execution_format=_ENV_EXECUTION_FORMAT)


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
