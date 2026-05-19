"""
Dual PR-Tree Prompt Templates for Mind2Web (v6.1 - Skill Format)

v6.1 修复: 移除 .format() 预处理，常量块直接内联，和 ALFWorld/SciWorld 保持一致，
           避免双层大括号在两次 format 后被过度消耗导致 KeyError。
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

{memory_header}

{memory_context}

Now, it's your turn and here is the task.
{task}"""

# 按 memory 类型自适应的上下文引导语（含 Mind2web 特有的元素 ID 警告）
MEMORY_HEADERS: dict = {
    "prtree": (
        "The following memory is retrieved from your hierarchical memory system:\n"
        "• Task Skill Memory — HOW to navigate this task type. Base Skill = foundation procedure; Skill Deltas = patches for specific UI variants.\n"
        "• Website Knowledge Memory — WHAT UI components exist and HOW to identify them. Read as background facts, not steps to execute.\n"
        "⚠️ Do NOT use element IDs from memory — they are unique to past pages. Locate elements by aria-label, placeholder, visible text, or structural position."
    ),
    "synapse": (
        "The following past task trajectories are retrieved from memory as few-shot examples.\n"
        "⚠️ IMPORTANT: Element IDs (numbers in brackets like [1234]) in these examples are "
        "from DIFFERENT past episodes and DO NOT exist on the current page — ignore them completely. "
        "Locate elements by semantic attributes: aria-label, placeholder, visible text, role, "
        "or structural position."
    ),
    "awm": (
        "The following workflow procedure was distilled from past similar web navigation tasks. "
        "Follow it as a semantic step-by-step guide if the task type matches.\n"
        "⚠️ Do not use any element IDs from this workflow — "
        "locate all elements by semantic attributes on the current page."
    ),
    "reasoningbank": (
        "The following memory items are distilled lessons from past web navigation interactions. "
        "[✅ SUCCESS] entries describe navigation strategies that worked; "
        "[⚠️ FAILURE] entries highlight mistakes to avoid.\n"
        "⚠️ Do not use any element IDs from past experiences — they are invalid on the current page. "
        "Always locate elements by semantic attributes: aria-label, placeholder, visible text, or role."
    ),
}


# =================================================================
# 任务树反思 Prompt (Task Tree) — Skill 格式
# 注意: 模板不做 .format() 预处理，直接用 {env_description} 等单括号占位符，
#       和 ALFWorld/SciWorld 保持一致，agent 只 format 一次。
# =================================================================

TaskTree_Prompt_Map = {}

TaskTree_Prompt_Map['root_success'] = """You are a Skill Extractor. Based on this successful web navigation trajectory, extract a **Base Skill** for this task type.

IMPORTANT: Do NOT include any element IDs (numbers in brackets like [2058], [19433]).
These IDs are unique to this specific episode's HTML and are useless for other tasks.
Describe elements ONLY by their semantic attributes:
  - HTML role/type: input with type='text', button with role='button'
  - ARIA attributes: element with aria-label='Departure city', aria-expanded='true'
  - Visible text: button labeled 'Search Flights', link text 'Sign In'
  - Placeholder: input with placeholder='Enter city name'
  - Structural position: search bar in the top header, leftmost filter panel
  - Interaction behavior: dropdown that opens on CLICK, autocomplete that shows suggestions after typing

**Full Scenario:**
Website: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps})

Trajectory:
{trajectory}

**Output Requirements:**
Your output will be placed in a global skill cache and triggered DIRECTLY with NO access to this trajectory.

Output a **self-contained Base Skill** as JSON:
- `activation_condition`: The task TYPE + website context that triggers this skill (e.g., for flight booking tasks on travel websites where user must search for one-way flights). Describe using semantic task features, NOT element IDs.
- `execution_procedure`: Complete self-contained procedure. Follow this format:

Required format for `execution_procedure`:

Task Category: [Exactly one of: flight_booking | hotel_booking | rental_booking |
  product_search_filter | form_fill_submit | event_booking |
  account_settings | content_search | navigation | other]

Element Identification Strategies:
- For [step description, e.g. origin city autocomplete input]:
    Identify: [semantic description — role, aria-label, placeholder, visible text, position]
    Interact: [exact action type and sequence]
    Value format: [what value to pass]
- For [next step type]: ...

Element Disambiguation Rules:
- Component: [e.g. origin city autocomplete]
    Correct element signals: [aria-label/placeholder/text that marks the RIGHT element]
    Confusable elements to avoid: [what nearby elements look similar but are wrong]

Action Sequence Outline:
1. [What to do — describe element semantically, NOT by ID]
2. ...

Key Pitfalls to Avoid:
- [Specific mistake → correct approach, using semantic descriptions only]

- `termination_condition`: When to consider this web task skill complete (e.g., search results page loaded with results, or confirmation page displayed).

Output ONLY the JSON (no element IDs anywhere):
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}"""

TaskTree_Prompt_Map['root_failure'] = """You are a Skill Extractor. Based on this FAILED web navigation trajectory, extract a **corrective Base Skill**.

IMPORTANT: Do NOT include any element IDs (numbers in brackets like [2058], [19433]).
These IDs are unique to this specific episode's HTML and are useless for other tasks.
Describe elements ONLY by their semantic attributes:
  - HTML role/type: input with type='text', button with role='button'
  - ARIA attributes: element with aria-label='Departure city', aria-expanded='true'
  - Visible text: button labeled 'Search Flights', link text 'Sign In'
  - Placeholder: input with placeholder='Enter city name'
  - Structural position: search bar in the top header, leftmost filter panel
  - Interaction behavior: dropdown that opens on CLICK, autocomplete that shows suggestions after typing

**Full Scenario:**
Website: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps})

Trajectory:
{trajectory}

**Step-Level Failure Analysis:**
{failed_steps_analysis}

**Output Requirements:**
Output a **self-contained corrective Base Skill** as JSON:
- `activation_condition`: Task type + what went wrong (e.g., for flight booking tasks where agent selects wrong input for origin city).
- `execution_procedure`: Corrected procedure with explicit error-avoidance rules. No element IDs.
  Use exactly these two labeled sections:
  [FAILED]: Which element types were misidentified or which interactions were wrong, and what the page returned.
  [UNEXPLORED]: Plausible alternative approaches this agent never attempted — prevents future agents from treating this failure as proof those approaches are impossible.
- `termination_condition`: When this corrective skill is complete.

Output ONLY the JSON (no element IDs anywhere):
{{
    "activation_condition": "...",
    "execution_procedure": "[FAILED]: ...\\n[UNEXPLORED]: ...",
    "termination_condition": ""
}}"""

TaskTree_Prompt_Map['node_success'] = """You are a Skill Delta Extractor. Extract the **Task Skill Delta** — what NEW strategic knowledge this trajectory adds beyond existing memories.

IMPORTANT: Do NOT include any element IDs (numbers in brackets like [2058], [19433]).
These IDs are unique to this specific episode's HTML and are useless for other tasks.
Describe elements ONLY by their semantic attributes:
  - HTML role/type: input with type='text', button with role='button'
  - ARIA attributes: element with aria-label='Departure city', aria-expanded='true'
  - Visible text: button labeled 'Search Flights', link text 'Sign In'
  - Placeholder: input with placeholder='Enter city name'
  - Structural position: search bar in the top header, leftmost filter panel
  - Interaction behavior: dropdown that opens on CLICK, autocomplete that shows suggestions after typing

=== EXISTING TASK SKILL MEMORIES (already stored — DO NOT REPEAT) ===
{retrieved_task_memory}
=== END ===

**Current Experience:**
Website: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps})

Trajectory:
{trajectory}

Output {{"skip": true}} ONLY if you can satisfy ALL of the following, with direct evidence:
1. Identify ONE existing skill whose execution_procedure explicitly covers every distinct element type and interaction performed in this trajectory — quote the exact phrase from that skill for each action.
2. No action in this trajectory required a different UI component, a different interaction sequence, or an edge case not covered by that quoted text.
If you cannot quote matching text for even ONE action in this trajectory, you MUST write a delta.

Otherwise, output ONLY the new Skill Delta as JSON. Must genuinely differ from existing conditions.

- `activation_condition`: The SPECIFIC NEW trigger — new UI pattern, edge case, or condition NOT covered by existing skills. No element IDs.
- `execution_procedure`: The NEW incremental steps/rules only. Self-contained, no references to existing memories.

Required format for `execution_procedure`:

Task Category: [Exactly one of: flight_booking | hotel_booking | rental_booking |
  product_search_filter | form_fill_submit | event_booking |
  account_settings | content_search | navigation | other]

Element Identification Strategies:
- For [step description, e.g. origin city autocomplete input]:
    Identify: [semantic description — role, aria-label, placeholder, visible text, position]
    Interact: [exact action type and sequence]
    Value format: [what value to pass]
- For [next step type]: ...

Element Disambiguation Rules:
- Component: [e.g. origin city autocomplete]
    Correct element signals: [aria-label/placeholder/text that marks the RIGHT element]
    Confusable elements to avoid: [what nearby elements look similar but are wrong]

Action Sequence Outline:
1. [What to do — describe element semantically, NOT by ID]
2. ...

Key Pitfalls to Avoid:
- [Specific mistake → correct approach, using semantic descriptions only]

- `termination_condition`: When this delta's modification is complete.

Output ONLY one of these two JSON formats (no element IDs anywhere):
{{"skip": true}}
OR
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}"""

TaskTree_Prompt_Map['node_failure'] = """You are a Skill Delta Extractor. Identify the **gap in existing task skills** that caused this failure.

IMPORTANT: Do NOT include any element IDs (numbers in brackets like [2058], [19433]).
These IDs are unique to this specific episode's HTML and are useless for other tasks.
Describe elements ONLY by their semantic attributes:
  - HTML role/type: input with type='text', button with role='button'
  - ARIA attributes: element with aria-label='Departure city', aria-expanded='true'
  - Visible text: button labeled 'Search Flights', link text 'Sign In'
  - Placeholder: input with placeholder='Enter city name'
  - Structural position: search bar in the top header, leftmost filter panel
  - Interaction behavior: dropdown that opens on CLICK, autocomplete that shows suggestions after typing

=== EXISTING TASK SKILL MEMORIES (already stored — DO NOT REPEAT) ===
{retrieved_task_memory}
=== END ===

**Current Experience:**
Website: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps})

Trajectory:
{trajectory}

**Step-Level Failure Analysis:**
{failed_steps_analysis}

Output {{"skip": true}} ONLY if an existing failure record describes the EXACT SAME failure: you must quote the specific failed element type, interaction, and environment response from that record that match this trajectory. If the failure pattern differs in any way, you MUST write a new record.

Otherwise, output ONLY the corrective Skill Delta as JSON:
- `activation_condition`: The specific new situation the existing skills failed to handle. No element IDs.
- `execution_procedure`: Use exactly these two labeled sections:
  [FAILED]: Which element types were misidentified or which interactions were wrong, and what the environment returned. Reference existing skill guidance that was followed but did not work.
  [UNEXPLORED]: Plausible alternative approaches this agent never attempted — prevents future agents from treating this failure as proof those approaches are impossible.
- `termination_condition`: Leave empty string.

Output ONLY one of these two JSON formats (no element IDs anywhere):
{{"skip": true}}
OR
{{
    "activation_condition": "...",
    "execution_procedure": "[FAILED]: ...\\n[UNEXPLORED]: ...",
    "termination_condition": ""
}}"""


# =================================================================
# 网站树反思 Prompt (Website/Env Tree) — Skill 格式
# =================================================================

EnvTree_Prompt_Map = {}

EnvTree_Prompt_Map['root_success'] = """You are a Website Knowledge Extractor. Extract declarative **Base Website Knowledge** from this successful web navigation.

⚠️ ABSOLUTE PROHIBITION: Do NOT include any element IDs (numbers in brackets like [9123], [8494]).
Describe UI components ONLY by their observable properties:
  • Visual/structural position: "top navigation bar", "left sidebar filter panel", "modal dialog"
  • Component type: "text input", "dropdown", "calendar date picker", "toggle button", "checkbox"
  • Visible labels/text: "labeled 'Departure'", "button text 'Search'", "placeholder 'MM/DD/YYYY'"
  • ARIA attributes: "aria-label='Select date'", "role='combobox'"
  • Interaction trigger: "opens on CLICK", "requires hover to reveal", "appears after typing"

**Full Scenario:**
Website: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps})

Trajectory:
{trajectory}

**Output Requirements:**
This is DECLARATIVE KNOWLEDGE — facts about the website's UI, not a procedure to execute.
Your output will be retrieved in similar websites with NO access to this trajectory.

Output **self-contained Base Website Knowledge** as JSON:
- `activation_condition`: Website type + key UI features that make this knowledge applicable. Format: "Applicable on [website_type] websites like {env_description} where ..."
- `execution_procedure`: Factual observations about the website's UI components.

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
- [Pitfall: what fails → correct action]

- `termination_condition`: When website-specific navigation facts have been fully applied (e.g., all required form fields identified and filled).

Output ONLY the JSON (no element IDs):
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}"""

EnvTree_Prompt_Map['root_failure'] = """You are a Website Knowledge Extractor. Record what website UI traps or wrong assumptions caused this failure as **Base Website Knowledge**.

⚠️ ABSOLUTE PROHIBITION: Do NOT include any element IDs (numbers in brackets like [9123], [8494]).
Describe UI components ONLY by their observable properties:
  • Visual/structural position: "top navigation bar", "left sidebar filter panel", "modal dialog"
  • Component type: "text input", "dropdown", "calendar date picker", "toggle button", "checkbox"
  • Visible labels/text: "labeled 'Departure'", "button text 'Search'", "placeholder 'MM/DD/YYYY'"
  • ARIA attributes: "aria-label='Select date'", "role='combobox'"
  • Interaction trigger: "opens on CLICK", "requires hover to reveal", "appears after typing"

**Full Scenario:**
Website: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps})

Trajectory:
{trajectory}

**Step-Level Failure Analysis:**
{failed_steps_analysis}

**Output Requirements:**
Record only what was observed. Do not state correct interaction rules unless demonstrated in the trajectory.

Output **self-contained corrective Base Website Knowledge** as JSON:
- `activation_condition`: Website type + the specific trap condition that caused failure. Format: "Applicable on [website_type] websites where ..."
- `execution_procedure`: Factual observations — what UI components were encountered, what interactions failed, what pitfalls exist. No element IDs.
  Use exactly these two labeled sections:
  [FAILED]: What UI components were misidentified or what interactions failed, and what the page returned.
  [UNEXPLORED]: Plausible alternative UI patterns or interaction strategies not attempted.
- `termination_condition`: Leave empty.

Output ONLY the JSON (no element IDs):
{{
    "activation_condition": "...",
    "execution_procedure": "[FAILED]: ...\\n[UNEXPLORED]: ...",
    "termination_condition": ""
}}"""

EnvTree_Prompt_Map['node_success'] = """You are a Website Knowledge Extractor. Extract the **Website Knowledge Update** — new declarative UI facts NOT covered by existing memories.

⚠️ ABSOLUTE PROHIBITION: Do NOT include any element IDs (numbers in brackets like [9123], [8494]).
Describe UI components ONLY by their observable properties:
  • Visual/structural position: "top navigation bar", "left sidebar filter panel", "modal dialog"
  • Component type: "text input", "dropdown", "calendar date picker", "toggle button", "checkbox"
  • Visible labels/text: "labeled 'Departure'", "button text 'Search'", "placeholder 'MM/DD/YYYY'"
  • ARIA attributes: "aria-label='Select date'", "role='combobox'"
  • Interaction trigger: "opens on CLICK", "requires hover to reveal", "appears after typing"

=== EXISTING WEBSITE KNOWLEDGE (already stored — DO NOT REPEAT) ===
{retrieved_env_memory}
=== END ===

**Current Experience:**
Website: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps})

Trajectory:
{trajectory}

Output {{"skip": true}} ONLY if EVERY specific UI component type, interaction rule, and website-specific pattern observed in this trajectory is already explicitly stated in the existing knowledge above — quote the exact phrase for each. If even ONE component or rule is not explicitly covered, you MUST write an update.

Otherwise, output ONLY the new Website Knowledge Update as JSON:
- `activation_condition`: The specific new website condition / UI pattern that activates this update. Format: "Applicable on [website_type] websites where ..."
- `execution_procedure`: NEW declarative website facts only. Self-contained, no element IDs.

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
- [Pitfall: what fails → correct action]

- `termination_condition`: When this website adaptation is complete.

Output ONLY one of these two JSON formats (no element IDs):
{{"skip": true}}
OR
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}"""

EnvTree_Prompt_Map['node_failure'] = """You are a Website Knowledge Extractor. Identify the **website knowledge gap** that caused this failure.

⚠️ ABSOLUTE PROHIBITION: Do NOT include any element IDs (numbers in brackets like [9123], [8494]).
Describe UI components ONLY by their observable properties:
  • Visual/structural position: "top navigation bar", "left sidebar filter panel", "modal dialog"
  • Component type: "text input", "dropdown", "calendar date picker", "toggle button", "checkbox"
  • Visible labels/text: "labeled 'Departure'", "button text 'Search'", "placeholder 'MM/DD/YYYY'"
  • ARIA attributes: "aria-label='Select date'", "role='combobox'"
  • Interaction trigger: "opens on CLICK", "requires hover to reveal", "appears after typing"

=== EXISTING WEBSITE KNOWLEDGE (already stored — DO NOT REPEAT) ===
{retrieved_env_memory}
=== END ===

**Current Experience:**
Website: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps})

Trajectory:
{trajectory}

**Step-Level Failure Analysis:**
{failed_steps_analysis}

Output {{"skip": true}} ONLY if an existing knowledge entry already explicitly describes the EXACT SAME wrong assumption or UI trap observed in this trajectory — quote the specific entry. If the failure pattern differs in any way, you MUST write a new record.

Otherwise, output ONLY the corrective Website Knowledge Update as JSON:
- `activation_condition`: The specific new website situation existing knowledge failed to handle. Format: "Applicable on [website_type] websites where ..."
- `execution_procedure`: Use exactly these two labeled sections:
  [FAILED]: What UI components were misidentified or what interactions failed, and what the page returned.
  [UNEXPLORED]: Plausible alternative UI patterns or interaction strategies not attempted — prevents future agents from treating this failure as proof those approaches are impossible.
- `termination_condition`: Leave empty.

Output ONLY one of these two JSON formats (no element IDs):
{{"skip": true}}
OR
{{
    "activation_condition": "...",
    "execution_procedure": "[FAILED]: ...\\n[UNEXPLORED]: ...",
    "termination_condition": ""
}}"""


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
