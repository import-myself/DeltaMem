"""
Dual PR-Tree Prompt Templates (v7.0)

v7.0 核心改进:
1. 环境经验扩展: 不仅是静态物品位置mapping，还包括「在该环境下高效完成任务的操作套路」
2. 残差经验严格差异化: 给出前序经验的完整 description+body 供对比，明确要求不得重复
3. scenario description 给完整: 所有 prompt 统一提供 env_description + task_description
"""

# =================================================================
# 基础 ALFWorld Instruction
# =================================================================

alfworld_instruction = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
For each of your turn, you will be given the observation of the last turn. You should choose from two actions: "Thought" or "Action". If you choose "Thought", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.
 Action: your next action"; If you choose "Action", you should directly output the action in this turn. Your output must strictly follow this format:"Action: your next action". 
The available actions are:
1. go to {recep}
2. take {obj} from {recep}
3. put {obj} in/on {recep}
4. open {recep}
5. close {recep}
6. toggle {obj} {recep}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
where {obj} and {recep} correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.
Reminder: 
1. The action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal.
2. Think when necessary, try to act directly more in the process.
"""

# =================================================================
# Prompt 模板 (无记忆 / 有记忆)
# =================================================================

PROMPT_WITH_ICL_TEMPLATE = """{instruction}
---
Here is an example for a complete task trajectory.

{examples}
---

Now, it's your turn and here is the task.
{task}
"""

PROMPT_WITH_ICL_TEMPLATE_DUAL_MEMORY = """{instruction}
---
Here is an example for a complete task trajectory.

{examples}
---

The following relevant experiences may help you complete the task:

{memory_context}

Now, it's your turn and here is the task.
{task}
"""


# =================================================================
# 任务树反思 Prompt
# =================================================================

TaskTree_Prompt_Map = {}

TaskTree_Prompt_Map['root_success'] = """You have successfully completed a household task.
Your job: Extract a **general task workflow strategy** that helps future agents solve similar task types.

**Focus on TASK STRATEGY — the general workflow applicable across different environments:**
- What type of task is this? (e.g., heat-then-place, cool-then-place, pick-clean-then-place, etc.)
- What is the correct step-by-step action sequence?
- What critical action syntax or rules must be followed? (e.g., must you hold the object to use 'heat X with Y'?)
- What decision points, preconditions, or common pitfalls exist?

**Full Scenario:**
Environment: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps})

Trajectory:
{trajectory}

**CRITICAL: Self-Contained Output**
Your output will be stored and later shown to a future agent on a DIFFERENT but similar task.
That future agent will NOT see the current trajectory, environment, or any memory chain.
Rules:
1. "content_body" must be fully understandable on its own — no references to "the above", "the retrieved memory", etc.
2. Mention the task type explicitly (e.g., "for tasks requiring heating an object and placing it somewhere").
3. Include specific action syntax rules where relevant (e.g., "use 'heat X with microwave' while holding X in hand").
4. Be concrete and actionable — include step-by-step instructions with decision points and pitfalls.

**Output (JSON):**
1. "memory_description": One sentence summarizing the task type and key strategy insight.
   Example: "For heat-then-place tasks, you must hold the object in hand when using 'heat X with microwave' — do not put it inside first."
2. "content_body": Self-contained, step-by-step workflow for this task type with action syntax, decision points, and pitfall warnings.

{{
    "memory_description": "string",
    "content_body": "string"
}}
"""

TaskTree_Prompt_Map['root_failure'] = """You attempted a household task but FAILED.
Your job: Generate a **corrective task strategy** so future agents avoid the same mistake.

**Focus on TASK STRATEGY — what went wrong in the workflow logic:**
- What type of task is this?
- What was the wrong action, missing step, or incorrect action syntax?
- What is the correct workflow?

**Full Scenario:**
Environment: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps})

Trajectory:
{trajectory}

**CRITICAL: Self-Contained Output**
Your output will be stored and later shown to a future agent on a DIFFERENT but similar task.
That future agent will NOT see the current trajectory, environment, or any memory chain.
Rules:
1. "content_body" must be fully understandable on its own — no references to "the above", "the retrieved memory", etc.
2. Mention the task type explicitly (e.g., "for tasks requiring heating an object and placing it somewhere").
3. Include specific action syntax rules where relevant (e.g., "use 'heat X with microwave' while holding X in hand").
4. Be concrete and actionable — include step-by-step instructions with decision points and pitfalls.

**Output (JSON):**
1. "memory_description": One sentence: task type + what went wrong + how to fix it.
2. "content_body": Self-contained corrective guide with correct action sequence and syntax.

{{
    "memory_description": "string",
    "content_body": "string"
}}
"""

TaskTree_Prompt_Map['node_success'] = """You successfully completed a household task. There are existing task strategy memories stored.
Your job: identify what **NEW strategic insight** this experience adds that is NOT already covered.

=== EXISTING TASK MEMORIES (already stored — DO NOT REPEAT any of this) ===
{retrieved_task_memory}
=== END OF EXISTING MEMORIES ===

**Current Experience:**
Environment: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps})

Trajectory:
{trajectory}

**Residual Generation Instructions:**
1. READ the existing memories above carefully. List (mentally) what they already cover.
2. ANALYZE the current trajectory. Find knowledge that is genuinely NEW:
   - A different action syntax rule not mentioned in existing memories
   - An edge case or failure-recovery pattern not covered
   - A more efficient workflow variant
   - A new precondition or verification step
3. Your output MUST contain ONLY the new incremental knowledge.
   DO NOT repeat, rephrase, or summarize anything from the existing memories.
4. If you find yourself writing something similar to an existing memory, STOP and think of what is truly different.

**CRITICAL: Self-Contained Output**
Your output will be stored and later shown to a future agent on a DIFFERENT but similar task.
That future agent will NOT see the current trajectory, environment, or any memory chain.
Rules:
1. "content_body" must be fully understandable on its own — no references to "the above", "the retrieved memory", etc.
2. Mention the task type explicitly (e.g., "for tasks requiring heating an object and placing it somewhere").
3. Include specific action syntax rules where relevant (e.g., "use 'heat X with microwave' while holding X in hand").
4. Be concrete and actionable — include step-by-step instructions with decision points and pitfalls.

**Output (JSON):**
1. "memory_description": One sentence about the NEW insight only. Must clearly differ from all existing memory descriptions.
2. "content_body": Self-contained new advice. Readable without the existing memories.

{{
    "memory_description": "string",
    "content_body": "string"
}}
"""

TaskTree_Prompt_Map['node_failure'] = """You attempted a household task but FAILED despite existing task strategy memories.
Your job: identify the **specific gap** in existing strategies that caused the failure.

=== EXISTING TASK MEMORIES (already stored — DO NOT REPEAT any of this) ===
{retrieved_task_memory}
=== END OF EXISTING MEMORIES ===

**Current Experience:**
Environment: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps})

Trajectory:
{trajectory}

**Residual Generation Instructions:**
1. READ the existing memories. What strategies do they recommend?
2. ANALYZE the failure. At which step did things go wrong? Why didn't existing strategies prevent it?
3. Identify the SPECIFIC gap — what rule, edge case, or situation is NOT covered?
4. Your output MUST contain ONLY the gap-filling correction.
   DO NOT repeat, rephrase, or summarize anything from the existing memories.

**CRITICAL: Self-Contained Output**
Your output will be stored and later shown to a future agent on a DIFFERENT but similar task.
That future agent will NOT see the current trajectory, environment, or any memory chain.
Rules:
1. "content_body" must be fully understandable on its own — no references to "the above", "the retrieved memory", etc.
2. Mention the task type explicitly (e.g., "for tasks requiring heating an object and placing it somewhere").
3. Include specific action syntax rules where relevant (e.g., "use 'heat X with microwave' while holding X in hand").
4. Be concrete and actionable — include step-by-step instructions with decision points and pitfalls.

**Output (JSON):**
1. "memory_description": One sentence: the specific gap and correction. Must differ from existing descriptions.
2. "content_body": Self-contained corrective rule. Readable without the existing memories.

{{
    "memory_description": "string",
    "content_body": "string"
}}
"""


# =================================================================
# 环境树反思 Prompt
# =================================================================

EnvTree_Prompt_Map = {}

EnvTree_Prompt_Map['root_success'] = """You completed a task in a household environment.
Your job: Extract **environment-adaptive knowledge** — practical knowledge for operating in this type of environment.

**Your output should cover TWO aspects:**

A) **Environment Layout Knowledge:**
   - What type of environment is this? (kitchen, bathroom, bedroom, etc.)
   - What receptacles are present? Which are open surfaces vs. closed containers?
   - Where were specific objects found? (object-location patterns)
   - What is the efficient search order for finding objects in this environment?

B) **Environment-Specific Operation Rules (IMPORTANT — this is what makes your output valuable):**
   Based on the trajectory, extract operational rules that are specific to how this environment works:
   - How do appliances work in this environment? (e.g., 'to heat with microwave: hold object in hand, go to microwave, use "heat X with microwave" — do NOT put the object inside first')
   - Which receptacles must be opened before interaction? (e.g., fridge, cabinets must be opened; countertops are open)
   - What interaction pitfalls were encountered? (e.g., 'if you put an object inside a container and try to interact from another location, you must "go to" the container first')
   - What is the efficient workflow pattern for this environment type?

**Full Scenario:**
Environment: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps})

Trajectory:
{trajectory}

**CRITICAL: Self-Contained Output**
Your output will be stored and later shown to a future agent in a SIMILAR environment.
That future agent will NOT see the current environment description, trajectory, or any memory chain.
Rules:
1. "content_body" must be fully understandable on its own — no references to "the above", "the retrieved memory", etc.
2. Describe the environment type explicitly (e.g., "in kitchen environments with 10+ cabinets, a fridge, countertops, and a microwave").
3. Include BOTH:
   a) Object-location patterns (where things are typically found)
   b) Environment-specific operation rules and pitfalls (e.g., "to heat an object with a microwave, you must hold the object in hand and use 'heat X with microwave' — do NOT put the object inside the microwave first, or it will fail with 'Nothing happens'")
4. Be concrete: mention specific receptacle types, interaction rules, and search priorities.

**Output (JSON):**
1. "memory_description": One sentence: environment type + the most important operational insight.
   Example: "In kitchen environments with a microwave, you must hold an object in hand to heat it — putting it inside the microwave first causes failure."
2. "content_body": Self-contained environment knowledge covering BOTH layout patterns AND operation rules/pitfalls.

{{
    "memory_description": "string",
    "content_body": "string"
}}
"""

EnvTree_Prompt_Map['root_failure'] = """You attempted a task in a household environment but FAILED.
Your job: Extract **environment-adaptive warnings** — what environmental factors caused the failure.

**Your output should cover TWO aspects:**

A) **Environment Layout Pitfalls:**
   - Were objects not where expected?
   - Were receptacles in unexpected states (closed when expected open, etc.)?

B) **Environment-Specific Interaction Traps:**
   - What actions failed because of how this environment works?
   - What is the correct way to interact with appliances/receptacles in this environment?
   - What operation rules were violated?

**Full Scenario:**
Environment: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps})

Trajectory:
{trajectory}

**CRITICAL: Self-Contained Output**
Your output will be stored and later shown to a future agent in a SIMILAR environment.
That future agent will NOT see the current environment description, trajectory, or any memory chain.
Rules:
1. "content_body" must be fully understandable on its own — no references to "the above", "the retrieved memory", etc.
2. Describe the environment type explicitly (e.g., "in kitchen environments with 10+ cabinets, a fridge, countertops, and a microwave").
3. Include BOTH:
   a) Object-location patterns (where things are typically found)
   b) Environment-specific operation rules and pitfalls
4. Be concrete: mention specific receptacle types, interaction rules, and search priorities.

**Output (JSON):**
1. "memory_description": One sentence: environment type + the key environmental pitfall.
2. "content_body": Self-contained environment warning covering BOTH layout issues AND interaction traps.

{{
    "memory_description": "string",
    "content_body": "string"
}}
"""

EnvTree_Prompt_Map['node_success'] = """You completed a task in a household environment. There are existing environment knowledge memories stored.
Your job: identify what **NEW environment-adaptive knowledge** this experience adds that is NOT already covered.

=== EXISTING ENVIRONMENT MEMORIES (already stored — DO NOT REPEAT any of this) ===
{retrieved_env_memory}
=== END OF EXISTING MEMORIES ===

**Current Experience:**
Environment: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps})

Trajectory:
{trajectory}

**Residual Generation Instructions:**
1. READ the existing environment memories carefully. List what layout patterns and operation rules they already cover.
2. ANALYZE the current trajectory. Find genuinely NEW environment knowledge:
   - New object-location mappings not previously recorded
   - New receptacle interaction rules or pitfalls discovered
   - New operational patterns for this environment type (e.g., a workflow shortcut)
   - New search priority insights
3. Your output MUST contain ONLY the new incremental knowledge.
   DO NOT repeat, rephrase, or summarize anything from the existing memories.
4. Remember to include both layout knowledge AND operational tips if the trajectory reveals new ones.

**CRITICAL: Self-Contained Output**
Your output will be stored and later shown to a future agent in a SIMILAR environment.
That future agent will NOT see the current environment description, trajectory, or any memory chain.
Rules:
1. "content_body" must be fully understandable on its own — no references to "the above", "the retrieved memory", etc.
2. Describe the environment type explicitly (e.g., "in kitchen environments with 10+ cabinets, a fridge, countertops, and a microwave").
3. Include BOTH:
   a) Object-location patterns (where things are typically found)
   b) Environment-specific operation rules and pitfalls
4. Be concrete: mention specific receptacle types, interaction rules, and search priorities.

**Output (JSON):**
1. "memory_description": One sentence about the NEW insight only. Must clearly differ from all existing memory descriptions.
2. "content_body": Self-contained new environment knowledge. Readable without the existing memories.

{{
    "memory_description": "string",
    "content_body": "string"
}}
"""

EnvTree_Prompt_Map['node_failure'] = """You attempted a task in a household environment but FAILED despite existing environment knowledge.
Your job: identify what **environment knowledge gap** caused the failure.

=== EXISTING ENVIRONMENT MEMORIES (already stored — DO NOT REPEAT any of this) ===
{retrieved_env_memory}
=== END OF EXISTING MEMORIES ===

**Current Experience:**
Environment: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps})

Trajectory:
{trajectory}

**Residual Generation Instructions:**
1. READ the existing environment memories. What do they know about this environment type?
2. ANALYZE the failure. Was it caused by:
   - Wrong assumption about object locations?
   - Missed receptacle interaction rule?
   - An environment-specific action pitfall not previously recorded?
3. Identify the SPECIFIC environment gap not covered by existing memories.
4. Your output MUST contain ONLY the gap-filling knowledge.
   DO NOT repeat, rephrase, or summarize anything from the existing memories.

**CRITICAL: Self-Contained Output**
Your output will be stored and later shown to a future agent in a SIMILAR environment.
That future agent will NOT see the current environment description, trajectory, or any memory chain.
Rules:
1. "content_body" must be fully understandable on its own — no references to "the above", "the retrieved memory", etc.
2. Describe the environment type explicitly (e.g., "in kitchen environments with 10+ cabinets, a fridge, countertops, and a microwave").
3. Include BOTH:
   a) Object-location patterns (where things are typically found)
   b) Environment-specific operation rules and pitfalls
4. Be concrete: mention specific receptacle types, interaction rules, and search priorities.

**Output (JSON):**
1. "memory_description": One sentence: the specific environment gap. Must differ from existing descriptions.
2. "content_body": Self-contained environment correction. Readable without the existing memories.

{{
    "memory_description": "string",
    "content_body": "string"
}}
"""


# =================================================================
# Helper 函数
# =================================================================

def get_task_prompt_key(is_root: bool, is_success: bool) -> str:
    if is_root:
        return "root_success" if is_success else "root_failure"
    else:
        return "node_success" if is_success else "node_failure"

def get_env_prompt_key(is_root: bool, is_success: bool) -> str:
    if is_root:
        return "root_success" if is_success else "root_failure"
    else:
        return "node_success" if is_success else "node_failure"
