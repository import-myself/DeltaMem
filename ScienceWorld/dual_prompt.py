"""
Dual PR-Tree Prompt Templates for ScienceWorld (v1.0)

针对 ScienceWorld 的双树 Prompt 模板：
- ScienceWorld 是一个科学实验环境，需要完成各种科学任务
- 任务树：存储特定科学任务的操作流程策略（如沸腾、融化、混合化学品等）
- 环境树：存储场景导航经验（各房间的物品分布、设备位置等）
"""

# =================================================================
# ScienceWorld 基础 Instruction
# =================================================================

scienceworld_instruction = """You are a helpful assistant to do some scientific experiment in an environment.
In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway.
You should explore the environment and find the items you need to complete the experiment.
You can teleport to any room in one step.
All containers in the environment have already been opened, you can directly get items from the containers.
For each of your turn, you will be given the observation of the last turn. You should choose from two actions: "Thought" or "Action". If you choose "Thought", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\n Action: your next action"; If you choose "Action", you should directly output the action in this turn. Your output must strictly follow this format:"Action: your next action". Remember that you can only output one "Action:" in per response.

The available actions are:
open OBJ: open a container
close OBJ: close a container
activate OBJ: activate a device
deactivate OBJ: deactivate a device
connect OBJ to OBJ: connect electrical components
disconnect OBJ: disconnect electrical components
use OBJ [on OBJ]: use a device/item
look around: describe the current room
examine OBJ: describe an object in detail
look at OBJ: describe a container's contents
read OBJ: read a note or book
move OBJ to OBJ: move an object to a container
pick up OBJ: move an object to the inventory
pour OBJ into OBJ: pour a liquid into a container
mix OBJ: chemically mix a container
teleport to LOC: teleport to a specific room
focus on OBJ: signal intent on a task object
wait: task no action for 10 steps
wait1: task no action for a step
"""

# =================================================================
# Prompt 模板 (无记忆 / 有记忆)
# =================================================================

PROMPT_WITH_ICL_TEMPLATE = """{instruction}
Here is an example for a complete task trajectory.

{examples}
---

Now, it's your turn and here is the task.
{task}
"""

PROMPT_WITH_ICL_TEMPLATE_DUAL_MEMORY = """{instruction}
Here is an example for a complete task trajectory.

{examples}
---

The following relevant experiences may help you complete the task:

{memory_context}

Now, it's your turn and here is the task.
{task}
"""


# =================================================================
# 任务树反思 Prompt（ScienceWorld 版）
# =================================================================

TaskTree_Prompt_Map = {}

TaskTree_Prompt_Map['root_success'] = """You have successfully completed a scientific experiment task.
Your job: Extract a **general task workflow strategy** that helps future agents solve similar science tasks.

**Focus on TASK STRATEGY — the general workflow applicable across different environments:**
- What type of scientific task is this? (e.g., boiling, melting, freezing, chemistry mixing, measuring, testing conductivity, finding living things, growing plants, etc.)
- What is the correct step-by-step action sequence?
- What critical action syntax or rules must be followed?
- What scientific knowledge or decision points are involved?

**Full Scenario:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps}, Reward: {reward:.2f}/1.0)

Trajectory:
{trajectory}

**CRITICAL: Self-Contained Output**
Your output will be stored and later shown to a future agent on a DIFFERENT but similar science task.
That future agent will NOT see the current trajectory, environment, or any memory chain.
Rules:
1. "content_body" must be fully understandable on its own — no references to "the above", "the retrieved memory", etc.
2. Mention the science task type explicitly (e.g., "for tasks requiring boiling water").
3. Include specific action syntax rules where relevant.
4. Be concrete and actionable — include step-by-step instructions with decision points and pitfalls.

**Output (JSON):**
1. "memory_description": One sentence summarizing the task type and key strategy insight.
2. "content_body": Self-contained, step-by-step workflow for this science task type with action syntax, decision points, and pitfall warnings.

{{
    "memory_description": "string",
    "content_body": "string"
}}
"""

TaskTree_Prompt_Map['root_failure'] = """You attempted a scientific experiment task but FAILED.
Your job: Generate a **corrective task strategy** so future agents avoid the same mistake.

**Focus on TASK STRATEGY — what went wrong in the workflow logic:**
- What type of scientific task is this?
- What was the wrong action, missing step, or incorrect action syntax?
- What is the correct workflow based on scientific principles?

**Full Scenario:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps}, Reward: {reward:.2f}/1.0)
Note: Reward > 0 means some sub-goals were completed in correct order before failure.

Trajectory:
{trajectory}

**CRITICAL: Self-Contained Output**
Your output will be stored and later shown to a future agent on a DIFFERENT but similar science task.
That future agent will NOT see the current trajectory, environment, or any memory chain.
Rules:
1. "content_body" must be fully understandable on its own.
2. Mention the science task type explicitly.
3. Include specific action syntax rules where relevant.
4. Be concrete and actionable. If reward > 0, analyze which sub-goals were completed and where the breakdown occurred.

**Output (JSON):**
1. "memory_description": One sentence: task type + what went wrong + how to fix it.
2. "content_body": Self-contained corrective guide with correct action sequence and syntax.

{{
    "memory_description": "string",
    "content_body": "string"
}}
"""

TaskTree_Prompt_Map['node_success'] = """You successfully completed a scientific task. There are existing task strategy memories stored.
Your job: identify what **NEW strategic insight** this experience adds that is NOT already covered.

=== EXISTING TASK MEMORIES (already stored — DO NOT REPEAT any of this) ===
{retrieved_task_memory}
=== END OF EXISTING MEMORIES ===

**Current Experience:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps}, Reward: {reward:.2f}/1.0)

Trajectory:
{trajectory}

**Residual Generation Instructions:**
1. READ the existing memories carefully. List what they already cover.
2. ANALYZE the current trajectory. Find genuinely NEW knowledge:
   - A different action syntax rule not mentioned
   - An edge case or failure-recovery pattern not covered
   - A more efficient workflow variant
   - A new scientific principle or precondition
3. Your output MUST contain ONLY the new incremental knowledge.
   DO NOT repeat, rephrase, or summarize anything from existing memories.

**CRITICAL: Self-Contained Output**
Rules:
1. "content_body" must be fully understandable on its own.
2. Mention the science task type explicitly.
3. Be concrete and actionable.

**Output (JSON):**
1. "memory_description": One sentence about the NEW insight only.
2. "content_body": Self-contained new advice.

{{
    "memory_description": "string",
    "content_body": "string"
}}
"""

TaskTree_Prompt_Map['node_failure'] = """You attempted a scientific task but FAILED despite existing task strategy memories.
Your job: identify the **specific gap** in existing strategies that caused the failure.

=== EXISTING TASK MEMORIES (already stored — DO NOT REPEAT any of this) ===
{retrieved_task_memory}
=== END OF EXISTING MEMORIES ===

**Current Experience:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps}, Reward: {reward:.2f}/1.0)
Note: Reward > 0 means some sub-goals were completed in correct order before failure.

Trajectory:
{trajectory}

**Residual Generation Instructions:**
1. READ the existing memories. What strategies do they recommend?
2. ANALYZE the failure. At which step did things go wrong? If reward > 0, identify which sub-goals succeeded and where the breakdown occurred.
3. Identify the SPECIFIC gap — what rule, edge case, or scientific principle is NOT covered?
4. Your output MUST contain ONLY the gap-filling correction.

**CRITICAL: Self-Contained Output**
Rules:
1. "content_body" must be fully understandable on its own.
2. Mention the science task type explicitly.
3. Be concrete and actionable.

**Output (JSON):**
1. "memory_description": One sentence: the specific gap and correction.
2. "content_body": Self-contained corrective rule.

{{
    "memory_description": "string",
    "content_body": "string"
}}
"""


# =================================================================
# 环境树反思 Prompt（ScienceWorld 版）
# =================================================================

EnvTree_Prompt_Map = {}

EnvTree_Prompt_Map['root_success'] = """You completed a scientific task in a ScienceWorld environment.
Your job: Extract **environment-adaptive knowledge** — practical knowledge for navigating and operating in this type of environment.

**Your output should cover TWO aspects:**

A) **Environment Layout Knowledge:**
   - What rooms are present and what items/equipment are in each?
   - Where were specific items found? (item-room patterns)
   - What is the efficient search order for finding items needed for science tasks?

B) **Environment-Specific Operation Rules (IMPORTANT):**
   - How do specific devices work in this environment? (e.g., thermometer, bunsen burner, electrical components)
   - What interaction pitfalls were encountered?
   - What is the efficient workflow pattern for operating in this environment?

**Full Scenario:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps}, Reward: {reward:.2f}/1.0)

Trajectory:
{trajectory}

**CRITICAL: Self-Contained Output**
Your output will be stored and later shown to a future agent in a SIMILAR environment.
That future agent will NOT see the current environment description, trajectory, or any memory chain.
Rules:
1. "content_body" must be fully understandable on its own.
2. Describe the environment type explicitly (e.g., "in a ScienceWorld environment with a kitchen, greenhouse, and workshop").
3. Include BOTH:
   a) Item-location patterns (where things are typically found)
   b) Environment-specific operation rules and pitfalls
4. Be concrete: mention specific rooms, device interaction rules, and search priorities.

**Output (JSON):**
1. "memory_description": One sentence: environment type + the most important operational insight.
2. "content_body": Self-contained environment knowledge covering BOTH layout patterns AND operation rules/pitfalls.

{{
    "memory_description": "string",
    "content_body": "string"
}}
"""

EnvTree_Prompt_Map['root_failure'] = """You attempted a scientific task in a ScienceWorld environment but FAILED.
Your job: Extract **environment-adaptive warnings** — what environmental factors caused the failure.

**Your output should cover TWO aspects:**

A) **Environment Layout Pitfalls:**
   - Were items not where expected?
   - Were devices in unexpected states?

B) **Environment-Specific Interaction Traps:**
   - What actions failed because of how this environment works?
   - What is the correct way to interact with devices in this environment?

**Full Scenario:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps}, Reward: {reward:.2f}/1.0)
Note: Reward > 0 means some sub-goals were completed in correct order before failure.

Trajectory:
{trajectory}

**CRITICAL: Self-Contained Output**
Rules:
1. "content_body" must be fully understandable on its own.
2. Describe the environment type explicitly.
3. Include BOTH:
   a) Item-location patterns
   b) Environment-specific operation rules and pitfalls. If reward > 0, analyze which environment interactions succeeded and where the environment caused the failure.

**Output (JSON):**
1. "memory_description": One sentence: environment type + the key environmental pitfall.
2. "content_body": Self-contained environment warning.

{{
    "memory_description": "string",
    "content_body": "string"
}}
"""

EnvTree_Prompt_Map['node_success'] = """You completed a scientific task in a ScienceWorld environment. There are existing environment knowledge memories stored.
Your job: identify what **NEW environment-adaptive knowledge** this experience adds that is NOT already covered.

=== EXISTING ENVIRONMENT MEMORIES (already stored — DO NOT REPEAT any of this) ===
{retrieved_env_memory}
=== END OF EXISTING MEMORIES ===

**Current Experience:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps}, Reward: {reward:.2f}/1.0)

Trajectory:
{trajectory}

**Residual Generation Instructions:**
1. READ the existing environment memories carefully.
2. ANALYZE the current trajectory. Find genuinely NEW environment knowledge:
   - New item-location mappings not previously recorded
   - New device interaction rules or pitfalls discovered
   - New operational patterns for this environment
3. Your output MUST contain ONLY the new incremental knowledge.
   DO NOT repeat, rephrase, or summarize anything from existing memories.

**CRITICAL: Self-Contained Output**
Rules:
1. "content_body" must be fully understandable on its own.
2. Describe the environment type explicitly.
3. Be concrete: mention specific rooms, devices, and interaction rules.

**Output (JSON):**
1. "memory_description": One sentence about the NEW insight only.
2. "content_body": Self-contained new environment knowledge.

{{
    "memory_description": "string",
    "content_body": "string"
}}
"""

EnvTree_Prompt_Map['node_failure'] = """You attempted a scientific task in a ScienceWorld environment but FAILED despite existing environment knowledge.
Your job: identify what **environment knowledge gap** caused the failure.

=== EXISTING ENVIRONMENT MEMORIES (already stored — DO NOT REPEAT any of this) ===
{retrieved_env_memory}
=== END OF EXISTING MEMORIES ===

**Current Experience:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps}, Reward: {reward:.2f}/1.0)
Note: Reward > 0 means some sub-goals were completed in correct order before failure.

Trajectory:
{trajectory}

**Residual Generation Instructions:**
1. READ the existing environment memories.
2. ANALYZE the failure. If reward > 0, identify which sub-goals succeeded in the environment before the failure. Was the remaining failure caused by:
   - Wrong assumption about item locations?
   - Missed device interaction rule?
   - An environment-specific action pitfall not previously recorded?
3. Identify the SPECIFIC gap not covered by existing memories.
4. Your output MUST contain ONLY the gap-filling knowledge.

**CRITICAL: Self-Contained Output**
Rules:
1. "content_body" must be fully understandable on its own.
2. Describe the environment type explicitly.
3. Be concrete.

**Output (JSON):**
1. "memory_description": One sentence: the specific environment gap.
2. "content_body": Self-contained environment correction.

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
