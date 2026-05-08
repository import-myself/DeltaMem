"""
Dual PR-Tree Prompt Templates for ScienceWorld (v2.0 - Skill Format)

v2.0 核心改进:
- 反思 Prompt 重构为面向 Skill 的格式：activation_condition / execution_procedure / termination_condition
- Root 节点 → Base Skill（基础科学任务技能）
- Residual 节点 → Skill Delta（技能修正残差）
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
# 任务树反思 Prompt（ScienceWorld 版）— Skill 格式
# =================================================================

TaskTree_Prompt_Map = {}

TaskTree_Prompt_Map['root_success'] = """You are a Skill Extractor. Based on this successful scientific experiment trajectory, extract a **Base Skill** for this science task type.

**Full Scenario:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps}, Reward: {reward:.2f}/1.0)

Trajectory:
{trajectory}

**Output Requirements:**
Your output will be placed in a global skill cache and triggered DIRECTLY with NO access to this trajectory.

Output a **self-contained Base Skill** as JSON:
- `activation_condition`: The science task TYPE precisely (e.g., "for boiling tasks where the goal is to boil water or a liquid"). Include all prerequisites (required rooms, equipment needed).
- `execution_procedure`: Complete self-contained step-by-step procedure: (1) where to find items, (2) exact action sequence with action syntax (e.g., "use thermometer on OBJ"), (3) critical preconditions and pitfalls.
- `termination_condition`: When this skill is complete (e.g., "target substance has reached required state / reward > 0 confirmed").

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

TaskTree_Prompt_Map['root_failure'] = """You are a Skill Extractor. Based on this FAILED scientific experiment, extract a **corrective Base Skill** to prevent future agents from repeating the mistake.

**Full Scenario:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps}, Reward: {reward:.2f}/1.0)
Note: Reward > 0 means some sub-goals were completed correctly before failure.

Trajectory:
{trajectory}

**Output Requirements:**
Output a **self-contained corrective Base Skill** as JSON:
- `activation_condition`: Task type + what went wrong (e.g., "for boiling tasks where agent failed to use correct container or heat source").
- `execution_procedure`: Corrected procedure with explicit error-avoidance rules. If reward > 0, identify which sub-goals succeeded and what broke down.
- `termination_condition`: When this corrective skill is complete.

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

TaskTree_Prompt_Map['node_success'] = """You are a Skill Delta Extractor. Extract the **Science Task Skill Delta** — new strategic knowledge NOT covered by existing memories.

=== EXISTING TASK SKILL MEMORIES (already stored — DO NOT REPEAT) ===
{retrieved_task_memory}
=== END ===

**Current Experience:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps}, Reward: {reward:.2f}/1.0)

Trajectory:
{trajectory}

**Output Requirements:**
1. READ existing memories. What task types and procedures do they cover?
2. FIND genuinely NEW knowledge: different action syntax, edge case, efficiency improvement, or new precondition.
3. Output ONLY the new Skill Delta as JSON.

- `activation_condition`: The SPECIFIC NEW condition that activates this delta. Must differ from existing.
- `execution_procedure`: NEW steps/rules only. Self-contained, no references to existing memories.
- `termination_condition`: When this delta modification is complete.

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

TaskTree_Prompt_Map['node_failure'] = """You are a Skill Delta Extractor. Identify the **gap in existing science task skills** that caused this failure.

=== EXISTING TASK SKILL MEMORIES (already stored — DO NOT REPEAT) ===
{retrieved_task_memory}
=== END ===

**Current Experience:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps}, Reward: {reward:.2f}/1.0)
Note: Reward > 0 means some sub-goals were completed correctly before failure.

Trajectory:
{trajectory}

**Output Requirements:**
1. READ existing skills. What do they recommend?
2. IDENTIFY the specific gap. If reward > 0, identify which sub-goals succeeded before breakdown.
3. Output ONLY the corrective Skill Delta as JSON.

- `activation_condition`: The specific new situation existing skills failed to handle.
- `execution_procedure`: Corrective procedure filling the gap. Self-contained.
- `termination_condition`: When this corrective delta is complete.

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""


# =================================================================
# 环境树反思 Prompt（ScienceWorld 版）— Skill 格式
# =================================================================

EnvTree_Prompt_Map = {}

EnvTree_Prompt_Map['root_success'] = """You are a Skill Extractor for Environment Knowledge. Extract a **Base Environment Skill** for navigating this ScienceWorld environment.

**Full Scenario:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps}, Reward: {reward:.2f}/1.0)

Trajectory:
{trajectory}

**Output Requirements:**
Your output will be triggered DIRECTLY in similar environments with NO access to this trajectory.

Output a **self-contained Base Environment Skill** as JSON:
- `activation_condition`: The environment type + key features (e.g., "in ScienceWorld environments containing a kitchen with a stove, a foundry with a metal pot, and a greenhouse").
- `execution_procedure`: Complete self-contained environment knowledge: (A) item-room location patterns (what is where), (B) device operation rules and action syntax, (C) efficient search order, (D) common pitfalls.
- `termination_condition`: When environment navigation is complete (e.g., "required items located and retrieved; ready for experiment actions").

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

EnvTree_Prompt_Map['root_failure'] = """You are a Skill Extractor for Environment Knowledge. Extract a **corrective Base Environment Skill** from this failed experiment.

**Full Scenario:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps}, Reward: {reward:.2f}/1.0)
Note: Reward > 0 means some sub-goals were completed correctly before failure.

Trajectory:
{trajectory}

**Output Requirements:**
Output a **self-contained corrective Base Environment Skill** as JSON:
- `activation_condition`: Environment type + the specific trap/pitfall that caused failure.
- `execution_procedure`: Corrective environment knowledge. If reward > 0, analyze which environment interactions succeeded before failure.
- `termination_condition`: When environment-specific pitfalls have been addressed.

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

EnvTree_Prompt_Map['node_success'] = """You are a Skill Delta Extractor for Environment Knowledge. Extract the **Environment Skill Delta** — new knowledge NOT covered by existing memories.

=== EXISTING ENVIRONMENT SKILL MEMORIES (already stored — DO NOT REPEAT) ===
{retrieved_env_memory}
=== END ===

**Current Experience:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps}, Reward: {reward:.2f}/1.0)

Trajectory:
{trajectory}

**Output Requirements:**
1. READ existing memories. What item locations and device rules do they cover?
2. FIND genuinely NEW environment knowledge: new item-room mapping, new device rule, new pitfall.
3. Output ONLY the new Environment Skill Delta as JSON.

- `activation_condition`: The specific new environment condition that activates this delta.
- `execution_procedure`: NEW environment knowledge only. Self-contained.
- `termination_condition`: When this environment adaptation is complete.

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

EnvTree_Prompt_Map['node_failure'] = """You are a Skill Delta Extractor for Environment Knowledge. Identify the **environment knowledge gap** that caused this failure.

=== EXISTING ENVIRONMENT SKILL MEMORIES (already stored — DO NOT REPEAT) ===
{retrieved_env_memory}
=== END ===

**Current Experience:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps}, Reward: {reward:.2f}/1.0)
Note: Reward > 0 means some sub-goals were completed correctly before failure.

Trajectory:
{trajectory}

**Output Requirements:**
1. READ existing memories.
2. IDENTIFY the gap. If reward > 0, identify which environment interactions succeeded before failure.
3. Output ONLY the corrective Environment Skill Delta as JSON.

- `activation_condition`: The specific new environment situation existing memories failed to cover.
- `execution_procedure`: Corrective environment rule. Self-contained.
- `termination_condition`: When this environment correction is complete.

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
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
