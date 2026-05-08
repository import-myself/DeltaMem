"""
Dual PR-Tree Prompt Templates (v8.0 - Skill Format)

v8.0 核心改进:
- 反思 Prompt 统一重构为面向 Skill 的格式，输出 activation_condition / execution_procedure / termination_condition
- Root 节点 → Base Skill（基础技能提取）
- Residual 节点 → Skill Delta（技能修正残差）
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
# 任务树反思 Prompt (Task Tree) — Skill 格式
# =================================================================

TaskTree_Prompt_Map = {}

TaskTree_Prompt_Map['root_success'] = """You are a Skill Extractor. Based on the following successful task trajectory (starting from scratch), extract a **Base Skill** that encapsulates the complete executable strategy.

**Full Scenario:**
Environment: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps})

Trajectory:
{trajectory}

**Output Requirements:**
Your output will be placed in a global skill cache and triggered DIRECTLY when similar tasks are encountered — with NO access to any trajectory, environment description, or memory chain.

Output a **self-contained Base Skill** as JSON with these three fields:
- `activation_condition`: When to trigger this skill — describe the task TYPE precisely (e.g., "for heat-then-place tasks where the goal is to heat an object and place it on a receptacle"). Include all prerequisites.
- `execution_procedure`: The complete, self-contained step-by-step action sequence. Must specify action syntax explicitly (e.g., "use 'heat X with microwave' while holding X"). Must NOT reference 'the above trajectory' or any external context.
- `termination_condition`: When to consider this skill complete and hand back control (e.g., "task marked as complete by environment" or "object placed on target receptacle").

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

TaskTree_Prompt_Map['root_failure'] = """You are a Skill Extractor. Based on the following FAILED task trajectory, extract a **corrective Base Skill** that prevents future agents from making the same mistake.

**Full Scenario:**
Environment: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps})

Trajectory:
{trajectory}

**Output Requirements:**
Your output will be placed in a global skill cache and triggered DIRECTLY when similar tasks are encountered — with NO access to any trajectory, environment description, or memory chain.

Output a **self-contained corrective Base Skill** as JSON:
- `activation_condition`: Task type this skill applies to, including what went wrong (e.g., "for heat-then-place tasks where agent incorrectly puts object inside appliance first").
- `execution_procedure`: The corrected self-contained action sequence with explicit error-avoidance rules.
- `termination_condition`: When to consider the skill complete.

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

TaskTree_Prompt_Map['node_success'] = """You are a Skill Delta Extractor. Compare the existing skill memories with this new successful trajectory and extract the **Skill Delta** — the incremental modification needed to adapt the existing skill to this new scenario.

=== EXISTING SKILL MEMORIES (already stored — DO NOT REPEAT) ===
{retrieved_task_memory}
=== END ===

**Current Experience:**
Environment: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps})

Trajectory:
{trajectory}

**Output Requirements:**
1. READ existing memories. Identify exactly what they cover.
2. FIND what is genuinely NEW: a different precondition, an additional step, an edge case, or a more efficient variant.
3. Output ONLY the new incremental Skill Delta as a self-contained JSON:
   - `activation_condition`: The SPECIFIC NEW trigger condition — the precise new premise that activates this delta (e.g., "when the target receptacle requires opening before placing"). Must DIFFER from existing conditions.
   - `execution_procedure`: The NEW/modified steps only — what to ADD, REPLACE, or DELETE relative to the base skill. Must be self-contained (readable without the existing memories).
   - `termination_condition`: When this delta's modification is complete and the base skill resumes.

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

TaskTree_Prompt_Map['node_failure'] = """You are a Skill Delta Extractor. Identify the **gap in existing skills** that caused this failure and extract a corrective Skill Delta.

=== EXISTING SKILL MEMORIES (already stored — DO NOT REPEAT) ===
{retrieved_task_memory}
=== END ===

**Current Experience:**
Environment: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps})

Trajectory:
{trajectory}

**Output Requirements:**
1. READ existing skills. What do they recommend?
2. IDENTIFY the specific gap: what rule, edge case, or precondition is NOT covered?
3. Output ONLY the corrective Skill Delta as self-contained JSON:
   - `activation_condition`: The precise new trigger — the situation the existing skills failed to handle.
   - `execution_procedure`: The corrective action sequence that fills the gap. Self-contained, no references to existing memories.
   - `termination_condition`: When this corrective delta is complete.

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""


# =================================================================
# 环境树反思 Prompt (Env Tree) — Skill 格式
# =================================================================

EnvTree_Prompt_Map = {}

EnvTree_Prompt_Map['root_success'] = """You are a Skill Extractor for Environment Knowledge. Based on this successful trajectory, extract a **Base Environment Skill** covering layout and operation rules for this environment type.

**Full Scenario:**
Environment: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps})

Trajectory:
{trajectory}

**Output Requirements:**
Your output will be triggered DIRECTLY in similar environments with NO access to this trajectory.

Output a **self-contained Base Environment Skill** as JSON:
- `activation_condition`: The environment type + key features that make this skill applicable (e.g., "in kitchen environments with a microwave, countertops, and multiple cabinets").
- `execution_procedure`: Complete self-contained environment knowledge: (A) object-location patterns, (B) appliance/receptacle operation rules, (C) efficient search order and pitfalls.
- `termination_condition`: When environment-adaptive navigation is complete (e.g., "target object located and retrieved; ready for task-specific operations").

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

EnvTree_Prompt_Map['root_failure'] = """You are a Skill Extractor for Environment Knowledge. Based on this FAILED trajectory, extract a **corrective Base Environment Skill** that warns future agents about environmental traps.

**Full Scenario:**
Environment: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps})

Trajectory:
{trajectory}

**Output Requirements:**
Output a **self-contained corrective Base Environment Skill** as JSON:
- `activation_condition`: The environment type + the specific trap condition that triggered this failure.
- `execution_procedure`: Corrective environment knowledge: what went wrong, correct operation rules, pitfall avoidance strategies.
- `termination_condition`: When environment-specific pitfalls have been addressed.

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

EnvTree_Prompt_Map['node_success'] = """You are a Skill Delta Extractor for Environment Knowledge. Extract the **Environment Skill Delta** — new environment knowledge NOT covered by existing memories.

=== EXISTING ENVIRONMENT MEMORIES (already stored — DO NOT REPEAT) ===
{retrieved_env_memory}
=== END ===

**Current Experience:**
Environment: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps})

Trajectory:
{trajectory}

**Output Requirements:**
1. READ existing environment memories. What layout patterns and operation rules do they cover?
2. FIND genuinely NEW knowledge: new object-location mapping, new appliance rule, new pitfall.
3. Output ONLY the new Environment Skill Delta as self-contained JSON:
   - `activation_condition`: The specific new environment condition that activates this delta.
   - `execution_procedure`: The NEW environment knowledge only. Self-contained, no references to existing memories.
   - `termination_condition`: When this environment adaptation is complete.

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

EnvTree_Prompt_Map['node_failure'] = """You are a Skill Delta Extractor for Environment Knowledge. Identify the **environment knowledge gap** that caused this failure.

=== EXISTING ENVIRONMENT MEMORIES (already stored — DO NOT REPEAT) ===
{retrieved_env_memory}
=== END ===

**Current Experience:**
Environment: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps})

Trajectory:
{trajectory}

**Output Requirements:**
1. READ existing environment memories.
2. IDENTIFY the specific gap: wrong object-location assumption, missed appliance rule, or uncovered interaction trap.
3. Output ONLY the corrective Environment Skill Delta as self-contained JSON:
   - `activation_condition`: The specific new environment situation the existing memories failed to cover.
   - `execution_procedure`: The corrective environment rule. Self-contained.
   - `termination_condition`: When this environmental correction is applied.

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
