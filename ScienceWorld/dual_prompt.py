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

{memory_header}

{memory_context}

Now, it's your turn and here is the task.
{task}
"""

MEMORY_HEADERS: dict = {
    "prtree": (
        "Retrieved from your hierarchical memory system:\n"
        "• Task Skill Memory — HOW to perform this science task type (Base Skill + Skill Deltas as patches)\n"
        "• Environment Knowledge Memory — WHERE items are located and HOW devices/equipment operate\n"
        "Note: 'Trigger'/'Applicable scenario' labels describe PAST episodes — your actual task is at the end."
    ),
    "synapse": (
        "The following past experiment trajectories are retrieved from memory as few-shot examples. "
        "Use them as reference if the experimental pattern is similar to the current one:"
    ),
    "awm": (
        "The following workflow procedure was distilled from past similar science experiments. "
        "Follow it as a step-by-step guide if the experiment type matches:"
    ),
    "reasoningbank": (
        "The following memory items are distilled lessons from past science experiment interactions. "
        "[✅ SUCCESS] entries describe experimental strategies that worked; "
        "[⚠️ FAILURE] entries highlight mistakes to avoid. "
        "Apply the relevant insights to guide your experiment:"
    ),
}


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
- `activation_condition`: The science task TYPE precisely (e.g., for boiling tasks where the goal is to boil water or a liquid). Include all prerequisites (required rooms, equipment needed).
- `execution_procedure`: Complete self-contained step-by-step procedure: (1) where to find items, (2) exact action sequence with action syntax (e.g., use thermometer on OBJ), (3) critical preconditions and pitfalls.
- `termination_condition`: When this skill is complete (e.g., target substance has reached required state / reward > 0 confirmed).

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

TaskTree_Prompt_Map['root_failure'] = """You are a Failure Recorder. This output will be stored as a FAILURE RECORD — it is NOT a skill to execute. It will be shown to future agents with a ⛔ warning so they know what to avoid.

**Full Scenario:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: FAILURE (Steps: {steps}, Reward: {reward:.2f}/1.0)
Note: Reward > 0 means some sub-goals were completed correctly before failure.

Trajectory:
{trajectory}

**Output Requirements:**
- `activation_condition`: Task type + the specific wrong assumption or failure condition. Format: "Applicable when [task_type_tag] ..."
- `execution_procedure`: Use exactly these two labeled sections:
  [FAILED]: Actions attempted and exact environment responses showing failure. If reward > 0, identify which sub-goals succeeded before breakdown.
  [UNEXPLORED]: Plausible approaches this agent never attempted — prevents future agents from treating this failure as proof those approaches are impossible.
- `termination_condition`: Leave empty string.

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "[FAILED]: ...\n[UNEXPLORED]: ...",
    "termination_condition": ""
}}
"""

TaskTree_Prompt_Map['node_success'] = """You are a Skill Delta Extractor. Extract ONLY the minimal new patch not covered by existing skills.

=== EXISTING TASK SKILL MEMORIES (already stored — DO NOT REPEAT) ===
{retrieved_task_memory}
=== END ===

**Current Experience:**
Environment Description: {env_description}
Task Goal: {task_description}
Result: SUCCESS (Steps: {steps}, Reward: {reward:.2f}/1.0)

Trajectory:
{trajectory}

Output {{"skip": true}} ONLY if you can satisfy ALL of the following, with direct evidence:
1. Identify ONE existing skill whose execution_procedure explicitly lists every distinct action type performed in this trajectory — quote the exact phrase from that skill for each action.
2. No action in this trajectory required a recovery step, a different item category, or a procedural order not covered by that quoted text.
If you cannot quote matching text for even ONE action in this trajectory, you MUST write a delta.

Otherwise, output the smallest delta (1-3 new observations max):
- `activation_condition`: The SPECIFIC NEW condition that activates this delta — must differ from existing triggers.
- `execution_procedure`: NEW steps/rules only. Concrete and self-contained, no references to existing memories.
- `termination_condition`: When this delta modification is complete.

Output ONLY one of these two JSON formats:
{{"skip": true}}
OR
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

TaskTree_Prompt_Map['node_failure'] = """You are a Failure Recorder. Record the gap in existing skills that caused this failure.

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

Output {{"skip": true}} ONLY if an existing failure record describes the EXACT SAME failure: you must quote the specific failed actions and the exact environment responses from that record that match this trajectory. If the failed action sequence or environment response differs in any way, you MUST write a new record.

Otherwise, output the gap as a trap record:
- `activation_condition`: The specific new situation existing skills failed to handle. Format: "Applicable when [task_type_tag] and ..."
- `execution_procedure`: Use exactly these two labeled sections:
  [FAILED]: Actions attempted and exact environment responses. Note which existing skill guidance was followed but did not work. If reward > 0, identify which sub-goals succeeded before breakdown.
  [UNEXPLORED]: Plausible approaches this agent never attempted — prevents future agents from treating this failure as proof those approaches are impossible.
- `termination_condition`: Leave empty string.

Output ONLY one of these two JSON formats:
{{"skip": true}}
OR
{{
    "activation_condition": "...",
    "execution_procedure": "[FAILED]: ...\n[UNEXPLORED]: ...",
    "termination_condition": ""
}}
"""


# =================================================================
# 环境树反思 Prompt（ScienceWorld 版）— Skill 格式
# =================================================================

EnvTree_Prompt_Map = {}

EnvTree_Prompt_Map['root'] = """You are an Environment Knowledge Extractor. Extract declarative **Base Environment Knowledge** for navigating this ScienceWorld environment — regardless of whether the task succeeded or failed, the environmental observations are valid knowledge.

**Full Scenario:**
Environment Description: {env_description}
Task Goal: {task_description}
Outcome: Steps={steps}, Reward={reward:.2f}/1.0

Trajectory:
{trajectory}

**Output Requirements:**
This is DECLARATIVE KNOWLEDGE — facts about the world, not a procedure to execute.
Your output will be retrieved in similar environments with NO access to this trajectory.

Output **self-contained Base Environment Knowledge** as JSON:
- `activation_condition`: Environment type + key structural features. Format: "Applicable in [environment_type] environments where ..."
- `execution_procedure`: Factual observations only — (A) item-room location patterns observed (what is where), (B) device operation rules and action syntax from environment feedback, (C) efficient search order, (D) common pitfalls. Write as observations ("items tend to be in..."), NOT as commands.
- `termination_condition`: When this environment knowledge has been fully applied.

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

EnvTree_Prompt_Map['node'] = """You are an Environment Knowledge Extractor. Extract ONLY new environment facts not covered by existing knowledge — regardless of task outcome.

=== EXISTING ENVIRONMENT KNOWLEDGE (already stored — DO NOT REPEAT) ===
{retrieved_env_memory}
=== END ===

**Current Experience:**
Environment Description: {env_description}
Task Goal: {task_description}
Outcome: Steps={steps}, Reward={reward:.2f}/1.0

Trajectory:
{trajectory}

Output {{"skip": true}} ONLY when EVERY item-room mapping AND EVERY device/equipment interaction rule observed in this trajectory is already explicitly stated in the existing knowledge above. If even ONE location pattern or device rule is not explicitly covered, you MUST write an update.

Otherwise, output the smallest new update (1-3 new facts max):
- `activation_condition`: The specific new environment condition. Format: "Applicable in [environment_type] environments where ..."
- `execution_procedure`: NEW declarative observations only — new item-room tendencies or device rules not in existing knowledge. Written as facts ("items tend to be in..."), not commands.
- `termination_condition`: When this environment adaptation is complete.

Output ONLY one of these two JSON formats:
{{"skip": true}}
OR
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

def get_env_prompt_key(is_root: bool, is_success: bool = True) -> str:
    return "root" if is_root else "node"
