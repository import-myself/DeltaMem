"""
Dual PR-Tree Prompt Templates for ScienceWorld (v3.0)

v3.0 改进:
- Env tree: 允许记录具体房间→物品映射（ScienceWorld 房间固定，不像 ALFWorld 随机）
- Task tree: 格式对齐 ALFWorld lean 风格；强调 ScienceWorld 特有动作语法
- node_success skip 规则：新增 partial reward 约束
- Memory header：更具体描述 ScienceWorld 双树内容
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
        "• Task Skill Memory — HOW to perform this science task type: exact action sequence, which room to find each item, device operation syntax (use/activate/connect), and what to watch out for.\n"
        "• Environment Knowledge Memory — WHERE specific items are located across ScienceWorld rooms, and HOW devices/equipment respond to actions.\n"
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
# 任务树反思 Prompt（ScienceWorld 版）
# =================================================================

TaskTree_Prompt_Map = {}

TaskTree_Prompt_Map['root_success'] = """You are a Skill Extractor. Extract a reusable Base Skill from this successful science experiment trajectory.

Environment: {env_description}
Task: {task_description}
Result: SUCCESS (Steps: {steps}, Reward: {reward:.2f}/1.0)

{trajectory}

Output a self-contained Base Skill — future agents have NO access to this trajectory:
- `activation_condition`: The science task TYPE and distinguishing features (e.g., "boiling tasks where the goal is to heat a liquid until it reaches its boiling point, requiring a heat source and thermometer").
- `execution_procedure`: Concrete step-by-step procedure derived from this trajectory. Write out each action individually using exact ScienceWorld syntax (teleport to LOC / pick up OBJ / use OBJ on OBJ / focus on OBJ / pour OBJ into OBJ / mix OBJ / activate OBJ / connect OBJ to OBJ). Include: (1) which room each item is found in, (2) the full action sequence in order, (3) any critical preconditions or pitfalls.
- `termination_condition`: When the skill is complete (e.g., reward > 0 confirmed / target substance reached required state).

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

TaskTree_Prompt_Map['root_failure'] = """You are a Failure Recorder. This is a FAILURE RECORD — NOT a skill to execute. Future agents will see a ⛔ warning with this.

Environment: {env_description}
Task: {task_description}
Result: FAILURE (Steps: {steps}, Reward: {reward:.2f}/1.0)
Note: Reward > 0 means some sub-goals were completed correctly before failure.

{trajectory}

- `activation_condition`: Task type + the specific wrong assumption or failure condition that caused this outcome.
- `execution_procedure`:
  [FAILED]: Actions attempted and exact environment responses showing failure. If reward > 0, identify which sub-goals succeeded before breakdown.
  [UNEXPLORED]: Plausible approaches this agent never attempted.
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

Environment: {env_description}
Task: {task_description}
Result: SUCCESS (Steps: {steps}, Reward: {reward:.2f}/1.0)

{trajectory}

Output {{"skip": true}} ONLY if you can satisfy ALL of the following with direct evidence:
1. Reward is 1.0 (full success — partial reward means a sub-goal was missed, which requires a delta).
2. Identify ONE existing skill whose execution_procedure explicitly lists every distinct action type performed in this trajectory — quote the exact phrase from that skill for each action.
3. No action in this trajectory required a recovery step, a different item, or a procedural order not covered by that quoted text.
If ANY condition fails, you MUST write a delta.

Otherwise, output the smallest delta (1-3 new observations max):
- `activation_condition`: The SPECIFIC NEW condition that activates this delta — must differ from existing triggers.
- `execution_procedure`: NEW steps/rules only. Concrete and self-contained, using exact ScienceWorld action syntax.
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

Environment: {env_description}
Task: {task_description}
Result: FAILURE (Steps: {steps}, Reward: {reward:.2f}/1.0)
Note: Reward > 0 means some sub-goals were completed correctly before failure.

{trajectory}

Output {{"skip": true}} ONLY if an existing failure record describes the EXACT SAME failure: you must quote the specific failed actions and exact environment responses from that record matching this trajectory. If the failed action sequence or environment response differs in any way, write a new record.

Otherwise, output the gap as a trap record:
- `activation_condition`: The specific situation existing skills failed to handle.
- `execution_procedure`:
  [FAILED]: Actions attempted and exact environment responses. Note which existing skill guidance was followed but did not work. If reward > 0, identify which sub-goals succeeded before breakdown.
  [UNEXPLORED]: Plausible approaches this agent never attempted.
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
# 环境树反思 Prompt（ScienceWorld 版）
# =================================================================
#
# 关键设计原则：ScienceWorld 的房间结构是固定的（kitchen / workshop / greenhouse 等），
# 物品位置跨 episode 基本一致（不像 ALFWorld 每次随机摆放）。
# 因此 env tree 应记录具体的 房间→物品 映射，而非 ALFWorld 式的类目归纳。

EnvTree_Prompt_Map = {}

EnvTree_Prompt_Map['root'] = """You are an Environment Knowledge Extractor. Extract declarative facts about this ScienceWorld environment from the trajectory — regardless of task outcome, the observations are valid knowledge.

Environment: {env_description}
Task: {task_description}
Outcome: Steps={steps}, Reward={reward:.2f}/1.0

{trajectory}

Output self-contained Base Environment Knowledge — FACTS about the world, not a procedure to execute.

Important: Unlike ALFWorld, ScienceWorld rooms have CONSISTENT item placement across episodes. Record SPECIFIC room→item mappings (e.g., "the thermometer is in the kitchen"), not just category-level tendencies.

- `activation_condition`: "Applicable in ScienceWorld environments where ..." — describe the task context or room configuration.
- `execution_procedure`: Factual observations: (A) specific item locations by room (e.g., "thermometer: kitchen", "battery: workshop"), (B) device/equipment operation rules with exact action syntax (e.g., "use thermometer on OBJ to read temperature"), (C) efficient room search order for common items, (D) any pitfalls observed. Write as facts, not commands.
- `termination_condition`: When this environment knowledge has been fully applied.

Output ONLY the JSON:
{{
    "activation_condition": "Applicable in ScienceWorld environments where ...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

EnvTree_Prompt_Map['node'] = """You are an Environment Knowledge Extractor. Extract ONLY new environment facts not covered by existing knowledge — regardless of task outcome.

=== EXISTING ENVIRONMENT KNOWLEDGE (already stored — DO NOT REPEAT) ===
{retrieved_env_memory}
=== END ===

Environment: {env_description}
Task: {task_description}
Outcome: Steps={steps}, Reward={reward:.2f}/1.0

{trajectory}

Important: ScienceWorld rooms have CONSISTENT item placement. Record specific room→item mappings when newly observed.

Output {{"skip": true}} ONLY when EVERY item location AND EVERY device/equipment interaction rule observed in this trajectory is already explicitly stated in the existing knowledge above. If even ONE specific location or device rule is not explicitly covered, you MUST write an update.

Otherwise, output the smallest new update (1-3 new facts max):
- `activation_condition`: "Applicable in ScienceWorld environments where ..." — the new context covered.
- `execution_procedure`: NEW facts only — specific item locations or device rules not in existing knowledge. Written as facts, not commands.
- `termination_condition`: When this environment adaptation is complete.

Output ONLY one of these two JSON formats:
{{"skip": true}}
OR
{{
    "activation_condition": "Applicable in ScienceWorld environments where ...",
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
