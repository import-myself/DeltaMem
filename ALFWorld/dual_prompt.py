"""
Dual PR-Tree Prompt Templates for ALFWorld (v11.0 - Lean)

设计原则:
- Root: 冷启动，从轨迹提取完整自包含 skill/knowledge
- Node: 增量，只提取与已有记忆不同的最小 delta
- Failure: 记录陷阱，不是可执行步骤
- 不用 Rules 列表，直接在字段描述里说清要求
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

{memory_header}

{memory_context}

Now, it's your turn and here is the task.
{task}
"""

MEMORY_HEADERS: dict = {
    "prtree": (
        "Retrieved from your hierarchical memory system:\n"
        "• Task Skill Memory — HOW to solve this task type (Base Skill + Skill Deltas as patches)\n"
        "• Environment Knowledge Memory — WHERE objects are and HOW to operate receptacles/appliances\n"
        "Note: 'Trigger'/'Applicable scenario' labels describe PAST episodes — your actual task is at the end."
    ),
    "synapse": (
        "The following past task trajectories are retrieved from memory as few-shot examples. "
        "Use them as reference if the task pattern is similar to the current one:"
    ),
    "awm": (
        "The following workflow procedure was distilled from past similar household tasks. "
        "Follow it as a step-by-step guide if the task type matches:"
    ),
    "reasoningbank": (
        "The following memory items are distilled lessons from past household task interactions. "
        "[✅ SUCCESS] entries describe strategies that worked; "
        "[⚠️ FAILURE] entries highlight mistakes to avoid. "
        "Apply the relevant insights to guide your actions:"
    ),
}


# =================================================================
# 任务树反思 Prompt (Task Tree)
# =================================================================

TaskTree_Prompt_Map = {}

TaskTree_Prompt_Map['root_success'] = """You are a Skill Extractor. Extract a reusable Base Skill from this successful trajectory.

Environment: {env_description}
Task: {task_description}
Result: SUCCESS ({steps}/{max_steps} steps)

{trajectory}

Output a self-contained Base Skill — future agents have NO access to this trajectory:
- `activation_condition`: In plain natural language, describe when this skill applies and what distinguishes it from other tasks.
- `execution_procedure`: Concrete step sequence derived from this trajectory. Write out each individual action — do not collapse multi-step operations into a single abstract phrase.
- `termination_condition`: When the skill is complete.

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
Result: FAILURE ({steps}/{max_steps} steps)

{trajectory}

- `activation_condition`: In plain natural language, describe what task situation this applies to and what wrong assumption caused the failure.
- `execution_procedure`:
  [FAILED]: Actions tried and environment responses showing they failed.
  [UNEXPLORED]: Plausible approaches never attempted.
- `termination_condition`: Leave empty string.

Output ONLY the JSON:
{{
    "activation_condition": "...",
    "execution_procedure": "[FAILED]: ...\n[UNEXPLORED]: ...",
    "termination_condition": ""
}}
"""

TaskTree_Prompt_Map['node_success'] = """You are a Skill Delta Extractor. Extract ONLY the minimal new patch not covered by existing skills.

=== EXISTING SKILL MEMORIES ===
{retrieved_task_memory}
=== END ===

Environment: {env_description}
Task: {task_description}
Result: SUCCESS ({steps}/{max_steps} steps)

{trajectory}

Output {{"skip": true}} ONLY if you can satisfy ALL of the following, with direct evidence:
1. Identify ONE existing skill whose execution_procedure explicitly lists every distinct action type performed in this trajectory — quote the exact phrase from that skill for each action.
2. No action in this trajectory required a recovery step, a different object category, or a procedural order not covered by that quoted text.
If you cannot quote matching text for even ONE action in this trajectory, you MUST write a delta.

Otherwise, output the smallest delta (1-3 new observations max):
- `activation_condition`: In plain natural language, describe the specific new condition or variant that makes this delta necessary — must differ from existing triggers.
- `execution_procedure`: New steps only, concrete and self-contained.
- `termination_condition`: When this delta is done.

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

=== EXISTING SKILL MEMORIES ===
{retrieved_task_memory}
=== END ===

Environment: {env_description}
Task: {task_description}
Result: FAILURE ({steps}/{max_steps} steps)

{trajectory}

Output {{"skip": true}} ONLY if an existing failure record describes the EXACT SAME failure: you must quote the specific failed actions and the exact environment responses from that record that match this trajectory. If the failed action sequence or environment response differs in any way, you MUST write a new record.

Otherwise, output the gap as a trap record:
- `activation_condition`: In plain natural language, describe the specific situation that existing skills failed to handle.
- `execution_procedure`:
  [FAILED]: What was tried and why it failed.
  [UNEXPLORED]: Approaches never attempted.
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
# 环境树反思 Prompt (Env Tree)
# =================================================================

EnvTree_Prompt_Map = {}

EnvTree_Prompt_Map['root'] = """You are an Environment Knowledge Extractor. Extract declarative facts about this household environment from the trajectory — regardless of whether the task succeeded or failed, the environmental observations are valid knowledge.

Environment: {env_description}
Task: {task_description}
Outcome: {result} ({steps}/{max_steps} steps)

{trajectory}

Output self-contained Base Environment Knowledge — FACTS about the world, not a procedure:
- `activation_condition`: "Applicable in [environment_type] environments where ..." — key structural features.
- `execution_procedure`: CATEGORY-LEVEL patterns only — (A) what TYPES of objects tend to be in what TYPES of locations (e.g., "condiment-type objects tend to be on countertops or in kitchen drawers"), (B) how receptacle/appliance types behave, (C) pitfalls observed. Write as observations ("X-type objects tend to be on Y"), not commands. ⚠️ Do NOT record specific instances (e.g., "cabinet 3 had saltshaker 1") — ALFWorld randomizes object placement each episode, making specific locations immediately stale. Focus on generalizable object-category → location-type tendencies.
- `termination_condition`: When this knowledge has been fully applied.

Output ONLY the JSON:
{{
    "activation_condition": "Applicable in [environment_type] environments where ...",
    "execution_procedure": "...",
    "termination_condition": "..."
}}
"""

EnvTree_Prompt_Map['node'] = """You are an Environment Knowledge Extractor. Extract ONLY new environment facts not covered by existing knowledge — regardless of whether the task succeeded or failed.

=== EXISTING ENVIRONMENT KNOWLEDGE ===
{retrieved_env_memory}
=== END ===

Environment: {env_description}
Task: {task_description}
Outcome: {result} ({steps}/{max_steps} steps)

{trajectory}

Output {{"skip": true}} ONLY when EVERY specific object location AND EVERY appliance/receptacle interaction rule observed in this trajectory is already explicitly stated in the existing knowledge above. Each new trajectory visits different rooms and finds objects in specific spots — if even ONE location or rule is not explicitly in the existing knowledge, you MUST write an update.

Otherwise, output the smallest new update (1-3 new facts max):
- `activation_condition`: "Applicable in [environment_type] environments where ..." — the new structural feature.
- `execution_procedure`: CATEGORY-LEVEL patterns only — new object-type → location-type tendencies or new appliance behavior rules not in existing knowledge. Write as observations ("X-type objects tend to be on Y"), not commands. ⚠️ Do NOT record specific instances (e.g., "cabinet 3 had saltshaker 1") — ALFWorld randomizes object placement each episode, making specific locations immediately stale.
- `termination_condition`: When this adaptation is complete.

Output ONLY one of these two JSON formats:
{{"skip": true}}
OR
{{
    "activation_condition": "Applicable in [environment_type] environments where ...",
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
