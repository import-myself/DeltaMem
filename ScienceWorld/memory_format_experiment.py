"""
Memory Format Experiment
用同一条源轨迹测试3种extraction格式在不同variation上的得分
- 从源轨迹提取记忆
- 注入不同variation的episode（不更新树，纯测试）
- 对比格式A/B/C vs no-memory
"""

import json, os, sys, re, time, argparse, logging

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
# ScienceWorld 目录必须在最前面，避免被父目录同名模块覆盖
for _p in [_THIS_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, _THIS_DIR)  # 确保 ScienceWorld/ 在最前

from common.llm_client import LLMClient
from agent_sciworld_dual import DualTreeSciWorldAgent
from scienceworld import ScienceWorldEnv

# 用绝对路径加载 utils，避免 sys.path 歧义
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("sw_utils", os.path.join(_THIS_DIR, "utils.py"))
_sw_utils = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sw_utils)
scienceworld_monkey_patch = _sw_utils.sciworld_monkey_patch

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── 源轨迹 ────────────────────────────────────────────────────────────────
TRAJ_DIR = os.path.join(os.path.dirname(__file__), "trajectories/dev-sciworld-deepseek-v4-flash-prtree")

SOURCES = {
    "conductivity": ("11_task-2a-test-conductivity_var528.json", "task-2a-test-conductivity", 528),
    "find_plant":   ("101_task-3-find-plant_var164.json",         "task-3-find-plant",         164),
}

TESTS = {
    "conductivity": [
        ("task-2a-test-conductivity", 579),
        ("task-2a-test-conductivity", 474),
        ("task-2a-test-conductivity", 662),
    ],
    "find_plant": [
        ("task-3-find-plant", 154),
        ("task-3-find-plant", 175),
        ("task-3-find-plant", 209),
    ],
}

# ── 提取 trajectory string（与 agent 内部一致）──────────────────────────
def build_traj_str(traj_path: str) -> tuple[str, str]:
    with open(traj_path) as f:
        msgs = json.load(f)
    first = msgs[0]["content"]
    task_desc = first.split("Now, it's your turn")[-1].strip()
    task_desc = re.sub(r"Task Description:\s*", "", task_desc).strip()

    traj_lines = []
    for m in msgs[1:]:
        if not isinstance(m, dict) or "role" not in m:
            continue
        if m["role"] == "user":
            traj_lines.append(f"Observation: {m['content'][:300]}")
        else:
            traj_lines.append(m["content"][:300])
    return task_desc, "\n".join(traj_lines)


# ── 三种 extraction 格式 ──────────────────────────────────────────────────

FORMAT_A = """You are a Skill Extractor. Extract a reusable Base Skill from this successful science experiment trajectory.

Task: {task_description}
Result: SUCCESS

{trajectory}

Output a self-contained Base Skill. Output ONLY the JSON:
{{
    "activation_condition": "The science task TYPE and distinguishing features. ≤ 30 words.",
    "execution_procedure": "Step-by-step procedure using exact ScienceWorld action syntax (teleport to LOC / pick up OBJ / focus on OBJ / use OBJ on OBJ / activate OBJ / connect OBJ to OBJ / pour OBJ into OBJ / mix OBJ). Include: (1) SPECIFIC rooms and items observed in this episode — ScienceWorld item placements are consistent across episodes so these are reliable hints; (2) the ACTION SEQUENCE in order; (3) critical preconditions or pitfalls to avoid.",
    "termination_condition": "When the skill is complete."
}}"""

FORMAT_B = """You are a Skill Extractor. Extract a reusable Base Skill from this successful science experiment trajectory.

Task: {task_description}
Result: SUCCESS

{trajectory}

Output a self-contained Base Skill. For execution_procedure use this EXACT structure:
LOCATIONS: list each key item and the specific room it was found in, e.g. "battery → workshop table, red wire → workshop table, plant → greenhouse flower pot"
STEPS: numbered action sequence using exact ScienceWorld syntax, e.g. "1. teleport to workshop  2. pick up battery  3. ..."
PITFALLS: 1-2 critical mistakes to avoid

Output ONLY the JSON:
{{
    "activation_condition": "The science task TYPE and distinguishing features. ≤ 30 words.",
    "execution_procedure": "LOCATIONS: ...\\nSTEPS: 1. ... 2. ... 3. ...\\nPITFALLS: ...",
    "termination_condition": "When the skill is complete."
}}"""

FORMAT_C = """You are a Skill Extractor. Extract a reusable Base Skill from this successful science experiment trajectory.

Task: {task_description}
Result: SUCCESS

{trajectory}

Copy the EXACT action commands that solved this task verbatim from the trajectory, as a numbered list. Then add: (a) which specific rooms/items were used, and (b) which parts to adapt for different variations of the same task type.

Output ONLY the JSON:
{{
    "activation_condition": "The science task TYPE and distinguishing features. ≤ 30 words.",
    "execution_procedure": "EXACT STEPS (from trajectory):\\n1. [exact action]\\n2. [exact action]\\n...\\nADAPT: [what changes across variations] ROOMS: [item→room mappings]",
    "termination_condition": "When the skill is complete."
}}"""

FORMATS = {"A": FORMAT_A, "B": FORMAT_B, "C": FORMAT_C}

# ── 调用 LLM 提取记忆 ─────────────────────────────────────────────────────
def extract_memory(llm: LLMClient, fmt_template: str, task_desc: str, traj_str: str) -> dict:
    prompt = fmt_template.format(task_description=task_desc, trajectory=traj_str[:8000])
    resp = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=2048)
    m = re.search(r'\{.*\}', resp, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}

# ── 渲染为 external_memory_str ────────────────────────────────────────────
def render_memory(skill: dict, fmt_name: str) -> str:
    if not skill:
        return ""
    act = skill.get("activation_condition", "")
    proc = skill.get("execution_procedure", "")
    term = skill.get("termination_condition", "")
    return (
        f"# Task Skill Memory (Format {fmt_name})\n"
        f"Use the following skill as a guide for your current task. Adapt item names and room locations to your current observation.\n\n"
        f"### [Base Skill] [✅ SUCCESS]\n"
        f"- **Trigger**: {act}\n"
        f"- **Procedure**: {proc}\n"
        f"- **Done when**: {term}\n"
    )

# ── 运行单个 episode ──────────────────────────────────────────────────────
def run_one(agent, env, task_name, var_idx, ext_mem=None):
    env.load(task_name, var_idx)
    messages = agent.run_episode(
        env=env,
        task_name=task_name,
        variation_idx=var_idx,
        no_memory=(ext_mem is None),
        no_prtree_update=True,
        external_memory_str=ext_mem,
    )
    # run_episode 末尾会 append 一个 result dict 到 messages
    for m in reversed(messages):
        if isinstance(m, dict) and "reward" in m:
            return m["reward"]
    return 0.0

# ── 主实验 ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--task-type", default="conductivity",
                        choices=["conductivity", "find_plant"])
    args = parser.parse_args()

    os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"
    os.environ["DEEPSEEK_API_KEY"] = args.api_key

    llm = LLMClient(model_name="deepseek-v4-flash")
    scienceworld_monkey_patch()
    env = ScienceWorldEnv()

    agent = DualTreeSciWorldAgent(agent_name="mem_exp", llm_client=llm)

    task_type = args.task_type
    src_file, src_task, src_var = SOURCES[task_type]
    traj_path = os.path.join(TRAJ_DIR, src_file)
    task_desc, traj_str = build_traj_str(traj_path)

    print(f"\n=== Source: {src_task} var={src_var} ===")
    print(f"Task: {task_desc[:100]}")
    print(f"Traj tokens (approx): {len(traj_str.split())}\n")

    # 提取3种格式的记忆
    memories = {}
    for fmt_name, fmt_template in FORMATS.items():
        print(f"Extracting Format {fmt_name}...")
        skill = extract_memory(llm, fmt_template, task_desc, traj_str)
        memories[fmt_name] = render_memory(skill, fmt_name)
        proc_preview = skill.get("execution_procedure", "")[:200]
        print(f"  execution_procedure preview: {proc_preview}\n")

    # 跑测试episode
    test_cases = TESTS[task_type]
    results = {fmt: [] for fmt in [*FORMATS.keys(), "no_memory"]}

    print("=" * 60)
    print(f"{'Episode':<35} {'no_mem':>7} {'Fmt_A':>7} {'Fmt_B':>7} {'Fmt_C':>7}")
    print("-" * 60)

    for (task_name, var_idx) in test_cases:
        row = {}
        row["no_memory"] = run_one(agent, env, task_name, var_idx, ext_mem=None)
        for fmt_name, mem_str in memories.items():
            row[fmt_name] = run_one(agent, env, task_name, var_idx, ext_mem=mem_str)

        label = f"{task_name} var={var_idx}"
        print(f"{label:<35} {row['no_memory']:>7.3f} {row['A']:>7.3f} {row['B']:>7.3f} {row['C']:>7.3f}")
        for k, v in row.items():
            results[k].append(v)

    print("-" * 60)
    avgs = {k: sum(v)/len(v) for k, v in results.items() if v}
    print(f"{'AVG':<35} {avgs['no_memory']:>7.3f} {avgs['A']:>7.3f} {avgs['B']:>7.3f} {avgs['C']:>7.3f}")


if __name__ == "__main__":
    main()
