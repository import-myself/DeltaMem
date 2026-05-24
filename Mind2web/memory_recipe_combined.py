"""
Combined experiment: 5 recipes × 6 episodes × 3 metrics.

Recipes:
  01_baseline       — no memory
  02_oracle_pitfall — hand-written pitfall (upper bound for pitfall-only)
  03_auto_pitfall   — LLM-generated pitfall from baseline's wrong-step trajectory
  04_step_template  — numbered procedure with element descriptors
  05_combined       — pitfall (highlighted) + first 3 steps of step_template

Metrics: element_acc, action_f1, step_success — all reported per recipe.
"""

import json
import glob
import logging
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_mind2web_dual import DualTreeMind2WebAgent
from common.llm_client import create_llm_client
from mind2web_utils import add_scores
from memory_recipe_multi import EPISODES as ORACLE_RECIPES

logging.getLogger().setLevel(logging.WARNING)

DATA_GLOB = "/hdd/REDACTED_USER/DeltaMem/Mind2web/data/test_website/*.json"
SCORE_PATH = "/hdd/REDACTED_USER/DeltaMem/Mind2web/data/scores_all_data.pkl"
EPISODES_TO_TEST = [132, 131, 137, 149, 88, 122]


AUTO_PITFALL_PROMPT = """You are a Decision Pitfall Extractor for a web navigation agent.

The agent failed at one step. Write ONE short pitfall sentence to help future agents
avoid the same wrong choice in similar situations.

Task: {task}
Website: {website}

Observation when the wrong action was chosen:
```
{observation}
```

WRONG action the agent took:        {pred_act}
RIGHT action it should have taken:  {target_act}

Output ONE JSON object only, no markdown fence:
{{
  "trigger": "Situation: which elements appear together (refer by aria-label/text/role, NO element IDs).",
  "pitfall": "One sentence ≤ 60 words: name the DEFAULT correct element (by semantic attribute) and the wrong one to avoid. NO element IDs."
}}

Rules:
- NEVER write any numeric element ID like [3128]. They change every page.
- Refer to elements ONLY by visible text, aria-label, placeholder, or role.
- Generalize: pitfall must apply to similar tasks on this website type.
"""


def extract_observation(user_content: str) -> str:
    m = re.search(r"Observation: `(.+?)`", user_content, re.DOTALL)
    return m.group(1) if m else user_content


def find_wrong_step_obs(result):
    real_idx = 0
    wrong_real_idx = None
    wrong_line = None
    for line in result.get('trajectory', []):
        if line.startswith('Step '):
            real_idx += 1
            if 'elem_acc=0' in line:
                wrong_real_idx = real_idx
                wrong_line = line
                break
    if wrong_real_idx is None:
        return None
    llm_call_idx = 0
    for turn in result.get('conversation', []):
        if isinstance(turn, dict) and 'input' in turn and 'output' in turn:
            llm_call_idx += 1
            if llm_call_idx == wrong_real_idx:
                last_user = turn['input'][-1]
                obs = extract_observation(last_user.get('content', ''))
                m = re.search(r"pred=`(.+?)`\s*\|\s*target=`(.+?)`", wrong_line)
                if not m:
                    return None
                return obs, m.group(1), m.group(2)
    return None


def generate_pitfall(llm, task, website, observation, pred_act, target_act):
    prompt = AUTO_PITFALL_PROMPT.format(
        task=task, website=website,
        observation=observation[:2500],
        pred_act=pred_act, target_act=target_act,
    )
    resp = llm.chat([{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=400)
    raw = resp.strip()
    if "```" in raw:
        raw = re.sub(r"```[a-z]*\s*", "", raw, count=1)
        raw = raw.rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def pitfall_dict_to_memory(pitfall: dict) -> str:
    return (
        "# Task Skill Memory (decision pitfall)\n"
        "### [Base Skill] [✅ SUCCESS]\n"
        f"- **Trigger**: {pitfall.get('trigger', '')}\n"
        f"- **Pitfall**: {pitfall.get('pitfall', '')}\n"
    )


def make_combined(ep_idx: int, auto_pitfall: dict | None) -> str:
    """Combined recipe: prominent pitfall on top + 3 minimal steps."""
    recipes = ORACLE_RECIPES[ep_idx]
    # extract pitfall text from oracle pitfall memory
    oracle_pitfall_mem = recipes['pitfall']
    pitfall_text = ""
    trigger_text = ""
    for line in oracle_pitfall_mem.split('\n'):
        if '**Pitfall**' in line:
            pitfall_text = line.split('**Pitfall**:', 1)[-1].strip().lstrip(':').strip()
        elif '**Trigger**' in line:
            trigger_text = line.split('**Trigger**:', 1)[-1].strip().lstrip(':').strip()

    # extract first 3 Step N: ... lines from oracle step_template
    steps = []
    for line in recipes['step_template'].split('\n'):
        s = line.strip()
        if re.match(r'^Step \d+: ', s):
            steps.append(s)
            if len(steps) >= 3:
                break

    parts = [
        "# Task Skill Memory (combined: pitfall + minimal steps)",
        "### [Base Skill] [✅ SUCCESS]",
        f"⚠️ **CRITICAL DECISION HINT — READ FIRST**:",
        f"  {pitfall_text}",
        f"  (Applies when: {trigger_text})",
        "",
        "**Minimal procedure** (use only as fallback after applying the pitfall):",
        *[f"  {s}" for s in steps],
    ]
    return "\n".join(parts)


def run_one(agent, sample, mem):
    return agent.run_episode(
        sample, model_name="deepseek-v4-flash",
        no_memory=(mem is None),
        no_prtree_update=True,
        external_memory_str=mem,
        memory_type="prtree",
        episode_idx=-1,
    )


def metrics(r):
    elem = r['element_acc']
    f1 = r['action_f1']
    ss = r['step_success']
    return (
        sum(elem) / len(elem) if elem else 0.0,
        sum(f1) / len(f1) if f1 else 0.0,
        sum(ss) / len(ss) if ss else 0.0,
    )


def main():
    samples = []
    for fn in sorted(glob.glob(DATA_GLOB)):
        with open(fn) as f:
            samples.extend(json.load(f))
    samples = add_scores(samples, score_path=SCORE_PATH)

    llm = create_llm_client(model_name="deepseek-v4-flash")
    agent = DualTreeMind2WebAgent(agent_name="combined-exp", llm_client=llm)

    grid = {}
    auto_pitfalls = {}

    for ep_idx in EPISODES_TO_TEST:
        sample = samples[ep_idx]
        print(f"\n########## ep{ep_idx} ({sample['website']}) ##########")
        print(f"Task: {sample['confirmed_task'][:130]}")

        # 1. baseline (also provides wrong-step obs for auto pitfall)
        baseline_r = run_one(agent, sample, None)
        b_E, b_F, b_S = metrics(baseline_r)
        print(f"  01_baseline       E={b_E:.2f} F={b_F:.2f} S={b_S:.2f}")

        # 2. oracle pitfall
        oracle_mem = ORACLE_RECIPES[ep_idx]['pitfall']
        oracle_r = run_one(agent, sample, oracle_mem)
        o_E, o_F, o_S = metrics(oracle_r)
        print(f"  02_oracle_pitfall E={o_E:.2f} F={o_F:.2f} S={o_S:.2f}")

        # 3. auto pitfall — generate from baseline trajectory
        info = find_wrong_step_obs(baseline_r)
        if info:
            obs, pred_act, target_act = info
            pitfall_dict = generate_pitfall(
                llm, sample['confirmed_task'], sample['website'],
                obs, pred_act, target_act,
            )
        else:
            pitfall_dict = None
        if pitfall_dict:
            auto_pitfalls[ep_idx] = pitfall_dict
            print(f"  auto pitfall →   trigger={pitfall_dict.get('trigger','')[:90]}")
            print(f"                   pitfall={pitfall_dict.get('pitfall','')[:90]}")
            auto_mem = pitfall_dict_to_memory(pitfall_dict)
            auto_r = run_one(agent, sample, auto_mem)
            a_E, a_F, a_S = metrics(auto_r)
            print(f"  03_auto_pitfall   E={a_E:.2f} F={a_F:.2f} S={a_S:.2f}")
        else:
            a_E = a_F = a_S = None
            print(f"  03_auto_pitfall   ⚠️ generation failed, skipping")

        # 4. step_template
        st_mem = ORACLE_RECIPES[ep_idx]['step_template']
        st_r = run_one(agent, sample, st_mem)
        st_E, st_F, st_S = metrics(st_r)
        print(f"  04_step_template  E={st_E:.2f} F={st_F:.2f} S={st_S:.2f}")

        # 5. combined
        comb_mem = make_combined(ep_idx, pitfall_dict)
        comb_r = run_one(agent, sample, comb_mem)
        c_E, c_F, c_S = metrics(comb_r)
        print(f"  05_combined       E={c_E:.2f} F={c_F:.2f} S={c_S:.2f}")

        grid[ep_idx] = {
            "baseline":   (b_E, b_F, b_S),
            "oracle_pf":  (o_E, o_F, o_S),
            "auto_pf":    (a_E, a_F, a_S),
            "step_tpl":   (st_E, st_F, st_S),
            "combined":   (c_E, c_F, c_S),
        }

    # ---------- aggregate ----------
    print("\n\n================== AGGREGATE (E=element_acc  F=action_f1  S=step_success) ==================")
    cfgs = ["baseline", "oracle_pf", "auto_pf", "step_tpl", "combined"]
    header = ["ep   "] + [f"{c:>16}" for c in cfgs]
    print(" | ".join(header))
    print("-" * 110)
    for ep_idx, row in grid.items():
        cells = [f"{ep_idx:>5}"]
        for c in cfgs:
            v = row[c]
            if v[0] is None:
                cells.append("    --/--/--    ")
            else:
                cells.append(f"E{v[0]:.2f}/F{v[1]:.2f}/S{v[2]:.2f}")
        print(" | ".join(cells))

    # mean (skip None)
    print("-" * 110)
    sums = {c: [0, 0, 0, 0] for c in cfgs}  # E, F, S, count
    for row in grid.values():
        for c in cfgs:
            v = row[c]
            if v[0] is None:
                continue
            sums[c][0] += v[0]
            sums[c][1] += v[1]
            sums[c][2] += v[2]
            sums[c][3] += 1
    cells = ["mean "]
    for c in cfgs:
        n = sums[c][3] or 1
        e, f, s = sums[c][0] / n, sums[c][1] / n, sums[c][2] / n
        cells.append(f"E{e:.2f}/F{f:.2f}/S{s:.2f}")
    print(" | ".join(cells))

    print("\n--- Δ vs baseline ---")
    b_n = sums['baseline'][3]
    b_e = sums['baseline'][0] / b_n
    b_f = sums['baseline'][1] / b_n
    b_s = sums['baseline'][2] / b_n
    for c in cfgs[1:]:
        n = sums[c][3] or 1
        e, f, s = sums[c][0] / n, sums[c][1] / n, sums[c][2] / n
        print(f"  {c:14s}  ΔE={e-b_e:+.3f}  ΔF={f-b_f:+.3f}  ΔS={s-b_s:+.3f}  (n={n})")

    out = Path(__file__).parent / "memory_recipe_combined_results.json"
    with open(out, "w") as f:
        json.dump({"grid": grid, "auto_pitfalls": auto_pitfalls}, f, indent=2)
    print(f"\nDetailed → {out}")


if __name__ == "__main__":
    main()
