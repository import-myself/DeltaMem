"""
Auto-pitfall generation experiment.

For each failing episode:
  1. Run baseline (no memory), capture the wrong-step's observation and pred/target.
  2. Ask the LLM to generate a single decision-pitfall sentence given:
       task + observation + pred_act + target_act
     This simulates what reflection could learn automatically from a failed trajectory.
  3. Inject that auto-pitfall as memory and re-run the same episode.
  4. Compare element_acc against (a) baseline (b) oracle hand-written pitfall.

The point: validate that the reflection prompt can produce pitfalls that recover
most of the gain we saw with hand-written ones.
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

# import hand-written pitfalls so we can compare auto vs oracle directly
from memory_recipe_multi import EPISODES as ORACLE_RECIPES

logging.getLogger().setLevel(logging.WARNING)

DATA_GLOB = "/hdd/REDACTED_USER/DeltaMem/Mind2web/data/test_website/*.json"
SCORE_PATH = "/hdd/REDACTED_USER/DeltaMem/Mind2web/data/scores_all_data.pkl"

EPISODES_TO_TEST = [132, 131, 137, 149, 88, 122]


# ---------------------------------------------------------------------------
# Auto-pitfall generation prompt
# ---------------------------------------------------------------------------

AUTO_PITFALL_PROMPT = """You are a Decision Pitfall Extractor for a web navigation agent.

The agent failed at one step of this task. Your job is to write ONE short pitfall
sentence that future agents could use to avoid making the SAME wrong choice in
similar situations.

Task: {task}
Website: {website}

Observation when the wrong action was chosen:
```
{observation}
```

WRONG action the agent took:        {pred_act}
RIGHT action it should have taken:  {target_act}

Write your output as a JSON object with this exact shape:

{{
  "trigger": "Describe the SITUATION that causes this ambiguity — what UI elements
              are present together (use aria-label/text/role only, NO element IDs).",
  "pitfall": "ONE sentence ≤ 60 words. State the DEFAULT correct choice (by semantic
              attribute) and the wrong choice to avoid. NO element IDs."
}}

Rules:
- Refer to elements ONLY by visible text, aria-label, placeholder, or role.
- NEVER write any numeric element ID (e.g., [3128]) — these are different every page.
- Generalize: this pitfall must apply to similar tasks on this website type,
  not only this exact task.
- Output ONLY the JSON, no markdown fence, no preamble.
"""


def extract_observation(user_content: str) -> str:
    m = re.search(r"Observation: `(.+?)`", user_content, re.DOTALL)
    return m.group(1) if m else user_content


def find_wrong_step_user_msg(result_messages, traj):
    """Return (observation_text, pred_act, target_act) for the first wrong step."""
    # walk traj to find wrong step's real-step index (1-based among Step ... lines)
    real_idx = 0
    wrong_real_idx = None
    wrong_line = None
    for line in traj:
        if line.startswith('Step '):
            real_idx += 1
            if 'elem_acc=0' in line:
                wrong_real_idx = real_idx
                wrong_line = line
                break
    if wrong_real_idx is None:
        return None
    # the wrong real-step corresponds to the wrong_real_idx-th user/assistant pair
    # messages layout: system, user1, ass1, user2, ass2, ...
    user_msg_idx = 1 + 2 * (wrong_real_idx - 1)
    if user_msg_idx >= len(result_messages):
        return None
    obs = extract_observation(result_messages[user_msg_idx]['content'])
    # parse pred/target from wrong_line
    m = re.search(r"pred=`(.+?)`\s*\|\s*target=`(.+?)`", wrong_line)
    if not m:
        return None
    return obs, m.group(1), m.group(2)


def generate_pitfall(llm, task: str, website: str, observation: str,
                     pred_act: str, target_act: str) -> dict | None:
    prompt = AUTO_PITFALL_PROMPT.format(
        task=task, website=website,
        observation=observation[:2500],
        pred_act=pred_act, target_act=target_act,
    )
    resp = llm.chat([{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=400)
    raw = resp.strip()
    if "```" in raw:
        # strip code fence
        raw = re.sub(r"```[a-z]*\s*", "", raw, count=1)
        raw = raw.rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except Exception:
        # fallback: try to extract braces
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def pitfall_to_memory(pitfall: dict) -> str:
    return (
        "# Task Skill Memory (decision pitfall)\n"
        "### [Base Skill] [✅ SUCCESS]\n"
        f"- **Trigger**: {pitfall.get('trigger', '')}\n"
        f"- **Pitfall**: {pitfall.get('pitfall', '')}\n"
    )


# ---------------------------------------------------------------------------
# Episode runner — also returns the message list so we can recover obs
# ---------------------------------------------------------------------------

def run_one(agent, sample, mem):
    return agent.run_episode(
        sample, model_name="deepseek-v4-flash",
        no_memory=(mem is None),
        no_prtree_update=True,
        external_memory_str=mem,
        memory_type="prtree",
        episode_idx=-1,
    )


def collect_messages_from_result(result):
    """Reconstruct the message list from result['conversation']."""
    # conversation entries can be {"input":..., "output":...} or {"pred_act":..., "target_act":...}
    # we need the first {"input"} that carries the FAILING step's user message.
    # For simplicity, scan all input blocks and merge messages.
    msgs = []
    for turn in result.get('conversation', []):
        if isinstance(turn, dict) and 'input' in turn:
            # last user message of this input block is the latest obs
            msgs.append(turn['input'])
    return msgs


def find_wrong_step_observation_v2(result):
    """Walk through conversation in order, find the wrong-step's last user obs."""
    real_idx = 0
    wrong_real_idx = None
    for line in result.get('trajectory', []):
        if line.startswith('Step '):
            real_idx += 1
            if 'elem_acc=0' in line:
                wrong_real_idx = real_idx
                wrong_line = line
                break
    if wrong_real_idx is None:
        return None
    # conversation has input/output pairs per LLM call. Each LLM call = one real step.
    llm_call_idx = 0
    for turn in result.get('conversation', []):
        if isinstance(turn, dict) and 'input' in turn and 'output' in turn:
            llm_call_idx += 1
            if llm_call_idx == wrong_real_idx:
                last_user = turn['input'][-1]
                obs = extract_observation(last_user.get('content', ''))
                # parse pred/target
                m = re.search(r"pred=`(.+?)`\s*\|\s*target=`(.+?)`", wrong_line)
                if not m:
                    return None
                return obs, m.group(1), m.group(2)
    return None


def main():
    samples = []
    for fn in sorted(glob.glob(DATA_GLOB)):
        with open(fn) as f:
            samples.extend(json.load(f))
    samples = add_scores(samples, score_path=SCORE_PATH)

    llm = create_llm_client(model_name="deepseek-v4-flash")
    agent = DualTreeMind2WebAgent(agent_name="auto-pitfall-exp", llm_client=llm)

    rows = []
    auto_pitfalls = {}

    for ep_idx in EPISODES_TO_TEST:
        sample = samples[ep_idx]
        print(f"\n############### ep{ep_idx} ({sample['website']}) ###############")
        print(f"Task: {sample['confirmed_task'][:130]}")

        # 1. baseline
        baseline_r = run_one(agent, sample, None)
        b_elem = sum(baseline_r['element_acc']) / len(baseline_r['element_acc']) if baseline_r['element_acc'] else 0
        b_ss = sum(baseline_r['step_success']) / len(baseline_r['step_success']) if baseline_r['step_success'] else 0
        print(f"  baseline       elem={b_elem:.2f} ss={b_ss:.2f}")

        # 2. extract wrong-step info
        info = find_wrong_step_observation_v2(baseline_r)
        if not info:
            print(f"  ⚠️  could not locate wrong step — skip auto pitfall")
            rows.append({"ep": ep_idx, "baseline": b_elem, "auto": None, "oracle": None})
            continue
        obs, pred_act, target_act = info
        print(f"  wrong obs ({len(obs)} chars), pred={pred_act} target={target_act}")

        # 3. generate auto pitfall
        pitfall = generate_pitfall(llm, sample['confirmed_task'], sample['website'],
                                    obs, pred_act, target_act)
        if pitfall is None:
            print(f"  ⚠️  pitfall JSON parse failed")
            rows.append({"ep": ep_idx, "baseline": b_elem, "auto": None, "oracle": None})
            continue
        auto_pitfalls[ep_idx] = pitfall
        print(f"  auto pitfall:")
        print(f"    trigger: {pitfall.get('trigger', '')[:160]}")
        print(f"    pitfall: {pitfall.get('pitfall', '')[:160]}")

        # 4. inject auto pitfall, re-run
        auto_mem = pitfall_to_memory(pitfall)
        auto_r = run_one(agent, sample, auto_mem)
        a_elem = sum(auto_r['element_acc']) / len(auto_r['element_acc']) if auto_r['element_acc'] else 0
        a_ss = sum(auto_r['step_success']) / len(auto_r['step_success']) if auto_r['step_success'] else 0
        print(f"  with auto pf   elem={a_elem:.2f} ss={a_ss:.2f}")

        # 5. run oracle pitfall for direct comparison
        oracle_mem = ORACLE_RECIPES.get(ep_idx, {}).get('pitfall')
        if oracle_mem:
            oracle_r = run_one(agent, sample, oracle_mem)
            o_elem = sum(oracle_r['element_acc']) / len(oracle_r['element_acc']) if oracle_r['element_acc'] else 0
            o_ss = sum(oracle_r['step_success']) / len(oracle_r['step_success']) if oracle_r['step_success'] else 0
            print(f"  with oracle pf elem={o_elem:.2f} ss={o_ss:.2f}")
        else:
            o_elem = o_ss = None

        rows.append({
            "ep": ep_idx,
            "baseline_elem": b_elem, "baseline_ss": b_ss,
            "auto_elem": a_elem, "auto_ss": a_ss,
            "oracle_elem": o_elem, "oracle_ss": o_ss,
        })

    print("\n\n================ AGGREGATE ================")
    print(f"{'ep':>5} | {'baseline':>9} | {'auto_pf':>9} | {'oracle_pf':>9}  (element_acc / step_success)")
    print("-" * 70)
    for r in rows:
        if r.get('auto_elem') is None:
            print(f"{r['ep']:>5} | (auto generation failed)")
            continue
        line = f"{r['ep']:>5} | {r['baseline_elem']:.2f}/{r['baseline_ss']:.2f}  | "
        line += f"{r['auto_elem']:.2f}/{r['auto_ss']:.2f}  | "
        if r['oracle_elem'] is not None:
            line += f"{r['oracle_elem']:.2f}/{r['oracle_ss']:.2f}"
        print(line)

    valid = [r for r in rows if r.get('auto_elem') is not None]
    if valid:
        n = len(valid)
        b_elem = sum(r['baseline_elem'] for r in valid) / n
        a_elem = sum(r['auto_elem'] for r in valid) / n
        o_elem = sum(r['oracle_elem'] for r in valid if r['oracle_elem'] is not None) / n
        b_ss = sum(r['baseline_ss'] for r in valid) / n
        a_ss = sum(r['auto_ss'] for r in valid) / n
        o_ss = sum(r['oracle_ss'] for r in valid if r['oracle_ss'] is not None) / n
        print("-" * 70)
        print(f"  mean | {b_elem:.2f}/{b_ss:.2f}  | {a_elem:.2f}/{a_ss:.2f}  | {o_elem:.2f}/{o_ss:.2f}")

    out = Path(__file__).parent / "memory_recipe_auto_pitfall_results.json"
    with open(out, "w") as f:
        json.dump({"rows": rows, "auto_pitfalls": auto_pitfalls}, f, indent=2)
    print(f"\nDetailed results → {out}")


if __name__ == "__main__":
    main()
