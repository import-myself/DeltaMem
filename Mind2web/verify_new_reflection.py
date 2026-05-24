"""
Verify the new pitfall-style reflection prompt by running it on saved
no-memory trajectories (oracle-free — only what reflection naturally sees).
"""

import json
import logging
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from dual_prompt import TaskTree_Prompt_Map
from common.llm_client import create_llm_client

logging.getLogger().setLevel(logging.WARNING)

EPISODES = [132, 131, 137, 149, 88, 122]
TRAJ_DIR = Path("/hdd/REDACTED_USER/DeltaMem/Mind2web/trajectories/online-test_website-deepseek-v4-flash-no-memory")


def build_step_causal_analysis(trajectory):
    """Mirror agent_mind2web_dual._build_step_causal_analysis with stored trajectory only."""
    real_steps = [t for t in trajectory if t.startswith("Step ")]
    lines = []
    for line in real_steps:
        m = re.search(r"elem_acc=(\d+)\s+action_f1=([\d.]+)\s+step_success=(\d+)", line)
        if not m:
            lines.append(line)
            continue
        elem = int(m.group(1))
        f1 = float(m.group(2))
        e_tag = "✓" if elem == 1 else "✗"
        a_tag = "✓" if f1 == 1.0 else "✗"
        if elem == 1 and f1 == 1.0:
            hint = "Both correct — what signal identified this element and action type?"
        elif elem == 1 and f1 < 1.0:
            hint = "Element correct but action wrong — what indicates the right action type?"
        elif elem == 0 and f1 > 0:
            hint = "Element wrong, action partly matched — what attribute distinguishes the correct element?"
        else:
            hint = "Both wrong — what led to the wrong element, what alternative signal should have been used?"
        lines.append(f"[{e_tag}elem {a_tag}act] {line}\n  → {hint}")
    return "\n".join(lines)


def main():
    llm = create_llm_client(model_name="deepseek-v4-flash")
    samples_path = "/hdd/REDACTED_USER/DeltaMem/Mind2web/data/test_website"
    import glob
    all_samples = []
    for fn in sorted(glob.glob(f"{samples_path}/*.json")):
        with open(fn) as f:
            all_samples.extend(json.load(f))

    for ep in EPISODES:
        with open(TRAJ_DIR / f"{ep}.json") as f:
            traj_data = json.load(f)
        r = traj_data[-1]
        trajectory = r["trajectory"]
        sample = all_samples[ep]
        task = sample["confirmed_task"]
        website = sample["website"]
        domain = sample.get("domain", "")
        subdomain = sample.get("subdomain", "")
        env_key = "::".join([domain, subdomain, website]) if domain else website
        success = r.get("success", False)
        prompt_key = "root_success" if success else "root_failure"
        traj_str = "\n".join(trajectory)
        causal = build_step_causal_analysis(trajectory)
        prompt = TaskTree_Prompt_Map[prompt_key].format(
            env_description=env_key,
            task_description=task,
            steps=len([t for t in trajectory if t.startswith("Step ")]),
            trajectory=traj_str,
            step_causal_analysis=causal,
        )
        print(f"\n========== ep{ep} ({website}) | success={success} ==========")
        print(f"Task: {task[:140]}")
        resp = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=800)
        print(f"LLM raw response:\n{resp}\n")
        # try to parse
        raw = resp.strip()
        if "```" in raw:
            raw = re.sub(r"```[a-z]*\s*", "", raw, count=1).rsplit("```", 1)[0]
        try:
            data = json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(m.group(0)) if m else None
        if data:
            ac = data.get("activation_condition", "")
            ep_text = data.get("execution_procedure", "")
            n_words = len(ep_text.split())
            has_id = bool(re.search(r"\[\d+\]", ac + " " + ep_text))
            mentions_step = "Step " in ep_text or "step 1" in ep_text.lower() or "step 2" in ep_text.lower()
            print(f"  parsed: trigger={ac[:120]}")
            print(f"          pitfall={ep_text[:160]}")
            print(f"  checks: words={n_words}  has_element_id={has_id}  reads_like_steps={mentions_step}")
        else:
            print("  ⚠️ JSON parse failed")


if __name__ == "__main__":
    main()
