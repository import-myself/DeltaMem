"""
End-to-end verification of the new pitfall reflection.

For each failing episode:
  1. Run baseline (no memory) → capture conversation
  2. Use the NEW reflection prompts to extract a pitfall
  3. Re-run with that pitfall injected as memory
  4. Compare element_acc / action_f1 / step_success to baseline
"""

import json
import glob
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_mind2web_dual import DualTreeMind2WebAgent
from common.llm_client import create_llm_client
from mind2web_utils import add_scores

logging.getLogger().setLevel(logging.WARNING)
EPISODES = [132, 131, 137, 149, 88, 122]


def metrics(r):
    e = r['element_acc']; f = r['action_f1']; s = r['step_success']
    return (
        sum(e)/len(e) if e else 0.0,
        sum(f)/len(f) if f else 0.0,
        sum(s)/len(s) if s else 0.0,
    )


def pitfall_to_memory(refl):
    return (
        "# Task Pitfall Memory (decision hints)\n"
        "### [Base Skill] [✅ SUCCESS]\n"
        f"- **Trigger**: {refl.get('activation_condition','')}\n"
        f"- **Pitfall**: {refl.get('execution_procedure','')}\n"
    )


def main():
    samples = []
    for fn in sorted(glob.glob("/hdd/REDACTED_USER/DeltaMem/Mind2web/data/test_website/*.json")):
        with open(fn) as f:
            samples.extend(json.load(f))
    samples = add_scores(samples, score_path="/hdd/REDACTED_USER/DeltaMem/Mind2web/data/scores_all_data.pkl")

    llm = create_llm_client(model_name="deepseek-v4-flash")
    agent = DualTreeMind2WebAgent(agent_name="e2e-pitfall", llm_client=llm)

    rows = []
    for ep in EPISODES:
        sample = samples[ep]
        print(f"\n========== ep{ep} ({sample['website']}) ==========")
        print(f"Task: {sample['confirmed_task'][:130]}")

        # 1. baseline
        b = agent.run_episode(
            sample, model_name="deepseek-v4-flash",
            no_memory=True, no_prtree_update=True,
            external_memory_str=None, episode_idx=-1,
        )
        bE, bF, bS = metrics(b)
        print(f"  baseline       E={bE:.2f} F={bF:.2f} S={bS:.2f}")

        # 2. reflection with new prompt + observation-aware causal
        task_refl, _ = agent._generate_dual_reflection(
            task_goal=sample['confirmed_task'],
            env_description=f"{sample.get('domain','')}::{sample.get('subdomain','')}::{sample['website']}".strip(':'),
            success=bool(b['success']),
            steps=len(b['step_success']),
            trajectory=b['trajectory'],
            task_path=[], env_path=[],
            element_acc_list=b['element_acc'],
            action_f1_list=b['action_f1'],
            step_success_list=b['step_success'],
            conversation=b.get('conversation', []),
        )
        trig = task_refl.get('activation_condition','')
        pf = task_refl.get('execution_procedure','')
        print(f"  ↳ trigger: {trig[:140]}")
        print(f"  ↳ pitfall: {pf[:160]}")

        # 3. inject and rerun
        mem = pitfall_to_memory(task_refl)
        rr = agent.run_episode(
            sample, model_name="deepseek-v4-flash",
            no_memory=False, no_prtree_update=True,
            external_memory_str=mem, episode_idx=-1,
        )
        rE, rF, rS = metrics(rr)
        print(f"  with reflect   E={rE:.2f} F={rF:.2f} S={rS:.2f}   (ΔE={rE-bE:+.2f} ΔS={rS-bS:+.2f})")

        rows.append((ep, bE, bF, bS, rE, rF, rS))

    print("\n\n================ AGGREGATE ================")
    print(f"{'ep':>5} | {'baseline':>16} | {'+self_reflect':>18}")
    print("-" * 60)
    for ep, bE, bF, bS, rE, rF, rS in rows:
        print(f"{ep:>5} | E{bE:.2f}/F{bF:.2f}/S{bS:.2f} | E{rE:.2f}/F{rF:.2f}/S{rS:.2f}")
    n = len(rows)
    bE_m = sum(r[1] for r in rows)/n
    bF_m = sum(r[2] for r in rows)/n
    bS_m = sum(r[3] for r in rows)/n
    rE_m = sum(r[4] for r in rows)/n
    rF_m = sum(r[5] for r in rows)/n
    rS_m = sum(r[6] for r in rows)/n
    print("-" * 60)
    print(f" mean | E{bE_m:.2f}/F{bF_m:.2f}/S{bS_m:.2f} | E{rE_m:.2f}/F{rF_m:.2f}/S{rS_m:.2f}")
    print(f"  Δ vs baseline → ΔE={rE_m-bE_m:+.3f} ΔF={rF_m-bF_m:+.3f} ΔS={rS_m-bS_m:+.3f}")

    out = Path(__file__).parent / "verify_e2e_pitfall_results.json"
    with open(out, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
