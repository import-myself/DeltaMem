"""
End-to-end reflection verification with the new pitfall prompt
+ observation-aware causal analysis.

For each failing episode:
  1. Run baseline (no memory) — captures conversation
  2. Manually invoke the agent's reflection helper on that conversation
  3. Print the reflected pitfall and check whether it identifies the actual wrong-step ambiguity
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

logging.getLogger().setLevel(logging.WARNING)

EPISODES = [132, 131, 137, 149, 88, 122]


def main():
    samples = []
    for fn in sorted(glob.glob("/hdd/REDACTED_USER/DeltaMem/Mind2web/data/test_website/*.json")):
        with open(fn) as f:
            samples.extend(json.load(f))
    samples = add_scores(samples, score_path="/hdd/REDACTED_USER/DeltaMem/Mind2web/data/scores_all_data.pkl")

    llm = create_llm_client(model_name="deepseek-v4-flash")
    agent = DualTreeMind2WebAgent(agent_name="verify-v2", llm_client=llm)

    for ep in EPISODES:
        sample = samples[ep]
        print(f"\n========== ep{ep} ({sample['website']}) ==========")
        print(f"Task: {sample['confirmed_task'][:140]}")

        # Run baseline to populate conversation
        result = agent.run_episode(
            sample, model_name="deepseek-v4-flash",
            no_memory=True, no_prtree_update=True,
            external_memory_str=None,
            episode_idx=-1,
        )
        elem = result['element_acc']
        f1 = result['action_f1']
        ss = result['step_success']
        print(f"  baseline elem={sum(elem)/len(elem):.2f} ss={sum(ss)/len(ss):.2f}")
        # find wrong step line for sanity
        wrong = [t for t in result['trajectory'] if t.startswith('Step ') and 'elem_acc=0' in t]
        if wrong:
            print(f"  wrong step (truth): {wrong[0]}")

        # Manually trigger reflection (since no_memory was True it was skipped in the run)
        task_refl, env_refl = agent._generate_dual_reflection(
            task_goal=sample['confirmed_task'],
            env_description=f"{sample.get('domain','')}::{sample.get('subdomain','')}::{sample['website']}".strip(':'),
            success=False,
            steps=len(ss),
            trajectory=result['trajectory'],
            task_path=[],
            env_path=[],
            element_acc_list=elem,
            action_f1_list=f1,
            step_success_list=ss,
            conversation=result.get('conversation', []),
        )
        print(f"  → reflected trigger: {task_refl.get('activation_condition','')[:200]}")
        print(f"  → reflected pitfall: {task_refl.get('execution_procedure','')[:240]}")


if __name__ == "__main__":
    main()
