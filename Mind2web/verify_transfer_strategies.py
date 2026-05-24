"""
Verify which transfer-improvement strategy actually helps.

10 episodes (ep 80-176, NOT seen during v2 memory building) × 4 conditions:
  baseline : no memory
  v0       : current v2 memory + default prompt + default threshold (0.75)
  +B       : v2 memory + self-check trigger guidance
  +C       : v2 memory + tighter retrieval threshold (0.85)

We REUSE the frozen v2 memory bank (80 ep already built).
For B/C we inject memory via external_memory_str so no writes happen.
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
from common.retriever import VectorRetriever
from mind2web_utils import add_scores
from memory.prtree.dual_tree_manager import DualTreeMemory
from memory.prtree.dual_memory_reader import DualMemoryReader

logging.getLogger().setLevel(logging.WARNING)

EPISODES = [83, 112, 119, 131, 137, 156, 157, 158, 168, 174]
V2_MEM_BASE = "storage/online-test_website-deepseek-v4-flash-prtree-v2"


def metrics(r):
    e = r['element_acc']; f = r['action_f1']; s = r['step_success']
    return (
        sum(e)/len(e) if e else 0.0,
        sum(f)/len(f) if f else 0.0,
        sum(s)/len(s) if s else 0.0,
    )


def env_key(sample):
    d = sample.get('domain',''); sd = sample.get('subdomain',''); w = sample['website']
    return "::".join(p for p in [d, sd, w] if p)


def build_memory_context(reader, task_path, env_path, prefix_self_check: bool):
    """Render task+env memory from given paths, optionally with self-check guidance."""
    task_text = reader.render_task_memory(task_path)
    env_text  = reader.render_env_memory(env_path)
    if not task_text and not env_text:
        return None
    sections = []
    TASK_INTRO = (
        "# Task Pitfall Memory (decision hints)\n"
        "Each entry below is a SHORT decision-pitfall extracted from past episodes. "
        "Element IDs are stale — match elements by aria-label / placeholder / text / role.\n\n"
    )
    if prefix_self_check:
        TASK_INTRO = (
            "# Task Pitfall Memory (decision hints)\n"
            "⚠️ DECISION CHECK (READ BEFORE USING): Each pitfall below has a 'Trigger' situation. "
            "BEFORE following any pitfall, FIRST verify that the trigger's described UI elements "
            "actually appear in your current Observation. If the trigger does NOT match the current "
            "page, IGNORE that pitfall — it was learned on a DIFFERENT task/website and will mislead "
            "you. Only follow a pitfall when its trigger truly fits the situation you face right now.\n\n"
        )
    ENV_INTRO = (
        "# Website Knowledge Memory (declarative UI facts)\n"
        "Background knowledge about this website's UI behaviour — not a procedure. "
        "Object IDs are from past episodes — do NOT reuse them.\n\n"
    )
    if task_text:
        sections.append(TASK_INTRO + task_text)
    if env_text:
        sections.append(ENV_INTRO + env_text)
    return "\n---\n\n".join(sections) if sections else None


def run_condition(agent, sample, memory_str):
    return agent.run_episode(
        sample, model_name="deepseek-v4-flash",
        no_memory=(memory_str is None),
        no_prtree_update=True,
        external_memory_str=memory_str,
        memory_type="prtree",
        episode_idx=-1,
    )


def main():
    samples = []
    for fn in sorted(glob.glob('/hdd/REDACTED_USER/DeltaMem/Mind2web/data/test_website/*.json')):
        with open(fn) as f: samples.extend(json.load(f))
    samples = add_scores(samples, score_path='/hdd/REDACTED_USER/DeltaMem/Mind2web/data/scores_all_data.pkl')

    llm = create_llm_client(model_name="deepseek-v4-flash")
    agent = DualTreeMind2WebAgent(agent_name="transfer-strategies", llm_client=llm)

    # 加载 v2 memory（同一份给所有条件用，frozen）
    retriever = VectorRetriever()
    dm = DualTreeMemory(retriever)
    dm.load_trees(f"{V2_MEM_BASE}_task.json", f"{V2_MEM_BASE}_env.json")
    reader = DualMemoryReader(dm)

    # 为 condition C 准备一个高阈值版本
    dm_high = DualTreeMemory(retriever)
    dm_high.load_trees(f"{V2_MEM_BASE}_task.json", f"{V2_MEM_BASE}_env.json")
    dm_high.task_tree.base_threshold = 0.85
    dm_high.task_tree.depth_step = 0.02
    dm_high.env_tree.base_threshold = 0.94
    reader_high = DualMemoryReader(dm_high)

    rows = []
    for ep in EPISODES:
        sample = samples[ep]
        task = sample['confirmed_task']
        ek = env_key(sample)
        print(f"\n========== ep{ep} ({sample['website']}) ==========")
        print(f"Task: {task[:120]}")

        # default retrieval paths
        task_path = dm.retrieve_task_path(task)
        env_path  = dm.retrieve_env_path(ek)
        task_hit = not (len(task_path)==1 and 'GLOBAL_ROOT' in task_path[0].payload.get('scenario_description',''))
        env_hit = not (len(env_path)==1 and 'GLOBAL_ROOT' in env_path[0].payload.get('scenario_description',''))
        print(f"  default retrieval: task_hit={task_hit} (depth={len(task_path)-1})  env_hit={env_hit} (depth={len(env_path)-1})")

        # high threshold paths
        task_path_h = dm_high.retrieve_task_path(task)
        env_path_h  = dm_high.retrieve_env_path(ek)
        task_hit_h = not (len(task_path_h)==1 and 'GLOBAL_ROOT' in task_path_h[0].payload.get('scenario_description',''))
        env_hit_h = not (len(env_path_h)==1 and 'GLOBAL_ROOT' in env_path_h[0].payload.get('scenario_description',''))
        print(f"  high-thresh retr  : task_hit={task_hit_h} (depth={len(task_path_h)-1})  env_hit={env_hit_h} (depth={len(env_path_h)-1})")

        # build memory strings
        mem_v0 = build_memory_context(reader, task_path, env_path, prefix_self_check=False)
        mem_B  = build_memory_context(reader, task_path, env_path, prefix_self_check=True)
        mem_C  = build_memory_context(reader_high, task_path_h, env_path_h, prefix_self_check=False)

        # run 4 conditions
        baseline = run_condition(agent, sample, None)
        v0 = run_condition(agent, sample, mem_v0)
        vB = run_condition(agent, sample, mem_B)
        vC = run_condition(agent, sample, mem_C)

        bE,bF,bS = metrics(baseline); vE,vF,vS = metrics(v0); BE,BF,BS = metrics(vB); CE,CF,CS = metrics(vC)
        print(f"  baseline E={bE:.2f} F={bF:.2f} S={bS:.2f}")
        print(f"  v0       E={vE:.2f} F={vF:.2f} S={vS:.2f}  (ΔS={vS-bS:+.2f})")
        print(f"  +B       E={BE:.2f} F={BF:.2f} S={BS:.2f}  (ΔS={BS-bS:+.2f})")
        print(f"  +C       E={CE:.2f} F={CF:.2f} S={CS:.2f}  (ΔS={CS-bS:+.2f})")
        rows.append((ep, bE,bF,bS, vE,vF,vS, BE,BF,BS, CE,CF,CS, task_hit, task_hit_h))

    print("\n\n================== AGGREGATE ==================")
    print(f"{'ep':>5} | {'baseline':>16} | {'v0':>16} | {'+B':>16} | {'+C':>16} | task_hit (default/high)")
    print("-" * 130)
    for r in rows:
        ep, bE,bF,bS, vE,vF,vS, BE,BF,BS, CE,CF,CS, th, thh = r
        print(f"{ep:>5} | E{bE:.2f}/F{bF:.2f}/S{bS:.2f} | E{vE:.2f}/F{vF:.2f}/S{vS:.2f} | "
              f"E{BE:.2f}/F{BF:.2f}/S{BS:.2f} | E{CE:.2f}/F{CF:.2f}/S{CS:.2f} | {th}/{thh}")
    n = len(rows)
    cols = list(zip(*[r[1:13] for r in rows]))
    means = [sum(c)/n for c in cols]
    bE,bF,bS, vE,vF,vS, BE,BF,BS, CE,CF,CS = means
    print("-" * 130)
    print(f" mean | E{bE:.2f}/F{bF:.2f}/S{bS:.2f} | E{vE:.2f}/F{vF:.2f}/S{vS:.2f} | "
          f"E{BE:.2f}/F{BF:.2f}/S{BS:.2f} | E{CE:.2f}/F{CF:.2f}/S{CS:.2f}")
    print(f"\nΔ vs baseline:")
    print(f"  v0  ΔE={vE-bE:+.3f}  ΔF={vF-bF:+.3f}  ΔS={vS-bS:+.3f}")
    print(f"  +B  ΔE={BE-bE:+.3f}  ΔF={BF-bF:+.3f}  ΔS={BS-bS:+.3f}")
    print(f"  +C  ΔE={CE-bE:+.3f}  ΔF={CF-bF:+.3f}  ΔS={CS-bS:+.3f}")

    out = Path(__file__).parent / "verify_transfer_strategies_results.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
