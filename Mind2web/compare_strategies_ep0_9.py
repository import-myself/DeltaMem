"""
Full-pipeline comparison on ep 0-9 (each strategy starts from empty tree).

Strategies:
  baseline : no memory (run all 10 ep with no_memory=True)
  v0       : current v2 PRTree (default prompt, tb=0.75)
  +A       : abstract-trigger prompt (forces trigger to describe PATTERNS, not specific sites/values)
  +B       : self-check trigger (read-time guidance asking LLM to verify match)
  +C       : high threshold (tb=0.85)

Each strategy uses its own isolated DualTreeMemory; reflections write to its tree.
Episodes are processed in order 0..9 to simulate online learning.
"""

import json
import glob
import logging
import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_mind2web_dual import DualTreeMind2WebAgent
from common.llm_client import create_llm_client
from common.retriever import VectorRetriever
from mind2web_utils import add_scores
from memory.prtree.dual_tree_manager import DualTreeMemory
from memory.prtree.dual_memory_reader import DualMemoryReader
from memory.prtree.dual_memory_writer import DualMemoryWriter
import dual_prompt

logging.getLogger().setLevel(logging.WARNING)

EPISODES = list(range(0, 10))


# ---------------------------------------------------------------------------
# Abstract-trigger prompts for strategy A
# These replace TaskTree_Prompt_Map at runtime when running strategy A.
# Key change: trigger MUST be a generic PATTERN, no specific website / button text.
# ---------------------------------------------------------------------------

A_ROOT_FAILURE = """You are a Decision Pitfall Extractor (from a failed trajectory).

Capture the SINGLE most consequential wrong decision as a TRANSFERABLE pitfall.
The TRIGGER must describe a UI PATTERN that applies on MANY websites — NOT a
specific website name or specific element label.

⚠️ Trigger rules (strict):
  ❌ FORBIDDEN: "On recreation.gov...", "the 'Tickets & Tours' tab", "the 'Best Value' button",
                "a search input labeled 'Search Recreation.gov'"
  ✅ REQUIRED:  describe by CATEGORY only — "a category-tab navigation", "a sort-by control",
                "a search input alongside a category nav", "a submit button vs a filter toggle"

⚠️ Pitfall rules:
  - ONE sentence ≤ 60 words
  - Refer to elements by ROLE / GENERIC LABEL TYPE only, no specific text or site
  - No element IDs

Website: {env_description}
Task: {task_description}
Result: FAILURE (Steps: {steps})

{trajectory}

**Step-by-Step Causal Analysis:**
{step_causal_analysis}

Output ONLY valid JSON (no markdown fence):
{{
    "activation_condition": "Generic UI pattern: '<element-role-A> together with <element-role-B> on a <site-category> page'. NO website names, NO specific text values.",
    "execution_procedure": "ONE sentence: 'Click <element by ROLE/category>, not <decoy by ROLE/category>, because <reason at the category level>.' No specific text labels.",
    "termination_condition": ""
}}"""

A_ROOT_SUCCESS = """You are a Decision Pitfall Extractor (from a successful trajectory).

Capture the most TRANSFERABLE decision as a pitfall. The TRIGGER must describe
a UI PATTERN, not specific text or site name.

⚠️ Trigger rules (strict):
  ❌ FORBIDDEN: site names, specific button text, specific input placeholders
  ✅ REQUIRED:  UI pattern by role/category — "two visually similar nav links", "a primary submit button vs a toggle"

⚠️ Pitfall rules: ONE sentence ≤ 60 words, no specific text, no element IDs.

Website: {env_description}
Task: {task_description}
Result: SUCCESS (Steps: {steps})

{trajectory}

**Step-by-Step Causal Analysis:**
{step_causal_analysis}

Output ONLY valid JSON:
{{
    "activation_condition": "Generic UI pattern, no website names, no specific text.",
    "execution_procedure": "ONE sentence: 'Click <element by ROLE/category>, not <decoy by ROLE/category>, because <category-level reason>.'",
    "termination_condition": ""
}}"""

A_NODE_FAILURE = A_ROOT_FAILURE.replace(
    "{step_causal_analysis}\n\nOutput ONLY",
    "{step_causal_analysis}\n\n=== EXISTING PITFALLS ===\n{retrieved_task_memory}\n=== END ===\n\nIf an existing pitfall covers this same generic pattern, output {{\"reasoning\":\"...\",\"skip\":true}}.\nOtherwise output ONLY"
)
A_NODE_SUCCESS = A_ROOT_SUCCESS.replace(
    "{step_causal_analysis}\n\nOutput ONLY",
    "{step_causal_analysis}\n\n=== EXISTING PITFALLS ===\n{retrieved_task_memory}\n=== END ===\n\nIf an existing pitfall covers this same generic pattern, output {{\"reasoning\":\"...\",\"skip\":true}}.\nOtherwise output ONLY"
)


# Self-check memory header for B
B_HEADER_OVERRIDE = (
    "Below are decision-pitfall hints from past episodes.\n"
    "⚠️ DECISION CHECK (mandatory): Each pitfall has a 'Trigger' situation. "
    "BEFORE applying ANY pitfall, you MUST first verify in your thought whether "
    "the trigger's described UI elements actually appear in the CURRENT Observation. "
    "If NO — the pitfall is about a different task/site and WILL MISLEAD; ignore it. "
    "If YES — follow the pitfall's recommendation. "
    "Element IDs in memory are stale — match elements by aria-label / placeholder / text / role."
)


def metrics(r):
    e = r['element_acc']; f = r['action_f1']; s = r['step_success']
    return (
        sum(e)/len(e) if e else 0.0,
        sum(f)/len(f) if f else 0.0,
        sum(s)/len(s) if s else 0.0,
    )


def reset_agent_memory(agent, tb_task=0.75, tb_env=0.92):
    """Replace agent.dual_memory / reader / writer with a fresh isolated tree."""
    dm = DualTreeMemory(agent.retriever)
    dm.task_tree.base_threshold = tb_task
    dm.task_tree.depth_step = 0.03
    dm.env_tree.base_threshold = tb_env
    agent.dual_memory = dm
    agent.reader = DualMemoryReader(dm)
    agent.writer = DualMemoryWriter(dm)


def apply_prompts(strategy: str):
    """Patch dual_prompt module in-place for strategy A; restore for others."""
    if strategy == "A":
        dual_prompt.TaskTree_Prompt_Map['root_failure'] = A_ROOT_FAILURE
        dual_prompt.TaskTree_Prompt_Map['root_success'] = A_ROOT_SUCCESS
        dual_prompt.TaskTree_Prompt_Map['node_failure'] = A_NODE_FAILURE
        dual_prompt.TaskTree_Prompt_Map['node_success'] = A_NODE_SUCCESS
    else:
        # Restore originals if previously patched.
        import importlib
        importlib.reload(dual_prompt)


def apply_memory_header(agent, strategy: str):
    """B uses a different header; others use default."""
    if strategy == "B":
        dual_prompt.MEMORY_HEADERS["prtree"] = B_HEADER_OVERRIDE
    else:
        import importlib
        importlib.reload(dual_prompt)


def run_strategy(samples, strategy: str, agent: DualTreeMind2WebAgent):
    print(f"\n{'#'*60}")
    print(f"# Strategy: {strategy}")
    print(f"{'#'*60}")

    apply_prompts(strategy)
    apply_memory_header(agent, strategy)
    # for C, raise thresholds; for others, default
    tb_task = 0.85 if strategy == "C" else 0.75
    tb_env  = 0.94 if strategy == "C" else 0.92
    if strategy == "baseline":
        # we still reset memory but won't be used (no_memory=True)
        reset_agent_memory(agent, 0.75, 0.92)
    else:
        reset_agent_memory(agent, tb_task, tb_env)

    rows = []
    for ep_idx in EPISODES:
        sample = samples[ep_idx]
        r = agent.run_episode(
            sample, model_name="deepseek-v4-flash",
            no_memory=(strategy == "baseline"),
            no_prtree_update=(strategy == "baseline"),
            external_memory_str=None,
            memory_type="prtree",
            episode_idx=ep_idx,
        )
        E, F, S = metrics(r)
        thit = r.get('task_memory_used', False)
        ehit = r.get('env_memory_used', False)
        skip_t = (r.get('task_reflection', {}) or {}).get('skip', False)
        rows.append((ep_idx, E, F, S, thit, ehit, skip_t))
        print(f"  ep{ep_idx:>2}  E={E:.2f} F={F:.2f} S={S:.2f}  t_hit={thit} e_hit={ehit} skip_t={skip_t}")
    return rows


def main():
    samples = []
    for fn in sorted(glob.glob('/hdd/REDACTED_USER/DeltaMem/Mind2web/data/test_website/*.json')):
        with open(fn) as f: samples.extend(json.load(f))
    samples = add_scores(samples, score_path='/hdd/REDACTED_USER/DeltaMem/Mind2web/data/scores_all_data.pkl')

    llm = create_llm_client(model_name="deepseek-v4-flash")
    agent = DualTreeMind2WebAgent(agent_name="strategy-cmp", llm_client=llm)

    all_results = {}
    for strategy in ["baseline", "v0", "A", "B", "C"]:
        rows = run_strategy(samples, strategy, agent)
        all_results[strategy] = rows

    # Aggregate
    print("\n\n================== AGGREGATE (mean over 10 ep) ==================")
    print(f"{'strategy':<10}  {'E':>6}  {'F':>6}  {'S':>6}  {'t_hit':>6}  {'e_hit':>6}  {'skip':>5}")
    base_E = base_F = base_S = None
    for strat, rows in all_results.items():
        n = len(rows)
        E = sum(r[1] for r in rows)/n
        F = sum(r[2] for r in rows)/n
        S = sum(r[3] for r in rows)/n
        th = sum(1 for r in rows if r[4])
        eh = sum(1 for r in rows if r[5])
        sk = sum(1 for r in rows if r[6])
        print(f"{strat:<10}  E{E:.3f}  F{F:.3f}  S{S:.3f}  {th:>6}  {eh:>6}  {sk:>5}")
        if strat == "baseline":
            base_E, base_F, base_S = E, F, S
    print(f"\nΔ vs baseline:")
    for strat in ["v0", "A", "B", "C"]:
        rows = all_results[strat]
        n = len(rows)
        E = sum(r[1] for r in rows)/n
        F = sum(r[2] for r in rows)/n
        S = sum(r[3] for r in rows)/n
        print(f"  {strat:<3}  ΔE={E-base_E:+.3f}  ΔF={F-base_F:+.3f}  ΔS={S-base_S:+.3f}")

    out = Path(__file__).parent / "compare_strategies_ep0_9_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
