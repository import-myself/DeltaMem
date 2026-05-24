"""
Memory Recipe Experiment for ep132 (Mind2Web test_website)

Task: "Find cheapest flight from Toronto, Canada to New York City, USA for 1 adult."
Website: tripadvisor

Failure point identified from previous runs:
  Step 8 — Observation shows: One-way span [33630] + Find-flights button [32796]
  No-memory & PRTree both predict: CLICK [33630]  (wrong: changed trip type)
  Target:                          CLICK [32796]  (right: trigger search)

Hypothesis: different memory shapes will or will not redirect the LLM to the
correct element. We iterate over recipes and report element_acc at each step.

NOTE: This script does NOT mutate any tree on disk. We use external_memory_str
to inject the recipe and `no_prtree_update=True` to skip reflection/writing.
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

logging.getLogger().setLevel(logging.WARNING)  # quiet down noisy retrievers

EPISODE_IDX = 132
DATA_GLOB = "/hdd/REDACTED_USER/DeltaMem/Mind2web/data/test_website/*.json"


def load_sample(idx: int):
    samples = []
    for fn in sorted(glob.glob(DATA_GLOB)):
        with open(fn) as f:
            samples.extend(json.load(f))
    score_path = "/hdd/REDACTED_USER/DeltaMem/Mind2web/data/scores_all_data.pkl"
    samples = add_scores(samples, score_path=score_path)
    return samples[idx]


# ---------------------------------------------------------------------------
# Recipes — each returns the memory_context string injected into the system msg
# ---------------------------------------------------------------------------

R_BASELINE = ""  # equivalent to no-memory

R_ABSTRACT = """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Flight booking tasks on travel websites
- **Procedure**:
  In flight booking workflows, locate the origin and destination fields by their
  visible labels (e.g., "From"/"To" or "City or Airport"). After entering both
  cities and selecting from the autocomplete suggestions, trigger the search.
  The search results page typically lets users sort by price; choose the option
  whose label conveys cheapest/best-value when the task asks for the cheapest flight.
- **Done when**: A list of flights is shown and the cheapest is selectable.
"""

R_STEP_DESCRIPTOR = """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Flight booking on travel-meta sites (tripadvisor/kayak/etc.)
- **Procedure** (step-by-step template — match each step against the current Observation by aria-label / placeholder / visible text):
  Step 1: CLICK <menuitem/span text='Flights'> — enter the flights section
  Step 2: CLICK <input text='City or Airport'/placeholder containing 'From'> — focus origin field
  Step 3: TYPE the origin city name into the focused origin field
  Step 4: CLICK <span/option containing the origin city or its airport code> — confirm origin from autocomplete
  Step 5: CLICK <input placeholder containing 'To'/'Destination'> — focus destination field
  Step 6: TYPE the destination city name
  Step 7: CLICK <span/option containing destination city or 'All Airports (XXX)'> — confirm destination
  Step 8: CLICK <button text='Find flights'/'Search'/'Go'> — trigger results
  Step 9: CLICK <generic/div text='Best Value'/'Cheapest'/'Price (low to high)'> — sort results for cheapest-flight tasks
  Step 10: CLICK <button text='View Deal'/'Select'/'Book'> — proceed with chosen flight
- **Done when**: The chosen flight detail page is opened.
"""

R_STEP_DESCRIPTOR_PLUS_PITFALL = """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Flight booking on travel-meta sites (tripadvisor/kayak/etc.)
- **Procedure** (step-by-step template — match each step against the current Observation by aria-label / placeholder / visible text):
  Step 1: CLICK <menuitem/span text='Flights'> — enter the flights section
  Step 2-7: fill origin & destination fields, confirm from autocomplete
  Step 8: CLICK <button text='Find flights'/'Search'/'Go'> — trigger results
  Step 9: CLICK <generic/div text='Best Value'/'Cheapest'/'Price (low to high)'> — sort for cheapest
- **Pitfalls (must follow)**:
  • Trip-type toggles ('Round Trip' / 'One-way' / 'Multi-city') are ALREADY at the
    default 'Round Trip'. NEVER change the trip type unless the task explicitly says
    "one-way" or "multi-city". If the task does not mention it, skip those toggles
    and CLICK 'Find flights' directly.
  • The 'Find flights' button is the only action that actually launches the search;
    do not mistake it for any sibling option.
- **Done when**: The chosen flight detail page is opened.
"""

R_JUST_PITFALL = """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Flight booking on travel-meta sites where the user did not specify a trip type.
- **Procedure**:
  After filling origin & destination, IGNORE any 'Round Trip / One-way / Multi-city'
  toggles — they default to 'Round Trip' which is what the task wants. Immediately
  CLICK the 'Find flights' / 'Search' button to launch results. Selecting a non-default
  trip-type when the task did not request it is a known error.
- **Done when**: Results page is shown.
"""

R_NEGATIVE_DECOY = """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Flight booking on travel sites.
- **Procedure**:
  After entering origin and destination, change the trip type to 'One-way' before
  clicking Find flights. This often gives cheaper fares.
- **Done when**: Results page is shown.
"""
# ^ deliberately misleading — used to confirm memory actually changes behaviour.


RECIPES = {
    "01_baseline_no_memory": None,             # no external memory injected
    "02_abstract_prose":     R_ABSTRACT,
    "03_step_descriptor":    R_STEP_DESCRIPTOR,
    "04_step_desc_pitfall":  R_STEP_DESCRIPTOR_PLUS_PITFALL,
    "05_just_pitfall":       R_JUST_PITFALL,
    "06_negative_decoy":     R_NEGATIVE_DECOY,
}


def run_recipe(agent, sample, name: str, mem: str | None):
    print(f"\n==================== {name} ====================")
    if mem:
        print(f"[memory injected: {len(mem)} chars]")
    else:
        print("[no memory]")
    result = agent.run_episode(
        sample,
        model_name="deepseek-v4-flash",
        no_memory=(mem is None),
        no_prtree_update=True,
        external_memory_str=mem,
        memory_type="prtree",
        episode_idx=-1,
    )
    for line in result["trajectory"]:
        print("  ", line)
    elem = result["element_acc"]
    f1 = result["action_f1"]
    ss = result["step_success"]
    print(f"  → element_acc={sum(elem)/len(elem):.2f}  action_f1_mean={sum(f1)/len(f1):.2f}  "
          f"step_success={sum(ss)/len(ss):.2f}  success={result['success']}")

    # Pull the per-step pred/target from conversation for the key Step 8.
    pred_targets = []
    for turn in result.get("conversation", []):
        if isinstance(turn, dict) and "pred_act" in turn:
            pred_targets.append((turn["pred_act"], turn["target_act"]))
    return {
        "name": name,
        "element_acc": elem,
        "action_f1": f1,
        "step_success": ss,
        "success": result["success"],
        "pred_target_pairs": pred_targets,
    }


def main():
    sample = load_sample(EPISODE_IDX)
    print(f"Task: {sample['confirmed_task']}")
    print(f"Website: {sample['website']} | Steps: {len(sample['actions'])}")

    llm = create_llm_client(model_name="deepseek-v4-flash")
    agent = DualTreeMind2WebAgent(agent_name="recipe-exp", llm_client=llm)

    summary = []
    for name, mem in RECIPES.items():
        try:
            r = run_recipe(agent, sample, name, mem)
            summary.append(r)
        except Exception as e:
            print(f"  ✗ recipe {name} crashed: {e}")
            summary.append({"name": name, "error": str(e)})

    print("\n\n================ SUMMARY ================")
    for r in summary:
        if "error" in r:
            print(f"  {r['name']:30s} ERROR: {r['error']}")
            continue
        elem = r["element_acc"]
        ss = r["step_success"]
        elem_avg = sum(elem) / len(elem) if elem else 0
        ss_avg = sum(ss) / len(ss) if ss else 0
        # show pred for the wrong step (index 1 among real steps)
        pts = r["pred_target_pairs"]
        step8_str = ""
        if len(pts) >= 2:
            p, t = pts[1]
            mark = "✓" if p == t else "✗"
            step8_str = f"  step8 {mark}: pred={p} target={t}"
        print(f"  {r['name']:30s} elem={elem_avg:.2f} step_succ={ss_avg:.2f}{step8_str}")

    out = Path(__file__).parent / "memory_recipe_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
