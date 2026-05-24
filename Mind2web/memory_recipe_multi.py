"""
Multi-episode memory recipe experiment (Mind2Web test_website).

For each failing episode we test 4 memory recipes:
  01_baseline       — no memory injected
  02_abstract       — abstract prose skill (current PRTree style)
  03_step_template  — numbered step list with element descriptors
  04_pitfall_only   — single-sentence decision pitfall (oracle, from trajectory)

Outputs per-episode element_acc, step_success and whether the previously-wrong
step is fixed.
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

DATA_GLOB = "/hdd/REDACTED_USER/DeltaMem/Mind2web/data/test_website/*.json"
SCORE_PATH = "/hdd/REDACTED_USER/DeltaMem/Mind2web/data/scores_all_data.pkl"


# ---------------------------------------------------------------------------
# Recipes per episode
#   abstract: copy-paste of current PRTree prose style (no episode-specific signal)
#   step_template: numbered procedure with element descriptors (no pitfall)
#   pitfall_only: ONE decision-disambiguation sentence written from oracle trajectory
# ---------------------------------------------------------------------------

EPISODES = {
    132: {
        "abstract": """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Flight booking on travel websites.
- **Procedure**: Identify origin/destination fields by labels like 'From'/'To',
  fill them, then trigger the search. Use the cheapest-sort option on results.
""",
        "step_template": """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Flight booking on travel-meta sites.
- **Procedure** (step-by-step):
  Step 1: CLICK <menuitem text='Flights'>
  Step 2: CLICK <input text='City or Airport'/'From'>
  Step 3: TYPE origin city
  Step 4: CLICK <option containing origin>
  Step 5: TYPE destination city
  Step 6: CLICK <option containing destination>
  Step 7: CLICK <button text='Find flights'/'Search'>
  Step 8: CLICK <div text='Best Value'/'Cheapest'>
""",
        "pitfall": """# Task Skill Memory (decision pitfall)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Flight booking where the task does not specify Round-trip vs One-way.
- **Pitfall**: Round-trip is the DEFAULT and is what the task wants when unspecified.
  Do NOT click any 'One-way' / 'Multi-city' toggle. After filling origin & destination
  you should CLICK the 'Find flights' button directly.
""",
    },
    131: {
        "abstract": """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Writing/creating a review on a travel website.
- **Procedure**: Navigate to the website's review creation entry, choose the right
  category (hotel/restaurant/attraction), then proceed to the review form.
""",
        "step_template": """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Creating a review on tripadvisor.
- **Procedure** (step-by-step):
  Step 1: CLICK <a text='Review'>  — open review entry
  Step 2: CLICK <button text='Hotels'/'Restaurants'>  — choose category
  Step 3: TYPE name of place
""",
        "pitfall": """# Task Skill Memory (decision pitfall)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: 'Create a review' tasks on tripadvisor when the navbar contains a
  short 'Review' icon link AND there is an inline 'Write a review' link elsewhere.
- **Pitfall**: ALWAYS take the top-nav 'Review' icon link (short text='Review') as
  the canonical entry. Do NOT click the inline 'Write a review' link — it leads to
  a different flow and is the wrong target for this task type.
""",
    },
    137: {
        "abstract": """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Searching appliances on bestbuy.
- **Procedure**: Type the product category in the search box, pick a suggestion,
  then filter by specs (size, features, price).
""",
        "step_template": """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Appliance search on bestbuy.
- **Procedure** (step-by-step):
  Step 1: CLICK <input search>
  Step 2: TYPE the broad category name
  Step 3: CLICK <a> in autocomplete matching the most specific variant available
  Step 4: Apply spec filters on results
""",
        "pitfall": """# Task Skill Memory (decision pitfall)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: bestbuy autocomplete shows a generic term and several pre-filtered
  variants (e.g. 'cooktop', 'cooktop induction', 'cooktop 30"').
- **Pitfall**: Always pick the GENERIC term suggestion (shortest text, only the
  category word). Bestbuy's canonical landing page is the broad category;
  pre-filtered suggestions go to a narrower scope that loses available filters.
""",
    },
    149: {
        "abstract": """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Booking concert tickets on stubhub.
- **Procedure**: Search the artist, pick the right date and venue, choose seats,
  view pricing.
""",
        "step_template": """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Concert ticket browsing on stubhub.
- **Procedure** (step-by-step):
  Step 1: CLICK <input search>
  Step 2: TYPE artist name
  Step 3: CLICK <a> for the matching artist event
  Step 4: CLICK <button text='Continue'>  — confirm artist+quantity
  Step 5: Filter results by date/location
""",
        "pitfall": """# Task Skill Memory (decision pitfall)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: stubhub interstitial showing an artist title at the top, a
  'Not looking for X?' hint, and a 'Continue' button.
- **Pitfall**: Re-clicking the artist title leads to a generic event listing.
  When the page already confirms the right artist, CLICK 'Continue' — that's
  the action that advances to date/seat selection.
""",
    },
    88: {
        "abstract": """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Creating a product collection on shopping.google.
- **Procedure**: Open the navigation menu, find the Collections feature, create
  a new collection with the chosen name.
""",
        "step_template": """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Account features on shopping.google.
- **Procedure** (step-by-step):
  Step 1: CLICK <button aria-label='menu'/text='menu'>  — open main menu
  Step 2: CLICK <a/span text='Collections'>  — enter the feature
  Step 3: CLICK <button text='Create'/'New collection'>
  Step 4: TYPE the desired collection name
""",
        "pitfall": """# Task Skill Memory (decision pitfall)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: shopping.google task that needs the Collections feature, page shows
  a Collections link inside a side iframe/app-switcher AND a top-left main-menu
  button.
- **Pitfall**: The side iframe 'Collections' link goes to a static landing page.
  ALWAYS open the main menu (top-left button labeled 'menu') first; collection
  management is reached from inside that menu, not the side link.
""",
    },
    122: {
        "abstract": """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: Product search on macys with multiple filters.
- **Procedure**: Use either the keyword search bar or category navigation,
  then refine with filters (price, rating).
""",
        "step_template": """# Task Skill Memory (procedural)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: macys multi-filter shopping (price, rating, category).
- **Procedure** (step-by-step):
  Step 1: CLICK the category nav matching the task (e.g. <a text='Shoes'/'Women'>)
  Step 2: Use category landing page filters to narrow size/age range
  Step 3: Apply price and rating filters
""",
        "pitfall": """# Task Skill Memory (decision pitfall)
### [Base Skill] [✅ SUCCESS]
- **Trigger**: macys task needing filters (price, rating) AND a category that
  matches a top-nav link (e.g. 'Shoes', 'Women', 'Kids').
- **Pitfall**: Do NOT TYPE the query into the keyword search bar — that loses
  the category filter sidebar. CLICK the matching top-nav category link first;
  filtering is only reachable from category landing pages.
""",
    },
}


def load_samples():
    samples = []
    for fn in sorted(glob.glob(DATA_GLOB)):
        with open(fn) as f:
            samples.extend(json.load(f))
    return add_scores(samples, score_path=SCORE_PATH)


def run_one(agent, sample, mem):
    return agent.run_episode(
        sample,
        model_name="deepseek-v4-flash",
        no_memory=(mem is None),
        no_prtree_update=True,
        external_memory_str=mem,
        memory_type="prtree",
        episode_idx=-1,
    )


def extract_wrong_step_pair(traj):
    for line in traj:
        if line.startswith('Step ') and 'elem_acc=0' in line:
            return line
    return None


def main():
    samples = load_samples()
    llm = create_llm_client(model_name="deepseek-v4-flash")
    agent = DualTreeMind2WebAgent(agent_name="recipe-multi", llm_client=llm)

    grid = {}
    for ep_idx, recipes in EPISODES.items():
        sample = samples[ep_idx]
        print(f"\n############### ep{ep_idx} ({sample['website']}) ###############")
        print(f"Task: {sample['confirmed_task'][:130]}")
        ep_results = {}
        configs = [
            ("01_baseline", None),
            ("02_abstract", recipes["abstract"]),
            ("03_step_template", recipes["step_template"]),
            ("04_pitfall_only", recipes["pitfall"]),
        ]
        for name, mem in configs:
            r = run_one(agent, sample, mem)
            elem = r["element_acc"]
            ss = r["step_success"]
            wrong = extract_wrong_step_pair(r["trajectory"])
            elem_avg = sum(elem) / len(elem) if elem else 0
            ss_avg = sum(ss) / len(ss) if ss else 0
            ep_results[name] = {
                "element_acc_avg": elem_avg,
                "step_success_avg": ss_avg,
                "remaining_wrong": wrong,
                "trajectory": r["trajectory"],
            }
            print(f"  {name:18s} elem={elem_avg:.2f}  step_succ={ss_avg:.2f}  "
                  f"wrong_left={wrong if wrong else 'none'}")
        grid[ep_idx] = ep_results

    print("\n\n================== AGGREGATE ==================")
    rows = ["ep    | baseline | abstract | step_tpl | pitfall"]
    rows.append("-" * 60)
    for ep_idx, res in grid.items():
        rows.append(
            f"{ep_idx:5d} |  {res['01_baseline']['element_acc_avg']:.2f}    |  "
            f"{res['02_abstract']['element_acc_avg']:.2f}    |  "
            f"{res['03_step_template']['element_acc_avg']:.2f}    |  "
            f"{res['04_pitfall_only']['element_acc_avg']:.2f}"
        )
    print("\n".join(rows))

    # Step-success aggregate
    print("\n--- step_success ---")
    sums = {k: 0.0 for k in ("01_baseline", "02_abstract", "03_step_template", "04_pitfall_only")}
    for ep_idx, res in grid.items():
        for k in sums:
            sums[k] += res[k]["step_success_avg"]
    n = len(grid)
    print(f"  mean step_success across {n} episodes:")
    for k, v in sums.items():
        print(f"    {k:18s} {v / n:.3f}")

    out = Path(__file__).parent / "memory_recipe_multi_results.json"
    with open(out, "w") as f:
        json.dump(grid, f, indent=2)
    print(f"\nDetailed results → {out}")


if __name__ == "__main__":
    main()
