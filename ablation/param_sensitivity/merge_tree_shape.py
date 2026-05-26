"""
跑完 tree_shape_*.csv 后执行：
  python merge_tree_shape.py
仅将 task/env_tree_level_counts 与 *_total_nodes / *_max_depth / consolidation_count
写回 sensitivity_all.csv，不覆盖 avg_reward / sr。
"""
import csv, os

RESULTS = os.path.join(os.path.dirname(__file__), "results")
MAIN    = os.path.join(RESULTS, "sensitivity_all.csv")
PATCHES = [
    os.path.join(RESULTS, "tree_shape_alf_k5.csv"),
    os.path.join(RESULTS, "tree_shape_sci_k5.csv"),
]
TREE_COLS = [
    "task_tree_total_nodes", "env_tree_total_nodes",
    "task_tree_max_depth",   "env_tree_max_depth",
    "task_tree_level_counts","env_tree_level_counts",
    "consolidation_count",
]

def key(row):
    return (row["benchmark"],
            round(float(row["tb"]), 4),
            round(float(row["eb"]), 4),
            int(row["k"]))

# load patches
patch_map = {}
for p in PATCHES:
    if not os.path.isfile(p):
        print(f"[skip] {p} not found yet")
        continue
    with open(p) as f:
        for row in csv.DictReader(f):
            patch_map[key(row)] = row
    print(f"[loaded] {p}  ({len(patch_map)} patch entries so far)")

if not patch_map:
    print("No patch files found — nothing to do.")
    raise SystemExit

# update main CSV
with open(MAIN) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

updated = 0
for row in rows:
    k = key(row)
    if k in patch_map:
        for col in TREE_COLS:
            if col in patch_map[k]:
                row[col] = patch_map[k][col]
        updated += 1
        print(f"  patched: bm={k[0]} tb={k[1]} eb={k[2]} K={k[3]}")

with open(MAIN, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"\nDone — {updated} rows updated in {MAIN}")
