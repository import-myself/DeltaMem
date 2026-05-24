# DeltaMem

**DeltaMem** is a dual-tree residual memory framework for LLM-based interactive agents. Rather than storing complete trajectories, it organizes experience into two independent residual trees — a **Task Tree** for goal-conditioned skills and an **Environment Tree** for scene-level knowledge. Each tree stores a generalized base experience as a root node and records new episodes as compact **delta nodes**, preserving only incremental differences. A global similarity retrieval with failure-node penalty reconstructs the full context via root-to-match chain composition. High-frequency convergent paths are autonomously consolidated into new root nodes, enabling the memory to self-organize over time.

## Supported Benchmarks

| Benchmark | Directory |
|-----------|-----------|
| ALFWorld  | `ALFWorld/` |
| ScienceWorld | `ScienceWorld/` |
| WebShop   | `WebShop/` |

## Project Structure

```
memory/          # Core PR-Tree memory implementation
  prtree/        #   Dual-tree manager, node definition, reader/writer, consolidation
common/          # Shared utilities (LLM client, retriever)
ALFWorld/        # ALFWorld agent and prompts
ScienceWorld/    # ScienceWorld agent and prompts
Mind2web/        # (not included)
WebShop/         # WebShop agent and prompts
ablation/        # Ablation scripts and parameter sensitivity analysis
embedding/       # Sentence embedding model (e5-base-v2)
```

## Setup

```bash
pip install -r requirements.txt
```

Each benchmark may require additional environment setup; see the corresponding subdirectory for details.

## Running

```bash
# ALFWorld
python ALFWorld/agent_alfworld_dual.py

# ScienceWorld
python ScienceWorld/run_sciworld.py

# WebShop
python WebShop/run.py

# Parameter sensitivity analysis
python ablation/param_sensitivity/plot_sensitivity.py --figure-type combined_trees
```

## Ablation & Parameter Search

See `ablation/` for threshold grid search, memory mode ablation, and train-to-test transfer experiments.
