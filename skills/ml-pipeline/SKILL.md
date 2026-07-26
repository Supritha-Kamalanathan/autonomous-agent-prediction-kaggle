---
name: ml-pipeline
description: Backend scripts for the grandmaster ensemble pipeline. Manages experiment state, EDA, model sequencing, and CPU hill climbing ensemble optimization.
---

# ML Pipeline Skill

## Scripts

### `scripts/eda.py`
Profiles the dataset: detects target column, feature types, class balance, null rates, and correlation hints. Writes findings to `/work/state.json` so every subsequent iteration knows the data schema without re-reading the full CSV.

### `scripts/planner.py`
Reads `/work/state.json` and decides what to build next. Registers the next model config from the 57-item sequence, emits a structured task block (MODEL_TYPE, NODE_ID, file paths, column names), and signals HILL_CLIMB when the sequence is exhausted or the budget is low.

### `scripts/state.py`
CLI tool for reading and writing `/work/state.json`. Used by the agent after each model run to record status (valid/buggy), CV AUC, and submission ID for each node.

### `scripts/hill_climb.py`
CPU greedy hill climbing over all valid OOF prediction arrays. Finds optimal ensemble weights that maximize OOF AUC ROC. Generates the final blended test submission. Adapted from Chris Deotte's GPU hill climbing notebook for CPU-only execution.
