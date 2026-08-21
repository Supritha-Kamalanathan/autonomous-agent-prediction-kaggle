---
name: ensemble-manager
description: Manages the solution tree on disk, executes search policy, generates cross-iteration memory summaries, rich data previews, evaluates candidate scripts, and runs a greedy OOF hill-climb ensemble at the end.
---

# Ensemble Manager Skill

## Overall Strategy

This skill drives a **two-phase ensemble search**:

### Phase 1 - Diversity Search (Iterations 1-50)
Build up to **20 diverse baseline models** followed by **iterative improvements** using the search policy:
- **DRAFT** (first 20 nodes): Each draft targets a different model family - LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, GradientBoosting, LogisticRegression, and feature-engineered variants.
- **DEBUG** (stochastic, prob=0.5): If a buggy leaf node exists and its script file exists on disk, attempt to fix it.
- **IMPROVE** (all subsequent nodes): Apply one atomic Deotte feature engineering technique to the best valid node.

All candidate scripts must:
1. Save `submission_{NODE-ID}.csv` - the per-model submission file.
2. Save `oof_{NODE-ID}.npy` - out-of-fold predictions (shape `[n_train,]`, float probabilities) used for hill climbing.

### Phase 2 - Hill Climb Ensemble (Final Step)
When budget is low (time < 20 minutes OR submissions < 3), **run the greedy hill climber** on all valid OOF predictions.

---

## Available Scripts

### `scripts/search_policy.py`
Reads `solution_tree.json`, applies the search policy (Draft -> Debug -> Improve), and outputs a structured task prompt for the agent.

**Usage**: `run_skill_script(skill_name="ensemble-manager", file_path="scripts/search_policy.py")`

### `scripts/data_preview.py`
Generates a comprehensive dataset overview.

**Usage**: `run_skill_script(skill_name="ensemble-manager", file_path="scripts/data_preview.py")`

### `scripts/update_feedback.py`
Persists structured LLM feedback into `solution_tree.json`.

**Usage**:
```
run_skill_script(
    skill_name="ensemble-manager",
    file_path="scripts/update_feedback.py",
    args=["--node-id", "<id>", "--is-bug", "<true|false>", "--analysis", "<summary>",
          "--metric", "<score_float_or_null>", "--submission-id", "<sub_id>"]
)
```

### `scripts/hill_climb.py`
Greedy OOF hill climber over all valid models. Creates `submission_ensemble.csv`.

**Usage**: `run_skill_script(skill_name="ensemble-manager", file_path="scripts/hill_climb.py")`
**Trigger**: Call this when `time_minutes_remaining < 20` OR `submissions_remaining < 3`.
