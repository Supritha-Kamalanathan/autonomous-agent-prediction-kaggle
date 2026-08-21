"""Search policy - guides toward diverse model families and Deotte techniques."""
import json
import os
import random
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
from data_preview import generate_preview

WORK_DIR = "/work"
TREE_FILE = os.path.join(WORK_DIR, "solution_tree.json")
MAX_DRAFTS = 20
MAX_DEBUG_DEPTH = 3
DEBUG_PROB = 0.5

# 20 diverse model families spanning all diversity axes
DRAFT_GUIDES = [
    "LGBM-RAW: LightGBM with label-encoded categoricals only. n_estimators=3000, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, early stopping patience=50. Fast strong baseline.",
    "XGB-RAW: XGBoost with label-encoded categoricals. max_depth=6, learning_rate=0.05, n_estimators=3000, subsample=0.8, colsample_bytree=0.8, eval_metric='auc', early stopping 50. Different tree growth from LGBM.",
    "CATBOOST-NATIVE: CatBoost with ALL categorical columns passed as cat_features — no manual encoding needed. iterations=1500, learning_rate=0.05, depth=6, od_wait=50. Symmetric oblivious trees = very different error pattern.",
    "RF-SKLEARN: RandomForestClassifier, n_estimators=500, max_depth=None, min_samples_leaf=5, max_features='sqrt'. Bagging with random splits — completely different from boosting. High variance, low bias.",
    "EXTRATREES: ExtraTreesClassifier, n_estimators=500, max_depth=None, min_samples_leaf=3. Extreme randomization: splits are fully random. Even more decorrelated than RF.",
    "LOGREG-SCALED: LogisticRegression with StandardScaler on all numeric features. C=0.1. Linear decision boundary — captures global linear trends that tree models miss.",
    "GBM-SKLEARN: GradientBoostingClassifier (sklearn), n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8. Different boosting implementation from XGB/LGBM.",
    "LGBM-NESTED-TE: LightGBM with nested 5-fold target encoding on ALL categorical columns (fit TE inside each training fold — zero leakage). num_leaves=127. TE adds powerful smooth signal.",
    "XGB-FREQ: XGBoost with frequency encoding for every categorical column (map each value to its count/total in training fold). max_depth=6, early stopping. Frequency signal captures popularity/rarity.",
    "LGBM-DECIMAL: LightGBM with decimal digit extraction on ALL float features: d1=floor(frac*10), d2=floor(frac*100)%10, mod10=floor(x)%10. Captures hidden discretization patterns in floats.",
    "LGBM-BIGRAMS: LightGBM with categorical bigrams — combine top-2 categorical columns as 'A__B' cross-feature, then nested TE on bigrams. Captures interaction effects between categoricals.",
    "CATBOOST-FREQ: CatBoost with cat_features + frequency encoding on top. iterations=1500, depth=6. Different base model + frequency signal.",
    "XGB-NESTED-TE: XGBoost with nested 5-fold target encoding on ALL categoricals. max_depth=5, learning_rate=0.05. Same FE as LGBM-NESTED-TE but different tree structure.",
    "LGBM-BINNING: LightGBM with multi-scale quantile binning (q=10, q=50) on top numeric features, binned columns target-encoded. num_leaves=63. Captures non-linear numeric patterns.",
    "LGBM-DEEP: LightGBM with num_leaves=255, max_depth=12, learning_rate=0.02, n_estimators=5000, early stopping 100. Deep trees capture complex interactions.",
    "LGBM-SHALLOW: LightGBM with num_leaves=31, max_depth=5, learning_rate=0.1, n_estimators=1000, min_child_samples=50. Shallow trees — high regularization, low overfitting.",
    "XGB-DEEP: XGBoost with max_depth=10, learning_rate=0.02, n_estimators=5000, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0. Deep trees with L1/L2 regularization.",
    "CATBOOST-DEEP: CatBoost with depth=8, iterations=2000, learning_rate=0.03, l2_leaf_reg=3. Deeper symmetric trees.",
    "LGBM-ALL-FE: LightGBM combining nested TE + frequency encoding + decimal extraction + bigrams all at once. num_leaves=127. Kitchen-sink feature engineering.",
    "XGB-TE-FREQ: XGBoost with both nested TE AND frequency encoding for all categoricals. max_depth=6, learning_rate=0.05. Two complementary FE techniques combined.",
]

# 11 IMPROVE techniques cycling through Deotte's full FE arsenal
IMPROVE_GUIDES = [
    "Add nested 5-fold target encoding for ALL categorical and binned numeric columns. Also add TE stats: std, min, max per group.",
    "Extract decimal digits from float features: floor(frac*10) and floor(frac*100)%10 as new integer features.",
    "Add categorical bigrams: combine top-2 categorical columns into a single cross-feature, then target-encode it.",
    "Add multi-scale quantile binning (q=10, q=50) on top numeric features, convert to string category, then target-encode.",
    "Add frequency encoding for all categorical columns: map each value to its normalized frequency in the training fold.",
    "Tune LightGBM: increase num_leaves to 127, add reg_alpha=0.1, reg_lambda=0.1, try learning_rate=0.02.",
    "Use CatBoost with pre-computed arithmetic interactions: difference, ratio, and product of top correlated numeric pairs.",
    "Use ExtraTreesClassifier (sklearn): n_estimators=500, extreme randomization. Bagging with fully random splits = high OOF diversity.",
    "Add TE statistics beyond mean: compute std, min, max, q10, q90 of target per category group, use all as features.",
    "Add radix interaction features: (int(NumCol * 100) + cat_code * 100_000) for top numeric x categorical pairs.",
    "Use GradientBoostingClassifier (sklearn): n_estimators=500, learning_rate=0.05, max_depth=5. Different implementation than XGB/LGBM.",
]


def read_script_content(script_path):
    """Read a script file and return its content, or a placeholder if missing."""
    try:
        with open(script_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"# Could not read script: {e}"


def load_tree():
    if not os.path.exists(TREE_FILE):
        return {"nodes": {}, "next_id": 1, "draft_count": 0, "meta": {}}
    with open(TREE_FILE, "r") as f:
        tree = json.load(f)
        if "meta" not in tree:
            tree["meta"] = {}
        return tree


def save_tree(tree):
    with open(TREE_FILE, "w") as f:
        json.dump(tree, f, indent=2)


def generate_memory_summary(nodes):
    valid_nodes = [n for n in nodes.values() if n.get("status") == "valid"]
    buggy_nodes = [n for n in nodes.values() if n.get("status") == "buggy"]
    if not valid_nodes and not buggy_nodes:
        return "No completed experiments yet."
    parts = []
    if valid_nodes:
        parts.append(f"VALID MODELS ({len(valid_nodes)} total):")
        for n in sorted(valid_nodes, key=lambda x: x["id"]):
            part = f"  Node {n['id']} ({n['type']}): score={n.get('score', 'N/A')}"
            if n.get("plan"):
                part += f" | Approach: {n['plan'][:120]}"
            if n.get("analysis"):
                part += f" | Reviewer: {n['analysis'][:80]}"
            parts.append(part)
    if buggy_nodes:
        parts.append(f"\nFAILED MODELS ({len(buggy_nodes)} total — avoid repeating these approaches):")
        for n in sorted(buggy_nodes, key=lambda x: x["id"])[-5:]:
            parts.append(f"  Node {n['id']}: {n.get('error', 'unknown error')[:100]}")
    return "\n".join(parts)


def compute_debug_depth(nodes, parent_node):
    depth = 0
    cursor = parent_node
    while cursor and cursor.get("type") == "debug":
        depth += 1
        pid = cursor.get("parent")
        cursor = nodes.get(str(pid)) if pid is not None else None
    return depth


def main():
    tree = load_tree()
    nodes = tree["nodes"]
    meta = tree.get("meta", {})

    # Check for already-pending node (retry it)
    pending = [n for n in nodes.values() if n.get("status") == "pending"]
    if pending:
        target = pending[-1]
        node_id = target["id"]
        memory_str = generate_memory_summary(nodes)
        preview = generate_preview()
        base_script = f"model_{target.get('parent', 'None')}.py" if target.get("parent") else "None"
        base_content = ""
        if target.get("parent"):
            path = os.path.join(WORK_DIR, f"model_{target['parent']}.py")
            base_content = f"\nBASE_SCRIPT_CONTENT:\n{read_script_content(path)}"
        print(f"TASK_TYPE: {target['type'].upper()}")
        print(f"NODE-ID: {node_id}")
        print(f"BASE_SCRIPT: {base_script}{base_content}")
        print(f"DATA_PREVIEW:\n{preview}")
        print(f"SUMMARY: Retrying pending {target['type']} task.")
        print(f"MEMORY:\n{memory_str}")
        return

    memory_str = generate_memory_summary(nodes)
    preview = generate_preview()
    draft_count = tree.get("draft_count", 0)

    # Rule 1: Draft new baselines
    if draft_count < MAX_DRAFTS:
        node_id = tree["next_id"]
        tree["next_id"] += 1
        tree["draft_count"] = draft_count + 1
        guide = DRAFT_GUIDES[draft_count % len(DRAFT_GUIDES)]
        tree["nodes"][str(node_id)] = {
            "id": node_id, "parent": None, "status": "pending",
            "type": "draft", "debug_depth": 0
        }
        save_tree(tree)
        print(f"TASK_TYPE: DRAFT")
        print(f"NODE-ID: {node_id}")
        print(f"BASE_SCRIPT: None")
        print(f"DATA_PREVIEW:\n{preview}")
        print(f"SUMMARY: Draft baseline #{draft_count + 1}/{MAX_DRAFTS}. GUIDANCE: {guide}")
        print(f"MEMORY:\n{memory_str}")
        return

    # Rule 2: Stochastic debug
    parent_ids = {n.get("parent") for n in nodes.values() if n.get("parent") is not None}
    buggy_leaves = [
        n for n in nodes.values()
        if n.get("status") == "buggy"
        and n["id"] not in parent_ids
        and compute_debug_depth(nodes, n) < MAX_DEBUG_DEPTH
        and os.path.exists(os.path.join(WORK_DIR, f"model_{n['id']}.py"))
    ]
    if buggy_leaves and random.random() < DEBUG_PROB:
        parent = random.choice(buggy_leaves)
        new_id = tree["next_id"]
        tree["next_id"] += 1
        debug_depth = compute_debug_depth(nodes, parent) + 1
        tree["nodes"][str(new_id)] = {
            "id": new_id, "parent": parent["id"], "status": "pending",
            "type": "debug", "debug_depth": debug_depth
        }
        save_tree(tree)
        parent_script_path = os.path.join(WORK_DIR, f"model_{parent['id']}.py")
        parent_script_content = read_script_content(parent_script_path)
        print(f"TASK_TYPE: DEBUG")
        print(f"NODE-ID: {new_id}")
        print(f"BASE_SCRIPT: model_{parent['id']}.py")
        print(f"BASE_SCRIPT_CONTENT:\n{parent_script_content}")
        print(f"DATA_PREVIEW:\n{preview}")
        print(f"SUMMARY: Script model_{parent['id']}.py failed:\n{parent.get('error', 'Unknown error')}\nFix the bug while preserving the approach.")
        print(f"MEMORY:\n{memory_str}")
        return

    # Rule 3: Improve best valid node
    valid_nodes = [n for n in nodes.values() if n.get("status") == "valid"]
    if not valid_nodes:
        # Fallback: draft another
        node_id = tree["next_id"]
        tree["next_id"] += 1
        tree["nodes"][str(node_id)] = {
            "id": node_id, "parent": None, "status": "pending",
            "type": "draft", "debug_depth": 0
        }
        save_tree(tree)
        print(f"TASK_TYPE: DRAFT")
        print(f"NODE-ID: {node_id}")
        print(f"BASE_SCRIPT: None")
        print(f"DATA_PREVIEW:\n{preview}")
        print(f"SUMMARY: No valid nodes yet. Draft another baseline.")
        print(f"MEMORY:\n{memory_str}")
        return

    lower_is_better = meta.get("lower_is_better", False)
    if lower_is_better:
        best_node = min(valid_nodes, key=lambda x: x.get("score", float("inf")))
    else:
        best_node = max(valid_nodes, key=lambda x: x.get("score", -float("inf")))

    # Pick an IMPROVE guide not yet applied
    improve_index = len([n for n in nodes.values() if n.get("type") == "improve"]) % len(IMPROVE_GUIDES)
    guide = IMPROVE_GUIDES[improve_index]

    new_id = tree["next_id"]
    tree["next_id"] += 1
    tree["nodes"][str(new_id)] = {
        "id": new_id, "parent": best_node["id"], "status": "pending",
        "type": "improve", "debug_depth": 0
    }
    save_tree(tree)
    best_script_path = os.path.join(WORK_DIR, f"model_{best_node['id']}.py")
    best_script_content = read_script_content(best_script_path)
    print(f"TASK_TYPE: IMPROVE")
    print(f"NODE-ID: {new_id}")
    print(f"BASE_SCRIPT: model_{best_node['id']}.py")
    print(f"BASE_SCRIPT_CONTENT:\n{best_script_content}")
    print(f"DATA_PREVIEW:\n{preview}")
    print(f"SUMMARY: Best node model_{best_node['id']}.py scored {best_node.get('score')}. GUIDANCE: {guide}")
    print(f"MEMORY:\n{memory_str}")


if __name__ == "__main__":
    main()
