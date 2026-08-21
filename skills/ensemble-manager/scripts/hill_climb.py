"""Greedy OOF hill climber over all valid models with saved OOF predictions."""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

WORK_DIR = "/work"
TREE_FILE = os.path.join(WORK_DIR, "solution_tree.json")


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


def greedy_hill_climb(matrix, target, steps=100, max_iter=200):
    """Greedy weight hill climber that incrementally builds an ensemble."""
    individual = np.array([roc_auc_score(target, row) for row in matrix])
    weights = np.zeros(len(matrix))
    weights[int(individual.argmax())] = 1.0
    score = float(individual.max())

    for _ in range(max_iter):
        best_score, best_idx = score, None
        for i in range(len(matrix)):
            w = weights.copy()
            w[i] += 1.0 / steps
            w /= w.sum()
            s = roc_auc_score(target, matrix.T @ w)
            if s > best_score + 1e-8:
                best_score, best_idx = s, i
        if best_idx is None:
            break
        weights[best_idx] += 1.0 / steps
        weights /= weights.sum()
        score = best_score

    return weights, score, individual


def main():
    tree = load_tree()

    # Find valid nodes that have OOF files
    valid_with_oof = [
        n for n in tree["nodes"].values()
        if n.get("status") == "valid"
        and n.get("oof_file")
        and os.path.exists(n["oof_file"])
        and n.get("submission_file")
        and os.path.exists(n["submission_file"])
    ]

    if len(valid_with_oof) < 2:
        print(f"HILL_CLIMB_SKIP: Only {len(valid_with_oof)} models with OOF predictions - need at least 2.")
        # Fall back to best individual
        valid_nodes = [n for n in tree["nodes"].values() if n.get("status") == "valid" and n.get("submission_file")]
        if valid_nodes:
            lower_is_better = tree.get("meta", {}).get("lower_is_better", False)
            valid_nodes.sort(key=lambda x: x.get("score", -float("inf")), reverse=not lower_is_better)
            best = valid_nodes[0]
            print(f"HILL_CLIMB_SUBMIT: {best['submission_file']}")
        else:
            print("HILL_CLIMB_ERROR: No valid models found.")
        return

    # Load target column
    target_col = tree.get("eda", {}).get("target_col") if "eda" in tree else None
    if not target_col:
        # Detect from sample_submission
        try:
            sample = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
            train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"), nrows=5)
            for col in sample.columns:
                if col.lower() not in ("id", "row_id", "index") and col in train.columns:
                    target_col = col
                    break
            if not target_col:
                target_col = pd.read_csv(os.path.join(WORK_DIR, "train.csv")).columns[-1]
        except Exception as e:
            print(f"HILL_CLIMB_ERROR: Could not detect target column - {e}")
            return

    try:
        target = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))[target_col].values
    except Exception as e:
        print(f"HILL_CLIMB_ERROR: Could not load target - {e}")
        return

    # Load OOF and test predictions
    nodes_used = []
    oof_list = []
    test_list = []

    sample = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
    pred_col = [c for c in sample.columns if c.lower() not in ("id", "row_id", "index")]
    if len(pred_col) != 1:
        print("HILL_CLIMB_ERROR: Cannot determine prediction column from sample_submission.csv")
        return
    pred_col = pred_col[0]

    for node in valid_with_oof:
        try:
            oof = np.load(node["oof_file"])
            # Load test preds from the submission CSV
            sub_df = pd.read_csv(node["submission_file"])
            test_preds = sub_df[pred_col].values

            if len(oof) == len(target) and np.isfinite(oof).all() and np.std(oof) > 1e-8:
                nodes_used.append(node)
                oof_list.append(oof)
                test_list.append(test_preds)
        except Exception as e:
            print(f"  Skipping node {node['id']}: {e}")

    if len(nodes_used) < 2:
        print(f"HILL_CLIMB_SKIP: Only {len(nodes_used)} valid OOF arrays after filtering.")
        valid_nodes = [n for n in tree["nodes"].values() if n.get("status") == "valid" and n.get("submission_file")]
        if valid_nodes:
            lower_is_better = tree.get("meta", {}).get("lower_is_better", False)
            valid_nodes.sort(key=lambda x: x.get("score", -float("inf")), reverse=not lower_is_better)
            print(f"HILL_CLIMB_SUBMIT: {valid_nodes[0]['submission_file']}")
        return

    matrix = np.vstack(oof_list)        # shape (n_models, n_train)
    test_matrix = np.vstack(test_list)  # shape (n_models, n_test)

    weights, auc, individual = greedy_hill_climb(matrix, target)

    print("=== HILL CLIMB RESULTS ===")
    for node, w, s in zip(nodes_used, weights, individual):
        print(f"  Node {node['id']} ({node['type']}): weight={w:.4f}, individual_auc={s:.6f}")
    print(f"ENSEMBLE_AUC: {auc:.6f}")

    # Write ensemble submission
    ensemble_preds = test_matrix.T @ weights
    sample[pred_col] = ensemble_preds
    out_path = os.path.join(WORK_DIR, "submission_ensemble.csv")
    sample.to_csv(out_path, index=False)

    # Save to tree
    tree["hill_climb"] = {
        "ensemble_auc": auc,
        "weights": {str(n["id"]): float(w) for n, w in zip(nodes_used, weights)},
        "ensemble_file": out_path,
    }
    save_tree(tree)

    print(f"HILL_CLIMB_SUBMIT: {out_path}")


if __name__ == "__main__":
    main()
