"""
hill_climb.py — Greedy ensemble weight optimizer (CPU version).

Adapted from Chris Deotte's GPU hill climbing notebook:
  https://www.kaggle.com/code/cdeotte/gpu-hill-climbing-cv-0-05930

Algorithm:
  1. Start with the single best model by OOF AUC
  2. Greedily add weight (1/N_STEPS) to whichever model improves ensemble AUC
  3. Repeat until convergence
  4. Apply final weights to test predictions, save blended submission

Run twice per session:
  - After round 1 baselines: python hill_climb.py --round 1
  - After creative round:    python hill_climb.py --round 2  (final)

Round 1 results are written to state['hill_climb_1'] and displayed to the
agent so it can reason about what complementary models to build in round 2.
"""
import os
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from state import load, save, get_valid_nodes, WORK_DIR


def greedy_hill_climb(oof_matrix: np.ndarray, y_true: np.ndarray,
                      n_steps: int = 100, max_iterations: int = 300):
    """
    Greedy hill climbing over OOF predictions to find optimal ensemble weights.

    Args:
        oof_matrix:     shape (n_models, n_train)
        y_true:         shape (n_train,)
        n_steps:        weight granularity (step = 1/n_steps)
        max_iterations: maximum greedy steps

    Returns:
        weights:        shape (n_models,), normalized, sum to 1
        ensemble_auc:   final OOF AUC of the weighted ensemble
    """
    n_models = oof_matrix.shape[0]
    step = 1.0 / n_steps

    single_aucs = [roc_auc_score(y_true, oof_matrix[i]) for i in range(n_models)]
    best_start = int(np.argmax(single_aucs))
    weights = np.zeros(n_models)
    weights[best_start] = 1.0
    current_auc = single_aucs[best_start]

    print(f"Starting AUC (best single, index={best_start}): {current_auc:.6f}")

    for iteration in range(max_iterations):
        best_gain = 0.0
        best_i = -1
        for i in range(n_models):
            candidate = weights.copy()
            candidate[i] += step
            candidate /= candidate.sum()
            auc = roc_auc_score(y_true, oof_matrix.T @ candidate)
            if auc - current_auc > best_gain:
                best_gain = auc - current_auc
                best_i = i
        if best_i == -1:
            print(f"Converged at iteration {iteration + 1}.")
            break
        weights[best_i] += step
        weights /= weights.sum()
        current_auc += best_gain

    final_auc = roc_auc_score(y_true, oof_matrix.T @ weights)
    return weights, final_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=[1, 2], default=2,
                        help="Which hill climb round: 1 (after baselines) or 2 (final)")
    args = parser.parse_args()

    state = load()
    valid = get_valid_nodes(state)

    if len(valid) < 2:
        print(f"ERROR: Need at least 2 valid models. Only have {len(valid)}.")
        return

    print(f"Loading OOF predictions from {len(valid)} valid models...")

    train_df = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
    target_col = state["eda"]["target_col"]
    y_true = train_df[target_col].values

    oof_arrays, test_arrays, node_ids, model_types, single_aucs = [], [], [], [], []

    for node in valid:
        oof_path = node.get("oof_file")
        sub_path = node.get("submission_file")

        if not oof_path or not os.path.exists(oof_path):
            print(f"  SKIP node {node['id']}: OOF file missing")
            continue
        if not sub_path or not os.path.exists(sub_path):
            print(f"  SKIP node {node['id']}: submission file missing")
            continue

        oof = np.load(oof_path)
        if len(oof) != len(y_true):
            print(f"  SKIP node {node['id']}: OOF length mismatch")
            continue

        sub_df = pd.read_csv(sub_path)
        pred_col = [c for c in sub_df.columns if c.lower() not in ("id", "row_id", "index")][0]

        auc = node.get("cv_auc") or roc_auc_score(y_true, oof)
        print(f"  Node {node['id']} ({node['model_type']}): OOF AUC = {auc:.6f}")

        oof_arrays.append(oof)
        test_arrays.append(sub_df[pred_col].values)
        node_ids.append(node["id"])
        model_types.append(node["model_type"])
        single_aucs.append(auc)

    if len(oof_arrays) < 2:
        print("ERROR: Not enough loadable OOF arrays.")
        return

    oof_matrix = np.array(oof_arrays)
    test_matrix = np.array(test_arrays)

    print(f"\nRunning greedy hill climbing on {len(oof_arrays)} models...")
    weights, ensemble_auc = greedy_hill_climb(oof_matrix, y_true)

    best_single_auc = max(single_aucs)
    print(f"\n=== HILL CLIMB ROUND {args.round} RESULTS ===")
    print(f"Ensemble OOF AUC:     {ensemble_auc:.6f}")
    print(f"Best single OOF AUC:  {best_single_auc:.6f}")
    print(f"Gain from ensembling: +{ensemble_auc - best_single_auc:.6f}")
    print()

    weight_dict = {}
    for nid, mtype, w, sa in zip(node_ids, model_types, weights, single_aucs):
        if w > 0.001:
            print(f"  weight={w:.4f}  node={nid}  {mtype}  (single={sa:.6f})")
        weight_dict[str(nid)] = float(w)

    # Save blended submission
    suffix = f"r{args.round}"
    sample_sub = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
    pred_col = [c for c in sample_sub.columns if c.lower() not in ("id", "row_id", "index")][0]
    sample_sub[pred_col] = test_matrix.T @ weights
    ensemble_path = os.path.join(WORK_DIR, f"submission_ensemble_{suffix}.csv")
    sample_sub.to_csv(ensemble_path, index=False)

    # Best individual backup
    best_idx = int(np.argmax(single_aucs))
    best_individual_path = valid[best_idx].get("submission_file", "")

    print(f"\nVALIDATION_SCORE: {ensemble_auc:.6f}")
    print(f"ENSEMBLE_SUBMIT: {ensemble_path}")
    if best_individual_path and os.path.exists(best_individual_path):
        print(f"BEST_INDIVIDUAL_SUBMIT: {best_individual_path}")

    # Persist to state
    climb_key = f"hill_climb_{args.round}"
    state[climb_key]["done"] = True
    state[climb_key]["ensemble_auc"] = float(ensemble_auc)
    state[climb_key]["weights"] = weight_dict
    state[climb_key]["ensemble_submission_file"] = ensemble_path

    if args.round == 1:
        # Transition to creative phase
        state["phase"] = "creative"
    else:
        state["phase"] = "done"

    save(state)


if __name__ == "__main__":
    main()
