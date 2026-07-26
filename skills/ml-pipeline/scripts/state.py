"""
state.py — Experiment state manager for the ML pipeline.

Reads and writes /work/state.json which tracks:
  - EDA results (target column, feature lists, class balance)
  - Every model node (type, status, CV AUC, file paths, submission ID)
  - Hill climbing results per round (weights, ensemble AUC)
  - Current pipeline phase

Used as both an importable module and a CLI tool called by the agent.

CLI usage:
  # Update an existing node after running it:
  python state.py --node-id 3 --status valid --cv-auc 0.923456

  # Register a new creative node (agent decides what to build):
  python state.py --register --model-type catboost_groupby_te_rs2
  > NODE_ID: 21
  > SCRIPT: /work/model_21.py
  > OOF_OUT: /work/oof_21.npy
  > SUB_OUT: /work/submission_21.csv
"""
import os
import json
import argparse

WORK_DIR = "/work"
STATE_FILE = os.path.join(WORK_DIR, "state.json")

# Round 1 sequence — 20 models chosen for MAXIMUM diversity across:
#   - Model families: LGBM, XGB, CatBoost, LogReg
#   - Feature engineering: raw, target-enc, groupby, products, binned, freq, combos
#   - Hyperparameter axes: lr, depth, leaves, regularization
# Order matters: put the broadest coverage first so early hill climbing is meaningful
ROUND_1_SEQUENCE = [
    "lgbm_raw__lr005_l63",          # LGBM baseline
    "xgb_raw__lr005_d6",             # XGB baseline
    "catboost_raw__d6",              # CatBoost baseline
    "lgbm_target_enc",               # Target encoding
    "lgbm_groupby__mean_std",        # Groupby aggregations
    "lgbm_products",                 # Interaction features
    "xgb_groupby",                   # XGB + groupby
    "catboost_groupby",              # CatBoost + groupby
    "lgbm_raw__lr002_l127",          # LGBM: deeper trees
    "xgb_raw__lr01_d4",              # XGB: shallower, faster
    "catboost_raw__d8",              # CatBoost: deeper trees
    "lgbm_target_enc__l127",         # Target enc + deeper LGBM
    "xgb_target_enc",                # XGB + target encoding
    "lgbm_binned",                   # Binned interaction features
    "lgbm_freq_enc",                 # Frequency encoding
    "lgbm_target_enc_groupby",       # Target enc + groupby combo
    "lgbm_groupby_products",         # Groupby + product combo
    "xgb_target_enc_groupby",        # XGB + target enc + groupby
    "catboost_target_enc_groupby",   # CatBoost + target enc + groupby
    "logreg",                        # Linear baseline for diversity
]

# How many creative models the agent builds in round 2 (guided by climber output)
CREATIVE_BUDGET = 7

# Minimum valid models before hill climbing is triggered
MIN_MODELS_FOR_CLIMBING = 6


def load() -> dict:
    if not os.path.exists(STATE_FILE):
        return {
            "phase": "eda",
            "eda": {"done": False},
            "model_index": 0,       # position in ROUND_1_SEQUENCE
            "creative_count": 0,    # creative models built so far
            "next_node_id": 1,
            "nodes": {},
            "hill_climb_1": {"done": False, "ensemble_auc": None, "weights": {}},
            "hill_climb_2": {"done": False, "ensemble_auc": None, "weights": {}},
            "submitted_ids": [],
        }
    with open(STATE_FILE) as f:
        return json.load(f)


def save(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_valid_nodes(state: dict) -> list:
    return [n for n in state["nodes"].values() if n.get("status") == "valid"]


def get_pending_node(state: dict):
    for n in state["nodes"].values():
        if n.get("status") == "pending":
            return n
    return None


def register_node(state: dict, model_type: str) -> dict:
    """Add a new node to state and return it. Used for creative round nodes."""
    node_id = state["next_node_id"]
    node = {
        "id": node_id,
        "model_type": model_type,
        "status": "pending",
        "cv_auc": None,
        "oof_file": os.path.join(WORK_DIR, f"oof_{node_id}.npy"),
        "submission_file": os.path.join(WORK_DIR, f"submission_{node_id}.csv"),
        "submission_id": None,
        "round": state.get("phase", "unknown"),
    }
    state["nodes"][str(node_id)] = node
    state["next_node_id"] += 1
    save(state)
    return node


def main():
    parser = argparse.ArgumentParser(description="Update experiment state after a model run.")
    parser.add_argument("--node-id", type=str, default=None, help="Node ID to update")
    parser.add_argument("--status", type=str, choices=["valid", "buggy", "pending"])
    parser.add_argument("--cv-auc", type=float, default=None)
    parser.add_argument("--submission-id", type=str, default=None)
    parser.add_argument("--register", action="store_true",
                        help="Register a new creative node and get back its ID")
    parser.add_argument("--model-type", type=str, default=None,
                        help="Model type name for --register")
    args = parser.parse_args()

    state = load()

    # Register mode: add a new creative node and print its paths
    if args.register:
        if not args.model_type:
            print("ERROR: --model-type is required with --register")
            return
        node = register_node(state, args.model_type)
        print(f"NODE_ID: {node['id']}")
        print(f"SCRIPT: /work/model_{node['id']}.py")
        print(f"OOF_OUT: {node['oof_file']}")
        print(f"SUB_OUT: {node['submission_file']}")
        return

    # Update mode: record results for an existing node
    if not args.node_id:
        print("ERROR: --node-id is required (or use --register to add a new node)")
        return

    key = str(args.node_id)
    if key not in state["nodes"]:
        print(f"ERROR: Node {key} not found in state.")
        return

    node = state["nodes"][key]
    if args.status:
        node["status"] = args.status
    if args.cv_auc is not None:
        node["cv_auc"] = args.cv_auc
    if args.submission_id:
        node["submission_id"] = args.submission_id
        if args.submission_id not in state["submitted_ids"]:
            state["submitted_ids"].append(args.submission_id)

    save(state)
    print(f"State updated — node {key}: status={node.get('status')}, cv_auc={node.get('cv_auc', 'N/A')}")


if __name__ == "__main__":
    main()
