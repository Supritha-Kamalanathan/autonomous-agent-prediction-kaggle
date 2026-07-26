"""
planner.py — Decides what the agent should do next.

Pipeline phases:
  eda          → run eda.py first
  forge_r1     → build round-1 baseline models from ROUND_1_SEQUENCE
  hill_climb_1 → first hill climb (agent triggers and reads results)
  creative     → agent designs new targeted models based on climber feedback
  hill_climb_2 → final hill climb over all models
  done         → session complete

The key insight: after hill_climb_1, the agent is no longer following a
fixed list. It reads which models contributed (non-zero weight) and decides
what complementary models to build next. This is where the LLM adds value.
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from state import (load, save, ROUND_1_SEQUENCE, CREATIVE_BUDGET,
                   MIN_MODELS_FOR_CLIMBING, WORK_DIR,
                   get_pending_node, get_valid_nodes, register_node)


def format_climber_summary(hill_climb_result: dict) -> str:
    """Format hill climb weights into a readable summary for the agent."""
    weights = hill_climb_result.get("weights", {})
    ensemble_auc = hill_climb_result.get("ensemble_auc")
    nodes = []

    # Get model types for each node_id
    state = load()
    for node_id_str, weight in sorted(weights.items(), key=lambda x: -x[1]):
        if weight > 0.005:
            node = state["nodes"].get(node_id_str, {})
            nodes.append(f"  [{weight:.3f}] {node.get('model_type', '?')} (node {node_id_str}, AUC={node.get('cv_auc', '?')})")

    ignored = [
        state["nodes"][nid]["model_type"]
        for nid, w in weights.items()
        if w <= 0.005 and nid in state["nodes"]
    ]

    lines = [f"ENSEMBLE_AUC: {ensemble_auc:.6f}"]
    lines.append("SELECTED_MODELS (non-zero weight, sorted by contribution):")
    lines.extend(nodes)
    if ignored:
        lines.append(f"IGNORED_MODELS (weight ~0, not contributing): {', '.join(ignored)}")
    return "\n".join(lines)


def main():
    state = load()
    eda = state.get("eda", {})
    phase = state.get("phase", "eda")

    # EDA not done
    if phase == "eda" or not eda.get("done"):
        print("PHASE: EDA")
        return

    # Budget low — skip straight to final hill climb
    if state.get("budget_low"):
        print("PHASE: HILL_CLIMB_2")
        return

    # Resume any pending (unfinished) node
    pending = get_pending_node(state)
    if pending:
        _emit_forge(pending, eda)
        return

    # ── Round 1: build baseline models ──────────────────────────────────────
    if phase == "forge_r1":
        model_index = state.get("model_index", 0)
        if model_index < len(ROUND_1_SEQUENCE):
            # Register and emit next round-1 model
            model_type = ROUND_1_SEQUENCE[model_index]
            node = register_node(state, model_type)
            state["model_index"] = model_index + 1
            save(state)
            _emit_forge(node, eda)
            return
        else:
            # Round 1 complete — trigger first hill climb
            state["phase"] = "hill_climb_1"
            save(state)
            print("PHASE: HILL_CLIMB_1")
            print(f"All {len(ROUND_1_SEQUENCE)} baseline models built. Run hill climbing now.")
            return

    # ── After hill climb 1: emit creative context ────────────────────────────
    if phase == "hill_climb_1":
        # This phase is set by the agent after it runs hill_climb.py --round 1
        # Transition to creative if hill climb 1 is recorded
        if state["hill_climb_1"].get("done"):
            state["phase"] = "creative"
            save(state)
            phase = "creative"

    if phase == "creative":
        creative_count = state.get("creative_count", 0)
        if creative_count >= CREATIVE_BUDGET:
            # Creative budget exhausted — final hill climb
            state["phase"] = "hill_climb_2"
            save(state)
            print("PHASE: HILL_CLIMB_2")
            print(f"Creative round complete ({CREATIVE_BUDGET} models built). Run final hill climbing.")
            return

        # Emit creative task with hill climb 1 context
        print("PHASE: CREATIVE")
        print(f"CREATIVE_MODEL_NUMBER: {creative_count + 1} of {CREATIVE_BUDGET}")
        print(f"ITERATIONS_REMAINING: ~{CREATIVE_BUDGET - creative_count - 1} creative slots left")
        print()
        print("=== HILL CLIMB 1 RESULTS (use this to decide what to build) ===")
        print(format_climber_summary(state["hill_climb_1"]))
        print()
        print("=== ALL MODELS ALREADY BUILT ===")
        for node in state["nodes"].values():
            auc_str = f"{node['cv_auc']:.5f}" if node.get("cv_auc") else "FAILED"
            print(f"  node={node['id']}  {node['model_type']}  AUC={auc_str}")
        print()
        print(f"TARGET_COL: {eda.get('target_col')}")
        print(f"NUMERIC_COLS: {', '.join(eda.get('numeric_cols', []))}")
        print(f"CATEGORICAL_COLS: {', '.join(eda.get('categorical_cols', []))}")
        print()
        print("NEXT_NODE_ID:", state["next_node_id"])
        print("INSTRUCTIONS: Design ONE new model that complements the selected ensemble.")
        print("  Use state.py --register --model-type YOUR_CHOSEN_TYPE to claim a node ID.")
        return

    # ── Final hill climb ─────────────────────────────────────────────────────
    if phase == "hill_climb_2":
        if state["hill_climb_2"].get("done"):
            print("PHASE: DONE")
        else:
            print("PHASE: HILL_CLIMB_2")
            print("Run final hill climbing over all models.")
        return

    if phase == "done":
        print("PHASE: DONE")
        return

    # Default: start forge_r1 if phase not set properly
    state["phase"] = "forge_r1"
    save(state)
    print("PHASE: FORGE_R1_STARTING")


def _emit_forge(node, eda):
    node_id = node["id"]
    print("PHASE: FORGE")
    print(f"MODEL_TYPE: {node['model_type']}")
    print(f"NODE_ID: {node_id}")
    print(f"TARGET_COL: {eda.get('target_col', 'target')}")
    print(f"NUMERIC_COLS: {', '.join(eda.get('numeric_cols', []))}")
    print(f"CATEGORICAL_COLS: {', '.join(eda.get('categorical_cols', []))}")
    print(f"N_TRAIN: {eda.get('n_train', '?')}")
    print(f"N_TEST: {eda.get('n_test', '?')}")
    print(f"CLASS_BALANCE: {eda.get('class_balance', {})}")
    print(f"SCRIPT: /work/model_{node_id}.py")
    print(f"OOF_OUT: /work/oof_{node_id}.npy")
    print(f"SUB_OUT: /work/submission_{node_id}.csv")


if __name__ == "__main__":
    main()
