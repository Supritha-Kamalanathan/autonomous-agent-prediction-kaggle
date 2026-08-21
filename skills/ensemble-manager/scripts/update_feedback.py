"""Persist structured LLM feedback into solution_tree.json - mirrors aideml's review_func_spec."""
import argparse
import json
import os

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", type=str, required=True)
    parser.add_argument("--analysis", type=str, default="")
    parser.add_argument("--is-bug", type=str, default=None, choices=["true", "false", "True", "False"])
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--lower-is-better", type=str, default=None, choices=["true", "false", "True", "False"])
    parser.add_argument("--submission-id", type=str, default=None)
    args = parser.parse_args()

    tree = load_tree()
    node_key = str(args.node_id)

    if node_key not in tree["nodes"]:
        print(f"ERROR: Node {node_key} not found.")
        return

    node = tree["nodes"][node_key]

    if args.analysis:
        node["analysis"] = args.analysis
    if args.submission_id:
        node["submission_id"] = args.submission_id
    if args.metric and args.metric.lower() != "null":
        try:
            node["score"] = float(args.metric)
        except ValueError:
            pass
    if args.lower_is_better is not None:
        tree["meta"]["lower_is_better"] = args.lower_is_better.lower() == "true"
    if args.is_bug is not None:
        if args.is_bug.lower() == "true":
            node["status"] = "buggy"
            if not node.get("error"):
                node["error"] = "Reviewer flagged as buggy."

    save_tree(tree)
    print(f"SUCCESS: Updated feedback for node {node_key}.")
    print(f"  Status: {node['status']}")
    print(f"  Score: {node.get('score', 'N/A')}")
    print(f"  Submission ID: {node.get('submission_id', 'N/A')}")
    print(f"  Analysis: {node.get('analysis', '')}")


if __name__ == "__main__":
    main()
