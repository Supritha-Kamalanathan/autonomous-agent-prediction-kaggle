"""
eda.py — Exploratory data analysis for the ML pipeline.

Runs once at the start of each session. Dynamically detects the dataset
schema (target column, feature types, class balance, null rates, correlations)
since each competition mini-session uses a different dataset with different
column names and structures.

Writes findings to /work/state.json so the planner and agent have full
schema context on every subsequent iteration without re-reading the CSVs.
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import pandas as pd
import numpy as np
from state import load, save, WORK_DIR


def detect_target_column(train_df: pd.DataFrame, sample_sub: pd.DataFrame) -> str:
    """Find the target column by cross-referencing train with sample submission."""
    sub_cols = set(sample_sub.columns)
    id_keywords = {"id", "row_id", "index"}
    non_id_sub_cols = [c for c in sub_cols if c.lower() not in id_keywords]

    # If submission has a non-ID column that exists in train, that's the target
    for col in non_id_sub_cols:
        if col in train_df.columns:
            return col

    # Fallback: find a binary column in train that is NOT in test
    try:
        test_df = pd.read_csv(os.path.join(WORK_DIR, "test.csv"), nrows=5)
        test_cols = set(test_df.columns)
    except Exception:
        test_cols = set()

    for col in reversed(train_df.columns):
        if col in test_cols:
            continue
        unique_vals = set(train_df[col].dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
            return col

    # Last resort: last column
    return train_df.columns[-1]


def main():
    state = load()

    train_df = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
    sample_sub_path = os.path.join(WORK_DIR, "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path) if os.path.exists(sample_sub_path) else pd.DataFrame()

    try:
        test_df = pd.read_csv(os.path.join(WORK_DIR, "test.csv"))
        n_test = len(test_df)
    except Exception:
        n_test = 0

    target_col = detect_target_column(train_df, sample_sub)
    feature_cols = [c for c in train_df.columns if c != target_col]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    categorical_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train_df[c])]

    y = train_df[target_col]
    class_counts = y.value_counts().to_dict()
    class_balance = {str(k): round(v / len(y) * 100, 1) for k, v in class_counts.items()}
    null_rates = (train_df.isnull().mean() * 100).round(2).to_dict()
    high_null_cols = [c for c, r in null_rates.items() if r > 20 and c != target_col]
    high_cardinality_cats = {c: int(train_df[c].nunique()) for c in categorical_cols if train_df[c].nunique() > 50}

    # Top correlations with target
    top_correlations = {}
    try:
        for col in numeric_cols[:30]:
            corr = abs(train_df[col].fillna(train_df[col].median()).corr(y))
            top_correlations[col] = round(float(corr), 4)
        top_correlations = dict(sorted(top_correlations.items(), key=lambda x: -x[1])[:10])
    except Exception:
        pass

    state["eda"] = {
        "done": True,
        "target_col": target_col,
        "n_train": len(train_df),
        "n_test": n_test,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "class_balance": class_balance,
        "high_null_cols": high_null_cols,
        "high_cardinality_cats": high_cardinality_cats,
        "top_correlations": top_correlations,
    }
    state["phase"] = "forge"
    save(state)

    print("=== EDA COMPLETE ===")
    print(f"TARGET_COL: {target_col}")
    print(f"N_TRAIN: {len(train_df)} | N_TEST: {n_test}")
    print(f"NUMERIC_COLS ({len(numeric_cols)}): {', '.join(numeric_cols[:15])}{'...' if len(numeric_cols) > 15 else ''}")
    print(f"CATEGORICAL_COLS ({len(categorical_cols)}): {', '.join(categorical_cols[:10])}{'...' if len(categorical_cols) > 10 else ''}")
    print(f"CLASS_BALANCE: {class_balance}")
    if high_null_cols:
        print(f"HIGH_NULL_COLS: {high_null_cols}")
    if top_correlations:
        print(f"TOP_CORRELATIONS: {top_correlations}")


if __name__ == "__main__":
    main()
