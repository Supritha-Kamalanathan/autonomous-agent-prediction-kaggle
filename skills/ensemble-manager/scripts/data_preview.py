"""Rich data preview generator matching aideml's data_preview.py."""
import os
import pandas as pd

WORK_DIR = "/work"
MAX_PREVIEW_CHARS = 6000


def generate_preview():
    parts = []

    parts.append("=== FILES ===")
    if os.path.exists(WORK_DIR):
        for f in sorted(os.listdir(WORK_DIR)):
            fpath = os.path.join(WORK_DIR, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                parts.append(f"  {f} ({size:,} bytes)")
    else:
        parts.append("  /work directory not found")

    for csv_name in ["train.csv", "test.csv"]:
        csv_path = os.path.join(WORK_DIR, csv_name)
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path, nrows=1000)
            parts.append(f"\n=== {csv_name} ({df.shape[0]}+ rows, {df.shape[1]} cols) ===")
            for col in df.columns:
                dtype = str(df[col].dtype)
                nulls = int(df[col].isnull().sum())
                nuniq = int(df[col].nunique())
                samples = df[col].dropna().head(3).tolist()
                parts.append(f"  {col}: {dtype}, {nulls} nulls, {nuniq} unique, samples={samples}")
        except Exception as e:
            parts.append(f"\n=== {csv_name} (error reading: {e}) ===")

    sub_path = os.path.join(WORK_DIR, "sample_submission.csv")
    if os.path.exists(sub_path):
        try:
            sub = pd.read_csv(sub_path, nrows=3)
            parts.append(f"\n=== sample_submission.csv ===")
            parts.append(f"  Columns: {list(sub.columns)}")
            parts.append(f"  {sub.head(3).to_string(index=False)}")
        except Exception as e:
            parts.append(f"\n=== sample_submission.csv (error reading: {e}) ===")

    preview = "\n".join(parts)
    if len(preview) > MAX_PREVIEW_CHARS:
        preview = preview[:MAX_PREVIEW_CHARS] + "\n... (truncated)"
    return preview


if __name__ == "__main__":
    print(generate_preview())
