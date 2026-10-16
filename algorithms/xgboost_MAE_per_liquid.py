"""XGBoost errors by liquid on the shared validation and test rows."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import DATASETS, TARGET, evaluate_model, split_rows  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASETS[0])
    args = parser.parse_args()
    df = pd.read_csv(args.dataset, encoding="utf-8-sig")
    rows = split_rows(df)
    _, scores, predictions, _ = evaluate_model("XGBoost", df, rows)
    print(f"XGBoost | {args.dataset.name} | train={len(rows['train'])}")
    for split in ("validation", "test"):
        subset = df.iloc[rows[split]].copy()
        subset["absolute_error"] = abs(subset[TARGET] - predictions[split])
        print(f"\n{split} MAE: {scores[split]['mae']:.4f} s")
        print(subset.groupby("Υγρό")["absolute_error"].agg(["count", "mean"]).to_string())


if __name__ == "__main__":
    main()
