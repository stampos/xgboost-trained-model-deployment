"""Evaluate a mean DummyRegressor on the shared folds and 80/20 holdout."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import DATASETS, FEATURES, SEED, TARGET, metrics  # noqa: E402


def evaluate_split(df, fit_rows, test_rows):
    model = DummyRegressor(strategy="mean")
    model.fit(df.iloc[fit_rows][FEATURES], df.iloc[fit_rows][TARGET])
    return metrics(
        df.iloc[test_rows][TARGET],
        model.predict(df.iloc[test_rows][FEATURES]),
    )


def main():
    output = {"model": "DummyRegressor(strategy='mean')", "seed": SEED, "datasets": []}
    for path in DATASETS:
        df = pd.read_csv(path, encoding="utf-8-sig")
        folds = [
            {"fold": number, "train_rows": len(fit), "test_rows": len(test),
             **evaluate_split(df, fit, test)}
            for number, (fit, test) in enumerate(
                KFold(n_splits=5, shuffle=True, random_state=SEED).split(df), start=1
            )
        ]
        fit_80, test_20 = train_test_split(
            np.arange(len(df)), test_size=0.20, shuffle=True, random_state=SEED
        )
        output["datasets"].append({
            "name": path.name,
            "folds_80_20": folds,
            "mean_5_fold": {
                key: float(np.mean([fold[key] for fold in folds]))
                for key in ("mae", "mse", "rmse", "r2")
            },
            "single_80_20": {
                "train_rows": len(fit_80), "test_rows": len(test_20),
                **evaluate_split(df, fit_80, test_20),
            },
        })
    result_path = ROOT / "tmp" / "dummy_regressor_results.json"
    result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
