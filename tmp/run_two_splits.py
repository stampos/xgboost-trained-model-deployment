"""Evaluate all models on nested 64/16/20 and direct 80/20 splits."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import (  # noqa: E402
    DATASETS, FEATURES, SEED, TARGET, evaluate_model, make_models, metrics, split_rows,
)

OUTPUT_PATH = ROOT / "tmp" / "two_split_results.json"


def main():
    result = {"seed": SEED, "features": FEATURES, "target": TARGET, "datasets": []}
    for path in DATASETS:
        df = pd.read_csv(path, encoding="utf-8-sig")
        rows_nested = split_rows(df)
        train_80, test_80 = train_test_split(
            np.arange(len(df)), test_size=0.20, shuffle=True, random_state=SEED
        )
        # The outer test rows stay fixed; the inner validation rows join
        # training in the direct 80/20 experiment.
        np.testing.assert_array_equal(test_80, rows_nested["test"])
        assert set(train_80) == set(rows_nested["train"]) | set(rows_nested["validation"])
        dataset = {
            "name": path.name,
            "total_rows": len(df),
            "splits": {
                "64/16/20": {key: [int(i) + 1 for i in value] for key, value in rows_nested.items()},
                "80/20": {
                    "train": [int(i) + 1 for i in train_80],
                    "test": [int(i) + 1 for i in test_80],
                },
            },
            "models": [],
        }
        for name in make_models():
            _, scores_nested, _, _ = evaluate_model(name, df, rows_nested)
            model_80 = make_models()[name]
            model_80.fit(df.iloc[train_80][FEATURES], df.iloc[train_80][TARGET])
            scores_80 = metrics(
                df.iloc[test_80][TARGET],
                model_80.predict(df.iloc[test_80][FEATURES]),
            )
            dataset["models"].append({
                "algorithm": name,
                "64/16/20": scores_nested,
                "80/20": {"test": scores_80},
            })
            print(f"{path.name}: {name}", flush=True)
        result["datasets"].append(dataset)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
