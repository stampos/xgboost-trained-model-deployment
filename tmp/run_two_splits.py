"""Evaluate all models on shared 60/20/20 and 80/20 splits for both datasets."""

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
        rows_60 = split_rows(df)
        train_80, test_80 = train_test_split(
            np.arange(len(df)), test_size=0.20, shuffle=True, random_state=SEED
        )
        # The 20% test rows are identical in the two experiments. The extra
        # 20% validation rows from version 1 join training in version 2.
        np.testing.assert_array_equal(test_80, rows_60["test"])
        assert set(train_80) == set(rows_60["train"]) | set(rows_60["validation"])
        dataset = {
            "name": path.name,
            "total_rows": len(df),
            "splits": {
                "60/20/20": {key: [int(i) + 1 for i in value] for key, value in rows_60.items()},
                "80/20": {
                    "train": [int(i) + 1 for i in train_80],
                    "test": [int(i) + 1 for i in test_80],
                },
            },
            "models": [],
        }
        for name in make_models():
            _, scores_60, _, _ = evaluate_model(name, df, rows_60)
            model_80 = make_models()[name]
            model_80.fit(df.iloc[train_80][FEATURES], df.iloc[train_80][TARGET])
            scores_80 = metrics(
                df.iloc[test_80][TARGET],
                model_80.predict(df.iloc[test_80][FEATURES]),
            )
            dataset["models"].append({
                "algorithm": name,
                "60/20/20": scores_60,
                "80/20": {"test": scores_80},
            })
            print(f"{path.name}: {name}", flush=True)
        result["datasets"].append(dataset)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
