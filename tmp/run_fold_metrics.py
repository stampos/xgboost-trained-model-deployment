"""Five-fold cross-validation on each entire dataset, as in the original tabs."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import DATASETS, FEATURES, SEED, TARGET, make_models, metrics  # noqa: E402

OUTPUT_PATH = ROOT / "tmp" / "fold_metrics_full.json"
METRICS = ("mae", "rmse", "r2")


def main():
    result = {"scope": "full_dataset", "folds": 5, "seed": SEED, "datasets": []}
    for path in DATASETS:
        df = pd.read_csv(path, encoding="utf-8-sig")
        splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
        folds = list(splitter.split(df))
        dataset = {"name": path.name, "rows": len(df), "models": []}
        for name in make_models():
            model_result = {"algorithm": name, "folds": []}
            for number, (fit, held_out) in enumerate(folds, start=1):
                model = make_models()[name]
                model.fit(df.iloc[fit][FEATURES], df.iloc[fit][TARGET])
                score = metrics(
                    df.iloc[held_out][TARGET],
                    model.predict(df.iloc[held_out][FEATURES]),
                )
                model_result["folds"].append({
                    "fold": number,
                    "fit_rows": len(fit),
                    "held_out_rows": len(held_out),
                    "held_out_row_ids": [int(i) + 1 for i in held_out],
                    "mae": score["mae"],
                    "rmse": score["rmse"],
                    "r2": score["r2"],
                })
            model_result["mean"] = {
                key: float(np.mean([fold[key] for fold in model_result["folds"]]))
                for key in METRICS
            }
            model_result["std"] = {
                key: float(np.std([fold[key] for fold in model_result["folds"]]))
                for key in METRICS
            }
            dataset["models"].append(model_result)
            print(f"{path.name}: {name}", flush=True)
        result["datasets"].append(dataset)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
