"""Five-fold model selection inside an independent 80/20 test holdout."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import DATASETS, FEATURES, SEED, TARGET, TEST_SIZE, make_models, metrics  # noqa: E402

OUTPUT_PATH = ROOT / "outputs" / "ml-comparison-independent-cv" / "results.json"
METRICS = ("mae", "mse", "rmse", "r2")
FOLD_COUNT = 5


def evaluate_dataset(df):
    training_pool, test = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, shuffle=True, random_state=SEED
    )
    folds = []
    for number, (fit, validation) in enumerate(
        KFold(n_splits=FOLD_COUNT, shuffle=True, random_state=SEED).split(training_pool),
        start=1,
    ):
        folds.append({
            "fold": number,
            "fit_rows": training_pool[fit],
            "validation_rows": training_pool[validation],
        })

    results = []
    for name in make_models():
        fold_scores = []
        for fold in folds:
            model = make_models()[name]
            fit = df.iloc[fold["fit_rows"]]
            validation = df.iloc[fold["validation_rows"]]
            model.fit(fit[FEATURES], fit[TARGET])
            fold_scores.append({
                "fold": fold["fold"],
                "fit_rows": len(fit),
                "validation_rows": len(validation),
                **metrics(validation[TARGET], model.predict(validation[FEATURES])),
            })

        # Test data are touched only after all CV scores have been recorded.
        final_model = make_models()[name]
        final_model.fit(df.iloc[training_pool][FEATURES], df.iloc[training_pool][TARGET])
        held_out = df.iloc[test]
        prediction = final_model.predict(held_out[FEATURES])
        results.append({
            "algorithm": name,
            "folds": fold_scores,
            "cv_mean": {key: float(np.mean([score[key] for score in fold_scores])) for key in METRICS},
            "cv_std": {key: float(np.std([score[key] for score in fold_scores])) for key in METRICS},
            "final_fit_rows": len(training_pool),
            "test": metrics(held_out[TARGET], prediction),
        })

    return {
        "rows": len(df),
        "training_pool_rows": [int(row) + 1 for row in training_pool],
        "test_rows": [int(row) + 1 for row in test],
        "folds": [{
            "fold": fold["fold"],
            "fit_rows": [int(row) + 1 for row in fold["fit_rows"]],
            "validation_rows": [int(row) + 1 for row in fold["validation_rows"]],
        } for fold in folds],
        "models": results,
        "selected_by_cv_mae": min(results, key=lambda item: item["cv_mean"]["mae"])["algorithm"],
    }


def main():
    result = {
        "protocol": "independent 20% test; five-fold CV on outer 80%; final refit on outer 80%",
        "random_state": SEED,
        "fold_count": FOLD_COUNT,
        "selection_metric": "mean CV MAE",
        "datasets": [],
    }
    for path in DATASETS:
        df = pd.read_csv(path, encoding="utf-8-sig")
        dataset = evaluate_dataset(df)
        dataset["dataset"] = path.name
        dataset["dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result["datasets"].append(dataset)
        print(f"{path.name}: selected {dataset['selected_by_cv_mae']}", flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
