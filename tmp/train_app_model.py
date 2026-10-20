"""Train the app's XGBoost on the no-Meli100 outer training pool."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import DATASETS, FEATURES, SEED, TARGET, TEST_SIZE, make_models, metrics  # noqa: E402

MODEL_PATH = ROOT / "xgboost_nomeli100_model.joblib"


def train_model(df):
    train, test = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, shuffle=True, random_state=SEED
    )
    model = make_models()["XGBoost"]
    model.fit(df.iloc[train][FEATURES], df.iloc[train][TARGET])
    test_rows = df.iloc[test]
    scores = metrics(test_rows[TARGET], model.predict(test_rows[FEATURES]))
    return model, train, test, scores


def main():
    df = pd.read_csv(DATASETS[1], encoding="utf-8-sig")
    if "Μέλι100" in set(df["Υγρό"]):
        raise ValueError("The app training dataset must exclude Μέλι100")
    model, train, test, scores = train_model(df)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")
    print(f"Training rows: {len(train)}; independent test rows: {len(test)}")
    print(f"Test MAE: {scores['mae']:.4f} s; RMSE: {scores['rmse']:.4f} s; R2: {scores['r2']:.4f}")


if __name__ == "__main__":
    main()
