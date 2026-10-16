"""One reproducible experiment protocol for every regression model.

Each dataset has its own fixed 60/20/20 train/validation/test rows. All models
see the same rows within a dataset. The test set is never used for fitting.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DATASETS = (
    ROOT / "data" / "mesos_oros.csv",
    ROOT / "data" / "mesos_oros_nomeli.csv",
)
FEATURES = ["Υγρό", "Ύψος (cm)", "Σχήμα", "Μάζα (g)"]
TARGET = "Μέσος χρόνος (s)"
SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20


def split_rows(df):
    """Return disjoint positional row IDs, shared by every model."""
    rows = np.arange(len(df))
    train_val, test = train_test_split(
        rows, test_size=TEST_SIZE, shuffle=True, random_state=SEED
    )
    train, validation = train_test_split(
        train_val,
        test_size=VALIDATION_SIZE / (1 - TEST_SIZE),
        shuffle=True,
        random_state=SEED,
    )
    assert len(set(train) | set(validation) | set(test)) == len(df)
    assert not (set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test))
    return {"train": train, "validation": validation, "test": test}


def categorical_preprocessor():
    return ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), ["Υγρό", "Σχήμα"])],
        remainder="passthrough",
    )


def make_models():
    """Create fresh estimators so preprocessing is fitted on train rows only."""
    return {
        "Linear Regression": Pipeline([
            ("preprocessor", categorical_preprocessor()),
            ("regressor", LinearRegression()),
        ]),
        "Polynomial Ridge": Pipeline([
            ("preprocessor", categorical_preprocessor()),
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler(with_mean=False)),
            ("regressor", Ridge(alpha=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", categorical_preprocessor()),
            ("regressor", RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", categorical_preprocessor()),
            ("regressor", XGBRegressor(random_state=SEED)),
        ]),
        "Neural Network": Pipeline([
            ("preprocessor", ColumnTransformer([
                ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Υγρό", "Σχήμα"]),
                ("numerical", StandardScaler(), ["Ύψος (cm)", "Μάζα (g)"]),
            ])),
            ("neural_network", MLPRegressor(
                hidden_layer_sizes=(2,), activation="relu", solver="adam",
                learning_rate_init=0.001, max_iter=5000, random_state=SEED,
            )),
        ]),
    }


def metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)),
    }


def evaluate_model(name, df, rows):
    model = make_models()[name]
    train = rows["train"]
    x_train = df.iloc[train][FEATURES]
    y_train = df.iloc[train][TARGET]
    started = perf_counter()
    model.fit(x_train, y_train)
    fit_time = perf_counter() - started
    scores = {}
    predictions = {}
    for split in ("validation", "test"):
        subset = df.iloc[rows[split]]
        predicted = model.predict(subset[FEATURES])
        scores[split] = metrics(subset[TARGET], predicted)
        predictions[split] = predicted
    return model, scores, predictions, fit_time


def run_standalone(name):
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=f"Evaluate {name} on the shared split")
    parser.add_argument("--dataset", type=Path, default=DATASETS[0])
    args = parser.parse_args()
    df = pd.read_csv(args.dataset, encoding="utf-8-sig")
    rows = split_rows(df)
    _, scores, _, _ = evaluate_model(name, df, rows)
    print(f"{name} | {args.dataset.name} | train={len(rows['train'])}, validation={len(rows['validation'])}, test={len(rows['test'])}")
    for split, values in scores.items():
        print(f"{split}: MAE={values['mae']:.4f} s, MSE={values['mse']:.4f}, RMSE={values['rmse']:.4f} s, R²={values['r2']:.4f}")
