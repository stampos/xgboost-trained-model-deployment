import json
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "tmp" / "ml_comparison_results.json"
DATASETS = [
    ROOT / "data" / "mesos_oros.csv",
    ROOT / "data" / "mesos_oros_nomeli.csv",
]

FEATURES = ["Υγρό", "Ύψος (cm)", "Σχήμα", "Μάζα (g)"]
CATEGORICAL = ["Υγρό", "Σχήμα"]
NUMERICAL = ["Ύψος (cm)", "Μάζα (g)"]
TARGET = "Μέσος χρόνος (s)"


def base_preprocessor(dense=False):
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=not dense),
                CATEGORICAL,
            )
        ],
        remainder="passthrough",
    )


def make_models():
    return {
        "Linear Regression": Pipeline(
            [
                ("preprocessor", base_preprocessor()),
                ("regressor", LinearRegression()),
            ]
        ),
        "Polynomial Ridge": Pipeline(
            [
                ("preprocessor", base_preprocessor()),
                ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler(with_mean=False)),
                ("regressor", Ridge(alpha=1.0)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", base_preprocessor()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=200, random_state=42, n_jobs=-1
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("preprocessor", base_preprocessor()),
                (
                    "regressor",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Neural Network": Pipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer(
                        transformers=[
                            (
                                "categorical",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                                CATEGORICAL,
                            ),
                            ("numerical", StandardScaler(), NUMERICAL),
                        ]
                    ),
                ),
                (
                    "neural_network",
                    MLPRegressor(
                        hidden_layer_sizes=(2,),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=0.001,
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


MODEL_CONFIGS = [
    {
        "algorithm": "Linear Regression",
        "source_file": "algorithms/linear_regression.py",
        "parameters": "LinearRegression defaults",
    },
    {
        "algorithm": "Polynomial Ridge",
        "source_file": "algorithms/polynomial_ridge.py",
        "parameters": "degree=2; include_bias=False; StandardScaler(with_mean=False); alpha=1.0",
    },
    {
        "algorithm": "Random Forest",
        "source_file": "algorithms/random_forest.py",
        "parameters": "n_estimators=200; random_state=42; n_jobs=-1",
    },
    {
        "algorithm": "XGBoost",
        "source_file": "algorithms/xgboost_regression.py",
        "parameters": "objective=reg:squarederror; n_estimators=300; learning_rate=0.05; max_depth=4; subsample=0.9; colsample_bytree=0.9; random_state=42; n_jobs=-1",
    },
    {
        "algorithm": "Neural Network",
        "source_file": "algorithms/NN.py",
        "parameters": "hidden_layer_sizes=(2,); activation=relu; solver=adam; learning_rate_init=0.001; max_iter=5000; random_state=42",
    },
    {
        "algorithm": "XGBoost MAE per liquid",
        "source_file": "algorithms/xgboost_MAE_per_liquid.py",
        "parameters": "Same XGBoost model; out-of-fold predictions and MAE grouped by liquid",
    },
]


def safe_float(value):
    value = float(value)
    return None if not math.isfinite(value) else value


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cv": {
            "method": "KFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
        },
        "features": FEATURES,
        "categorical_features": CATEGORICAL,
        "target": TARGET,
        "model_configs": MODEL_CONFIGS,
        "datasets": [],
        "fold_results": [],
        "summary_results": [],
        "xgboost_liquid_summary": [],
        "xgboost_oof_predictions": [],
        "warnings": [],
    }

    for dataset_path in DATASETS:
        dataset_name = dataset_path.name
        df = pd.read_csv(dataset_path)
        X = df[FEATURES]
        y = df[TARGET]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        result["datasets"].append(
            {
                "dataset": dataset_name,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "missing_values": int(df.isna().sum().sum()),
                "liquids": int(df["Υγρό"].nunique()),
                "shapes": int(df["Σχήμα"].nunique()),
                "target_mean": safe_float(y.mean()),
                "target_std": safe_float(y.std(ddof=1)),
                "target_min": safe_float(y.min()),
                "target_max": safe_float(y.max()),
                "liquid_counts": {
                    str(k): int(v) for k, v in df["Υγρό"].value_counts().items()
                },
            }
        )

        for algorithm, model in make_models().items():
            print(f"Running {algorithm} on {dataset_name}...", flush=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                started = time.perf_counter()
                scores = cross_validate(
                    model,
                    X,
                    y,
                    cv=kf,
                    scoring={
                        "MAE": "neg_mean_absolute_error",
                        "MSE": "neg_mean_squared_error",
                        "R2": "r2",
                    },
                    error_score="raise",
                )
                elapsed = time.perf_counter() - started

            for warning_item in caught:
                result["warnings"].append(
                    {
                        "dataset": dataset_name,
                        "algorithm": algorithm,
                        "category": warning_item.category.__name__,
                        "message": str(warning_item.message),
                    }
                )

            mae = -scores["test_MAE"]
            mse = -scores["test_MSE"]
            rmse = np.sqrt(mse)
            r2 = scores["test_R2"]

            for fold_index in range(5):
                result["fold_results"].append(
                    {
                        "dataset": dataset_name,
                        "algorithm": algorithm,
                        "fold": fold_index + 1,
                        "mae": safe_float(mae[fold_index]),
                        "mse": safe_float(mse[fold_index]),
                        "rmse": safe_float(rmse[fold_index]),
                        "r2": safe_float(r2[fold_index]),
                        "fit_time_s": safe_float(scores["fit_time"][fold_index]),
                        "score_time_s": safe_float(scores["score_time"][fold_index]),
                    }
                )

            result["summary_results"].append(
                {
                    "dataset": dataset_name,
                    "algorithm": algorithm,
                    "mae_mean": safe_float(mae.mean()),
                    "mae_std": safe_float(mae.std()),
                    "mse_mean": safe_float(mse.mean()),
                    "mse_std": safe_float(mse.std()),
                    "rmse_mean": safe_float(rmse.mean()),
                    "rmse_std": safe_float(rmse.std()),
                    "r2_mean": safe_float(r2.mean()),
                    "r2_std": safe_float(r2.std()),
                    "fit_time_mean_s": safe_float(scores["fit_time"].mean()),
                    "score_time_mean_s": safe_float(scores["score_time"].mean()),
                    "total_elapsed_s": safe_float(elapsed),
                }
            )

        print(f"Running XGBoost out-of-fold analysis on {dataset_name}...", flush=True)
        xgb_model = make_models()["XGBoost"]
        y_pred = cross_val_predict(xgb_model, X, y, cv=kf)
        fold_ids = np.zeros(len(df), dtype=int)
        for fold_number, (_, test_indices) in enumerate(kf.split(X), start=1):
            fold_ids[test_indices] = fold_number

        pred_df = df.copy()
        pred_df["Fold"] = fold_ids
        pred_df["Πρόβλεψη"] = y_pred
        pred_df["Σφάλμα"] = pred_df[TARGET] - pred_df["Πρόβλεψη"]
        pred_df["Απόλυτο σφάλμα"] = pred_df["Σφάλμα"].abs()

        overall_mae = mean_absolute_error(y, y_pred)
        liquid_summary = (
            pred_df.groupby("Υγρό", as_index=False)
            .agg(
                count=(TARGET, "count"),
                actual_mean=(TARGET, "mean"),
                prediction_mean=("Πρόβλεψη", "mean"),
                mae=("Απόλυτο σφάλμα", "mean"),
                error_std=("Απόλυτο σφάλμα", "std"),
            )
            .sort_values("mae")
        )
        for _, row in liquid_summary.iterrows():
            result["xgboost_liquid_summary"].append(
                {
                    "dataset": dataset_name,
                    "liquid": str(row["Υγρό"]),
                    "count": int(row["count"]),
                    "actual_mean": safe_float(row["actual_mean"]),
                    "prediction_mean": safe_float(row["prediction_mean"]),
                    "mae": safe_float(row["mae"]),
                    "error_std": safe_float(row["error_std"]),
                    "overall_mae": safe_float(overall_mae),
                }
            )

        for row_index, row in pred_df.iterrows():
            result["xgboost_oof_predictions"].append(
                {
                    "dataset": dataset_name,
                    "row": int(row_index) + 1,
                    "fold": int(row["Fold"]),
                    "liquid": str(row["Υγρό"]),
                    "height_cm": safe_float(row["Ύψος (cm)"]),
                    "shape": str(row["Σχήμα"]),
                    "mass_g": safe_float(row["Μάζα (g)"]),
                    "actual_s": safe_float(row[TARGET]),
                    "prediction_s": safe_float(row["Πρόβλεψη"]),
                    "error_s": safe_float(row["Σφάλμα"]),
                    "absolute_error_s": safe_float(row["Απόλυτο σφάλμα"]),
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
