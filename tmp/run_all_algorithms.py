"""Run every model on the same train/validation/test rows per dataset."""

import json
import hashlib
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import (  # noqa: E402
    DATASETS, FEATURES, SEED, TARGET, TEST_SIZE, VALIDATION_SIZE,
    evaluate_model, make_models, split_rows,
)

OUTPUT_PATH = ROOT / "tmp" / "ml_comparison_results.json"
MODEL_PATH = ROOT / "xgboost_model.joblib"


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def main():
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "method": "shared_holdout",
            "train_fraction": round((1 - TEST_SIZE) * (1 - VALIDATION_SIZE), 2),
            "validation_fraction": round((1 - TEST_SIZE) * VALIDATION_SIZE, 2),
            "test_fraction": TEST_SIZE,
            "shuffle": True,
            "random_state": SEED,
            "selection_metric": "validation MAE",
            "fit_rows": "train only",
        },
        "features": FEATURES,
        "target": TARGET,
        "datasets": [],
        "split_rows": [],
        "model_results": [],
        "xgboost_liquid_summary": [],
        "xgboost_predictions": [],
        "warnings": [],
        "deployment_model": None,
    }

    for path in DATASETS:
        df = pd.read_csv(path, encoding="utf-8-sig")
        rows = split_rows(df)
        result["datasets"].append({
            "dataset": path.name,
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": int(df.isna().sum().sum()),
            "liquids": int(df["Υγρό"].nunique()),
            "shapes": int(df["Σχήμα"].nunique()),
            "train_rows": len(rows["train"]),
            "validation_rows": len(rows["validation"]),
            "test_rows": len(rows["test"]),
        })
        for split, indices in rows.items():
            result["split_rows"].extend(
                {"dataset": path.name, "row": int(index) + 1, "split": split}
                for index in indices
            )

        for name in make_models():
            print(f"{path.name}: {name}", flush=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model, scores, predictions, fit_time = evaluate_model(name, df, rows)
            result["model_results"].append({
                "dataset": path.name,
                "algorithm": name,
                "train_rows": len(rows["train"]),
                "validation_rows": len(rows["validation"]),
                "test_rows": len(rows["test"]),
                "fit_time_s": finite(fit_time),
                "validation": scores["validation"],
                "test": scores["test"],
            })
            result["warnings"].extend({
                "dataset": path.name,
                "algorithm": name,
                "category": item.category.__name__,
                "message": str(item.message),
            } for item in caught)

            if name == "XGBoost" and path == DATASETS[0]:
                joblib.dump(model, MODEL_PATH)
                result["deployment_model"] = {
                    "path": str(MODEL_PATH.relative_to(ROOT)),
                    "dataset": path.name,
                    "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "train_rows": [int(index) + 1 for index in rows["train"]],
                    "validation": scores["validation"],
                    "test": scores["test"],
                }

            if name != "XGBoost":
                continue
            for split in ("validation", "test"):
                records = df.iloc[rows[split]].copy()
                records["prediction"] = predictions[split]
                records["absolute_error"] = (records[TARGET] - records["prediction"]).abs()
                overall_mae = scores[split]["mae"]
                for liquid, group in records.groupby("Υγρό"):
                    result["xgboost_liquid_summary"].append({
                        "dataset": path.name,
                        "split": split,
                        "liquid": str(liquid),
                        "count": len(group),
                        "actual_mean": finite(group[TARGET].mean()),
                        "prediction_mean": finite(group["prediction"].mean()),
                        "mae": finite(group["absolute_error"].mean()),
                        "overall_mae": overall_mae,
                    })
                for index, record in records.iterrows():
                    result["xgboost_predictions"].append({
                        "dataset": path.name,
                        "split": split,
                        "row": int(index) + 1,
                        "liquid": str(record["Υγρό"]),
                        "height_cm": finite(record["Ύψος (cm)"]),
                        "shape": str(record["Σχήμα"]),
                        "mass_g": finite(record["Μάζα (g)"]),
                        "actual_s": finite(record[TARGET]),
                        "prediction_s": finite(record["prediction"]),
                        "absolute_error_s": finite(record["absolute_error"]),
                    })

    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
