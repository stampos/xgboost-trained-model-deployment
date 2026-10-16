"""Report XGBoost MAE per liquid on the shared 80/20 held-out test set."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import DATASETS, FEATURES, SEED, TARGET, make_models  # noqa: E402


def main():
    data = pd.read_csv(DATASETS[0], encoding="utf-8-sig")
    train, test = train_test_split(
        np.arange(len(data)), test_size=0.20, shuffle=True, random_state=SEED
    )
    model = make_models()["XGBoost"]
    model.fit(data.iloc[train][FEATURES], data.iloc[train][TARGET])

    held_out = data.iloc[test].copy()
    held_out["absolute_error"] = np.abs(
        held_out[TARGET].to_numpy() - model.predict(held_out[FEATURES])
    )
    per_liquid = [
        {
            "liquid": liquid,
            "count": int(len(group)),
            "mean_actual_seconds": float(group[TARGET].mean()),
            "mae_seconds": float(group["absolute_error"].mean()),
        }
        for liquid, group in held_out.groupby(FEATURES[0], sort=True)
    ]
    result = {
        "dataset": DATASETS[0].name,
        "algorithm": "XGBoost",
        "seed": SEED,
        "train_count": int(len(train)),
        "test_count": int(len(test)),
        "overall_mae_seconds": float(held_out["absolute_error"].mean()),
        "per_liquid": per_liquid,
    }
    output = ROOT / "tmp" / "mae_per_liquid_80_20.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
