import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor


# --------------------------------------------------
# 1. Φόρτωση dataset
# --------------------------------------------------

df = pd.read_csv("data/mesos_oros.csv")
#df = pd.read_csv("data/mesos_oros_nomeli.csv")

# --------------------------------------------------
# 2. Features και Target
# --------------------------------------------------

X = df[
    [
        "Υγρό",
        "Ύψος (cm)",
        "Σχήμα",
        "Μάζα (g)"
    ]
]

y = df["Μέσος χρόνος (s)"]


# --------------------------------------------------
# 3. Κατηγορικά χαρακτηριστικά
# --------------------------------------------------

categorical_features = [
    "Υγρό",
    "Σχήμα"
]


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# --------------------------------------------------
# 5. XGBoost
# --------------------------------------------------

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 6. Pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", xgb_model)
    ]
)


# --------------------------------------------------
# 7. 5-Fold Cross Validation
# --------------------------------------------------

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# 8. Out-of-fold predictions
# --------------------------------------------------

y_pred = cross_val_predict(
    model,
    X,
    y,
    cv=kf
)


# --------------------------------------------------
# 9. Αποθήκευση predictions στο DataFrame
# --------------------------------------------------

results = df.copy()

results["Πρόβλεψη"] = y_pred

results["Απόλυτο σφάλμα"] = abs(
    results["Μέσος χρόνος (s)"] - results["Πρόβλεψη"]
)


# --------------------------------------------------
# 10. Συνολικό MAE
# --------------------------------------------------

overall_mae = mean_absolute_error(
    y,
    y_pred
)

print("\n----- ΣΥΝΟΛΙΚΟ MAE -----\n")

print(f"MAE: {overall_mae:.4f} s")


# --------------------------------------------------
# 11. MAE ανά υγρό
# --------------------------------------------------

mae_per_liquid = (
    results
    .groupby("Υγρό")["Απόλυτο σφάλμα"]
    .mean()
    .sort_values()
)


print("\n----- MAE ΑΝΑ ΥΓΡΟ -----\n")

for liquid, mae in mae_per_liquid.items():

    print(
        f"{liquid}: {mae:.4f} s"
    )


# --------------------------------------------------
# 12. Επιπλέον πληροφορίες ανά υγρό
# --------------------------------------------------

summary = (
    results
    .groupby("Υγρό")
    .agg(
        Πλήθος=("Μέσος χρόνος (s)", "count"),
        Μέσος_πραγματικός_χρόνος=("Μέσος χρόνος (s)", "mean"),
        Μέσο_απόλυτο_σφάλμα=("Απόλυτο σφάλμα", "mean")
    )
)


print("\n----- ΑΝΑΛΥΤΙΚΑ ΑΝΑ ΥΓΡΟ -----\n")

print(summary)