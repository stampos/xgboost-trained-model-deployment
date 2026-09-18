import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


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
# 3. Κατηγορικές μεταβλητές
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
# 5. Δημιουργία Linear Regression Pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# --------------------------------------------------
# 6. Δημιουργία 5-Fold Cross Validation
# --------------------------------------------------

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# 7. Εκτέλεση Cross Validation
# --------------------------------------------------

scores = cross_validate(
    model,
    X,
    y,
    cv=kf,
    scoring={
        "MAE": "neg_mean_absolute_error",
        "MSE": "neg_mean_squared_error",
        "R2": "r2"
    }
)


# --------------------------------------------------
# 8. Μετατροπή αποτελεσμάτων
# --------------------------------------------------

mae_scores = -scores["test_MAE"]

mse_scores = -scores["test_MSE"]

rmse_scores = mse_scores ** 0.5

r2_scores = scores["test_R2"]


# --------------------------------------------------
# 9. Αποτελέσματα κάθε Fold
# --------------------------------------------------

print("\n----- ΑΠΟΤΕΛΕΣΜΑΤΑ 5-FOLD CROSS VALIDATION -----\n")

for i in range(5):

    print(f"Fold {i + 1}")

    print(f"MAE:  {mae_scores[i]:.4f}")
    print(f"MSE:  {mse_scores[i]:.4f}")
    print(f"RMSE: {rmse_scores[i]:.4f}")
    print(f"R²:   {r2_scores[i]:.4f}")

    print()


# --------------------------------------------------
# 10. Μέσοι όροι
# --------------------------------------------------

print("----- ΜΕΣΟΙ ΟΡΟΙ -----\n")

print(f"Μέσο MAE:  {mae_scores.mean():.4f}")
print(f"Μέσο MSE:  {mse_scores.mean():.4f}")
print(f"Μέσο RMSE: {rmse_scores.mean():.4f}")
print(f"Μέσο R²:   {r2_scores.mean():.4f}")


# --------------------------------------------------
# 11. Τυπικές αποκλίσεις
# --------------------------------------------------

print("\n----- ΤΥΠΙΚΗ ΑΠΟΚΛΙΣΗ -----\n")

print(f"MAE std:  {mae_scores.std():.4f}")
print(f"MSE std:  {mse_scores.std():.4f}")
print(f"RMSE std: {rmse_scores.std():.4f}")
print(f"R² std:   {r2_scores.std():.4f}")