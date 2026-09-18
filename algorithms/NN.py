import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor


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
# 3. Κατηγορικές και αριθμητικές μεταβλητές
# --------------------------------------------------

categorical_features = [
    "Υγρό",
    "Σχήμα"
]

numerical_features = [
    "Ύψος (cm)",
    "Μάζα (g)"
]


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),

        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)


# --------------------------------------------------
# 5. Νευρωνικό δίκτυο
# --------------------------------------------------

neural_network = MLPRegressor(
    hidden_layer_sizes=(2,),
    activation="relu",
    solver="adam",

    learning_rate_init=0.001,
    max_iter=5000,

    random_state=42
)


# --------------------------------------------------
# 6. Pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("neural_network", neural_network)
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
# 8. Εκτέλεση Cross Validation
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
# 9. Μετατροπή αποτελεσμάτων
# --------------------------------------------------

mae_scores = -scores["test_MAE"]

mse_scores = -scores["test_MSE"]

rmse_scores = mse_scores ** 0.5

r2_scores = scores["test_R2"]


# --------------------------------------------------
# 10. Αποτελέσματα κάθε Fold
# --------------------------------------------------

print("\n----- NEURAL NETWORK: 5-FOLD CROSS VALIDATION -----\n")

for i in range(5):

    print(f"Fold {i + 1}")

    print(f"MAE:  {mae_scores[i]:.4f}")
    print(f"MSE:  {mse_scores[i]:.4f}")
    print(f"RMSE: {rmse_scores[i]:.4f}")
    print(f"R²:   {r2_scores[i]:.4f}")

    print()


# --------------------------------------------------
# 11. Μέσοι όροι
# --------------------------------------------------

print("----- ΜΕΣΟΙ ΟΡΟΙ -----\n")

print(f"Μέσο MAE:  {mae_scores.mean():.4f}")
print(f"Μέσο MSE:  {mse_scores.mean():.4f}")
print(f"Μέσο RMSE: {rmse_scores.mean():.4f}")
print(f"Μέσο R²:   {r2_scores.mean():.4f}")


# --------------------------------------------------
# 12. Τυπικές αποκλίσεις
# --------------------------------------------------

print("\n----- ΤΥΠΙΚΗ ΑΠΟΚΛΙΣΗ -----\n")

print(f"MAE std:  {mae_scores.std():.4f}")
print(f"MSE std:  {mse_scores.std():.4f}")
print(f"RMSE std: {rmse_scores.std():.4f}")
print(f"R² std:   {r2_scores.std():.4f}")