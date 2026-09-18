import pandas as pd
import joblib

model = joblib.load("xgboost_model.joblib")

for height in [5, 10, 15, 17.5, 20, 25, 30]:

    data = pd.DataFrame({
        "Υγρό": ["Μέλι70"],
        "Ύψος (cm)": [height],
        "Σχήμα": ["Σφαίρα"],
        "Μάζα (g)": [59]
    })

    prediction = model.predict(data)[0]

    print(f"{height:4.1f} cm -> {prediction:.6f} s")