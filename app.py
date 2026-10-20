import hashlib

import pandas as pd
import streamlit as st

from app_logic import (
    DATASET_OPTIONS,
    HEIGHT,
    LIQUID,
    MASS,
    MODEL_NAMES,
    SHAPE,
    predict_time,
    prediction_points,
    train_experiment,
)


st.set_page_config(page_title="Πρόβλεψη Χρόνου Κίνησης", page_icon="🔬", layout="centered")
st.title("🔬 Πρόβλεψη Χρόνου Κίνησης")
st.write(
    "Επίλεξε δεδομένα και μοντέλο, εκπαίδευσέ το και δοκίμασε διαφορετικά "
    "ύψη και μάζες, με ή χωρίς παρεμβολή."
)

st.subheader("1. Δεδομένα και εκπαίδευση")
dataset_col, model_col = st.columns(2)
with dataset_col:
    dataset_name = st.selectbox("Dataset", list(DATASET_OPTIONS), key="dataset")
with model_col:
    model_name = st.selectbox(
        "Μοντέλο", MODEL_NAMES, index=MODEL_NAMES.index("XGBoost"), key="model"
    )

try:
    df = pd.read_csv(DATASET_OPTIONS[dataset_name], encoding="utf-8-sig")
except (OSError, ValueError) as exc:
    st.error(f"Δεν ήταν δυνατή η ανάγνωση του dataset: {exc}")
    st.stop()

signature = (dataset_name, model_name, hashlib.sha256(df.to_csv(index=False).encode()).hexdigest())
liquids = [
    name for name in ["Νερό", "Αλατόνερο", "Λάδι", "Μέλι30", "Μέλι70", "Μέλι100"]
    if name in set(df[LIQUID])
]
st.caption(
    f"{len(df)} εγγραφές · {len(liquids)} υγρά. "
    "Η επιλογή dataset αφορά το Μέλι100· τα Μέλι30 και Μέλι70 περιλαμβάνονται και στα δύο."
)
st.caption(
    "Κάθε μοντέλο εκπαιδεύεται στο ίδιο 80% των εγγραφών. "
    "Το υπόλοιπο 20% κρατείται για έλεγχο και δεν χρησιμοποιείται στην εκπαίδευση."
)

if st.button("Εκπαίδευση μοντέλου", type="primary", key="train"):
    try:
        with st.spinner(f"Εκπαίδευση {model_name}…"):
            result = train_experiment(df, model_name)
        st.session_state["experiment"] = {"signature": signature, "result": result}
    except Exception as exc:
        st.session_state.pop("experiment", None)
        st.error(f"Η εκπαίδευση δεν ολοκληρώθηκε: {exc}")

experiment = st.session_state.get("experiment")
ready = experiment is not None and experiment["signature"] == signature
if ready:
    result = experiment["result"]
    st.success(
        f"Εκπαιδευμένο μοντέλο: {model_name} · {dataset_name} · "
        f"{len(result.train_rows)} εγγραφές εκπαίδευσης / {len(result.test_rows)} ελέγχου."
    )
    st.caption(f"Χρόνος εκπαίδευσης: {result.fit_seconds:.2f} s. Μετρικές στο 20% ελέγχου:")
    mae_col, rmse_col, r2_col = st.columns(3)
    mae_col.metric("MAE (s)", f"{result.scores['mae']:.4f}")
    rmse_col.metric("RMSE (s)", f"{result.scores['rmse']:.4f}")
    r2_col.metric("R²", f"{result.scores['r2']:.4f}")
    st.caption("Οι μετρικές αφορούν τις απευθείας προβλέψεις του μοντέλου, χωρίς παρεμβολή.")
elif experiment is not None:
    st.info("Άλλαξες dataset ή μοντέλο. Πάτησε «Εκπαίδευση μοντέλου» για τη νέα επιλογή.")
else:
    st.info("Πάτησε «Εκπαίδευση μοντέλου» για να ενεργοποιήσεις τις προβλέψεις.")

with st.expander("Προβολή δεδομένων"):
    st.dataframe(df, hide_index=True, width="stretch")

st.divider()
st.subheader("2. Χαρακτηριστικά και πρόβλεψη")
liquid_col, shape_col = st.columns(2)
with liquid_col:
    liquid = st.selectbox("Υγρό", liquids, key="liquid")
with shape_col:
    shape = st.selectbox("Σχήμα", ["Κύβος", "Σφαίρα", "Δίσκος"], key="shape")

shape_data = df.loc[df[SHAPE] == shape]
height_col, mass_col = st.columns(2)
with height_col:
    height = st.number_input(
        "Ύψος (cm)", min_value=float(shape_data[HEIGHT].min()),
        max_value=float(shape_data[HEIGHT].max()), value=15.0, step=0.5,
        format="%.2f", key="height",
    )
with mass_col:
    mass = st.number_input(
        "Μάζα (g)", min_value=float(shape_data[MASS].min()),
        max_value=float(shape_data[MASS].max()), value=float(shape_data[MASS].min()),
        step=0.5, format="%.2f", key=f"mass_{shape}",
    )

grid = df.loc[(df[LIQUID] == liquid) & (df[SHAPE] == shape)]
measured_heights = ", ".join(f"{value:g}" for value in sorted(grid[HEIGHT].unique()))
measured_masses = ", ".join(f"{value:g}" for value in sorted(grid[MASS].unique()))
st.caption(f"Μετρημένα ύψη: {measured_heights} cm · Μετρημένες μάζες: {measured_masses} g.")
st.caption("Μπορείς να πληκτρολογήσεις και ενδιάμεσες τιμές ύψους και μάζας.")

mode = st.radio(
    "Τρόπος πρόβλεψης", ["Χωρίς παρεμβολή", "Με παρεμβολή"],
    horizontal=True, key="prediction_mode",
)
interpolate = mode == "Με παρεμβολή"
if interpolate:
    st.caption(
        "Γραμμική παρεμβολή στο ύψος και στη μάζα, ανάμεσα στις προβλέψεις "
        "του μοντέλου στα γειτονικά μετρημένα σημεία."
    )
else:
    st.caption("Το μοντέλο προβλέπει απευθείας για το ύψος και τη μάζα που επέλεξες.")

valid_inputs = True
try:
    prediction_points(df, liquid, height, shape, mass, interpolate)
except ValueError as exc:
    valid_inputs = False
    st.warning(str(exc))

if liquid == "Μέλι100" and shape == "Κύβος":
    st.caption(
        "Στο Μέλι100 υπάρχουν μετρήσεις κύβου μόνο στα 77 g. "
        "Η παρεμβολή υποστηρίζεται μόνο σε αυτή τη μάζα."
    )

if st.button("Υπολογισμός πρόβλεψης", disabled=not (ready and valid_inputs), key="predict"):
    try:
        prediction, anchors = predict_time(
            result.model, df, liquid, height, shape, mass, interpolate
        )
        st.metric("Εκτιμώμενος χρόνος", f"{prediction:.5f} s")
        if prediction < 0:
            st.warning("Το μοντέλο επέστρεψε αρνητικό χρόνο, που δεν έχει φυσική ερμηνεία.")
        if interpolate and len(anchors) > 1:
            st.info("Η πρόβλεψη προέκυψε με παρεμβολή στις προβλέψεις του μοντέλου.")
            with st.expander("Σημεία και συντελεστές παρεμβολής", expanded=True):
                st.dataframe(anchors, hide_index=True, width="stretch")
        elif interpolate:
            st.info("Το σημείο είναι μετρημένος συνδυασμός· χρησιμοποιήθηκε απευθείας το μοντέλο.")
        else:
            st.info("Χρησιμοποιήθηκε η απευθείας πρόβλεψη του μοντέλου, χωρίς παρεμβολή.")
    except Exception as exc:
        st.error(f"Δεν ήταν δυνατή η πρόβλεψη: {exc}")

st.divider()
with st.expander("Πώς να ερμηνεύσεις τις δοκιμές"):
    st.write(
        "Οι ενδιάμεσες μάζες και τα ενδιάμεσα ύψη είναι εκτιμήσεις, όχι νέες πειραματικές "
        "μετρήσεις. Η παρεμβολή χρησιμοποιεί τις προβλέψεις του επιλεγμένου μοντέλου, "
        "όχι τους καταγεγραμμένους χρόνους. Οι επιλογές περιορίζονται στο εύρος ύψους "
        "και μάζας που υπάρχει για κάθε σχήμα."
    )
    st.write(
        "Οι δοκιμές μοντέλων εδώ είναι διερευνητικές. Για επιλογή μοντέλου με "
        "διασταυρούμενη επικύρωση χρησιμοποίησε την ξεχωριστή ανάλυση του έργου."
    )
st.caption("Η εκπαίδευση ισχύει για την τρέχουσα συνεδρία της εφαρμογής.")