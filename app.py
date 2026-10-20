import hashlib

import pandas as pd
import streamlit as st

from app_i18n import display_label, display_table, translate
from app_logic import (
    AppValidationError,
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


language = st.session_state.get("language", "el")
st.set_page_config(
    page_title=translate(language, "title"), page_icon="🔬", layout="centered"
)
language = st.selectbox(
    "Γλώσσα / Language", ["el", "en"],
    format_func={"el": "Ελληνικά", "en": "English"}.get, key="language",
)


def t(key, **values):
    return translate(language, key, **values)


def label(value):
    return display_label(value, language)


def error_message(error):
    return error.localized_message(language) if isinstance(error, AppValidationError) else str(error)


st.title(f"🔬 {t('title')}")
st.write(t("intro"))

st.subheader(t("training_heading"))
dataset_col, model_col = st.columns(2)
with dataset_col:
    dataset_name = st.selectbox(
        "Dataset", list(DATASET_OPTIONS), format_func=label, key="dataset"
    )
with model_col:
    model_name = st.selectbox(
        t("model"), MODEL_NAMES, index=MODEL_NAMES.index("XGBoost"), key="model"
    )

try:
    df = pd.read_csv(DATASET_OPTIONS[dataset_name], encoding="utf-8-sig")
except (OSError, ValueError) as exc:
    st.error(t("read_error", error=error_message(exc)))
    st.stop()

signature = (dataset_name, model_name, hashlib.sha256(df.to_csv(index=False).encode()).hexdigest())
liquids = [
    name for name in ["Νερό", "Αλατόνερο", "Λάδι", "Μέλι30", "Μέλι70", "Μέλι100"]
    if name in set(df[LIQUID])
]
st.caption(t("dataset_summary", rows=len(df), liquids=len(liquids)))
st.caption(t("split_help"))

if st.button(t("train"), type="primary", key="train"):
    try:
        with st.spinner(t("training", model=model_name)):
            result = train_experiment(df, model_name)
        st.session_state["experiment"] = {"signature": signature, "result": result}
    except Exception as exc:
        st.session_state.pop("experiment", None)
        st.error(t("training_error", error=error_message(exc)))

experiment = st.session_state.get("experiment")
ready = experiment is not None and experiment["signature"] == signature
if ready:
    result = experiment["result"]
    st.success(t(
        "trained", model=model_name, dataset=label(dataset_name),
        train=len(result.train_rows), test=len(result.test_rows),
    ))
    st.caption(t("fit_time", seconds=result.fit_seconds))
    mae_col, rmse_col, r2_col = st.columns(3)
    mae_col.metric("MAE (s)", f"{result.scores['mae']:.4f}")
    rmse_col.metric("RMSE (s)", f"{result.scores['rmse']:.4f}")
    r2_col.metric("R²", f"{result.scores['r2']:.4f}")
    st.caption(t("metrics_help"))
elif experiment is not None:
    st.info(t("stale"))
else:
    st.info(t("not_ready"))

with st.expander(t("view_data")):
    st.dataframe(display_table(df, language), hide_index=True, width="stretch")

st.divider()
st.subheader(t("prediction_heading"))
liquid_col, shape_col = st.columns(2)
with liquid_col:
    liquid = st.selectbox(label(LIQUID), liquids, format_func=label, key="liquid")
with shape_col:
    shape = st.selectbox(
        label(SHAPE), ["Κύβος", "Σφαίρα", "Δίσκος"], format_func=label, key="shape"
    )

shape_data = df.loc[df[SHAPE] == shape]
height_col, mass_col = st.columns(2)
with height_col:
    height = st.number_input(
        label(HEIGHT), min_value=float(shape_data[HEIGHT].min()),
        max_value=float(shape_data[HEIGHT].max()), value=15.0, step=0.5,
        format="%.2f", key="height",
    )
with mass_col:
    mass = st.number_input(
        label(MASS), min_value=float(shape_data[MASS].min()),
        max_value=float(shape_data[MASS].max()), value=float(shape_data[MASS].min()),
        step=0.5, format="%.2f", key=f"mass_{shape}",
    )

grid = df.loc[(df[LIQUID] == liquid) & (df[SHAPE] == shape)]
measured_heights = ", ".join(f"{value:g}" for value in sorted(grid[HEIGHT].unique()))
measured_masses = ", ".join(f"{value:g}" for value in sorted(grid[MASS].unique()))
st.caption(t("measured", heights=measured_heights, masses=measured_masses))
st.caption(t("intermediate_help"))

mode = st.radio(
    t("prediction_mode"), ["Χωρίς παρεμβολή", "Με παρεμβολή"],
    format_func=label, horizontal=True, key="prediction_mode",
)
interpolate = mode == "Με παρεμβολή"
st.caption(t("interpolation_help" if interpolate else "direct_help"))

valid_inputs = True
try:
    prediction_points(df, liquid, height, shape, mass, interpolate)
except ValueError as exc:
    valid_inputs = False
    st.warning(error_message(exc))

if liquid == "Μέλι100" and shape == "Κύβος":
    st.caption(t("honey_cube_help"))

if st.button(t("predict"), disabled=not (ready and valid_inputs), key="predict"):
    try:
        prediction, anchors = predict_time(
            result.model, df, liquid, height, shape, mass, interpolate
        )
        st.metric(t("estimated_time"), f"{prediction:.5f} s")
        if prediction < 0:
            st.warning(t("negative_time"))
        if interpolate and len(anchors) > 1:
            st.info(t("interpolated_result"))
            with st.expander(t("anchors"), expanded=True):
                st.dataframe(display_table(anchors, language), hide_index=True, width="stretch")
        elif interpolate:
            st.info(t("exact_result"))
        else:
            st.info(t("direct_result"))
    except Exception as exc:
        st.error(t("prediction_error", error=error_message(exc)))

st.divider()
with st.expander(t("interpretation")):
    st.write(t("interpretation_help"))
    st.write(t("exploratory_help"))
st.caption(t("session"))