"""Training and prediction rules for the interactive Streamlit experiment."""

from dataclasses import dataclass
from itertools import product
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluation import DATASETS, FEATURES, SEED, TARGET, TEST_SIZE, make_models, metrics
from app_i18n import translate


DATASET_OPTIONS = {
    "Χωρίς Μέλι100": DATASETS[1],
    "Με Μέλι100": DATASETS[0],
}
LIQUID, HEIGHT, SHAPE, MASS = FEATURES
MODEL_NAMES = tuple(make_models())


class AppValidationError(ValueError):
    """Keep validation messages translatable without changing model inputs."""

    def __init__(self, message_key, **values):
        self.message_key = message_key
        self.values = values
        super().__init__(self.localized_message("el"))

    def localized_message(self, language):
        values = self.values.copy()
        if "axis" in values:
            values["axis"] = translate(language, values["axis"])
        return translate(language, self.message_key, **values)


@dataclass
class TrainingResult:
    model: object
    train_rows: np.ndarray
    test_rows: np.ndarray
    scores: dict
    fit_seconds: float


def train_experiment(df, model_name):
    """Fit only on the outer 80%, using the same split for every model."""
    models = make_models()
    if model_name not in models:
        raise AppValidationError("invalid_model")
    train, test = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, shuffle=True, random_state=SEED
    )
    model = models[model_name]
    started = perf_counter()
    model.fit(df.iloc[train][FEATURES], df.iloc[train][TARGET])
    elapsed = perf_counter() - started
    held_out = df.iloc[test]
    scores = metrics(held_out[TARGET], model.predict(held_out[FEATURES]))
    return TrainingResult(model, train, test, scores, elapsed)


def _axis_points(values, value, label):
    """Return one measured coordinate, or two bounding coordinates and weights."""
    values = np.sort(np.unique(np.asarray(values, dtype=float)))
    exact = values[np.isclose(values, value, rtol=0, atol=1e-9)]
    if len(exact):
        return [(float(exact[0]), 1.0)]
    if value < values[0] or value > values[-1]:
        raise AppValidationError("interpolation_bounds", axis=label, low=values[0], high=values[-1])
    upper = int(np.searchsorted(values, value))
    low, high = values[upper - 1], values[upper]
    weight = float((value - low) / (high - low))
    return [(float(low), 1 - weight), (float(high), weight)]


def prediction_points(df, liquid, height, shape, mass, interpolate=False):
    """Build model inputs; interpolation uses only existing liquid/shape anchors."""
    height, mass = float(height), float(mass)
    if not np.isfinite(height) or not np.isfinite(mass):
        raise AppValidationError("nonfinite_inputs")
    grid = df.loc[(df[LIQUID] == liquid) & (df[SHAPE] == shape)]
    if grid.empty:
        raise AppValidationError("unknown_combination")
    shape_data = df.loc[df[SHAPE] == shape]
    if not shape_data[HEIGHT].min() <= height <= shape_data[HEIGHT].max():
        raise AppValidationError("height_bounds")
    if not shape_data[MASS].min() <= mass <= shape_data[MASS].max():
        raise AppValidationError("mass_bounds")
    if liquid == "Μέλι100" and shape == "Κύβος" and np.isclose(mass, 2, rtol=0, atol=1e-9):
        raise AppValidationError("missing_honey_cube")

    if not interpolate:
        return pd.DataFrame([[liquid, height, shape, mass]], columns=FEATURES), np.array([1.0])

    masses = _axis_points(grid[MASS], mass, "mass_axis")
    # Use heights shared by the mass anchors, so every corner actually exists.
    height_sets = [set(grid.loc[grid[MASS] == value, HEIGHT]) for value, _ in masses]
    shared_heights = set.intersection(*height_sets)
    if not shared_heights:
        raise AppValidationError("no_shared_heights")
    heights = _axis_points(list(shared_heights), height, "height_axis")
    rows, weights = [], []
    for (h, h_weight), (m, m_weight) in product(heights, masses):
        rows.append([liquid, h, shape, m])
        weights.append(h_weight * m_weight)
    return pd.DataFrame(rows, columns=FEATURES), np.asarray(weights)


def predict_time(model, df, liquid, height, shape, mass, interpolate=False):
    """Interpolate model predictions, never the observed target values."""
    points, weights = prediction_points(df, liquid, height, shape, mass, interpolate)
    predictions = np.asarray(model.predict(points), dtype=float)
    if not np.all(np.isfinite(predictions)):
        raise AppValidationError("invalid_time")
    prediction = float(np.dot(predictions, weights))
    anchors = points.copy()
    anchors["Πρόβλεψη μοντέλου (s)"] = predictions
    anchors["Συντελεστής παρεμβολής"] = weights
    return prediction, anchors
