import unittest

import numpy as np
import pandas as pd

from app_logic import (
    DATASET_OPTIONS, HEIGHT, MASS, MODEL_NAMES, predict_time,
    prediction_points, train_experiment,
)
from evaluation import FEATURES, TARGET, split_rows


class QuadraticModel:
    """Nonlinear oracle: direct predictions must differ from interpolation."""

    def predict(self, points):
        return points[HEIGHT].to_numpy() ** 2 + points[MASS].to_numpy() ** 2


class PlaygroundTrainingTests(unittest.TestCase):
    def test_all_models_train_on_both_datasets_with_the_shared_outer_split(self):
        for dataset, path in DATASET_OPTIONS.items():
            df = pd.read_csv(path, encoding="utf-8-sig")
            expected_test = split_rows(df)["test"]
            for name in MODEL_NAMES:
                with self.subTest(dataset=dataset, model=name):
                    result = train_experiment(df, name)
                    np.testing.assert_array_equal(result.test_rows, expected_test)
                    self.assertFalse(set(result.train_rows) & set(result.test_rows))
                    self.assertEqual(len(result.train_rows) + len(result.test_rows), len(df))
                    self.assertEqual(len(result.train_rows), 144 if len(df) == 180 else 168)
                    self.assertTrue(all(np.isfinite(list(result.scores.values()))))
                    self.assertTrue(np.all(np.isfinite(result.model.predict(df[FEATURES]))))

    def test_test_targets_never_affect_the_trained_model(self):
        df = pd.read_csv(DATASET_OPTIONS["Χωρίς Μέλι100"], encoding="utf-8-sig")
        first = train_experiment(df, "XGBoost")
        changed = df.copy()
        changed.loc[first.test_rows, TARGET] += 1000
        second = train_experiment(changed, "XGBoost")
        np.testing.assert_array_equal(
            first.model.predict(df[FEATURES]), second.model.predict(df[FEATURES])
        )
        self.assertNotEqual(first.scores["mae"], second.scores["mae"])


class PlaygroundPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATASET_OPTIONS["Με Μέλι100"], encoding="utf-8-sig")
        cls.model = QuadraticModel()

    def test_direct_mode_passes_intermediate_height_and_mass_unchanged(self):
        value, points = predict_time(self.model, self.df, "Νερό", 17.5, "Κύβος", 39.5)
        self.assertAlmostEqual(value, 17.5**2 + 39.5**2)
        self.assertEqual(len(points), 1)
        self.assertEqual(points.iloc[0][HEIGHT], 17.5)
        self.assertEqual(points.iloc[0][MASS], 39.5)

    def test_bilinear_interpolation_matches_the_four_corner_average(self):
        value, points = predict_time(
            self.model, self.df, "Νερό", 17.5, "Κύβος", 39.5, interpolate=True
        )
        expected = np.mean([h**2 + m**2 for h in (15, 20) for m in (2, 77)])
        self.assertAlmostEqual(value, expected)
        self.assertNotAlmostEqual(value, 17.5**2 + 39.5**2)
        self.assertEqual(len(points), 4)
        np.testing.assert_allclose(points["Συντελεστής παρεμβολής"], 0.25)
        measured = set(self.df[FEATURES].itertuples(index=False, name=None))
        self.assertTrue(set(points[FEATURES].itertuples(index=False, name=None)) <= measured)

    def test_one_axis_and_exact_endpoint_interpolation(self):
        cases = [
            (17.5, 2, (15**2 + 20**2) / 2 + 2**2, 2),
            (15, 39.5, 15**2 + (2**2 + 77**2) / 2, 2),
            (5, 2, 5**2 + 2**2, 1),
            (30, 77, 30**2 + 77**2, 1),
        ]
        for height, mass, expected, anchors in cases:
            with self.subTest(height=height, mass=mass):
                value, points = predict_time(
                    self.model, self.df, "Νερό", height, "Κύβος", mass, True
                )
                self.assertAlmostEqual(value, expected)
                self.assertEqual(len(points), anchors)

    def test_interpolation_does_not_read_the_measured_target_times(self):
        changed = self.df.copy()
        changed[TARGET] = 1e9
        original, _ = predict_time(self.model, self.df, "Νερό", 17.5, "Κύβος", 39.5, True)
        modified, _ = predict_time(self.model, changed, "Νερό", 17.5, "Κύβος", 39.5, True)
        self.assertEqual(original, modified)

    def test_missing_honey_cube_mass_is_never_an_interpolation_anchor(self):
        for interpolate in (False, True):
            with self.subTest(interpolate=interpolate), self.assertRaises(ValueError):
                prediction_points(self.df, "Μέλι100", 15, "Κύβος", 2, interpolate)
        with self.assertRaises(ValueError):
            prediction_points(self.df, "Μέλι100", 17.5, "Κύβος", 39.5, True)
        points, _ = prediction_points(self.df, "Μέλι100", 17.5, "Κύβος", 77, True)
        self.assertEqual(set(points[MASS]), {77})
        direct, _ = prediction_points(self.df, "Μέλι100", 17.5, "Κύβος", 39.5, False)
        self.assertEqual(direct.iloc[0][MASS], 39.5)

    def test_out_of_range_nonfinite_and_unseen_categories_are_rejected(self):
        cases = [
            ("Νερό", 4, "Κύβος", 2), ("Νερό", 31, "Κύβος", 2),
            ("Νερό", 15, "Σφαίρα", 2), ("Νερό", 15, "Δίσκος", 77),
            ("Νερό", np.nan, "Κύβος", 2), ("Νερό", 15, "Κύβος", np.inf),
            ("Άγνωστο", 15, "Κύβος", 2), ("Νερό", 15, "Άγνωστο", 2),
        ]
        for args in cases:
            for interpolate in (False, True):
                with self.subTest(args=args, interpolate=interpolate), self.assertRaises(ValueError):
                    prediction_points(self.df, *args, interpolate=interpolate)


if __name__ == "__main__":
    unittest.main()
