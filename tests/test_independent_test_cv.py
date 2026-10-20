import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

from evaluation import DATASETS, SEED, TARGET, TEST_SIZE
from tmp.run_independent_test_cv import evaluate_dataset


class IndependentTestCVTests(unittest.TestCase):
    def test_outer_test_is_excluded_from_every_fold_and_final_fit_uses_outer_80_percent(self):
        df = pd.read_csv(DATASETS[0], encoding="utf-8-sig")
        with patch("tmp.run_independent_test_cv.make_models", side_effect=lambda: {"Dummy": DummyRegressor()}):
            result = evaluate_dataset(df)

        training = set(result["training_pool_rows"])
        test = set(result["test_rows"])
        self.assertEqual((len(training), len(test)), (168, 42))
        self.assertFalse(training & test)
        self.assertEqual(training | test, set(range(1, len(df) + 1)))
        validation_rows = []
        for fold in result["folds"]:
            fit = set(fold["fit_rows"])
            validation = set(fold["validation_rows"])
            self.assertFalse(fit & validation or fit & test or validation & test)
            self.assertEqual(fit | validation, training)
            validation_rows.extend(validation)
        self.assertEqual(set(validation_rows), training)
        self.assertEqual(len(validation_rows), len(training))
        self.assertEqual(result["models"][0]["final_fit_rows"], len(training))

        expected_prediction = df.iloc[np.array(result["training_pool_rows"]) - 1][TARGET].mean()
        actual_test = df.iloc[np.array(result["test_rows"]) - 1][TARGET]
        expected_mae = np.abs(actual_test - expected_prediction).mean()
        self.assertAlmostEqual(result["models"][0]["test"]["mae"], expected_mae)

    def test_test_targets_cannot_change_cv_ranking(self):
        df = pd.read_csv(DATASETS[0], encoding="utf-8-sig")
        _, test = train_test_split(
            np.arange(len(df)), test_size=TEST_SIZE, shuffle=True, random_state=SEED
        )
        with patch("tmp.run_independent_test_cv.make_models", side_effect=lambda: {"Dummy": DummyRegressor()}):
            original = evaluate_dataset(df)
            changed = df.copy()
            changed.loc[test, TARGET] += 1000
            modified = evaluate_dataset(changed)
        self.assertEqual(original["models"][0]["cv_mean"], modified["models"][0]["cv_mean"])
        self.assertNotEqual(original["models"][0]["test"], modified["models"][0]["test"])


if __name__ == "__main__":
    unittest.main()
