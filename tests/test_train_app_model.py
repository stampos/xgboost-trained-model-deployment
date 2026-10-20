import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluation import DATASETS, SEED, TEST_SIZE
from tmp.train_app_model import train_model


class AppModelTests(unittest.TestCase):
    def test_app_model_uses_only_no_meli100_outer_training_rows(self):
        df = pd.read_csv(DATASETS[1], encoding="utf-8-sig")
        model, train, test, scores = train_model(df)
        expected_train, expected_test = train_test_split(
            np.arange(len(df)), test_size=TEST_SIZE, shuffle=True, random_state=SEED
        )
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(test, expected_test)
        self.assertEqual((len(train), len(test)), (144, 36))
        liquids = set(model.named_steps["preprocessor"].named_transformers_["categorical"].categories_[0])
        self.assertEqual(liquids, set(df["Υγρό"]))
        self.assertNotIn("Μέλι100", liquids)
        self.assertAlmostEqual(scores["mae"], 0.05626662185801426)

        inputs = pd.DataFrame({
            "Υγρό": ["Νερό", "Νερό"],
            "Ύψος (cm)": [15, 15],
            "Σχήμα": ["Κύβος", "Κύβος"],
            "Μάζα (g)": [2, 77],
        })
        predictions = model.predict(inputs)
        self.assertNotAlmostEqual(float(predictions[0]), float(predictions[1]))


if __name__ == "__main__":
    unittest.main()
