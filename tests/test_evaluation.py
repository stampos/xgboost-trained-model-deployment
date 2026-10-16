import unittest

import numpy as np
import pandas as pd

from evaluation import DATASETS, TARGET, evaluate_model, make_models, split_rows


class SharedEvaluationTests(unittest.TestCase):
    def test_same_disjoint_rows_for_each_dataset(self):
        for path in DATASETS:
            df = pd.read_csv(path, encoding="utf-8-sig")
            first = split_rows(df)
            second = split_rows(df)
            for split in ("train", "validation", "test"):
                np.testing.assert_array_equal(first[split], second[split])
            sets = [set(first[split]) for split in ("train", "validation", "test")]
            self.assertEqual(len(set.union(*sets)), len(df))
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            self.assertEqual(len(make_models()), 5)

    def test_unseen_targets_do_not_affect_training(self):
        df = pd.read_csv(DATASETS[0], encoding="utf-8-sig")
        rows = split_rows(df)
        _, _, original_predictions, _ = evaluate_model("XGBoost", df, rows)
        changed = df.copy()
        changed.loc[np.concatenate((rows["validation"], rows["test"])), TARGET] += 1000
        _, _, changed_predictions, _ = evaluate_model("XGBoost", changed, rows)
        for split in ("validation", "test"):
            np.testing.assert_array_equal(original_predictions[split], changed_predictions[split])


if __name__ == "__main__":
    unittest.main()
