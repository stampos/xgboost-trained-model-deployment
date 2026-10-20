# Interactive motion-time prediction

Run `python -m streamlit run app.py` from the repository root. On Windows with
the existing virtual environment, use `.venv\Scripts\python.exe -m streamlit run app.py`.

The app lets users:

- Switch between Greek and English, including category names, tables, and
  validation messages, while keeping their selections and trained model.
- Select a dataset with Meli100 (210 rows) or without it (180 rows). Meli30 and
  Meli70 remain available in both datasets.
- Select Linear Regression, Polynomial Ridge, Random Forest, XGBoost, or Neural
  Network, then explicitly train the selected model.
- Enter intermediate heights and masses within the measured range for a shape.
- Choose direct model prediction or interpolation in height and mass. When both
  vary, interpolation uses the four neighboring model predictions, not measured
  target times.

Every model uses the same seeded outer 80/20 split: 168/42 training/test rows
with Meli100, or 144/36 without it. Test rows never enter fitting. The app
displays direct-prediction test MAE, RMSE, and R2; it does not select a winner
from test scores. Changing dataset or model requires a matching training run.
Models stay in the current session and do not require saved model artifacts.

Meli100 has no Cube/2 g measurements, so that combination is unavailable. Cube
interpolation in Meli100 supports only 77 g. Intermediate predictions are model
estimates rather than additional experimental measurements.

## Streamlit Community Cloud

Use repository `stampos/xgboost-trained-model-deployment`, branch `main`, and
entrypoint `app.py`. Use Python 3.11 or 3.12; the pinned dependencies require
Python 3.11 or newer. Local verification uses Python 3.11. Install dependencies
with `python -m pip install -r requirements.txt`.

Community Cloud automatically updates a connected app when its branch changes;
see [Manage your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).
The app trains from the CSVs in `data/`, rather than loading the older
`xgboost_model.joblib` artifact.

Run checks with `python -m unittest discover -s tests -v`.

## Separate offline model evaluation

For the independent-test protocol, run `python tmp/run_independent_test_cv.py`.
It first reserves 20% of each dataset as an untouched test set. Five-fold
cross-validation then runs only on the remaining 80%; each fold fits on four
parts and validates on the fifth. Models are ranked by mean CV validation MAE.
After CV, every algorithm is fitted again on all of the outer 80% and evaluated
once on the independent test set. Fold scores, their means and population
standard deviations, test scores, row assignments, and the selected model are
written to `outputs/ml-comparison-independent-cv/results.json`. The test set
does not participate in model selection.

The optional `python tmp/train_app_model.py` command still writes the legacy
`xgboost_nomeli100_model.joblib` on the 144 no-Meli100 outer-training rows;
the interactive app now trains the selected model instead of loading that file.

The commands below describe older, separate comparisons and reports.

Run `python tmp/run_all_algorithms.py` to evaluate all five regression models on
each dataset under the older 64/16/20 protocol. This also refreshes the legacy
`xgboost_model.joblib`, which the app no longer loads. Then run
`python tmp/build_results_workbook.py` for that protocol's Excel report. All
commands should be run from the project root.

For each dataset, `evaluation.split_rows` shuffles the row positions with seed
42 and first reserves 20% for the independent test set. The remaining 80% is
split 80/20 into train and validation (about 64%/16%/20% overall). Every model
uses exactly the same rows, features, and target within that dataset. Feature
encoding and scaling are fitted on train rows inside each model's pipeline.
The five models are Linear Regression, Polynomial Ridge, Random Forest,
XGBoost, and Neural Network. The per-liquid script analyzes the same XGBoost
model; it is not a sixth model.

Model ranking uses **validation MAE**. Test metrics are reported separately;
test rows never enter fitting or model selection. The JSON results record the
row assignment, both sets of metrics, and the legacy model artifact's source.
The Excel report is written to
`outputs/ml-comparison-shared-split/ml_algorithm_results.xlsx`.

The standalone scripts in `algorithms/` use the same protocol and accept an
optional `--dataset` path. For example:

```powershell
python algorithms/random_forest.py --dataset data/mesos_oros_nomeli.csv
python algorithms/xgboost_MAE_per_liquid.py
```

The older report under `outputs/ml-comparison-20260918/` is a historical
five-fold analysis and does not follow this protocol.

Run `python tmp/run_two_splits.py` to compare nested 64/16/20 and direct
80/20 holdouts for both datasets. It writes
`tmp/two_split_results.json`. In the 80/20 comparison, the held-out 20% is
the same test rows as in the 64/16/20 comparison; its training set includes
the other version's training and validation rows.

Run `python tmp/run_fold_metrics.py` for five-fold cross-validation over each
entire dataset, matching the method in the original Google Sheet tabs. It
writes `tmp/fold_metrics_full.json`, which backs the two updated `5-fold` tabs.
This is a separate analysis from the 64/16/20 and 80/20 holdout results.
