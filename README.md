# Common model evaluation

Run `python tmp/run_all_algorithms.py` to evaluate all five regression models on
each dataset. The script also refreshes `xgboost_model.joblib`, which the
Streamlit app loads, using the XGBoost trained on the full dataset's **train
rows only**. Then run `python tmp/build_results_workbook.py` for the Excel
report. All commands should be run from the project root.

For each dataset, `evaluation.split_rows` shuffles the row positions with seed
42 and assigns 60% to train, 20% to validation, and 20% to test. Every model
uses exactly the same rows, features, and target within that dataset. Feature
encoding and scaling are fitted on train rows inside each model's pipeline.
The five models are Linear Regression, Polynomial Ridge, Random Forest,
XGBoost, and Neural Network. The per-liquid script analyzes the same XGBoost
model; it is not a sixth model.

Model ranking uses **validation MAE**. Test metrics are reported separately;
test rows never enter fitting or model selection. The JSON results record the
row assignment, both sets of metrics, and the deployment model's source.
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

Run `python tmp/run_two_splits.py` to reproduce the two comparisons recorded
in the Google Sheet: 60/20/20 and 80/20 for both datasets. It writes
`tmp/two_split_results.json`. In the 80/20 comparison, the held-out 20% is
the same test rows as in the 60/20/20 comparison; its training set includes
the other version's training and validation rows.

Run `python tmp/run_fold_metrics.py` for five-fold cross-validation over each
entire dataset, matching the method in the original Google Sheet tabs. It
writes `tmp/fold_metrics_full.json`, which backs the two updated `5-fold` tabs.
This is a separate analysis from the 60/20/20 and 80/20 holdout results.
