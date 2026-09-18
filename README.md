# USD/LKR Exchange Rate Forecasting

Forecasting monthly USD/LKR exchange rates using rolling-window validation, comparing Random Forest and XGBoost, with MLflow experiment tracking for reproducibility.

Status: notebook complete and reproducible, not deployed as a service.

## Summary

Across 5 expanding walk-forward windows, XGBoost's average R2 was 0.1614 versus Random Forest's -0.2175, meaning Random Forest performed worse than simply predicting the mean on average, while XGBoost was consistently, if modestly, better than that baseline. The best single window (Window 5, trained on the full history through the 2022 crisis) reached R2 = 0.8821, MAE = 2.96 LKR, RMSE = 3.50. That is the strongest evidence that more training history, including the crisis period, materially improves the forecast, not a claim of typical performance across all windows.

## Results

| Metric | Random Forest | XGBoost |
|---|---|---|
| Avg R2 (5 windows) | -0.2175 | 0.1614 |
| Avg MAE | 20.17 | 16.48 |
| Best window (5) R2 | 0.8304 | 0.8821 |
| Improvement, best window | | +37.9% |

The average and the best window are not interchangeable. The average row is the honest expectation for an arbitrary rolling window; the best-window row is what happened in the single most data-rich window, whose test period (2026) was also the calmest, post-crisis stretch in the dataset. Random Forest's negative average R2 indicates it does not extrapolate well outside its training range, a known weakness of tree ensembles that do not boost residuals. That is the actual reason XGBoost was selected, not just its higher headline number.

### Per-window results

| Window | Random Forest R2 | XGBoost R2 |
|---|---|---|
| 1 | -0.7060 | -0.6638 |
| 2 | -0.5999 | 0.2937 |
| 3 | -0.0187 | 0.5011 |
| 4 | -0.5933 | -0.2061 |
| 5 | 0.8304 | 0.8821 |

XGBoost is positive in 3 of 5 windows; Random Forest is positive in only 1 of 5. Both struggle most on Window 1, which has the least training history and no exposure to the 2022 crisis yet. This is the real basis for preferring XGBoost: not just its higher Window 5 score, but its consistency across most of the timeline.

## Data and methodology

- Period: 2010 to 2026, 199 raw monthly observations, 196 usable after feature engineering
- Special event: 2022 Sri Lanka macroeconomic crisis (structural break)
- Features: 7 engineered (3 lagged rates, current volatility and range, lagged volatility)
- Validation: 5 expanding rolling windows, each testing on the following 12 months
- Hyperparameters were chosen by reasoning about dataset size, not by grid or Bayesian search (see the notebook for the full rationale)

## Feature importance

Verified directly against the saved model via model.feature_importances_, matching model_metadata.json.

| Feature | Importance |
|---|---|
| lag_rate_1 (previous month) | 93.7% |
| rate_max | 2.6% |
| lag_rate_3 | 2.1% |
| rate_min | 1.4% |
| lag_rate_2 (2 months ago) | 0.2% |
| lag_vol_1, rate_std | ~0.0% |

The model is overwhelmingly a persistence forecaster, not a multi-feature blend. This is consistent with exchange rates behaving close to a random walk.

## Naive baseline comparison

On the same Window 5 test set, a trivial baseline that predicts "next month's rate equals this month's rate" (using lag_rate_1 directly, with no model at all) was compared against the trained XGBoost model:

| Metric | Naive baseline | XGBoost model | Better |
|---|---|---|---|
| MAE | 2.58 | 2.96 | Naive, about 14.9% lower |
| RMSE | 3.81 | 3.50 | XGBoost, about 8.2% lower |
| R2 | 0.860 | 0.882 | XGBoost, marginal |

The trained model does not uniformly beat a one-line naive forecast. It has higher typical error (MAE) than simply copying last month's rate forward, but lower RMSE, meaning it makes fewer or smaller large misses, since RMSE penalizes big errors more heavily than MAE does. This is consistent with the feature importance result above: with 93.7% of the model's weight on lag_rate_1, XGBoost has effectively learned a regularized, smoothed version of the naive forecast rather than something qualitatively different from it. Whether the added complexity of a trained model is worth trading typical-case accuracy for outlier robustness is an open question, and with only 12 test points, the RMSE gap itself is close enough that its statistical significance hasn't been checked.

## Files

```
usd_forecasting.ipynb          complete analysis
xgboost_window5_best.pkl       best-window model
model_metadata.json            configuration and performance, including the naive baseline
prediction_visualization.png   actual vs predicted rates
feature_importance.png         feature importance breakdown
mlflow.db                      MLflow tracking store (SQLite), 11 logged runs
```

## Using the model

```python
import pickle
import json

with open('xgboost_window5_best.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

# feature order: lag_rate_1, lag_rate_2, lag_rate_3, rate_std, rate_max, rate_min, lag_vol_1
new_features = [[300, 295, 290, 2.5, 302, 298, 2.3]]
prediction = model.predict(new_features)[0]

print(f"predicted USD/LKR rate: {prediction:.2f}")
print(f"model R2 (Window 5, best window, see Results for average): {metadata['model_performance']['R2']:.4f}")
```

## MLflow tracking

Experiment: USDLKR-Model-Comparison, tracked in a local SQLite store (mlflow.db). 11 runs logged: 5 Random Forest windows, 5 XGBoost windows, and the best-window run, all verified FINISHED. View locally with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

## Limitations

- Small test windows: each fold tests on only 12 months, so R2 and RMSE are sensitive to a handful of points. Window 5's R2 of 0.88 could look meaningfully different with a slightly different cutoff.
- No hyperparameter tuning: parameters were chosen by reasoning, not by grid or Bayesian search. A tuned model might close or widen the naive-baseline gap.
- Single crisis, single country: the 2022 structural break is the only regime shift in this data. Whether XGBoost's advantage generalizes to a different kind of shock is untested.
- Not a clean win over doing nothing: the naive baseline has a lower MAE than the trained model, though the model has a lower RMSE. That trade-off isn't resolved here, it's disclosed.

## Next steps

- Test whether the RMSE gap vs. the naive baseline is statistically significant, given only 12 test points
- Deploy the model behind a prediction API
- Set up automated retraining
- Monitor for data drift

## Author

Chanindu Dahanayake
