# Model Card — taxi-demand-forecaster

_Generated: 2026-10-24 03:04 UTC_

## Model details

- **Problem:** demand
- **Algorithm:** LGBMRegressor
- **Target:** `pickup_count`
- **Registry version:** n/a

## Intended use

Operational forecasting of pickup_count to support fleet positioning and rider fare estimates. Not intended for individual pricing decisions or any use with legal/financial consequences for individuals.

## Evaluation — overall (test split)

| Metric | Value |
|--------|-------|
| rmse | 8.5903 |
| mae | 2.5218 |
| mape | 40.1239 |
| r2 | 0.9715 |

## Evaluation — subgroup fairness

### By borough

Disparity ratio (worst/best RMSE): **197.01** (threshold 1.5) — ⚠️ **flagged**. Worst: `Manhattan`, best: `Staten Island`.

| Group | n | RMSE | MAE | MAPE |
|-------|---|------|-----|------|
| Manhattan | 3862 | 15.417 | 7.682 | 29.65 |
| Queens | 3971 | 6.224 | 1.170 | 52.52 |
| Unknown | 116 | 3.279 | 2.033 | 47.41 |
| Brooklyn | 3457 | 0.739 | 0.439 | 53.98 |
| Bronx | 2476 | 0.443 | 0.261 | 64.10 |
| EWR | 58 | 0.271 | 0.104 | 97.32 |
| Staten Island | 633 | 0.078 | 0.052 | 91.24 |

### By time_of_day

Disparity ratio (worst/best RMSE): **2.68** (threshold 1.5) — ⚠️ **flagged**. Worst: `evening`, best: `overnight`.

| Group | n | RMSE | MAE | MAPE |
|-------|---|------|-----|------|
| evening | 4554 | 11.576 | 3.318 | 37.35 |
| afternoon | 3947 | 8.598 | 2.990 | 36.21 |
| morning | 3036 | 6.199 | 2.337 | 39.07 |
| overnight | 3036 | 4.320 | 0.904 | 57.08 |

### By day_type

Disparity ratio (worst/best RMSE): **1.00** (threshold 1.5) — ok. Worst: `weekday`, best: `weekday`.

| Group | n | RMSE | MAE | MAPE |
|-------|---|------|-----|------|
| weekday | 14573 | 8.590 | 2.522 | 40.12 |

### By holiday

Disparity ratio (worst/best RMSE): **1.00** (threshold 1.5) — ok. Worst: `non-holiday`, best: `non-holiday`.

| Group | n | RMSE | MAE | MAPE |
|-------|---|------|-----|------|
| non-holiday | 14573 | 8.590 | 2.522 | 40.12 |

### By airport

Disparity ratio (worst/best RMSE): **4.49** (threshold 1.5) — ⚠️ **flagged**. Worst: `airport`, best: `non-airport`.

| Group | n | RMSE | MAE | MAPE |
|-------|---|------|-----|------|
| airport | 116 | 35.917 | 25.576 | 36.24 |
| non-airport | 14457 | 8.002 | 2.337 | 40.20 |

## Top features (SHAP)

| Feature | Mean |SHAP| |
|---------|-----------|
| lag_1 | 12.5103 |
| lag_168 | 7.6655 |
| lag_24 | 4.0611 |
| rolling_mean_3 | 0.9907 |
| hour_cos | 0.6015 |
| rolling_std_3 | 0.4562 |
| lag_3 | 0.4362 |
| hour | 0.3967 |
| rolling_std_24 | 0.3619 |
| lag_2 | 0.3585 |

## Limitations

- Trained on a single month of NYC Yellow Taxi data; may not generalise to other periods, boroughs served sparsely, or green/for-hire vehicles.
- Predictions degrade under distribution shift (see drift monitoring).

## Ethical considerations

- No demographic or protected attributes are used or available; fairness is assessed as geographic/temporal service equity, not demographic parity.
- Large borough-level disparities could translate into unequal service quality across neighbourhoods and should be monitored.
