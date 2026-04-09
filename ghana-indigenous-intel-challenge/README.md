# Ghana Indigenous Intel Challenge — Rainfall Classification

## Overview

Entry for the [Ghana Indigenous Intel Challenge](https://zindi.africa/competitions/ghana-indigenous-intel-challenge) hosted on Zindi. The objective is to predict the type of rainfall — heavy, moderate, small, or none — expected in the next 12 to 24 hours, based solely on indigenous ecological indicators submitted by trained farmers across three districts in Ghana.

## Dataset

The data consists of farmer-submitted observations collected via a mobile app, including:

- **user_id** — identifier of the farmer making the prediction
- **community / district** — location of the observation
- **prediction_time** — timestamp of when the prediction was made
- **indicator / indicator_description** — ecological indicator observed (e.g., cloud movement, insect behavior) — mostly missing in the dataset
- **confidence / predicted_intensity** — farmer's self-reported confidence and predicted rainfall intensity
- **forecast_length** — whether the prediction is for the next 12 or 24 hours
- **Target** — rainfall class: `NORAIN`, `SMALLRAIN`, `MEDIUMRAIN`, or `HEAVYRAIN`

The dataset is heavily imbalanced: ~88% of observations are `NORAIN`.

## Approach

1. **Feature engineering** — Extracted hour, day-of-week, and date from the prediction timestamp. Dropped columns that were mostly NaN (`time_observed`, `indicator_description`) or not informative (`confidence`, `predicted_intensity`).
2. **Preprocessing** — Median imputation for numeric features, most-frequent imputation + one-hot encoding for categorical features, all within an sklearn `Pipeline`.
3. **Model** — CatBoost classifier with default hyperparameters.
4. **Validation** — 5-fold stratified cross-validation to preserve class balance.

## Results

| Metric | Score |
|--------|-------|
| CV Macro F1 | 0.9852 +/- 0.0018 |
| CV Accuracy | 0.9964 +/- 0.0004 |
| Zindi Leaderboard (F1 macro) | **0.9587** |

## Project Structure

```
├── README.md
├── notebook.ipynb       # Full pipeline: EDA, feature engineering, training, submission
└── data/
    ├── train.csv
    ├── test.csv
    └── SampleSubmission.csv
```

## Requirements

- pandas, numpy, matplotlib
- scikit-learn
- catboost
